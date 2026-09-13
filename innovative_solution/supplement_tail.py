"""Frozen-tail error, conditional intervals and sequential recalibration diagnostics."""
from __future__ import annotations
import json,time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp
from sklearn.metrics import mean_pinball_loss
from .supplement_common import OUT,BASE,SEEDS,matrix,columns_for,prediction_rows,summary,write_json,sha
from .supplement_protocol import freeze
from .pipeline import prepare_data,RunTrace,validation_groups,metric_row,conformal_quantile
from .config import PipelineConfig
from .review_validation import model_for


def wilson(k,n):
    z=norm.ppf(.975); p=k/n; d=1+z*z/n
    a=(p+z*z/(2*n))/d; b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(a-b),float(a+b)


def rolling_radii(cal_errors, cal_times, tail_y, tail_pred, tail_times, window=80):
    """Predict all same-time records before ingesting their outcomes.

    This simulates immediate target availability after each time group, which is
    an explicit hypothetical scenario; it does not assert actual inspection lag.
    """
    history=list(zip(cal_times,np.abs(cal_errors)))
    radii=np.empty(len(tail_y))
    for stamp in np.unique(tail_times):
        positions=np.flatnonzero(tail_times==stamp)
        past=[err for t,err in history if t<stamp][-window:]
        radius=conformal_quantile(np.asarray(past),alpha=.1)
        radii[positions]=radius
        history.extend((stamp,float(abs(tail_y[i]-tail_pred[i]))) for i in positions)
    return radii


def main():
    freeze();started=time.perf_counter()
    data=prepare_data(PipelineConfig(),RunTrace());groups=validation_groups(data)
    splits=pd.read_csv(BASE/"split_audit.csv")
    calrow=splits[splits.stage.eq("calibration")].iloc[0]
    tailrow=splits[splits.stage.eq("tail_evaluation")].iloc[0]
    train=np.array(json.loads(calrow.train_indices));cal=np.array(json.loads(calrow.valid_indices));valid=np.array(json.loads(tailrow.valid_indices))
    frozen=joblib.load(BASE/"frozen_historical_model.joblib")
    cal_preds=[];tail_preds=[];latencies=[]
    for member in frozen["members"]:
        builder=member["builder"];model=member["model"];cols=member["columns"]
        stamp=time.perf_counter()
        C=builder.transform(data.train_numeric.iloc[cal],data.train_categorical.iloc[cal],data.train_time.iloc[cal])
        V=builder.transform(data.train_numeric.iloc[valid],data.train_categorical.iloc[valid],data.train_time.iloc[valid])
        cal_preds.append(model.predict(C[cols]));tail_preds.append(model.predict(V[cols]))
        latencies.append(time.perf_counter()-stamp)
    cp=np.mean(np.asarray(cal_preds,dtype=float),axis=0);vp=np.mean(np.asarray(tail_preds,dtype=float),axis=0)
    saved=pd.read_csv(BASE/"frozen_tail_predictions.csv")
    assert np.allclose(vp,saved.prediction,atol=1e-12,rtol=0)
    details=[prediction_rows(data,groups,valid,vp,"frozen_full_d3",0,"tail")]
    cutoff=float(data.y.iloc[train].quantile(.1))
    modelnames=["full_d2","no_time_d2","absolute_loss_d2","low_tail_weight_d2"]
    predpool={n:[] for n in modelnames}
    selectionreports=[]
    for seed in SEEDS:
        builder,X,V=matrix(data,train,valid,seed)
        selectionreports.append(builder.selection_report.assign(seed=seed))
        for name in modelnames:
            cols=columns_for(X,"process_only" if name=="no_time_d2" else "full")
            m=model_for("full_d2",seed)
            if name=="absolute_loss_d2":m.set_params(objective="reg:absoluteerror")
            weights=np.where(data.y.iloc[train]<=cutoff,3.,1.) if name=="low_tail_weight_d2" else None
            m.fit(X[cols],data.y.iloc[train],sample_weight=weights)
            predpool[name].append(m.predict(V[cols]).astype(float))
    for name,p in predpool.items():details.append(prediction_rows(data,groups,valid,np.mean(p,axis=0),name,0,"tail"))
    frame=pd.concat(details);frame.to_csv(OUT/"tail_model_predictions.csv",index=False)
    scores=[]
    for name,f in frame.groupby("model"):
        low=f.actual<=cutoff;flag=f.prediction<=cutoff
        recall=float((flag&low).sum()/low.sum()) if low.sum() else None
        row={"model":name,**metric_row(f.actual.to_numpy(),f.prediction.to_numpy()),"low_cutoff_from_development":cutoff,
            "n_low":int(low.sum()),"low_tail_mse":float(((f.actual[low]-f.prediction[low])**2).mean()),
            "low_tail_mae":float(abs(f.actual[low]-f.prediction[low]).mean()),"low_value_recall":recall,
            "predicted_low_count":int(flag.sum()),"median_pinball_loss":mean_pinball_loss(f.actual,f.prediction,alpha=.5),
            "max_abs_error":float(abs(f.actual-f.prediction).max())}
        scores.append(row)
    pd.DataFrame(scores).to_csv(OUT/"tail_model_metrics.csv",index=False)
    y=data.y.iloc[valid].to_numpy();cal_y=data.y.iloc[cal].to_numpy()
    fixed=np.full(len(valid),float(frozen["radius"]))
    rolling=rolling_radii(cal_y-cp,data.order_key.iloc[cal].to_numpy(),y,vp,data.order_key.iloc[valid].to_numpy())
    q1,q2=np.quantile(cp,[1/3,2/3]);predbins=np.where(vp<q1,"lower_predicted",np.where(vp<q2,"middle_predicted","upper_predicted"))
    coverage=[];intervals=[]
    for strategy,radii in [("fixed",fixed),("rolling80_immediate_labels",rolling)]:
        covered=np.abs(y-vp)<=radii
        intervals.append(pd.DataFrame({"strategy":strategy,"row_index":valid,"ID":data.train_ids.iloc[valid].to_numpy(),"time_proxy":data.order_key.iloc[valid].to_numpy(),"actual":y,"prediction":vp,"radius":radii,"covered":covered}))
        categories=[("overall","all",np.ones(len(valid),dtype=bool))]
        categories += [("predicted_tercile",b,predbins==b) for b in np.unique(predbins)]
        # All categorical equipment columns, no selection using tail score.
        for col in data.train_categorical:
            labels=data.train_categorical.iloc[valid][col].astype("string").fillna("__MISSING__").to_numpy()
            categories += [(str(col),str(v),labels==v) for v in pd.unique(labels)]
        for variable,label,mask in categories:
            n=int(mask.sum());k=int(covered[mask].sum());lo,hi=wilson(k,n)
            coverage.append({"strategy":strategy,"stratifier":variable,"level":label,"n":n,"covered":k,"coverage":k/n,
                "mean_width":float(np.mean(2*radii[mask])),"wilson_low":lo,"wilson_high":hi,"small_group":n<20,
                "interval_note":"Wilson is descriptive; within-device temporal dependence not modeled"})
    pd.concat(intervals).to_csv(OUT/"conditional_interval_predictions.csv",index=False)
    pd.DataFrame(coverage).to_csv(OUT/"conditional_coverage.csv",index=False)
    influence=saved.copy();influence["squared_error"]=(influence.actual-influence.prediction)**2
    influence["fraction_total_squared_error"]=influence.squared_error/influence.squared_error.sum()
    influence.sort_values("squared_error",ascending=False).to_csv(OUT/"tail_error_influence.csv",index=False)
    targets=[]
    for name,idx in [("development",train),("calibration",cal),("tail",valid)]:
        v=data.y.iloc[idx]
        targets.append({"segment":name,"n":len(v),"mean":v.mean(),"std":v.std(),"min":v.min(),"q10":v.quantile(.1),"median":v.median(),"max":v.max(),"low_fraction":(v<=cutoff).mean()})
    pd.DataFrame(targets).to_csv(OUT/"target_segment_distribution.csv",index=False)
    # Numerical shift distances only; no p-value/causality claim.
    bundlebuilder=frozen["members"][0]["builder"]
    drift=[]
    for col in bundlebuilder.selected_features:
        a=data.train_numeric.iloc[train][col].dropna();b=data.train_numeric.iloc[valid][col].dropna()
        scale=max(float(a.quantile(.75)-a.quantile(.25)),1e-12)
        drift.append({"feature":col,"operation":data.feature_operation.get(col),"ks_distance":ks_2samp(a,b).statistic if len(a) and len(b) else None,
            "median_shift_iqr":abs(float(b.median()-a.median()))/scale,"development_missing":1-len(a)/len(train),"tail_missing":1-len(b)/len(valid)})
    pd.DataFrame(drift).sort_values("ks_distance",ascending=False).to_csv(OUT/"tail_covariate_shift.csv",index=False)
    worst=int(saved.iloc[np.argmax(influence.squared_error)].row_index)
    report=pd.concat(selectionreports).groupby("feature").importance.mean().sort_values(ascending=False)
    sample=[]
    for col in report.head(30).index:
        v=data.train_numeric.iloc[train][col].dropna();obs=data.train_numeric.iloc[worst][col]
        sample.append({"feature":col,"operation":data.feature_operation.get(col),"sample_value":obs,"development_median":v.median(),"development_q01":v.quantile(.01),"development_q99":v.quantile(.99),"percentile_rank":float((v<=obs).mean()) if np.isfinite(obs) else None})
    pd.DataFrame(sample).to_csv(OUT/"NH1835_feature_diagnosis.csv",index=False)
    write_json(OUT/"tail_diagnostic_manifest.json",{"elapsed_sec":time.perf_counter()-started,"calibration80_plus_tail100_three_member_seconds":sum(latencies),
        "timing_scope":"180 records,warm model+data in memory,feature transformation and predictions;not production SLA", "sample_missing_rate":float(data.train_numeric.iloc[worst].isna().mean()),
        "prediction_identity_max_difference":float(max(abs(saved.prediction-vp))),"rolling_assumption":"labels observed immediately after strictly earlier time group;actual label delay unavailable",
        "script_sha256":sha(__file__)})
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


if __name__=="__main__":main()
