"""Same-row strong baselines and controlled feature/model sensitivity experiments."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .supplement_common import (OUT, BASE, SEEDS, matrix, columns_for, prediction_rows, summary, rerank, sha, write_json)
from .supplement_protocol import freeze
from .pipeline import prepare_data, RunTrace, validation_groups, log
from .config import PipelineConfig
from .review_validation import model_for, clean_train
from xgboost import XGBRegressor

GRIDS = {
    "Ridge": [{"alpha": x} for x in [10,100,1000]],
    "RandomForest": [{"min_samples_leaf": x} for x in [2,5]],
    "GBR": [{"max_depth": x} for x in [2,3]],
    "SVR": [{"C": x} for x in [1,10]],
    "LightGBM": [{"num_leaves": x} for x in [7,15]],
    "CatBoost": [{"depth": x} for x in [4,6]],
    "XGBoost": [{"max_depth": x} for x in [2,3]],
}


def learner(name, params, seed):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, max_features=.7, n_jobs=4, random_state=seed, **params)
    if name == "GBR":
        return GradientBoostingRegressor(n_estimators=500, learning_rate=.03, min_samples_leaf=3, random_state=seed, **params)
    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", gamma="scale", epsilon=.05, **params))
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=800, learning_rate=.03, min_child_samples=10, reg_lambda=2,
                            verbosity=-1, n_jobs=4, random_state=seed, **params)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=800, learning_rate=.03, l2_leaf_reg=3, loss_function="RMSE",
                                  thread_count=4, random_seed=seed, verbose=False, allow_writing_files=False, **params)
    if name == "XGBoost":
        return model_for("full_d"+str(params["max_depth"]), seed)
    raise ValueError(name)


def model_input(frame, family):
    # Device category strings may contain JSON punctuation. LightGBM's field-name
    # restrictions must not change values, order, or the feature set.
    if family == "LightGBM":
        out=frame.copy(deep=False)
        out.columns=[f"feature_{i}" for i in range(frame.shape[1])]
        return out
    return frame


def load():
    freeze()
    data = prepare_data(PipelineConfig(), RunTrace())
    groups = validation_groups(data)
    splits = pd.read_csv(BASE / "split_audit.csv")
    return data, groups, splits


def indices(row):
    return np.array(json.loads(row.train_indices)), np.array(json.loads(row.valid_indices))


def baseline():
    data, groups, splits = load()
    write_json(OUT / "baseline_grid.json", GRIDS)
    details, searches, timings, choices = [], [], [], []
    started = time.perf_counter()
    for outer in splits[splits.stage.eq("outer")].itertuples():
        train, valid = indices(outer)
        inner = []
        for split in splits[splits.stage.eq(f"outer_{outer.fold}")].itertuples():
            ti, vi = indices(split)
            builder, X, V = matrix(data, ti, vi)
            inner.append((X, V, data.y.iloc[ti].to_numpy(), data.y.iloc[vi].to_numpy(), vi))
        selected, oof = {}, {}
        for name, grid in GRIDS.items():
            candidate_results = []
            for p in grid:
                preds = []
                for j, (X,V,y,z,vi) in enumerate(inner, 1):
                    stamp = time.perf_counter()
                    model = learner(name,p,42).fit(model_input(X,name),y)
                    pred = model.predict(model_input(V,name))
                    loss = float(np.mean((pred-z)**2))
                    searches.append({"outer_fold": outer.fold, "inner_fold":j,"family":name,"parameters":json.dumps(p),"mse":loss,"n":len(z)})
                    timings.append({"fold":outer.fold,"stage":"inner","model":name,"seconds":time.perf_counter()-stamp})
                    preds.append(pred)
                actual = np.concatenate([entry[3] for entry in inner])
                pred = np.concatenate(preds)
                candidate_results.append((float(np.mean((actual-pred)**2)), p, pred))
            best = min(candidate_results,key=lambda t:t[0])
            selected[name],oof[name] = best[1],best[2]
            choices.append({"fold":outer.fold,"model":name,"selected_parameters":json.dumps(best[1]),"pooled_inner_mse":best[0]})
            log(f"Baseline outer {outer.fold} {name}: selected {best[1]}")
        outer_preds = {name:[] for name in GRIDS}
        for seed in SEEDS:
            _, X,V = matrix(data,train,valid,seed)
            for name in GRIDS:
                stamp = time.perf_counter()
                model = learner(name,selected[name],seed).fit(model_input(X,name),data.y.iloc[train])
                outer_preds[name].append(model.predict(model_input(V,name)))
                timings.append({"fold":outer.fold,"stage":"outer","model":name,"seconds":time.perf_counter()-stamp})
        averaged = {name:np.mean(pred,axis=0) for name,pred in outer_preds.items()}
        for name,pred in averaged.items():
            details.append(prediction_rows(data,groups,valid,pred,name,outer.fold,"strong_baseline"))
        stacking_names = ["GBR","XGBoost","RandomForest","SVR"]
        meta = LinearRegression().fit(np.column_stack([oof[name] for name in stacking_names]),np.concatenate([entry[3] for entry in inner]))
        stacked = meta.predict(np.column_stack([averaged[name] for name in stacking_names]))
        details.append(prediction_rows(data,groups,valid,stacked,"TemporalStacking",outer.fold,"strong_baseline"))
        choices.append({"fold":outer.fold,"model":"TemporalStacking","selected_parameters":json.dumps({"meta_weights":meta.coef_.tolist(),"intercept":float(meta.intercept_)}),"pooled_inner_mse":None})
        pd.concat(details).to_csv(OUT/"baseline_predictions.csv",index=False)
        pd.DataFrame(searches).to_csv(OUT/"baseline_inner_scores.csv",index=False)
        pd.DataFrame(choices).to_csv(OUT/"baseline_choices.csv",index=False)
        log(f"Strong baseline outer {outer.fold} saved")
    summary(pd.concat(details)).to_csv(OUT/"baseline_summary.csv",index=False)
    pd.DataFrame(timings).to_csv(OUT/"baseline_timings.csv",index=False)
    write_json(OUT/"baseline_manifest.json",{"elapsed_sec":time.perf_counter()-started,"grid":GRIDS,
        "final_members":3,"inner_seed":42,"versions":{n:importlib.metadata.version(n) for n in ["scikit-learn","xgboost","lightgbm","catboost"]},"script_sha256":sha(__file__)})


def feature_experiments():
    data,groups,splits=load()
    details, dims, selected_rows, support = [],[],[],[]
    variants = {"median_only":{"median_only":True},"group_min5":{"min_count":5},"group_min5_shrink10":{"min_count":5,"prior_count":10}}
    sensitivity = [ ("importance_0.5",{"importance_weight":.5}), ("importance_0.9",{"importance_weight":.9}),
        ("drift_0",{"drift_power":0}), ("drift_0.5",{"drift_power":.5}),
        ("top160",{"top_k":160}), ("top640",{"top_k":640}),
        ("mask0",{"missing_threshold":0}), ("mask0.02",{"missing_threshold":.02})]
    views=["measurements","basic","process_only","time_only","full","no_mask"]
    started=time.perf_counter()
    for outer in splits[splits.stage.eq("outer")].itertuples():
        train,valid=indices(outer)
        predpool={name:[] for name in views+list(variants)}
        for seed in SEEDS:
            builder,X,V=matrix(data,train,valid,seed)
            for feature in builder.selected_features:
                selected_rows.append({"fold":outer.fold,"seed":seed,"feature":feature})
            support.extend([{"fold":outer.fold,"seed":seed,**r} for r in builder.support_audit_])
            for view in views:
                cols=columns_for(X,view)
                model=model_for("full_d2",seed).fit(X[cols],data.y.iloc[train])
                predpool[view].append(model.predict(V[cols]))
                dims.append({"fold":outer.fold,"seed":seed,"variant":view,"numeric_kept":len(builder.numeric_columns_),"raw_selected":len(builder.selected_features),"time_fields":len(builder.time_columns_),"engineered":len(cols)})
                if seed==42 and view=="full":
                    details.append(prediction_rows(data,groups,valid,predpool[view][-1],"reference_seed42",outer.fold,"sensitivity"))
            if seed==42:
                for name,opts in sensitivity:
                    rerank(builder,data.train_numeric.iloc[train],**opts)
                    A=builder.transform(data.train_numeric.iloc[train],data.train_categorical.iloc[train],data.train_time.iloc[train])
                    B=builder.transform(data.train_numeric.iloc[valid],data.train_categorical.iloc[valid],data.train_time.iloc[valid])
                    m=model_for("full_d2",42).fit(A,data.y.iloc[train])
                    details.append(prediction_rows(data,groups,valid,m.predict(B),name,outer.fold,"sensitivity"))
                for name,opts in [("rounds600",{"n_estimators":600}), ("lr0.025",{"learning_rate":.025}),
                                  ("reg0_1",{"reg_alpha":0,"reg_lambda":1}),("reg0.1_10",{"reg_alpha":.1,"reg_lambda":10})]:
                    m=model_for("full_d2",42).set_params(**opts).fit(X,data.y.iloc[train])
                    details.append(prediction_rows(data,groups,valid,m.predict(V),name,outer.fold,"sensitivity"))
            for name,opts in variants.items():
                other,A,B=matrix(data,train,valid,seed,**opts)
                m=model_for("full_d2",seed).fit(A,data.y.iloc[train])
                predpool[name].append(m.predict(B))
        for name,pred in predpool.items():
            details.append(prediction_rows(data,groups,valid,np.mean(pred,axis=0),name,outer.fold,"ablation"))
        # Reverse-time stress check uses exactly the same evaluation records.
        order=np.argsort(data.order_key.to_numpy(),kind="stable")
        after=order[data.order_key.iloc[order].to_numpy()>data.order_key.iloc[valid].max()]
        after=after[~np.isin(groups[after],groups[valid])]
        reverse={view:[] for view in ["basic","full"]}
        for seed in SEEDS:
            _,X,V=matrix(data,after,valid,seed)
            for view in reverse:
                cols=columns_for(X,view)
                m=model_for("full_d2",seed).fit(X[cols],data.y.iloc[after])
                reverse[view].append(m.predict(V[cols]))
        for view,pred in reverse.items():
            details.append(prediction_rows(data,groups,valid,np.mean(pred,axis=0),view,outer.fold,"reverse_time"))
        log(f"Feature ablations/sensitivity/reverse outer {outer.fold} complete")
        pd.concat(details).to_csv(OUT/"feature_experiment_predictions.csv",index=False)
        pd.DataFrame(dims).to_csv(OUT/"fold_feature_dimensions.csv",index=False)
        pd.DataFrame(selected_rows).to_csv(OUT/"selected_fields.csv",index=False)
        pd.DataFrame(support).to_csv(OUT/"device_group_support.csv",index=False)
    summary(pd.concat(details)).to_csv(OUT/"feature_experiment_summary.csv",index=False)
    write_json(OUT/"features_manifest.json",{"elapsed_sec":time.perf_counter()-started,"sensitivity_members":1,"ablation_members":3,"no_final_model_promotion":True,"script_sha256":sha(__file__)})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["baseline","features"])
    args=p.parse_args()
    baseline() if args.stage=="baseline" else feature_experiments()
