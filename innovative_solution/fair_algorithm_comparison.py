"""Equal candidate-count and tree-round sensitivity, on the same outer records."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .supplement_models import learner, model_input, indices
from .supplement_common import matrix, columns_for, prediction_rows, summary, SEEDS, sha, write_json
from .pipeline import prepare_data, RunTrace, validation_groups, log
from .config import PipelineConfig

OUT = Path(__file__).resolve().parent / "outputs" / "fair_comparison"
BASE = OUT.parent / "review"
PARAMETERS = {"Ridge":{"alpha":100}, "RandomForest":{"min_samples_leaf":2},
    "GBR":{"max_depth":2}, "SVR":{"C":1}, "LightGBM":{"num_leaves":7},
    "CatBoost":{"depth":4}, "XGBoost":{"max_depth":2}}
PROTOCOL = {"interpretation":"retrospective_budget_sensitivity_not_exhaustive_ranking",
    "candidate_views":["basic","time_only"], "trials_per_family":2,
    "tree_rounds":500, "inner_seed":42, "outer_seeds":SEEDS,
    "parameters":PARAMETERS, "same_outer_records":True,
    "selection":"pooled inner MSE over two feature views; fixed other parameters",
    "budget_limit":"equal candidates, folds, seeds and tree count; not equal compute, capacity or optimization effort",
    "categorical_policy":"same one-hot representation; CatBoost native categorical treatment not evaluated",
    "modern_foundation_models":"not evaluated; no claim of state-of-the-art"}


def model(name, seed):
    m = learner(name, PARAMETERS[name], seed)
    if name == "CatBoost": m.set_params(iterations=500)
    elif name in {"RandomForest","GBR","LightGBM","XGBoost"}: m.set_params(n_estimators=500)
    return m


def main():
    OUT.mkdir(parents=True,exist_ok=True); write_json(OUT/'protocol.json',PROTOCOL)
    started = time.perf_counter()
    data = prepare_data(PipelineConfig(),RunTrace()); groups = validation_groups(data)
    splits = pd.read_csv(BASE/'split_audit.csv')
    details, searches, timings, choices, preptimes = [], [], [], [], []
    for outer in splits[splits.stage.eq('outer')].itertuples():
        train, valid = indices(outer); inner = []
        for split in splits[splits.stage.eq(f'outer_{outer.fold}')].itertuples():
            ti, vi = indices(split); stamp=time.perf_counter()
            _, X,V = matrix(data,ti,vi,42)
            preptimes.append(dict(fold=outer.fold,stage='inner',seconds=time.perf_counter()-stamp))
            inner.append((X,V,data.y.iloc[ti].to_numpy(),data.y.iloc[vi].to_numpy()))
        selected = {}
        for name in PARAMETERS:
            scores = []
            for view in PROTOCOL['candidate_views']:
                errors=[]
                for fold,(X,V,y,z) in enumerate(inner,1):
                    cols=columns_for(X,view);stamp=time.perf_counter()
                    m=model(name,42).fit(model_input(X[cols],name),y)
                    pred=m.predict(model_input(V[cols],name));err=(pred-z)**2;errors.extend(err)
                    searches.append(dict(outer_fold=outer.fold,inner_fold=fold,model=name,view=view,mse=float(err.mean()),n=len(z)))
                    timings.append(dict(fold=outer.fold,stage='inner',model=name,view=view,seconds=time.perf_counter()-stamp))
                scores.append((float(np.mean(errors)),view))
            mse,selected[name]=min(scores,key=lambda x:x[0])
            choices.append(dict(fold=outer.fold,model=name,selected_view=selected[name],pooled_inner_mse=mse))
        pool={name:[] for name in PARAMETERS}
        for seed in SEEDS:
            stamp=time.perf_counter();_,X,V=matrix(data,train,valid,seed)
            preptimes.append(dict(fold=outer.fold,stage='outer',seconds=time.perf_counter()-stamp))
            for name in PARAMETERS:
                view=selected[name];cols=columns_for(X,view);stamp=time.perf_counter()
                m=model(name,seed).fit(model_input(X[cols],name),data.y.iloc[train])
                pool[name].append(m.predict(model_input(V[cols],name)).astype(float))
                timings.append(dict(fold=outer.fold,stage='outer',model=name,view=view,seconds=time.perf_counter()-stamp))
        for name,preds in pool.items():
            details.append(prediction_rows(data,groups,valid,np.mean(preds,axis=0),name,outer.fold,'equal_trial_tree_budget'))
        pd.concat(details).to_csv(OUT/'predictions.csv',index=False)
        pd.DataFrame(searches).to_csv(OUT/'inner_scores.csv',index=False)
        pd.DataFrame(choices).to_csv(OUT/'choices.csv',index=False)
        log(f'Equal-budget algorithm comparison outer {outer.fold} complete')
    result=summary(pd.concat(details));timings=pd.DataFrame(timings)
    totals=timings.groupby('model').seconds.sum().rename('learner_seconds')
    result.merge(totals,on='model').to_csv(OUT/'summary.csv',index=False)
    timings.to_csv(OUT/'timings.csv',index=False);pd.DataFrame(preptimes).to_csv(OUT/'preprocessing_times.csv',index=False)
    write_json(OUT/'manifest.json',dict(protocol=PROTOCOL,elapsed_sec=time.perf_counter()-started,
        code_sha256=sha(__file__),splits_sha256=sha(BASE/'split_audit.csv'),
        timing_note='shared preprocessing logged separately; wall times are local observations and can reflect concurrent load'))


if __name__ == '__main__':main()
