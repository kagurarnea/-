"""Paired group inference and data/selection diagnostics from saved predictions."""
from __future__ import annotations
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from .supplement_common import OUT,BASE,write_json,sha,summary
from .supplement_protocol import freeze
from .pipeline import prepare_data,RunTrace,validation_groups
from .config import PipelineConfig


def paired_cluster(frame, reference, candidate, B=5000, seed=20260913):
    ref=frame[frame.model.eq(reference)]
    other=frame[frame.model.eq(candidate)]
    paired=ref.merge(other,on=["fold","row_index"],suffixes=("_ref","_candidate"),validate="one_to_one")
    if len(paired)!=len(ref) or len(paired)!=len(other):
        raise ValueError("Paired methods must have identical evaluation records")
    assert np.array_equal(paired.actual_ref,paired.actual_candidate)
    paired["difference"]=(paired.actual_ref-paired.prediction_ref)**2-(paired.actual_ref-paired.prediction_candidate)**2
    clustered=paired.groupby("group_ref").agg(loss_sum=("difference","sum"),size=("difference","size"),time=("time_proxy_ref","min")).sort_values("time")
    s=clustered.loss_sum.to_numpy(); n=clustered["size"].to_numpy(); G=len(s)
    rng=np.random.default_rng(seed)
    sampled=rng.integers(G,size=(B,G))
    draws=s[sampled].sum(axis=1)/n[sampled].sum(axis=1)
    pergroup=s/n
    groupdraws=pergroup[sampled].mean(axis=1)
    signs=rng.choice([-1,1],size=(B,G))
    observed=s.sum()/n.sum()
    perm=(signs*s).sum(axis=1)/n.sum()
    out={"reference":reference,"candidate":candidate,"n_records":int(n.sum()),"n_groups":G,
        "mse_difference_ref_minus_candidate":float(observed),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),
        "group_equal_difference":float(pergroup.mean()),"group_equal_ci_low":float(np.quantile(groupdraws,.025)),"group_equal_ci_high":float(np.quantile(groupdraws,.975)),
        "group_sign_flip_p":float((1+(np.abs(perm)>=abs(observed)).sum())/(B+1)),
        "interpretation":"exploratory conditional-on-predictions; sign symmetry and between-cluster independence assumptions; no adaptive-selection correction"}
    for length in [5,10,20]:
        starts=rng.integers(G,size=(B,int(np.ceil(G/length))))
        idx=((starts[:,:,None]+np.arange(length))%G).reshape(B,-1)[:,:G]
        values=s[idx].sum(axis=1)/n[idx].sum(axis=1)
        out[f"block{length}_low"],out[f"block{length}_high"]=map(float,np.quantile(values,[.025,.975]))
    return out


def cross_audit(data):
    cfg=PipelineConfig()
    raw={name:pd.read_pickle(cfg.cache_dir/f"{file.removesuffix('.xlsx')}.pkl") for name,file in [("train",cfg.train_file),("A",cfg.test_a_file),("B",cfg.test_b_file)]}
    features=list(raw["A"].columns[1:]); rows=[]
    frames={name:frame[features].copy() for name,frame in raw.items()}
    hashes={name:pd.util.hash_pandas_object(frame,index=False).to_numpy() for name,frame in frames.items()}
    for a,b in combinations(raw,2):
        ids=set(raw[a].iloc[:,0].astype(str))&set(raw[b].iloc[:,0].astype(str))
        common=set(hashes[a])&set(hashes[b]); pairs=[]
        for sig in common:
            for i in np.flatnonzero(hashes[a]==sig):
                for j in np.flatnonzero(hashes[b]==sig):
                    if frames[a].iloc[i].equals(frames[b].iloc[j]):pairs.append((int(i),int(j)))
        rows.append({"dataset_1":a,"dataset_2":b,"shared_ID_count":len(ids),"shared_IDs":json.dumps(sorted(ids)),"exact_X_pairs":len(pairs),"exact_X_positions":json.dumps(pairs)})
    pd.DataFrame(rows).to_csv(OUT/"cross_dataset_identity_audit.csv",index=False)
    # Name-based temporal clues do not establish information availability.
    fields=[]
    for col in data.train_time:
        fields.append({"field":col,"parse_count":int(data.train_time[col].notna().sum()),
          "contains_quality_keyword":any(x in col.lower() for x in ["quality","inspect","test","value","label","result"]),
          "physical_role":"unknown","availability_at_mid_process":"unverified"})
    pd.DataFrame(fields).to_csv(OUT/"timestamp_availability_audit.csv",index=False)
    med=data.train_time.median(axis=1); order=np.argsort(med.to_numpy(),kind="stable")
    checks=[]
    for name,proxy in [("minimum",data.train_time.min(axis=1)),("maximum",data.train_time.max(axis=1)),("median",med)]:
        alternative=np.argsort(proxy.to_numpy(),kind="stable")
        checks.append({"proxy":name,"spearman_vs_median":spearmanr(med,proxy).statistic,
            "kendall_vs_median":kendalltau(med,proxy).statistic,
            "tail100_overlap":len(set(order[700:])&set(alternative[700:])),
            "outer300_overlap":len(set(order[320:620])&set(alternative[320:620]))})
    pd.DataFrame(checks).to_csv(OUT/"time_proxy_sensitivity.csv",index=False)


def selection_audit():
    scores=pd.read_csv(BASE/"inner_selection.csv")
    scores["weighted_loss"]=scores.mse*scores.n_valid
    pooled=scores.groupby(["stage","candidate"]).agg(loss=("weighted_loss","sum"),n=("n_valid","sum"),fold_min=("mse","min"),fold_max=("mse","max")).reset_index()
    pooled["pooled_mse"]=pooled.loss/pooled.n
    pooled["rank"]=pooled.groupby("stage").pooled_mse.rank(method="first")
    pooled.to_csv(OUT/"selection_score_distribution.csv",index=False)
    selected=pooled[pooled["rank"].eq(1)].copy()
    selected.to_csv(OUT/"selected_configs.csv",index=False)
    selected[selected.stage.str.startswith("outer_")].candidate.value_counts().rename_axis("configuration").reset_index(name="outer_selection_count").to_csv(OUT/"selection_frequency.csv",index=False)
    if (OUT/"selected_fields.csv").exists():
        fields=pd.read_csv(OUT/"selected_fields.csv")
        sets={(fold,seed):set(g.feature) for (fold,seed),g in fields.groupby(["fold","seed"])}
        overlap=[]
        for a,b in combinations(sets,2):
            overlap.append({"fold_a":a[0],"seed_a":a[1],"fold_b":b[0],"seed_b":b[1],"jaccard":len(sets[a]&sets[b])/len(sets[a]|sets[b]),"intersection":len(sets[a]&sets[b])})
        pd.DataFrame(overlap).to_csv(OUT/"selection_jaccard.csv",index=False)
        fields.groupby("feature").size().rename("selection_count_of_9").sort_values(ascending=False).to_csv(OUT/"selection_stability.csv")


def main():
    freeze()
    data=prepare_data(PipelineConfig(),RunTrace())
    cross_audit(data);selection_audit()
    primary=pd.read_csv(BASE/"nested_predictions.csv")
    inference=[paired_cluster(primary,"fixed_raw_d2","fixed_full_d2")]
    if (OUT/"feature_experiment_predictions.csv").exists():
        f=pd.read_csv(OUT/"feature_experiment_predictions.csv")
        ab=f[f.scope.eq("ablation")]
        if ab.groupby("model").size().min()==300:
            for comparison in ["process_only","time_only","full","no_mask","median_only","group_min5","group_min5_shrink10"]:
                inference.append(paired_cluster(ab,"basic" if comparison in ["process_only","time_only","full"] else "full",comparison))
            # Leave-one-record influence is descriptive; official scores retain all rows.
    result=pd.DataFrame(inference)
    if len(result)>1:
        p=result.group_sign_flip_p.to_numpy(); ranking=np.argsort(p); adjusted=np.empty(len(p))
        adjusted[ranking]=np.minimum(1,np.maximum.accumulate(p[ranking]*(len(p)-np.arange(len(p)))))
        result["holm_adjusted_p_across_reported_comparisons"]=adjusted
    result.to_csv(OUT/"paired_group_inference.csv",index=False)
    eq=primary.copy();eq["squared_error"]=(eq.actual-eq.prediction)**2;eq["absolute_error"]=abs(eq.actual-eq.prediction)
    eq.groupby(["model","group"])[["squared_error","absolute_error"]].mean().groupby("model").mean().rename(columns={"squared_error":"product_equal_mse","absolute_error":"product_equal_mae"}).to_csv(OUT/"group_equal_metrics.csv")
    print(result[["reference","candidate","mse_difference_ref_minus_candidate","ci_low","ci_high","group_sign_flip_p"]].to_string(index=False),flush=True)


if __name__=="__main__": main()
