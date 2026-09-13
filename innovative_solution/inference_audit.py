"""One declared family of distinct reported feature/search comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .review_validation import json_write

ROOT=Path(__file__).resolve().parent/'outputs'
OUT=ROOT/'branch_selection'


def main():
    # Exclude the numerical replication of basic-vs-combined; it is the same hypothesis.
    ab=pd.read_csv(ROOT/'supplement'/'paired_group_inference.csv')
    ab=ab[ab.reference.ne('fixed_raw_d2')].copy()
    ab['comparison_origin']='feature_ablation'
    branch=pd.read_csv(OUT/'paired_group_inference.csv')
    branch=branch[~((branch.reference=='fixed_raw_d2')&(branch.candidate=='fixed_time_d2'))].copy()
    branch['comparison_origin']='branch_search'
    scope=pd.read_csv(OUT/'search_scope_comparison.csv');scope['comparison_origin']='candidate_scope'
    frame=pd.concat([ab,branch,scope],ignore_index=True)
    frame=frame.drop(columns=[c for c in frame if c.startswith('holm_')])
    names={'basic':'基础','process_only':'基础加工序','time_only':'基础加时间','full':'组合',
        'fixed_full_d2':'组合','fixed_time_d2':'基础加时间','no_mask':'组合去缺失',
        'median_only':'中位数填充','group_min5':'设备至少5个','group_min5_shrink10':'至少5个并收缩',
        'nested_selected':'八候选选择','restricted_selected':'四候选选择'}
    frame['reference_label']=frame.reference.map(names);frame['candidate_label']=frame.candidate.map(names)
    assert frame[['reference_label','candidate_label']].notna().all().all()
    assert not frame.duplicated(['reference_label','candidate_label']).any()
    p=frame.group_sign_flip_p.to_numpy(); order=np.argsort(p); q=np.empty(len(p))
    q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    frame['holm_p']=q;frame['holm_family_size']=len(p);frame['resamples']=5000
    frame.to_csv(OUT/'inference_family.csv',index=False,encoding='utf-8-sig')
    json_write(OUT/'inference_family_manifest.json',dict(family_size=len(frame),resamples=5000,
        hypotheses=[f'{r.reference_label} minus {r.candidate_label}' for r in frame.itertuples()],
        duplicate_reproductions_excluded=True,
        adjustment_scope='11 distinct reported feature and selection comparisons; not all possible model or research-history tests',
        CI_scope='individual 95 percent group-cluster intervals, not simultaneous confidence intervals',
        conditional_on_saved_predictions=True,adaptive_research_history_corrected=False,
        assumptions=['cluster resampling treats entities as independent; temporal blocks are sensitivity only','sign-flip requires symmetric group differences'],
        interpretation='descriptive retrospective evidence, not confirmatory significance or causal process value'))


if __name__=='__main__':main()
