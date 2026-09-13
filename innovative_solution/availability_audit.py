"""Inventory evidence of feature availability without inventing process semantics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .pipeline import prepare_data, RunTrace
from .config import PipelineConfig
from .review_validation import json_write

OUT=Path(__file__).resolve().parent/'outputs'/'availability'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=prepare_data(PipelineConfig(),RunTrace())
    fields=list(data.train_numeric)+list(data.train_categorical)
    temporal=set(data.train_time)
    rows=[]
    for field in fields:
        rows.append(dict(field=field,operation=data.feature_operation.get(field),
            observed_type='parseable_timestamp' if field in temporal else ('categorical' if field in data.train_categorical else 'numeric'),
            physical_meaning='unverified',unit='unverified',measurement_event='unverified',
            source_system='unverified',earliest_available_time='unverified',Y_measurement_time='unverified',
            available_before_Y='unverified',evidence_reference='no field dictionary or event log supplied',
            prospective_eligibility='not established',offline_complete_record_use='retrospective only'))
    assert len(rows)==5952 and len(set(fields))==5952
    pd.DataFrame(rows).to_csv(OUT/'all_input_availability.csv',index=False,encoding='utf-8-sig')
    train=data.order_key
    direction=[];records=[]
    for name,t,ids in [('train',train,data.train_ids),('A',data.test_a_time.median(axis=1),data.test_a_ids),('B',data.test_b_time.median(axis=1),data.test_b_ids)]:
        before=t<train.min();after=t>train.max()
        direction.append(dict(dataset=name,n=len(t),before_train_min=int(before.sum()),within_train_range=int((~before&~after).sum()),
            after_train_max=int(after.sum()),minimum_proxy=str(pd.to_datetime(t.min(),unit='s')),
            maximum_proxy=str(pd.to_datetime(t.max(),unit='s')),certified_production_time=False))
        for i,value in enumerate(t):
            records.append(dict(dataset=name,row_index=i,ID=ids.iloc[i],time_proxy=float(value),
                relation_to_train_range='before' if before.iloc[i] else ('after' if after.iloc[i] else 'within'),
                evaluation_role='retrospective offline; not prospective forecasting'))
    pd.DataFrame(direction).to_csv(OUT/'prediction_direction.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'record_prediction_direction.csv',index=False)
    json_write(OUT/'availability_manifest.json',dict(input_fields=len(fields),timestamp_fields=len(temporal),
        verified_available_before_target=0,unverified_input_fields=len(fields),
        input_inventory_complete=True,physical_availability_audit_complete=False,
        user_confirmed_no_dictionary_or_timing_documentation=True,
        prospective_model_certified=False,removing_explicit_time_does_not_certify_other_inputs=True,
        required_evidence=['field dictionary','source system and event meaning','earliest release timestamp per field','Y measurement event','actual prediction decision point'],
        conclusion='All current model comparisons are retrospective associations on complete anonymous records.'))
    print(json.dumps(direction,ensure_ascii=False))


if __name__=='__main__':main()
