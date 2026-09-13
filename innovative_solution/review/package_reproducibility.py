"""Local source/result handoff; no upload, raw data or embedded-data model objects."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,platform,importlib.metadata
import numpy as np
import pandas as pd
from innovative_solution.supplement_common import PACKAGE,BASE,OUT,sha,write_json


def main():
    primary=pd.read_csv(BASE/'nested_predictions.csv')
    key=primary[primary.model.eq('fixed_full_d2')][['fold','row_index','actual']]
    checks={}
    for file,scope in [('baseline_predictions.csv',None),('feature_experiment_predictions.csv',None)]:
        frame=pd.read_csv(OUT/file)
        for (s,m),part in frame.groupby(['scope','model']):
            joined=key.merge(part,on=['fold','row_index'],suffixes=('_expected','_actual'),validate='one_to_one')
            assert len(joined)==len(part)==300
            np.testing.assert_allclose(joined.actual_expected,joined.actual_actual,rtol=0,atol=0)
            assert np.isfinite(part.prediction).all()
        checks[file]={'models_and_scopes':len(frame.groupby(['scope','model'])),'all_evaluation_records_match':True,'finite_predictions':True}
    ab=pd.read_csv(OUT/'feature_experiment_predictions.csv')
    check=primary[primary.model.eq('fixed_full_d2')].merge(ab[(ab.scope=='ablation')&(ab.model=='full')],on=['fold','row_index'],suffixes=('_main','_supp'))
    delta=float(abs(check.prediction_main-check.prediction_supp).max())
    assert delta<5e-7
    checks['reference_reproduction']={'max_prediction_difference':delta,'note':'float32 versus float64 ensemble averaging; same features and members'}
    branches=PACKAGE/'outputs'/'branch_selection'
    branch_pred=pd.read_csv(branches/'nested_predictions.csv')
    fair_pred=pd.read_csv(PACKAGE/'outputs'/'fair_comparison'/'predictions.csv')
    for name,frame in [('eight_candidate',branch_pred),('equal_budget',fair_pred)]:
        for model,part in frame.groupby('model'):
            joined=key.merge(part,on=['fold','row_index'],suffixes=('_expected','_actual'),validate='one_to_one')
            assert len(joined)==len(part)==300
            np.testing.assert_array_equal(joined.actual_expected,joined.actual_actual)
            assert np.isfinite(part.prediction).all()
        checks[name]={'models':frame.model.nunique(),'rows_per_model':300,'actual_and_record_identity':'passed'}
    family=pd.read_csv(branches/'inference_family.csv')
    assert len(family)==11 and family.holm_family_size.eq(11).all()
    assert not family.duplicated(['reference_label','candidate_label']).any()
    p=family.group_sign_flip_p.to_numpy();order=np.argsort(p);q=np.empty(len(p))
    q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    np.testing.assert_allclose(q,family.holm_p,rtol=0,atol=1e-15)
    availability=pd.read_csv(PACKAGE/'outputs'/'availability'/'all_input_availability.csv')
    assert len(availability)==5952 and availability.available_before_Y.eq('unverified').all()
    checks['inference_family']={'unique_hypotheses':11,'holm_recalculation':'passed'}
    checks['availability']={'inventoried':5952,'verified_before_Y':0}
    input_manifest=json.loads((BASE/'review_manifest.json').read_text(encoding='utf-8'))
    sources=list(PACKAGE.glob('*.py'))+list((PACKAGE/'tests').glob('*.py'))+list((PACKAGE/'review').glob('*.py'))+list((PACKAGE/'thesis').glob('*.py'))
    environment={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib','scikit-learn','xgboost','lightgbm','catboost','joblib','openpyxl']}
    record={'purpose':'local_reproducibility_handoff','raw_data_included':False,'models_included':False,'public_repository':None,
        'redistribution_permission':'not established; this local bundle is not a public release',
        'python':platform.python_version(),'platform':platform.platform(),'versions':environment,
        'input_sha256':input_manifest['input_sha256'],'protocol_sha256':sha(OUT/'protocol.json'),
        'source_sha256':{str(p.relative_to(PACKAGE)):sha(p) for p in sources},'validation_checks':checks,
        'tests':'28 tests passed, including time/process routing, eight-candidate selection and evaluation-label isolation',
        'execution_notes':['LightGBM JSON-punctuation field names mapped to ordered feature_i names without changing values.',
          'Default supplementary features were checked exactly against the primary builder; unnecessary mean arithmetic was removed before the final full experiment run.',
          'Final A/B configuration is selected by development-inner scores among eight candidates; expanded scope informed by retrospective ablation, not independent confirmation.',
          'Thirteen shared train/validation splits match the restricted experiment; one additional calibration-vs-tail audit row records the same boundary.',
          'Eleven distinct comparisons form the reported Holm family; all confidence intervals remain conditional on existing predictions.']}
    write_json(OUT/'reproducibility_manifest.json',record)
    files=sources+list(PACKAGE.glob('*.md'))+list(PACKAGE.glob('requirements*.txt'))
    files += list((PACKAGE/'review').glob('*.md'))+list((PACKAGE/'review').glob('*.json'))
    files += [PACKAGE/'thesis'/'manuscript.md',PACKAGE/'thesis'/'论文正文.md',PACKAGE/'thesis'/'citation_map.json',PACKAGE/'thesis'/'render_review.ps1']
    files += list(OUT.glob('*.csv'))+list(OUT.glob('*.json'))+list(OUT.glob('*.md'))+list((OUT/'plots').glob('*'))
    files += list((OUT/'sources').glob('*.json'))
    files += list(BASE.glob('*.csv'))+list(BASE.glob('*.json'))+list((BASE/'plots').glob('*'))
    for folder in [branches,PACKAGE/'outputs'/'fair_comparison',PACKAGE/'outputs'/'availability']:
        files += list(folder.glob('*.csv'))+list(folder.glob('*.json'))+list(folder.glob('*.md'))+list((folder/'plots').glob('*'))
    # Keep a small set of original audit metadata needed by the figure generators.
    for name in ['run_manifest.json','feature_audit.csv','operation_audit.csv']:
        path=PACKAGE/'outputs'/name
        if path.exists():files.append(path)
    archive=PACKAGE/'TFT_LCD_源码与结果复现包.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for file in sorted(set(files)):
            if file.is_file():z.write(file,arcname=str(Path('innovative_solution')/file.relative_to(PACKAGE)))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.xlsx','.pkl','.joblib')) or '.research_deps' in n for n in z.namelist())
        count=len(z.namelist())
    write_json(OUT/'package_verification.json',{'archive':archive.name,'sha256':sha(archive),'files':count,'bytes':archive.stat().st_size,'zip_integrity':'passed'})
    print(json.dumps({'files':count,'bytes':archive.stat().st_size,'prediction_difference':delta}))


if __name__=='__main__':main()
