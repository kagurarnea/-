import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from innovative_solution.config import PipelineConfig
from innovative_solution.supplement_common import SupportedGroupBuilder
from innovative_solution.supplement_statistics import paired_cluster
from innovative_solution.supplement_tail import rolling_radii


class SupplementTests(unittest.TestCase):
    def test_device_support_counts_observed_cells_and_unseen_falls_back(self):
        X=pd.DataFrame({"210X1":[10.,np.nan,20.,21.,22.,23.]})
        C=pd.DataFrame({"Tool":["a","a","b","b","b","b"]})
        empty=pd.DataFrame(index=X.index)
        metadata=SimpleNamespace(feature_operation={"210X1":"210"},operation_tool={"210":"Tool"})
        builder=SupportedGroupBuilder(metadata,PipelineConfig(selector_estimators=5,top_k_raw_features=1),42,min_count=3)
        builder.fit(X,C,empty,pd.Series([1.,2.,3.,4.,5.,6.]),pd.Series(np.arange(6)))
        self.assertEqual(builder.impute(X,C).iloc[1,0],21.)
        unseen=builder.impute(pd.DataFrame({"210X1":[np.nan]}),pd.DataFrame({"Tool":["never_seen"]}))
        self.assertEqual(unseen.iloc[0,0],21.)
        self.assertEqual(builder.support_audit_[0]["singleton_cells"],1)

    def test_rolling_never_uses_same_time_or_future_labels(self):
        actual=np.r_[np.full(20,100.),0.];pred=np.zeros(21);times=np.r_[np.full(20,2.),3.]
        radii=rolling_radii(np.zeros(80),np.ones(80),actual,pred,times)
        np.testing.assert_equal(radii[:20],np.zeros(20))
        self.assertEqual(radii[-1],100.)

    def test_pairing_preserves_records_and_clusters(self):
        base=pd.DataFrame({"model":"a","fold":1,"row_index":[1,2,3],"group":[8,8,9],"time_proxy":[1,1,2],"actual":[0.,0.,0.],"prediction":[2.,2.,2.]})
        alt=base.assign(model="b",prediction=1.)
        result=paired_cluster(pd.concat([base,alt]),"a","b",B=100)
        self.assertEqual(result["n_groups"],2)
        self.assertEqual(result["n_records"],3)
        self.assertAlmostEqual(result["mse_difference_ref_minus_candidate"],3.)
        with self.assertRaises(ValueError):
            paired_cluster(pd.concat([base,alt.iloc[:2]]),"a","b",B=100)


if __name__=="__main__":unittest.main()
