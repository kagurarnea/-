"""Warm end-to-end feature transformation + 3-member prediction latency."""
import json,time,platform,os
import joblib,numpy as np
from innovative_solution.supplement_common import BASE,OUT,write_json

bundle=joblib.load(BASE/"model_A.joblib")
data=bundle["members"][0]["builder"].data
times=[]
for size in [1,100]:
 for repeat in range(4):
  stamp=time.perf_counter();pred=[]
  for entry in bundle["members"]:
   X=entry["builder"].transform(data.test_a_numeric.iloc[:size],data.test_a_categorical.iloc[:size],data.test_a_time.iloc[:size])
   pred.append(entry["model"].predict(X[entry["columns"]]))
  result=np.mean(pred,axis=0)
  if repeat:times.append({"batch_size":size,"repeat":repeat,"seconds":time.perf_counter()-stamp})
write_json(OUT/"inference_latency.json",{"platform":platform.platform(),"cpu":platform.processor(),"logical_cores":os.cpu_count(),
 "scope":"resident model and input frame; all feature transformation + 3 model predictions; excludes I/O, acquisition, queuing and label delay; local snapshot, not SLA",
 "measurements":times,"medians":{str(n):float(np.median([t['seconds'] for t in times if t['batch_size']==n])) for n in [1,100]}})
print(json.dumps({str(n):float(np.median([t['seconds'] for t in times if t['batch_size']==n])) for n in [1,100]}))
