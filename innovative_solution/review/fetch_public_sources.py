"""Fetch public source metadata/text for rule and algorithm verification; no uploads."""
from pathlib import Path
import hashlib,json,re,urllib.request

OUT=Path(__file__).resolve().parents[1]/"outputs"/"supplement"/"sources"
OUT.mkdir(parents=True,exist_ok=True)
URLS={
 "competition_intro":"https://tianchi.aliyun.com/competition/entrance/231633/introduction",
 "competition_rules":"https://tianchi.aliyun.com/competition/entrance/231633/rule",
 "lightgbm_paper":"https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
 "catboost_paper":"https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
 "catboost_arxiv":"https://arxiv.org/abs/1706.09516",
 "lightgbm_parameters":"https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html",
 "catboost_parameters":"https://catboost.ai/en/docs/concepts/python-reference_catboostregressor",
}
records=[]
for name,url in URLS.items():
 if name != "catboost_arxiv" and (OUT/f"{name}.txt").exists():
  continue
 try:
  req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
  with urllib.request.urlopen(req,timeout=25) as resp:
   raw=resp.read();status=resp.status
  text=raw.decode("utf-8",errors="replace")
  text=re.sub(r"(?is)<(script|style)\b.*?</\1>","",text)
  text=re.sub(r"(?s)<[^>]+>"," ",text)
  text=re.sub(r"\s+"," ",text).strip()
  (OUT/f"{name}.txt").write_text(text,encoding="utf-8")
  records.append({"name":name,"url":url,"http_status":status,"text_length":len(text),"sha256":hashlib.sha256(raw).hexdigest(),"rule_permission_verified":False if name.startswith("competition") else None})
 except Exception as error:
  records.append({"name":name,"url":url,"error":str(error)})
 print(name,records[-1].get("text_length",records[-1].get("error")),flush=True)
existing=json.loads((OUT/"source_access.json").read_text(encoding="utf-8")) if (OUT/"source_access.json").exists() else []
names={r['name'] for r in records}
(OUT/"source_access.json").write_text(json.dumps([r for r in existing if r['name'] not in names]+records,ensure_ascii=False,indent=2),encoding="utf-8")
