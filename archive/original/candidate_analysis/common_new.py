import os,sys,json,hashlib,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
P1=ROOT/'output/mad_etd_icassp2027_p1_20260920'
R2=ROOT/'output/mad_etd_icassp2027_v55_r2'
sys.path.insert(0,str(P1/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from prepare_p1 import load_inputs
from run_e1 import meta,learned,metric
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    path=Path(path).resolve();assert path.is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def table(path,records):
    path=Path(path).resolve();assert path.is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(path,index=False)
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%S%z')
