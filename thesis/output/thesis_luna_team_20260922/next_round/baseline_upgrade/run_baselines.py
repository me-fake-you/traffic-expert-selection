"""Fixed same-backend baseline development comparison, train -> selection only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import csv, json, hashlib, time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
HIST=ROOT/'output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001'
SOURCE=ROOT/'data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz'
PROV=ROOT/'output/mad_etd_icassp2027_v55_r1/runs/raw_audit_001/selected_provenance.csv'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def labels_for(hashes):
    wanted=set(hashes); found={}
    with PROV.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['partition']=='development40k' and row['sample_hash'] in wanted:
                assert row['sample_hash'] not in found
                found[row['sample_hash']]=int(row['label'])
    assert set(found)==wanted
    return np.array([found[h] for h in hashes],dtype=int)
def scores(y,p):
    h=(p>=.5).astype(int)
    tn=int(((y==0)&(h==0)).sum());fp=int(((y==0)&(h==1)).sum())
    fn=int(((y==1)&(h==0)).sum());tp=int(((y==1)&(h==1)).sum())
    return dict(macro_f1=float(f1_score(y,h,average='macro',labels=[0,1],zero_division=0)),
        malicious_recall=tp/(tp+fn),false_positive_rate=fp/(tn+fp),tn=tn,fp=fp,fn=fn,tp=tp)
def main():
    start=time.perf_counter(); cfg=json.loads((HERE/'preregistration.json').read_text())
    out=HERE/'results';out.mkdir(exist_ok=False)
    logs=HERE/'logs';logs.mkdir(exist_ok=True)
    models=HERE/'models';models.mkdir(exist_ok=False)
    tracked=[Path(__file__),HERE/'preregistration.json',SOURCE,PROV,HIST/'split_manifest.json',HIST/'config.json']
    splits=json.loads((HIST/'split_manifest.json').read_text())
    historical=json.loads((HIST/'config.json').read_text())
    assert {k:v for k,v in cfg['expert_parameters'].items() if k!='random_state'}==historical['expert']
    with np.load(SOURCE,allow_pickle=False) as z:
        xs=z['x_stats'].copy();xh=z['x_hybrid'].copy();g=z['group'].astype(str);h=z['sample_hash'].astype(str)
    assert xs.shape==(40000,8) and xh.shape==(40000,48)
    assert np.allclose(xs,xh[:,:8],equal_nan=True)
    rows=[];individual=[];lock=[];log=[]
    with threadpool_limits(limits=1):
        for fold in cfg['folds']:
            folder=HIST/f'seed_42_fold_{fold}';oofp=folder/'oof_predictions.npz';tracked.append(oofp)
            with np.load(oofp,allow_pickle=False) as z:
                tr=z['indices'].astype(int); ytr=z['y'].astype(int);pt=z['temporal'];ps=z['stats']
            roles=splits[fold]['groups'];va=np.flatnonzero(np.isin(g,roles['selection']))
            assert set(g[tr])==set(roles['train']) and len(tr)==24000 and len(va)==4000
            assert not set(g[tr])&set(g[va]) and not set(tr)&set(va)
            weights=[0,.25,.5,.75,1]
            weight=max(weights,key=lambda w:(scores(ytr,w*pt+(1-w)*ps)['macro_f1'],-abs(w-.5),-w))
            lock.append(dict(fold=fold,temporal_weight=weight,selection_labels_used_for_weight=False))
            dump(out/'weight_lock.json',lock)
            m=HistGradientBoostingClassifier(**cfg['expert_parameters'])
            fitstart=time.perf_counter();m.fit(xh[tr],ytr);fitsec=time.perf_counter()-fitstart
            joblib.dump(m,models/f'concat_fold_{fold}.joblib')
            probs={}
            for name,x in [('stats',xs),('temporal',xh[:,8:])]:
                path=folder/f'models/full_{name}.joblib';tracked.append(path)
                model=joblib.load(path); assert all(model.get_params()[k]==v for k,v in cfg['expert_parameters'].items())
                probs[name]=model.predict_proba(x[va])[:,1]
            probs['probability_average']=(probs['stats']+probs['temporal'])/2
            probs['oof_weighted_average']=weight*probs['temporal']+(1-weight)*probs['stats']
            probs['feature_concatenation']=m.predict_proba(xh[va])[:,1]
            y=labels_for(h[va])
            for method,p in probs.items():
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                rows.append(dict(fold=fold,method=method,rows=len(y),**scores(y,p)))
                individual.append(pd.DataFrame(dict(fold=fold,method=method,sample_hash=h[va],group=g[va],y=y,p=p,prediction=(p>=.5).astype(int))))
            log.append(f'fold={fold} concat_fit_seconds={fitsec:.6f} train=24000 selection=4000 temporal_weight={weight}')
            print(log[-1],flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'fold_metrics.csv',index=False)
    pd.concat(individual,ignore_index=True).to_csv(out/'selection_predictions.csv.gz',index=False,compression='gzip')
    mean=frame.groupby('method',sort=False)[['macro_f1','malicious_recall','false_positive_rate']].mean().reset_index()
    mean.to_csv(out/'descriptive_mean.csv',index=False)
    hashes={str(p):sha(p) for p in tracked};dump(out/'input_hashes.json',hashes)
    dump(out/'summary.json',dict(status='COMPLETED_DEVELOPMENT_BASELINE_COMPARISON',models_trained=5,folds=cfg['folds'],
        source_y_array_read=False,evaluation_scored=False,selection_rows_per_fold=4000,overlapping_folds=True,
        fresh_blind_test=False,wall_seconds=time.perf_counter()-start,descriptive_mean=mean.to_dict('records')))
    (logs/'run.log').write_text('\n'.join(log)+'\n',encoding='utf8')
    print(mean.to_string(index=False))
if __name__=='__main__':main()
