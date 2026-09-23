"""Separate within-capture early/late supplement; preserve all parent frozen artifacts."""
import sys,json,time
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
import authoritative_fusion_benchmark as b
import frozen_fusion_fast as f
from natural_shift_audit import OUT,SEEDS,dump,sha

TEMP=OUT/'temporal_refit'
OLD=['V0_Stats','V0_Temporal','V1_Average','V2_Yager','V3_Full','V6_EDL']

def run_cell(seed,fold):
    dest=TEMP/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}'
    if (dest/'NATURAL_DONE.json').exists():return dict(seed=seed,fold=fold,status='cached')
    b.CONDITIONS=['clean'];b.METHODS=OLD
    started=time.perf_counter();b.run_fold(str(TEMP),seed,fold)
    # Existing runner has trained every shared view and EDL before its one clean test call.
    # New heads receive ONLY cross-fitted training / validation records below.
    with np.load(TEMP/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz') as z:tr=z['train'];va=z['validation']
    with np.load(TEMP/'features.npz') as z:y=z['y'][tr];yv=z['y'][va]
    frozen=TEMP/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'
    with np.load(frozen/'train_oof.npz') as z:x=f.probability_features(z['p'],z['app']);gx=f.gating_features(z['p'],z['app']);assert np.array_equal(z['index'],tr)
    with np.load(frozen/'validation.npz') as z:xv=f.probability_features(z['p'],z['app']);gv=f.gating_features(z['p'],z['app']);assert np.array_equal(z['index'],va)
    from sklearn.linear_model import LogisticRegression
    import joblib
    lr=LogisticRegression(random_state=seed,**f.CONFIG['logistic']).fit(x,y)
    mlp,mm=f.train_network('mlp',x,y,xv,yv,seed);gate,gm=f.train_network('gate',gx,y,gv,yv,seed);torch=f.torchlib()
    joblib.dump(lr,dest/'natural_logistic.joblib');torch.save(dict(mlp=mlp.state_dict(),gate=gate.state_dict()),dest/'natural_heads.pt')
    dump(dest/'NATURAL_TRAINED.json',dict(seed=seed,fold=fold,mlp=mm,gate=gm,test_data_in_training=False,model_sha256={name:sha(dest/name) for name in ('natural_logistic.joblib','natural_heads.pt')}))
    with np.load(dest/'predictions.npz') as z:pred={k:z[k] for k in z.files if k in ('index','y','group') or any(k.startswith('clean__'+m+'__') for m in OLD)}
    with np.load(frozen/'test_clean.npz') as z:tx=f.probability_features(z['p'],z['app']);tg=f.gating_features(z['p'],z['app']);app=z['app'];assert np.array_equal(z['index'],pred['index'])
    with torch.no_grad():
        ps=[lr.predict_proba(tx)[:,1],torch.softmax(mlp(torch.tensor(tx)),1)[:,1].numpy(),f.gate_probability(gate(torch.tensor(tg)),torch.tensor(tg))[0].numpy()]
    for m,p in zip(f.METHODS,ps):
        key='clean__'+m+'__';pred[key+'pred']=(p>=.5).astype(np.int8);pred[key+'score']=2*abs(p-.5);pred[key+'available']=app.any(1);pred[key+'verdict']=np.where(app.any(1),p>=.5,3).astype(np.int8)
    np.savez_compressed(dest/'natural_predictions.npz',**pred)
    r=dict(seed=seed,fold=fold,status='completed',seconds=time.perf_counter()-started,ntrain=len(tr),nval=len(va),ntest=len(pred['y']),predictions_sha256=sha(dest/'natural_predictions.npz'))
    dump(dest/'NATURAL_DONE.json',r);return r

def main():
    assert (OUT/'NATURAL_PROTOCOL.json').exists();failures=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(run_cell,s,g):(s,g) for s in SEEDS for g in range(20)}
        for i,future in enumerate(as_completed(jobs),1):
            try:r=future.result()
            except Exception as e:
                import traceback
                r=dict(seed=jobs[future][0],fold=jobs[future][1],status='failed',error=repr(e),traceback=traceback.format_exc());failures.append(r)
            with (TEMP/'execution.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(r)+'\n')
            if i%20==0 or failures:print(i,'/200',round(time.perf_counter()-start,1),r,flush=True)
    if failures:raise RuntimeError('Temporal failures logged; no partial result accepted')
    for seed in SEEDS:
        arrays={}
        for fold in range(20):
            with np.load(TEMP/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}'/'natural_predictions.npz') as z:
                for k in z.files:arrays.setdefault(k,[]).append(z[k])
        arrays={k:np.concatenate(v) for k,v in arrays.items()};dest=TEMP/'cache';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'seed_{seed}.npz',**arrays)
    dump(TEMP/'COMPLETE.json',dict(cells=200,seconds=time.perf_counter()-start,parent_protocol_unchanged=True,study='within-capture early/late relative-time refit'))
if __name__=='__main__':main()
