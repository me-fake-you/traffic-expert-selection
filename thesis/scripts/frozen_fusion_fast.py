"""Train only small fusion heads on the existing authoritative frozen outputs.

Phases are separate: inspect/partition labels -> train every head -> seal -> test
once -> append. No view detector fit or inference function is imported/called.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,csv,hashlib,json,shutil,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(Path.home()/'AppData/Roaming/Python/Python312/site-packages'))
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

BASE=ROOT/'output/ustc_authoritative_fusion_v1'
OUT=BASE/'trained_fusion_fast_v1'
SEEDS=list(range(42,52))
METHODS=['V4_StackLogistic_Frozen','V5_StackMLP32x16_Frozen','V6_Attention_Frozen']
CONFIG=dict(version='trained-fusion-fast-v1',seeds=SEEDS,folds=20,
    stack_features='six binary class probabilities in Stats/Temporal/TLS order; missing=[0.5,0.5]',
    gate_features='same six probabilities plus three applicability bits',
    logistic=dict(C=1.,max_iter=500,solver='lbfgs'),mlp_dims=[6,32,16,2],gate_dims=[9,16,3],
    epochs=60,learning_rate=.005,weight_decay=.0001,checkpoint='lowest external validation BCE/CE',
    decision_threshold=.5,bootstrap='reuse locked 10000 stratified group weights',native_abstention=False)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csvwrite(path,rows,fields=None):
    rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cell(seed,fold):return OUT/'cells'/f'seed_{seed}'/f'fold_{fold:02d}'
def frozen(seed,fold):return BASE/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'
def split(seed,fold):return BASE/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz'

def probability_features(p,app):
    p=np.where(app,p,.5)
    return np.stack((1-p,p),axis=2).reshape(len(p),6).astype(np.float32)
def gating_features(p,app):return np.c_[probability_features(p,app),app.astype(np.float32)]
def torchlib():
    import torch
    torch.set_num_threads(1)
    return torch
def make_net(dims):
    torch=torchlib();layers=[]
    for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(torch.nn.Linear(a,b))
        if i<len(dims)-2:layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)
def gate_probability(logits,x):
    torch=torchlib();mask=x[:,6:]>0
    weights=torch.softmax(logits.masked_fill(~mask,-1e9),dim=1)*mask
    den=weights.sum(1,keepdim=True);weights=weights/den.clamp_min(1e-12)
    prob=(weights*x[:,[1,3,5]]).sum(1)
    prob=torch.where(mask.any(1),prob,torch.full_like(prob,.5))
    return prob,weights
def train_network(kind,x,y,xv,yv,seed):
    torch=torchlib();torch.manual_seed(seed);net=make_net(CONFIG[kind+'_dims'])
    xx=torch.tensor(x);yy=torch.tensor(y,dtype=torch.long);vx=torch.tensor(xv);vy=torch.tensor(yv,dtype=torch.long)
    opt=torch.optim.Adam(net.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    def loss(z,labels):
        logits=net(z)
        if kind=='gate':
            prob,_=gate_probability(logits,z)
            return torch.nn.functional.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),labels.float())
        return torch.nn.functional.cross_entropy(logits,labels)
    best=float('inf');state=None;epoch_best=None;history=[]
    for epoch in range(CONFIG['epochs']):
        net.train();opt.zero_grad();trainloss=loss(xx,yy);trainloss.backward();opt.step();net.eval()
        with torch.no_grad():val=float(loss(vx,vy))
        if not np.isfinite(val):raise RuntimeError('Non-finite validation loss')
        history.append(val)
        if val<best:best=val;state={k:v.detach().clone() for k,v in net.state_dict().items()};epoch_best=epoch+1
    net.load_state_dict(state);net.eval()
    return net,dict(params=sum(p.numel() for p in net.parameters()),best_epoch=epoch_best,validation_loss=best,validation_history=history)

def prepare():
    required=[BASE/'PROTOCOL.md',BASE/'protocol.json',BASE/'benchmark_results.csv',BASE/'BENCHMARK.md',BASE/'RUNS.md',BASE/'features.npz',BASE/'bootstrap_draws.npz']
    required += [frozen(s,f)/name for s in SEEDS for f in range(20) for name in ('train_oof.npz','validation.npz','test_clean.npz','oof_lineage.json')]
    required += [split(s,f) for s in SEEDS for f in range(20)]
    absent=[str(p) for p in required if not p.exists()]
    if absent:raise FileNotFoundError('STOP: locked inputs missing; do not rebuild: '+str(absent[:10]))
    old=json.loads((BASE/'protocol.json').read_text());assert old['seeds']==SEEDS and old['per_group']==2000
    if (OUT/'preflight.json').exists():
        assert json.loads((OUT/'config.json').read_text())==CONFIG
        return
    OUT.mkdir(parents=True,exist_ok=True);backup=OUT/'original_tables';backup.mkdir(exist_ok=True)
    for name in ('benchmark_results.csv','BENCHMARK.md','RUNS.md'):shutil.copy2(BASE/name,backup/name)
    dump(OUT/'config.json',CONFIG)
    # Coordinator partitions labels; fusion trainers subsequently receive TRAIN/VAL labels only.
    with np.load(BASE/'features.npz') as z:y=z['y'];groups=z['group']
    frozen_files=sorted((BASE/'frozen_predictions').rglob('*'))
    protected=[p for p in frozen_files if p.is_file()]+sorted((BASE/'splits').rglob('*.npz'))
    protected += [BASE/'PROTOCOL.md',BASE/'protocol.json',BASE/'features.npz',BASE/'bootstrap_draws.npz']
    protected += sorted((BASE/'fold_runs').glob('seed_*/fold_*/models.joblib'))+sorted((BASE/'fold_runs').glob('seed_*/fold_*/networks.pt'))
    hashes={str(p.relative_to(BASE)):sha(p) for p in protected}
    for seed in SEEDS:
        for fold in range(20):
            with np.load(split(seed,fold)) as z:tr,va,te,inner=z['train'],z['validation'],z['test'],z['inner_fold']
            assert (len(tr),len(va),len(te))==(34000,4000,2000)
            assert not(set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
            with np.load(frozen(seed,fold)/'train_oof.npz') as z:
                assert np.array_equal(z['index'],tr) and np.array_equal(z['inner_fold'],inner[tr])
                assert z['p'].shape==(34000,3) and np.isfinite(z['p']).all()
            for k in range(3):assert not set(groups[tr[inner[tr]==k]])&set(groups[tr[inner[tr]!=k]])
            lineage=json.loads((frozen(seed,fold)/'oof_lineage.json').read_text());assert len(lineage)==6 and all(r['group_disjoint'] for r in lineage)
            for r in lineage:
                held=inner[tr]==r['inner'];fit=tr[~held];pred=tr[held]
                assert hashlib.sha256(fit.tobytes()).hexdigest()==r['fit_indices_sha256']
                assert hashlib.sha256(pred.tobytes()).hexdigest()==r['predict_indices_sha256']
            dest=cell(seed,fold);dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'labels_trainval.npz',train_index=tr,validation_index=va,train_y=y[tr],validation_y=y[va])
            np.savez_compressed(dest/'labels_test.npz',index=te,y=y[te],group=groups[te])
    refs=readcsv(BASE/'benchmark_results.csv');refs={r['method']:r for r in refs if r['method'] in ('V1_Average','V2_Yager','V3_Full')}
    dump(OUT/'preflight.json',dict(status='ready',protected_sha256=hashes,protected_files=len(hashes),outer_folds_checked=200,inner_splits_checked=600,source_oof_lineage_entries_checked=1200,reference_rows=refs,view_retraining=False))
    print('Preflight ready: 200 folds, 600 inner splits; immutable sources',len(hashes),flush=True)

def train_cell(seed,fold):
    threadpool_limits(1);dest=cell(seed,fold)
    if (dest/'TRAINED.json').exists():return dict(seed=seed,fold=fold,status='trained_cached')
    if (OUT/'TEST_OPENED.json').exists():raise RuntimeError('No training allowed after the global test barrier opens')
    t=time.perf_counter()
    # Deliberate allowlist: no test frozen outputs, labels, base models or features.
    with np.load(dest/'labels_trainval.npz') as z:y=z['train_y'];yv=z['validation_y'];tr=z['train_index'];va=z['validation_index']
    with np.load(frozen(seed,fold)/'train_oof.npz') as z:
        assert np.array_equal(z['index'],tr);x=probability_features(z['p'],z['app']);gx=gating_features(z['p'],z['app'])
    with np.load(frozen(seed,fold)/'validation.npz') as z:
        assert np.array_equal(z['index'],va);xv=probability_features(z['p'],z['app']);gv=gating_features(z['p'],z['app'])
    lr=LogisticRegression(random_state=seed,**CONFIG['logistic']);lr.fit(x,y);joblib.dump(lr,dest/'stack_logistic.joblib')
    mlp,mm=train_network('mlp',x,y,xv,yv,seed);gate,gm=train_network('gate',gx,y,gv,yv,seed)
    torch=torchlib();torch.save({'mlp':mlp.state_dict(),'gate':gate.state_dict()},dest/'fusion_heads.pt')
    modelsha={name:sha(dest/name) for name in ('stack_logistic.joblib','fusion_heads.pt')}
    r=dict(seed=seed,fold=fold,status='trained',seconds=time.perf_counter()-t,params={METHODS[0]:int(lr.coef_.size+lr.intercept_.size),METHODS[1]:mm['params'],METHODS[2]:gm['params']},mlp=mm,gate=gm,model_sha256=modelsha,test_predictions_read=False,test_labels_read=False,view_detectors_fitted=0)
    dump(dest/'TRAINED.json',r);return {k:r[k] for k in ('seed','fold','status','seconds')}

def train_all(workers):
    assert (OUT/'preflight.json').exists()
    failures=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(train_cell,s,f):(s,f) for s in SEEDS for f in range(20)}
        for i,fut in enumerate(as_completed(jobs),1):
            try:r=fut.result()
            except Exception as e:r=dict(seed=jobs[fut][0],fold=jobs[fut][1],status='failed',error=repr(e),traceback=traceback.format_exc());failures.append(r)
            with (OUT/'execution.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(r)+'\n')
            if i%20==0 or failures:print(i,'/200',round(time.perf_counter()-start,1),'seconds',r,flush=True)
    if failures:raise RuntimeError(f'{len(failures)} training failures; see execution.jsonl')
    seals={f'{s}/{f}':sha(cell(s,f)/'TRAINED.json') for s in SEEDS for f in range(20)}
    dump(OUT/'TRAINING_BARRIER.json',dict(all_200_trained=True,training_manifest_hashes=seals,seconds=time.perf_counter()-start))

def cf(y,p):return np.bincount(2*y.astype(int)+p.astype(int),minlength=4).astype(float)
def metrics(c):
    tn,fp,fn,tp=np.moveaxis(np.asarray(c),-1,0)
    a=2*tn+fp+fn;b=2*tp+fp+fn;n=tn+fp+fn+tp
    f0=np.divide(2*tn,a,out=np.zeros_like(tn),where=a>0);f1=np.divide(2*tp,b,out=np.zeros_like(tp),where=b>0)
    return (f0+f1)/2,(tn+tp)/np.maximum(n,1)
def evaluate():
    barrier=json.loads((OUT/'TRAINING_BARRIER.json').read_text());assert barrier['all_200_trained']
    for s in SEEDS:
        for f in range(20):assert sha(cell(s,f)/'TRAINED.json')==barrier['training_manifest_hashes'][f'{s}/{f}']
    if not (OUT/'TEST_OPENED.json').exists():dump(OUT/'TEST_OPENED.json',dict(training_complete=True,models_locked=True,time_unix=time.time()))
    torch=torchlib();done=0
    for seed in SEEDS:
        for fold in range(20):
            dest=cell(seed,fold)
            if (dest/'EVALUATED.json').exists():done+=1;continue
            modelinfo=json.loads((dest/'TRAINED.json').read_text())
            for name,value in modelinfo['model_sha256'].items():assert sha(dest/name)==value
            lr=joblib.load(dest/'stack_logistic.joblib');state=torch.load(dest/'fusion_heads.pt',map_location='cpu',weights_only=True)
            mlp=make_net(CONFIG['mlp_dims']);mlp.load_state_dict(state['mlp']);mlp.eval();gate=make_net(CONFIG['gate_dims']);gate.load_state_dict(state['gate']);gate.eval()
            # First test access occurs after ALL 200 training cells were sealed.
            with np.load(frozen(seed,fold)/'test_clean.npz') as z:p=z['p'];app=z['app'];index=z['index']
            with np.load(dest/'labels_test.npz') as z:
                assert np.array_equal(index,z['index']);y=z['y']
            x=probability_features(p,app);gx=gating_features(p,app)
            with torch.no_grad():
                predp=[lr.predict_proba(x)[:,1],torch.softmax(mlp(torch.tensor(x)),1)[:,1].numpy()]
                gp,weights=gate_probability(gate(torch.tensor(gx)),torch.tensor(gx));predp.append(gp.numpy());weights=weights.numpy()
            assert np.all(weights[~app]==0);np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
            predictions=np.stack(predp);binary=(predictions>=.5).astype(np.int8)
            np.savez_compressed(dest/'predictions.npz',index=index,y=y,probabilities=predictions,predictions=binary,attention_weights=weights,applicability=app)
            rows=[]
            for method,yp in zip(METHODS,binary):
                confusion=cf(y,yp);mf,ac=metrics(confusion)
                rows.append(dict(seed=seed,fold=fold,method=method,macro_f1=float(mf),accuracy=float(ac),tn=int(confusion[0]),fp=int(confusion[1]),fn=int(confusion[2]),tp=int(confusion[3]),params=modelinfo['params'][method],calls_per_sample=float(app.sum(1).mean()),learned_view_calls_per_sample=float(app[:,:2].sum(1).mean()),conditional_tls_calls_per_sample=float(app[:,2].mean()),native_abstention='n',trained_layer='y'))
            csvwrite(dest/'metrics.csv',rows);dump(dest/'EVALUATED.json',dict(seed=seed,fold=fold,test_evaluation_count=1,predictions_sha256=sha(dest/'predictions.npz')));done+=1
        print('evaluated seed',seed,flush=True)
    assert done==200

def summarize():
    rows=[];counts=np.zeros((3,10,20,4))
    for si,s in enumerate(SEEDS):
        for f in range(20):
            dest=cell(s,f);info=json.loads((dest/'EVALUATED.json').read_text());assert sha(dest/'predictions.npz')==info['predictions_sha256']
            rr=readcsv(dest/'metrics.csv');rows.extend(rr)
            for mi,r in enumerate(rr):counts[mi,si,f]=[float(r[k]) for k in ('tn','fp','fn','tp')]
    df=pd.DataFrame(rows);csvwrite(OUT/'per_fold_results.csv',rows)
    with np.load(BASE/'bootstrap_draws.npz') as z:w=z['group_weights'];ref_f1={m:z[m+'__pooled'] for m in ('V1_Average','V2_Yager','V3_Full')}
    assert w.shape==(10000,20)
    old=pd.read_csv(BASE/'per_fold_results.csv');old=old[old.condition=='clean']
    summary=[];seedrows=[];paired=[];boot={}
    for mi,m in enumerate(METHODS):
        f1,acc=metrics(counts[mi]);pooled,pa=metrics(counts[mi].sum(1));b1=w@f1.mean(0)/20;ba=w@acc.mean(0)/20
        bp=np.stack([metrics(w@counts[mi,si])[0] for si in range(10)],1).mean(1)
        lo,hi=np.quantile(b1,[.025,.975]);pl,ph=np.quantile(bp,[.025,.975]);al,ah=np.quantile(ba,[.025,.975]);local=df[df.method==m]
        row=dict(method=m,status='completed_frozen_head_only',macro_f1_fold_mean=float(f1.mean()),macro_f1_fold_std=float(f1.std(ddof=1)),macro_f1_ci95_low=float(lo),macro_f1_ci95_high=float(hi),accuracy_fold_mean=float(acc.mean()),accuracy_fold_std=float(acc.std(ddof=1)),macro_f1_pooled_seed_mean=float(pooled.mean()),macro_f1_pooled_seed_std=float(pooled.std(ddof=1)),pooled_macro_f1_ci95_low=float(pl),pooled_macro_f1_ci95_high=float(ph),accuracy_pooled_seed_mean=float(pa.mean()),accuracy_pooled_seed_std=float(pa.std(ddof=1)),accuracy_ci95_low=float(al),accuracy_ci95_high=float(ah),calls_per_sample=float(local.calls_per_sample.astype(float).mean()),learned_view_calls_per_sample=float(local.learned_view_calls_per_sample.astype(float).mean()),conditional_tls_calls_per_sample=float(local.conditional_tls_calls_per_sample.astype(float).mean()),trained_layer='y',params=int(local.params.iloc[0]),native_abstention='n',native_abstention_scope='none; binary argmax without rejection',native_coverage_mean=1.,unknown_rate=0.,suspicious_rate=0.,seeds=10,folds_per_seed=20,frozen_output_fusion='y',experiment_version=CONFIG['version'],method_display={'V4_StackLogistic_Frozen':'V4 stack-logistic','V5_StackMLP32x16_Frozen':'V5 stack-MLP (32-16)','V6_Attention_Frozen':'V6 attention'}[m])
        summary.append(row);boot[m+'__foldmean']=b1;boot[m+'__pooled']=bp;boot[m+'__accuracy']=ba
        for si,s in enumerate(SEEDS):seedrows.append(dict(seed=s,method=m,macro_f1=float(pooled[si]),accuracy=float(pa[si])))
        for ref in ('V1_Average','V2_Yager','V3_Full'):
            ra=old[old.method==ref].pivot(index='seed',columns='fold',values='accuracy').reindex(index=SEEDS,columns=range(20)).to_numpy();rab=w@ra.mean(0)/20
            delta=ba-rab;dl,dh=np.quantile(delta,[.025,.975]);fl,fh=np.quantile(bp-ref_f1[ref],[.025,.975])
            paired.append(dict(method=m,reference=ref,accuracy_delta=float(acc.mean()-ra.mean()),accuracy_delta_ci95_low=float(dl),accuracy_delta_ci95_high=float(dh),pooled_macro_f1_delta=float(pooled.mean()-ref_f1[ref].mean()),pooled_f1_bootstrap_delta_mean=float((bp-ref_f1[ref]).mean()),pooled_f1_delta_ci95_low=float(fl),pooled_f1_delta_ci95_high=float(fh)))
    # Point differences use observed references, not the mean of bootstrap draws.
    refs={r['method']:r for r in readcsv(OUT/'original_tables/benchmark_results.csv')}
    for r in paired:
        r['pooled_macro_f1_delta']=next(x['macro_f1_pooled_seed_mean'] for x in summary if x['method']==r['method'])-float(refs[r['reference']]['macro_f1_pooled_seed_mean'])
    csvwrite(OUT/'results.csv',summary);csvwrite(OUT/'per_seed_results.csv',seedrows);csvwrite(OUT/'paired_comparisons.csv',paired);np.savez_compressed(OUT/'bootstrap_draws.npz',**boot)
    print(pd.DataFrame(summary)[['method','macro_f1_pooled_seed_mean','accuracy_pooled_seed_mean','params','calls_per_sample']].to_string(index=False))

def audit_saved_predictions():
    from sklearn.metrics import accuracy_score,f1_score
    barrier=json.loads((OUT/'TRAINING_BARRIER.json').read_text())
    opened=json.loads((OUT/'TEST_OPENED.json').read_text())['time_unix']
    checked=0;allrows=[]
    for seed in SEEDS:
        for fold in range(20):
            dest=cell(seed,fold)
            assert sha(dest/'TRAINED.json')==barrier['training_manifest_hashes'][f'{seed}/{fold}']
            assert (dest/'TRAINED.json').stat().st_mtime<=opened
            train=json.loads((dest/'TRAINED.json').read_text())
            assert not train['test_predictions_read'] and not train['test_labels_read'] and train['view_detectors_fitted']==0
            for name,value in train['model_sha256'].items():assert sha(dest/name)==value
            evaluated=json.loads((dest/'EVALUATED.json').read_text());assert evaluated['test_evaluation_count']==1
            assert evaluated['predictions_sha256']==sha(dest/'predictions.npz')
            with np.load(dest/'predictions.npz') as z:
                p=z['probabilities'];yp=z['predictions'];y=z['y'];weights=z['attention_weights'];app=z['applicability'];index=z['index']
            with np.load(split(seed,fold)) as z:assert np.array_equal(index,z['test'])
            assert p.shape==(3,2000) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            assert np.array_equal(yp,(p>=.5).astype(np.int8));assert np.all(weights[~app]==0)
            np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
            rows=readcsv(dest/'metrics.csv');assert [r['method'] for r in rows]==METHODS
            for i,r in enumerate(rows):
                np.testing.assert_allclose(float(r['macro_f1']),f1_score(y,yp[i],labels=[0,1],average='macro',zero_division=0),atol=1e-12)
                np.testing.assert_allclose(float(r['accuracy']),accuracy_score(y,yp[i]),atol=1e-12)
                assert int(r['params'])==[7,786,211][i]
                allrows.append(r);checked+=1
    result=readcsv(OUT/'results.csv')
    for m,r in zip(METHODS,result):
        rr=[x for x in allrows if x['method']==m]
        for metric in ('macro_f1','accuracy'):
            values=np.array([float(x[metric]) for x in rr])
            np.testing.assert_allclose(float(r[metric+'_fold_mean']),values.mean(),atol=1e-12)
            np.testing.assert_allclose(float(r[metric+'_fold_std']),values.std(ddof=1),atol=1e-12)
    dump(OUT/'prediction_audit.json',dict(status='passed',sklearn_metric_rows_verified=checked,model_hashes_verified=400,training_manifests_before_test_opening=200,attention_masks_verified=200,test_inference_repeated=False))

def publish():
    audit_saved_predictions()
    pre=json.loads((OUT/'preflight.json').read_text());changed=[name for name,value in pre['protected_sha256'].items() if sha(BASE/name)!=value]
    if changed:raise RuntimeError('Protected files changed: '+str(changed[:5]))
    original=readcsv(OUT/'original_tables/benchmark_results.csv');new=readcsv(OUT/'results.csv');existing=readcsv(BASE/'benchmark_results.csv')
    oldcols=list(original[0]);assert len(existing) in (len(original),len(original)+3)
    for a,b in zip(original,existing[:len(original)]):assert all(a[k]==b[k] for k in oldcols),'Original reference row changed'
    allfields=list(dict.fromkeys([*oldcols,*(k for r in new for k in r)]))
    for r in new:
        assert r['native_abstention']=='n' and r['trained_layer']=='y'
        assert abs(float(r['calls_per_sample'])-float(pre['reference_rows']['V1_Average']['calls_per_sample']))<1e-12
    csvwrite(BASE/'benchmark_results.csv',[*original,*new],allfields)
    lines=['','<!-- TRAINED_FUSION_FAST_V1 -->','## Frozen-output trained fusion addendum','',
        'Added on the existing 20 held-out groups × 10 seeds, using only existing OOF training probabilities and external validation groups. No view detector, OOD model or EDL model was fitted or queried. The three new model IDs are namespaced to preserve the earlier V4/V5/V6 rows, including the earlier EDL row.','',
        'The requested reference numbers do not identify this locked protocol. Its actual averaging, Yager-without-OOD, and full-pipeline rows are retained below unchanged. Two learned-view calls plus conditional TLS give 2.050325 total calls/sample; 2.0000 is the HGB-only component, not the total.','',
        '| Method | Fold Macro-F1 mean±std | 95% group CI | Accuracy mean±std | Pooled F1 mean±std | Trained fusion? | Params | Native abstention? | Calls/sample |',
        '|---|---:|---|---:|---:|:---:|---:|:---:|---:|']
    def show(r):
        num=lambda k:float(r[k])
        return f"| {r.get('method_display') or r['method']} | {num('macro_f1_fold_mean'):.4f}±{num('macro_f1_fold_std'):.4f} | [{num('macro_f1_ci95_low'):.4f}, {num('macro_f1_ci95_high'):.4f}] | {num('accuracy_fold_mean'):.4f}±{num('accuracy_fold_std'):.4f} | {num('macro_f1_pooled_seed_mean'):.4f}±{num('macro_f1_pooled_seed_std'):.4f} | {r['trained_layer']} | {int(num('params'))} | {r['native_abstention']} | {num('calls_per_sample'):.6f} |"
    lines += [show(pre['reference_rows'][m]) for m in ('V1_Average','V2_Yager','V3_Full')]+[show(r) for r in new]
    lines += ['',
        '**Comparison:** the trained fusion heads require OOF fusion training and have no native rejection; Yager requires no trained fusion layer and natively emits suspicious/unknown, while the measured accuracies differ and must not be described as proven equal.','',
        'Each test fold contains one class: the locked fixed-label binary fold F1 has maximum 0.5; the pooled F1 column instead merges 40,000 predictions per seed before averaging across ten seeds. Exact fold/seed values, paired group-bootstrap differences and full uncertainty fields are in `trained_fusion_fast_v1/`. Bootstrap uses the original 10,000 group-weight draws. No equivalence margin was specified or tested.','',
        'Input is the six class probabilities [Stats benign/malicious, Temporal benign/malicious, TLS benign/malicious], with unavailable pairs fixed at [0.5,0.5]. Logistic regression has C=1; MLP has 32 and 16 hidden units. Attention uses these probabilities plus three masks, a 16-unit hidden layer and a three-way masked softmax. Both neural heads use 60 fixed full-batch Adam epochs (lr=.005, weight decay=.0001), selecting the checkpoint by validation loss only. All three use the fixed .5 binary boundary. The gate falls back to p=.5 if every view is absent, without emitting unknown.','',
        'Only clean classification metrics and bookkeeping were requested for these appended rows. Their unrelated perturbation/latency/selective-curve columns are blank, not inherited from a different model. Existing figures and original rows remain the original benchmark results. Full reproduction and immutable-input validation: [FAST_FUSION_PROTOCOL.md](trained_fusion_fast_v1/FAST_FUSION_PROTOCOL.md), [validation.json](trained_fusion_fast_v1/validation.json).','<!-- END_TRAINED_FUSION_FAST_V1 -->','']
    section='\n'.join(lines)
    oldtext=(OUT/'original_tables/BENCHMARK.md').read_text(encoding='utf-8')
    current=(BASE/'BENCHMARK.md').read_text(encoding='utf-8')
    assert current==oldtext or current.startswith(oldtext+'\n'),'Concurrent benchmark markdown change'
    (BASE/'BENCHMARK.md').write_text(oldtext+section,encoding='utf-8')
    (OUT/'RESULTS.md').write_text(section,encoding='utf-8')
    barrier=json.loads((OUT/'TRAINING_BARRIER.json').read_text())
    events=[json.loads(s) for s in (OUT/'execution.jsonl').read_text().splitlines()]
    failures=[r for r in events if r['status']=='failed']
    runlines=['','## Frozen-output trained fusion addendum','',
        'Protocol: [trained_fusion_fast_v1/FAST_FUSION_PROTOCOL.md](trained_fusion_fast_v1/FAST_FUSION_PROTOCOL.md). Three heads only; no view retraining/inference or new EDL training. Seeds 42–51, all 20 folds per seed. Fixed C=1 logistic regression; MLP 32–16; attention hidden width 16; Adam lr=.005, weight decay=.0001, 60 full-batch epochs with validation-only checkpoint selection.',
        f"Training completed all 200 cells / 600 fusion heads in {barrier['seconds']:.2f} seconds with six workers. Recorded training failures: {len(failures)}. Test evaluation opened after every training manifest was sealed; each cell was evaluated once. Source hashes, environment and test audit accompany this addendum.",
        '', 'Commands executed from the workspace root:', '', '```powershell',
        '.\\.codex_mad_etd_v24_venv\\Scripts\\python.exe -X utf8 -m pytest tests\\test_frozen_fusion_fast.py -q',
        *[f'.\\.codex_mad_etd_v24_venv\\Scripts\\python.exe -X utf8 scripts\\frozen_fusion_fast.py {stage}'+(' --workers 6' if stage=='train' else '') for stage in ('prepare','train','evaluate','summarize','publish')],
        '```','',
        'No failed training/evaluation run or discarded result occurred in this addendum. Existing PyTorch was loaded from the user-site installation; no packages were installed. Existing test predictions were not passed to trainers. The requested reference scores and total-call figure differed from the locked files, so the actual reference rows were preserved; total calls and HGB-only calls are separate columns. Accuracy equality was not assumed. This log does not replace the earlier run history.','']
    runsection='\n'.join(runlines)
    oldruns=(OUT/'original_tables/RUNS.md').read_text(encoding='utf-8')
    currentruns=(BASE/'RUNS.md').read_text(encoding='utf-8')
    assert currentruns==oldruns or currentruns.startswith(oldruns+'\n'),'Concurrent run log change'
    (BASE/'RUNS.md').write_text(oldruns+runsection,encoding='utf-8')
    (OUT/'RUNS.md').write_text(runsection,encoding='utf-8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for source in (ROOT/'scripts/frozen_fusion_fast.py',ROOT/'tests/test_frozen_fusion_fast.py'):
        shutil.copy2(source,snapshot/source.name)
    import sklearn,scipy
    torch=torchlib()
    dump(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,torch=torch.__version__,source_sha256={p.name:sha(p) for p in snapshot.iterdir()}))
    outrows=readcsv(BASE/'benchmark_results.csv')
    for a,b in zip(original,outrows):assert all(a[k]==b[k] for k in oldcols)
    done=dict(status='passed',trained_fusion_models=600,view_detector_fit_calls=0,view_detector_inference_calls=0,edl_models_trained=0,seed_fold_cells=200,test_evaluations_per_cell=1,protected_files_verified=len(pre['protected_sha256']),protected_files_changed=0,original_reference_rows_preserved=len(original),appended_rows=3,missing_attention_weights_zero=True,global_training_before_test_barrier=True)
    dump(OUT/'validation.json',done)
    print(json.dumps(done))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','summarize','publish']);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    {'prepare':prepare,'train':lambda:train_all(a.workers),'evaluate':evaluate,'summarize':summarize,'publish':publish}[a.stage]()
if __name__=='__main__':main()
