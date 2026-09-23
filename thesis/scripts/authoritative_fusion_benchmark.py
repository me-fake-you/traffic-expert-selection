"""Reproducible USTC 11/30-D benchmark; run --help for stages.

No historical predictions or historical target metrics are used.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse, csv, gzip, hashlib, heapq, json, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
# Reuse installed CPU torch without enabling the broken user-site .pth startup.
USER_SITE = Path.home()/'AppData/Roaming/Python/Python312/site-packages'
if USER_SITE.exists(): sys.path.append(str(USER_SITE))
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from threadpoolctl import threadpool_limits
from mad_etd.features import STATS_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES, extract_stats_features, extract_temporal_features
from mad_etd.schemas import DetectorInput, SequenceFeatures
from mad_etd.detectors import TLSProtocolAgent

OUT = ROOT/'output/ustc_authoritative_fusion_v1'
SEEDS = list(range(42,52))
CONDITIONS = ['clean','no_stats','no_temporal','no_tls','stats_volume_mask','temporal_size_mask','temporal_timing_direction_mask']
METHODS = ['V0_Stats','V0_Temporal','V0_TLS','V1_Average','V2_Yager','V3_Full','V4_Logistic','V4_MLP','V5_Gating','V6_EDL','V3_OnDemand']
COVERAGES = [.1,.2,.2337,.3,.4,.5,.6,.7,.8,.9,1.0]
CONFIG = dict(version='ustc-authoritative-v1',seeds=SEEDS,sampling_seed=42,split_seed=42,
    per_group=2000,inner_folds=3,hgb=dict(max_iter=100,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,early_stopping=False,class_weight='balanced'),
    isolation_forest=dict(n_estimators=100,max_samples=1024,contamination='auto',n_jobs=1),
    neural_epochs=60,neural_lr=.005,weight_decay=.0001,bootstrap_iterations=10000,
    bootstrap_seed=42,conditions=CONDITIONS,coverages=COVERAGES,operating_coverage_validation=.8)

def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def csvwrite(path, rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metric(y,p):
    if not len(y):return (float('nan'),float('nan'))
    return float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0)),float(accuracy_score(y,p))
def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==CONFIG,'Protocol drift'
    else:dump(out/'protocol.json',CONFIG)
    if (out/'features.npz').exists():return
    heaps={};seen=set();sources=[];total=0;tlsagent=TLSProtocolAgent(contract_native_fields=True)
    for path in sorted((ROOT/'data/processed/ustc_tfc2016/v1/flows').rglob('*.jsonl.gz')):
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line); lab=r['labels'];y=int(lab['binary']=='malicious')
                g=('malware:' if y else 'benign:')+lab['family' if y else 'application'];sid=r['sample_id']
                if sid in seen:raise ValueError('Duplicate source sample ID '+sid)
                seen.add(sid);total+=1
                priority=int(hashlib.sha256(f"42:authoritative-v1:{g}:{sid}".encode()).hexdigest(),16)
                h=heaps.setdefault(g,[])
                item=(-priority,sid,r)
                if len(h)<2000:heapq.heappush(h,item)
                elif priority < -h[0][0]:heapq.heapreplace(h,item)
        print('source',path.name,'scanned',total,flush=True)
    gs=sorted(heaps);assert len(gs)==20 and sum(g.startswith('benign:') for g in gs)==10
    xs=[];xt=[];tf=[];tlsp=[];tlsm=[];app=[];ids=[];yy=[];group=[];manifest=[]
    safe=[]
    for gi,g in enumerate(gs):
        assert len(heaps[g])==2000,(g,len(heaps[g]))
        for _,sid,r in sorted(heaps[g],reverse=True):
            tls={k:v for k,v in r.get('tls',{}).items() if k in ('version','cipher_suite','certificate_valid','handshake_complete')}
            d=DetectorInput(stats=r['stats'],sequence=SequenceFeatures.model_validate(r['sequence']),tls=tls)
            # A visible version/cipher is required; no inferred TLS from ports/labels.
            ta=bool(tls.get('version') or tls.get('cipher_suite'))
            ev=tlsagent.analyze(d if ta else DetectorInput())
            mass=[ev.benign_support,ev.malicious_support,ev.uncertainty]
            p=mass[1]/max(mass[0]+mass[1],1e-12) if ta else .5
            ver=str(tls.get('version','')).lower()
            tf.append([float(ver in ('ssl3','tls1.0','tlsv1','tls1.1')),float(ver in ('tls1.3','tlsv1.3','quic')),float(tls.get('certificate_valid') is False),float(tls.get('certificate_valid') is True),float(tls.get('handshake_complete') is False)])
            xs.append(extract_stats_features(d));xt.append(extract_temporal_features(d));tlsp.append(p);tlsm.append(mass)
            app.append([bool(r['stats']),len(d.sequence.packet_lengths)>=1,ta]);ids.append(sid);yy.append(int(g.startswith('malware:')));group.append(gi)
            manifest.append(dict(index=len(ids)-1,sample_id=sid,group=g,label=yy[-1],source_file=r['provenance']['source_file'],tls_applicable=ta))
        print('features',g,flush=True)
    np.savez_compressed(out/'features.npz',stats=np.asarray(xs,np.float32),temporal=np.asarray(xt,np.float32),tls_features=np.asarray(tf,np.float32),tls_probability=np.asarray(tlsp),tls_mass=np.asarray(tlsm),app=np.asarray(app,bool),y=np.asarray(yy,np.int8),group=np.asarray(group,np.int8),sample_id=np.asarray(ids),group_names=np.asarray(gs))
    csvwrite(out/'sample_manifest.csv',manifest)
    rng=np.random.default_rng(42)
    benign=rng.permutation(np.arange(10));malware=rng.permutation(np.arange(10,20))
    splitrows=[];groups=np.asarray(group);y=np.asarray(yy)
    for seed in SEEDS:
        for fold in range(20):
            va=[int(next(x for x in benign if x!=fold)),int(next(x for x in malware if x!=fold))]
            # Rotate which validation groups are chosen by held-out group; independent of labels/results.
            va=[int([x for x in np.roll(benign,fold) if x!=fold][0]),int([x for x in np.roll(malware,fold) if x!=fold][0])]
            tr=np.flatnonzero(~np.isin(groups,[fold,*va]));val=np.flatnonzero(np.isin(groups,va));te=np.flatnonzero(groups==fold)
            inner=np.full(len(y),-1,np.int8)
            for k,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(tr,y[tr],groups[tr])):
                assert not set(groups[tr[a]])&set(groups[tr[b]])
                inner[tr[b]]=k
            sd=out/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz';sd.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(sd,train=tr,validation=val,test=te,inner_fold=inner)
            splitrows.append(dict(seed=seed,fold=fold,heldout=gs[fold],validation='|'.join(gs[x] for x in va),ntrain=len(tr),nvalidation=len(val),ntest=len(te),sha256=sha(sd)))
    csvwrite(out/'split_manifest.csv',splitrows)
    dump(out/'data_audit.json',dict(source_rows=total,selected_rows=len(y),source_files=sources,features_sha256=sha(out/'features.npz'),stats_dim=11,temporal_dim=30,tls_applicable=int(np.asarray(app)[:,2].sum()),tls_primitives=['legacy_version','modern_version','invalid_cert','valid_cert','incomplete_handshake'],selected_groups=gs))

def condition_arrays(data,condition):
    xs=data['stats'].copy();xt=data['temporal'].copy();app=data['app'].copy();tf=data['tls_features'].copy()
    if condition=='no_stats':xs[:]=np.nan;app[:,0]=False
    if condition=='no_temporal':xt[:]=np.nan;app[:,1]=False
    if condition=='no_tls':app[:,2]=False;tf[:]=0
    if condition=='stats_volume_mask':xs[:,[1,2,3,6,7,9,10]]=np.nan
    if condition=='temporal_size_mask':xt[:,3:14]=np.nan
    if condition=='temporal_timing_direction_mask':xt[:,14:30]=np.nan
    return xs,xt,tf,app

def hgb(seed):return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingClassifier(random_state=seed,**CONFIG['hgb']))
def masses(p):
    u=.08+.34*np.minimum(p,1-p)
    return np.stack(((1-p)*(1-u),p*(1-u),u),axis=-1)
def evidence(p,app,tlsmass,rel,shift=None,raw=None):
    m=masses(p);m[:,2]=tlsmass;m[~app]=[0,0,1]
    return dict(p=p,app=app,mass=m,reliability=np.broadcast_to(rel,p.shape).copy(),ood=np.zeros_like(p) if shift is None else shift,ood_raw=np.zeros_like(p) if raw is None else raw)
def freeze(path,ev,idx,**extra):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,index=idx,**ev,**extra)
def yager(ev,ood=False,available=None):
    app=ev['app'].copy() if available is None else available.copy()
    observed_app=app.copy()
    m=ev['mass'].copy();r=ev['reliability'].copy();shift=ev['ood']
    hard=(shift>=.99)&app;hard[:,2]=False
    if ood:
        a=np.clip((shift-.95)/.04,0,1);mr=np.maximum(.1,1-.9*a);mr[:,2]=1
        u=np.maximum(m[:,:,2],.25+.7*a);u[:,2]=m[:,2,2]
        m[:,:,0]=(1-ev['p'])*(1-u);m[:,:,1]=ev['p']*(1-u);m[:,:,2]=u
        r*=mr;app&=~hard
    b=np.zeros(len(app));mal=b.copy();u=np.ones(len(app));conf=b.copy()
    for j in range(3):
        rb=m[:,j,0]*r[:,j]*app[:,j];rm=m[:,j,1]*r[:,j]*app[:,j];ru=1-rb-rm
        c=b*rm+mal*rb
        b,mal,u=b*rb+b*ru+u*rb,mal*rm+mal*ru+u*rm,u*ru+c
        conf=1-(1-conf)*(1-c)
    p=mal/np.maximum(b+mal,1e-15)
    pred=(mal>=b).astype(np.int8);pred[(b+mal)==0]=0
    # Exact final=True FusionAgent._decide order. Codes 0/1 benign/malicious, 2 suspicious, 3 unknown.
    verdict=np.where(mal>=.25,2,3).astype(np.int8)
    verdict[(mal>=.8)&(u<=.25)]=1;verdict[(b>=.8)&(u<=.25)]=0
    conflict=(conf>=.3)&(mal>=.2);verdict[conflict]=2
    special=(u>=.5)|(~app.any(1))
    if ood:special|=hard.any(1)
    verdict[special]=np.where(mal[special]>=.3,2,3)
    both_hard=hard[:,:2].all(1) if ood else np.zeros(len(app),bool)
    if ood:
        verdict[both_hard]=3
        # FusionAgent returns before consuming TLS when BOTH primary views are hard OOD.
        p[both_hard]=0;pred[both_hard]=0
    score=np.maximum(b,mal)*(1-u)
    if ood:score*=1-np.max(shift[:,:2]*observed_app[:,:2],axis=1)
    score[both_hard]=0
    return p,pred,score,verdict
def meta_x(ev):return np.concatenate((np.where(ev['app'],ev['p'],.5),ev['app'].astype(float)),axis=1).astype(np.float32)
def raw_x(xs,xt,tf,app):return np.concatenate((xs,xt,tf,app.astype(float)),axis=1)

def torch_module():
    import torch
    torch.set_num_threads(1)
    return torch
def edl_loss(alpha,y,anneal):
    torch=torch_module();target=torch.nn.functional.one_hot(y,2).float();s=alpha.sum(1,keepdim=True)
    mse=((target-alpha/s)**2+alpha*(s-alpha)/(s*s*(s+1))).sum(1)
    at=target+(1-target)*alpha;st=at.sum(1,keepdim=True)
    kl=torch.lgamma(st).squeeze(1)-torch.lgamma(at).sum(1)+((at-1)*(torch.digamma(at)-torch.digamma(st))).sum(1)
    return (mse+anneal*kl).mean()
def train_net(kind,x,y,xv,yv,seed):
    torch=torch_module();torch.manual_seed(seed)
    dims=[x.shape[1],16,8,2] if kind=='mlp' else [x.shape[1],16,3] if kind=='gate' else [x.shape[1],32,16,2]
    layers=[]
    for a,b in zip(dims[:-1],dims[1:]):
        layers.append(torch.nn.Linear(a,b))
        if b!=dims[-1] or len(layers)<len(dims)-1:layers.append(torch.nn.ReLU())
    # Explicit construction avoids output-layer activation for gate/EDL/MLP.
    layers=[]
    for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(torch.nn.Linear(a,b))
        if i<len(dims)-2:layers.append(torch.nn.ReLU())
    net=torch.nn.Sequential(*layers)
    xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.long);vx=torch.tensor(xv,dtype=torch.float32);vy=torch.tensor(yv,dtype=torch.long)
    opt=torch.optim.Adam(net.parameters(),lr=CONFIG['neural_lr'],weight_decay=CONFIG['weight_decay'])
    best=float('inf');state=None;bestepoch=0;history=[]
    def loss(z,lab,epoch):
        logits=net(z)
        if kind=='edl':return edl_loss(torch.nn.functional.softplus(logits)+1,lab,min(1.,(epoch+1)/10))
        if kind=='gate':
            weights=torch.softmax(logits.masked_fill(z[:,3:]<=0,-1e9),dim=1)*z[:,3:]
            weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
            p=(weights*z[:,:3]).sum(1).clamp(1e-6,1-1e-6)
            return torch.nn.functional.binary_cross_entropy(p,lab.float())
        return torch.nn.functional.cross_entropy(logits,lab)
    for epoch in range(CONFIG['neural_epochs']):
        net.train();opt.zero_grad();l=loss(xx,yy,epoch);l.backward();opt.step()
        net.eval()
        with torch.no_grad():v=float(loss(vx,vy,CONFIG['neural_epochs']).item())
        history.append(v)
        if v<best:best=v;state={k:t.detach().clone() for k,t in net.state_dict().items()};bestepoch=epoch+1
    net.load_state_dict(state);net.eval()
    return net,dict(kind=kind,dims=dims,params=sum(p.numel() for p in net.parameters()),best_epoch=bestepoch,validation_loss=best,validation_history=history)
def net_predict(net,kind,x):
    torch=torch_module()
    with torch.no_grad():
        z=torch.tensor(x,dtype=torch.float32);logit=net(z)
        if kind=='edl':
            a=torch.nn.functional.softplus(logit)+1;s=a.sum(1);return (a[:,1]/s).numpy(),(2/s).numpy()
        if kind=='gate':
            w=torch.softmax(logit.masked_fill(z[:,3:]<=0,-1e9),1)*z[:,3:];w=w/w.sum(1,keepdim=True).clamp_min(1e-12)
            return (w*z[:,:3]).sum(1).numpy(),w.numpy()
        return torch.softmax(logit,1)[:,1].numpy(),None

def decision(method,ev,models,edlx):
    app=ev['app'];p=ev['p'];available=app.any(1)
    if method.startswith('V0_'):
        j=['Stats','Temporal','TLS'].index(method[3:]);prob=p[:,j];available=app[:,j];score=2*abs(prob-.5);verdict=np.where(available,prob>=.5,3).astype(np.int8);calls=available.astype(float)
    elif method=='V1_Average':
        prob=(p*app).sum(1)/np.maximum(app.sum(1),1);score=2*abs(prob-.5);verdict=np.where(available,prob>=.5,3).astype(np.int8);calls=app.sum(1)
    elif method in ('V2_Yager','V3_Full'):
        prob,pred,score,verdict=yager(ev,method=='V3_Full');calls=app.sum(1)
        return dict(p=prob,pred=pred,score=score,verdict=verdict,available=available,calls=calls)
    elif method=='V3_OnDemand':
        used=np.zeros_like(app);pending=np.ones(len(app),bool)
        for j in (0,1,2):
            used[:,j]=app[:,j]&pending
            prob,pred,score,verdict=yager(ev,True,used)
            pending=verdict>=2
        return dict(p=prob,pred=pred,score=score,verdict=verdict,available=available,calls=used.sum(1))
    elif method=='V4_Logistic':prob=models['lr'].predict_proba(meta_x(ev))[:,1];score=2*abs(prob-.5);verdict=np.where(available,prob>=.5,3).astype(np.int8);calls=app.sum(1)
    elif method in ('V4_MLP','V5_Gating'):
        kind='mlp' if method=='V4_MLP' else 'gate';prob,_=net_predict(models[kind],kind,meta_x(ev));score=2*abs(prob-.5);verdict=np.where(available,prob>=.5,3).astype(np.int8);calls=app.sum(1)
    elif method=='V6_EDL':
        prob,u=net_predict(models['edl'],'edl',edlx);score=1-u;verdict=np.where(available&(u<=.5),prob>=.5,3).astype(np.int8);calls=np.zeros(len(app))
    else:raise ValueError(method)
    prob=np.where(available,prob,.5);score=np.where(available,score,0)
    return dict(p=prob,pred=(prob>=.5).astype(np.int8),score=score,verdict=verdict,available=available,calls=calls)

def run_fold(outstr,seed,fold):
    threadpool_limits(1);out=Path(outstr);dest=out/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}'
    if (dest/'DONE.json').exists():return dict(seed=seed,fold=fold,resumed=True)
    dest.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    with np.load(out/'features.npz',allow_pickle=False) as d:data={k:d[k] for k in d.files}
    with np.load(out/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz') as d:sp={k:d[k] for k in d.files}
    tr,va,te=sp['train'],sp['validation'],sp['test'];y=data['y'];base=condition_arrays(data,'clean')
    ff=out/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}';ff.mkdir(parents=True,exist_ok=True)
    p_oof=np.full((len(tr),3),.5);p_oof[:,2]=data['tls_probability'][tr]
    pval=np.full((len(va),3),.5);pval[:,2]=data['tls_probability'][va]
    oodval=np.zeros_like(pval);rawval=np.zeros_like(pval);viewmodels=[];gates=[];viewms=[];oodms=[]
    rel=np.zeros(3);lineage=[]
    for j,x in enumerate(base[:2]):
        for inner in range(3):
            held=sp['inner_fold'][tr]==inner;itr=tr[~held];ite=tr[held]
            model=hgb(seed);model.fit(x[itr],y[itr]);p_oof[held,j]=model.predict_proba(x[ite])[:,1]
            lineage.append(dict(view=j,inner=inner,fit_indices_sha256=hashlib.sha256(itr.tobytes()).hexdigest(),predict_indices_sha256=hashlib.sha256(ite.tobytes()).hexdigest(),group_disjoint=not bool(set(data['group'][itr])&set(data['group'][ite]))))
        model=hgb(seed);model.fit(x[tr],y[tr]);pval[:,j]=model.predict_proba(x[va])[:,1];viewmodels.append(model)
        gate=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RobustScaler(),IsolationForest(random_state=seed,**CONFIG['isolation_forest']))
        gate.fit(x[tr]);rv=-gate.score_samples(x[va]);cal=np.sort(rv);gates.append((gate,cal));rawval[:,j]=rv;oodval[:,j]=np.searchsorted(cal,rv,side='right')/len(cal)
        mask=data['app'][va,j];rel[j]=np.clip(1-np.mean((pval[mask,j]-y[va][mask])**2),0,1) if mask.any() else 0
        # Warmed single-worker batch-128 component timing, no test labels involved.
        xx=x[va[:128]];model.predict_proba(xx);gate.score_samples(xx)
        times=[];ot=[]
        for _ in range(3):
            t=time.perf_counter();model.predict_proba(xx);times.append((time.perf_counter()-t)*1000/len(xx))
            t=time.perf_counter();gate.score_samples(xx);ot.append((time.perf_counter()-t)*1000/len(xx))
        viewms.append(float(np.median(times)));oodms.append(float(np.median(ot)))
    tlsmask=data['app'][va,2];rel[2]=(1-np.mean((pval[tlsmask,2]-y[va][tlsmask])**2)) if tlsmask.any() else 0
    evalev=evidence(pval,data['app'][va],data['tls_mass'][va],rel,oodval,rawval)
    train_ev=evidence(p_oof,data['app'][tr],data['tls_mass'][tr],np.zeros(3))
    # Only OOF probabilities and applicability are consumed by learned fusers.
    freeze(ff/'train_oof.npz',dict(p=p_oof,app=data['app'][tr]),tr,inner_fold=sp['inner_fold'][tr])
    freeze(ff/'validation.npz',evalev,va)
    dump(ff/'oof_lineage.json',lineage)
    mx=meta_x(train_ev);mv=meta_x(evalev)
    lr=LogisticRegression(C=1.,max_iter=500,random_state=seed);lr.fit(mx,y[tr]);models={'lr':lr};netmeta={}
    for kind in ('mlp','gate'):
        models[kind],netmeta[kind]=train_net(kind,mx,y[tr],mv,y[va],seed)
    allraw=raw_x(*base);rawpipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler());rx=rawpipe.fit_transform(allraw[tr]).astype(np.float32);rv=rawpipe.transform(allraw[va]).astype(np.float32)
    models['edl'],netmeta['edl']=train_net('edl',rx,y[tr],rv,y[va],seed)
    validation={m:decision(m,evalev,models,rv) for m in METHODS}
    thresholds={m:float(np.quantile(d['score'][d['available']],.2)) if d['available'].any() else float('inf') for m,d in validation.items()}
    params={m:0 for m in METHODS};params.update(V4_Logistic=int(lr.coef_.size+lr.intercept_.size),V4_MLP=netmeta['mlp']['params'],V5_Gating=netmeta['gate']['params'],V6_EDL=netmeta['edl']['params'])
    # Persist fitted artifacts and validation decisions before opening test labels.
    joblib.dump(dict(views=viewmodels,ood_gates=gates,lr=lr,edl_preprocess=rawpipe,reliability=rel),dest/'models.joblib',compress=3)
    torch=torch_module()
    torch.save({k:models[k].state_dict() for k in ('mlp','gate','edl')},dest/'networks.pt')
    dump(dest/'training.json',dict(seed=seed,fold=fold,neural=netmeta,thresholds=thresholds,params=params,view_ms=viewms,ood_ms=oodms,reliability=rel.tolist()))
    # Time frozen decision layers on validation only. Component costs are explicitly not wall-clock end-to-end latency.
    decisionms={}
    small={k:v[:128] for k,v in evalev.items()}
    for method in METHODS:
        decision(method,small,models,rv[:128]);tt=[]
        for _ in range(3):
            t=time.perf_counter();decision(method,small,models,rv[:128]);tt.append((time.perf_counter()-t)*1000/128)
        decisionms[method]=float(np.median(tt))
    result={};rows=[];pertrows=[];clean={};freezehash={}
    for condition in CONDITIONS:
        xs,xt,tf,app=condition_arrays(data,condition);pt=np.full((len(te),3),.5);pt[:,2]=data['tls_probability'][te];sh=np.zeros_like(pt);raw=np.zeros_like(pt)
        for j,x in enumerate((xs,xt)):
            active=app[te,j]
            if active.any():
                pt[active,j]=viewmodels[j].predict_proba(x[te][active])[:,1]
                gate,cal=gates[j];raw[active,j]=-gate.score_samples(x[te][active]);sh[active,j]=np.searchsorted(cal,raw[active,j],side='right')/len(cal)
        ev=evidence(pt,app[te],data['tls_mass'][te],rel,sh,raw);path=ff/f'test_{condition}.npz';freeze(path,ev,te);freezehash[condition]=sha(path)
        # All V0-V5 decisions below receive exactly this same frozen record.
        with np.load(path) as frozen:ev={k:frozen[k] for k in ('p','app','mass','reliability','ood','ood_raw')}
        ex=rawpipe.transform(raw_x(xs[te],xt[te],tf[te],app[te])).astype(np.float32)
        for method in METHODS:
            d=decision(method,ev,models,ex);prefix=f'{condition}__{method}__'
            for k,v in d.items():result[prefix+k]=v.astype(np.float32) if v.dtype.kind=='f' else v
            accept=(d['score']>=thresholds[method])&d['available'];result[prefix+'accepted_fixed']=accept
            if condition=='clean':clean[method]=d;clean[method]['accepted_fixed']=accept
            mf,ac=metric(y[te],d['pred']);native=d['verdict']<2;sf,sa=metric(y[te][native],d['pred'][native]);af,aa=metric(y[te][accept],d['pred'][accept])
            orig=clean[method];harm=(orig['accepted_fixed']&accept&(orig['pred']==y[te])&(d['pred']!=y[te])) if condition!='clean' else np.zeros(len(te),bool)
            result[prefix+'harmful']=harm
            row=dict(seed=seed,fold=fold,group=str(data['group_names'][fold]),condition=condition,method=method,macro_f1=mf,accuracy=ac,available_rate=float(d['available'].mean()),native_coverage=float(native.mean()),native_selective_macro_f1=sf,native_selective_error=1-sa,unknown_rate=float((d['verdict']==3).mean()),suspicious_rate=float((d['verdict']==2).mean()),fixed_coverage=float(accept.mean()),fixed_selective_error=1-aa,harmful_flips=int(harm.sum()),calls_per_sample=float(d['calls'].mean()),params=params[method],decision_latency_ms=decisionms[method],stats_view_latency_ms=viewms[0],temporal_view_latency_ms=viewms[1],stats_ood_latency_ms=oodms[0],temporal_ood_latency_ms=oodms[1])
            if method=='V0_TLS' and not d['available'].any():row['macro_f1']=row['accuracy']=float('nan')
            rows.append(row)
    result.update(index=te,y=y[te],group=data['group'][te])
    np.savez_compressed(dest/'predictions.npz',**result)
    csvwrite(dest/'metrics.csv',rows);dump(dest/'frozen_manifest.json',freezehash)
    done=dict(seed=seed,fold=fold,seconds=time.perf_counter()-started,source_features_sha256=sha(out/'features.npz'),predictions_sha256=sha(dest/'predictions.npz'))
    dump(dest/'DONE.json',done);return done

def run(out,workers,seeds,folds):
    out=Path(out);start=time.perf_counter();jobs=[(s,f) for s in seeds for f in folds]
    log=out/'execution.jsonl';fail=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(run_fold,str(out),s,f):(s,f) for s,f in jobs}
        for i,future in enumerate(as_completed(fs),1):
            try:r=future.result();r['status']='completed'
            except Exception as e:
                s,f=fs[future];r=dict(seed=s,fold=f,status='failed',error=repr(e),traceback=traceback.format_exc());fail.append(r)
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
            print(f'{i}/{len(jobs)} elapsed={time.perf_counter()-start:.1f}s '+json.dumps(r),flush=True)
    if fail:raise RuntimeError(f'{len(fail)} failed folds; see execution.jsonl')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','summarize']);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--folds',type=int,nargs='+',default=list(range(20)));a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='run':run(a.out,a.workers,a.seeds,a.folds)
    else:
        from authoritative_fusion_report import summarize
        summarize(a.out)
if __name__=='__main__':main()
