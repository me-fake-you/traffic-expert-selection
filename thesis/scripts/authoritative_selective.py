"""Post-hoc, pre-specified selective benchmark; never fit any detector or head."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import sys,json,time,hashlib,argparse,shutil,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(Path.home()/'AppData/Roaming/Python/Python312/site-packages'))
BASE=ROOT/'output/ustc_authoritative_fusion_v1'
FAST=BASE/'trained_fusion_fast_v1'
OUT=BASE/'selective_reliability_v1'
SEEDS=list(range(42,52))
CONDS=['clean','no_stats','no_temporal','no_tls','stats_volume_mask','temporal_size_mask','temporal_timing_direction_mask']
OLD=['V0_Stats','V0_Temporal','V1_Average','V2_Yager','V3_Full','V6_EDL']
NEW=['V4_StackLogistic_Frozen','V5_StackMLP32x16_Frozen','V6_Attention_Frozen']
METHODS=OLD+NEW
LABELS=['Stats','Temporal','Average','Yager (no OOD)','Full fusion','EDL (retrained)','Stack logistic','Stack MLP','Attention']
COV=np.unique(np.r_[1/40000,np.arange(1,101)/100,.2337])
K=np.rint(COV*40000).astype(np.int32)
FIXED_COV=[.2337,.5,.8,1.]
CIIDX=np.array([np.where(COV==q)[0][0] for q in FIXED_COV],dtype=np.int32)
REGIMES=['fixed_validation_80','fixed_validation_80_native']+[f'coverage_{q:.4f}' for q in FIXED_COV]
FIELDS=['corrected','introduced','c_minus_d','harmful_flips','native_harmful_flips','joint_accepted','clean_accepted','shift_accepted','operating_rejection_rate','native_rejection_rate','unknown_rate','suspicious_rate','clean_macro_f1','shift_macro_f1','macro_f1_delta','clean_selective_error','shift_selective_error']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def csv(path,rows):pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
def cfmetric(c):
    c=np.asarray(c,dtype=float);tn,fp,fn,tp=np.moveaxis(c,-1,0);n=c.sum(-1)
    a=2*tn+fp+fn;b=2*tp+fp+fn
    mf=.5*(np.divide(2*tn,a,out=np.zeros_like(a),where=a>0)+np.divide(2*tp,b,out=np.zeros_like(b),where=b>0))
    return np.where(n>0,mf,np.nan),np.divide(fp+fn,n,out=np.full_like(n,np.nan),where=n>0)
def ci(a):return np.quantile(a,[.025,.975],axis=0)
def oldpath(s,f):return BASE/'fold_runs'/f'seed_{s}'/f'fold_{f:02d}'
def fastpath(s,f):return FAST/'cells'/f'seed_{s}'/f'fold_{f:02d}'
def frozpath(s,f):return BASE/'frozen_predictions'/f'seed_{s}'/f'fold_{f:02d}'

def register():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():return
    for path in (BASE/'PROTOCOL.md',BASE/'features.npz',BASE/'bootstrap_draws.npz',FAST/'validation.json'):
        if not path.exists():raise FileNotFoundError(f'STOP: locked prerequisite missing: {path}')
    config=json.loads((BASE/'protocol.json').read_text());assert config['seeds']==SEEDS and config['per_group']==2000
    from sklearn.metrics import f1_score
    anchors=[];protected={}
    targets=dict(V0_Stats=.9152,V0_Temporal=.8684,V1_Average=.9099,V3_Full=.8821)
    vals={m:[] for m in targets}
    for s in SEEDS:
        ys=[];pred={m:[] for m in targets}
        for f in range(20):
            d=oldpath(s,f);done=json.loads((d/'DONE.json').read_text())
            assert sha(d/'predictions.npz')==done['predictions_sha256']
            with np.load(d/'predictions.npz') as z:
                ys.append(z['y'])
                for m in targets:pred[m].append(z[f'clean__{m}__pred'])
            manifest=json.loads((d/'frozen_manifest.json').read_text())
            for c in CONDS:
                p=frozpath(s,f)/f'test_{c}.npz';assert sha(p)==manifest[c]
        for m in targets:vals[m].append(f1_score(np.concatenate(ys),np.concatenate(pred[m]),labels=[0,1],average='macro',zero_division=0))
    for m,t in targets.items():
        val=float(np.mean(vals[m]));anchors.append(dict(method=m,target=t,reproduced=val,std=float(np.std(vals[m],ddof=1)),pass_tolerance=abs(val-t)<=.0001))
    csv(OUT/'anchors.csv',anchors)
    if not all(r['pass_tolerance'] for r in anchors):raise RuntimeError('STOP: anchor mismatch; see anchors.csv')
    # Hash only existing locked sources, never this new output subtree.
    paths=[BASE/n for n in ('PROTOCOL.md','protocol.json','features.npz','bootstrap_draws.npz','benchmark_results.csv','BENCHMARK.md')]
    paths+=list((BASE/'frozen_predictions').rglob('*.npz'))+list((BASE/'splits').rglob('*.npz'))
    for s in SEEDS:
        for f in range(20):
            paths += [oldpath(s,f)/n for n in ('predictions.npz','training.json','models.joblib','networks.pt')]
            paths += [fastpath(s,f)/n for n in ('predictions.npz','TRAINED.json','stack_logistic.joblib','fusion_heads.pt')]
    for p in paths:protected[str(p.relative_to(BASE))]=sha(p)
    with np.load(BASE/'features.npz') as z:
        dims={k:z[k].shape for k in z.files if z[k].ndim==2}
    dump(OUT/'PREREGISTRATION.json',dict(registered_unix=time.time(),analysis_type='post-hoc analysis of already observed locked results; analysis rules frozen before new curves',seeds=SEEDS,conditions=CONDS,methods=METHODS,coverages=COV.tolist(),fixed_coverages=FIXED_COV,representative_shift='no_temporal',bootstrap_draws=10000,bootstrap_reselect_each_draw=True,ranking='native score; verdict>=2 or unavailable becomes -1; tie=global sample index',dominance='paired percentile 95% upper bound of full error minus average error < 0; pointwise, not familywise',simultaneous_supplement='95th percentile of maximum absolute centered bootstrap difference over all 7 conditions and all coverages',aurc='exact mean of selective error at every accepted rank 1..40000; force binary above native coverage; undefined if no view available',cd_regimes=REGIMES,cd_fields=FIELDS,dimension_shapes=dims,protected_sha256=protected,new_model_training=False,edl='reuse previously retrained feature-based EDL predictions, no new EDL inference or fitting'))
    print('Anchors passed; preregistration sealed',flush=True)

def head_outputs(models,p,app):
    import torch
    from frozen_fusion_fast import probability_features,gating_features,gate_probability
    x=probability_features(p,app);gx=gating_features(p,app)
    with torch.no_grad():
        a=models[0].predict_proba(x)[:,1];b=torch.softmax(models[1](torch.tensor(x)),1)[:,1].numpy()
        c,w=gate_probability(models[2](torch.tensor(gx)),torch.tensor(gx))
    assert np.all(w.numpy()[~app]==0)
    return np.stack((a,b,c.numpy()))

def collect_seed(seed):
    """Copy original decisions; infer only new heads on not-yet-scored occlusions."""
    dest=OUT/'cache'/f'seed_{seed}.npz'
    if dest.exists():return dict(seed=seed,status='cached')
    import joblib,torch
    from frozen_fusion_fast import make_net,CONFIG
    torch.set_num_threads(1)
    arrays={};labels=[];groups=[];ids=[]
    def add(key,value):arrays.setdefault(key,[]).append(value)
    for fold in range(20):
        dp=oldpath(seed,fold);fp=fastpath(seed,fold)
        thresholds=json.loads((dp/'training.json').read_text())['thresholds']
        with np.load(dp/'predictions.npz') as z:
            labels.append(z['y']);groups.append(z['group']);ids.append(z['index'])
            for cond in CONDS:
                for m in OLD:
                    for field in ('pred','score','verdict','available','accepted_fixed'):
                        add(f'{cond}__{m}__{field}',z[f'{cond}__{m}__{field}'])
                    add(f'{cond}__{m}__threshold',np.full(2000,thresholds[m]))
        trained=json.loads((fp/'TRAINED.json').read_text())
        for name,h in trained['model_sha256'].items():assert sha(fp/name)==h
        mlp=make_net(CONFIG['mlp_dims']);gate=make_net(CONFIG['gate_dims'])
        state=torch.load(fp/'fusion_heads.pt',map_location='cpu',weights_only=True)
        mlp.load_state_dict(state['mlp']);gate.load_state_dict(state['gate']);mlp.eval();gate.eval()
        models=[joblib.load(fp/'stack_logistic.joblib'),mlp,gate]
        # Thresholds are validation-only and fixed before any new occlusion evaluation.
        with np.load(frozpath(seed,fold)/'validation.npz') as z:vp=head_outputs(models,z['p'],z['app'])
        th=np.quantile(2*np.abs(vp-.5),.2,axis=1)
        for cond in CONDS:
            if cond=='clean':
                with np.load(fp/'predictions.npz') as z:p=z['probabilities'];app=z['applicability'];assert np.array_equal(z['index'],ids[-1])
            else:
                with np.load(frozpath(seed,fold)/f'test_{cond}.npz') as z:
                    assert np.array_equal(z['index'],ids[-1]);app=z['app'];p=head_outputs(models,z['p'],app)
            for j,m in enumerate(NEW):
                score=2*np.abs(p[j]-.5);available=app.any(1);pred=(p[j]>=.5).astype(np.int8)
                fields=dict(pred=pred,score=score,verdict=np.where(available,pred,3),available=available,accepted_fixed=available&(score>=th[j]),threshold=np.full(2000,th[j]))
                for field,v in fields.items():add(f'{cond}__{m}__{field}',v)
    arrays={k:np.concatenate(v) for k,v in arrays.items()};arrays.update(y=np.concatenate(labels),group=np.concatenate(groups),index=np.concatenate(ids))
    assert len(set(arrays['index']))==40000
    dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**arrays)
    dump(dest.with_suffix('.json'),dict(seed=seed,sha256=sha(dest),new_occlusion_head_predictions=20*6*3,clean_head_predictions_reused=True,view_calls=0,fit_calls=0))
    return dict(seed=seed,status='collected')

def ordered(d,cond,m):
    key=f'{cond}__{m}__';available=d[key+'available'];native=d[key+'verdict']<2
    score=np.where(available&native,d[key+'score'],-1.)
    # Entirely unavailable single views have no class evidence and no feasible curve.
    ix=np.flatnonzero(available);order=ix[np.lexsort((d['index'][ix],-score[ix]))]
    return order

@njit(cache=True)
def weighted_curves(group,code,weights,ks,harmonic):
    """Exact duplicated-group bootstrap ranks, including partial boundary copies."""
    B=len(weights);L=len(ks);N=len(group)
    f1=np.full((B,L),np.nan);error=np.full((B,L),np.nan);aurc=np.full(B,np.nan)
    cut=np.full((B,L),-1,np.int32);take=np.zeros((B,L),np.int16)
    for b in range(B):
        cf=np.zeros(4,np.int64);n=0;err=0;area=0.;j=0
        for r in range(N):
            w=int(weights[b,group[r]])
            if w==0:continue
            c=code[r];wrong=(c==1 or c==2);dh=harmonic[n+w]-harmonic[n]
            area += w+(err-n)*dh if wrong else err*dh
            cf[c]+=w;n+=w
            if wrong:err+=w
            while j<L and n>=ks[j]:
                excess=n-ks[j];cf[c]-=excess
                a=2*cf[0]+cf[1]+cf[2];bb=2*cf[3]+cf[1]+cf[2]
                f1[b,j]=.5*((2*cf[0]/a if a else 0.)+(2*cf[3]/bb if bb else 0.))
                error[b,j]=(cf[1]+cf[2])/ks[j]
                cut[b,j]=r;take[b,j]=w-excess
                cf[c]+=excess;j+=1
        if n==ks[-1]:aurc[b]=area/n
    return f1,error,aurc,cut,take

def run_curve_job(method,condition):
    dest=OUT/'curves'/f'{condition}__{method}.npz'
    if dest.exists():return dict(method=method,condition=condition,status='cached')
    with np.load(BASE/'bootstrap_draws.npz') as z:w=z['group_weights'].astype(np.int16)
    weights=np.vstack((np.ones((1,20),np.int16),w));h=np.r_[0,np.cumsum(1/np.arange(1,40001))]
    fs=[];es=[];au=[];cuts=[];takes=[];point_f=[];point_e=[];point_a=[];cache={};nc=[]
    for seed in SEEDS:
        with np.load(OUT/'cache'/f'seed_{seed}.npz') as z:
            d={k:z[k] for k in ('y','group','index',*(f'{condition}__{method}__{v}' for v in ('pred','score','verdict','available')))}
        order=ordered(d,condition,method);code=(2*d['y']+d[f'{condition}__{method}__pred'])[order].astype(np.int8);group=d['group'][order].astype(np.int16)
        key=hashlib.sha256(group.tobytes()+code.tobytes()).hexdigest()
        if key not in cache:cache[key]=weighted_curves(group,code,weights,K,h)
        f,e,a,c,t=cache[key]
        point_f.append(f[0]);point_e.append(e[0]);point_a.append(a[0]);fs.append(f[1:]);es.append(e[1:]);au.append(a[1:]);cuts.append(c[:,CIIDX]);takes.append(t[:,CIIDX]);nc.append(np.mean(d[f'{condition}__{method}__verdict']<2))
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,point_f1=point_f,point_error=point_e,point_aurc=point_a,boot_f1=np.mean(fs,axis=0),boot_error=np.mean(es,axis=0),boot_aurc=np.mean(au,axis=0),cuts=cuts,takes=takes,native_coverage=nc)
    return dict(method=method,condition=condition,status='curves',unique_orderings=len(cache))

@njit(cache=True)
def paired_coverage_counts(order,shift_rank,groups,flags,weights,cut0,take0,cut1,take1):
    # Duplicate copy IDs are ordered identically in both conditions; joint multiplicity=min.
    B=len(weights);Q=cut0.shape[1];out=np.zeros((B,Q,5))
    for b in range(B):
        for q in range(Q):
            if cut0[b,q]<0 or cut1[b,q]<0:
                out[b,q,:]=np.nan;continue
            for r in range(cut0[b,q]+1):
                i=order[r];w=int(weights[b,groups[i]])
                if w==0:continue
                sr=shift_rank[i]
                if sr>cut1[b,q]:continue
                a=take0[b,q] if r==cut0[b,q] else w
                bb=take1[b,q] if sr==cut1[b,q] else w
                n=min(a,bb)
                out[b,q,4]+=n
                for j in range(4):out[b,q,j]+=n*flags[i,j]
    return out

def group_sum(group,values):
    values=np.asarray(values)
    if values.ndim==1:values=values[:,None]
    return np.stack([np.bincount(group,weights=values[:,j],minlength=20) for j in range(values.shape[1])],1)

def run_cd_job(method,condition):
    dest=OUT/'cd'/f'{condition}__{method}.npz'
    if dest.exists():return dict(method=method,condition=condition,status='cached')
    with np.load(BASE/'bootstrap_draws.npz') as z:w=z['group_weights'].astype(np.int16)
    weights=np.vstack((np.ones((1,20),np.int16),w));B=len(weights)
    with np.load(OUT/'curves'/f'clean__{method}.npz') as z:cut0=z['cuts'];take0=z['takes'];bf0=z['boot_f1'];be0=z['boot_error'];pf0=z['point_f1'];pe0=z['point_error']
    with np.load(OUT/'curves'/f'{condition}__{method}.npz') as z:cut1=z['cuts'];take1=z['takes'];bf1=z['boot_f1'];be1=z['boot_error'];pf1=z['point_f1'];pe1=z['point_error']
    points=[];boot=np.zeros((B-1,len(REGIMES),len(FIELDS)));cache={}
    for si,seed in enumerate(SEEDS):
        with np.load(OUT/'cache'/f'seed_{seed}.npz') as z:
            names=['y','group','index']+[f'{c}__{method}__{v}' for c in ('clean',condition) for v in ('pred','score','verdict','available','accepted_fixed')]
            d={k:z[k] for k in names}
        key0=f'clean__{method}__';key1=f'{condition}__{method}__';group=d['group'];y=d['y'];p0=d[key0+'pred'];p1=d[key1+'pred']
        n0=d[key0+'verdict']<2;n1=d[key1+'verdict']<2
        conf0=d[key0+'accepted_fixed'];conf1=d[key1+'accepted_fixed']
        C=(p0!=y)&(p1==y);D=(p0==y)&(p1!=y)
        flags=np.stack((C,D,D&conf0&conf1,D&conf0&conf1&n0&n1),1).astype(np.int8)
        result=np.full((B,len(REGIMES),len(FIELDS)),np.nan)
        native=weights@group_sum(group,np.stack((~n1,d[key1+'verdict']==3,d[key1+'verdict']==2),1))/40000
        result[:,:,9:12]=native[:,None,:]
        for ri in range(2):
            a0=conf0 if ri==0 else conf0&n0;a1=conf1 if ri==0 else conf1&n1;joint=a0&a1
            vals=np.c_[flags*joint[:,None],joint,a0,a1]
            v=weights@group_sum(group,vals)
            result[:,ri,0:2]=v[:,:2];result[:,ri,2]=v[:,0]-v[:,1];result[:,ri,3:8]=v[:,2:7]
            result[:,ri,8]=1-v[:,6]/40000
            cf0=weights@group_sum(group,np.eye(4)[2*y+p0]*a0[:,None]);cf1=weights@group_sum(group,np.eye(4)[2*y+p1]*a1[:,None])
            f0,e0=cfmetric(cf0);f1,e1=cfmetric(cf1)
            result[:,ri,12:]=np.stack((f0,f1,f1-f0,e0,e1),1)
        order0=ordered(d,'clean',method);order1=ordered(d,condition,method);r1=np.full(40000,40000,np.int32);r1[order1]=np.arange(len(order1))
        signature=hashlib.sha256(order0.tobytes()+r1.tobytes()+flags.tobytes()+cut0[si].tobytes()+take0[si].tobytes()+cut1[si].tobytes()+take1[si].tobytes()).hexdigest()
        if signature not in cache:
            cache[signature]=paired_coverage_counts(order0,r1,group,flags,weights,cut0[si],take0[si],cut1[si],take1[si])
        v=cache[signature]
        result[:,2:,0:2]=v[:,:,:2];result[:,2:,2]=v[:,:,0]-v[:,:,1];result[:,2:,3:6]=v[:,:,2:5]
        for qi,coverage in enumerate(FIXED_COV):
            ri=qi+2;feasible=(cut0[si,:,qi]>=0)&(cut1[si,:,qi]>=0)
            result[:,ri,6]=np.where(cut0[si,:,qi]>=0,round(coverage*40000),np.nan)
            result[:,ri,7]=np.where(cut1[si,:,qi]>=0,round(coverage*40000),np.nan)
            result[:,ri,8]=1-result[:,ri,7]/40000
            result[0,ri,12:]=[pf0[si,CIIDX[qi]],pf1[si,CIIDX[qi]],pf1[si,CIIDX[qi]]-pf0[si,CIIDX[qi]],pe0[si,CIIDX[qi]],pe1[si,CIIDX[qi]]]
            # Mean-of-seed bootstrap curve metrics are assigned once below.
        points.append(result[0]);boot+=result[1:]/10
    for qi,kk in enumerate(CIIDX):
        boot[:,qi+2,12:]=np.stack((bf0[:,kk],bf1[:,kk],bf1[:,kk]-bf0[:,kk],be0[:,kk],be1[:,kk]),1)
    dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,point=points,bootstrap=boot)
    return dict(method=method,condition=condition,status='cd',unique_pairs=len(cache))

def parallel(function,args,workers):
    failures=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(function,*a):a for a in args}
        for i,f in enumerate(as_completed(pending),1):
            try:r=f.result()
            except Exception as exc:r=dict(status='failed',arguments=pending[f],error=repr(exc),traceback=traceback.format_exc());failures.append(r)
            r['elapsed_seconds']=time.perf_counter()-start
            with (OUT/'execution.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(r)+'\n')
            print(i,'/',len(args),r,flush=True)
    if failures:raise RuntimeError('Failed jobs retained in execution.jsonl')

def finite_ci(x):
    x=np.asarray(x);flat=x.reshape(len(x),-1);lo=np.full(flat.shape[1],np.nan);hi=lo.copy();valid=np.isfinite(flat).sum(0)
    for j in range(flat.shape[1]):
        if valid[j]:lo[j],hi[j]=np.quantile(flat[np.isfinite(flat[:,j]),j],[.025,.975])
    return lo.reshape(x.shape[1:]),hi.reshape(x.shape[1:]),valid.reshape(x.shape[1:])

def regions(condition,kind,lower,upper):
    status=np.where(upper<0,'full_lower_error',np.where(lower>0,'average_lower_error','no_direction_established'))
    rows=[];start=0
    for end in range(1,len(COV)+1):
        if end==len(COV) or status[end]!=status[start]:
            rows.append(dict(condition=condition,interval_type=kind,coverage_start=float(COV[start]),coverage_end=float(COV[end-1]),grid_points=end-start,status=status[start],full_strictly_lower=status[start]=='full_lower_error',interpretation='tested grid points only; no interpolation claim'))
            start=end
    return rows

def report():
    reg=json.loads((OUT/'PREREGISTRATION.json').read_text())
    assert reg['dimension_shapes']['stats']==[40000,11] and reg['dimension_shapes']['temporal']==[40000,30]
    changed=[p for p,h in reg['protected_sha256'].items() if sha(BASE/p)!=h]
    assert not changed,changed
    curve_rows=[];aurc_rows=[];cd_rows=[];dominance=[];curves={};cds={};delta_boot={};delta_point={};pair_rows=[]
    for c in CONDS:
        for m in METHODS:
            with np.load(OUT/'curves'/f'{c}__{m}.npz') as z:
                curves[c,m]={k:z[k] for k in ('point_f1','point_error','point_aurc','boot_f1','boot_error','boot_aurc','native_coverage')}
        full=curves[c,'V3_Full'];avg=curves[c,'V1_Average']
        delta_boot[c]=full['boot_error']-avg['boot_error'];delta_point[c]=(full['point_error']-avg['point_error']).mean(0)
    maxdev=np.zeros(10000)
    for c in CONDS:maxdev=np.maximum(maxdev,np.max(np.abs(delta_boot[c]-delta_point[c]),axis=1))
    critical=float(np.quantile(maxdev,.95))
    for c in CONDS:
        dl,dh=ci(delta_boot[c]);sl=delta_point[c]-critical;sh=delta_point[c]+critical
        dominance+=regions(c,'pointwise_95',dl,dh)+regions(c,'simultaneous_all_conditions_95',sl,sh)
        for j,q in enumerate(COV):pair_rows.append(dict(condition=c,coverage=q,error_difference_full_minus_average=delta_point[c][j],paired_ci95_low=dl[j],paired_ci95_high=dh[j],full_lower_error=dh[j]<0,simultaneous_ci95_low=sl[j],simultaneous_ci95_high=sh[j],simultaneous_full_lower_error=sh[j]<0))
        for m in METHODS:
            d=curves[c,m];fl,fh,fv=finite_ci(d['boot_f1']);el,eh,ev=finite_ci(d['boot_error']);al,ah,av=finite_ci(d['boot_aurc'])
            # Class composition and forced-rejection fraction support interpretation of selective F1.
            composition=[];forced=[]
            for s in SEEDS:
                with np.load(OUT/'cache'/f'seed_{s}.npz') as z:
                    names=['index','y']+[f'{c}__{m}__{v}' for v in ('score','verdict','available')];dd={k:z[k] for k in names}
                ix=ordered(dd,c,m)
                if len(ix)==40000:
                    composition.append(np.cumsum(dd['y'][ix])[K-1]);forced.append(np.cumsum(dd[f'{c}__{m}__verdict'][ix]>=2)[K-1]/K)
                else:composition.append(np.full(len(K),np.nan));forced.append(np.full(len(K),np.nan))
            malicious=np.mean(composition,axis=0);forced=np.mean(forced,axis=0)
            for j,q in enumerate(COV):
                curve_rows.append(dict(method=m,condition=c,coverage=float(q),accepted_count=int(K[j]) if np.isfinite(d['point_error'][:,j]).all() else 0,feasible=bool(np.isfinite(d['point_error'][:,j]).all()),selective_macro_f1_mean=float(d['point_f1'][:,j].mean()),selective_macro_f1_std=float(d['point_f1'][:,j].std(ddof=1)),macro_f1_ci95_low=fl[j],macro_f1_ci95_high=fh[j],macro_f1_valid_bootstrap_draws=fv[j],selective_error_mean=float(d['point_error'][:,j].mean()),selective_error_std=float(d['point_error'][:,j].std(ddof=1)),selective_error_ci95_low=el[j],selective_error_ci95_high=eh[j],error_valid_bootstrap_draws=ev[j],accepted_benign_mean=K[j]-malicious[j],accepted_malicious_mean=malicious[j],native_coverage_mean=float(d['native_coverage'].mean()),forced_native_rejected_fraction=forced[j],edl_retrained_feature_model=m=='V6_EDL'))
            row=dict(method=m,condition=c,aurc_mean=float(d['point_aurc'].mean()),aurc_std=float(d['point_aurc'].std(ddof=1)),aurc_ci95_low=float(al),aurc_ci95_high=float(ah),valid_bootstrap_draws=int(av),feasible=bool(np.isfinite(d['point_aurc']).all()),integration='exact mean risk over all 40000 ranks')
            if m=='V3_Full':
                dd=d['boot_aurc']-curves[c,'V1_Average']['boot_aurc'];a,b=ci(dd)
                row.update(aurc_delta_vs_average=float((d['point_aurc']-curves[c,'V1_Average']['point_aurc']).mean()),delta_ci95_low=float(a),delta_ci95_high=float(b))
            aurc_rows.append(row)
    for m in METHODS:
        for c in CONDS[1:]:
            with np.load(OUT/'cd'/f'{c}__{m}.npz') as z:cds[c,m]=dict(point=z['point'],bootstrap=z['bootstrap'])
        point=np.sum([cds[c,m]['point'] for c in CONDS[1:]],axis=0);boot=np.sum([cds[c,m]['bootstrap'] for c in CONDS[1:]],axis=0)
        point[:,:,8:]/=6;boot[:,:,8:]/=6
        cds['all_six_occlusions',m]=dict(point=point,bootstrap=boot)
        for c in [*CONDS[1:],'all_six_occlusions']:
            d=cds[c,m];lo,hi,n=finite_ci(d['bootstrap'])
            for j,op in enumerate(REGIMES):
                row=dict(method=m,condition=c,operating_point=op,coverage_target=FIXED_COV[j-2] if j>=2 else np.nan,unit='sum of six paired perturbation counts per seed; condition-mean rates/metrics' if c=='all_six_occlusions' else 'paired sample counts per seed',exposures_per_seed=240000 if c=='all_six_occlusions' else 40000)
                for fi,field in enumerate(FIELDS):
                    values=d['point'][:,j,fi];row[field+'_mean']=float(values.mean());row[field+'_std']=float(values.std(ddof=1));row[field+'_ci95_low']=lo[j,fi];row[field+'_ci95_high']=hi[j,fi];row[field+'_valid_bootstrap_draws']=n[j,fi]
                    if fi<8:row[field+'_total_10_seeds']=int(values.sum()) if np.isfinite(values).all() else np.nan
                cd_rows.append(row)
    csv(OUT/'risk_coverage_curves.csv',curve_rows);csv(OUT/'aurc.csv',aurc_rows);csv(OUT/'cd_decomposition.csv',cd_rows);csv(OUT/'paired_error_differences.csv',pair_rows);csv(OUT/'dominance_regions.csv',dominance)
    dump(OUT/'simultaneous_band.json',dict(family='7 conditions x 102 coverages: full vs average selective error',critical_absolute_deviation_95=critical,bootstrap_draws=10000))
    draw_figures(pd.DataFrame(curve_rows),pd.DataFrame(pair_rows),pd.DataFrame(cd_rows))
    write_report(pd.DataFrame(aurc_rows),pd.DataFrame(dominance),pd.DataFrame(cd_rows),pd.DataFrame(curve_rows))
    final_audit(reg,curves,cds,len(curve_rows),len(cd_rows))

def draw_figures(curve,pairs,cd):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    visible=curve[curve.condition.isin(['clean','no_temporal'])&curve.method.isin(['V3_Full','V1_Average'])]
    ymax=min(1.,np.ceil(float(visible.selective_error_ci95_high.max())*20)/20)
    for ax,c,title in zip(axes,['clean','no_temporal'],['Clean','Temporal view occluded (pre-specified)']):
        pp=pairs[pairs.condition==c]
        # Green strips encode pointwise evidence only, not simultaneous significance.
        for j in np.flatnonzero(pp.full_lower_error.to_numpy()):
            left=COV[j] if j==0 else (COV[j-1]+COV[j])/2;right=COV[j] if j==len(COV)-1 else (COV[j]+COV[j+1])/2
            ax.axvspan(left,right,color='#AADCA9',alpha=.45,lw=0,zorder=0)
        for m,color,label in [('V3_Full','#B64342','Full fusion'),('V1_Average','#3775BA','Probability average')]:
            d=curve[(curve.condition==c)&(curve.method==m)]
            ax.plot(d.coverage,d.selective_error_mean,color=color,lw=1.8,ls='--' if m=='V1_Average' else '-',label=label)
            ax.fill_between(d.coverage,d.selective_error_ci95_low,d.selective_error_ci95_high,color=color,alpha=.14,lw=0)
        ax.set(title=title,xlabel='Coverage',xlim=(0,1),ylim=(0,ymax));ax.grid(axis='y',alpha=.2)
        if not pp.full_lower_error.any():ax.text(.03,.95,'No pointwise full-fusion dominance',transform=ax.transAxes,va='top',fontsize=8)
    axes[0].set_ylabel('Selective error (lower is better)');axes[1].legend(loc='upper right',frameon=False)
    fig.text(.5,.01,'Bands: paired group-bootstrap 95% CI. Green: pointwise full-fusion advantage; not multiplicity corrected.\nAbove native coverage, curves force binary projections of rejected decisions.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1))
    for ext in ('pdf','png'):fig.savefig(OUT/f'risk_coverage.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    x=np.arange(len(METHODS));width=.36
    for ax,op,title in zip(axes,REGIMES[:2],['Fixed validation threshold: forced binary','Fixed threshold + native acceptance']):
        data=cd[(cd.condition=='all_six_occlusions')&(cd.operating_point==op)].set_index('method').loc[METHODS]
        for offset,field,color,label in [(-width/2,'corrected','#42949E','C: wrong → right'),(width/2,'introduced','#B64342','D: right → wrong')]:
            xx=x+offset;means=data[field+'_mean'].to_numpy();lo=data[field+'_ci95_low'].to_numpy();hi=data[field+'_ci95_high'].to_numpy()
            ax.bar(xx,means,width,color=color,label=label,hatch='//' if field=='introduced' else None,edgecolor='#333333',linewidth=.4);ax.vlines(xx,lo,hi,color='#222222',lw=.7);ax.hlines(lo,xx-.05,xx+.05,color='#222222',lw=.7);ax.hlines(hi,xx-.05,xx+.05,color='#222222',lw=.7)
        ax.set_xticks(x,LABELS,rotation=48,ha='right',fontsize=8);ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Paired changes per seed (sum over six occlusions)');axes[1].legend(loc='upper left',frameon=False,fontsize=8)
    fig.text(.5,.01,'Bars: ten-seed means; whiskers: paired group-bootstrap 95% CI. C/D use the clean–shift acceptance intersection.\nC−D is a count, not a Macro-F1 difference. EDL was previously trained on features; no models were refitted here.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,1))
    for ext in ('pdf','png'):fig.savefig(OUT/f'corrected_introduced.{ext}',dpi=300)
    plt.close(fig)

def write_report(aurc,dom,cd,curve):
    anchors=pd.read_csv(OUT/'anchors.csv');lines=['# Selective reliability on locked USTC predictions','',
        'Completed analysis of the locked 11-D Stats / 30-D Temporal protocol, 20 whole-group folds × seeds 42–51. No model was refitted. Three newly added heads were evaluated once on each previously frozen occlusion; their clean predictions and all original method/EDL decisions were reused. EDL is a previously retrained feature-based model, not a frozen-output fuser.','',
        'This is post-hoc analysis of an existing test set. Analysis rules were sealed before generating the new curves, not before the original results were known. The requested expectation that clean accuracy matches averaging is not assumed.','',
        '## Anchor reproduction','', '| Method | Exact pooled Macro-F1 |','|---|---:|']
    lines += [f"| {r.method} | {r.reproduced:.12f} |" for r in anchors.itertuples()]
    lines += ['','All four anchors passed tolerance 0.0001. No historical score or alternate cohort was substituted.','',
        '## Exact AURC (lower is better)','', '| Method | Clean mean±SD | Clean 95% group CI | Temporal occluded mean±SD |','|---|---:|---|---:|']
    for m,label in zip(METHODS,LABELS):
        a=aurc[(aurc.method==m)&(aurc.condition=='clean')].iloc[0];b=aurc[(aurc.method==m)&(aurc.condition=='no_temporal')].iloc[0]
        shifted=f'{b.aurc_mean:.5f}±{b.aurc_std:.5f}' if np.isfinite(b.aurc_mean) else 'N/A: entire view absent'
        lines.append(f'| {label} | {a.aurc_mean:.5f}±{a.aurc_std:.5f} | [{a.aurc_ci95_low:.5f}, {a.aurc_ci95_high:.5f}] | {shifted} |')
    lines += ['','AURC averages error at every prefix rank 1…40,000. Selective Macro-F1, error, standard deviations and paired 95% group CIs are saved for every method × seven conditions × 102 coverages. Each bootstrap draw resamples whole groups and reselects at the target coverage; all seeds/methods/conditions share the same 10,000 draws.','',
        '## Full fusion versus probability averaging','', '| Condition | AURC difference (full − average) | Paired 95% CI |','|---|---:|---|']
    for r in aurc[aurc.method=='V3_Full'].itertuples():lines.append(f'| {r.condition} | {r.aurc_delta_vs_average:.5f} | [{r.delta_ci95_low:.5f}, {r.delta_ci95_high:.5f}] |')
    lines += ['','Pointwise dominance requires the entire paired CI for full-minus-average selective error to be below zero. The following are ranges of **tested grid points**, not guarantees between points. They are not corrected for multiple coverage comparisons. The simultaneous supplement controls a common bootstrap deviation band across all seven conditions and 102 points.','', '| Condition | Pointwise full-lower-error regions | Simultaneous supplement |','|---|---|---|']
    def describe(dd):
        dd=dd[dd.status=='full_lower_error']
        if dd.empty:return 'none exists'
        return '; '.join(f'{r.coverage_start:.4f}–{r.coverage_end:.4f}' if r.grid_points>1 else f'{r.coverage_start:.4f} (one grid point)' for r in dd.itertuples())
    for c in CONDS:lines.append(f"| {c} | {describe(dom[(dom.condition==c)&(dom.interval_type=='pointwise_95')])} | {describe(dom[(dom.condition==c)&(dom.interval_type=='simultaneous_all_conditions_95')])} |")
    dominance_count=int(((dom.interval_type=='pointwise_95')&(dom.status=='full_lower_error')).sum())
    worse=aurc[(aurc.method=='V3_Full')&(aurc.delta_ci95_low>0)]
    if dominance_count==0:
        lines += ['', '**No full-fusion dominance region exists on the tested grid in any of the seven conditions.** This is an observed negative result, not an unreported or omitted analysis.']
    if len(worse)==7:
        lines += ['', '**Probability averaging has lower AURC in all seven conditions, with each paired 95% difference interval excluding zero.** These measurements do not support a claim that the full fusion has superior selective reliability under this ranking rule. Lower native flip counts, if present, must be assessed together with rejection and coverage.']
    lines += ['','Every non-dominance interval, including intervals favoring averaging and unresolved intervals, is retained in `dominance_regions.csv`; every paired contrast and both bands are in `paired_error_differences.csv`. No failed claim or baseline-favoring interval is omitted.','',
        '## Corrected / introduced','', 'Below, counts sum the six occlusions per seed (240,000 paired exposures). Selection is the unchanged validation threshold intersected with native acceptance in both conditions. Full per-condition and coverage-specific CIs are in `cd_decomposition.csv`.','',
        '| Method | C mean [95% CI] | D mean [95% CI] | C−D count | Native harmful flips | Shift operating coverage | Shift native rejection rate |','|---|---|---|---:|---:|---:|---:|']
    rows=cd[(cd.condition=='all_six_occlusions')&(cd.operating_point=='fixed_validation_80_native')].set_index('method')
    for m,label in zip(METHODS,LABELS):
        r=rows.loc[m];lines.append(f'| {label} | {r.corrected_mean:.1f} [{r.corrected_ci95_low:.1f}, {r.corrected_ci95_high:.1f}] | {r.introduced_mean:.1f} [{r.introduced_ci95_low:.1f}, {r.introduced_ci95_high:.1f}] | {r.c_minus_d_mean:.1f} | {r.native_harmful_flips_mean:.1f} | {1-r.operating_rejection_rate_mean:.4f} | {r.native_rejection_rate_mean:.4f} |')
    full=rows.loc['V3_Full'];avg=rows.loc['V1_Average']
    if full.native_harmful_flips_mean==0 and avg.native_harmful_flips_mean==0:
        lines += ['',f'At this fixed native operating point, both full fusion and averaging have zero harmful flips. Full fusion accepts {1-full.operating_rejection_rate_mean:.2%} of perturbed exposures versus {1-avg.operating_rejection_rate_mean:.2%} for averaging. Thus zero native flips does not establish an advantage over averaging. Native rejection alone is {full.native_rejection_rate_mean:.2%} versus {avg.native_rejection_rate_mean:.2%}; threshold-based rejection is accounted for separately.']
    lines += ['','C/D use only paired observations selected before and after perturbation. Leaving the accepted set is reported as rejection, not a corrected label. Harmful flips additionally require both unchanged clean-validation confidence thresholds; native harmful flips also require binary native verdicts. C−D is a count and is never substituted for Macro-F1 change. Coverage regimes are 0.2337, 0.5, 0.8, 1.0; unavailable single-view cases remain N/A.','',
        '## Interpretation and limits','',
        '- Curves rank native suspicious/unknown last. Above native coverage, forced binary projections are used; these are not native accepted decisions. Their fraction is recorded for every point.',
        '- Ties use the locked global row index. Large rejected-score ties inherit the manifest group order and can affect the shape of high-coverage forced-projection curves.',
        '- Fixed-label Macro-F1 uses both labels even if the selected subset contains one class; selected benign/malicious counts are retained. This analysis pools all twenty held-out groups within each seed before ranking.',
        '- Group CIs condition on these groups and saved models. They do not include refitting uncertainty or establish clean-accuracy equivalence. A low flip count accompanied by rejection is not a claim of better full-coverage accuracy.',
        '- The representative shift was fixed as `no_temporal`; every other condition remains in the CSVs. Existing benchmark tables and models were preserved.',
        '', '## Files and reproduction','',
        '[Analysis protocol](ANALYSIS_PROTOCOL.md), [pre-registration](PREREGISTRATION.json), [run log](RUNS.md), [validation](validation.json). Commands and failed-run records are in RUNS.md; exact source and environment snapshots accompany the results.','',
        '![Risk–coverage](risk_coverage.png)','', '![Corrected and introduced](corrected_introduced.png)','']
    (OUT/'SELECTIVE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

def final_audit(reg,curves,cds,curve_count,cd_count):
    from sklearn.metrics import f1_score
    from PIL import Image
    import numba,scipy,sklearn
    with np.load(BASE/'bootstrap_draws.npz') as z:
        weights=z['group_weights'];assert weights.shape==(10000,20)
        assert np.all(weights[:,:10].sum(1)==10) and np.all(weights[:,10:].sum(1)==10)
    cases=0;feature_hash=sha(BASE/'features.npz')
    for s in SEEDS:
        for f in range(20):assert json.loads((oldpath(s,f)/'DONE.json').read_text())['source_features_sha256']==feature_hash
    for si,s in enumerate(SEEDS):
        path=OUT/'cache'/f'seed_{s}.npz';assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
        with np.load(path) as z:
            for c in CONDS:
                for m in METHODS:
                    d=curves[c,m]
                    if z[f'{c}__{m}__available'].all():
                        y=z['y'];pred=z[f'{c}__{m}__pred']
                        np.testing.assert_allclose(d['point_f1'][si,-1],f1_score(y,pred,labels=[0,1],average='macro'),atol=1e-12)
                        np.testing.assert_allclose(d['point_error'][si,-1],np.mean(y!=pred),atol=1e-12)
                    else:assert np.isnan(d['point_f1'][si]).all()
                    cases+=1
    for d in cds.values():
        np.testing.assert_allclose(d['point'][:,:,0]-d['point'][:,:,1],d['point'][:,:,2],equal_nan=True)
        assert np.all((d['point'][:,:,4]<=d['point'][:,:,3])|np.isnan(d['point'][:,:,3]))
    reference_audit()
    image_info={}
    for name in ('risk_coverage','corrected_introduced'):
        im=Image.open(OUT/f'{name}.png');assert min(im.info.get('dpi',(0,0)))>=299
        image_info[name]=dict(pixels=im.size,dpi=im.info['dpi'],png_sha256=sha(OUT/f'{name}.png'),pdf_sha256=sha(OUT/f'{name}.pdf'))
    # Read-only PDF inspection; images remain vector charts and fonts must be embedded.
    try:
        import fitz
        for name in image_info:
            doc=fitz.open(OUT/f'{name}.pdf');fonts=doc[0].get_fonts();assert fonts and all(doc.extract_font(font[0])[3] for font in fonts)
            assert not doc[0].get_images();image_info[name]['embedded_fonts']=len(fonts);image_info[name]['raster_images']=0;doc.close()
    except ImportError:raise RuntimeError('PDF verification needs the already available PyMuPDF installation')
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'scripts/authoritative_selective.py',ROOT/'tests/test_authoritative_selective.py',ROOT/'scripts/frozen_fusion_fast.py'):shutil.copy2(p,snapshot/p.name)
    dump(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,numba=numba.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,source_sha256={p.name:sha(p) for p in snapshot.iterdir()}))
    events=[json.loads(line) for line in (OUT/'execution.jsonl').read_text().splitlines()]
    timings={stage:max(r['elapsed_seconds'] for r in events if r['status']==stage) for stage in ('collected','curves','cd')}
    dump(OUT/'run_summary.json',dict(stage_seconds=timings,parallel_jobs_completed=len(events),parallel_failures=sum(r['status']=='failed' for r in events)))
    dump(OUT/'validation.json',dict(status='passed',anchors=4,seed_method_condition_checks=cases,curve_rows=curve_count,coverage_grid_points=len(COV),cd_rows=cd_count,bootstrap_draws=10000,group_bootstrap_reranks=True,source_files_unchanged=len(reg['protected_sha256']),new_detector_fits=0,new_head_fits=0,new_edl_fits=0,new_view_inferences=0,new_occlusion_head_inferences=200*6*3,old_clean_predictions_reused=True,figures=image_info))
    print('All selective deliverable checks passed',flush=True)

def reference_audit():
    """Independent literal group duplication verifies optimized full-size calculations."""
    from sklearn.metrics import f1_score
    with np.load(OUT/'cache/seed_42.npz') as z:
        m='V3_Full';shift='no_stats';keys=['y','group','index']+[f'{c}__{m}__{v}' for c in ('clean',shift) for v in ('pred','score','verdict','available','accepted_fixed')];d={k:z[k] for k in keys}
    with np.load(BASE/'bootstrap_draws.npz') as z:w=z['group_weights'][0].astype(np.int16)
    order0=ordered(d,'clean',m);order1=ordered(d,shift,m);group=d['group'];y=d['y'];p0=d[f'clean__{m}__pred'];p1=d[f'{shift}__{m}__pred']
    expanded=np.repeat(order0,w[group[order0]]);yy=y[expanded];pp=p0[expanded];risk=np.cumsum(yy!=pp)/np.arange(1,40001)
    result=weighted_curves(group[order0].astype(np.int16),(2*y[order0]+p0[order0]).astype(np.int8),w[None],K,np.r_[0,np.cumsum(1/np.arange(1,40001))])
    f,e,ar,cut,take=result;np.testing.assert_allclose(e[0],risk[K-1],atol=1e-12);np.testing.assert_allclose(ar[0],risk.mean(),atol=1e-12)
    for j,k in enumerate(K):np.testing.assert_allclose(f[0,j],f1_score(yy[:k],pp[:k],labels=[0,1],average='macro',zero_division=0),atol=1e-12)
    c0=np.array([[i,j] for i in order0 for j in range(w[group[i]])]);c1=np.array([[i,j] for i in order1 for j in range(w[group[i]])])
    with np.load(OUT/'curves'/f'clean__{m}.npz') as z:cut0=z['cuts'][0,1:2];take0=z['takes'][0,1:2]
    with np.load(OUT/'curves'/f'{shift}__{m}.npz') as z:cut1=z['cuts'][0,1:2];take1=z['takes'][0,1:2]
    D=(p0==y)&(p1!=y);conf=d[f'clean__{m}__accepted_fixed']&d[f'{shift}__{m}__accepted_fixed'];native=(d[f'clean__{m}__verdict']<2)&(d[f'{shift}__{m}__verdict']<2)
    flags=np.stack(((p0!=y)&(p1==y),D,D&conf,D&conf&native),1).astype(np.int8);rank=np.argsort(order1).astype(np.int32)
    actual=paired_coverage_counts(order0,rank,group,flags,w[None],cut0,take0,cut1,take1)
    for j,q in enumerate(FIXED_COV):
        k=round(q*40000);a=set(map(tuple,c0[:k]));b=set(map(tuple,c1[:k]));joint=a&b
        expected=np.r_[np.sum([flags[i] for i,copy in joint],axis=0),len(joint)]
        np.testing.assert_allclose(actual[0,j],expected,atol=1e-12)
    dump(OUT/'independent_full_size_check.json',dict(status='passed',seed=42,method=m,bootstrap_draw_index=0,explicitly_duplicated_rows=40000,coverage_points_checked=len(K),cd_coverages_checked=FIXED_COV,aurc_explicit=float(risk.mean()),aurc_kernel=float(ar[0]),model_inference_repeated=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','collect','curves','cd','report']);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage=='register':register()
    else:
        assert (OUT/'PREREGISTRATION.json').exists()
        if a.stage=='collect':parallel(collect_seed,[(s,) for s in SEEDS],a.workers)
        elif a.stage=='curves':parallel(run_curve_job,[(m,c) for c in CONDS for m in METHODS],a.workers)
        elif a.stage=='cd':parallel(run_cd_job,[(m,c) for c in CONDS[1:] for m in METHODS],a.workers)
        else:report()

if __name__=='__main__':main()
