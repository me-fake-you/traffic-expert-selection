"""Natural observed-cohort conditions and separate system-boundary replication."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import sys,json,gzip,hashlib,time,argparse,collections
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.append(str(Path.home()/'AppData/Roaming/Python/Python312/site-packages'))
BASE=ROOT/'output/ustc_authoritative_fusion_v1'
OUT=BASE/'natural_shift_audit_v1'
SEEDS=list(range(42,52))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def csv(p,x):pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')

def inspect_data():
    for p in (BASE/'PROTOCOL.md',BASE/'features.npz',BASE/'frozen_predictions',BASE/'selective_reliability_v1/PREREGISTRATION.json'):
        if not p.exists():raise FileNotFoundError(f'STOP: prerequisite missing: {p}')
    OUT.mkdir(exist_ok=True)
    if (OUT/'metadata.csv').exists():return
    audit=json.loads((BASE/'data_audit.json').read_text());assert audit['stats_dim']==11 and audit['temporal_dim']==30 and sha(BASE/'features.npz')==audit['features_sha256']
    with np.load(BASE/'features.npz') as z:ids=z['sample_id'];groups=z['group'];gnames=z['group_names'];app=z['app']
    mapping={sid:i for i,sid in enumerate(ids)};rows=[];source_rows=[];hist=collections.Counter();joint=collections.Counter();examples=[]
    selected_for_audit=set(np.concatenate([np.flatnonzero(groups==g)[:10] for g in range(20)]).tolist())
    for source in audit['source_files']:
        path=ROOT/source['path'];assert sha(path)==source['sha256']
        count=0;starts=[];end=[];tls_count=0;trunc_count=0;lens=[];segments=[];mismatch=0;source_group=None
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for lineno,line in enumerate(f,1):
                r=json.loads(line);sq=r['sequence'];st=r['stats'];pr=r['provenance'];tls=r.get('tls',{})
                n=len(sq['packet_lengths']);pc=int(st['packet_count']);iat=sq['iats'];duration=float(st.get('duration',0));start=float(pr.get('capture_start_epoch',np.nan));segment=int(pr.get('segment_index',0));visible=bool(tls.get('version') or tls.get('cipher_suite'));trunc=bool(sq['truncated'])
                aligned=(len(sq['directions'])==n and len(iat)==max(n-1,0));fullwindow=(pc==n and abs(sum(iat)-duration)<=1e-5)
                group=('malware:'+r['labels']['family']) if r['labels']['binary']=='malicious' else ('benign:'+r['labels']['application']);source_group=group
                lengthbin='1_4' if n<=4 else '5_16' if n<=16 else '17_63' if n<64 else '64'
                count+=1;starts.append(start);end.append(start+duration);lens.append(n);segments.append(segment);tls_count+=visible;trunc_count+=trunc;mismatch+=not aligned
                hist[('all','tls',int(visible))]+=1;hist[('all','truncated',int(trunc))]+=1;hist[('all','length',n)]+=1;hist[('all','alignment',int(aligned))]+=1;hist[('all','full_window',int(fullwindow))]+=1;hist[('all','segment',segment)]+=1
                joint[(group,int(visible),int(trunc),lengthbin,int(segment>0))]+=1
                sid=r['sample_id']
                if sid in mapping:
                    i=mapping[sid];assert str(gnames[groups[i]])==group
                    assert app[i].tolist()==[bool(st),n>=1,visible]
                    rows.append(dict(index=i,sample_id=sid,group=int(groups[i]),group_name=group,source_file=pr['source_file'],source_json=str(path.relative_to(ROOT)),source_line=lineno,flow_start=start,flow_end=start+duration,packet_count=pc,sequence_length=n,sequence_truncated=trunc,sequence_span=sum(iat),stats_duration=duration,sequence_arrays_aligned=aligned,views_full_window_aligned=fullwindow,segment_index=segment,tls_present=visible,stats_applicable=bool(st),temporal_applicable=n>=1))
                    if i in selected_for_audit:examples.append(dict(index=i,record=r))
                    for dimension,value in [('tls',int(visible)),('truncated',int(trunc)),('length',n),('alignment',int(aligned)),('full_window',int(fullwindow)),('segment',segment)]:hist[('selected',dimension,value)]+=1
        source_rows.append(dict(source_file=r['provenance']['source_file'],group_name=source_group,records=count,start_min=min(starts),start_max=max(starts),end_max=max(end),plausible_absolute_time_min=min(starts)>=946684800,tls_count=tls_count,tls_rate=tls_count/count,truncated_count=trunc_count,truncated_rate=trunc_count/count,sequence_min=min(lens),sequence_max=max(lens),segment_max=max(segments),sequence_misaligned=mismatch))
        print(path.name,count,flush=True)
    assert len(rows)==40000 and len(set(r['index'] for r in rows))==40000
    csv(OUT/'metadata.csv',sorted(rows,key=lambda r:r['index']));csv(OUT/'capture_inventory.csv',source_rows)
    csv(OUT/'empirical_distributions.csv',[dict(population=pop,dimension=dim,value=val,count=count,denominator=40000 if pop=='selected' else audit['source_rows'],rate=count/(40000 if pop=='selected' else audit['source_rows'])) for (pop,dim,val),count in sorted(hist.items())])
    csv(OUT/'population_joint_distribution.csv',[dict(group_name=g,tls_present=t,sequence_truncated=tr,length_bin=l,later_segment=seg,count=n) for (g,t,tr,l,seg),n in sorted(joint.items())])
    with gzip.open(OUT/'audit_flow_examples.jsonl.gz','wt',encoding='utf-8') as f:
        for r in examples:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(OUT/'inspection.json',dict(source_records=sum(r['records'] for r in source_rows),selected_records=len(rows),captures=len(source_rows),source_feature_sha256=audit['features_sha256'],audit_examples=len(examples),stats_dim=11,temporal_dim=30,source_hashes_verified=True,temporal_clock_domains_comparable=all(r['plausible_absolute_time_min'] for r in source_rows)))

CONDITIONS=['observed_all','tls_absent','tls_present','truncated_prefix','complete_sequence','short_sequence_1_4','full_window_mismatch','later_flow_segment','capture_late_half','capture_early_half','temporal_early_late_refit']
def prepare():
    import shutil
    from sklearn.model_selection import StratifiedGroupKFold
    if (OUT/'NATURAL_PROTOCOL.json').exists():return
    meta=pd.read_csv(OUT/'metadata.csv').sort_values('index');assert np.array_equal(meta['index'],np.arange(40000))
    early=np.zeros(40000,bool);late=early.copy();cutrows=[]
    # Absolute timestamps cannot be compared across captures. A separate relative-time study
    # uses a per-capture median START boundary, purging early records that overlap it.
    for cap,sub in meta.groupby('source_file',sort=True):
        cutoff=float(sub.flow_start.median());a=sub['index'].to_numpy();early[a]=sub.flow_end.to_numpy()<cutoff;late[a]=sub.flow_start.to_numpy()>cutoff
        cutrows.append(dict(source_file=cap,cutoff=cutoff,early=int(early[a].sum()),late=int(late[a].sum()),purged_ties_or_overlaps=int((~(early[a]|late[a])).sum()),clock='within-capture only'))
    masks=dict(observed_all=np.ones(40000,bool),tls_absent=~meta.tls_present.to_numpy(),tls_present=meta.tls_present.to_numpy(),truncated_prefix=meta.sequence_truncated.to_numpy(),complete_sequence=~meta.sequence_truncated.to_numpy(),short_sequence_1_4=meta.sequence_length.to_numpy()<=4,full_window_mismatch=~meta.views_full_window_aligned.to_numpy(),later_flow_segment=meta.segment_index.to_numpy()>0,capture_late_half=late,capture_early_half=early)
    rows=[];groups=meta.group.to_numpy()
    for c,mask in masks.items():
        sel=meta[mask];rows.append(dict(condition=c,samples=int(mask.sum()),cohort_rate=float(mask.mean()),groups=sel.group.nunique(),benign=int((sel.group<10).sum()),malicious=int((sel.group>=10).sum()),tls_rate=float(sel.tls_present.mean()) if len(sel) else None,truncation_rate=float(sel.sequence_truncated.mean()) if len(sel) else None))
    np.savez_compressed(OUT/'condition_indices.npz',**{c:np.flatnonzero(mask) for c,mask in masks.items()})
    csv(OUT/'condition_inventory.csv',rows);csv(OUT/'within_capture_time_boundaries.csv',cutrows)
    temporal=OUT/'temporal_refit';temporal.mkdir(exist_ok=True);shutil.copy2(BASE/'features.npz',temporal/'features.npz')
    with np.load(BASE/'features.npz') as z:y=z['y']
    capture_splits=[];temporal_splits=[]
    for seed in SEEDS:
        for fold in range(20):
            with np.load(BASE/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz') as z:tr=z['train'];va=z['validation'];te=z['test']
            for cap,sub in meta.iloc[te].groupby('source_file',sort=True):
                capid=hashlib.sha256(cap.encode()).hexdigest()[:12];test=sub['index'].to_numpy()
                assert not set(meta.iloc[tr].source_file)&set(sub.source_file)
                dest=OUT/'leave_capture_out'/f'seed_{seed}'/f'{capid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,train=tr,validation=va,test=test,parent_fold=fold)
                capture_splits.append(dict(seed=seed,parent_fold=fold,source_file=cap,ntrain=len(tr),nvalidation=len(va),ntest=len(test),path=str(dest.relative_to(OUT)),sha256=sha(dest)))
            t=tr[early[tr]];v=va[early[va]];e=te[late[te]]
            assert len(t)>0 and len(v)>0 and len(e)>0 and len(np.unique(y[t]))==2 and len(np.unique(y[v]))==2
            assert not (set(groups[t])&set(groups[e]) or set(groups[v])&set(groups[e]) or set(groups[t])&set(groups[v]))
            inner=np.full(40000,-1,np.int8)
            for k,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(t,y[t],groups[t])):inner[t[b]]=k
            dest=temporal/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,train=t,validation=v,test=e,inner_fold=inner)
            temporal_splits.append(dict(seed=seed,fold=fold,ntrain=len(t),nvalidation=len(v),ntest=len(e),path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    csv(OUT/'leave_capture_out_indices.csv',capture_splits);csv(OUT/'temporal_indices.csv',temporal_splits)
    protected=json.loads((BASE/'selective_reliability_v1/PREREGISTRATION.json').read_text())['protected_sha256']
    for p,h in protected.items():assert sha(BASE/p)==h
    dump(OUT/'NATURAL_PROTOCOL.json',dict(seeds=SEEDS,outer_folds=20,conditions=CONDITIONS,metadata_used_for_condition_definition_only=True,source_feature_sha256=sha(BASE/'features.npz'),conditions_are_observed_joint_strata=True,random_independent_missingness_masks=False,bootstrap_draws=10000,bootstrap_weights='original paired stratified group draws; reselect coverage within each condition',rank_rule='same as selective_reliability_v1; native rejected last, global index breaks ties',absolute_calendar_split='infeasible: only four captures have plausible calendar epochs; all four are malware',temporal_supplement='new model training on earlier half of each TRAIN/VAL capture, testing later half of held-out-group captures; overlap/ties purged; within-capture clock only, no global-calendar forecast claim',leave_capture_out='24 capture partitions nested in same 20 held-out groups; parent train/validation unchanged, no same-group capture leaks',representative_figure_conditions=['tls_absent','temporal_early_late_refit'],cross_dataset='optional skipped; local CIC-IDS2017 processed directory has manifests only, no aligned flow/packet records; no synthetic reconstruction or pooling',protected_sha256=protected,registered_unix=time.time()))
    print('Natural conditions and 200 temporal split indices locked',flush=True)

@njit(cache=True)
def natural_bootstrap(group,code,w,cov,harmonic):
    B=len(w);Q=len(cov);f1=np.full((B,Q),np.nan);err=f1.copy();aurc=np.full(B,np.nan);sizes=np.zeros(B,np.int32)
    ng=np.zeros(20,np.int32)
    for g in group:ng[g]+=1
    for b in range(B):
        N=0
        for g in range(20):N+=int(w[b,g])*ng[g]
        sizes[b]=N
        if N==0:continue
        ks=np.maximum(1,np.rint(cov*N).astype(np.int32));cf=np.zeros(4,np.int64);n=0;wrong=0;area=0.;j=0
        for r in range(len(group)):
            copies=int(w[b,group[r]])
            if copies==0:continue
            c=code[r];bad=(c==1 or c==2);dh=harmonic[n+copies]-harmonic[n]
            area+=copies+(wrong-n)*dh if bad else wrong*dh
            cf[c]+=copies;n+=copies
            if bad:wrong+=copies
            while j<Q and n>=ks[j]:
                excess=n-ks[j];cf[c]-=excess;a=2*cf[0]+cf[1]+cf[2];bb=2*cf[3]+cf[1]+cf[2]
                f1[b,j]=.5*((2*cf[0]/a if a else 0)+(2*cf[3]/bb if bb else 0));err[b,j]=(cf[1]+cf[2])/ks[j]
                cf[c]+=excess;j+=1
        aurc[b]=area/N
    return f1,err,aurc,sizes

def natural_job(method,condition):
    import authoritative_selective as a
    dest=OUT/'curves'/f'{condition}__{method}.npz'
    if dest.exists():return dict(method=method,condition=condition,status='cached')
    with np.load(BASE/'bootstrap_draws.npz') as z:w=np.vstack((np.ones((1,20)),z['group_weights'])).astype(np.int16)
    h=np.r_[0,np.cumsum(1/np.arange(1,40001))];mask=np.ones(40000,bool)
    if condition!='temporal_early_late_refit':
        with np.load(OUT/'condition_indices.npz') as z:ix=z[condition]
        mask[:]=False;mask[ix]=True
    fs=[];es=[];ars=[];nc=[];cache={};cf=[];nss=[];groups_seen=[]
    for seed in SEEDS:
        source=(OUT/'temporal_refit/cache' if condition=='temporal_early_late_refit' else BASE/'selective_reliability_v1/cache')/f'seed_{seed}.npz'
        with np.load(source) as z:
            index=z['index'];sel=mask[index];y=z['y'][sel];g=z['group'][sel];index=index[sel]
            prefix='clean__'+method+'__';pred=z[prefix+'pred'][sel];score=z[prefix+'score'][sel];verdict=z[prefix+'verdict'][sel];app=z[prefix+'available'][sel]
        assert app.all();ranking=np.where(verdict<2,score,-1.)
        order=np.lexsort((index,-ranking));code=(2*y+pred)[order].astype(np.int8);group=g[order].astype(np.int16)
        key=hashlib.sha256(group.tobytes()+code.tobytes()).hexdigest()
        if key not in cache:cache[key]=natural_bootstrap(group,code,w,a.COV,h)
        f,e,ar,sizes=cache[key];fs.append(f);es.append(e);ars.append(ar);nss.append(sizes);nc.append(float(np.mean(verdict<2)) if len(verdict) else np.nan)
        cf.append(np.bincount(2*y+pred,minlength=4));groups_seen.append(len(set(g)))
    dest.parent.mkdir(exist_ok=True);fs=np.asarray(fs);es=np.asarray(es);ars=np.asarray(ars)
    np.savez_compressed(dest,point_f1=fs[:,0],point_error=es[:,0],point_aurc=ars[:,0],boot_f1=np.mean(fs[:,1:],axis=0),boot_error=np.mean(es[:,1:],axis=0),boot_aurc=np.mean(ars[:,1:],axis=0),sizes=nss,native_coverage=nc,confusion=cf,groups=groups_seen)
    return dict(method=method,condition=condition,status='complete',samples=int(nss[0][0]),unique_orderings=len(cache))

def curves():
    import authoritative_selective as a
    assert (OUT/'temporal_refit/COMPLETE.json').exists(),'Temporal study must finish before joint reporting'
    failed=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(natural_job,m,c):(m,c) for c in CONDITIONS for m in a.METHODS}
        for i,future in enumerate(as_completed(jobs),1):
            try:r=future.result()
            except Exception as e:
                import traceback
                r=dict(status='failed',cell=jobs[future],error=repr(e),traceback=traceback.format_exc());failed.append(r)
            r['elapsed']=time.perf_counter()-start
            with (OUT/'curve_execution.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(r)+'\n')
            if i%9==0 or failed:print(i,'/',len(jobs),r,flush=True)
    if failed:raise RuntimeError('Natural curve failures logged')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inspect','prepare','curves']);a=p.parse_args()
    globals()[dict(inspect='inspect_data',prepare='prepare',curves='curves')[a.stage]]()
if __name__=='__main__':main()
