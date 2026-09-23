"""One bounded Stats8 replacement. Preserve old Temporal chain; regenerate Stats targets."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse,itertools,traceback,warnings
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];DEST=OUT/'backend'
BASE=OUT.parent/'mad_etd_icassp2027_upgrade_014';MATRICES=OUT.parent/'mad_etd_icassp2027_upgrade_013/analysis'
P1=OUT.parent/'mad_etd_icassp2027_p1_20260920'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,p):
    tn,fp,fn,tp=[int(v.sum()) for v in [(y==0)&(p==0),(y==0)&(p==1),(y==1)&(p==0),(y==1)&(p==1)]]
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,macro_f1=tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1),malicious_recall=tp/max(tp+fn,1),false_positive_rate=fp/max(tn+fp,1))
def comparison(y,p,first):
    return dict(n=len(y),**metric(y,p),C=int(((first!=y)&(p==y)).sum()),D=int(((first==y)&(p!=y)).sum()),switches=int((first!=p).sum()),
        corrected_FN=int(((first==0)&(y==1)&(p==1)).sum()),corrected_FP=int(((first==1)&(y==0)&(p==0)).sum()),
        introduced_FN=int(((first==1)&(y==1)&(p==0)).sum()),introduced_FP=int(((first==0)&(y==0)&(p==1)).sum()))
def meta(a,b,ra,rb):return np.column_stack((a,b,abs(a-.5),abs(b-.5),np.broadcast_to(ra,a.shape),np.broadcast_to(rb,a.shape),a-b,abs(a-b),a*b,a>=.5,b>=.5))
def arbiter(k,cfg):
    return HistGradientBoostingClassifier(**cfg['arbiter_hgb']) if k=='hgb' else make_pipeline(StandardScaler(),LogisticRegression(**cfg['arbiter_logistic']))
def choose(d,mat,ids=None):
    ids=list(range(mat.shape[1])) if ids is None else list(ids);first=d['first_probability']>=.5;y=d['label']
    rows=[dict(candidate=j,switches=int((mat[:,j]!=first).sum()),**metric(y,mat[:,j])) for j in range(mat.shape[1])]
    recall=metric(y,first)['malicious_recall'];F=[j for j in ids if rows[j]['malicious_recall']>=recall-1e-12];assert F,'no feasible candidate'
    keys=[(r['macro_f1'],r['malicious_recall'],-r['switches']) for r in rows]
    best=max(F,key=lambda j:keys[j]);opt=[j for j in F if keys[j]==keys[best]]
    return best,F,opt,rows
def learned(d,q,grid):
    first=d['first_probability']>=.5;second=d['second_probability']>=.5;conf=d['trigger']&(first!=second)
    return np.column_stack([np.where(conf&(q>=np.where(first,hi,lo)),second,first).astype(np.uint8) for lo,hi in grid])
def fixed(d,ra,rb,grid):
    first=d['first_probability']>=.5;second=d['second_probability']>=.5;aT=abs(d['first_probability']-.5)*ra;aS=abs(d['second_probability']-.5)*rb
    cols=[]
    for lo,hi in grid:
        switch=np.zeros(len(first),dtype=bool)
        for direction,c in [(0,lo),(1,hi)]:
            if c!='no_switch':switch|=(first==direction)&d['trigger']&(first!=second)&(aS>c*aT)
        cols.append(np.where(switch,second,first).astype(np.uint8))
    return np.column_stack(cols)
def preflight():
    cfg=read(DEST/'config.json');original=read(ROOT/cfg['original_protocol']);hist=ROOT/cfg['history'];inputlock=read(P1/'development/input_lock.json')
    source=ROOT/cfg['source'];assert sha(source)==inputlock[str(source)]
    with np.load(source,allow_pickle=False) as d:data={k:d[k].copy() for k in ['x_stats','y','group','sample_hash']}
    assert data['x_stats'].shape==(40000,8) and np.isfinite(data['x_stats']).all()
    splits=read(hist/'split_manifest.json');assert sha(hist/'split_manifest.json')==inputlock[str(hist/'split_manifest.json')]
    pv=pd.read_csv(ROOT/original['provenance']);pv=pv[pv.partition=='development40k'].set_index('sample_hash').loc[data['sample_hash']]
    assert np.array_equal(pv.label,data['y']) and np.array_equal(pv.group,data['group'])
    locks={str(source):sha(source),str(DEST/'config.json'):sha(DEST/'config.json'),str(ROOT/cfg['original_protocol']):sha(ROOT/cfg['original_protocol']),str(hist/'split_manifest.json'):sha(hist/'split_manifest.json')}
    for sp in splits:
        f=sp['outer_fold'];roles=sp['groups'];ix={r:np.flatnonzero(np.isin(data['group'],g)) for r,g in roles.items()}
        for a,b in itertools.combinations(roles,2):
            assert not set(roles[a])&set(roles[b]);assert not set(pv.iloc[ix[a]].capture)&set(pv.iloc[ix[b]].capture)
        for r,idx in ix.items():assert set(data['y'][idx])=={0,1}
        deps=read(hist/f'seed_42_fold_{f}/oof_dependencies.json')
        for dep in deps:
            assert set(dep['fit_groups'])|set(dep['predict_groups'])==set(roles['train'])
            assert not set(dep['predict_groups'])&(set(dep['fit_groups'])|set(dep['calibration_groups']))
            assert set(dep['calibration_groups'])==set(roles['calibration'])
        files=[hist/f'seed_42_fold_{f}/oof_predictions.npz',hist/f'seed_42_fold_{f}/oof_dependencies.json',hist/f'seed_42_fold_{f}/temporal_stats_locked_policy.json']
        files += [hist/f'seed_42_fold_{f}/models/{p}_temporal.joblib' for p in ['full','inner_0','inner_1','inner_2']]
        for p in files:assert sha(p)==inputlock[str(p)];locks[str(p)]=sha(p)
        # Frozen Temporal probabilities only; no old Stats correctness is reused.
        for role in ['selection','evaluation']:
            p=MATRICES/f'row100_20260920_logistic_fold{f}_{role}.npz';locks[str(p)]=sha(p)
    dump(DEST/'input_lock.json',locks)
    dump(DEST/'preflight.json',dict(status='PASS_ORIGINAL_ROLES_AND_FINITE_STATS8',sklearn=sklearn.__version__,rows=40000,groups=20,temporal_retrained=False,old_stats_targets_used=False,new_blind_test=False))
    return cfg,original,hist,data,splits,locks
def temporal_cache(data,f,role,ids):
    with np.load(MATRICES/f'row100_20260920_logistic_fold{f}_{role}.npz',allow_pickle=False) as z:
        assert np.array_equal(data['sample_hash'][ids],z['sample_hash']) and np.array_equal(data['y'][ids],z['label'])
        return z['first_probability'].copy()
def pilot(cfg,original,hist,data,splits):
    dest=DEST/'pilot_001';dest.mkdir(exist_ok=False);sp=splits[0];dep=read(hist/'seed_42_fold_0/oof_dependencies.json')[0]
    idx=lambda groups:np.flatnonzero(np.isin(data['group'],groups));fit=idx(dep['fit_groups']);pr=idx(dep['predict_groups']);cal=idx(dep['calibration_groups'])
    start=time.perf_counter();et=ExtraTreesClassifier(**cfg['expert']).fit(data['x_stats'][fit],data['y'][fit]);b=et.predict_proba(data['x_stats'][pr])[:,1];cb=et.predict_proba(data['x_stats'][cal])[:,1];seconds=time.perf_counter()-start
    joblib.dump(et,dest/'stats.joblib');rb=metric(data['y'][cal],cb>=.5)['macro_f1']
    with np.load(hist/'seed_42_fold_0/oof_predictions.npz',allow_pickle=False) as z:
        pos=np.flatnonzero(np.isin(z['indices'],pr));assert np.array_equal(z['indices'][pos],pr);a=z['temporal'][pos];ra=z['reliability_temporal'][pos]
    mask=(a>=.5)!=(b>=.5);target=((b>=.5)==data['y'][pr]).astype(int);X=meta(a,b,ra,rb)[mask];times={};fit_count=1
    for k in ['logistic','hgb']:
        if len(np.unique(target[mask]))<2:times[k]=None;continue
        start=time.perf_counter();model=arbiter(k,original).fit(X,target[mask]);assert np.isfinite(model.predict_proba(X[:16])).all();times[k]=time.perf_counter()-start;fit_count+=1;joblib.dump(model,dest/f'{k}.joblib')
    projected=max(seconds,1)*20*3+sum(300 if v is None else max(v,30) for v in times.values())*5*4+600
    result=dict(status='PILOT_COMPLETED',fit_rows=len(fit),prediction_rows=len(pr),fresh_disagreements=int(mask.sum()),new_stats_target_classes=np.unique(target[mask]).tolist(),stats_fit_and_development_inference_seconds=seconds,arbiter_seconds=times,
        projected_conservative_seconds=projected,within_budget=projected<=cfg['budget_seconds'],budget_seconds=cfg['budget_seconds'],outer_evaluated=False,pilot_fits=fit_count,config_sha256=sha(DEST/'config.json'),script_sha256=sha(__file__))
    dump(dest/'pilot.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
def run(cfg,original,hist,data,splits,locks):
    pil=read(DEST/'pilot_001/pilot.json');assert pil['within_budget'] and pil['config_sha256']==sha(DEST/'config.json')
    dest=DEST/'run_001';dest.mkdir(exist_ok=False);started=time.perf_counter();grid=list(itertools.product(cfg['arbiter_thresholds'],repeat=2));fgrid=list(itertools.product(cfg['fixed_coefficients'],repeat=2))
    selections=[];candidate_rows=[];fits=[];support=[];refs={};rolesrows=[]
    dump(dest/'run_lock.json',dict(start=time.strftime('%Y-%m-%dT%H:%M:%S%z'),script_sha256=sha(__file__),config_sha256=sha(DEST/'config.json'),input_lock_sha256=sha(DEST/'input_lock.json'),exploratory_now_not_pre_E1=True,outer_evaluation_started=False))
    def fit_model(model,X,y,path,fold,component):
        assert time.perf_counter()-started<cfg['budget_seconds'],'8h runtime budget reached'
        assert set(y)=={0,1},f'one-class {component}'
        ts=time.perf_counter()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');model.fit(X,y)
        joblib.dump(model,path);fits.append(dict(fold_id=fold,component=component,rows=len(y),seconds=time.perf_counter()-ts,model=str(path),sha256=sha(path),warnings=[str(w.message) for w in ws]));return model
    for sp in splits:
        f=sp['outer_fold'];folder=dest/f'fold_{f}';folder.mkdir();(folder/'models').mkdir();g=sp['groups'];ix={r:np.flatnonzero(np.isin(data['group'],v)) for r,v in g.items()};tr=ix['train'];cal=ix['calibration'];va=ix['selection']
        with np.load(hist/f'seed_42_fold_{f}/oof_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['indices'],tr);oa=z['temporal'].copy();ora=z['reliability_temporal'].copy()
        ob=np.full(len(tr),np.nan);orb=np.full(len(tr),np.nan);inner=np.full(len(tr),-1);deps=read(hist/f'seed_42_fold_{f}/oof_dependencies.json')
        for dep in deps:
            fitix=np.flatnonzero(np.isin(data['group'],dep['fit_groups']));pos=np.flatnonzero(np.isin(data['group'][tr],dep['predict_groups']));j=dep['fold']
            model=fit_model(ExtraTreesClassifier(**cfg['expert']),data['x_stats'][fitix],data['y'][fitix],folder/f'models/inner_{j}_stats.joblib',f,f'inner_{j}_stats')
            ob[pos]=model.predict_proba(data['x_stats'][tr[pos]])[:,1];orb[pos]=metric(data['y'][cal],model.predict_proba(data['x_stats'][cal])[:,1]>=.5)['macro_f1'];inner[pos]=j
        assert np.isfinite(ob).all() and np.isfinite(orb).all() and (inner>=0).all()
        target=((ob>=.5)==data['y'][tr]).astype(np.uint8);mask=(oa>=.5)!=(ob>=.5);X=meta(oa,ob,ora,orb)[mask]
        for direction in [0,1]:
            dmask=mask&((oa>=.5)==direction);assert set(target[dmask])=={0,1},f'fold {f}: direction {direction} lacks correctness support'
            for t in [0,1]:support.append(dict(fold_id=f,direction=direction,target=t,rows=int((dmask&(target==t)).sum()),groups=len(set(data['group'][tr[dmask&(target==t)]]))))
        np.savez_compressed(folder/'oof.npz',indices=tr,sample_hash=data['sample_hash'][tr],group=data['group'][tr],label=data['y'][tr],temporal=oa,stats=ob,reliability_temporal=ora,reliability_stats=orb,disagreement=mask,target=target,inner=inner)
        et=fit_model(ExtraTreesClassifier(**cfg['expert']),data['x_stats'][tr],data['y'][tr],folder/'models/full_stats.joblib',f,'full_stats')
        history=read(hist/f'seed_42_fold_{f}/temporal_stats_locked_policy.json');ra=history['reliability']['temporal'];cb=et.predict_proba(data['x_stats'][cal])[:,1];rb=metric(data['y'][cal],cb>=.5)['macro_f1']
        np.savez_compressed(folder/'reliability.npz',indices=cal,label=data['y'][cal],stats_probability=cb)
        a=temporal_cache(data,f,'selection',va);b=et.predict_proba(data['x_stats'][va])[:,1]
        d=dict(sample_hash=data['sample_hash'][va],group=data['group'][va],label=data['y'][va],first_probability=a,second_probability=b,trigger=abs(a-.5)<.495)
        fm=fixed(d,ra,rb,fgrid);models={};matrices={'Fixed-Dev16':fm};grids={'Fixed-Dev16':fgrid}
        for k,name in [('logistic','Logistic'),('hgb','HGB')]:
            models[k]=fit_model(arbiter(k,original),X,target[mask],folder/f'models/{k}.joblib',f,k);q=models[k].predict_proba(meta(a,b,ra,rb))[:,1];mat=learned(d,q,grid)
            matrices[name+'-Dev16']=mat;matrices[name+'-Global4']=mat;grids[name+'-Dev16']=grid;grids[name+'-Global4']=grid
            np.savez_compressed(folder/f'{k}_selection.npz',**d,q=q,thresholds=np.array(grid),predictions=mat)
        for method,mat in matrices.items():
            ids=[0,5,10,15] if method.endswith('Global4') else list(range(16));best,F,opt,rows=choose(d,mat,ids);lo,hi=grids[method][best]
            selections.append(dict(fold_id=f,method=method,selected=best,low=lo,high=hi,feasible_candidates=len(F),best_rank_candidates=len(opt),**rows[best]))
            candidate_rows.extend(dict(fold_id=f,method=method,low=grids[method][j][0],high=grids[method][j][1],feasible=j in F,chosen=j==best,**rows[j]) for j in ids)
        np.savez_compressed(folder/'fixed_selection.npz',**d,predictions=fm)
        refs[str(f)]=dict(reliability_temporal=ra,reliability_stats=rb,roles=g,inner_dependencies=deps,arbiter_fit_rows=int(mask.sum()))
        for role,indices in ix.items():
            rolesrows.extend(dict(fold_id=f,role=role,sample_hash=str(data['sample_hash'][i]),group=str(data['group'][i]),label=int(data['y'][i])) for i in indices)
        pd.DataFrame(selections).to_csv(dest/'selection_records.csv',index=False);dump(dest/'fit_records.json',fits)
        print(f'fold {f}: fresh Stats OOF/reliability and both arbiters fitted; development choices saved; no outer scoring',flush=True)
    pd.DataFrame(candidate_rows).to_csv(dest/'selection_candidates.csv',index=False);pd.DataFrame(support).to_csv(dest/'fitting_support.csv',index=False);pd.DataFrame(rolesrows).to_csv(dest/'role_manifest.csv.gz',index=False,compression='gzip');dump(dest/'reference_lock.json',refs)
    dump(dest/'evaluation_gate.json',dict(all5_folds_fitted=True,all25_choices_saved=len(selections)==25,policy_sha256=sha(dest/'selection_records.csv'),reference_sha256=sha(dest/'reference_lock.json'),outer_evaluated=False,elapsed_seconds=time.perf_counter()-started))
    frames=[];foldmetrics=[];mode=[];behavior=[];transfer=[];members=[];counts=[]
    for sp in splits:
        f=sp['outer_fold'];folder=dest/f'fold_{f}';ev=np.flatnonzero(np.isin(data['group'],sp['groups']['evaluation']));r=refs[str(f)];ra=r['reliability_temporal'];rb=r['reliability_stats'];et=joblib.load(folder/'models/full_stats.joblib')
        a=temporal_cache(data,f,'evaluation',ev);b=et.predict_proba(data['x_stats'][ev])[:,1];first=a>=.5
        d=dict(sample_hash=data['sample_hash'][ev],group=data['group'][ev],label=data['y'][ev],first_probability=a,second_probability=b,trigger=abs(a-.5)<.495)
        fm=fixed(d,ra,rb,fgrid);sel={r['method']:r for r in selections if r['fold_id']==f}
        outputs={'Temporal':first.astype(int),'Stats-ExtraTrees':(b>=.5).astype(int),'Fixed':fm[:,0],'Fixed-Dev16':fm[:,sel['Fixed-Dev16']['selected']]}
        np.savez_compressed(folder/'fixed_evaluation.npz',**d,predictions=fm)
        for k,name in [('logistic','Logistic'),('hgb','HGB')]:
            model=joblib.load(folder/f'models/{k}.joblib');q=model.predict_proba(meta(a,b,ra,rb))[:,1];em=learned(d,q,grid)
            np.savez_compressed(folder/f'{k}_evaluation.npz',**d,q=q,thresholds=np.array(grid),predictions=em)
            outputs[name+'-0.5']=em[:,5];outputs[name+'-Dev16']=em[:,sel[name+'-Dev16']['selected']];outputs[name+'-Global4']=em[:,sel[name+'-Global4']['selected']]
            with np.load(folder/f'{k}_selection.npz',allow_pickle=False) as z:s={v:z[v] for v in z.files}
            best,F,opt,scores=choose(s,s['predictions']);assert best==sel[name+'-Dev16']['selected'];sm=s['predictions'];E=[j for j in F if np.array_equal(sm[:,j],sm[:,best])]
            L=(em!=d['label'][:,None]).sum(axis=0);within=int(L[best]-min(L[E]));between=int(min(L[E])-min(L[F]));assert within>=0 and between>=0 and within+between==L[best]-min(L[F])
            mode.append(dict(fold_id=f,arbiter=k,selected=best,selected_low=grid[best][0],selected_high=grid[best][1],best_numeric_candidates=len(opt),best_prediction_patterns=len(np.unique(sm[:,opt].T,axis=0)),J=len(E),U=len(np.unique(em[:,E].T,axis=0)),selected_errors=int(L[best]),oracle_E=int(min(L[E])),oracle_F=int(min(L[F])),within=within,between=between,closure=0))
            for label,ids in [('all',list(range(16))),('feasible',F)]:
                vectors,inverse=np.unique(sm[:,ids].T,axis=0,return_inverse=True)
                counts.append(dict(fold_id=f,arbiter=k,set=label,n_candidates=len(ids),m=int((sm[:,ids].min(axis=1)!=sm[:,ids].max(axis=1)).sum()),K=len(vectors)))
                for cl in range(len(vectors)):
                    js=[j for n,j in enumerate(ids) if inverse[n]==cl];assert all(np.array_equal(sm[:,j],vectors[cl]) for j in js)
                    ms=[metric(d['label'],em[:,j]) for j in js]
                    behavior.append(dict(fold_id=f,arbiter=k,set=label,selection_class=cl,members=';'.join(map(str,js)),J=len(js),U=len(np.unique(em[:,js].T,axis=0)),outer_FP_min=min(v['fp'] for v in ms),outer_FP_max=max(v['fp'] for v in ms),outer_FN_min=min(v['fn'] for v in ms),outer_FN_max=max(v['fn'] for v in ms),outer_errors_min=int(min(L[js])),outer_errors_max=int(max(L[js]))))
            for j in range(16):members.append(dict(fold_id=f,arbiter=k,candidate=j,low=grid[j][0],high=grid[j][1],feasible=j in F,in_selected_class=j in E,best_rank=j in opt,**comparison(d['label'],em[:,j],first)))
            transfer.append(dict(fold_id=f,arbiter=k,selection_delta_pp=100*(metric(s['label'],sm[:,best])['macro_f1']-metric(s['label'],sm[:,5])['macro_f1']),outer_delta_pp=100*(metric(d['label'],em[:,best])['macro_f1']-metric(d['label'],em[:,5])['macro_f1']),**comparison(d['label'],em[:,best],em[:,5])))
        frame=pd.DataFrame({**d,**outputs});frame['fold_id']=f;frame.to_csv(folder/'predictions.csv.gz',index=False,compression='gzip');frames.append(frame)
        foldmetrics.extend(dict(fold_id=f,method=m,**comparison(d['label'],p,first)) for m,p in outputs.items());print(f'fold {f}: locked outer evaluation and all candidate records saved',flush=True)
    full=pd.concat(frames,ignore_index=True);assert len(full)==full.sample_hash.nunique()==40000
    pooled=[dict(method=m,**comparison(full.label.to_numpy(),full[m].to_numpy(),full.Temporal.to_numpy()),calls=1 if m in ['Temporal','Stats-ExtraTrees'] else 1+full.trigger.mean()) for m in cfg['policies']]
    groupmetrics=[dict(group=g,method=m,**comparison(p.label.to_numpy(),p[m].to_numpy(),p.Temporal.to_numpy())) for g,p in full.groupby('group') for m in cfg['policies']]
    for filename,rows in [('pooled_metrics.csv',pooled),('fold_metrics.csv',foldmetrics),('group_metrics.csv',groupmetrics),('mode_decomposition.csv',mode),('behavior_groups.csv',behavior),('candidate_metrics.csv',members),('candidate_counts.csv',counts),('selection_transfer.csv',transfer)]:pd.DataFrame(rows).to_csv(dest/filename,index=False)
    for p,h in locks.items():assert sha(p)==h,p
    dump(dest/'completion.json',dict(status='ONE_STATS_EXTRATREES_BACKEND_COMPLETED',seconds=time.perf_counter()-started,new_stats_fits=20,new_arbiter_fits=10,new_temporal_fits=0,pilot_fits=pil['pilot_fits'],outer_unique_rows=40000,all_locked_policies_retained=True,fresh_blind_test=False,external_generalization=False,pcap_rereads=0,bootstrap_runs=0))
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','run'],required=True);args=parser.parse_args()
    original=sys.stdout
    class Tee:
        def __init__(self,f):self.f=f
        def write(self,s):original.write(s);self.f.write(s);self.f.flush()
        def flush(self):original.flush();self.f.flush() if not self.f.closed else None
    with (OUT/'logs'/f'backend_{args.stage}.log').open('x',encoding='utf-8') as log:
        try:
            sys.stdout=Tee(log);print('command: '+str(sys.argv),flush=True)
            with threadpool_limits(limits=1):
                cfg,originalcfg,hist,data,splits,locks=preflight()
                if args.stage=='pilot':pilot(cfg,originalcfg,hist,data,splits)
                else:run(cfg,originalcfg,hist,data,splits,locks)
        except Exception:
            failure=traceback.format_exc();log.write(failure);dump(DEST/f'failure_{args.stage}.json',dict(command=sys.argv,error=failure));raise
        finally:sys.stdout=original
if __name__=='__main__':main()
