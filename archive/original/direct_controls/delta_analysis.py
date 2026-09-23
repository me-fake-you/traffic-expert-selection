"""Bounded selection/decomposition on saved arrays only. No estimator loading."""
from pathlib import Path
import sys,json,time,hashlib,itertools
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
BASE=OUT.parent/'mad_etd_icassp2027_upgrade_013';P1=OUT.parent/'mad_etd_icassp2027_p1_20260920'
DEST=OUT/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(f,kind,role,condition='row100'):
    p=BASE/f'analysis/{condition}_20260920_{kind}_fold{f}_{role}.npz'
    with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def metric(y,p):
    tn,fp,fn,tp=[int(v.sum()) for v in [(y==0)&(p==0),(y==0)&(p==1),(y==1)&(p==0),(y==1)&(p==1)]]
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,macro_f1=tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1),malicious_recall=tp/max(tp+fn,1),false_positive_rate=fp/max(tn+fp,1))
def scores(d,mat):
    first=d['first_probability']>=.5;y=d['label']
    return [dict(candidate=j,switches=int((mat[:,j]!=first).sum()),**metric(y,mat[:,j])) for j in range(mat.shape[1])]
def choose(d,mat,ids=None):
    rows=scores(d,mat);ids=list(range(mat.shape[1])) if ids is None else list(ids)
    recall=metric(d['label'],d['first_probability']>=.5)['malicious_recall']
    feasible=[j for j in ids if rows[j]['malicious_recall']>=recall-1e-12]
    assert feasible
    keys=[(r['macro_f1'],r['malicious_recall'],-r['switches']) for r in rows]
    chosen=max(feasible,key=lambda j:keys[j]) # input ascending order fixes remaining ties
    optimal=[j for j in feasible if keys[j]==keys[chosen]]
    return chosen,feasible,optimal,rows
def fixed_matrix(d,ref):
    a=d['first_probability'];b=d['second_probability'];first=a>=.5;second=b>=.5
    aT=abs(a-.5)*ref['reliability_temporal'];aS=abs(b-.5)*ref['reliability_stats']
    columns=[]
    for lo,hi in itertools.product([.25,.5,1.,None],repeat=2):
        change=np.zeros(len(a),dtype=bool)
        for direction,coef in [(False,lo),(True,hi)]:
            if coef is not None:
                mask=(first==direction)&d['trigger']&(first!=second)
                change[mask]=aS[mask]>coef*aT[mask]
        columns.append(np.where(change,second,first).astype(np.uint8))
    return np.column_stack(columns)
def comparison(y,p,first):
    return dict(n=len(y),**metric(y,p),C=int(((first!=y)&(p==y)).sum()),D=int(((first==y)&(p!=y)).sum()),switches=int((first!=p).sum()))
def main():
    assert not (DEST/'completion.json').exists();start=time.perf_counter()
    refs=read(P1/'e1/run_001/reference_lock.json');oldlock=pd.read_csv(P1/'e1/run_001/policy_lock.csv')
    sources=list((BASE/'analysis').glob('*.npz'))+[P1/'e1/run_001/reference_lock.json',P1/'e1/run_001/policy_lock.csv']
    ci_pairs=[('Fixed-Dev16','Fixed'),('Logistic-Dev16','Fixed-Dev16'),('HGB-Dev16','Fixed-Dev16'),('Logistic-Global4','Logistic-Dev16'),('HGB-Global4','HGB-Dev16')]
    dump(DEST/'delta_lock.json',dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=sys.argv,
        fixed_grid=list(itertools.product([.25,.5,1.,'no_switch'],repeat=2)),global_candidates=[0,5,10,15],
        selection='Original selection roles; recall >= Temporal-1e-12; descending F1,recall,-switches; first ascending pair; no-switch sentinel explicit.',
        population='same previously exposed 40k; no new independent source',new_training=0,new_model_inference=0,
        decomposition='Error counts: selected-min_F = (selected-min_E)+(min_E-min_F); E is selected full-vector class within F; oracle only.',
        conditional_ci=dict(pairs=ci_pairs,unit='20 original family/application groups, class stratified',repetitions=1000,seed=20260920,scope='frozen fitted models; no refitting or independent-activity inference'),
        stop='Any original-lock/vector mismatch stops; no grid changes or new models after scores.',inputs={str(p):sha(p) for p in sources}))
    selections=[];candrows=[];chosen_map={};fixed_grid=list(itertools.product([.25,.5,1.,None],repeat=2))
    # Finish all new selection using only selection matrices before loading outer arrays.
    for f in range(5):
        for method,kind in [('Fixed-Dev16','logistic'),('Logistic-Global4','logistic'),('HGB-Global4','hgb')]:
            d=matrix(f,kind,'selection');mat=fixed_matrix(d,refs[str(f)]) if method=='Fixed-Dev16' else d['predictions']
            ids=list(range(16)) if method=='Fixed-Dev16' else [0,5,10,15]
            best,feas,opt,rows=choose(d,mat,ids);chosen_map[f,method]=best
            grid=fixed_grid if method=='Fixed-Dev16' else d['thresholds'];lo,hi=grid[best]
            param=lambda x:'no_switch' if x is None else float(x)
            selections.append(dict(fold_id=f,method=method,candidate=best,parameter0=param(lo),parameter1=param(hi),candidate_budget=len(ids),feasible_candidates=len(feas),best_numeric_candidates=len(opt),**{k:v for k,v in rows[best].items() if k!='candidate'}))
            for j in ids:
                candrows.append(dict(fold_id=f,method=method,**rows[j],parameter0=param(grid[j][0]),parameter1=param(grid[j][1]),feasible=j in feas,optimal=j in opt,selected=j==best))
            np.savez_compressed(DEST/f'new_{method}_fold{f}_selection.npz',sample_hash=d['sample_hash'],label=d['label'],candidate_ids=np.array(ids),predictions=mat[:,ids])
    pd.DataFrame(selections).to_csv(DEST/'selection_records.csv',index=False)
    pd.DataFrame(candrows).to_csv(DEST/'selection_candidates.csv',index=False)
    dump(DEST/'selection_gate.json',dict(all15_new_selections_saved=True,selection_records_sha256=sha(DEST/'selection_records.csv'),outer_scoring_started=False))
    frames=[];foldmetrics=[];decomp=[];members=[]
    for f in range(5):
        d=matrix(f,'logistic','evaluation');h=matrix(f,'hgb','evaluation');assert np.array_equal(d['sample_hash'],h['sample_hash'])
        first=d['first_probability']>=.5;fm=fixed_matrix(d,refs[str(f)])
        lock=oldlock[(oldlock.outer_fold==f)&(oldlock.condition=='row100')]
        outputs={'Temporal':first.astype(int),'Fixed':fm[:,0],'Fixed-Dev16':fm[:,chosen_map[f,'Fixed-Dev16']],
            'Logistic-0.5':d['predictions'][:,5],'HGB-0.5':h['predictions'][:,5]}
        for kind,name,z in [('logistic','Logistic',d),('hgb','HGB',h)]:
            r=lock[lock.kind==kind].iloc[0];sid=int(np.flatnonzero((z['thresholds']==[r.selected_low,r.selected_high]).all(axis=1))[0])
            outputs[name+'-Dev16']=z['predictions'][:,sid];outputs[name+'-Global4']=z['predictions'][:,chosen_map[f,name+'-Global4']]
        frame=pd.DataFrame(dict(sample_hash=d['sample_hash'],group=d['group'],label=d['label'],fold_id=f,trigger=d['trigger'],**outputs));frames.append(frame)
        frame.to_csv(DEST/f'fold_{f}_direct_predictions.csv.gz',index=False,compression='gzip')
        foldmetrics.extend(dict(fold_id=f,method=name,**comparison(d['label'],p,first)) for name,p in outputs.items())
        for condition in ['row100','source_concentrated']:
            for kind in ['logistic','hgb']:
                s=matrix(f,kind,'selection',condition);e=matrix(f,kind,'evaluation',condition);sm=s['predictions'];em=e['predictions']
                best,F,opt,sr=choose(s,sm);r=oldlock[(oldlock.outer_fold==f)&(oldlock.condition==condition)&(oldlock.kind==kind)&(oldlock.subset_seed==20260920)].iloc[0]
                assert np.array_equal(s['thresholds'][best],[r.selected_low,r.selected_high])
                # Equality is checked on vectors directly, never hash alone.
                E=[j for j in F if np.array_equal(sm[:,j],sm[:,best])]
                assert set(E).issubset(F) and best in E
                losses=(em!=e['label'][:,None]).sum(axis=0);picked=int(losses[best]);emin=int(losses[E].min());fmin=int(losses[F].min())
                within=picked-emin;between=emin-fmin;total=picked-fmin
                assert min(within,between,total)>=0 and total==within+between
                er=scores(e,em);maxf=max(sr[j]['macro_f1'] for j in F);f1best=[j for j in F if sr[j]['macro_f1']==maxf]
                row=dict(fold_id=f,condition=condition,arbiter=kind,selected_candidate=best,selected_low=float(s['thresholds'][best,0]),selected_high=float(s['thresholds'][best,1]),
                    feasible_candidates=len(F),best_rank_tuple_candidates=len(opt),best_rank_tuple_patterns=len(np.unique(sm[:,opt].T,axis=0)),
                    best_f1_candidates=len(f1best),best_f1_patterns=len(np.unique(sm[:,f1best].T,axis=0)),selected_pattern_candidates=len(E),
                    selected_pattern_outer_patterns=len(np.unique(em[:,E].T,axis=0)),outer_FP_min=min(er[j]['fp'] for j in E),outer_FP_max=max(er[j]['fp'] for j in E),outer_FN_min=min(er[j]['fn'] for j in E),outer_FN_max=max(er[j]['fn'] for j in E),
                    selected_outer_errors=picked,oracle_E_errors=emin,oracle_F_errors=fmin,within_class_error_space=within,between_class_error_space=between,total_error_space=total,closure_residual=total-within-between)
                decomp.append(row)
                for j in F:
                    members.append(dict(fold_id=f,condition=condition,arbiter=kind,candidate=j,low=s['thresholds'][j,0],high=s['thresholds'][j,1],same_selection_pattern_as_selected=j in E,best_rank_tuple=j in opt,selected=j==best,outer_same_as_selected=np.array_equal(em[:,j],em[:,best]),outer_errors=int(losses[j]),outer_fp=er[j]['fp'],outer_fn=er[j]['fn']))
    pd.DataFrame(decomp).to_csv(DEST/'mode_decomposition.csv',index=False);pd.DataFrame(members).to_csv(DEST/'mode_members.csv',index=False)
    full=pd.concat(frames,ignore_index=True);assert len(full)==full.sample_hash.nunique()==40000
    methods=list(outputs);aggregate=[dict(method=m,**comparison(full.label.to_numpy(),full[m].to_numpy(),full.Temporal.to_numpy()),policy_calls=1.0 if m=='Temporal' else 1+full.trigger.mean()) for m in methods]
    pd.DataFrame(aggregate).to_csv(DEST/'direct_results.csv',index=False);pd.DataFrame(foldmetrics).to_csv(DEST/'direct_fold_results.csv',index=False)
    groups=[]
    for (g,y),part in full.groupby(['group','label']):
        for m in methods:groups.append(dict(group=g,label=y,method=m,**comparison(part.label.to_numpy(),part[m].to_numpy(),part.Temporal.to_numpy())))
    gg=pd.DataFrame(groups);gg.to_csv(DEST/'direct_group_results.csv',index=False)
    grouporder=sorted(full.group.unique());glabels=[int(full.loc[full.group==g,'label'].iloc[0]) for g in grouporder]
    counts={m:gg[gg.method==m].set_index('group').loc[grouporder,['tn','fp','fn','tp','C','D']].to_numpy() for m in methods}
    def reduce(x):
        tn,fp,fn,tp,C,D=x
        return np.array([tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1),tp/max(tp+fn,1),fp/max(tn+fp,1),C,D])
    rng=np.random.default_rng(20260920);strata=[np.flatnonzero(np.array(glabels)==v) for v in [0,1]]
    samples=[np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(1000)]
    intervals=[]
    for a,b in ci_pairs:
        point=reduce(counts[a].sum(axis=0))-reduce(counts[b].sum(axis=0))
        diffs=np.array([reduce(counts[a][ix].sum(axis=0))-reduce(counts[b][ix].sum(axis=0)) for ix in samples]);lo,hi=np.quantile(diffs,[.025,.975],axis=0)
        for j,name in enumerate(['macro_f1','malicious_recall','false_positive_rate','C','D']):intervals.append(dict(left=a,right=b,metric=name,difference=point[j],conditional_low=lo[j],conditional_high=hi[j],replicates=1000,unit='existing group, class-stratified'))
    pd.DataFrame(intervals).to_csv(DEST/'direct_conditional_intervals.csv',index=False)
    for p,hsh in read(DEST/'delta_lock.json')['inputs'].items():assert sha(p)==hsh
    dump(DEST/'completion.json',dict(status='COMPLETED_BOUNDED_SAVED_PREDICTION_DELTA',seconds=time.perf_counter()-start,new_selection_records=15,new_candidate_selection_rows=len(candrows),outer_samples=40000,mode_cases=len(decomp),new_training=0,new_model_inference=0,raw_pcap_rereads=0,all_predeclared_results_retained=True,script_sha256=sha(__file__)))
    print(pd.DataFrame(aggregate).to_string(index=False));print(pd.DataFrame(decomp)[['condition','arbiter','fold_id','best_rank_tuple_candidates','selected_pattern_outer_patterns','within_class_error_space','between_class_error_space']].to_string(index=False))
if __name__=='__main__':
    class Tee:
        def __init__(self,a,b):self.a,self.b=a,b
        def write(self,s):self.a.write(s);self.b.write(s);self.b.flush()
        def flush(self):
            self.a.flush()
            if not self.b.closed:self.b.flush()
    with (OUT/'logs/delta_analysis.log').open('x',encoding='utf-8') as log:
        original_stdout=sys.stdout
        try:sys.stdout=Tee(original_stdout,log);main()
        finally:sys.stdout=original_stdout
