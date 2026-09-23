"""One fixed diagnostic; no training, inference, sampling or grid expansion."""
from pathlib import Path
import argparse, hashlib, itertools, json, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
PREV=OUT.parent/'mad_etd_icassp2027_upgrade_022'
OLD=PREV/'experiment/run_001'
BASE=OUT.parent/'mad_etd_icassp2027_upgrade_013/analysis'
CFG=OUT/'config/constraint_ablation.json'
RUN=OUT/'experiment/run_001'

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,d):
    assert p.resolve().is_relative_to(OUT.resolve())
    with p.open('x',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(p,rs):
    assert not p.exists()
    pd.DataFrame(rs).to_csv(p,index=False)
def load(f,k,role):
    with np.load(BASE/f'row100_20260920_{k}_fold{f}_{role}.npz',allow_pickle=False) as z:
        return {n:z[n] for n in z.files}
def counts(y,p,first):
    tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum()); tp=int(((y==1)&(p==1)).sum())
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,
                macro_f1=tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1),
                malicious_recall=tp/max(tp+fn,1),false_positive_rate=fp/max(tn+fp,1),
                C=int(((first!=y)&(p==y)).sum()),D=int(((first==y)&(p!=y)).sum()),
                switches=int((first!=p).sum()))
def pattern(p):
    return dict(m=int(np.any(p!=p[:,:1],axis=1).sum()),K=int(np.unique(p.T,axis=0).shape[0]))
def baseline_seals():
    for n in ['DELIVERABLES.json','EXPERIMENT_FILES.json']:
        for r in read(PREV/'validation'/n): assert sha(PREV/r['path'])==r['sha256'],r['path']

def select():
    assert not (RUN/'input_lock.json').exists()
    cfg=read(CFG); baseline_seals()
    inputs=[CFG,Path(__file__),OLD/'selection_choices.csv',OLD/'selection_gate.json']
    inputs += sorted(OLD.glob('subset_fold*.npz'))
    inputs += [BASE/f'row100_20260920_{k}_fold{f}_selection.npz' for f in cfg['folds'] for k in cfg['arbiters']]
    put(RUN/'input_lock.json',dict(created=now(),inputs={str(p):sha(p) for p in inputs},outer_opened=False))
    old_choices=pd.read_csv(OLD/'selection_choices.csv').set_index(['fold_id','seed','arm','arbiter'])
    chosen=[]; candidates=[]; diagnostics=[]
    for f,k in itertools.product(cfg['folds'],cfg['arbiters']):
        d=load(f,k,'selection'); full=d['predictions']; first=(d['first_probability']>=.5).astype(np.uint8)
        assert np.array_equal(d['thresholds'],np.array(list(itertools.product(cfg['thresholds'],repeat=2))))
        assert np.array_equal(full[:,15],first)
        for seed,arm in itertools.product(cfg['seeds'],cfg['coverage_arms']):
            path=OLD/f'subset_fold{f}_{seed}_{arm}.npz'
            with np.load(path,allow_pickle=False) as s:
                idx=s['indices']; assert np.array_equal(s['sample_hash'],d['sample_hash'][idx])
            y=d['label'][idx]; pp=full[idx]; first_s=first[idx]
            assert len(idx)==len(np.unique(idx))==2000 and set(np.bincount(y))=={1000}
            key=dict(fold_id=f,arbiter=k,seed=seed,arm=arm)
            rs=[dict(candidate=j,**counts(y,pp[:,j],first_s)) for j in range(16)]
            recall=counts(y,first_s,first_s)['malicious_recall']
            feasible=[j for j,r in enumerate(rs) if r['malicious_recall']>=recall-1e-12]
            assert 15 in feasible
            score=lambda j:(rs[j]['macro_f1'],rs[j]['malicious_recall'],-rs[j]['switches'])
            a=max(feasible,key=score); b=max(range(16),key=score)
            old=old_choices.loc[(f,seed,arm,k)]
            assert a==int(old.candidate)
            for r in rs: candidates.append(dict(**key,**r,feasible=r['candidate'] in feasible))
            for variant,j in [('original_recall_constraint',a),('constraint_removed',b)]:
                chosen.append(dict(**key,variant=variant,**rs[j],theta0=d['thresholds'][j,0],
                                   theta1=d['thresholds'][j,1],pool_vector_sha256=hashlib.sha256(full[:,j].tobytes()).hexdigest()))
            s0=pattern(pp[y==0]); s1=pattern(pp[y==1])
            diagnostics.append(dict(**key,m_benign=s0['m'],K_benign=s0['K'],m_malicious=s1['m'],K_malicious=s1['K'],
                FP_min=min(r['fp'] for r in rs),FP_max=max(r['fp'] for r in rs),
                TP_min=min(r['tp'] for r in rs),TP_max=max(r['tp'] for r in rs),
                feasible_candidates=len(feasible),half_feasible=5 in feasible,
                selected_original=a,selected_ablated=b,choice_changed=a!=b,
                all_max_F1_candidates_feasible=all(j in feasible for j,r in enumerate(rs)
                    if r['macro_f1']==max(t['macro_f1'] for t in rs)),
                selected_TP_is_max=rs[a]['tp']==max(r['tp'] for r in rs)))
    assert len(chosen)==180 and len(candidates)==1440 and len(diagnostics)==90
    table(RUN/'choices.csv',chosen); table(RUN/'candidate_counts.csv',candidates); table(RUN/'selection_diagnostics.csv',diagnostics)
    sealed=[RUN/'choices.csv',RUN/'candidate_counts.csv',RUN/'selection_diagnostics.csv',RUN/'input_lock.json']
    put(RUN/'selection_gate.json',dict(created=now(),choices=180,paired_cases=90,outer_opened=False,
                                     sealed={str(p):sha(p) for p in sealed}))
    print(pd.DataFrame(diagnostics).groupby('arbiter')[['choice_changed','m_benign','FP_min','FP_max']].agg(['min','max','sum']).to_string())

def evaluate():
    assert not (RUN/'completion.json').exists()
    cfg=read(CFG); baseline_seals()
    for p,h in read(RUN/'input_lock.json')['inputs'].items(): assert sha(p)==h,p
    for p,h in read(RUN/'selection_gate.json')['sealed'].items(): assert sha(p)==h,p
    choice=pd.read_csv(RUN/'choices.csv')
    assert len(choice)==180
    inputs=[BASE/f'row100_20260920_{k}_fold{f}_evaluation.npz' for f in cfg['folds'] for k in cfg['arbiters']]
    put(RUN/'evaluation_start.json',dict(started=now(),gate_sha256=sha(RUN/'selection_gate.json'),
                                       inputs={str(p):sha(p) for p in inputs},previously_exposed=True))
    records=[]; contrasts=[]; all_ids=set()
    for f,k in itertools.product(cfg['folds'],cfg['arbiters']):
        d=load(f,k,'evaluation'); y=d['label']; first=(d['first_probability']>=.5).astype(np.uint8)
        with np.load(OLD/f'outer_policies_fold{f}.npz',allow_pickle=False) as old:
            assert np.array_equal(old['sample_hash'],d['sample_hash']) and np.array_equal(old['label'],y)
        assert len(y)==len(set(d['sample_hash']))==8000
        if k=='logistic':
            assert not all_ids.intersection(d['sample_hash']); all_ids.update(d['sample_hash'])
        columns=[]; predictions=[]
        for seed,arm in itertools.product(cfg['seeds'],cfg['coverage_arms']):
            key=dict(fold_id=f,arbiter=k,seed=seed,arm=arm); vals={}; ps={}
            for variant in cfg['variants']:
                r=choice[(choice.fold_id==f)&(choice.arbiter==k)&(choice.seed==seed)&(choice.arm==arm)&(choice.variant==variant)].iloc[0]
                j=int(r.candidate); p=d['predictions'][:,j]; vals[variant]=counts(y,p,first); ps[variant]=p
                records.append(dict(**key,variant=variant,candidate=j,n=len(y),**vals[variant]))
                columns.append(f'{seed}|{arm}|{variant}'); predictions.append(p)
            a=vals[cfg['variants'][0]]; b=vals[cfg['variants'][1]]
            contrasts.append(dict(**key,prediction_changes=int((ps[cfg['variants'][0]]!=ps[cfg['variants'][1]]).sum()),
                **{'delta_'+metric:b[metric]-a[metric] for metric in ['macro_f1','malicious_recall','false_positive_rate','fp','fn','C','D','switches']}))
        np.savez_compressed(RUN/f'outer_fold{f}_{k}.npz',sample_hash=d['sample_hash'],label=y,
                            columns=np.array(columns),predictions=np.column_stack(predictions))
    assert len(all_ids)==40000 and len(records)==180 and len(contrasts)==90
    table(RUN/'outer_metrics.csv',records); table(RUN/'constraint_contrasts.csv',contrasts)
    df=pd.DataFrame(records); pooled=[]
    for (k,seed,arm,variant),g in df.groupby(['arbiter','seed','arm','variant']):
        r={c:int(g[c].sum()) for c in ['n','tn','fp','fn','tp','C','D','switches']}
        tn,fp,fn,tp=[r[c] for c in ['tn','fp','fn','tp']]
        r.update(macro_f1=tp/(2*tp+fp+fn)+tn/(2*tn+fp+fn),malicious_recall=tp/(tp+fn),false_positive_rate=fp/(tn+fp))
        pooled.append(dict(arbiter=k,seed=int(seed),arm=arm,variant=variant,**r))
    table(RUN/'pooled_metrics.csv',pooled)
    dg=pd.read_csv(RUN/'selection_diagnostics.csv')
    summary=dict(finished=now(),status='COMPLETE_FINITE_CONSTRAINT_ABLATION',paired_cases=90,
        changed_choices=int(dg.choice_changed.sum()),changed_outer_vectors=sum(r['prediction_changes']>0 for r in contrasts),
        total_outer_prediction_changes=sum(r['prediction_changes'] for r in contrasts),
        m_benign_zero_cases=int((dg.m_benign==0).sum()),constant_FP_cases=int((dg.FP_min==dg.FP_max).sum()),
        half_infeasible_cases=int((~dg.half_feasible).sum()),filtered_cases=int((dg.feasible_candidates<16).sum()),
        all_max_F1_feasible_cases=int(dg.all_max_F1_candidates_feasible.sum()),max_TP_selected_cases=int(dg.selected_TP_is_max.sum()),
        unique_outer_samples=len(all_ids),new_training=0,new_model_inference=0,new_sampling=0,
        interpretation='conditional diagnostic, not new data or independent experiments; zero and nonzero all retained',
        script_sha256=sha(__file__),config_sha256=sha(CFG))
    put(RUN/'completion.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    stage=argparse.ArgumentParser(); stage.add_argument('stage',choices=['select','evaluate']); a=stage.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    log=RUN/f'{a.stage}.log'; assert not log.exists()
    class Tee:
        def __init__(self,f): self.f=f; self.console=sys.stdout
        def write(self,s): self.console.write(s); self.f.write(s); self.f.flush()
        def flush(self): self.console.flush(); self.f.flush()
    with log.open('x',encoding='utf-8') as f:
        old=sys.stdout; sys.stdout=Tee(f)
        try:
            print(json.dumps(dict(started=now(),command=sys.argv,stage=a.stage),ensure_ascii=False))
            (select if a.stage=='select' else evaluate)()
        except Exception:
            import traceback; traceback.print_exc(file=sys.stdout); raise
        finally: sys.stdout=old
if __name__=='__main__': main()
