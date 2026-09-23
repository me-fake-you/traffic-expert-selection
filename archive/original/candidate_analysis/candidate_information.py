"""New post-hoc inference of the frozen original 16 candidates, never refit."""
from common_new import *
import itertools

def support(matrix,indices):
    selected=matrix[:,indices]
    changing=np.any(selected!=selected[:,0,None],axis=1)
    k=np.unique(selected.T,axis=0).shape[0]
    return changing,int(changing.sum()),int(k)

def main():
    dest=OUT/'analysis';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'completion.json').exists()
    started=time.perf_counter();cfg,data,pv,hist=load_inputs();run=P1/'e1/run_001'
    plans=read(hist/'split_manifest.json');refs=read(run/'reference_lock.json')
    locks=pd.read_csv(run/'policy_lock.csv');old_grid=pd.read_csv(run/'selection_candidates.csv')
    grid=list(itertools.product(cfg['secondary_threshold_grid'],repeat=2))
    # Second condition is chosen by design, before inspecting new candidate outcomes.
    conditions=['row100','source_concentrated'];seed=20260920
    lock={'created':stamp(),'post_hoc':True,'command':sys.argv,'grid':grid,'conditions':conditions,'subset_seed':seed,
          'population':'all 4000 selection and 8000 evaluation rows per outer fold, both arbiters',
          'cross_condition_scope':'one existing source-concentrated fit condition; same exposed USTC benchmark, not external validation',
          'hypothesis':'Total role size and triggered disagreements need not equal candidate-distinguishing support; dev-equivalent candidate vectors may split on evaluation.',
          'judgment':'Report all folds, models, groups and mixed/zero cases; no predictive or causal rule claim. Preserve unique-best Logistic fold-1 counterexample.',
          'stop':'Mismatch with original scores, selected predictions, models or grid stops this run. No grid/model expansion.',
          'inputs':{str(p):sha(p) for p in [P1/'config/e1_protocol.json',run/'policy_lock.csv',run/'selection_candidates.csv',hist/'split_manifest.json',Path(__file__)]}}
    dump(dest/'analysis_lock.json',lock)
    summaries=[];candidates=[];groups=[];transfers=[];hashes={};checks=[]
    views={'temporal':data['x_hybrid'][:,8:],'stats':data['x_stats']}
    with threadpool_limits(limits=1):
        for split in plans:
            f=split['outer_fold'];r=refs[str(f)];ra,rb=r['reliability_temporal'],r['reliability_stats']
            role_data={}
            for role in ['selection','evaluation']:
                ix=np.flatnonzero(np.isin(data['group'],split['groups'][role]));y=data['y'][ix]
                probs=[]
                for view in ['temporal','stats']:
                    path=hist/f'seed_42_fold_{f}/models/full_{view}.joblib';hashes[str(path)]=sha(path)
                    probs.append(joblib.load(path).predict_proba(views[view][ix])[:,1])
                a,b=probs;first=(a>=.5).astype(int);second=(b>=.5).astype(int);tr=abs(a-.5)<cfg['trigger_margin']
                fixed=np.where(tr&(abs(b-.5)*rb>.25*abs(a-.5)*ra),second,first)
                role_data[role]=(ix,y,a,b,first,second,tr,fixed)
            old=pd.read_csv(run/f'fold_{f}/predictions.csv.gz',float_precision='round_trip').set_index('sample_hash')
            for condition in conditions:
                for kind in ['logistic','hgb']:
                    item=locks[(locks.outer_fold==f)&(locks.condition==condition)&(locks.subset_seed==seed)&(locks.kind==kind)].iloc[0]
                    mid=item.model_id;path=run/f'fold_{f}/models/{mid}.joblib';hashes[str(path)]=sha(path);model=joblib.load(path)
                    matrices={};scores={};meta_rows={}
                    selected_id=grid.index((item.selected_low,item.selected_high))
                    for role,vals in role_data.items():
                        ix,y,a,b,first,second,tr,fixed=vals;q=model.predict_proba(meta(a,b,ra,rb))[:,1]
                        matrix=np.column_stack([learned(a,b,q,cfg['trigger_margin'],lo,hi) for lo,hi in grid]).astype(np.uint8)
                        matrices[role]=matrix
                        score=[{**metric(y,matrix[:,j]),'switches':int((matrix[:,j]!=first).sum())} for j in range(16)]
                        scores[role]=score
                        if role=='selection':
                            recall=metric(y,first)['malicious_recall'];feasible=[j for j,s in enumerate(score) if s['malicious_recall']>=recall-1e-12]
                            keys=[(s['macro_f1'],s['malicious_recall'],-s['switches']) for s in score]
                            best=max(feasible,key=lambda j:keys[j]);optimal=[j for j in feasible if keys[j]==keys[best]]
                            assert best==selected_id,(f,kind,condition,best,selected_id)
                            prior=old_grid[(old_grid.outer_fold==f)&(old_grid.model_id==mid)]
                            for j,(lo,hi) in enumerate(grid):
                                row=prior[(prior.low==lo)&(prior.high==hi)].iloc[0]
                                assert abs(row.macro_f1-score[j]['macro_f1'])<1e-12
                                assert bool(row.feasible)==(j in feasible)
                        else:
                            ids=data['sample_hash'][ix];assert np.array_equal(matrix[:,selected_id],old.loc[ids,mid+'_selected'])
                            assert np.array_equal(matrix[:,grid.index((.5,.5))],old.loc[ids,mid+'_primary'])
                        np.savez_compressed(dest/f'{mid}_fold{f}_{role}.npz',sample_hash=data['sample_hash'][ix],group=data['group'][ix],
                            label=y,first_probability=a,second_probability=b,q=q,trigger=tr,thresholds=np.array(grid),predictions=matrix)
                        base={'fold_id':f,'condition':condition,'arbiter':kind,'role':role,'selected_candidate':selected_id}
                        for category,inds in [('all',list(range(16))),('dev_feasible',feasible)]:
                            diff,m,k=support(matrix,inds)
                            summaries.append({**base,'candidate_set':category,'candidates':len(inds),'n':len(y),'triggered':int(tr.sum()),'triggered_disagreements':int((tr&(first!=second)).sum()),'m':m,'K':k,
                                'm_benign':int((diff&(y==0)).sum()),'m_malicious':int((diff&(y==1)).sum()),
                                'm_first_benign':int((diff&(first==0)).sum()),'m_first_malicious':int((diff&(first==1)).sum()),
                                'best_dev_patterns':support(matrices['selection'],optimal)[2], 'best_dev_candidates':len(optimal)})
                        fm=metric(y,fixed)
                        for j,(lo,hi) in enumerate(grid):
                            p=matrix[:,j]
                            candidates.append({**base,'candidate':j,'low':lo,'high':hi,'dev_feasible':j in feasible,'dev_optimal':j in optimal,'actually_selected':j==selected_id,
                                **score[j],'delta_F1_vs_fixed_pp':100*(score[j]['macro_f1']-fm['macro_f1']),
                                'delta_FP_vs_fixed':score[j]['fp']-fm['fp'],'delta_FN_vs_fixed':score[j]['fn']-fm['fn'],
                                'corrections_vs_fixed':int(((fixed!=y)&(p==y)).sum()),'damage_vs_fixed':int(((fixed==y)&(p!=y)).sum())})
                    for category,inds in [('all',list(range(16))),('dev_feasible',feasible)]:
                        by_vector={}
                        for j in inds:
                            fingerprint=hashlib.sha256(matrices['selection'][:,j].tobytes()).hexdigest()
                            by_vector.setdefault(fingerprint,[]).append(j)
                        for number,(fingerprint,members) in enumerate(by_vector.items()):
                            ix,y,a,b,first,second,tr,fixed=role_data['evaluation'];diff,m,k=support(matrices['evaluation'],members)
                            record={'fold_id':f,'condition':condition,'arbiter':kind,'candidate_set':category,'behavior_group':number,
                                    'selection_vector_sha256':fingerprint,'members':';'.join(map(str,members)),'member_count':len(members),
                                    'contains_selected':selected_id in members,'contains_dev_optimal':bool(set(members)&set(optimal)),
                                    'outer_K':k,'outer_m':m,'outer_m_benign':int((diff&(y==0)).sum()),'outer_m_malicious':int((diff&(y==1)).sum()),
                                    'outer_FP_min':min(scores['evaluation'][j]['fp'] for j in members),'outer_FP_max':max(scores['evaluation'][j]['fp'] for j in members),
                                    'outer_FN_min':min(scores['evaluation'][j]['fn'] for j in members),'outer_FN_max':max(scores['evaluation'][j]['fn'] for j in members)}
                            groups.append(record)
                    primary=grid.index((.5,.5));evy=role_data['evaluation'][1];pp=matrices['evaluation'][:,primary];sp=matrices['evaluation'][:,selected_id]
                    transfers.append({'fold_id':f,'condition':condition,'arbiter':kind,'selected_candidate':selected_id,
                        'dev_delta_pp':100*(scores['selection'][selected_id]['macro_f1']-scores['selection'][primary]['macro_f1']),
                        'outer_delta_pp':100*(scores['evaluation'][selected_id]['macro_f1']-scores['evaluation'][primary]['macro_f1']),
                        'corrected':int(((pp!=evy)&(sp==evy)).sum()),'damaged':int(((pp==evy)&(sp!=evy)).sum()),
                        'best_dev_patterns':support(matrices['selection'],optimal)[2]})
            print(f'fold {f}: all candidates replayed for two arbiters and two predeclared fit conditions',flush=True)
    for name,rows in [('support',summaries),('candidates',candidates),('behavior_groups',groups),('selection_transfer',transfers)]:table(dest/f'{name}.csv',rows)
    assert all(sha(p)==v for p,v in hashes.items());dump(dest/'model_hashes.json',hashes)
    dump(dest/'completion.json',{'status':'COMPLETED_POST_HOC_CANDIDATE_REPLAY','finished':stamp(),'seconds':time.perf_counter()-started,
        'new_fits':0,'original_models_reused':len(hashes),'candidate_role_records':len(candidates),'new_inference':True,
        'all_original_scores_and_selected_predictions_matched':True,'second_condition':'within-benchmark fit-source concentration; not independent replication'})
    print(pd.DataFrame(summaries).query("condition=='row100' and role=='selection' and candidate_set=='all'")[['fold_id','arbiter','n','triggered_disagreements','m','K','best_dev_patterns']].to_string(index=False))
if __name__=='__main__':main()
