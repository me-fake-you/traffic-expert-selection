"""Frozen-protocol E1: train-side pilot, all policy locks, then outer scoring."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from prepare_p1 import OUT, ROOT, CFG, sha, dump, load_inputs, development_frame

def meta(a,b,ra,rb):
    return np.column_stack((a,b,abs(a-.5),abs(b-.5),np.broadcast_to(ra,a.shape),np.broadcast_to(rb,a.shape),a-b,abs(a-b),a*b,a>=.5,b>=.5))

def metric(y,p):
    tn,fp,fn,tp = [int(v.sum()) for v in ((y==0)&(p==0),(y==0)&(p==1),(y==1)&(p==0),(y==1)&(p==1))]
    f1 = tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1)
    return {'macro_f1':float(f1),'malicious_recall':tp/max(tp+fn,1),'false_positive_rate':fp/max(fp+tn,1),'tn':tn,'fp':fp,'fn':fn,'tp':tp}

def changes(y,p,first):
    return {'corrected_FN':int(((first==0)&(y==1)&(p==1)).sum()),'corrected_FP':int(((first==1)&(y==0)&(p==0)).sum()),'introduced_FN':int(((first==1)&(y==1)&(p==0)).sum()),'introduced_FP':int(((first==0)&(y==0)&(p==1)).sum())}

def learned(a,b,q,margin,lo=.5,hi=.5):
    trigger = abs(a-.5)<margin
    first,second = a>=.5,b>=.5
    switch = trigger & (first!=second) & (q>=np.where(first,hi,lo))
    return np.where(switch,second,first).astype(int)

def select_threshold(a,b,y,q,cfg):
    recall = metric(y,(a>=.5).astype(int))['malicious_recall']
    candidates = []
    for lo,hi in itertools.product(cfg['secondary_threshold_grid'],repeat=2):
        p = learned(a,b,q,cfg['trigger_margin'],lo,hi)
        m = metric(y,p)
        candidates.append({'low':lo,'high':hi,'feasible':m['malicious_recall']>=recall-1e-12,'switches':int((p!=(a>=.5)).sum()),**m})
    eligible = [v for v in candidates if v['feasible']]
    best = max(eligible,key=lambda v:(v['macro_f1'],v['malicious_recall'],-v['switches']))
    return best,candidates

def new_model(kind,cfg):
    if kind=='hgb':
        return HistGradientBoostingClassifier(**cfg['arbiter_hgb'])
    return make_pipeline(StandardScaler(),LogisticRegression(**cfg['arbiter_logistic']))

def locked_inputs():
    lock = json.loads((OUT/'development/input_lock.json').read_text(encoding='utf-8'))
    for p,h in lock.items():
        assert sha(p)==h, f'Frozen input changed: {p}'
    return lock

def pilot(cfg,data,hist,members):
    dest = OUT/'e1/pilot_001'
    dest.mkdir(parents=True,exist_ok=False)
    frame,_ = development_frame(data,hist,0)
    selected = members[(members.outer_fold==0)&(members.condition=='row25')&(members.subset_seed==cfg['subset_seeds'][0])]
    part = frame.loc[selected.oof_position.to_numpy()]
    X = meta(part.temporal.to_numpy(),part.stats.to_numpy(),part.reliability_temporal.to_numpy(),part.reliability_stats.to_numpy())
    times = {}
    for kind in ('logistic','hgb'):
        t = time.perf_counter()
        model = new_model(kind,cfg).fit(X,part.target.to_numpy())
        assert np.isfinite(model.predict_proba(X[:16])).all()
        times[kind] = time.perf_counter()-t
        joblib.dump(model,dest/f'{kind}.joblib')
    # Scale with measured full-vs-pilot row ratios, deliberately conservative for small HGB fits.
    support = pd.read_csv(OUT/'development/condition_support.csv')
    projected = float((support.rows/len(part)).clip(lower=1).sum()*sum(times.values())*2)
    result = {'command':sys.argv,'rows':len(part),'fit_seconds':times,'projected_fit_seconds_conservative':projected,'outer_evaluation_performed':False,'all_conditions':len(support),'planned_new_fits':2*len(support),'within_frozen_local_budget':projected<=cfg['pilot']['maximum_projected_cpu_seconds']}
    dump(dest/'pilot.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def run(cfg,data,hist,members,lock):
    pilot_result = json.loads((OUT/'e1/pilot_001/pilot.json').read_text(encoding='utf-8'))
    assert pilot_result['within_frozen_local_budget'], 'Pilot exceeded predeclared local budget'
    dest = OUT/'e1/run_001'
    dest.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    dump(dest/'run_manifest.json',{'command':sys.argv,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'config_sha256':sha(CFG),'script_sha256':sha(__file__),'input_lock_sha256':sha(OUT/'development/input_lock.json'),'scope':cfg['exposure']})
    splits = json.loads((hist/'split_manifest.json').read_text(encoding='utf-8'))
    views = {'stats':data['x_stats'],'temporal':data['x_hybrid'][:,8:]}
    records,searches,reference_locks,fit_times = [],[],{},[]
    # Complete every fit and development-only selection before evaluating any outer row.
    for split in splits:
        f = split['outer_fold']; folder = dest/f'fold_{f}'; folder.mkdir(); (folder/'models').mkdir()
        frame,_ = development_frame(data,hist,f)
        historical = json.loads((hist/f'seed_42_fold_{f}/temporal_stats_locked_policy.json').read_text(encoding='utf-8'))
        ra,rb = historical['reliability']['temporal'],historical['reliability']['stats']
        va = np.flatnonzero(np.isin(data['group'],split['groups']['selection']))
        a,b = [joblib.load(hist/f'seed_42_fold_{f}/models/full_{v}.joblib').predict_proba(views[v][va])[:,1] for v in cfg['order']]
        yy = data['y'][va]; vm = meta(a,b,ra,rb)
        weight = max(cfg['weighted_average_grid'],key=lambda w:(metric(yy,(w*a+(1-w)*b>=.5).astype(int))['macro_f1'],-abs(w-.5)))
        single = 0 if metric(yy,(a>=.5).astype(int))['macro_f1']>=metric(yy,(b>=.5).astype(int))['macro_f1'] else 1
        reference_locks[str(f)] = {'reliability_temporal':ra,'reliability_stats':rb,'weighted_first':weight,'single':single}
        fold_members = members[members.outer_fold==f]
        for (condition,seed), subset in fold_members.groupby(['condition','subset_seed'],sort=True):
            part = frame.loc[subset.oof_position.to_numpy()]
            assert np.array_equal(part.sample_hash.to_numpy(),subset.sample_hash.to_numpy())
            X = meta(part.temporal.to_numpy(),part.stats.to_numpy(),part.reliability_temporal.to_numpy(),part.reliability_stats.to_numpy())
            for kind in ('logistic','hgb'):
                t = time.perf_counter(); model = new_model(kind,cfg).fit(X,part.target.to_numpy()); elapsed=time.perf_counter()-t
                model_id = f'{condition}_{seed}_{kind}'
                joblib.dump(model,folder/f'models/{model_id}.joblib')
                chosen,candidates = select_threshold(a,b,yy,model.predict_proba(vm)[:,1],cfg)
                rec = {'outer_fold':f,'condition':condition,'subset_seed':int(seed),'kind':kind,'model_id':model_id,'rows':len(part),'selected_low':chosen['low'],'selected_high':chosen['high']}
                records.append(rec); fit_times.append({**rec,'fit_seconds':elapsed})
                searches.extend({**rec,**c} for c in candidates)
        print(f'E1 fold {f}: {len([r for r in records if r["outer_fold"]==f])} arbiters fitted and development policies locked; no outer scoring yet',flush=True)
    pd.DataFrame(records).to_csv(dest/'policy_lock.csv',index=False)
    pd.DataFrame(searches).to_csv(dest/'selection_candidates.csv',index=False)
    pd.DataFrame(fit_times).to_csv(dest/'fit_times.csv',index=False)
    dump(dest/'reference_lock.json',reference_locks)
    dump(dest/'evaluation_gate.json',{'all_arbiter_fits_and_policies_saved':True,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'policy_sha256':sha(dest/'policy_lock.csv'),'reference_sha256':sha(dest/'reference_lock.json'),'config_sha256':sha(CFG)})
    metrics,group_metrics,index = [],[],{}
    for split in splits:
        f = split['outer_fold']; folder=dest/f'fold_{f}'
        ev = np.flatnonzero(np.isin(data['group'],split['groups']['evaluation']))
        a,b = [joblib.load(hist/f'seed_42_fold_{f}/models/full_{v}.joblib').predict_proba(views[v][ev])[:,1] for v in cfg['order']]
        yy=data['y'][ev]; first=(a>=.5).astype(int); second=(b>=.5).astype(int); trigger=abs(a-.5)<cfg['trigger_margin']
        r=reference_locks[str(f)];ra,rb=r['reliability_temporal'],r['reliability_stats'];mx=meta(a,b,ra,rb)
        fixed=np.where(trigger&(abs(b-.5)*rb>.25*abs(a-.5)*ra),second,first)
        predictions={'first_only':first,'second_only':second,'static_equal':((a+b)/2>=.5).astype(int),'validation_weighted':(r['weighted_first']*a+(1-r['weighted_first'])*b>=.5).astype(int),'validation_selected_single':first if r['single']==0 else second,'same_trigger_fixed':fixed}
        meta_index={k:{'condition':'reference','kind':k,'subset_seed':-1,'operating_point':'reference'} for k in predictions}
        for record in [r for r in records if r['outer_fold']==f]:
            mid=record['model_id']; model=joblib.load(folder/f'models/{mid}.joblib');q=model.predict_proba(mx)[:,1]
            for mode,lo,hi in [('primary',.5,.5),('selected',record['selected_low'],record['selected_high'])]:
                col=mid+'_'+mode; predictions[col]=learned(a,b,q,cfg['trigger_margin'],lo,hi)
                meta_index[col]={k:record[k] for k in ('condition','kind','subset_seed')};meta_index[col]['operating_point']=mode
        output=pd.DataFrame({'sample_hash':data['sample_hash'][ev],'group':data['group'][ev],'label':yy,'first_probability':a,'second_probability':b,'trigger':trigger.astype(int),**predictions})
        output.to_csv(folder/'predictions.csv.gz',index=False,compression='gzip')
        for name,p in predictions.items():
            calls=1+float(trigger.mean()) if name=='same_trigger_fixed' or name not in ('first_only','second_only','static_equal','validation_weighted','validation_selected_single') else (2. if name in ('static_equal','validation_weighted') else 1.)
            header={'outer_fold':f,'prediction_column':name,**meta_index[name]}
            metrics.append({**header,'rows':len(ev),**metric(yy,p),**changes(yy,p,first),'evidence_calls':calls,'trigger_rate':float(trigger.mean()) if calls not in (1.,2.) else (calls-1.),'switch_rate':float((p!=first).mean())})
            for group in sorted(set(data['group'][ev])):
                mask=data['group'][ev]==group
                group_metrics.append({**header,'group':group,'label':int(yy[mask][0]),'rows':int(mask.sum()),**metric(yy[mask],p[mask]),**changes(yy[mask],p[mask],first[mask]),'switches':int((p[mask]!=first[mask]).sum())})
        index.update(meta_index)
        print(f'E1 fold {f}: outer predictions saved for every prespecified condition',flush=True)
    pd.DataFrame(metrics).to_csv(dest/'fold_metrics.csv',index=False)
    pd.DataFrame(group_metrics).to_csv(dest/'group_metrics.csv',index=False)
    dump(dest/'prediction_index.json',index)
    for p,h in lock.items(): assert sha(p)==h, f'input changed during E1: {p}'
    summary={'status':'COMPLETED_CONDITIONAL_E1','new_arbiter_fits':len(records),'pilot_fits':2,'new_base_fits':0,'outer_rows_per_condition':40000,'subset_repeats_are_not_independent_test_sets':True,'fresh_blind_test':False,'all_predeclared_conditions_kept':True,'historical_inputs_unchanged':True,'duration_seconds':time.perf_counter()-start}
    dump(dest/'completion.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','run'],required=True);args=p.parse_args()
    class Tee:
        def __init__(self,stream,file): self.stream,self.file=stream,file
        def write(self,text): self.stream.write(text);self.file.write(text);self.file.flush()
        def flush(self): self.stream.flush();self.file.flush()
    if args.mode=='run':
        log=(OUT/'logs/e1_run_001.stdout.log').open('x',encoding='utf-8')
        sys.stdout=Tee(sys.stdout,log)
    lock=locked_inputs();cfg,data,pv,hist=load_inputs()
    assert json.loads((OUT/'development/load_check.json').read_text(encoding='utf-8'))['status']=='PASS_FOR_CONDITIONAL_USTC_E1_ONLY'
    members=pd.read_csv(OUT/'development/subset_manifest.csv')
    with threadpool_limits(limits=1):
        if args.mode=='pilot': pilot(cfg,data,hist,members)
        else: run(cfg,data,hist,members,lock)

if __name__=='__main__':
    try: main()
    except Exception:
        failure=traceback.format_exc()
        (OUT/'logs'/f'e1_failure_{time.time_ns()}.log').write_text(failure,encoding='utf-8')
        raise
