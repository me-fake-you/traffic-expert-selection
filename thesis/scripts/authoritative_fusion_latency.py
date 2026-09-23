"""Serial, warmed validation feature-to-decision latency with real lazy calls.

Never trains or inspects test labels. Verifies replay parity on validation data.
"""
import gzip,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from authoritative_fusion_benchmark import OUT,ROOT,SEEDS,METHODS,CONDITIONS,torch_module,raw_x,evidence,decision,yager,condition_arrays,dump,csvwrite,TLSProtocolAgent,DetectorInput

def load_networks(dest):
    torch=torch_module();meta=json.loads((dest/'training.json').read_text());state=torch.load(dest/'networks.pt',map_location='cpu',weights_only=True);nets={}
    for kind in ('mlp','gate','edl'):
        dims=meta['neural'][kind]['dims'];layers=[]
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(torch.nn.Linear(a,b))
            if i<len(dims)-2:layers.append(torch.nn.ReLU())
        net=torch.nn.Sequential(*layers);net.load_state_dict(state[kind]);net.eval();nets[kind]=net
    return nets

def main(out=OUT):
    out=Path(out)
    with np.load(out/'features.npz') as z:data={k:z[k] for k in z.files}
    tlsfile=out/'tls_visible_inputs.json'
    if not tlsfile.exists():
        wanted=set(data['sample_id'][data['app'][:,2]].astype(str));lookup={}
        for path in sorted((ROOT/'data/processed/ustc_tfc2016/v1/flows').rglob('*.jsonl.gz')):
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line)
                    if r['sample_id'] in wanted:lookup[r['sample_id']]={k:v for k,v in r.get('tls',{}).items() if k in ('version','cipher_suite','certificate_valid','handshake_complete')}
        assert len(lookup)==len(wanted);dump(tlsfile,lookup)
    tlslookup=json.loads(tlsfile.read_text());tlsagent=TLSProtocolAgent(contract_native_fields=True);rows=[];checks=0
    for seed in SEEDS:
        for fold in range(20):
            dest=out/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}';art=joblib.load(dest/'models.joblib');models={'lr':art['lr'],**load_networks(dest)}
            with np.load(out/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz') as z:va=z['validation']
            local=np.linspace(0,len(va)-1,128,dtype=int);idx=va[local];n=len(idx)
            with np.load(out/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'/'validation.npz') as z:cached={k:z[k][local] for k in ('p','app','mass','reliability','ood','ood_raw')}
            xs=data['stats'][idx];xt=data['temporal'][idx];tf=data['tls_features'][idx];app=data['app'][idx]
            raw=raw_x(xs,xt,tf,app);edlx=art['edl_preprocess'].transform(raw).astype(np.float32)
            def infer(method):
                prob=np.full((n,3),.5);shift=np.zeros((n,3));rawscores=shift.copy();used=np.zeros((n,3),bool);tm=np.tile([0.,0.,1.],(n,1));calls=np.zeros(n)
                def acquire(j,mask):
                    active=mask&app[:,j];used[:,j]|=active;calls[active]+=1
                    if j<2 and active.any():
                        x=(xs,xt)[j][active];prob[active,j]=art['views'][j].predict_proba(x)[:,1]
                        if method in ('V3_Full','V3_OnDemand'):
                            gate,cal=art['ood_gates'][j];rs=-gate.score_samples(x);rawscores[active,j]=rs;shift[active,j]=np.searchsorted(cal,rs,side='right')/len(cal)
                    elif j==2:
                        for pos in np.flatnonzero(active):
                            ev=tlsagent.analyze(DetectorInput(tls=tlslookup[str(data['sample_id'][idx[pos]])]));tm[pos]=[ev.benign_support,ev.malicious_support,ev.uncertainty]
                            prob[pos,2]=tm[pos,1]/max(tm[pos,0]+tm[pos,1],1e-15)
                if method=='V6_EDL':
                    ev=evidence(prob,app,tm,art['reliability']);x=art['edl_preprocess'].transform(raw).astype(np.float32)
                    return decision(method,ev,models,x),calls
                if method.startswith('V0_'):acquire(['Stats','Temporal','TLS'].index(method[3:]),np.ones(n,bool))
                elif method=='V3_OnDemand':
                    pending=np.ones(n,bool)
                    for j in range(3):
                        acquire(j,pending);ev=evidence(prob,used,tm,art['reliability'],shift,rawscores);p,pred,score,verdict=yager(ev,True);pending=verdict>=2
                    return dict(p=p,pred=pred,score=score,verdict=verdict,calls=calls),calls
                else:
                    for j in range(3):acquire(j,np.ones(n,bool))
                ev=evidence(prob,used,tm,art['reliability'],shift,rawscores)
                return decision(method,ev,models,None),calls
            for method in METHODS:
                actual,calls=infer(method);reference=decision(method,cached,models,edlx)
                np.testing.assert_array_equal(actual['pred'],reference['pred']);np.testing.assert_array_equal(actual['verdict'],reference['verdict']);checks+=1
                timings=[]
                for repeat in range(3):
                    t=time.perf_counter();infer(method);timings.append((time.perf_counter()-t)*1000/n)
                rows.append(dict(seed=seed,fold=fold,method=method,batch_size=n,repeat_1_ms_per_sample=timings[0],repeat_2_ms_per_sample=timings[1],repeat_3_ms_per_sample=timings[2],median_ms_per_sample=float(np.median(timings)),validation_calls_per_sample=float(calls.mean()),validation_decision_matches_frozen=True))
            if fold%5==4:print('latency',seed,fold,flush=True)
    csvwrite(out/'runtime_latency.csv',rows)
    timing=pd.DataFrame(rows).groupby('method').median_ms_per_sample.agg(['mean','std']).rename(columns={'mean':'feature_to_decision_latency_ms_mean','std':'feature_to_decision_latency_ms_std'})
    result=pd.read_csv(out/'benchmark_results.csv').drop(columns=list(timing.columns),errors='ignore').merge(timing,left_on='method',right_index=True,how='left');result.to_csv(out/'benchmark_results.csv',index=False)
    report=['# Serial feature-to-decision runtime measurement','','Measured after training with one worker, all 200 seed/fold models, 128 deterministic equally spaced validation rows per fold, one warmup and three repeats. Timings include actual HGB inference, applicable TLS rule calls, applicable OOD scoring for V3, preprocessing required by EDL, and the decision rule. V3_OnDemand actually skips downstream model calls after a native decision. Each measured method was verified against its frozen validation decisions. Feature extraction from packets, disk/network access and model loading are outside the timed interval. No test labels or test scores are used.','','| Method | ms/sample mean±std over 200 cells |','|---|---:|']
    for method,r in timing.iterrows():report.append(f"| {method} | {r.iloc[0]:.6f}±{r.iloc[1]:.6f} |")
    report+=['','Raw repeats and observed validation call counts: `runtime_latency.csv`. Test-cohort calls remain the separate frozen-replay counts in `benchmark_results.csv`. Differences in acquisition policy can change decisions, so call savings are not reported as free accuracy-preserving optimization.']
    (out/'RUNTIME_LATENCY.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    dump(out/'runtime_latency_validation.json',dict(parity_checks=checks,all_passed=True,training=False,test_labels_used=False,workers=1,batch_size=128,repetitions=3,cells=len(rows)))
    from authoritative_fusion_report import write_report
    write_report(out,result.to_dict('records'),json.loads((out/'paired_fusion_vs_average.json').read_text()))

if __name__=='__main__':main()
