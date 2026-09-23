"""Deterministic parity repair: both-primary hard OOD returns before TLS.

No fitting, hyperparameter selection, or base-view inference. Preserve original
derived decision artifacts, recalculate V3 decisions from immutable evidence.
"""
from pathlib import Path
import shutil, json
import numpy as np
import pandas as pd
from authoritative_fusion_benchmark import OUT,SEEDS,CONDITIONS,decision,metric,sha,dump,csvwrite

def repair(out=OUT):
    out=Path(out);events=[]
    for seed in SEEDS:
        for fold in range(20):
            dest=out/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}'
            if (dest/'runtime_parity_repair.json').exists():continue
            ff=out/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'
            with np.load(dest/'predictions.npz') as z:preds={k:z[k] for k in z.files}
            y=preds['y'];frame=pd.read_csv(dest/'metrics.csv');training=json.loads((dest/'training.json').read_text())
            with np.load(ff/'validation.npz') as z:val={k:z[k] for k in ('p','app','mass','reliability','ood','ood_raw')}
            beforehash=sha(dest/'predictions.npz');changed=0;clean={}
            for method in ('V3_Full','V3_OnDemand'):
                vd=decision(method,val,{},None)
                threshold=float(np.quantile(vd['score'][vd['available']],.2)) if vd['available'].any() else float('inf')
                training['thresholds'][method]=threshold
                for condition in CONDITIONS:
                    with np.load(ff/f'test_{condition}.npz') as z:ev={k:z[k] for k in ('p','app','mass','reliability','ood','ood_raw')}
                    d=decision(method,ev,{},None);accept=(d['score']>=threshold)&d['available'];prefix=f'{condition}__{method}__'
                    changed+=int(np.sum(preds[prefix+'pred']!=d['pred']))
                    for k,v in d.items():preds[prefix+k]=v.astype(np.float32) if v.dtype.kind=='f' else v
                    preds[prefix+'accepted_fixed']=accept
                    if condition=='clean':clean[method]={**d,'accepted_fixed':accept}
                    orig=clean[method];harm=(orig['accepted_fixed']&accept&(orig['pred']==y)&(d['pred']!=y)) if condition!='clean' else np.zeros(len(y),bool)
                    preds[prefix+'harmful']=harm
                    mf,ac=metric(y,d['pred']);native=d['verdict']<2;sf,sa=metric(y[native],d['pred'][native]);af,aa=metric(y[accept],d['pred'][accept])
                    updates=dict(macro_f1=mf,accuracy=ac,available_rate=float(d['available'].mean()),native_coverage=float(native.mean()),native_selective_macro_f1=sf,native_selective_error=1-sa,unknown_rate=float((d['verdict']==3).mean()),suspicious_rate=float((d['verdict']==2).mean()),fixed_coverage=float(accept.mean()),fixed_selective_error=1-aa,harmful_flips=int(harm.sum()),calls_per_sample=float(d['calls'].mean()))
                    mask=(frame.method==method)&(frame.condition==condition)
                    for k,v in updates.items():frame.loc[mask,k]=v
            archive=dest/'before_runtime_parity_repair';archive.mkdir()
            for name in ('predictions.npz','metrics.csv','training.json','DONE.json'):shutil.copy2(dest/name,archive/name)
            np.savez_compressed(dest/'predictions.npz',**preds);frame.to_csv(dest/'metrics.csv',index=False);dump(dest/'training.json',training)
            done=json.loads((dest/'DONE.json').read_text());done['predictions_sha256']=sha(dest/'predictions.npz');done['runtime_parity_repair']=True;dump(dest/'DONE.json',done)
            r=dict(seed=seed,fold=fold,changed_forced_predictions=changed,before_sha256=beforehash,after_sha256=done['predictions_sha256'],reason='Match existing FusionAgent early hard-primary OOD return before TLS',frozen_evidence_unchanged=True,no_model_refit=True)
            dump(dest/'runtime_parity_repair.json',r);events.append(r)
    csvwrite(out/'runtime_parity_repairs.csv',events)
    print(json.dumps(dict(repaired_folds=len(events),changed_forced_predictions=sum(x['changed_forced_predictions'] for x in events))))
if __name__=='__main__':repair()
