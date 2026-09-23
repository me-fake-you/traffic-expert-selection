"""Line-indexed native rejection replay, using production FusionAgent operations."""
import gzip,json,time,inspect
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from natural_shift_audit import ROOT,BASE,OUT,SEEDS,dump,sha,csv
from mad_etd.schemas import AgentEvidence,FeatureGroup,ReliabilityProfile
from mad_etd.fusion import FusionAgent,Mass
from mad_etd.ood import apply_ood_policy_to_evidence

NAMES=['StatsDetectorAgent','TemporalBehaviorAgent','TLSProtocolAgent']
FG=[FeatureGroup.STATS,FeatureGroup.SEQUENCE,FeatureGroup.TLS]

def one_trace(seed,study):
    dest=OUT/'rejection_traces'/study;dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'seed_{seed}.jsonl.gz';done=dest/f'seed_{seed}.json'
    if done.exists():return json.loads(done.read_text())
    base=BASE if study=='locked' else OUT/'temporal_refit'
    counts={m:0 for m in ('V2_Yager','V3_Full','V6_EDL')};native={};line=0;parity=0;lines=[]
    with np.load(OUT/'condition_indices.npz') as z:conditions={c:set(z[c].tolist()) for c in z.files}
    with gzip.open(path,'wt',encoding='utf8',compresslevel=3) as log:
        for fold in range(20):
            dp=base/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}';frozen=base/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'/'test_clean.npz'
            with np.load(frozen) as z:ev={k:z[k] for k in z.files}
            with np.load(dp/('predictions.npz' if study=='locked' else 'natural_predictions.npz')) as z:
                pred={m:{k:z[f'clean__{m}__{k}'] for k in ('verdict','score','p')} for m in counts}
                assert np.array_equal(z['index'],ev['index'])
            frozen_hash=sha(frozen)
            for i,index in enumerate(ev['index']):
                memberships=[c for c,v in conditions.items() if int(index) in v] if study=='locked' else ['temporal_early_late_refit']
                for method in counts:
                    verdict=int(pred[method]['verdict'][i])
                    if verdict<2:continue
                    counts[method]+=1;line+=1
                    record=dict(study=study,seed=seed,fold=fold,index=int(index),method=method,verdict=['benign','malicious','suspicious','unknown'][verdict],conditions=memberships,frozen_file=str(frozen.relative_to(ROOT)),frozen_row=i,frozen_sha256=frozen_hash,trace_line=line)
                    if method=='V6_EDL':
                        uncertainty=1-float(pred[method]['score'][i]);assert uncertainty>.5
                        record.update(trigger='Dirichlet total uncertainty > 0.5',uncertainty=uncertainty,threshold=.5,view_evidence_consumed=False,feature_row=int(index),feature_archive=str((base/'features.npz').relative_to(ROOT)),model=str((dp/'networks.pt').relative_to(ROOT)),source_rule='scripts/authoritative_fusion_benchmark.py: decision V6_EDL',traceable=True)
                    else:
                        use_ood=method=='V3_Full';items=[];viewrecords=[];hard=(ev['ood'][i,:2]>=.99)&ev['app'][i,:2]
                        rel=ev['reliability'][i]
                        reliability=ReliabilityProfile(stats_reliability=rel[0],sequence_reliability=rel[1],tls_reliability=rel[2],payload_reliability=0,input_completeness=1,ood_suspected=bool(use_ood and hard.any()))
                        for j in range(3):
                            v=dict(view=NAMES[j],applicable=bool(ev['app'][i,j]),p_malicious=float(ev['p'][i,j]),original_mass=ev['mass'][i,j].tolist(),validation_reliability=float(rel[j]),ood_percentile=float(ev['ood'][i,j]),ood_raw=float(ev['ood_raw'][i,j]))
                            if ev['app'][i,j]:
                                mass=ev['mass'][i,j];level=('hard' if ev['ood'][i,j]>=.99 else 'warning' if ev['ood'][i,j]>=.95 else 'in_domain') if j<2 else 'off'
                                item=AgentEvidence(agent_name=NAMES[j],feature_group=FG[j],benign_support=mass[0],malicious_support=mass[1],uncertainty=mass[2],confidence=max(mass[:2]),calibration_quality=1,distribution_shift_score=ev['ood'][i,j],distribution_shift_level=level)
                                if use_ood:item=apply_ood_policy_to_evidence(item,'hybrid')
                                items.append(item);v.update(post_policy_mass=[item.benign_support,item.malicious_support,item.uncertainty],model_reliability=float(item.model_reliability),abstained=bool(item.abstained))
                            viewrecords.append(v)
                        fusion=FusionAgent();result=fusion.fuse(items,reliability,final=True)
                        actual={'benign':0,'malicious':1,'suspicious':2,'unknown':3}[result.verdict.value];assert actual==verdict,(study,seed,fold,index,method,actual,verdict)
                        stages=[];combined=Mass(0,0,1);conflict=0.
                        if not(use_ood and hard.all()):
                            for item in items:
                                if item.abstained or not item.contributes_to_verdict:continue
                                discounted=fusion._discount(item,reliability);combined,c=fusion._combine_yager(combined,discounted);conflict=1-(1-conflict)*(1-c)
                                stages.append(dict(view=item.agent_name,discounted_mass=[discounted.benign,discounted.malicious,discounted.unknown],combined_mass=[combined.benign,combined.malicious,combined.unknown],local_conflict=c,cumulative_conflict=conflict))
                        if use_ood and hard.all():branch='both_primary_hard_ood_early_unknown'
                        elif reliability.ood_suspected or result.uncertainty>=.5:branch='ood_or_ignorance_gate'
                        elif result.conflict_score>=.3 and result.malicious_support>=.2:branch='conflict_gate'
                        elif result.malicious_support>=.45:branch='malicious_support_0.45_suspicious'
                        elif result.malicious_support>=.25:branch='final_malicious_support_0.25_suspicious'
                        else:branch='insufficient_benign_or_malicious_support_unknown'
                        record.update(views=viewrecords,yager_stages=stages,trigger_branch=branch,hard_ood_views=[NAMES[j] for j in range(2) if hard[j]] if use_ood else [],evidence_contributors=[s['view'] for s in stages],result=result.model_dump(mode='json'),runtime_verdict_matches=True,traceable=True,source_rule='src/mad_etd/fusion.py: FusionAgent.fuse/_discount/_combine_yager/_decide; src/mad_etd/ood.py: apply_ood_policy_to_evidence')
                    log.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');parity+=1
                    lines.append(dict(study=study,seed=seed,fold=fold,index=int(index),method=method,verdict=record['verdict'],trace_file=str(path.relative_to(OUT)),line=line))
    csv(dest/f'seed_{seed}_index.csv',lines)
    info=dict(study=study,seed=seed,native_rejects=counts,trace_lines=line,traceable=parity,sha256=sha(path));dump(done,info);return info

def main():
    failures=[];rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(one_trace,s,study):(s,study) for study in ('locked','temporal') for s in SEEDS}
        for f in as_completed(jobs):
            try:r=f.result();rows.append(r);print('trace',r['study'],r['seed'],r['trace_lines'],flush=True)
            except Exception as e:
                import traceback
                failures.append(dict(job=jobs[f],error=repr(e),traceback=traceback.format_exc()))
    if failures:dump(OUT/'trace_failures.json',failures);raise RuntimeError('Trace replay failures logged')
    dump(OUT/'trace_summary.json',dict(status='passed',seconds=time.perf_counter()-start,runs=rows,total_native_rejections=sum(r['trace_lines'] for r in rows),traceable=sum(r['traceable'] for r in rows),source_sha256={p:sha(ROOT/p) for p in ('src/mad_etd/fusion.py','src/mad_etd/ood.py','scripts/authoritative_fusion_benchmark.py')}))
if __name__=='__main__':main()
