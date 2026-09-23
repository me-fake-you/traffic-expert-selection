"""Separate system-level replication; no detection-accuracy claims from synthetic probes."""
import sys,json,gzip,copy,hashlib,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from natural_shift_audit import ROOT,OUT,BASE,SEEDS,dump,csv,sha
from mad_etd.audit import AuditLogger
from mad_etd.evidence_admission import AgentEvidenceAdmissionGate
from mad_etd.layered_security_governance_w266_w270 import _formal_registry,_base_gate_evidence,run_systematic_threat_model_w268

def main():
    dest=OUT/'system_audit';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        old=json.loads((dest/'COMPLETE.json').read_text())
        if old.get('baseline_controls_logged'):return
        shutil.copytree(dest,OUT/'system_audit_before_baseline_log',dirs_exist_ok=True)
    existing=ROOT/'data/runs/mad_etd_layered_security_governance_w266_w270'
    # Required context documentation only; all threat cases are executed afresh.
    for name in ('architecture_current_state_audit.json','layered_ablation_report.json'):shutil.copy2(existing/name,dest/name)
    structured=run_systematic_threat_model_w268(dest)
    registry=_formal_registry();base,context,specialist=_base_gate_evidence(registry)
    kinds=['missing_prediction','illegal_final_verdict','ood_override','feature_policy_hash_mismatch','artifact_hash_mismatch','capability_mismatch','safety_flag','llm_source','rag_source','stale_evidence','future_evidence','unregistered_specialist','invalid_source_hash','unsupported_evidence','replay_same_ref']
    rows=[];control=[];boundaries=[];events=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);logger=AuditLogger(f'natural-audit-{seed}');gate=AgentEvidenceAdmissionGate(registry,audit_logger=logger)
        for i in range(500):
            kind=kinds[i%len(kinds)];payload=copy.deepcopy(base);ref=f'{seed}-{i}-{rng.integers(1,2**60)}';src='detector';seq=4;sid=specialist
            if kind=='missing_prediction':payload.pop('prediction')
            elif kind=='illegal_final_verdict':payload['final_verdict']='malicious'
            elif kind=='ood_override':payload['ood_override']='in_domain'
            elif kind=='feature_policy_hash_mismatch':payload['input_feature_policy_hash']=hashlib.sha256(ref.encode()).hexdigest()
            elif kind=='artifact_hash_mismatch':payload['artifact_hash']=hashlib.sha256(ref.encode()).hexdigest()
            elif kind=='capability_mismatch':payload['supported_capabilities']=['stats.source_ip']
            elif kind=='safety_flag':payload['safety_flags']=['blocked_field_access_attempt']
            elif kind=='llm_source':src='llm'
            elif kind=='rag_source':src='rag'
            elif kind=='stale_evidence':seq=0
            elif kind=='future_evidence':seq=5
            elif kind=='unregistered_specialist':sid='unregistered'
            elif kind=='invalid_source_hash':payload['source_evidence_sha256']='not-a-sha'
            elif kind=='unsupported_evidence':payload.update(applicability='unsupported',unsupported_reason='missing',contributes_to_verdict=False)
            elif kind=='replay_same_ref':
                setup=gate.admit(payload,specialist_id=sid,evidence_ref=ref,generated_sequence=seq,source_kind=src,context=context)
                control.append(dict(seed=seed,kind='replay_setup',admitted=setup.admitted));assert setup.admitted
            before=len(logger.events);decision=gate.admit(payload,specialist_id=sid,evidence_ref=ref,generated_sequence=seq,source_kind=src,context=context)
            rows.append(dict(seed=seed,case=i,case_type=kind,admitted=decision.admitted,rejected=not decision.admitted,reasons='|'.join(decision.reason_codes),audit_sequence=decision.audit_sequence_no,exactly_one_event=len(logger.events)==before+1))
        for i in range(50):
            d=gate.admit(base,specialist_id=specialist,evidence_ref=f'control-{seed}-{i}',generated_sequence=4,source_kind='detector',context=context)
            control.append(dict(seed=seed,kind='fresh_valid_control',admitted=d.admitted))
        for kind in ('semantic_payload_forgery','wellformed_forged_source_hash','replay_new_ref','cross_case_rebinding'):
            for i in range(10):
                p=copy.deepcopy(base);ctx=context
                if kind=='semantic_payload_forgery':p.update(prediction='malicious',probabilities={'benign':.01,'malicious':.99},confidence=.99,uncertainty=.01)
                if kind=='wellformed_forged_source_hash':p['source_evidence_sha256']=hashlib.sha256(f'fabricated-{seed}-{i}'.encode()).hexdigest()
                if kind=='cross_case_rebinding':ctx=context.model_copy(update={'trace_id':f'another-case-{seed}-{i}','case_state_hash':hashlib.sha256(f'another-{seed}-{i}'.encode()).hexdigest()})
                before=len(logger.events)
                d=gate.admit(p,specialist_id=specialist,evidence_ref=f'boundary-{seed}-{kind}-{i}',generated_sequence=4,source_kind='detector',context=ctx)
                boundaries.append(dict(seed=seed,kind=kind,case=i,admitted=d.admitted,reasons='|'.join(d.reason_codes),audit_sequence=d.audit_sequence_no,exactly_one_event=len(logger.events)==before+1,semantic_truth_verified=False))
        events.extend([e.model_dump(mode='json') for e in logger.events])
        print('audit gate seed',seed,flush=True)
    csv(dest/'mutation_cases.csv',rows);csv(dest/'valid_controls.csv',control);csv(dest/'semantic_boundary_probes.csv',boundaries)
    with gzip.open(dest/'gate_events.jsonl.gz','wt',encoding='utf8') as f:
        for r in events:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Real-source field mutation: stable safe inputs/features and independent system-rule outputs.
    from mad_etd.field_audit import FieldAuditAgent
    from mad_etd.schemas import FlowRecord
    from mad_etd.features import extract_stats_features,extract_temporal_features
    from mad_etd.engine import build_default_engine
    from mad_etd.security_acceptance import _snapshot,_fusion_ownership_violation,REQUIRED_AUDIT_EVENTS
    auditor=FieldAuditAgent();engine=build_default_engine(detector_backend='rule',use_nvidia=False,cache_dir=None,enrichment_policy='off',max_workers=1)
    with gzip.open(OUT/'audit_flow_examples.jsonl.gz','rt',encoding='utf8') as f:examples=[json.loads(s) for s in f]
    field_rows=[];unknowns=[]
    with gzip.open(dest/'field_mutation_events.jsonl.gz','wt',encoding='utf8') as log, gzip.open(dest/'baseline_control_events.jsonl.gz','wt',encoding='utf8') as baseline_log:
        for ei,item in enumerate(examples):
            original=FlowRecord.model_validate(item['record']);safe=auditor.make_detector_input(original,auditor.audit(original));orig_report,orig_audit=engine.analyze(original)
            baseline_events=[e.model_dump(mode='json') for e in orig_audit.events]
            baseline_log.write(json.dumps(dict(index=item['index'],events=baseline_events),ensure_ascii=False)+'\n')
            if orig_report.verdict.value in ('unknown','suspicious'):
                unknowns.append(dict(index=item['index'],seed='baseline_control',verdict=orig_report.verdict.value,event_file='baseline_control_events.jsonl.gz',line=ei+1,fusion_event_sequence=[e['sequence_no'] for e in baseline_events if e['event_type']=='FINAL_FUSION'],traceable=REQUIRED_AUDIT_EVENTS<={e['event_type'] for e in baseline_events}))
            for seed in SEEDS:
                mut=original.model_copy(deep=True);mut.trace_id=f'mut-{seed}-{item["index"]}';mut.sample_id=f'changed-{seed}-{item["index"]}'
                mut.labels={'binary':'malicious' if original.labels.get('binary')=='benign' else 'benign','family':'forged-family','application':'forged-application'}
                mut.context.update(src_ip='203.0.113.1',dst_ip='198.51.100.2',src_port=str(seed),dst_port='65535',timestamp=seed*1000,stream_id=f'forged-{seed}')
                mut.provenance={'source_file':f'forged-{seed}.pcap','capture_start_epoch':seed,'segment_index':seed};mut.stats['label_proxy']=seed;mut.tls['external_identity']=f'forged-{seed}'
                safe2=auditor.make_detector_input(mut,auditor.audit(mut));report,audit=engine.analyze(mut)
                features_equal=np.array_equal(extract_stats_features(safe),extract_stats_features(safe2),equal_nan=True) and np.array_equal(extract_temporal_features(safe),extract_temporal_features(safe2),equal_nan=True)
                events2=[e.model_dump(mode='json') for e in audit.events];types={e['event_type'] for e in events2};blocked=[e for e in events2 if e['event_type']=='AGENT_EVIDENCE' and e['input_summary'].get('blocked_field_intersection')]
                row=dict(index=item['index'],seed=seed,safe_input_identical=safe.model_dump()==safe2.model_dump(),features_11_30_identical=features_equal,fusion_snapshot_identical=_snapshot(orig_report)==_snapshot(report),blocked_field_admissions=len(blocked),fusion_ownership_violation=_fusion_ownership_violation(report,audit),required_events_complete=REQUIRED_AUDIT_EVENTS<=types,verdict=report.verdict.value)
                field_rows.append(row)
                log.write(json.dumps(dict(index=item['index'],seed=seed,events=events2),ensure_ascii=False)+'\n')
                if report.verdict.value in ('unknown','suspicious'):
                    unknowns.append(dict(index=item['index'],seed=seed,verdict=report.verdict.value,event_file='field_mutation_events.jsonl.gz',line=len(field_rows),fusion_event_sequence=[e['sequence_no'] for e in events2 if e['event_type']=='FINAL_FUSION'],traceable=REQUIRED_AUDIT_EVENTS<=types))
            if (ei+1)%40==0:print('field audit flows',ei+1,flush=True)
    csv(dest/'field_mutation_cases.csv',field_rows);csv(dest/'system_unknown_trace_index.csv',unknowns)
    summary=[]
    for kind,g in pd.DataFrame(rows).groupby('case_type'):summary.append(dict(study='synthetic_structural_gate',case_type=kind,denominator=len(g),admitted=int(g.admitted.sum()),rejected=int(g.rejected.sum()),failures=int(g.admitted.sum()),traceable=int(g.exactly_one_event.sum()),scope='default-off prototype gate; synthetic payloads'))
    for kind,g in pd.DataFrame(boundaries).groupby('kind'):summary.append(dict(study='semantic_boundary_probe',case_type=kind,denominator=len(g),admitted=int(g.admitted.sum()),rejected=int((~g.admitted).sum()),failures=int(g.admitted.sum()),traceable=int(g.exactly_one_event.sum()),scope='negative control: authentic content / cross-case binding is not verified'))
    for kind,g in pd.DataFrame(control).groupby('kind'):summary.append(dict(study='positive_control',case_type=kind,denominator=len(g),admitted=int(g.admitted.sum()),rejected=int((~g.admitted).sum()),failures=int((~g.admitted).sum()),scope='must admit valid fresh payloads'))
    fr=pd.DataFrame(field_rows)
    for column in ('safe_input_identical','features_11_30_identical','fusion_snapshot_identical','required_events_complete'):
        summary.append(dict(study='real_source_field_mutation',case_type=column,denominator=len(fr),matched=int(fr[column].sum()),failures=int((~fr[column]).sum()),scope='200 real USTC flows x 10 seeds; separate rule-runtime audit, not HGB benchmark'))
    threats=pd.read_csv(dest/'threat_model_and_results.csv')
    summary.append(dict(study='structured_threat_replication',case_type='original_19_cases',denominator=len(threats),failures=int(threats.attack_success_rate.sum()),scope='mixed gate/policy/ownership checks; T17 is a static protocol assertion, not runtime security enforcement'))
    csv(OUT/'audit_replication.csv',summary)
    dump(dest/'COMPLETE.json',dict(structural_probes=len(rows),structural_admitted=sum(r['admitted'] for r in rows),valid_controls=len(control),boundary_probes=len(boundaries),boundary_admitted=sum(r['admitted'] for r in boundaries),field_mutation_pairs=len(field_rows),baseline_controls=len(examples),baseline_controls_logged=True,system_native_rejected_verdicts=len(unknowns),system_rejects_traceable=sum(r['traceable'] for r in unknowns),rule_backend_only=True,gate_default_enabled=False))
if __name__=='__main__':main()
