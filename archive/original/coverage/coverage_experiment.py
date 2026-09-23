"""Finite-population selection coverage sensitivity; never train or infer models.

Stages are separate processes. Preparation/selection open selection arrays only;
outer arrays cannot be decoded until every planned selection has been sealed.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
PREV = OUT.parent / 'mad_etd_icassp2027_upgrade_021'
BASE = OUT.parent / 'mad_etd_icassp2027_upgrade_013'
P1 = OUT.parent / 'mad_etd_icassp2027_p1_20260920'
R1 = OUT.parent / 'mad_etd_icassp2027_v55_r1'
CONFIG = OUT / 'config/coverage_protocol.json'
PROVENANCE = R1 / 'runs/raw_audit_001/selected_provenance.csv'
SPLITS = R1 / 'runs/nested_controls_001/split_manifest.json'
POLICIES = P1 / 'e1/run_001/policy_lock.csv'
KINDS = ('logistic', 'hgb')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    assert Path(path).resolve().is_relative_to(OUT.resolve())
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def table(path, records):
    assert not path.exists(), path
    pd.DataFrame(records).to_csv(path, index=False)


def matrix_path(fold, kind, role):
    return BASE / f'analysis/row100_20260920_{kind}_fold{fold}_{role}.npz'


def load_matrix(fold, kind, role, run):
    if role == 'evaluation':
        gate = read(run / 'selection_gate.json')
        assert gate['all_planned_choices_saved'] and gate['outer_scoring_started'] is False
        for p, digest in gate['sealed_files'].items():
            assert sha(p) == digest, p
    else:
        assert role == 'selection'
    with np.load(matrix_path(fold, kind, role), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def counts(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return np.array([np.sum((y == 0) & (p == 0)), np.sum((y == 0) & (p == 1)),
                     np.sum((y == 1) & (p == 0)), np.sum((y == 1) & (p == 1))], dtype=int)


def metric_from_counts(c):
    tn, fp, fn, tp = map(int, c)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp,
                macro_f1=tp / max(2 * tp + fp + fn, 1) + tn / max(2 * tn + fp + fn, 1),
                malicious_recall=tp / max(tp + fn, 1), false_positive_rate=fp / max(tn + fp, 1))


def metric(y, p):
    return metric_from_counts(counts(y, p))


def score(y, p, first):
    C = int(np.sum((first != y) & (p == y)))
    D = int(np.sum((first == y) & (p != y)))
    switches = int(np.sum(first != p))
    assert switches == C + D
    return dict(n=len(y), **metric(y, p), C=C, D=D, switches=switches)


def choose(y, mat, first):
    records = [dict(candidate=j, **score(y, mat[:, j], first)) for j in range(16)]
    minimum = metric(y, first)['malicious_recall']
    feasible = [r['candidate'] for r in records if r['malicious_recall'] >= minimum - 1e-12]
    assert feasible and 15 in feasible
    selected = max(feasible, key=lambda j: (records[j]['macro_f1'], records[j]['malicious_recall'], -records[j]['switches']))
    return selected, feasible, records


def support(mat, candidates=None):
    mm = mat if candidates is None else mat[:, candidates]
    varying = np.any(mm != mm[:, :1], axis=1)
    return int(varying.sum()), int(np.unique(mm.T, axis=0).shape[0])


def verify_pair(ds):
    a, b = ds['logistic'], ds['hgb']
    for key in ['sample_hash', 'group', 'label', 'first_probability', 'second_probability', 'trigger', 'thresholds']:
        assert np.array_equal(a[key], b[key]), key
    for d in ds.values():
        assert len(d['sample_hash']) == len(set(d['sample_hash']))
        assert np.array_equal(d['thresholds'], np.array(list(itertools.product([.25, .5, .75, 1.01], repeat=2))))
        assert np.isin(d['predictions'], [0, 1]).all()
        first = (d['first_probability'] >= .5).astype(np.uint8)
        second = (d['second_probability'] >= .5).astype(np.uint8)
        for j, (lo, hi) in enumerate(d['thresholds']):
            use = d['trigger'] & (d['q'] >= np.where(first == 0, lo, hi))
            assert np.array_equal(np.where(use, second, first), d['predictions'][:, j])
        assert np.array_equal(d['predictions'][:, 15], first)


def provenance():
    cols = ['sample_hash', 'partition', 'group', 'label', 'capture', 'capture_sha256']
    p = pd.read_csv(PROVENANCE, usecols=cols)
    p = p[p.partition == 'development40k'].copy()
    assert len(p) == p.sample_hash.nunique() == 40000
    return p.set_index('sample_hash')


def prepare(run):
    t0 = time.perf_counter(); cfg = read(CONFIG)
    inputs = [CONFIG, Path(__file__), PROVENANCE, SPLITS, POLICIES,
              P1 / 'development/load_check.json', PREV / 'validation/DELIVERABLES.json']
    inputs += [matrix_path(f, k, 'selection') for f in cfg['folds'] for k in KINDS]
    locked = {str(p): sha(p) for p in inputs}
    dump(run / 'input_lock.json', dict(created=stamp(), inputs=locked, outer_arrays_opened=False,
                                     exposed_historical_evaluation=True, command=sys.argv))
    old_manifest = read(PREV / 'validation/DELIVERABLES.json')
    assert all(sha(PREV / r['path']) == r['sha256'] for r in old_manifest)
    prov = provenance(); splits = read(SPLITS); policies = pd.read_csv(POLICIES)
    quotas, info, source_rows = [], [], []
    for f in cfg['folds']:
        ds = {k: load_matrix(f, k, 'selection', run) for k in KINDS}; verify_pair(ds)
        d = ds['logistic']; pp = prov.loc[d['sample_hash']]
        assert np.array_equal(pp.group.to_numpy(), d['group'])
        assert np.array_equal(pp.label.to_numpy(), d['label'])
        split = next(s for s in splits if s['outer_fold'] == f)
        assert set(d['group']) == set(split['groups']['selection'])
        for role in ['train', 'calibration', 'evaluation']:
            assert set(d['group']).isdisjoint(split['groups'][role])
        assert set(pp.capture) == set(split['captures']['selection'])
        masks = {k: np.any(z['predictions'] != z['predictions'][:, :1], axis=1) for k, z in ds.items()}
        union = masks['logistic'] | masks['hgb']
        pp = pp.copy(); pp['union_informative'] = union
        for group, g in pp.groupby('group', sort=True):
            assert len(g) == 2000 and g.label.nunique() == 1
            target = len(g) // 2
            cap = g.groupby('capture_sha256', sort=True).size()
            allocation = {str(c): int(n * target // len(g)) for c, n in cap.items()}
            remainders = sorted(cap.index, key=lambda c: (-(int(cap[c]) * target % len(g)), str(c)))
            for c in remainders[:target - sum(allocation.values())]: allocation[str(c)] += 1
            for c, gg in g.groupby('capture_sha256', sort=True):
                n = len(gg); q = allocation[str(c)]; I = int(gg.union_informative.sum())
                quotas.append(dict(fold_id=f, group=group, label=int(gg.label.iloc[0]), capture_sha256=c,
                                   capture=gg.capture.iloc[0], available=n, quota=q, informative=I,
                                   min_m=max(0, q - (n - I)), max_m=min(q, I)))
        for k, z in ds.items():
            first = z['first_probability'] >= .5
            j, F, scores = choose(z['label'], z['predictions'], first)
            old = policies[(policies.outer_fold == f) & (policies.condition == 'row100') &
                           (policies.subset_seed == 20260920) & (policies.kind == k)]
            assert len(old) == 1
            assert np.array_equal(z['thresholds'][j], old[['selected_low', 'selected_high']].iloc[0].to_numpy())
            m, K = support(z['predictions'])
            info.append(dict(fold_id=f, arbiter=k, original_n=len(first), original_m=m, original_K=K,
                             union_m=int(union.sum()), m_benign=int(np.sum(masks[k] & (z['label'] == 0))),
                             m_malicious=int(np.sum(masks[k] & (z['label'] == 1))), full_pool_choice=j,
                             invariant_rows_equal_Temporal=bool(np.all(z['predictions'][~masks[k]] == first[~masks[k], None]))))
        # Hashes, classes and capture identifiers only; no packet contents/endpoints.
        source_rows += [dict(fold_id=f, sample_hash=str(h), group=str(d['group'][i]), label=int(d['label'][i]),
                            capture_sha256=str(pp.capture_sha256.iloc[i]), union_informative=bool(union[i]))
                        for i, h in enumerate(d['sample_hash'])]
    table(run / 'quotas.csv', quotas); table(run / 'feasibility.csv', info)
    pd.DataFrame(source_rows).to_csv(run / 'selection_source_manifest.csv.gz', index=False, compression='gzip')
    assert all(sha(p) == h for p, h in locked.items())
    dump(run / 'prepare_complete.json', dict(completed=stamp(), seconds=time.perf_counter()-t0,
         status='FEASIBLE_CONDITIONAL_CONTRAST_WITH_STRUCTURAL_ZERO_FOLDS', original_sealed_artifacts_unchanged=len(old_manifest),
         structural_zero_folds=[f for f in cfg['folds'] if not any(r['fold_id'] == f and r['union_m'] for r in info)],
         informative_benign_rows=sum(r['m_benign'] for r in info), outer_arrays_opened=False,
         new_training=0, new_inference=0, input_lock_sha256=sha(run / 'input_lock.json')))
    print(pd.DataFrame(info).to_string(index=False), flush=True)


def rank_rows(indices, hashes, fold, seed):
    return sorted(indices, key=lambda i: hashlib.sha256(f'coverage022|{seed}|{fold}|{hashes[i]}'.encode()).digest())


def verify_inputs(run):
    for p, digest in read(run / 'input_lock.json')['inputs'].items():
        assert sha(p) == digest, p


def select(run):
    t0 = time.perf_counter(); assert (run / 'prepare_complete.json').is_file(); verify_inputs(run)
    cfg = read(CONFIG); quotas = pd.read_csv(run / 'quotas.csv'); prov = provenance()
    all_choices, all_candidates, overlaps, manifests = [], [], [], []
    for f in cfg['folds']:
        ds = {k: load_matrix(f, k, 'selection', run) for k in KINDS}; verify_pair(ds)
        d = ds['logistic']; hashes = d['sample_hash']; pp = prov.loc[hashes]
        union = np.logical_or.reduce([np.any(z['predictions'] != z['predictions'][:, :1], axis=1) for z in ds.values()])
        for seed in cfg['seeds']:
            picks = {a: [] for a in cfg['arms']}
            for r in quotas[quotas.fold_id == f].itertuples():
                ix = np.flatnonzero((d['group'] == r.group) & (pp.capture_sha256.to_numpy() == r.capture_sha256) & (d['label'] == r.label))
                ordered = rank_rows(ix, hashes, f, seed)
                I = [i for i in ordered if union[i]]; U = [i for i in ordered if not union[i]]
                picks['low'] += (U + I)[:r.quota]
                picks['high'] += (I + U)[:r.quota]
                picks['random'] += ordered[:r.quota]
            for a in picks: picks[a] = np.array(sorted(picks[a]), dtype=int)
            for a, b in [('high', 'low'), ('high', 'random'), ('low', 'random')]:
                common = int(len(set(picks[a]) & set(picks[b])))
                overlaps.append(dict(fold_id=f, seed=seed, left=a, right=b, n_left=len(picks[a]), n_right=len(picks[b]),
                                     common_rows=common, symmetric_difference=len(picks[a])+len(picks[b])-2*common))
            if not union.any():
                assert np.array_equal(picks['low'], picks['high']) and np.array_equal(picks['low'], picks['random'])
            for arm, ix in picks.items():
                assert len(ix) == len(set(ix)) == 2000
                for r in quotas[quotas.fold_id == f].itertuples():
                    take = (d['group'][ix] == r.group) & (pp.capture_sha256.to_numpy()[ix] == r.capture_sha256) & (d['label'][ix] == r.label)
                    assert int(take.sum()) == r.quota
                    if arm in ['low', 'high']:
                        assert int(union[ix][take].sum()) == (r.min_m if arm == 'low' else r.max_m)
                np.savez_compressed(run / f'subset_fold{f}_{seed}_{arm}.npz', indices=ix, sample_hash=hashes[ix])
                manifests += [dict(fold_id=f, seed=seed, arm=arm, sample_hash=str(hashes[i]),
                                   group=str(d['group'][i]), label=int(d['label'][i]),
                                   capture_sha256=str(pp.capture_sha256.iloc[i]), union_informative=bool(union[i])) for i in ix]
                for kind, z in ds.items():
                    y = z['label'][ix]; mat = z['predictions'][ix]; first = z['first_probability'][ix] >= .5
                    j, F, scores = choose(y, mat, first); m, K = support(mat); mf, kf = support(mat, F)
                    equal = [k for k in F if np.array_equal(mat[:, k], mat[:, j])]
                    choice = dict(fold_id=f, seed=seed, arm=arm, arbiter=kind, n=len(ix), union_m=int(union[ix].sum()),
                                  m=m, K=K, m_feasible=mf, K_feasible=kf, candidate=j,
                                  low=float(z['thresholds'][j, 0]), high=float(z['thresholds'][j, 1]),
                                  feasible=';'.join(map(str, F)), half_feasible=5 in F, equivalent_candidates=len(equal),
                                  pool_prediction_sha256=hashlib.sha256(z['predictions'][:, j].tobytes()).hexdigest(),
                                  subset_sha256=sha(run / f'subset_fold{f}_{seed}_{arm}.npz'),
                                  **{k:v for k,v in scores[j].items() if k not in ['candidate','n']},
                                  **{'baseline_'+k:v for k,v in metric(y, first).items()})
                    all_choices.append(choice)
                    all_candidates += [dict(fold_id=f, seed=seed, arm=arm, arbiter=kind,
                                            feasible=s['candidate'] in F, selected=s['candidate'] == j, **s) for s in scores]
    assert len(all_choices) == 90 and len(all_candidates) == 1440
    table(run / 'selection_choices.csv', all_choices); table(run / 'selection_candidate_scores.csv', all_candidates)
    table(run / 'paired_sample_overlap.csv', overlaps)
    pd.DataFrame(manifests).to_csv(run / 'subset_manifest.csv.gz', index=False, compression='gzip')
    verify_inputs(run)
    files = [run/'selection_choices.csv', run/'selection_candidate_scores.csv', run/'quotas.csv',
             run/'subset_manifest.csv.gz', run/'input_lock.json', CONFIG, Path(__file__)] + sorted(run.glob('subset_*.npz'))
    dump(run / 'selection_gate.json', dict(created=stamp(), all_planned_choices_saved=True, choices=90,
         rows_per_selection=2000, independent_repetitions=False, seconds=time.perf_counter()-t0,
         outer_scoring_started=False, new_model_fits=0, new_model_inference=0,
         sealed_files={str(p):sha(p) for p in files}))
    print(pd.DataFrame(all_choices)[['fold_id','seed','arm','arbiter','union_m','m','K','candidate','half_feasible']].to_string(index=False), flush=True)


def evaluate(run):
    t0 = time.perf_counter(); verify_inputs(run); cfg=read(CONFIG)
    gate = read(run / 'selection_gate.json'); assert gate['all_planned_choices_saved']
    selected = pd.read_csv(run / 'selection_choices.csv')
    outer_inputs = {str(matrix_path(f,k,'evaluation')):sha(matrix_path(f,k,'evaluation')) for f in cfg['folds'] for k in KINDS}
    dump(run / 'evaluation_start.json', dict(started=stamp(), selection_gate_sha256=sha(run/'selection_gate.json'),
         input_hashes=outer_inputs, selection_complete_before_this_scoring=True, evaluation_is_historically_exposed=True))
    rows, refs, compare, paired = [], [], [], []
    for f in cfg['folds']:
        ds={k:load_matrix(f,k,'evaluation',run) for k in KINDS}; verify_pair(ds)
        columns=[]; values=[]
        for kind,z in ds.items():
            first=z['first_probability']>=.5; y=z['label']; base=z['predictions'][:,5]
            assert len(y)==8000
            sel=load_matrix(f,kind,'selection',run)
            assert set(sel['sample_hash']).isdisjoint(z['sample_hash'])
            assert set(sel['group']).isdisjoint(z['group'])
            refs.append(dict(fold_id=f,arbiter=kind,reference='arbiter_0.5',**score(y,base,first)))
            refs.append(dict(fold_id=f,arbiter=kind,reference='Temporal',**score(y,first,first)))
            columns += [kind+'_0.5']; values += [base]
            chosen=selected[(selected.fold_id==f)&(selected.arbiter==kind)]
            for r in chosen.itertuples():
                p=z['predictions'][:,r.candidate]
                transfer=score(y,p,base)
                row=dict(fold_id=f,arbiter=kind,seed=r.seed,arm=r.arm,candidate=r.candidate,
                         **score(y,p,first), transfer_C=transfer['C'],transfer_D=transfer['D'],
                         transfer_delta_f1_pp=100*(metric(y,p)['macro_f1']-metric(y,base)['macro_f1']),
                         corrected_FN=int(np.sum((base==0)&(y==1)&(p==1))),
                         corrected_FP=int(np.sum((base==1)&(y==0)&(p==0))),
                         introduced_FN=int(np.sum((base==1)&(y==1)&(p==0))),
                         introduced_FP=int(np.sum((base==0)&(y==0)&(p==1))))
                rows.append(row); columns.append(f'{kind}_{r.seed}_{r.arm}'); values.append(p)
            for seed in cfg['seeds']:
                subset=chosen[chosen.seed==seed].set_index('arm')
                for left,right in [('high','low'),('high','random')]:
                    a=z['predictions'][:,int(subset.loc[left,'candidate'])]
                    b=z['predictions'][:,int(subset.loc[right,'candidate'])]
                    ca=score(y,a,first); cb=score(y,b,first); tr=score(y,a,b)
                    compare.append(dict(fold_id=f,arbiter=kind,seed=seed,left=left,right=right,
                         numerical_choice_changed=int(subset.loc[left,'candidate'])!=int(subset.loc[right,'candidate']),
                         full_selection_pattern_changed=subset.loc[left,'pool_prediction_sha256']!=subset.loc[right,'pool_prediction_sha256'],
                         outer_prediction_changes=int(np.sum(a!=b)),delta_f1_pp=100*(ca['macro_f1']-cb['macro_f1']),
                         delta_recall_pp=100*(ca['malicious_recall']-cb['malicious_recall']),
                         delta_fpr_pp=100*(ca['false_positive_rate']-cb['false_positive_rate']),
                         delta_C=ca['C']-cb['C'],delta_D=ca['D']-cb['D'],delta_switches=ca['switches']-cb['switches'],
                         delta_FP=ca['fp']-cb['fp'],delta_FN=ca['fn']-cb['fn'],
                         left_corrects_right=tr['C'],left_damages_right=tr['D']))
        np.savez_compressed(run/f'outer_policies_fold{f}.npz',sample_hash=z['sample_hash'],group=z['group'],
                            label=z['label'],Temporal=first,columns=np.array(columns),predictions=np.column_stack(values))
    table(run/'outer_fold_metrics.csv',rows); table(run/'outer_reference_metrics.csv',refs); table(run/'paired_contrasts.csv',compare)
    frame=pd.DataFrame(rows); pooled=[]
    for (kind,seed,arm),g in frame.groupby(['arbiter','seed','arm'],sort=True):
        assert len(g)==5 and g.n.sum()==40000
        c=g[['tn','fp','fn','tp']].sum().to_numpy()
        pooled.append(dict(arbiter=kind,seed=int(seed),arm=arm,n=40000,**metric_from_counts(c),
                           C=int(g.C.sum()),D=int(g.D.sum()),switches=int(g.switches.sum()),
                           transfer_C=int(g.transfer_C.sum()),transfer_D=int(g.transfer_D.sum())))
    table(run/'pooled_metrics.csv',pooled)
    verify_inputs(run)
    assert all(sha(p)==digest for p,digest in outer_inputs.items())
    assert all(sha(PREV/r['path'])==r['sha256'] for r in read(PREV/'validation/DELIVERABLES.json'))
    dump(run/'completion.json',dict(status='COMPLETED_FROZEN_SELECTION_COVERAGE_SENSITIVITY',finished=stamp(),
          seconds=time.perf_counter()-t0,selection_choices=90,selection_candidate_scores=1440,
          unique_outer_rows=40000,fold_metric_rows=len(rows),paired_comparison_rows=len(compare),pooled_rows=len(pooled),
          new_training=0,new_model_inference=0,new_data=0,bootstrap_or_significance_tests=0,
          all_arms_seeds_and_folds_preserved=True,script_sha256=sha(__file__),protocol_sha256=sha(CONFIG)))
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(pd.DataFrame(compare)[lambda z:z.left.eq('high') & z.right.eq('low')].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','select','evaluate'])
    parser.add_argument('--run',default='run_001'); args=parser.parse_args()
    assert args.run.startswith('run_') and args.run[4:].isdigit()
    run=OUT/'experiment'/args.run; run.mkdir(parents=True,exist_ok=True)
    class Tee:
        def __init__(self,a,b): self.a,self.b=a,b
        def write(self,s): self.a.write(s); self.b.write(s); self.b.flush()
        def flush(self): self.a.flush(); self.b.flush()
    old=sys.stdout
    with (run/f'{args.stage}.log').open('x',encoding='utf-8') as log:
        try:
            sys.stdout=Tee(old,log)
            print(json.dumps(dict(command=sys.argv,started=stamp(),stage=args.stage),ensure_ascii=False),flush=True)
            globals()[args.stage](run)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stdout)
            raise
        finally: sys.stdout=old


if __name__=='__main__': main()
