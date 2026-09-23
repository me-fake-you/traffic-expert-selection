"""Development-only qualification; never evaluates an outer prediction."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
CFG = OUT / 'config/e1_protocol.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def load_inputs():
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    with np.load(ROOT / cfg['source'], allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    pv = pd.read_csv(ROOT / cfg['provenance'])
    pv = pv[pv.partition == 'development40k'].set_index('sample_hash').loc[data['sample_hash']].reset_index()
    hist = ROOT / cfg['historical_run']
    return cfg, data, pv, hist

def development_frame(data, hist, fold):
    folder = hist / f'seed_42_fold_{fold}'
    with np.load(folder / 'oof_predictions.npz', allow_pickle=False) as z:
        frame = pd.DataFrame({k: z[k] for k in z.files})
    idx = frame['indices'].to_numpy()
    frame['sample_hash'] = data['sample_hash'][idx]
    frame['group'] = data['group'][idx]
    frame['inner'] = -1
    deps = json.loads((folder / 'oof_dependencies.json').read_text(encoding='utf-8'))
    for dep in deps:
        frame.loc[frame.group.isin(dep['predict_groups']), 'inner'] = dep['fold']
    frame['direction'] = (frame.temporal >= .5).astype(int)
    frame['target'] = ((frame.stats >= .5) == frame.y).astype(int)
    frame['disagreement'] = (frame.temporal >= .5) != (frame.stats >= .5)
    return frame, deps

def subsets(frame, cfg):
    d = frame[frame.disagreement].copy()
    outputs = []
    for seed in cfg['subset_seeds']:
        rng = np.random.default_rng(seed)
        permutations = [(k, rng.permutation(v.index)) for k, v in d.groupby(['group','direction','target'], sort=True)]
        for fraction in cfg['row_fractions']:
            if fraction == 1 and seed != cfg['subset_seeds'][0]:
                continue  # identical full-data fits are not new repeats
            ids = np.concatenate([p[:max(1, math.ceil(len(p)*fraction))] for _, p in permutations])
            outputs.append((f'row{int(fraction*100)}', seed, np.sort(ids)))
        concentrated, dispersed = [], []
        for _, cell in d.groupby(['inner','direction','target'], sort=True):
            groups = sorted(cell.group.unique(), key=lambda g: (-int((cell.group == g).sum()), g))
            queues = [list(rng.permutation(cell[cell.group == g].index)) for g in groups]
            n = min(len(queues[0]), math.ceil(len(cell)/2))
            concentrated.extend(queues[0][:n])
            spread = []
            for j in range(max(map(len, queues))):
                for queue in queues:
                    if j < len(queue) and len(spread) < n:
                        spread.append(queue[j])
                if len(spread) == n:
                    break
            dispersed.extend(spread)
        outputs.extend([('source_concentrated', seed, np.sort(concentrated)), ('source_dispersed', seed, np.sort(dispersed))])
    return outputs

def main():
    started = time.perf_counter()
    destination = OUT / 'development'
    destination.mkdir(exist_ok=True)
    assert not any(destination.iterdir()), 'Refuse to overwrite an existing P1 development result'
    cfg, data, pv, hist = load_inputs()
    old = json.loads((hist / 'run_manifest.json').read_text(encoding='utf-8'))['input_sha256']
    recorded_models = json.loads((hist / 'model_sha256.json').read_text(encoding='utf-8'))
    hashes = {str(CFG): sha(CFG), str(Path(__file__)): sha(__file__)}
    for rel in (cfg['source'], cfg['provenance'], 'output/mad_etd_icassp2027_v55_r1/experiments/run_nested_controls.py'):
        p = ROOT / rel
        expected = next(v for k,v in old.items() if Path(k).resolve() == p.resolve())
        assert sha(p) == expected, f'historical input changed: {p}'
        hashes[str(p)] = expected
    assert len(set(data['sample_hash'])) == len(data['y'])
    assert np.array_equal(pv.label.to_numpy(), data['y'])
    assert np.array_equal(pv.group.to_numpy(), data['group'])
    assert np.array_equal(data['x_stats'], data['x_hybrid'][:,:8], equal_nan=True)
    assert data['x_stats'].shape == (40000,8) and data['x_hybrid'].shape == (40000,48)
    splits = json.loads((hist / 'split_manifest.json').read_text(encoding='utf-8'))
    manifests, supports, cells, checks, memberships, conditions, overlaps = [], [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for split in splits:
            f = split['outer_fold']
            frame, deps = development_frame(data, hist, f)
            assert np.array_equal(data['y'][frame['indices']], frame.y)
            assert (frame.inner >= 0).all()
            role_sets = {}
            for role, groups in split['groups'].items():
                rows = pv[pv.group.isin(groups)].copy()
                rows['outer_fold'] = f
                rows['role'] = {'train':'base_fit_and_group_crossfit_arbiter_development','calibration':'reliability_only','selection':'secondary_policy_selection','evaluation':'previously_exposed_evaluation'}[role]
                rows['array_index'] = rows.index
                manifests.append(rows[['outer_fold','array_index','sample_hash','group','label','capture','capture_sha256','session_signature','segment_index','role']])
                role_sets[role] = {'groups':set(groups),'captures':set(rows.capture),'sessions':set(rows.session_signature)}
            names = list(role_sets)
            for a in range(len(names)):
                for b in range(a+1,len(names)):
                    for unit in ('groups','captures','sessions'):
                        n = len(role_sets[names[a]][unit] & role_sets[names[b]][unit])
                        overlaps.append({'outer_fold':f,'left':names[a],'right':names[b],'unit':unit,'overlap':n})
                        assert n == 0, f'role overlap {f} {names[a]} {names[b]} {unit}'
            for dep in deps:
                assert not set(dep['predict_groups']) & (set(dep['fit_groups']) | set(dep['calibration_groups']))
                assert set(dep['fit_groups']) | set(dep['predict_groups']) == set(split['groups']['train'])
                sub = frame[frame.inner == dep['fold']]
                # Load real feature rows, execute the actual frozen experts, compare their OOF values.
                spot = sub.groupby('group',sort=True).head(8)
                for view in ('stats','temporal'):
                    mp = hist / f'seed_42_fold_{f}/models/inner_{dep["fold"]}_{view}.joblib'
                    assert sha(mp) == recorded_models[str(mp.relative_to(hist))]
                    hashes[str(mp)] = sha(mp)
                    model = joblib.load(mp)
                    X = data['x_stats'] if view == 'stats' else data['x_hybrid'][:,8:]
                    pred = model.predict_proba(X[spot['indices']])[:,1]
                    err = float(np.max(np.abs(pred-spot[view].to_numpy())))
                    assert err <= 1e-12
                    checks.append({'outer_fold':f,'inner':dep['fold'],'view':view,'loaded_rows':len(spot),'max_probability_error':err})
            for view in ('stats','temporal'):
                mp = hist / f'seed_42_fold_{f}/models/full_{view}.joblib'
                assert sha(mp) == recorded_models[str(mp.relative_to(hist))]
                hashes[str(mp)] = sha(mp)
            for rel in ('oof_predictions.npz','oof_dependencies.json','temporal_stats_locked_policy.json'):
                p = hist / f'seed_42_fold_{f}' / rel
                hashes[str(p)] = sha(p)
            d = frame[frame.disagreement]
            for keys, part in d.groupby(['inner','direction','target'],sort=True):
                cells.append({'outer_fold':f,'inner':keys[0],'direction':keys[1],'target':keys[2],'rows':len(part),'groups':part.group.nunique()})
            support = {'outer_fold':f,'oof_rows':len(frame),'disagreements':len(d),'disagreement_groups':d.group.nunique()}
            for direction in (0,1):
                for target in (0,1):
                    cell = d[(d.direction==direction)&(d.target==target)]
                    support[f'd{direction}_t{target}_rows'] = len(cell)
                    support[f'd{direction}_t{target}_groups'] = cell.group.nunique()
            supports.append(support)
            samples = subsets(frame,cfg)
            frames = {}
            for condition,seed,ids in samples:
                part = frame.loc[ids]
                assert len(part) == len(set(ids)) and part.disagreement.all()
                for direction in (0,1):
                    assert set(part.loc[part.direction == direction,'target']) == {0,1}, f'one-class direction: {f}, {condition}, {direction}'
                frames[condition,seed] = part
                memberships.append(pd.DataFrame({'outer_fold':f,'condition':condition,'subset_seed':seed,'oof_position':ids,'sample_hash':part.sample_hash.to_numpy()}))
                row = {'outer_fold':f,'condition':condition,'subset_seed':seed,'rows':len(part),'groups':part.group.nunique()}
                for direction in (0,1):
                    for target in (0,1):
                        cell = part[(part.direction==direction)&(part.target==target)]
                        row[f'd{direction}_t{target}_rows'] = len(cell)
                        row[f'd{direction}_t{target}_groups'] = cell.group.nunique()
                conditions.append(row)
            for seed in cfg['subset_seeds']:
                a,b = (frames[k,seed] for k in ('source_concentrated','source_dispersed'))
                strata = ['inner','direction','target']
                assert a.groupby(strata).size().equals(b.groupby(strata).size())
                ga = a.groupby(['direction','target']).group.nunique()
                gb = b.groupby(['direction','target']).group.nunique()
                assert (gb >= ga).all() and (gb > ga).any(), f'no directional source-support contrast in fold {f}'
            print(f'P1 fold {f}: actual frozen-model loading passed; {len(d)} OOF disagreements; {len(samples)} eligible conditions',flush=True)
    pd.concat(manifests,ignore_index=True).to_csv(destination/'input_role_manifest.csv',index=False)
    pd.DataFrame(supports).to_csv(destination/'oof_support.csv',index=False)
    pd.DataFrame(cells).to_csv(destination/'inner_direction_support.csv',index=False)
    pd.DataFrame(conditions).to_csv(destination/'condition_support.csv',index=False)
    pd.concat(memberships,ignore_index=True).to_csv(destination/'subset_manifest.csv',index=False)
    pd.DataFrame(checks).to_csv(destination/'actual_load_checks.csv',index=False)
    pd.DataFrame(overlaps).to_csv(destination/'role_overlap_checks.csv',index=False)
    hashes[str(hist/'split_manifest.json')] = sha(hist/'split_manifest.json')
    hashes[str(destination/'subset_manifest.csv')] = sha(destination/'subset_manifest.csv')
    dump(destination/'input_lock.json',hashes)
    summary = {'status':'PASS_FOR_CONDITIONAL_USTC_E1_ONLY','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'command':sys.argv,'unique_input_rows':len(pv),'role_manifest_rows':sum(len(v) for v in manifests),'outer_folds':len(splits),'spot_prediction_rows':sum(v['loaded_rows'] for v in checks),'base_models_retrained':0,'new_arbiters_trained':0,'outer_predictions_evaluated':False,'legitimate_historical_oof_reused_not_regenerated':True,'independent_attack_activity_metadata_available':False,'fresh_blind_test':False,'E2_ready':False,'duration_seconds':time.perf_counter()-started,'scope':'Conditional development-budget and matched family-composition experiments; not an independent-activity causal experiment or submission readiness.'}
    dump(destination/'load_check.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
