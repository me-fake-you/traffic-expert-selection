"""Extra metadata guard before invoking the already sealed evaluation runner.

Added after selection, before any outer scoring; it cannot alter selections or
the frozen protocol. Both this guard and the original runner are retained.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import coverage_experiment as c
import numpy as np


def main():
    run = c.OUT / 'experiment/run_001'
    assert not (run / 'completion.json').exists()
    c.verify_inputs(run)
    gate = c.read(run / 'selection_gate.json')
    assert gate['all_planned_choices_saved'] and gate['choices'] == 90
    for p, h in gate['sealed_files'].items():
        assert c.sha(p) == h
    splits = c.read(c.SPLITS); prov = c.provenance(); ids=[]; checks=[]
    for split in splits:
        fold = split['outer_fold']
        ds={k:c.load_matrix(fold,k,'evaluation',run) for k in c.KINDS}
        c.verify_pair(ds)
        d=ds['logistic']; pp=prov.loc[d['sample_hash']]
        assert len(d['sample_hash']) == 8000
        assert set(d['group']) == set(split['groups']['evaluation'])
        assert set(pp.capture) == set(split['captures']['evaluation'])
        assert np.array_equal(pp.group.to_numpy(),d['group'])
        assert np.array_equal(pp.label.to_numpy(),d['label'])
        _, ns=np.unique(d['group'],return_counts=True)
        assert np.all(ns==2000)
        for role in ['train','calibration','selection']:
            assert set(d['group']).isdisjoint(split['groups'][role])
            assert set(pp.capture).isdisjoint(split['captures'][role])
        ids.extend(d['sample_hash'].tolist())
        checks.append(dict(fold_id=fold,rows=len(d['sample_hash']),groups=sorted(set(d['group'])),
                           exact_evaluation_role_and_provenance_match=True))
    assert len(ids)==len(set(ids))==40000 and set(ids)==set(prov.index)
    c.dump(run/'outer_role_guard.json',dict(created=c.stamp(),guard_sha256=c.sha(__file__),
        original_runner_sha256=c.sha(Path(c.__file__)),all_checks_passed=True,
        stage='after_all90_choices_sealed_before_outer_scoring',checks=checks,
        scope='metadata and saved candidate array identities only; not re-audit of original label truth'))
    sys.argv=[c.__file__,'evaluate','--run','run_001']
    c.main()


if __name__=='__main__': main()
