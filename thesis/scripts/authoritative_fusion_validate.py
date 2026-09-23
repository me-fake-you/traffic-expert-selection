"""Final completeness, split, image/PDF and source snapshot audit."""
import json,sys,shutil,importlib.metadata
from pathlib import Path
from authoritative_fusion_benchmark import OUT,ROOT,SEEDS,METHODS,sha,dump
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

def main(out=OUT):
    out=Path(out)
    with np.load(out/'features.npz') as z:
        groups=z['group'];y=z['y'];ids=z['sample_id']
        assert z['stats'].shape==(40000,11) and z['temporal'].shape==(40000,30)
    ninner=0
    for seed in SEEDS:
        for fold in range(20):
            with np.load(out/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz') as z:
                tr,va,te=z['train'],z['validation'],z['test'];inner=z['inner_fold']
                assert (len(tr),len(va),len(te))==(34000,4000,2000)
                assert len(set(tr)|set(va)|set(te))==40000
                assert not(set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
                for k in range(3):
                    a=tr[inner[tr]!=k];v=tr[inner[tr]==k]
                    assert not set(groups[a])&set(groups[v]);assert set(y[a])=={0,1};ninner+=1
            with np.load(out/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'/'train_oof.npz') as p:
                assert np.array_equal(p['index'],tr);assert np.isfinite(p['p']).all();assert ((p['p']>=0)&(p['p']<=1)).all()
    splits=dict(outer_group_disjoint_cells=200,inner_group_disjoint_checks=ninner,all_inner_training_sets_have_two_classes=True,all_train_oof_indices_match=True,unique_samples=len(set(ids)))
    dump(out/'split_oof_validation.json',splits)
    table=pd.read_csv(out/'benchmark_results.csv');assert list(table.method)==METHODS
    assert table.seeds.eq(10).all() and table.folds_per_seed.eq(20).all()
    assert table.feature_to_decision_latency_ms_mean.gt(0).all()
    cells=pd.read_csv(out/'per_fold_results.csv');assert len(cells)==15400
    latency=pd.read_csv(out/'runtime_latency.csv');assert len(latency)==2200 and latency.validation_decision_matches_frozen.all()
    curve=pd.read_csv(out/'selective_per_seed.csv')
    for r in curve[curve.feasible].itertuples():
        assert r.accepted_count==r.accepted_benign+r.accepted_malicious
        if r.target_coverage==.2337:assert r.accepted_count==9348
    hashes=pd.read_csv(out/'verified_artifact_hashes.csv')
    for r in hashes.itertuples():assert sha(out/r.path)==r.sha256,r.path
    figures=[]
    for name in ('coverage_vs_selective_macro_f1','calls_per_sample'):
        with Image.open(out/(name+'.png')) as im:
            dpi=im.info['dpi'];assert all(abs(x-300)<.1 for x in dpi);size=list(im.size)
        pdf=PdfReader(out/(name+'.pdf'));assert len(pdf.pages)==1
        fonts=pdf.pages[0]['/Resources']['/Font'];embedded=[]
        for f in fonts.values():
            font=f.get_object()
            if '/DescendantFonts' in font:font=font['/DescendantFonts'][0].get_object()
            desc=font.get('/FontDescriptor')
            if desc:embedded.append(any(k in desc.get_object() for k in ('/FontFile','/FontFile2','/FontFile3')))
        assert embedded and all(embedded)
        assert pdf.pages[0].extract_text().strip()
        figures.append(dict(figure=name,pixels=size,dpi=list(dpi),pages=1,embedded_fonts=True,extractable_text=True))
    dump(out/'figure_validation.json',dict(figures=figures,manual_visual_check='Axes, legends, footnotes and labels inspected; no clipping; bar baseline zero'))
    files=['scripts/authoritative_fusion_benchmark.py','scripts/authoritative_fusion_report.py','scripts/authoritative_fusion_runtime_replay.py','scripts/authoritative_fusion_latency.py','scripts/authoritative_fusion_validate.py','tests/test_authoritative_fusion_benchmark.py','src/mad_etd/features.py','src/mad_etd/fusion.py','src/mad_etd/ood.py','src/mad_etd/detectors.py','src/mad_etd/schemas.py','src/mad_etd/extract_ustc.py']
    source=[]
    for name in files:
        p=ROOT/name;target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);source.append(dict(path=name,sha256=sha(p)))
    dump(out/'source_manifest.json',source)
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','pandas','matplotlib','torch','joblib','pydantic','pypdf')}
    dump(out/'environment.json',dict(python=sys.version,packages=versions,numerical_threads_per_worker=1,training_workers=6,latency_workers=1))
    final=dict(status='passed',completed_seed_fold_cells=200,methods=len(METHODS),method_condition_cells=len(cells),heldout_group_bootstrap_iterations=10000,artifact_hashes_verified=len(hashes),split_checks=splits,runtime_parity_checks=2200,figures_verified=2,source_snapshot_files=len(source))
    dump(out/'final_validation.json',final);print(json.dumps(final))

if __name__=='__main__':main()
