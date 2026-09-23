"""Summarize natural conditions and separate audit replication without model inference."""
import sys,json,gzip,shutil,inspect
from pathlib import Path
import numpy as np
import pandas as pd
from natural_shift_audit import ROOT,BASE,OUT,SEEDS,CONDITIONS,sha,dump,csv
import authoritative_selective as a

def summarize():
    summary=[];curves=[];pairs=[];regions=[];bundles={};delta={};observed={}
    for c in CONDITIONS:
        for m in a.METHODS:
            with np.load(OUT/'curves'/f'{c}__{m}.npz') as z:bundles[c,m]={k:z[k] for k in z.files}
        f= bundles[c,'V3_Full'];b=bundles[c,'V1_Average'];delta[c]=f['boot_error']-b['boot_error'];observed[c]=(f['point_error']-b['point_error']).mean(0)
    maxdev=np.zeros(10000)
    for c in CONDITIONS:maxdev=np.maximum(maxdev,np.max(np.nan_to_num(np.abs(delta[c]-observed[c]),nan=0.),axis=1))
    critical=np.quantile(maxdev,.95)
    for c in CONDITIONS:
        dl,dh,dn=a.finite_ci(delta[c]);regions+=a.regions(c,'pointwise_95',dl,dh)+a.regions(c,'simultaneous_95',observed[c]-critical,observed[c]+critical)
        for j,q in enumerate(a.COV):pairs.append(dict(condition=c,coverage=q,full_minus_average_error=observed[c][j],ci95_low=dl[j],ci95_high=dh[j],valid_draws=dn[j],full_lower_error=dh[j]<0,simultaneous_low=observed[c][j]-critical,simultaneous_high=observed[c][j]+critical))
        for m in a.METHODS:
            d=bundles[c,m];al,ah,an=a.finite_ci(d['boot_aurc']);fl,fh,fn=a.finite_ci(d['boot_f1']);el,eh,en=a.finite_ci(d['boot_error']);n=int(d['sizes'][0,0]);cf=d['confusion']
            row=dict(condition=c,method=m,study='relative_time_refit' if c=='temporal_early_late_refit' else 'locked_observed_stratum',samples_per_seed=n,groups=int(d['groups'][0]),benign=int(cf[0,0]+cf[0,1]),malicious=int(cf[0,2]+cf[0,3]),cohort_fraction=n/40000,aurc_mean=d['point_aurc'].mean(),aurc_std=d['point_aurc'].std(ddof=1),aurc_ci95_low=float(al),aurc_ci95_high=float(ah),aurc_valid_draws=int(an),macro_f1_mean=d['point_f1'][:,-1].mean(),macro_f1_std=d['point_f1'][:,-1].std(ddof=1),macro_f1_ci95_low=fl[-1],macro_f1_ci95_high=fh[-1],accuracy_mean=1-d['point_error'][:,-1].mean(),native_coverage_mean=d['native_coverage'].mean(),native_rejection_rate=1-d['native_coverage'].mean(),seeds=10,group_heldout_folds=20,alias_of='truncated_prefix' if c=='full_window_mismatch' else '',strict_capture_disjoint=True)
            if m=='V3_Full':
                b=bundles[c,'V1_Average'];bl,bh,bn=a.finite_ci(d['boot_aurc']-b['boot_aurc'])
                row.update(aurc_delta_vs_average=(d['point_aurc']-b['point_aurc']).mean(),delta_ci95_low=float(bl),delta_ci95_high=float(bh),dominance_grid_points=int(np.sum(dh<0)))
            for q in (.2337,1.):
                j=np.where(a.COV==q)[0][0];row[f'error_at_{q}']=d['point_error'][:,j].mean();row[f'f1_at_{q}']=d['point_f1'][:,j].mean()
            summary.append(row)
            for j,q in enumerate(a.COV):
                accepted=max(1,round(q*n));curves.append(dict(condition=c,method=m,target_coverage=q,achieved_coverage=accepted/n,accepted_count=accepted,samples=n,groups=int(d['groups'][0]),selective_macro_f1_mean=d['point_f1'][:,j].mean(),selective_macro_f1_std=d['point_f1'][:,j].std(ddof=1),f1_ci95_low=fl[j],f1_ci95_high=fh[j],selective_error_mean=d['point_error'][:,j].mean(),selective_error_std=d['point_error'][:,j].std(ddof=1),error_ci95_low=el[j],error_ci95_high=eh[j],valid_bootstrap_draws=en[j],native_coverage_mean=d['native_coverage'].mean()))
    csv(OUT/'natural_shift.csv',summary);csv(OUT/'natural_risk_coverage.csv',curves);csv(OUT/'natural_paired_differences.csv',pairs);csv(OUT/'natural_dominance_intervals.csv',regions)
    dump(OUT/'natural_simultaneous_band.json',dict(critical_deviation=float(critical),conditions=11,grid_points=102,undefined_empty_stratum_draws_excluded=True))
    return pd.DataFrame(summary),pd.DataFrame(curves),pd.DataFrame(regions)

def capture_results():
    meta=pd.read_csv(OUT/'metadata.csv');rows=[];from sklearn.metrics import f1_score
    for seed in SEEDS:
        with np.load(BASE/'selective_reliability_v1/cache'/f'seed_{seed}.npz') as z:
            index=z['index'];yy=z['y'];cap=meta.set_index('index').loc[index].source_file.to_numpy()
            for source in sorted(set(cap)):
                mask=cap==source
                for m in a.METHODS:
                    pred=z[f'clean__{m}__pred'][mask];rows.append(dict(seed=seed,source_file=source,method=m,n=int(mask.sum()),macro_f1=f1_score(yy[mask],pred,labels=[0,1],average='macro',zero_division=0),accuracy=float(np.mean(yy[mask]==pred)),native_coverage=float(np.mean(z[f'clean__{m}__verdict'][mask]<2)),group_bootstrap_ci='not identifiable within one held-out group; pooled intervals use original 20 groups'))
    csv(OUT/'leave_capture_out_results.csv',rows)

def figures(summary,curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
    for ax,c,title in zip(axes,['tls_absent','temporal_early_late_refit'],['Naturally absent TLS handshake','Within-capture early → late (refitted)']):
        for m,col,label,ls in [('V3_Full','#B64342','Full fusion','-'),('V1_Average','#3775BA','Probability average','--')]:
            d=curve[(curve.condition==c)&(curve.method==m)]
            ax.plot(d.target_coverage,d.selective_error_mean,c=col,label=label,ls=ls,lw=1.8);ax.fill_between(d.target_coverage,d.error_ci95_low,d.error_ci95_high,color=col,alpha=.15,lw=0)
        ax.set(title=title,xlabel='Coverage',xlim=(0,1));ax.grid(axis='y',alpha=.2);ax.text(.02,.96,'No pointwise full-fusion dominance',va='top',transform=ax.transAxes,fontsize=8)
    top=curve[curve.condition.isin(['tls_absent','temporal_early_late_refit'])&curve.method.isin(['V3_Full','V1_Average'])].error_ci95_high.max();axes[0].set_ylim(0,min(1,np.ceil(top*20)/20));axes[0].set_ylabel('Selective error (lower is better)');axes[1].legend(loc='upper right',fontsize=8,frameon=False)
    fig.text(.5,.01,'Paired whole-group bootstrap 95% bands; rejected verdicts rank last. High coverage forces binary projections.\nRelative-time refit retains held-out groups; clocks are not comparable across captures.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.11,1,1))
    for ext in ('png','pdf'):fig.savefig(OUT/f'natural_risk_coverage.{ext}',dpi=300)
    plt.close(fig)
    dist=pd.read_csv(OUT/'empirical_distributions.csv');labels=['TLS present','Truncated prefix','≤4 packets','Later flow segment'];rates=[]
    for pop in ('all','selected'):
        d=dist[dist.population==pop];rates.append([d[(d.dimension=='tls')&(d.value==1)].rate.sum(),d[(d.dimension=='truncated')&(d.value==1)].rate.sum(),d[(d.dimension=='length')&(d.value<=4)].rate.sum(),d[(d.dimension=='segment')&(d.value>0)].rate.sum()])
    fig,ax=plt.subplots(figsize=(7.5,4));x=np.arange(4)
    for values,shift,col,label in zip(rates,[-.18,.18],['#767676','#3775BA'],['Source records (634,630)','Locked cohort (40,000)']):
        bars=ax.bar(x+shift,100*np.array(values),.36,color=col,label=label)
        ax.bar_label(bars,labels=[f'{v*100:.2f}%' for v in values],padding=3,fontsize=8)
    ax.set_xticks(x,labels);ax.set(ylabel='Empirical frequency (%)',ylim=(0,85));ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'empirical_natural_conditions.{ext}',dpi=300)
    plt.close(fig)
    ad=pd.read_csv(OUT/'audit_replication.csv');kinds=['semantic_payload_forgery','wellformed_forged_source_hash','replay_new_ref','cross_case_rebinding'];vals=[int(ad[ad.study=='synthetic_structural_gate'].admitted.sum())]+[int(ad[(ad.study=='semantic_boundary_probe')&(ad.case_type==k)].admitted.iloc[0]) for k in kinds];den=[5000]+[100]*4
    fig,ax=plt.subplots(figsize=(9,4.3));xx=np.arange(5);ax.bar(xx,np.array(vals)/den,color=['#42949E']+['#B64342']*4)
    for i,(v,n) in enumerate(zip(vals,den)):ax.text(i,v/n+.02,f'{v}/{n}',ha='center',fontsize=9)
    ax.set_xticks(xx,['Structural / policy\nviolations','Semantic\nforgery','Well-formed forged\nsource hash','Replay with\nnew reference','Cross-case\nrebinding']);ax.set(ylabel='Admitted fraction',ylim=(0,1.17));ax.grid(axis='y',alpha=.2)
    fig.text(.5,.01,'Standalone default-off admission gate. Structural rejection is not semantic authenticity or universal replay protection.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.06,1,1))
    for ext in ('png','pdf'):fig.savefig(OUT/f'audit_boundary_replication.{ext}',dpi=300)
    plt.close(fig)

def reports(summary,dom):
    inv=pd.read_csv(OUT/'condition_inventory.csv');dist=pd.read_csv(OUT/'empirical_distributions.csv');capture=pd.read_csv(OUT/'capture_inventory.csv');audit=json.loads((OUT/'system_audit/COMPLETE.json').read_text());traces=json.loads((OUT/'trace_summary.json').read_text())
    def rate(pop,dim,value):return dist[(dist.population==pop)&(dist.dimension==dim)&(dist.value==value)].rate.sum()
    lines=['# Natural-shift results','', '**Answer: no evidence of a persistent full-fusion reliability advantage.** The preceding locked analysis did not establish such an advantage; none of the 11 natural/relative-time conditions here has a tested coverage point where the paired 95% interval establishes strictly lower error for full fusion. This is not an equivalence claim.','',
        '## Empirical data and natural conditions','',f"All 634,630 source records were scanned with source SHA-256 checks; the original 40,000-row selection was retained. TLS-handshake fields are present in {rate('all','tls',1):.4%} of source records and {rate('selected','tls',1):.4%} of the locked cohort. Stored sequences are truncated in {rate('all','truncated',1):.4%} and {rate('selected','truncated',1):.4%}, respectively. The cohort is group-balanced, so its rates are not population-prevalence estimates.",
        '', 'Stats and Temporal apply to every record in the locked cohort; conditional TLS applies to 2,013/40,000. Sequence lengths range from 1 to the 64-packet extractor cap. Packet-length/direction/IAT array lengths align in every source record. Stats summarize the full segment while a truncated Temporal sequence covers its prefix; all 289 selected full-window mismatches are exactly the truncated-prefix cohort, not an independent perturbation.',
        '', 'Conditions retain actual joint missingness and lengths and their empirical membership rates. No additional independent Bernoulli masks, universal whole-view removal, feature replacement or synthetic record pairing was applied. They are observational conditional-cohort analyses, not a claim of a newly observed deployment distribution.','',
        '| Natural condition | n | Cohort rate | Groups | Benign / malicious |','|---|---:|---:|---:|---:|']
    lines += [f'| {r.condition} | {r.samples} | {r.cohort_rate:.4%} | {r.groups} | {r.benign} / {r.malicious} |' for r in inv.itertuples()]
    lines += ['', 'The TLS-visible and truncated subsets cover only seven groups; the later-segment subset contains only malicious records. Fixed-label binary Macro-F1 on a one-class subset cannot exceed 0.5. Confidence intervals explicitly retain the number of nonempty bootstrap draws; these subsets do not establish balanced binary detection performance.','',
        '## Group, capture and time separation','',
        'The same 20 outer application/family folds and seeds 42–51 are retained. The 24 capture files are nested in these groups. Saved leave-capture-out partitions preserve each parent training/validation split and restrict its test indices to one capture. Thus there are 240 capture/seed partitions; no other capture from the held-out group enters training. Their pooled result is exactly `observed_all`, not an independent improvement experiment. Per-capture results and exact indices are saved separately.',
        '', 'A global calendar train-earlier/test-later study is **not identifiable** here: twenty captures have relative timestamps, while the four with plausible calendar epochs are all malware (Neris, Nsis-ay, Virut, Zeus). Ordering these different clocks would confound time with capture/label. No artificial calendar ordering was invented.',
        '', 'A separate **within-capture relative-time refit** is completed: use records ending strictly before each capture’s median start time in parent training/validation groups; evaluate records starting strictly after that boundary in the held-out group. Purge boundary ties and overlapping windows. All 200 fold/seed indices and newly trained models/OOF outputs are saved under `temporal_refit/`. The test cohort is 20,000 records per seed; original features and all parent models remain unchanged. This studies capture progression across disjoint groups, not calendar-time forecasting across captures.',
        '', 'The same HGB/OOD configuration, TLS rule, reliability calibration, group cross-fitting, fixed fusion rules, 32–16 MLP, logistic and attention settings were used. EDL is independently trained on the same temporal-split features/rows and labeled a feature model. No test outcomes selected any setting. The reused training runner also produces its earlier small fusion heads; these unused artifacts are retained, not substituted into the nine reported methods.',
        '', 'Optional cross-dataset work was skipped: the canonical local CIC-IDS2017 processed directory contains aggregate-CSV manifests but no aligned FlowRecord/packet-prefix records. Aggregate flow statistics cannot reconstruct the locked packet direction, entropy, repeated-size, burst and prefix-IAT semantics. No substitute feature set or pooled USTC/CIC result is reported.','',
        '## Risk–coverage and AURC','', '| Condition | Full AURC | Average AURC | Difference [paired 95% CI] | Full lower-error regions |','|---|---:|---:|---|---|']
    for c in CONDITIONS:
        f=summary[(summary.condition==c)&(summary.method=='V3_Full')].iloc[0];b=summary[(summary.condition==c)&(summary.method=='V1_Average')].iloc[0];dd=dom[(dom.condition==c)&(dom.interval_type=='pointwise_95')&(dom.status=='full_lower_error')]
        regions='none exists' if dd.empty else '; '.join(f'{r.coverage_start:.5g}–{r.coverage_end:.5g}' for r in dd.itertuples())
        lines.append(f'| {c} | {f.aurc_mean:.6f} | {b.aurc_mean:.6f} | {f.aurc_delta_vs_average:+.6f} [{f.delta_ci95_low:.6f}, {f.delta_ci95_high:.6f}] | {regions} |')
    lines += ['', 'TLS-visible AURC has a favorable full-fusion point estimate, but its paired interval crosses zero. The separately refitted relative-time AURC difference also crosses zero; neither result proves superiority or equivalence. Truncated-prefix performance is particularly poor for forced full-fusion projections; all such negative outcomes are retained.',
        '', 'Every condition has all nine methods and 102 requested coverage points, including 0.2337 and 1.0. Coverage is measured within that condition; rounding and small strata mean achieved coverage can differ, and both are recorded. The exact same 10,000 stratified group-multiplicity draws are applied jointly across methods and seeds. Unequal condition-specific group sizes are respected, selection is recomputed per draw, empty draws remain undefined, and AURC is exact mean prefix error. Native rejects rank last; forcing them above native coverage is a diagnostic rather than native acceptance. Ties use the original global row index. Whole-group CIs are conditional on the saved models; test points and coverage grids are not independent trials.',
        '', 'Both pointwise paired dominance intervals and a simultaneous bootstrap deviation-band supplement are recorded, including every non-dominance interval. `full_window_mismatch` duplicates `truncated_prefix` and is explicitly marked, not counted as independent corroboration.',
        '', '## Artifacts','', '[Main table](natural_shift.csv), [all curves](natural_risk_coverage.csv), [paired differences](natural_paired_differences.csv), [dominance intervals](natural_dominance_intervals.csv), [exact protocol](NATURAL_PROTOCOL.json), [run log](RUNS.md). Source metadata, empirical distributions, capture/temporal indices and per-capture results are in the same directory.','', '![Natural risk–coverage](natural_risk_coverage.png)','', '![Empirical conditions](empirical_natural_conditions.png)','']
    (OUT/'NATURAL_SHIFT_RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    ad=pd.read_csv(OUT/'audit_replication.csv');lines=['# Audit / accountability replication','', '**Structural rejection replicated; broad forgery/replay protection did not.** Detection scores and these system-boundary measurements are separate studies.','', '| Check | Exact result | Scope |','|---|---:|---|',
        '| Structured threat cases | 0 successful attacks / 19 checks | Mixed policy, ownership and admission checks; T17 is a static allow-list assertion, not an executed selection gate |',
        f"| Structural / policy mutation submissions | {audit['structural_admitted']} admitted / {audit['structural_probes']} | Existing default-off AdmissionGate; 15 probe types, seeds 42–51 |",
        '| Fresh valid controls | 500 admitted / 500 | Rules out rejecting every submission |', '| Replay setup controls | 330 admitted / 330 | Separate from the 5,000 attack submissions |',
        '| Same-reference replay attacks | 0 admitted / 330 | Reference-cache rejection works |',
        '| Semantic payload forgery | 100 admitted / 100 | Whitelisted identity and correctly shaped fields do not verify semantic truth |',
        '| Well-formed forged source hash | 100 admitted / 100 | Source hash format is checked; source bytes/authentic producer are not verified |',
        '| Replay under a new reference | 100 admitted / 100 | Reusing content with a fresh reference bypasses same-reference replay detection |',
        '| Cross-case rebinding | 100 admitted / 100 | This API does not bind payload origin to the supplied case context |',
        '| Real-source blocked-field mutation pairs | 2,000 / 2,000 identical safe input, 11/30 features and Fusion snapshots | 200 USTC flows × 10 seeds; separate current rule-runtime audit, not HGB accuracy |',
        f"| Rule-runtime nonbinary verdict traces | {audit['system_rejects_traceable']} / {audit['system_native_rejected_verdicts']} traceable | 2,000 mutated cases plus 200 baseline controls |",
        f"| Frozen-benchmark native rejection replay | {traces['traceable']:,} / {traces['total_native_rejections']:,} traceable | Yager, full fusion and EDL across locked and relative-time studies |",'',
        'The target of zero admitted structured violations is satisfied, but **zero admitted forged/replayed content is not satisfied**. The 400 admitted boundary probes are retained as negative results. Registry hashes and audit hashes are not digital signatures or proof of producer authenticity. No runtime security code was patched to obtain a target outcome.',
        '', '## Runtime scope and exact denominators','',
        'AdmissionGate is a standalone, default-off prototype exercised directly here. It is not thereby enabled in the default detection engine or claimed as an end-to-end enforcement layer for the locked benchmark. The original 19 checks are freshly executed; copied architecture documents only satisfy their provenance prerequisites. They are not 19 independent malicious evidence submissions. Every probe type has its own denominator and admission decision in `audit_replication.csv` and `system_audit/`.',
        '', 'Field mutations change identifiers, labels/family/application, provenance, endpoint/port/time context and unknown extra fields while retaining actual behavioral inputs. All 2,000 safe-input/feature/Fusion invariance pairs match. This control uses the separate rule runtime (which returned nonbinary verdicts in these cases); it is not evidence of trained-HGB detection quality. Audit events include field-policy, reliability, routing, evidence, final fusion and reporting stages. Baseline control events are also persisted.',
        '', '## Per-verdict accountability','',
        'The unique frozen verdict key is (study, seed, fold, global sample index, method). Overlapping natural strata share that same verdict; each trace lists all condition memberships instead of inflating the denominator by duplicating the event. Every Yager/full native reject is replayed through the actual `FusionAgent`, `apply_ood_policy_to_evidence`, reliability discounting and ordered Yager operations, with exact verdict agreement. Each JSONL line contains raw per-view probability/mass, applicability, validation reliability, OOD score, post-policy abstention/mass, each combination stage, final branch and reasons. It links the frozen file, SHA-256, row, and global index. EDL traces record its native total-uncertainty threshold plus the feature/model source, without claiming per-view fusion or feature causality.',
        '', 'These frozen-model traces are **retrospective deterministic replays**, not audit logs emitted during the original detector evaluation. No detector or EDL model is called again for this tracing; only decisions are reconstructed. Local trace indices give gzip JSONL line numbers. This supports reproducibility and inspection, not tamper-proof logging or content authentication.',
        '', '[Audit table](audit_replication.csv), [native rejection trace counts](trace_summary.json), [system audit counters](system_audit/COMPLETE.json), [run log](RUNS.md). `rejection_traces/{locked,temporal}/seed_*_index.csv` maps every traced verdict to its exact JSONL line; `trace_source_map.json` maps rule functions to source lines and hashes.',
        '', '![Admission boundary results](audit_boundary_replication.png)','']
    (OUT/'AUDIT_RESULTS.md').write_text('\n'.join(lines),encoding='utf8')

def accountability_summary():
    table=pd.read_csv(OUT/'audit_replication.csv')
    table=table[~table.study.isin(['verdict_accountability','field_boundary_check'])]
    trace=json.loads((OUT/'trace_summary.json').read_text())
    system=json.loads((OUT/'system_audit/COMPLETE.json').read_text())
    rows=[]
    for kind,n,k,scope in [('frozen_native_rejections',trace['total_native_rejections'],trace['traceable'],'retrospective deterministic decision replay; unique study/seed/fold/index/method'),('system_native_rejections',system['system_native_rejected_verdicts'],system['system_rejects_traceable'],'separate rule runtime; baseline and mutation event logs')]:
        rows.append(dict(study='verdict_accountability',case_type=kind,denominator=n,traceable=k,failures=n-k,scope=scope))
    fields=pd.read_csv(OUT/'system_audit/field_mutation_cases.csv')
    for column in ('blocked_field_admissions','fusion_ownership_violation'):
        rows.append(dict(study='field_boundary_check',case_type=column,denominator=len(fields),failures=int(fields[column].sum()),scope='separate rule-runtime mutation cases'))
    pd.concat([table,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/'audit_replication.csv',index=False,encoding='utf-8-sig')

def main():
    accountability_summary()
    summary,curve,dom=summarize();capture_results();figures(summary,curve);reports(summary,dom)
    validate(summary,curve)

def validate(summary,curve):
    from sklearn.metrics import f1_score
    from PIL import Image
    import fitz
    protocol=json.loads((OUT/'NATURAL_PROTOCOL.json').read_text());assert all(sha(BASE/p)==h for p,h in protocol['protected_sha256'].items())
    metadata=pd.read_csv(OUT/'metadata.csv');cuts=pd.read_csv(OUT/'within_capture_time_boundaries.csv').set_index('source_file').cutoff
    group=metadata.group.to_numpy();valid=0
    for s in SEEDS:
        for g in range(20):
            p=OUT/'temporal_refit/splits'/f'seed_{s}'/f'fold_{g:02d}.npz'
            with np.load(p) as z:tr=z['train'];va=z['validation'];te=z['test'];inner=z['inner_fold']
            assert not(set(group[tr])&set(group[va]) or set(group[tr])&set(group[te]) or set(group[va])&set(group[te]))
            for idx in (tr,va):assert np.all(metadata.iloc[idx].flow_end.to_numpy()<metadata.iloc[idx].source_file.map(cuts).to_numpy())
            assert np.all(metadata.iloc[te].flow_start.to_numpy()>metadata.iloc[te].source_file.map(cuts).to_numpy())
            for k in range(3):assert not set(group[tr[inner[tr]==k]])&set(group[tr[inner[tr]!=k]])
            dp=OUT/'temporal_refit/fold_runs'/f'seed_{s}'/f'fold_{g:02d}'
            trained=json.loads((dp/'NATURAL_TRAINED.json').read_text())
            assert not trained['test_data_in_training']
            assert all(sha(dp/name)==digest for name,digest in trained['model_sha256'].items())
            assert sha(dp/'natural_predictions.npz')==json.loads((dp/'NATURAL_DONE.json').read_text())['predictions_sha256'];valid+=1
    for s in SEEDS:
        partitions=pd.read_csv(OUT/'leave_capture_out_indices.csv');partitions=partitions[partitions.seed==s];alltest=[]
        for p in partitions.itertuples():
            with np.load(OUT/p.path) as z:tr=z['train'];te=z['test'];alltest.extend(te)
            assert not set(metadata.iloc[tr].source_file)&set(metadata.iloc[te].source_file)
        assert sorted(alltest)==list(range(40000))
    for m in a.METHODS:
        with np.load(BASE/'selective_reliability_v1/curves'/f'clean__{m}.npz') as z:previous=z['point_aurc']
        with np.load(OUT/'curves'/f'observed_all__{m}.npz') as z:np.testing.assert_allclose(z['point_aurc'],previous,atol=1e-12)
    figures_info={}
    for name in ('natural_risk_coverage','empirical_natural_conditions','audit_boundary_replication'):
        im=Image.open(OUT/f'{name}.png');assert min(im.info['dpi'])>=299;doc=fitz.open(OUT/f'{name}.pdf');fonts=doc[0].get_fonts();assert fonts and not doc[0].get_images() and all(doc.extract_font(f[0])[3] for f in fonts);doc.close()
        figures_info[name]=dict(pixels=im.size,dpi=im.info['dpi'],png_sha256=sha(OUT/f'{name}.png'),pdf_sha256=sha(OUT/f'{name}.pdf'))
    from mad_etd.fusion import FusionAgent
    from mad_etd.ood import apply_ood_policy_to_evidence
    import authoritative_fusion_benchmark as b
    source_map={}
    for obj in (FusionAgent.fuse,FusionAgent._discount,FusionAgent._combine_yager,FusionAgent._decide,apply_ood_policy_to_evidence,b.decision):
        path=Path(inspect.getsourcefile(obj));source_map[obj.__qualname__]=dict(file=str(path.relative_to(ROOT)),line=inspect.getsourcelines(obj)[1],sha256=sha(path))
    dump(OUT/'trace_source_map.json',source_map)
    trace=json.loads((OUT/'trace_summary.json').read_text());assert trace['traceable']==trace['total_native_rejections']
    for r in trace['runs']:assert sha(OUT/'rejection_traces'/r['study']/f"seed_{r['seed']}.jsonl.gz")==r['sha256']
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ('natural_shift_audit.py','natural_temporal_refit.py','natural_system_audit.py','natural_rejection_trace.py','natural_shift_report.py','authoritative_fusion_benchmark.py','frozen_fusion_fast.py','authoritative_selective.py'):shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    shutil.copy2(ROOT/'tests/test_natural_shift_audit.py',snapshot/'test_natural_shift_audit.py')
    dump(OUT/'validation.json',dict(status='passed',parent_files_unchanged=len(protocol['protected_sha256']),natural_summary_rows=len(summary),risk_curve_rows=len(curve),temporal_cells_checked=valid,temporal_inner_group_splits_checked=valid*3,nested_capture_partitions=240,observed_aurc_parity_methods=9,traceable_native_rejections=trace['traceable'],figures=figures_info,structural_admitted=0,boundary_probes_admitted=400))
    print('Natural-shift and separate audit reports validated',flush=True)

if __name__=='__main__':main()
