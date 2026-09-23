"""Label-only scoring and reporting after every protocol fold is frozen."""
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from authoritative_fusion_benchmark import SEEDS, METHODS, CONDITIONS, COVERAGES, CONFIG, ROOT, sha, dump, csvwrite

def counts(y,p,mask=None):
    if mask is not None:y,p=y[mask],p[mask]
    return np.bincount(2*y.astype(int)+p.astype(int),minlength=4).astype(float)
def cfmetric(c):
    tn,fp,fn,tp=np.moveaxis(np.asarray(c),-1,0)
    f0=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
    n=tn+fp+fn+tp
    mf=np.where(n>0,(f0+f1)/2,np.nan);ac=np.divide(tn+tp,n,out=np.full_like(n,np.nan),where=n>0)
    return mf,ac
def ms(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return (float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.) if len(x) else (np.nan,np.nan)
def ci(vals):
    vals=np.asarray(vals);vals=vals[np.isfinite(vals)]
    return tuple(np.quantile(vals,[.025,.975])) if len(vals) else (np.nan,np.nan)
def fmt(x):return 'N/A' if not np.isfinite(x) else f'{x:.4f}'
def pm(a,b):return 'N/A' if not np.isfinite(a) else f'{a:.4f}±{b:.4f}'
def interval(a,b):return 'N/A' if not np.isfinite(a) else f'[{a:.4f}, {b:.4f}]'
def bootstrap_weights():
    rng=np.random.default_rng(42);w=np.zeros((CONFIG['bootstrap_iterations'],20))
    for i in range(len(w)):
        idx=np.r_[rng.choice(10,10,replace=True),rng.choice(np.arange(10,20),10,replace=True)]
        w[i]=np.bincount(idx,minlength=20)
    return w
def summarize(out):
    out=Path(out);started=time.perf_counter();w=bootstrap_weights()
    paths=[out/'fold_runs'/f'seed_{s}'/f'fold_{f:02d}' for s in SEEDS for f in range(20)]
    missing=[str(p) for p in paths if not (p/'DONE.json').exists()]
    if missing:raise RuntimeError(f'Refuse partial benchmark: {len(missing)} missing folds')
    allcells=pd.concat([pd.read_csv(p/'metrics.csv') for p in paths],ignore_index=True)
    clean_cf=np.zeros((len(METHODS),10,20,4));native_cf=clean_cf.copy();app_cf=clean_cf.copy()
    native_harm_group=np.zeros((len(METHODS),10,20))
    sel_cf={c:clean_cf.copy() for c in COVERAGES};sel_valid={c:np.zeros((len(METHODS),10),bool) for c in COVERAGES}
    seedrows=[];curve_seed=[];pertrows=[];hashrows=[];auditrows=[]
    for si,seed in enumerate(SEEDS):
        folddata=[]
        for f in range(20):
            dest=out/'fold_runs'/f'seed_{seed}'/f'fold_{f:02d}';done=json.loads((dest/'DONE.json').read_text());predpath=dest/'predictions.npz'
            assert sha(predpath)==done['predictions_sha256'];hashrows.append(dict(seed=seed,fold=f,path=str(predpath.relative_to(out)),sha256=sha(predpath)))
            with np.load(predpath,allow_pickle=False) as z:folddata.append({k:z[k] for k in z.files})
            ff=out/'frozen_predictions'/f'seed_{seed}'/f'fold_{f:02d}';fh=json.loads((dest/'frozen_manifest.json').read_text())
            for cond in CONDITIONS:
                pp=ff/f'test_{cond}.npz';assert sha(pp)==fh[cond];hashrows.append(dict(seed=seed,fold=f,path=str(pp.relative_to(out)),sha256=fh[cond]))
                with np.load(pp) as z:
                    assert np.array_equal(z['index'],folddata[-1]['index'])
                    assert np.isfinite(z['p']).all() and np.allclose(z['mass'].sum(-1),1)
            lin=json.loads((ff/'oof_lineage.json').read_text());assert all(x['group_disjoint'] for x in lin)
            auditrows.append(dict(seed=seed,fold=f,view_output_hashes_verified=True,oof_group_disjoint=True,feature_hash_matches=done['source_features_sha256']==sha(out/'features.npz')))
        y=np.concatenate([d['y'] for d in folddata]);group=np.concatenate([d['group'] for d in folddata]);idx=np.concatenate([d['index'] for d in folddata]);assert len(set(idx))==40000
        for mi,method in enumerate(METHODS):
            clean={k:np.concatenate([d[f'clean__{method}__{k}'] for d in folddata]) for k in ('pred','score','verdict','available','calls','accepted_fixed')}
            for f in range(20):
                mask=group==f
                clean_cf[mi,si,f]=counts(y,clean['pred'],mask)
                native_cf[mi,si,f]=counts(y,clean['pred'],mask&(clean['verdict']<2))
                app_cf[mi,si,f]=counts(y,clean['pred'],mask&clean['available'])
            mf,ac=cfmetric(clean_cf[mi,si].sum(0));nf,na=cfmetric(native_cf[mi,si].sum(0));af,aa=cfmetric(app_cf[mi,si].sum(0))
            seedrows.append(dict(seed=seed,method=method,pooled_macro_f1=float(mf) if method!='V0_TLS' else np.nan,pooled_accuracy=float(ac) if method!='V0_TLS' else np.nan,available_macro_f1=float(af),available_accuracy=float(aa),available_coverage=float(clean['available'].mean()),native_coverage=float((clean['verdict']<2).mean()),native_selective_macro_f1=float(nf),native_selective_error=1-float(na),calls_per_sample=float(clean['calls'].mean())))
            for cover in COVERAGES:
                k=round(cover*len(y));eligible=np.flatnonzero(clean['available']);order=eligible[np.lexsort((idx[eligible],-clean['score'][eligible]))];ok=len(order)>=k
                if ok:
                    selected=np.zeros(len(y),bool);selected[order[:k]]=True;sel_valid[cover][mi,si]=True
                    for f in range(20):sel_cf[cover][mi,si,f]=counts(y,clean['pred'],selected&(group==f))
                    sf,sa=cfmetric(sel_cf[cover][mi,si].sum(0))
                else:sf=sa=np.nan
                cc=sel_cf[cover][mi,si].sum(0)
                curve_seed.append(dict(seed=seed,method=method,target_coverage=cover,achieved_coverage=k/len(y) if ok else np.nan,accepted_count=k if ok else 0,accepted_benign=int(cc[0]+cc[1]) if ok else 0,accepted_malicious=int(cc[2]+cc[3]) if ok else 0,feasible=ok,selective_macro_f1=float(sf),selective_accuracy=float(sa)))
            basecorrect=clean['accepted_fixed']&(clean['pred']==y)
            for cond in CONDITIONS[1:]:
                d={k:np.concatenate([z[f'{cond}__{method}__{k}'] for z in folddata]) for k in ('pred','accepted_fixed','verdict','available','harmful')}
                harmful=basecorrect&d['accepted_fixed']&(d['pred']!=y);assert np.array_equal(harmful,d['harmful'])
                c=counts(y,d['pred']);fm,am=cfmetric(c);sfc,sac=cfmetric(counts(y,d['pred'],d['accepted_fixed']))
                nn=d['verdict']<2;nfc,nac=cfmetric(counts(y,d['pred'],nn))
                native_fixed=d['accepted_fixed']&nn;native_base=basecorrect&(clean['verdict']<2)
                native_harm=native_base&native_fixed&(d['pred']!=y)
                for f in range(20):native_harm_group[mi,si,f]+=int(native_harm[group==f].sum())
                nff,nfa=cfmetric(counts(y,d['pred'],native_fixed))
                pertrows.append(dict(seed=seed,condition=cond,method=method,macro_f1=float(fm) if method!='V0_TLS' else np.nan,accuracy=float(am) if method!='V0_TLS' else np.nan,delta_vs_clean=float(fm-mf) if method!='V0_TLS' else np.nan,harmful_flips=int(harmful.sum()),eligible_clean_confident_correct=int(basecorrect.sum()),harmful_flip_rate=float(harmful.sum()/max(basecorrect.sum(),1)),fixed_coverage=float(d['accepted_fixed'].mean()),fixed_selective_macro_f1=float(sfc),fixed_selective_error=1-float(sac),native_fixed_harmful_flips=int(native_harm.sum()),native_fixed_eligible_clean_correct=int(native_base.sum()),native_fixed_coverage=float(native_fixed.mean()),native_fixed_selective_macro_f1=float(nff),native_fixed_selective_error=1-float(nfa),native_coverage=float(nn.mean()),native_selective_macro_f1=float(nfc),native_selective_error=1-float(nac),unknown_rate=float((d['verdict']==3).mean()),suspicious_rate=float((d['verdict']==2).mean())))
        print('scored seed',seed,flush=True)
    allcells.loc[allcells.method=='V0_TLS',['macro_f1','accuracy']]=np.nan
    allcells.to_csv(out/'per_fold_results.csv',index=False)
    csvwrite(out/'per_seed_results.csv',seedrows);csvwrite(out/'perturbation_results.csv',pertrows);csvwrite(out/'selective_per_seed.csv',curve_seed)
    csvwrite(out/'verified_artifact_hashes.csv',hashrows);csvwrite(out/'audit_checks.csv',auditrows)
    seeds=pd.DataFrame(seedrows);pert=pd.DataFrame(pertrows);curveseed=pd.DataFrame(curve_seed)
    curves=[];summary=[];bootstrap_arrays={}
    for mi,method in enumerate(METHODS):
        cells=allcells[(allcells.method==method)&(allcells.condition=='clean')];ss=seeds[seeds.method==method];pp=pert[pert.method==method]
        fm,fs=ms(cells.macro_f1);am,ast=ms(cells.accuracy);pmf,pstd=ms(ss.pooled_macro_f1);pam,past=ms(ss.pooled_accuracy)
        groupf=cfmetric(clean_cf[mi])[0].mean(0);bootfold=w@groupf/20
        bootpooled=np.stack([cfmetric(w@clean_cf[mi,si])[0] for si in range(10)],axis=1).mean(1)
        if method=='V0_TLS':bootfold[:]=np.nan;bootpooled[:]=np.nan
        lo,hi=ci(bootfold);plo,phi=ci(bootpooled);bootstrap_arrays[method+'__foldmean']=bootfold;bootstrap_arrays[method+'__pooled']=bootpooled
        h=pp.groupby('seed').harmful_flips.sum();hm,hs=ms(h)
        # Cluster intervals retain seed/condition pairing within held-out groups.
        pcells=allcells[(allcells.method==method)&(allcells.condition!='clean')].copy()
        hg=pcells.groupby(['seed','fold']).harmful_flips.sum().unstack('fold').reindex(index=SEEDS,columns=range(20)).to_numpy()
        hlo,hhi=ci(w@hg.mean(0))
        pcells['accepted_count']=np.rint(pcells.fixed_coverage*2000)
        pcells['accepted_wrong']=np.rint(pcells.fixed_selective_error.fillna(0)*pcells.accepted_count)
        gg=pcells.groupby(['seed','fold'])[['accepted_count','accepted_wrong']].sum()
        accepted=gg.accepted_count.unstack('fold').reindex(index=SEEDS,columns=range(20)).to_numpy()
        wrong=gg.accepted_wrong.unstack('fold').reindex(index=SEEDS,columns=range(20)).to_numpy()
        errs=[]
        for si in range(10):
            den=w@accepted[si];num=w@wrong[si];errs.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
        errmat=np.asarray(errs);finite=np.isfinite(errmat).sum(0)
        errmean=np.divide(np.nansum(errmat,axis=0),finite,out=np.full(len(w),np.nan),where=finite>0)
        elo,ehi=ci(errmean)
        fixerr,fixstd=ms(pp.fixed_selective_error);nc,ns=ms(ss.native_coverage);nf,nfs=ms(ss.native_selective_macro_f1)
        trained=method in ('V4_Logistic','V4_MLP','V5_Gating','V6_EDL');native=method in ('V2_Yager','V3_Full','V3_OnDemand','V6_EDL','V0_TLS')
        row=dict(method=method,status='conditional_TLS_only' if method=='V0_TLS' else 'completed',macro_f1_fold_mean=fm,macro_f1_fold_std=fs,macro_f1_ci95_low=lo,macro_f1_ci95_high=hi,accuracy_fold_mean=am,accuracy_fold_std=ast,macro_f1_pooled_seed_mean=pmf,macro_f1_pooled_seed_std=pstd,pooled_macro_f1_ci95_low=plo,pooled_macro_f1_ci95_high=phi,accuracy_pooled_seed_mean=pam,accuracy_pooled_seed_std=past,harmful_flips_total=int(h.sum()),harmful_flips_per_seed_mean=hm,harmful_flips_per_seed_std=hs,perturbed_fixed_selective_error_mean=fixerr,perturbed_fixed_selective_error_std=fixstd,perturbed_fixed_coverage_mean=float(pp.fixed_coverage.mean()),native_coverage_mean=nc,native_selective_macro_f1_mean=nf,unknown_rate=float(cells.unknown_rate.mean()),suspicious_rate=float(cells.suspicious_rate.mean()),calls_per_sample=float(cells.calls_per_sample.mean()),decision_latency_ms_mean=float(cells.decision_latency_ms.mean()),decision_latency_ms_std=float(cells.decision_latency_ms.std()),trained_layer='y' if trained else 'n',params=int(cells.params.iloc[0]),native_abstention='y' if native else 'n',native_abstention_scope='missing TLS only' if method=='V0_TLS' else 'decision rule' if native else 'unavailable views only',available_coverage=float(ss.available_coverage.mean()),available_subset_macro_f1=float(ss.available_macro_f1.mean()),seeds=10,folds_per_seed=20,frozen_output_fusion='n' if method=='V6_EDL' else 'y',edl_network_calls_per_sample=1 if method=='V6_EDL' else 0)
        row.update(harmful_flips_per_seed_ci95_low=hlo,harmful_flips_per_seed_ci95_high=hhi,
            perturbed_fixed_selective_error_pooled_mean=float(np.mean(np.divide(wrong.sum(1),accepted.sum(1),out=np.full(10,np.nan),where=accepted.sum(1)>0))),
            perturbed_fixed_selective_error_pooled_ci95_low=elo,perturbed_fixed_selective_error_pooled_ci95_high=ehi)
        nh=pp.groupby('seed').native_fixed_harmful_flips.sum();nhm,nhs=ms(nh);nlo,nhi=ci(w@native_harm_group[mi].mean(0))
        row.update(native_fixed_harmful_flips_total=int(nh.sum()),native_fixed_harmful_flips_per_seed_mean=nhm,native_fixed_harmful_flips_per_seed_std=nhs,native_fixed_harmful_flips_per_seed_ci95_low=nlo,native_fixed_harmful_flips_per_seed_ci95_high=nhi,native_fixed_perturbed_coverage_mean=float(pp.native_fixed_coverage.mean()),native_fixed_perturbed_selective_error_mean=float(pp.native_fixed_selective_error.mean()))
        blo,bhi=ci(np.stack([cfmetric(w@clean_cf[mi,si])[1] for si in range(10)],axis=1).mean(1))
        row.update(accuracy_ci95_low=blo if method!='V0_TLS' else np.nan,accuracy_ci95_high=bhi if method!='V0_TLS' else np.nan)
        for cover in COVERAGES:
            s=curveseed[(curveseed.method==method)&(curveseed.target_coverage==cover)];val,std=ms(s.selective_macro_f1)
            if sel_valid[cover][mi].all():bs=np.stack([cfmetric(w@sel_cf[cover][mi,si])[0] for si in range(10)],axis=1).mean(1);cl,ch=ci(bs)
            else:cl=ch=np.nan
            curves.append(dict(method=method,coverage=cover,selective_macro_f1_mean=val,selective_macro_f1_std=std,ci95_low=cl,ci95_high=ch,feasible_seeds=int(sel_valid[cover][mi].sum()),accepted_benign_mean=float(s.accepted_benign.mean()),accepted_malicious_mean=float(s.accepted_malicious.mean()),selective_accuracy_mean=float(s.selective_accuracy.mean())))
            if cover in (.2337,1.):
                name='02337' if cover==.2337 else '10000';row[f'selective_macro_f1_at_{name}']=val;row[f'selective_macro_f1_at_{name}_std']=std;row[f'selective_macro_f1_at_{name}_ci95_low']=cl;row[f'selective_macro_f1_at_{name}_ci95_high']=ch
        summary.append(row)
    if (out/'runtime_latency.csv').exists():
        rt=pd.read_csv(out/'runtime_latency.csv').groupby('method').median_ms_per_sample.agg(['mean','std'])
        for row in summary:
            row['feature_to_decision_latency_ms_mean']=float(rt.loc[row['method'],'mean'])
            row['feature_to_decision_latency_ms_std']=float(rt.loc[row['method'],'std'])
    csvwrite(out/'benchmark_results.csv',summary);csvwrite(out/'coverage_curve.csv',curves)
    np.savez_compressed(out/'native_harmful_group_counts.npz',counts=native_harm_group,methods=np.asarray(METHODS),seeds=np.asarray(SEEDS))
    np.savez_compressed(out/'bootstrap_draws.npz',group_weights=w,**bootstrap_arrays)
    # Compact paired group-bootstrap contrast, without changing either method.
    diff=bootstrap_arrays['V3_Full__pooled']-bootstrap_arrays['V1_Average__pooled'];dl,dh=ci(diff)
    contrast=dict(contrast='V3_Full minus V1_Average',pooled_macro_f1_delta=float(seeds[seeds.method=='V3_Full'].pooled_macro_f1.mean()-seeds[seeds.method=='V1_Average'].pooled_macro_f1.mean()),ci95_low=float(dl),ci95_high=float(dh))
    dump(out/'paired_fusion_vs_average.json',contrast)
    figures(out,pd.DataFrame(summary),pd.DataFrame(curves))
    write_report(out,summary,contrast)
    dump(out/'validation.json',dict(status='complete',completed_seed_fold_cells=200,seeds=SEEDS,methods=METHODS,condition_count=len(CONDITIONS),frozen_test_files_verified=1400,method_condition_cells=len(allcells),all_oof_group_disjoint=True,test_indices_aligned=True,primary_f1_fixed_labels=[0,1],tls_common_coverage_marked_infeasible=True,source_features_sha256=sha(out/'features.npz'),scoring_seconds=time.perf_counter()-started))

def figures(out,summary,curve):
    # Visual style adapted from scientific-figure-making / figures4papers (CC-BY-NC-4.0).
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
    colors=['#777777','#333333','#AAAAAA','#3775BA','#42949E','#B64342','#9A4D8E','#C57A24','#32884A','#6B60A8','#CC6677']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True,sharey=True)
    panes=[['V0_Stats','V0_Temporal','V1_Average','V2_Yager','V3_Full','V3_OnDemand'],['V1_Average','V3_Full','V4_Logistic','V4_MLP','V5_Gating','V6_EDL']]
    for ax,methods in zip(axes,panes):
        for method in methods:
            c=curve[curve.method==method];i=METHODS.index(method)
            ax.plot(c.coverage,c.selective_macro_f1_mean,label=method,color=colors[i],linewidth=1.6,marker='o',markersize=3,linestyle='--' if method in ('V2_Yager','V3_OnDemand') else '-')
        ax.axvline(.2337,color='.75',linestyle=':',linewidth=1);ax.set_xlabel('Retained fraction of all test rows');ax.set_ylim(0,1.02);ax.grid(axis='y',alpha=.18);ax.legend(fontsize=7,loc='lower right')
    axes[0].set_ylabel('Selective Macro-F1 (pooled, seed mean)');axes[0].set_title('Single views and evidence rules');axes[1].set_title('Learned decision methods')
    fig.suptitle('Common-coverage ranking; TLS-only cannot reach 0.2337 coverage',fontsize=10)
    fig.text(.01,.01,'Fixed binary labels: an accepted subset containing only one class can have F1 = 0.5 with zero errors. See class counts and accuracy in CSV.',fontsize=7)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('pdf','png'):fig.savefig(out/f'coverage_vs_selective_macro_f1.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.6));xx=np.arange(len(summary));bars=ax.bar(xx,summary.calls_per_sample,color=colors,edgecolor='white',linewidth=.5)
    for bar,value in zip(bars,summary.calls_per_sample):ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.3f}',ha='center',fontsize=8)
    ax.set_xticks(xx,summary.method,rotation=40,ha='right');ax.set_ylabel('Required specialist calls / sample');ax.set_ylim(0,max(summary.calls_per_sample)+.3);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Frozen-output call accounting: static methods and explicit on-demand variant')
    fig.text(.01,.01,'V6: one EDL network call; same feature extraction still required. Calls are replay accounting, not measured deployment speedup.',fontsize=7)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('pdf','png'):fig.savefig(out/f'calls_per_sample.{ext}',dpi=300)
    plt.close(fig)
    dump(out/'figure_sources.json',dict(coverage_figure='coverage_curve.csv',calls_figure='benchmark_results.csv',dpi=300,style_attribution='scientific-figure-making; ChenLiu-1996/figures4papers, CC-BY-NC-4.0',uncertainty='group-bootstrap intervals are in CSV; no intervals invented for figure'))

def write_report(out,rows,contrast):
    lines=['# USTC authoritative fusion benchmark v1','',
        'Completed all 20 whole-group-held-out folds × 10 seeds. These are newly trained 11-D Stats / 30-D Temporal results; no historical score is a target or comparator. Full details and commands: [PROTOCOL.md](PROTOCOL.md); run log: [RUNS.md](RUNS.md).','',
        '**Metric warning:** each test fold contains one class. Fixed-label binary fold Macro-F1 therefore tops out at 0.5. The requested 200-cell fold mean and its group-bootstrap CI are reported first. The second table reports the more interpretable two-class Macro-F1 after pooling 40,000 held-out predictions per seed. Neither is substituted for the other. Std uses ddof=1.','',
        '| Method | Fold Macro-F1 mean±std | 95% group CI | Accuracy mean±std | Trained layer? | Parameters | Native abstention? | Calls/sample |','|---|---:|---|---:|:---:|---:|:---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {pm(r['macro_f1_fold_mean'],r['macro_f1_fold_std'])} | {interval(r['macro_f1_ci95_low'],r['macro_f1_ci95_high'])} | {pm(r['accuracy_fold_mean'],r['accuracy_fold_std'])} | {r['trained_layer']} | {r['params']} | {r['native_abstention']} | {r['calls_per_sample']:.4f} |")
    lines+=['','| Method | Pooled Macro-F1, 10-seed mean±std | 95% group CI | Selective F1 @ 0.2337 | Selective F1 @ 1.0 | Native coverage | Decision ms/sample |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {pm(r['macro_f1_pooled_seed_mean'],r['macro_f1_pooled_seed_std'])} | {interval(r['pooled_macro_f1_ci95_low'],r['pooled_macro_f1_ci95_high'])} | {fmt(r['selective_macro_f1_at_02337'])} | {fmt(r['selective_macro_f1_at_10000'])} | {r['native_coverage_mean']:.4f} | {r['decision_latency_ms_mean']:.6f} |")
    lines+=['','TLS-only is evaluated on its applicable subset, not imputed to full coverage: common-cohort primary metrics and infeasible coverage points are N/A. Available-subset F1 and coverage are retained in the CSV. V6 is a separately trained end-to-end EDL detector; its parameter count includes the entire network. V3_OnDemand changes acquisition policy and has its own performance row.','',
        'Full-coverage scores are forced binary projections. Selective curves rank confidence at a specified retained fraction, including predictions the native policy might reject. Native four-state coverage/unknown/suspicious are separately recorded. Exact target 0.2337 retains 9,348 of 40,000 rows per seed; ranking uses no test labels. Fixed binary Macro-F1 also applies to selected subsets: if only one class is retained, a perfect subset scores 0.5. Selected benign/malicious counts and selective accuracy are provided in the curve CSVs; low-coverage F1 must be interpreted together with this class composition.','',
        '## Fixed perturbations and harmful flips','',
        'Six preregistered view/feature occlusions are applied to each frozen detector. The operating threshold comes from clean validation confidence at nominal 80% coverage and is unchanged after occlusion. A harmful flip must be accepted both before and after, correct before, and wrong after. The first table is the confidence-threshold/forced-binary diagnostic; the following native table additionally requires a benign/malicious native verdict at both ends, so suspicious/unknown cannot be counted as wrong binary decisions. Lower flip counts alone do not establish robustness: coverage and selective error are also shown. Counts sum all six conditions and ten seeds, with each repeated-seed prediction treated as an evaluation event.','',
        '| Method | Harmful flips, total | Per-seed mean±std | Perturbed fixed coverage | Perturbed fixed selective error | Native unknown (clean) | Native suspicious (clean) |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['harmful_flips_total']} | {r['harmful_flips_per_seed_mean']:.1f}±{r['harmful_flips_per_seed_std']:.1f} | {r['perturbed_fixed_coverage_mean']:.4f} | {fmt(r['perturbed_fixed_selective_error_mean'])} | {r['unknown_rate']:.4f} | {r['suspicious_rate']:.4f} |")
    lines+=['','Native binary decisions **and the same frozen confidence threshold**:','','| Method | Native harmful flips, total | Per-seed mean±std | 95% group CI of per-seed count | Perturbed accepted coverage | Selective error |','|---|---:|---:|---|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['native_fixed_harmful_flips_total']} | {r['native_fixed_harmful_flips_per_seed_mean']:.1f}±{r['native_fixed_harmful_flips_per_seed_std']:.1f} | [{r['native_fixed_harmful_flips_per_seed_ci95_low']:.1f}, {r['native_fixed_harmful_flips_per_seed_ci95_high']:.1f}] | {r['native_fixed_perturbed_coverage_mean']:.4f} | {fmt(r['native_fixed_perturbed_selective_error_mean'])} |")
    lines+=['','All condition/seed observations, including baseline-favoring outcomes, are in `perturbation_results.csv` and `per_fold_results.csv`. No result is omitted based on direction.','',
        '## Evidence and limits','',
        f"V3 Full minus V1 Average pooled Macro-F1: **{contrast['pooled_macro_f1_delta']:+.6f}**, paired stratified group-bootstrap 95% CI **[{contrast['ci95_low']:+.6f}, {contrast['ci95_high']:+.6f}]**. This is recorded as observed; neither equality nor superiority was imposed.",
        '',
        'Calls/sample is an auditable replay count of required applicable specialists, not a live deployment measurement. V6 uses zero specialist calls and one EDL network call; it still consumes the same feature groups. Reported latency measures the decision layer only, in warmed validation batches of 128 (three repetitions), while other fold workers may contend for CPU. HGB/OOD component timings are retained in per-fold results. Feature extraction, TLS rule execution, disk access, networking and end-to-end deployment latency are not included. No end-to-end speedup claim is supported by these timing numbers.','',
        'The group bootstrap retains all ten seeds for each resampled group. It estimates uncertainty across the twenty observed held-out groups, conditional on this dataset/training design; it is not an IID-row confidence interval or ten independent newly sampled datasets. Uncertainty for selective curves conditions on the confidence-ranked subset. Native abstention and common-coverage rankings answer different questions.','',
        '## Artifacts','',
        '- `benchmark_results.csv`: complete method-level numeric table, parameters, coverage, call and timing fields.','- `per_fold_results.csv`, `per_seed_results.csv`, `perturbation_results.csv`: all observations and diagnostics.','- `coverage_curve.csv`, `selective_per_seed.csv`, `bootstrap_draws.npz`: plot and interval evidence.','- `splits/`, `features.npz`, `frozen_predictions/`, `fold_runs/`: exact inputs, shared outputs, models and predictions.','- `audit_checks.csv`, `verified_artifact_hashes.csv`, `validation.json`: integrity and completeness checks.','- `coverage_vs_selective_macro_f1.pdf/.png`, `calls_per_sample.pdf/.png`: vector PDFs and 300-dpi raster figures.','',
        '![Selective performance](coverage_vs_selective_macro_f1.png)','', '![Calls per sample](calls_per_sample.png)','']
    by={r['method']:r for r in rows};full=by['V3_Full'];avg=by['V1_Average'];ond=by['V3_OnDemand'];stats=by['V0_Stats'];temporal=by['V0_Temporal']
    conclusions=['## Interpretation of this run','',
        f"Among the fixed single views, Stats pooled Macro-F1 is {stats['macro_f1_pooled_seed_mean']:.4f} and Temporal is {temporal['macro_f1_pooled_seed_mean']:.4f}. Their zero seed standard deviations reflect deterministic HGB fitting on identical outer splits, not missing seed runs. Neural initialization and OOF group assignments vary by seed; the OOD estimator also varies by seed.",
        f"V3 has {full['harmful_flips_total']:,} confidence-only forced-projection flips versus {avg['harmful_flips_total']:,} for averaging. Under the native binary-decision requirement, V3 has {full['native_fixed_harmful_flips_total']:,} flips versus {avg['native_fixed_harmful_flips_total']:,}, with perturbed accepted coverages {full['native_fixed_perturbed_coverage_mean']:.4f} versus {avg['native_fixed_perturbed_coverage_mean']:.4f}. These distinct operating modes are both retained, and neither the confidence threshold nor the native decision rule is retuned after observing the comparison. A blanket robustness advantage is not established by this run.",
        f"The explicit on-demand policy requires {ond['calls_per_sample']:.4f} calls/sample versus {full['calls_per_sample']:.4f} for static full fusion, a {100*(1-ond['calls_per_sample']/full['calls_per_sample']):.3f}% reduction. This is a small reduction; it does not support a substantial cost-saving claim.",
        '']
    if 'feature_to_decision_latency_ms_mean' in rows[0]:
        conclusions+=['## Actual serial inference latency','',
            'Supplemental measured feature-matrix-to-decision latency includes required HGB, TLS rules, OOD gates and neural/decision execution. Each of 2,200 method×seed×fold validation checks matched frozen decisions. Three warmed repetitions, batch 128, one worker; raw packet feature extraction, model loading and I/O are excluded. Details: [RUNTIME_LATENCY.md](RUNTIME_LATENCY.md).','',
            '| Method | Feature-to-decision ms/sample, mean±std |','|---|---:|']
        for r in rows:conclusions.append(f"| {r['method']} | {r['feature_to_decision_latency_ms_mean']:.6f}±{r['feature_to_decision_latency_ms_std']:.6f} |")
        conclusions+=['']
    lines+=conclusions
    (out/'BENCHMARK.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':summarize(Path(sys.argv[1]))
