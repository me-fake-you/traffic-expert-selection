"""Aggregate saved E1 predictions; fixed group-conditional intervals and figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_p1 import OUT, CFG, dump, sha

RUN=OUT/'e1/run_001'
COUNTS=['tn','fp','fn','tp','corrected_FN','corrected_FP','introduced_FN','introduced_FP']

def from_counts(a):
    tn,fp,fn,tp=(a[...,i] for i in range(4))
    f1=np.divide(tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)+np.divide(tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)>0)
    return {'macro_f1':f1,'malicious_recall':tp/np.maximum(tp+fn,1),'false_positive_rate':fp/np.maximum(fp+tn,1),'corrected':a[...,4]+a[...,5],'introduced':a[...,6]+a[...,7]}

def main():
    cfg=json.loads(CFG.read_text(encoding='utf-8'))
    assert json.loads((RUN/'completion.json').read_text(encoding='utf-8'))['status']=='COMPLETED_CONDITIONAL_E1'
    # These are deterministic derived summaries/figures; original run outputs remain read-only.
    dest=OUT/'results';dest.mkdir(exist_ok=True)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    g=pd.read_csv(RUN/'group_metrics.csv');f=pd.read_csv(RUN/'fold_metrics.csv')
    index=json.loads((RUN/'prediction_index.json').read_text(encoding='utf-8'))
    groups=sorted(g.group.unique())
    assert len(groups)==20
    arrays={name:part.set_index('group').loc[groups,COUNTS].to_numpy() for name,part in g.groupby('prediction_column')}
    aggregate=[]
    for name,a in arrays.items():
        total=a.sum(axis=0); values={k:float(v) for k,v in from_counts(total).items()}
        matching=f[f.prediction_column==name]
        assert matching.rows.sum()==40000
        aggregate.append({'prediction_column':name,**index[name],'rows':40000,**values,**dict(zip(COUNTS,total.astype(int))),'evidence_calls':float(np.average(matching.evidence_calls,weights=matching.rows)),'switch_rate':float(np.average(matching.switch_rate,weights=matching.rows))})
    agg=pd.DataFrame(aggregate);agg.to_csv(dest/'aggregate_metrics.csv',index=False)
    labels=g[['group','label']].drop_duplicates().set_index('group').loc[groups,'label'].to_numpy()
    rng=np.random.default_rng(cfg['conditional_intervals']['seed']);nboot=cfg['conditional_intervals']['replicates']
    boot=np.concatenate([rng.choice(np.flatnonzero(labels==c),size=(nboot,int((labels==c).sum())),replace=True) for c in (0,1)],axis=1)
    comparisons=[]
    def compare(left,right,kind,model,seed,mode):
        av,bv=arrays[left],arrays[right]
        observed_a,observed_b=from_counts(av.sum(axis=0)),from_counts(bv.sum(axis=0))
        ba,bb=from_counts(av[boot].sum(axis=1)),from_counts(bv[boot].sum(axis=1))
        for metric in ('macro_f1','malicious_recall','false_positive_rate','corrected','introduced'):
            lo,hi=np.quantile(ba[metric]-bb[metric],[.025,.975])
            comparisons.append({'contrast':kind,'kind':model,'subset_seed':seed,'operating_point':mode,'left':left,'right':right,'metric':metric,'difference':float(observed_a[metric]-observed_b[metric]),'conditional_low':float(lo),'conditional_high':float(hi),'interval_scope':'fixed-prediction class-stratified family-group bootstrap; not independent activities or fit uncertainty'})
    for mode in ('primary','selected'):
        for kind in ('hgb','logistic'):
            for seed in cfg['subset_seeds']:
                compare(f'row100_{cfg["subset_seeds"][0]}_{kind}_{mode}',f'row25_{seed}_{kind}_{mode}','row100_minus_row25',kind,seed,mode)
                compare(f'source_dispersed_{seed}_{kind}_{mode}',f'source_concentrated_{seed}_{kind}_{mode}','dispersed_minus_concentrated',kind,seed,mode)
    for row in aggregate:
        if row['operating_point']!='reference':compare(row['prediction_column'],'same_trigger_fixed','learned_minus_fixed',row['kind'],row['subset_seed'],row['operating_point'])
    pd.DataFrame(comparisons).to_csv(dest/'paired_conditional_intervals.csv',index=False)
    # Every fold is retained; no selection of the best family or seed.
    f['introduced']=f.introduced_FN+f.introduced_FP;f['corrected']=f.corrected_FN+f.corrected_FP
    contrast_folds=[]
    for mode in ('primary','selected'):
        for kind in ('hgb','logistic'):
            for seed in cfg['subset_seeds']:
                for fold in cfg['outer_folds']:
                    fold_rows=f[f.outer_fold==fold].set_index('prediction_column')
                    for contrast,left,right in [('row100_minus_row25',f'row100_{cfg["subset_seeds"][0]}_{kind}_{mode}',f'row25_{seed}_{kind}_{mode}'),('dispersed_minus_concentrated',f'source_dispersed_{seed}_{kind}_{mode}',f'source_concentrated_{seed}_{kind}_{mode}')]:
                        contrast_folds.append({'outer_fold':fold,'kind':kind,'subset_seed':seed,'operating_point':mode,'contrast':contrast,**{m:float(fold_rows.loc[left,m]-fold_rows.loc[right,m]) for m in ('macro_f1','introduced','corrected','malicious_recall','false_positive_rate')}})
    pd.DataFrame(contrast_folds).to_csv(dest/'per_fold_contrasts.csv',index=False)
    primary=agg[(agg.operating_point=='primary')|(agg.operating_point=='reference')]
    summary=primary.groupby(['condition','kind'],sort=False).agg(subset_fits=('macro_f1','size'),macro_f1_mean=('macro_f1','mean'),macro_f1_min=('macro_f1','min'),macro_f1_max=('macro_f1','max'),malicious_recall_mean=('malicious_recall','mean'),FPR_mean=('false_positive_rate','mean'),corrected_mean=('corrected','mean'),introduced_mean=('introduced','mean'),calls_mean=('evidence_calls','mean'),switch_rate_mean=('switch_rate','mean')).reset_index()
    summary.to_csv(dest/'primary_summary.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','lines.linewidth':1.2})
    colors={'hgb':'#0F4D92','logistic':'#B64342'}
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.55),layout='constrained')
    for kind in ('hgb','logistic'):
        for ax,metric,scale in [(axes[0],'macro_f1',100),(axes[1],'introduced',1)]:
            ys=[]
            for fraction in (25,50,100):
                rows=primary[(primary.kind==kind)&(primary.condition==f'row{fraction}')]
                vals=rows[metric].to_numpy()*scale;ys.append(vals.mean())
                ax.scatter(np.repeat(fraction,len(vals)),vals,s=16,color=colors[kind],alpha=.65,marker='o' if kind=='hgb' else 's',zorder=3)
            ax.plot([25,50,100],ys,color=colors[kind],label=kind.upper() if kind=='hgb' else 'Logistic')
    for name,color,style,label in [('first_only','#505050',':','First expert'),('same_trigger_fixed','#999999','--','Same-trigger fixed')]:
        row=agg[agg.prediction_column==name].iloc[0]
        axes[0].axhline(row.macro_f1*100,color=color,ls=style,lw=.9,label=label)
        axes[1].axhline(row.introduced,color=color,ls=style,lw=.9)
    axes[0].set_ylabel('Macro-F1 (%)');axes[1].set_ylabel('Introduced errors (40,000 segments)')
    axes[0].set_title('(a) Quality');axes[1].set_title('(b) Switching damage')
    for ax in axes:
        ax.set_xticks([25,50,100]);ax.set_xlabel('OOF cell-wise row budget (%)');ax.grid(axis='y',alpha=.15)
    axes[0].legend(loc='best',frameon=True,facecolor='white',edgecolor='white',framealpha=1,ncol=1)
    for ext in ('pdf','svg','png'):fig.savefig(figs/f'e1_row_budget.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
    primary[primary.condition.isin(['row25','row50','row100'])|primary.prediction_column.isin(['first_only','same_trigger_fixed'])].to_csv(figs/'e1_row_budget_data.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),layout='constrained')
    labels_plot=[]
    for i,(kind,condition) in enumerate(itertools_pairs()):
        rows=primary[(primary.kind==kind)&(primary.condition==condition)]
        labels_plot.append(('HGB' if kind=='hgb' else 'Logistic')+'\n'+('Concentrated' if condition=='source_concentrated' else 'Dispersed'))
        for ax,prefix in [(axes[0],'corrected'),(axes[1],'introduced')]:
            fn=rows[prefix+'_FN'].mean();fp=rows[prefix+'_FP'].mean()
            ax.bar(i,fn,color='#0F4D92',width=.6,label='FN component' if i==0 else None)
            ax.bar(i,fp,bottom=fn,color='#D9DFE7',edgecolor='#555555',linewidth=.5,hatch='///',width=.6,label='FP component' if i==0 else None)
            ax.scatter(i+np.linspace(-.15,.15,len(rows)),rows[prefix],s=12,color='#B64342',zorder=4)
    for ax,title in zip(axes,['(a) Corrected first-expert errors','(b) Introduced errors']):
        ax.set_xticks(range(4),labels_plot);ax.set_ylabel('Count per 40,000 segments');ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False,loc='upper left')
    for ext in ('pdf','svg','png'):fig.savefig(figs/f'e1_source_corrections.{ext}',dpi=320,bbox_inches='tight')
    plt.close(fig)
    primary[primary.condition.str.startswith('source_')].to_csv(figs/'e1_source_corrections_data.csv',index=False)
    mapping={'e1_row_budget':'Aggregate real saved E1 predictions, primary q=0.5, all three prespecified subsets; row100 fit once. Lines connect means, points are subset results, not independent test repetitions. All source groups retained, rounding changes finite-cell proportions slightly.','e1_source_corrections':'Within-fold matched rows, direction, correctness target and inner-model mixture; observed capture/family composition changes. Bars are three-subset means; dots show all totals. FN/FP components relative to frozen first expert. No independent-activity causal claim.','input_files':{str(p):sha(p) for p in (RUN/'group_metrics.csv',RUN/'fold_metrics.csv',CFG)},'style_attribution':'Adapted publication-style guidance from ChenLiu-1996/figures4papers, CC-BY-NC-4.0; plots use local empirical data and custom matplotlib code.'}
    dump(figs/'source_map.json',mapping)
    print(summary.to_string(index=False),flush=True)
    selected=pd.DataFrame(comparisons)
    print('\nPrimary planned differences:',flush=True)
    print(selected[(selected.operating_point=='primary')&(selected.contrast!='learned_minus_fixed')&selected.metric.isin(['macro_f1','corrected','introduced'])][['contrast','kind','subset_seed','metric','difference','conditional_low','conditional_high']].to_string(index=False),flush=True)

def itertools_pairs():
    return [(k,c) for k in ('hgb','logistic') for c in ('source_concentrated','source_dispersed')]

if __name__=='__main__':main()
