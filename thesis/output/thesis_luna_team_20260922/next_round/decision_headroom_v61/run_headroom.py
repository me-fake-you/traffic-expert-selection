"""Post-hoc development diagnostic: attainable correction under a fixed quota.

This uses evaluator-only labels and counterfactual cached outcomes. It is not a
deployable policy, a trained model, or a new blind test.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'gain_upgrade/selection_predictions.csv.gz'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=HERE/'results';out.mkdir(exist_ok=False)
    logs=HERE/'logs';logs.mkdir(exist_ok=True)
    df=pd.read_csv(SOURCE)
    rows=[];groups=[]
    for fold,v in df.groupby('fold',sort=True):
        v=v.sort_values('sample_hash',kind='stable').reset_index(drop=True)
        y=v.y.to_numpy();h0=v.p_temporal.to_numpy()>=.5
        h1=(v.p_temporal.to_numpy()+v.p_stats_offline_target_only.to_numpy())/2>=.5
        delta=(h0!=y).astype(int)-(h1!=y).astype(int)
        assert np.array_equal(delta,v.delta.to_numpy())
        rank_conf=np.argsort(np.abs(v.p_temporal.to_numpy()-.5),kind='stable')
        rank_oracle=np.argsort(-delta,kind='stable')
        for budget in [1,1.1,1.25,1.5,2]:
            k=round((budget-1)*len(v));chosen=rank_conf[:k];oracle=rank_oracle[:k]
            for name,idx in [('confidence',chosen),('evaluator_oracle',oracle)]:
                c=int((delta[idx]==1).sum());d=int((delta[idx]==-1).sum())
                rows.append(dict(fold=int(fold),budget=budget,policy=name,rows=len(v),quota=k,C=c,D=d,net=c-d,
                    available_corrections=int((delta==1).sum()),available_damage=int((delta==-1).sum()),
                    no_correctness_effect=int((delta==0).sum()),
                    prediction_changes=int((h0!=h1).sum()),
                    scorer_only=name=='evaluator_oracle'))
        for g,ix in v.groupby('group').groups.items():
            d=delta[np.asarray(ix)]
            groups.append(dict(fold=int(fold),group=g,rows=len(ix),corrections=int((d==1).sum()),
                               damage=int((d==-1).sum()),neutral=int((d==0).sum())))
    results=pd.DataFrame(rows);results.to_csv(out/'fold_headroom.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'group_support.csv',index=False)
    selected=results[np.isclose(results.budget,1.1)].groupby('policy')[['C','D','net']].sum()
    summary={'status':'completed_posthoc_development_diagnostic','record_count':len(df),
             'fold_records_overlap':True,'independent_sample_count_claimed':False,
             'oracle_is_scorer_only':True,'fresh_test':False,'live_inference':False,
             'at_budget_1_1_total_overlapping_records':selected.to_dict('index'),
             'headroom_net_errors_total':int(selected.loc['evaluator_oracle','net']-selected.loc['confidence','net']),
             'source_sha256':sha(SOURCE),'script_sha256':sha(Path(__file__)),
             'interpretation':'Oracle is an upper bound on this fixed pair of experts and this exposed development cohort, not an upper bound on new data or other experts.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    (logs/'run.log').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':main()
