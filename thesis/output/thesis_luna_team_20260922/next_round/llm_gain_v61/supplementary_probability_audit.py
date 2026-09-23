"""Post-run descriptive probability checks, valid atomic responses only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
a=json.loads((HERE/'results/api_attempts.json').read_text())
q=pd.read_csv(HERE/'results/scorer_only_queries.csv')
cards=json.loads((HERE/'results/training_cards.json').read_text())
rows=[]
for rec in a['requests']:
    if rec['arm']!='atomic' or rec['status']!='valid':continue
    v=q[q.fold==rec['fold']].sort_values('case_id');y=v.y.to_numpy()
    h0=v.p_temporal.to_numpy()>=.5;h1=(v.p_temporal.to_numpy()+v.p_stats_offline_target_only.to_numpy())/2>=.5
    target=(h0!=y).astype(int)-(h1!=y).astype(int);p=np.array([rec['probabilities'][x] for x in v.case_id])
    t=np.eye(3)[target+1]
    rows.append({'fold':rec['fold'],'rows':len(v),'target_minus_zero_plus_counts':[int((target==x).sum()) for x in [-1,0,1]],'multiclass_brier':float(((p-t)**2).sum(axis=1).mean()),'nll_natural_log_clip_1e_15':float(-np.log(np.clip(p[np.arange(len(v)),target+1],1e-15,1)).mean()),'maximum_probability_sum_error':float(np.max(abs(p.sum(axis=1)-1)))})
out={'status':'completed_descriptive_supplement','post_run_analysis':True,'valid_atomic_only':True,'probability_class_order':[-1,0,1],'rows':rows,'limitation':'Surviving 2 of 5 batches only; timeout exclusion is informative. These scores do not compare all arms or establish general calibration. No probabilities imputed for failed requests.'}
(HERE/'results/valid_atomic_probability_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out))
