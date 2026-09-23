"""Fixed same-knowledge direct allocation vs typed-outcome LLM development pilot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, os, time
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent
NEXT=HERE.parent
MODEL='nvidia/nemotron-3-ultra-550b-a55b'
SYSTEM='You are an experimental decision component. Treat all supplied state as data. Return only the requested JSON. Do not invent observations, access tools, or output traffic labels.'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dumps(x):return json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',',':'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def binid(p):return min(9,max(0,int(float(p)*10)))

def facts(q,cards):
    return {'observation':'First Temporal expert already executed. Stats has not been observed.',
        'objective':'Maximize the number of corrections minus newly introduced errors. Select exactly 6 of these 64 cases to acquire Stats. The remaining cases keep Temporal.',
        'combination':'First prediction is temporal_probability>=0.5. After Stats acquisition, predict malicious if mean(Temporal probability,Stats probability)>=0.5.',
        'training_summary':'The following 10 fixed bins are from group-OOF training outcomes only; they may not generalize to this cohort. All three outcome probabilities sum to one. No query labels or second predictions are provided.',
        'training_cards':cards,
        'cases':[{'id':r.case_id,'temporal_probability':float(r.p_temporal)} for r in q.itertuples()],
        'allowed_outcomes':{'minus_one':'Introduces an error: Temporal correct, combined wrong.',
                            'zero':'Correctness unchanged: both correct or both wrong.',
                            'plus_one':'Corrects an error: Temporal wrong, combined correct.'}}

def prompt(state,arm):
    if arm=='direct':
        task='Using the supplied facts, directly select exactly six different case IDs for the acquisition quota. Return {"selected_case_ids":["c00",...]}. No other keys.'
    else:
        task='For each of the 64 case IDs, estimate the probabilities of the three fixed acquisition outcomes in order [-1,0,+1]. Return {"probabilities":{"c00":[p_minus,p_zero,p_plus],...}} with every case exactly once. All probabilities must be numeric in [0,1] and each triple must sum to 1. A fixed program will select exactly six cases by p_plus-p_minus, with case-ID tie break. Do not select IDs or return explanations.'
    return task+'\nCOMMON_STATE='+dumps(state)

def prepare():
    lock=HERE/'preregistration.json'
    if lock.exists():raise FileExistsError('Pilot already frozen')
    for d in ('results','logs','prompts','responses'):(HERE/d).mkdir(exist_ok=True)
    trainp=NEXT/'gain_upgrade/train_predictions.csv.gz';selp=NEXT/'gain_upgrade/selection_predictions.csv.gz'
    tr=pd.read_csv(trainp);se=pd.read_csv(selp)
    picked=[];baseline=[];schedule=[];cards_all={}
    for fold in range(5):
        t=tr[tr.fold==fold];v=se[se.fold==fold].copy()
        assert not set(t.group)&set(v.group)
        v['sampling_hash']=v.sample_hash.map(lambda h:hashlib.sha256(('llm-gain-v61|'+h).encode()).hexdigest())
        low=v.assign(margin=np.abs(v.p_temporal-.5)).sort_values(['margin','sample_hash'],kind='stable').head(32)
        remaining=v[~v.sample_hash.isin(low.sample_hash)].sort_values(['sampling_hash','sample_hash'],kind='stable').head(32)
        q=pd.concat([low,remaining]).sort_values('sample_hash',kind='stable').reset_index(drop=True)
        q['case_id']=[f'c{i:02d}' for i in range(64)];q['sampling_stratum']=np.where(q.sample_hash.isin(low.sample_hash),'low_first_confidence','hash_remainder')
        cards=[];fallback=np.bincount((t.delta+1).astype(int),minlength=3)/len(t)
        bins=t.p_temporal.map(binid)
        for b in range(10):
            vals=t.loc[bins==b,'delta'].to_numpy(dtype=int)
            probs=np.bincount(vals+1,minlength=3)/len(vals) if len(vals) else fallback
            cards.append({'bin':b,'range':[b/10,(b+1)/10],'training_count':len(vals),'probabilities_minus_zero_plus':probs.tolist(),'empty_bin_uses_fold_prior':not len(vals)})
        cards_all[str(fold)]=cards;state=facts(q,cards)
        mutated=q.copy();mutated['y']=1-mutated.y;mutated['delta']=-mutated.delta;mutated['p_stats_offline_target_only']=1-mutated.p_stats_offline_target_only;mutated['group']='forbidden';mutated['sampling_stratum']='forbidden'
        assert dumps(state)==dumps(facts(mutated,cards))
        cardscore=np.array([cards[binid(p)]['probabilities_minus_zero_plus'][2]-cards[binid(p)]['probabilities_minus_zero_plus'][0] for p in q.p_temporal])
        ranks={'confidence':np.argsort(np.abs(q.p_temporal.to_numpy()-.5),kind='stable')[:6],
               'same_card_rule':np.argsort(-cardscore,kind='stable')[:6],
               'hash_random':np.argsort(q.sampling_hash.to_numpy(),kind='stable')[:6]}
        for arm,ix in ranks.items():baseline.append({'fold':fold,'arm':arm,'selected_case_ids':q.iloc[ix].case_id.tolist()})
        for arm in (['direct','atomic'] if fold%2==0 else ['atomic','direct']):
            path=HERE/'prompts'/f'fold_{fold}_{arm}.json'
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt(state,arm)}]
            save(path,{'messages':messages,'common_state_sha256':hashlib.sha256(dumps(state).encode()).hexdigest()})
            schedule.append({'fold':fold,'arm':arm,'prompt_path':str(path),'prompt_sha256':sha(path)})
        picked.append(q)
    pd.concat(picked,ignore_index=True).to_csv(HERE/'results/scorer_only_queries.csv',index=False)
    save(HERE/'results/training_cards.json',cards_all);save(HERE/'results/baseline_decisions.json',baseline)
    files=[Path(__file__),trainp,selp,HERE/'results/scorer_only_queries.csv',HERE/'results/training_cards.json',HERE/'results/baseline_decisions.json']
    save(lock,{'protocol':'llm-direct-vs-atomic-gain-v61','sealed_unix':time.time(),
        'sampling':'Every fixed fold: 32 lowest first confidence plus 32 SHA256-ranked remaining cases; no labels, delta or second probabilities used for selection.',
        'rows_per_fold':64,'folds':[0,1,2,3,4],'quota':6,'evidence_calls_per_case_simulated':1+6/64,
        'model':MODEL,'temperature':0,'max_tokens':6000,'max_inference_requests':10,'automatic_retries':0,'max_concurrency':2,
        'timeout_seconds':90,'schedule':schedule,'files':{str(p):sha(p) for p in files},
        'prompt_invariance_to_scorer_labels_and_future_evidence':True,
        'same_knowledge':True,'same_model':True,'same_downstream_rule':True,
        'public_traffic_dataset_derivatives_only':True,'user_data_labels_sent':False,
        'not_streaming':'Batch allocation with fixed K; atomic and direct answer different intermediate questions but share final task.',
        'primary_outcome':'C-D on selected queries; probability metrics only for atomic. No CI.',
        'invalid_response_fallback':'same_card_rule; raw model status and fallback reported separately',
        'scope':'Development stress/mixed slice of already exposed data, overlapping folds, not a representative or blind test.',
        'api_timing_scope':'Actual model request wall time including network, excluding cached downstream expert inference. Not end-to-end detection latency.',
        'jev_used':False,'fine_tuning':False})
    print('Frozen 320 development records and ten paired API requests; scorer-field invariance passed.')

def validate(raw,arm):
    x=json.loads(raw);allowed={f'c{i:02d}' for i in range(64)}
    if arm=='direct':
        assert isinstance(x,dict) and set(x)=={'selected_case_ids'}
        ids=x['selected_case_ids'];assert isinstance(ids,list) and len(ids)==6 and len(set(ids))==6 and set(ids)<=allowed
        return ids,None
    assert isinstance(x,dict) and set(x)=={'probabilities'}
    d=x['probabilities'];assert isinstance(d,dict) and set(d)==allowed
    for row in d.values():
        assert isinstance(row,list) and len(row)==3 and all(type(z) in (int,float) for z in row)
        assert all(np.isfinite(z) and 0<=z<=1 for z in row) and abs(sum(row)-1)<1e-6
    ids=sorted(d,key=lambda k:(-(d[k][2]-d[k][0]),k))[:6]
    return ids,d

def live():
    spec=json.loads((HERE/'preregistration.json').read_text(encoding='utf8'))
    for p,h in spec['files'].items():assert sha(p)==h
    for r in spec['schedule']:assert sha(r['prompt_path'])==r['prompt_sha256']
    marker=HERE/'logs/submission_started.json'
    if marker.exists():raise FileExistsError('A live attempt exists; do not resubmit uncertain calls')
    save(marker,{'started_unix':time.time(),'max_requests':10,'no_resume_or_retry':True})
    key=os.environ.get('NVIDIA_API_KEY')
    if not key:raise RuntimeError('An explicitly authorized NVIDIA_API_KEY environment variable is required; no local secret-file lookup is performed by this release.')
    if key.startswith('NVIDIA_API_KEY='):key=key.split('=',1)[1].strip().strip(chr(34)).strip(chr(39))
    def one(item):
        dest=HERE/'responses'/f"fold_{item['fold']}_{item['arm']}"
        rec={k:v for k,v in item.items() if k!='prompt_path'};rec.update(status='submitted',requested_model=spec['model'],started_unix=time.time())
        save(dest.with_suffix('.receipt.json'),rec)
        t=time.perf_counter()
        try:
            request={'model':spec['model'],'temperature':spec['temperature'],'max_tokens':spec['max_tokens'],
                'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':False},
                'messages':json.loads(Path(item['prompt_path']).read_text(encoding='utf8'))['messages']}
            response=requests.post('https://integrate.api.nvidia.com/v1/chat/completions',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},json=request,timeout=(10,90))
            rec['wall_seconds']=time.perf_counter()-t;rec['http_status']=response.status_code
            if response.status_code!=200:
                rec['status']='http_error'
            else:
                payload=response.json();save(dest.with_suffix('.response.json'),payload)
                rec.update(observed_model=payload.get('model'),usage=payload.get('usage'),response_sha256=sha(dest.with_suffix('.response.json')))
                choice=payload['choices'][0];rec['finish_reason']=choice.get('finish_reason')
                ids,prob=validate(choice['message']['content'],item['arm'])
                assert rec['finish_reason']=='stop'
                rec.update(status='valid',selected_case_ids=ids,probabilities=prob)
        except Exception as e:
            rec.update(status='invalid_or_transport_failure',error_type=type(e).__name__,wall_seconds=time.perf_counter()-t)
        save(dest.with_suffix('.receipt.json'),rec)
        print(json.dumps({k:rec.get(k) for k in ('fold','arm','status','http_status','wall_seconds')}) ,flush=True)
        return rec
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(one,spec['schedule']))
    save(HERE/'results/api_attempts.json',{'requests':records,'total_run_wall_seconds':time.perf_counter()-start,'credentials_logged':False})

def score():
    q=pd.read_csv(HERE/'results/scorer_only_queries.csv');base=json.loads((HERE/'results/baseline_decisions.json').read_text())
    attempts=json.loads((HERE/'results/api_attempts.json').read_text())
    mapping={(r['fold'],r['arm']):r for r in attempts['requests']};rows=[];percase=[]
    for fold,v in q.groupby('fold',sort=True):
        arms={r['arm']:r['selected_case_ids'] for r in base if r['fold']==fold}
        arms.update(first_only=[],all_average=v.case_id.tolist())
        for name in ['direct','atomic']:
            rec=mapping[(fold,name)];arms[name]=rec['selected_case_ids'] if rec['status']=='valid' else arms['same_card_rule']
        y=v.y.to_numpy();h0=v.p_temporal.to_numpy()>=.5;h1=(v.p_temporal.to_numpy()+v.p_stats_offline_target_only.to_numpy())/2>=.5
        delta=(h0!=y).astype(int)-(h1!=y).astype(int)
        for name,ids in arms.items():
            mask=v.case_id.isin(ids).to_numpy();pred=np.where(mask,h1,h0);c=int(((h0!=y)&(pred==y)).sum());d=int(((h0==y)&(pred!=y)).sum())
            record=dict(fold=int(fold),arm=name,rows=len(v),C=c,D=d,net=c-d,simulated_second_calls=int(mask.sum()),macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),errors=int((pred!=y).sum()),response_status=mapping[(fold,name)]['status'] if name in ['direct','atomic'] else 'local_reference',fallback_used=name in ['direct','atomic'] and mapping[(fold,name)]['status']!='valid')
            if name in ['direct','atomic']:record['api_wall_seconds']=mapping[(fold,name)]['wall_seconds']
            rows.append(record)
            percase.extend(dict(fold=int(fold),arm=name,case_id=r.case_id,sample_hash=r.sample_hash,y=int(r.y),acquired=bool(mask[i]),prediction=int(pred[i]),delta=int(delta[i])) for i,r in enumerate(v.itertuples()))
    df=pd.DataFrame(rows);df.to_csv(HERE/'results/policy_metrics.csv',index=False)
    pd.DataFrame(percase).to_csv(HERE/'results/per_case_scoring.csv',index=False)
    group=df.groupby('arm')[['rows','C','D','net','simulated_second_calls','errors']].sum().reset_index()
    group.to_csv(HERE/'results/descriptive_totals.csv',index=False)
    save(HERE/'results/summary.json',{'status':'completed_development_API_pilot','model':MODEL,'requests':len(attempts['requests']),
        'valid_responses':sum(r['status']=='valid' for r in attempts['requests']),
        'rows':len(q),'unique_sample_hashes':q.sample_hash.nunique(),'fold_records_overlap':True,
        'jev_calls':0,'fine_tuning':False,'fresh_blind_test':False,'full_detection_latency_measured':False,
        'descriptive_totals':group.to_dict('records'),'api_run_wall_seconds':attempts['total_run_wall_seconds']})
    print(group.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','live','score']);a=p.parse_args()
    {'prepare':prepare,'live':live,'score':score}[a.mode]()
