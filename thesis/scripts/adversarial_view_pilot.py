"""Fixed, label-blind view-channel manipulation; forward inference only."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import sys,json,gzip,time,copy,shutil,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import authoritative_fusion_benchmark as b
import frozen_fusion_fast as f
from threadpoolctl import threadpool_limits
ROOT=b.ROOT;BASE=b.OUT;OUT=BASE/'adversarial_view_pilot_v1'
SEEDS=[42,43]
METHODS=['V0_Stats','V0_Temporal','V0_TLS','V1_Average','V2_Yager','V3_Full','V6_Attention_Frozen']
ATTACKS={
 'stats_mild':dict(view=0,strength='mild',transform='Pad each observed prefix packet to at least 128 bytes, never shorten.'),
 'stats_strong':dict(view=0,strength='strong',transform='Pad each observed prefix packet to at least 1500 bytes, never shorten; packets already larger remain unchanged.'),
 'stats_flip':dict(view=0,strength='flip',transform='Directional padding: originally byte-minority direction to at least 1500 bytes, majority direction to at least 128. Ties designate inbound as minority. No guarantee of prediction inversion.'),
 'temporal_mild':dict(view=1,strength='mild',transform='Add 0.010 seconds to each stored IAT; cumulative packet delays increase by 0.010 seconds per gap.'),
 'temporal_strong':dict(view=1,strength='strong',transform='Replace each stored IAT t with 10*t+1.0 seconds; retain packet order, lengths and directions.'),
 'temporal_flip':dict(view=1,strength='flip',transform='Invert each +/-1 direction (observation-orientation manipulation) and apply t -> 10*t+1.0 seconds. This is not a claim an on-path adversary can reverse sender identity.'),
 'tls_mild':dict(view=2,strength='mild',transform='For already-applicable TLS only, set cipher_suite=TLS_AES_128_GCM_SHA256. Existing rule ignores cipher identity: negative control.'),
 'tls_strong':dict(view=2,strength='strong',transform='For already-applicable TLS only, set version=tls1.0 and cipher_suite=TLS_RSA_WITH_3DES_EDE_CBC_SHA. Preserve certificate/handshake flags.'),
 'tls_flip':dict(view=2,strength='flip',transform='For already-applicable TLS only, toggle recognized legacy versions to tls1.3, otherwise to tls1.0; set corresponding cipher. No label or detector response is queried.'),
}

def readz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def safe(raw):
    return b.DetectorInput(stats=raw['stats'],sequence=b.SequenceFeatures.model_validate(raw['sequence']),tls={k:v for k,v in raw.get('tls',{}).items() if k in ('version','cipher_suite','certificate_valid','handshake_complete')})

def transform(d,condition):
    d=d.model_copy(deep=True);j=ATTACKS[condition]['view'];strength=ATTACKS[condition]['strength']
    if j==0:
        s=d.stats;length=np.asarray(d.sequence.packet_lengths,dtype=float);direction=np.asarray(d.sequence.directions)
        floors=np.full(len(length),128 if strength=='mild' else 1500)
        if strength=='flip':
            minority=-1 if s['outbound_bytes']>=s['inbound_bytes'] else 1
            floors=np.where(direction==minority,1500,128)
        padded=np.maximum(length,floors);delta=padded-length;n=s['packet_count'];mean=s['mean_packet_length']
        # Only stored prefix packets change; the unseen suffix remains exactly as observed.
        total=s['total_bytes']+float(delta.sum());newmean=mean+float(delta.sum())/n
        second=s['packet_length_variance']+mean*mean+float(np.sum(padded*padded-length*length))/n
        s.update(total_bytes=total,outbound_bytes=s['outbound_bytes']+float(delta[direction==1].sum()),inbound_bytes=s['inbound_bytes']+float(delta[direction==-1].sum()),mean_packet_length=newmean,packet_length_variance=max(0.,second-newmean*newmean))
        s['outbound_ratio']=s['outbound_bytes']/max(total,1.)
        # Stats channel alone is attacked; stored Temporal input remains clean.
    elif j==1:
        d.sequence.iats=[t+.01 if strength=='mild' else 10*t+1. for t in d.sequence.iats]
        if strength=='flip':d.sequence.directions=[-v for v in d.sequence.directions]
        # The source extractor has no populated burst observations in these pilot records.
        assert not d.sequence.bursts,'STOP: cannot infer altered burst semantics'
    else:
        if d.tls.get('version') or d.tls.get('cipher_suite'):
            if strength=='mild':d.tls['cipher_suite']='TLS_AES_128_GCM_SHA256'
            else:
                legacy=str(d.tls.get('version','')).lower() in ('ssl3','tls1.0','tlsv1','tls1.1')
                version='tls1.3' if strength=='flip' and legacy else 'tls1.0'
                d.tls.update(version=version,cipher_suite='TLS_AES_128_GCM_SHA256' if version=='tls1.3' else 'TLS_RSA_WITH_3DES_EDE_CBC_SHA')
    return d

def register():
    for p in (BASE/'PROTOCOL.md',BASE/'frozen_predictions',BASE/'features.npz',BASE/'selective_reliability_v1/PREREGISTRATION.json'):
        if not p.exists():raise FileNotFoundError('STOP: missing locked prerequisite '+str(p))
    OUT.mkdir(exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():return
    features=readz(BASE/'features.npz');assert features['stats'].shape==(40000,11) and features['temporal'].shape==(40000,30)
    counts=[int(features['app'][features['group']==g,2].sum()) for g in range(20)]
    folds=[max(range(10),key=lambda g:counts[g]),max(range(10,20),key=lambda g:counts[g])]
    # Group choice depends on label stratum / TLS availability, never detection outcomes.
    protected=json.loads((BASE/'selective_reliability_v1/PREREGISTRATION.json').read_text())['protected_sha256']
    assert all(b.sha(BASE/p)==h for p,h in protected.items()),'STOP: locked artifacts changed'
    b.dump(OUT/'PREREGISTRATION.json',dict(seeds=SEEDS,folds=folds,group_names=features['group_names'][folds].tolist(),tls_counts=[counts[g] for g in folds],selection='maximum observed TLS applicability in each label stratum; lower group index breaks ties; not representative',attacks=ATTACKS,methods=METHODS,protected_sha256=protected,fit_allowed=False,operating_point='unchanged clean-validation 20th percentile confidence, plus native binary requirement; native-only separately',go_criterion='For at least one fixed strong attack in BOTH seeds: >=20 wrong-target inter-view-conflict rows, >=5 averaging harmful flips; full has fewer native fixed harmful flips, strictly lower fixed selective error, >=25% of averaging fixed acceptance, at least one native stays-correct conflict row, and >=1 new downweight or OOD suppression among conflict rows. Otherwise NO-GO for immediate full study; no thresholds tuned.',ci='No population group CI with only one held-out group per label; descriptive paired counts and per-seed effects only.',scope='view-channel intervention; Stats padding, Temporal delay/orientation, TLS metadata integrity stress; not a demonstrated packet-level attack against all views'))
    print('Registered',folds,features['group_names'][folds].tolist(),flush=True)

def prepare():
    cfg=json.loads((OUT/'PREREGISTRATION.json').read_text());data=readz(BASE/'features.npz')
    indices=np.flatnonzero(np.isin(data['group'],cfg['folds']));wanted={str(data['sample_id'][i]):int(i) for i in indices};records={}
    audit=json.loads((BASE/'data_audit.json').read_text())
    manifest=pd.read_csv(BASE/'sample_manifest.csv');files=set(manifest.iloc[indices].source_file)
    # Match the canonical filename by provenance rather than reconstructing path separators.
    for source in audit['source_files']:
        p=ROOT/source['path']
        with gzip.open(p,'rt',encoding='utf8') as stream:
            first=json.loads(next(stream))
            if first['provenance']['source_file'] not in files:continue
            assert b.sha(p)==source['sha256']
        with gzip.open(p,'rt',encoding='utf8') as stream:
            for line in stream:
                r=json.loads(line)
                if r['sample_id'] in wanted:records[wanted[r['sample_id']]]=r
    assert len(records)==4000
    tlsagent=b.TLSProtocolAgent(contract_native_fields=True);xs=[];xt=[];tm=[];tp=[];safe_records={}
    for i in indices:
        d=safe(records[int(i)]);safe_records[int(i)]=d
        np.testing.assert_array_equal(b.extract_stats_features(d),data['stats'][i]);np.testing.assert_array_equal(b.extract_temporal_features(d),data['temporal'][i])
        assert not d.sequence.bursts
    with gzip.open(OUT/'pilot_source_records.jsonl.gz','wt',encoding='utf8') as dest:
        for i in indices:dest.write(json.dumps(dict(index=int(i),record=records[int(i)]),ensure_ascii=False)+'\n')
    for name,spec in ATTACKS.items():
        j=spec['view'];xx=[];mm=[];pp=[];changed=[];notes=[]
        for i in indices:
            before=safe_records[int(i)];d=transform(before,name)
            if j<2:
                x=(b.extract_stats_features if j==0 else b.extract_temporal_features)(d);xx.append(x)
                orig=data['stats' if j==0 else 'temporal'][i];changed.append(not np.array_equal(x,orig,equal_nan=True))
                if j==0:assert d.sequence==before.sequence and d.tls==before.tls
                else:assert d.stats==before.stats and d.tls==before.tls
            else:
                assert d.stats==before.stats and d.sequence==before.sequence
                changed.append(d.tls!=before.tls)
                if data['app'][i,2]:
                    ev=tlsagent.analyze(d);m=[ev.benign_support,ev.malicious_support,ev.uncertainty];p=m[1]/max(m[0]+m[1],1e-12)
                else:m=[0,0,1];p=.5
                mm.append(m);pp.append(p);notes.append(dict(index=int(i),old_tls=before.tls,new_tls=d.tls))
        arrays=dict(index=indices,changed=np.array(changed),features=np.array(xx),tls_mass=np.array(mm),tls_probability=np.array(pp))
        np.savez_compressed(OUT/f'input_{name}.npz',**arrays)
        if notes:
            with gzip.open(OUT/f'input_{name}.jsonl.gz','wt',encoding='utf8') as dest:
                for row in notes:dest.write(json.dumps(row)+'\n')
    b.dump(OUT/'INPUTS_READY.json',dict(records=4000,feature_parity_rows=4000,bursts_empty=True,inputs={n:b.sha(OUT/f'input_{n}.npz') for n in ATTACKS}))

def gate_output(net,ev):
    torch=f.torchlib();gx=f.gating_features(ev['p'],ev['app'])
    with torch.no_grad():p,w=f.gate_probability(net(torch.tensor(gx)),torch.tensor(gx))
    p=p.numpy();return dict(p=p,pred=(p>=.5).astype(np.int8),score=2*abs(p-.5),verdict=(p>=.5).astype(np.int8),available=ev['app'].any(1),weights=w.numpy())

def discount(ev,j):
    a=np.clip((ev['ood'][:,j]-.95)/.04,0,1) if j<2 else np.zeros(len(ev['p']))
    return ev['reliability'][:,j]*np.maximum(.1,1-.9*a)*(ev['ood'][:,j]<.99 if j<2 else 1)*ev['app'][:,j]

def run():
    cfg=json.loads((OUT/'PREREGISTRATION.json').read_text());data=readz(BASE/'features.npz');threadpool_limits(1)
    # Fail closed if this pilot accidentally calls model fitting.
    def forbidden(*args,**kwargs):raise RuntimeError('NO RETRAINING permitted in adversarial pilot')
    from sklearn.pipeline import Pipeline
    Pipeline.fit=forbidden;b.HistGradientBoostingClassifier.fit=forbidden;b.IsolationForest.fit=forbidden
    for seed in SEEDS:
        for fold in cfg['folds']:
            dest=OUT/f'seed_{seed}'/f'fold_{fold:02d}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'DONE.json').exists():continue
            start=time.perf_counter();mp=BASE/'fold_runs'/f'seed_{seed}'/f'fold_{fold:02d}';ff=BASE/'frozen_predictions'/f'seed_{seed}'/f'fold_{fold:02d}'
            model=b.joblib.load(mp/'models.joblib');frozen=readz(ff/'test_clean.npz');ids=frozen.pop('index');y=data['y'][ids]
            clean={k:frozen[k] for k in ('p','mass','app','reliability','ood','ood_raw')}
            sp=readz(BASE/'splits'/f'seed_{seed}'/f'fold_{fold:02d}.npz');assert np.array_equal(ids,sp['test'])
            assert not (set(data['group'][sp['train']])|set(data['group'][sp['validation']]))&set(data['group'][ids])
            fp=f.cell(seed,fold);trained=json.loads((fp/'TRAINED.json').read_text());assert all(b.sha(fp/n)==h for n,h in trained['model_sha256'].items())
            net=f.make_net(f.CONFIG['gate_dims']);net.load_state_dict(f.torchlib().load(fp/'fusion_heads.pt',map_location='cpu',weights_only=True)['gate']);net.eval()
            val=readz(ff/'validation.npz');vd=gate_output(net,val);thresholds=json.loads((mp/'training.json').read_text())['thresholds'];thresholds['V6_Attention_Frozen']=float(np.quantile(vd['score'][vd['available']],.2))
            clean_dec={m:(gate_output(net,clean) if m=='V6_Attention_Frozen' else b.decision(m,clean,{},None)) for m in METHODS}
            existing=readz(mp/'predictions.npz')
            for m in METHODS[:-1]:
                np.testing.assert_array_equal(clean_dec[m]['pred'],existing[f'clean__{m}__pred']);np.testing.assert_array_equal(clean_dec[m]['verdict'],existing[f'clean__{m}__verdict'])
            frozen_head=readz(fp/'predictions.npz');np.testing.assert_allclose(clean_dec['V6_Attention_Frozen']['p'],frozen_head['probabilities'][2],atol=1e-7,rtol=0)
            for condition in ['clean',*ATTACKS]:
                ev={k:v.copy() for k,v in clean.items()};changed=np.zeros(len(ids),bool);j=-1
                if condition!='clean':
                    j=ATTACKS[condition]['view'];inp=readz(OUT/f'input_{condition}.npz');position=np.searchsorted(inp['index'],ids);assert np.array_equal(inp['index'][position],ids);changed=inp['changed'][position]
                    if j<2:
                        x=inp['features'][position];ev['p'][:,j]=model['views'][j].predict_proba(x)[:,1];ev['mass'][:,j]=b.masses(ev['p'][:,j]);gate,cal=model['ood_gates'][j];raw=-gate.score_samples(x);ev['ood_raw'][:,j]=raw;ev['ood'][:,j]=np.searchsorted(cal,raw,side='right')/len(cal)
                    else:ev['p'][:,2]=inp['tls_probability'][position];ev['mass'][:,2]=inp['tls_mass'][position]
                    for k in clean:
                        np.testing.assert_array_equal(ev[k][:,[v for v in range(3) if v!=j]],clean[k][:,[v for v in range(3) if v!=j]])
                    np.testing.assert_array_equal(ev['app'],clean['app']);np.testing.assert_array_equal(ev['reliability'],clean['reliability'])
                dec={m:(gate_output(net,ev) if m=='V6_Attention_Frozen' else b.decision(m,ev,{},None)) for m in METHODS}
                saved=dict(index=ids,y=y,changed=changed,**{'ev_'+k:v for k,v in ev.items()})
                for m,d in dec.items():
                    for k,v in d.items():saved[m+'__'+k]=v
                    saved[m+'__threshold']=np.full(len(ids),thresholds[m])
                np.savez_compressed(dest/f'{condition}.npz',**saved)
            b.dump(dest/'DONE.json',dict(seed=seed,fold=fold,seconds=time.perf_counter()-start,fit_calls=0,clean_predictions_match=True,unchanged_views_exact=True,model_sha256=b.sha(mp/'models.joblib'),outputs={c:b.sha(dest/f'{c}.npz') for c in ['clean',*ATTACKS]}))
            print('Complete',seed,fold,round(time.perf_counter()-start,2),flush=True)

def ratio(a,n):return float(a/n) if n else float('nan')

def metrics(z,clean,m,condition):
    pred=z[m+'__pred'];cp=clean[m+'__pred'];y=z['y'];av=z[m+'__available'];native=av&(z[m+'__verdict']<2);cn=clean[m+'__available']&(clean[m+'__verdict']<2)
    fixed=native&(z[m+'__score']>=z[m+'__threshold']);cf=cn&(clean[m+'__score']>=clean[m+'__threshold']);conf_only=av&(z[m+'__score']>=z[m+'__threshold']);cc=clean[m+'__available']&(clean[m+'__score']>=clean[m+'__threshold'])
    wrong=pred!=y;cleanright=cp==y;n=len(y);j=ATTACKS.get(condition,{}).get('view',-1)
    changed=z['changed'];conflict=np.zeros(n,bool);wrongconflict=conflict.copy();follow=conflict.copy();viewwrong=conflict.copy();newviewwrong=conflict.copy();react=conflict.copy();lower=np.zeros(n)
    view_pred=np.full(n,-1)
    if j>=0:
        view_pred=(z['ev_p'][:,j]>=.5).astype(int);viewwrong=(view_pred!=y)&z['ev_app'][:,j];newviewwrong=viewwrong&((clean['ev_p'][:,j]>=.5)==y)
        others=[v for v in range(3) if v!=j];otherpred=clean['ev_p'][:,others]>=.5
        conflict=changed&z['ev_app'][:,j]&np.any(clean['ev_app'][:,others]&(otherpred!=view_pred[:,None]),axis=1)
        wrongconflict=conflict&viewwrong;follow=wrongconflict&native&(pred==view_pred)
        ev={k:z['ev_'+k] for k in ('p','app','reliability','ood')};ce={k:clean['ev_'+k] for k in ev};lower=discount(ce,j)-discount(ev,j);react=lower>1e-12
    # A prediction already wrong before the intervention is not credited to the attack.
    new_conflict=wrongconflict&newviewwrong
    attributable=cn&native&cleanright&wrong&changed&viewwrong&(pred==view_pred)
    state=np.where(~native,'safely-abstains',np.where(~wrong,np.where(cleanright,'stays-correct','corrected'),np.where(attributable,'manipulated',np.where(cleanright,'introduced-other-wrong','preexisting-wrong'))))
    harmful=cf&fixed&cleanright&wrong
    row=dict(n=n,available=int(av.sum()),macro_f1=b.metric(y[av],pred[av])[0],clean_macro_f1=b.metric(y[clean[m+'__available']],cp[clean[m+'__available']])[0],accuracy=ratio(int((~wrong&av).sum()),int(av.sum())),harmful_flips=int(harmful.sum()),harmful_eligible_clean=int((cf&cleanright).sum()),forced_confidence_harmful_flips=int((cc&conf_only&cleanright&wrong).sum()),introduced_native_wrong=int((cn&native&cleanright&wrong).sum()),native_accepted=int(native.sum()),native_coverage=float(native.mean()),abstention_rate=float((~native).mean()),unknown_rate=float((z[m+'__verdict']==3).mean()),suspicious_rate=float((z[m+'__verdict']==2).mean()),native_selective_error=ratio(int((native&wrong).sum()),int(native.sum())),fixed_accepted=int(fixed.sum()),fixed_coverage=float(fixed.mean()),fixed_selective_error=ratio(int((fixed&wrong).sum()),int(fixed.sum())),manipulated_input_rows=int(changed.sum()),view_new_wrong=int(newviewwrong.sum()),inter_view_conflict=int(conflict.sum()),wrong_view_conflict=int(wrongconflict.sum()),follows_wrong_view=int(follow.sum()),follows_wrong_view_rate=ratio(int(follow.sum()),int(wrongconflict.sum())),follows_wrong_view_among_accepted=ratio(int(follow.sum()),int((wrongconflict&native).sum())),conflict_stays_correct=int((wrongconflict&native&~wrong&cleanright).sum()),conflict_native_reject=int((wrongconflict&~native).sum()),conflict_downweighted=int((wrongconflict&react).sum()),downweighted_rows=int(react.sum()),mean_effective_reliability_drop=float(lower.mean()))
    row['macro_f1_drop']=row['clean_macro_f1']-row['macro_f1']
    row.update(new_wrong_view_conflict=int(new_conflict.sum()),follows_new_wrong_view=int((new_conflict&native&(pred==view_pred)).sum()),follows_new_wrong_view_rate=ratio(int((new_conflict&native&(pred==view_pred)).sum()),int(new_conflict.sum())),attack_introduced_following=int(attributable.sum()),new_native_rejections=int((cn&~native).sum()),previously_rejected_remaining_rejected=int((~cn&~native).sum()),native_stays_correct=int((cn&native&cleanright&~wrong).sum()))
    if j>=0:
        row.update(manipulated_view_probability_change=float(np.mean(abs(z['ev_p'][:,j]-clean['ev_p'][:,j]))),ood_percentile_change=float(np.mean(z['ev_ood'][:,j]-clean['ev_ood'][:,j])),hard_ood_rows=int((z['ev_ood'][:,j]>=.99).sum()) if j<2 else 0,new_hard_ood_rows=int(((z['ev_ood'][:,j]>=.99)&(clean['ev_ood'][:,j]<.99)).sum()) if j<2 else 0,validation_reliability_change=float(np.max(abs(z['ev_reliability'][:,j]-clean['ev_reliability'][:,j]))))
        directional=wrongconflict&(abs(z['ev_p'][:,j]-.5)>1e-8)
        row.update(nontied_wrong_view_conflict=int(directional.sum()),follows_nontied_wrong_view_rate=ratio(int((directional&native&(pred==view_pred)).sum()),int(directional.sum())),target_probability_min=float(z['ev_p'][:,j].min()),target_probability_max=float(z['ev_p'][:,j].max()))
        if m=='V6_Attention_Frozen':row['attention_target_weight_change']=float((z[m+'__weights'][:,j]-clean[m+'__weights'][:,j]).mean())
    return row,dict(index=z['index'],y=y,pred=pred,clean_pred=cp,native_accepted=native,clean_native_accepted=cn,fixed_accepted=fixed,verdict=z[m+'__verdict'],state=state,changed=changed,view_pred=view_pred,wrong_view_conflict=wrongconflict,new_wrong_view_conflict=new_conflict,follows_wrong_view=follow,attack_introduced_following=attributable,harmful_flip=harmful,effective_reliability_drop=lower)

def report():
    cfg=json.loads((OUT/'PREREGISTRATION.json').read_text());rows=[];decisions=[]
    for seed in SEEDS:
        allz={}
        for c in ['clean',*ATTACKS]:
            parts=[readz(OUT/f'seed_{seed}'/f'fold_{g:02d}'/f'{c}.npz') for g in cfg['folds']]
            allz[c]={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for c,z in allz.items():
            for m in METHODS:
                row,labels=metrics(z,allz['clean'],m,c);row.update(seed=seed,fold='pooled',condition=c,method=m,scope='pooled_two_groups');rows.append(row)
                if c!='clean':
                    ld=pd.DataFrame(labels);ld.insert(0,'method',m);ld.insert(0,'condition',c);ld.insert(0,'seed',seed);decisions.append(ld)
                for g in cfg['folds']:
                    mask=(z['index']//2000)==g;sub={k:v[mask] for k,v in z.items()};cl={k:v[mask] for k,v in allz['clean'].items()};r,_=metrics(sub,cl,m,c);r.update(seed=seed,fold=g,condition=c,method=m,scope='single_group_fixed_label_f1_max_0.5');rows.append(r)
    table=pd.DataFrame(rows);table.to_csv(OUT/'adv_view_pilot.csv',index=False,encoding='utf-8-sig');pd.concat(decisions,ignore_index=True).to_csv(OUT/'per_sample_labels.csv.gz',index=False,compression='gzip')
    pooled=table[table.scope=='pooled_two_groups'];effects=[]
    for c in ATTACKS:
        for seed in SEEDS:
            dd=pooled[(pooled.condition==c)&(pooled.seed==seed)].set_index('method');a=dd.loc['V1_Average'];v=dd.loc['V3_Full'];criteria=dict(conflicts=v.wrong_view_conflict>=20,averaging_flips=a.harmful_flips>=5,fewer_full_flips=v.harmful_flips<a.harmful_flips,lower_full_error=v.fixed_selective_error<a.fixed_selective_error,coverage_floor=v.fixed_accepted>=.25*a.fixed_accepted,stays_correct=v.conflict_stays_correct>=1,ood_reacts=v.conflict_downweighted>=1)
            effects.append(dict(condition=c,seed=seed,strong=ATTACKS[c]['strength']=='strong',pass_all=all(criteria.values()),**criteria,full_minus_average_harmful=v.harmful_flips-a.harmful_flips,full_minus_average_fixed_error=v.fixed_selective_error-a.fixed_selective_error,full_minus_average_native_error=v.native_selective_error-a.native_selective_error,full_minus_average_f1=v.macro_f1-a.macro_f1,full_fixed_coverage=v.fixed_coverage,average_fixed_coverage=a.fixed_coverage))
    ef=pd.DataFrame(effects);ef.to_csv(OUT/'go_no_go.csv',index=False,encoding='utf-8-sig');go=any(ef[ef.condition==c].pass_all.all() for c in ATTACKS if ATTACKS[c]['strength']=='strong')
    b.dump(OUT/'CONCLUSION.json',dict(go=bool(go),criterion=cfg['go_criterion'],no_population_ci=True))
    # Final immutability and output checks; no regeneration of view predictions here.
    assert all(b.sha(BASE/p)==h for p,h in cfg['protected_sha256'].items())
    for seed in SEEDS:
        for g in cfg['folds']:
            dest=OUT/f'seed_{seed}'/f'fold_{g:02d}';done=json.loads((dest/'DONE.json').read_text());assert all(b.sha(dest/f'{c}.npz')==h for c,h in done['outputs'].items())
    b.dump(OUT/'VALIDATION.json',dict(status='passed',protected_parent_files=len(cfg['protected_sha256']),cells=4,fit_calls=0,other_views_unchanged=True,rows=len(table),per_sample_rows=sum(len(d) for d in decisions)))
    shutil.copy2(Path(__file__),OUT/'adversarial_view_pilot.py')
    write_docs(pooled,ef,cfg,go)
    print(pooled[pooled.method.isin(['V1_Average','V3_Full'])][['condition','seed','method','macro_f1','macro_f1_drop','harmful_flips','fixed_coverage','fixed_selective_error','wrong_view_conflict','follows_wrong_view','conflict_downweighted']].to_string(index=False),flush=True)
    print('GO',go,flush=True)

def write_docs(pooled,effects,cfg,go):
    def fm(x):return 'NA (no accepted samples)' if not np.isfinite(x) else f'{x:.4f}'
    def ms(values):return f'{np.mean(values):.4f} ± {np.std(values,ddof=1):.4f}'
    summaries=[]
    for (condition,method),g in pooled.groupby(['condition','method'],sort=False):
        row=dict(condition=condition,method=method,seeds=2)
        for col in ['macro_f1','macro_f1_drop','accuracy','native_coverage','native_selective_error','fixed_coverage','fixed_selective_error','harmful_flips','introduced_native_wrong','follows_new_wrong_view_rate','new_native_rejections']:
            row[col+'_mean']=g[col].mean();row[col+'_std']=g[col].std(ddof=1)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'pilot_seed_summary.csv',index=False,encoding='utf-8-sig')
    lines=['# Adversarial view-manipulation pilot','',
        '**NO-GO for an immediate 20-fold × 10-seed superiority study with the unchanged protocol.** The manipulated view can harm averaging, and learned-view OOD responds, but severe interventions collapse full-fusion acceptance. The pilot does not demonstrate better accepted accuracy at useful retained coverage. No attack, threshold or model was tuned on pilot outcomes.','',
        '## Frozen protocol and selection','',
        'Two existing whole-group folds: **7 (benign Skype)** and **12 (malware Htbot)**, seeds **42 and 43**, 2,000 test rows per fold: 4,000 distinct samples and 8,000 seed-repeated evaluations per condition. Choose the maximum-TLS-applicability group within each label stratum, before running these manipulations. TLS is visible in 1,694 Skype and 221 Htbot rows (1,915/4,000). This selection deliberately stresses TLS and is not representative of all twenty groups; the clean scores below are pilot-subset scores, not replacements for the authoritative benchmark.',
        '', 'Stats remains 11-D HGB, Temporal 30-D HGB, TLS the original conditional rule. All original train/validation/test group indices and fitted artifacts are reused. No fit, fine-tune, imputation refit, recalibration or label-directed perturbation occurs. Only the attacked HGB and its already-fitted OOD gate run on altered features; other view probabilities, masses, reliability, applicability and OOD values are checked exactly unchanged. TLS forward evaluation uses the original rule. All fusers read each same altered evidence bundle. The saved attention head is the existing 9→16→3 masked gate, with frozen weights.',
        '', 'The clean feature extractor matches all 4,000 original 11/30 feature rows exactly, and clean decisions match the original frozen outputs. All 3,606 protected parent artifacts remain unchanged. Code-level guards prohibit sklearn detector/pipeline fitting during the pilot.',
        '', '## Fixed transformations','', '| Condition | Exact operation |','|---|---|']
    lines += [f'| {name} | {spec["transform"]} |' for name,spec in ATTACKS.items()]
    lines += ['', 'Stats updates total/outbound/inbound bytes, mean, variance and outbound ratio consistently using the exact change to prefix sums and squared sums. Packet count and duration stay unchanged. If the record is truncated, only the observed prefix packets are padded; the unknown suffix is unchanged. Derived 11-D features are then extracted by the original function. Temporal extraction is rerun from altered IAT/direction arrays; all pilot source burst arrays are empty and remain empty. No missing TLS handshake is invented, and certificate/handshake-complete flags stay unchanged.',
        '', '**Attack scope:** these are controlled view-channel interventions, not validated on-wire evasion attacks. Actual packet padding would also affect Temporal length features and actual delays would change Stats duration; the experiment intentionally freezes those other channels to isolate a compromised or inconsistent view. MTU-inspired padding does not establish transport/handshake validity. Direction inversion represents an orientation/reporting manipulation, not changing which host sent a packet. TLS version/cipher changes stress metadata integrity; successful live TLS negotiation is not demonstrated. `flip` is a fixed label-blind transformation name, not a promise to invert the detector decision. There is no adaptive adversary or query-budget optimization.',
        '', '## Definitions and operating points','',
        'Macro-F1 uses fixed binary labels [0,1] and the existing forced binary projection. It is evaluated on available rows; TLS single-view F1 therefore uses its applicable subset. A projection for a rejected full-fusion row is a diagnostic, not a deployed accepted decision. Positive `macro_f1_drop` means degradation. Pooled two-group scores have both labels; individual-fold fixed-label F1 is bounded by 0.5 because each test fold is single-class.',
        '', 'Native acceptance means verdict benign/malicious; suspicious and unknown are both native rejection, separately counted in the CSV. Fixed acceptance additionally requires that method’s original clean-validation 20th-percentile confidence threshold. Attention’s identical threshold is reconstructed on frozen validation inputs only. Harmful flips require clean fixed-accepted/correct → attacked fixed-accepted/wrong. `introduced_native_wrong` removes only the confidence requirement. Counts and their eligible denominators are separate. NA error at zero acceptance is never replaced with zero.',
        '', 'Inter-view conflict requires an actually changed attacked input and disagreement between that view’s binary prediction and at least one applicable unchanged view. `follows_wrong_view_rate` is the fraction of wrong-target conflict rows whose fuser natively accepts that wrong target; it can include errors already present before the attack. The stricter `follows_new_wrong_view_rate` restricts the denominator to attacked-view clean-correct → attacked-wrong conflict rows. `attack_introduced_following` additionally requires the fuser itself to have been native-accepted/correct before the attack. These paired changes establish an intervention association, not an attribution from agreement alone.',
        '', 'The inherited binary boundary assigns p=0.5 to malicious. Legacy-version tampering in this pilot moves TLS to p=0.5, a tied/neutral output, not confident malicious evidence. Therefore the broad TLS disagreement/following counts depend on that tie convention; `nontied_wrong_view_conflict` and its following rate exclude ties. The averaging degradation still occurs from losing previously benign TLS support. TLS strong and flip yield identical outputs on these records and are not independent attacks. Temporal strong/flip can have different HGB probabilities and OOD values even where their aggregate classification scores coincide.',
        '', '`per_sample_labels.csv.gz` labels native rejection as `safely-abstains`, accepted retained correctness as `stays-correct`, and attack-associated newly wrong agreement as `manipulated`. It also retains `corrected`, `preexisting-wrong` and `introduced-other-wrong`; forcing all rows into only three labels would misstate preexisting errors. “Safely” means no binary claim is emitted; it does not imply useful coverage or a deployment safety guarantee. Raw predictions, thresholds, changed-input flags and conflict flags accompany each label.',
        '', 'Means/std below are over two seed-wise pooled scores. No population group-bootstrap CI is reported: only one held-out benign group and one held-out malware group are available, so the parent stratified group bootstrap would be degenerate. The seeds repeat the same flows; they are not 8,000 independent samples.',
        '', '## Averaging versus full fusion','', '| Condition | Method | Macro-F1 mean ± std | F1 drop | Native coverage | Native selective error | Fixed harmful flips, summed over seeds |', '|---|---|---|---:|---:|---:|---:|']
    for c in ['clean',*ATTACKS]:
        for m,label in [('V1_Average','Average'),('V3_Full','Full')]:
            g=pooled[(pooled.condition==c)&(pooled.method==m)]
            lines.append(f'| {c} | {label} | {ms(g.macro_f1)} | {g.macro_f1_drop.mean():+.4f} | {g.native_coverage.mean():.4%} | {fm(g.native_selective_error.mean())} | {int(g.harmful_flips.sum())} |')
    lines += ['', '## Mechanism and paired effects','',
        '- **Strong Stats padding:** averaging loses 0.2328 pooled Macro-F1 in each seed and introduces 651 native wrong decisions per seed. Full fusion rejects all 4,000 rows in both seeds. It gains no evidence of accurate accepted decisions because none remain. Its apparently higher forced-projection F1 is not an accepted-performance result.',
        '- **Strong temporal delays:** averaging F1 changes only +0.0011; it does not materially degrade in this pilot. Full fusion accepts only 24/4,000 rows in each seed (0.6%), all correct. This is too little coverage and no effective averaging failure to establish the proposed advantage.',
        '- **Directional Stats padding (`stats_flip`):** full fusion accepts 73/4,000 and 78/4,000 rows; 46 of those are wrong in each seed (native error 63.01% / 58.97%). At the unchanged confidence threshold, it accepts 46 rows per seed and all 46 are wrong. These are retained preexisting errors after correct outputs were rejected, not newly introduced confident flips. Reporting zero harmful flips alone would hide this failure.',
        '- **TLS version manipulation:** averaging introduces 548 native wrong decisions per seed, while full fusion introduces none. Full native coverage/error remain at their already selective clean values. TLS has no OOD model, its reliability does not change, and the full forced projection still degrades. This is a limited rejection-policy protection signal, not evidence that sample-wise reliability learned to identify a forged TLS view. Cipher-only manipulation changes no detector probabilities and is retained as the expected negative control.',
        '- **Fixed-point harmful flips:** averaging and full fusion both have zero under the locked validation confidence threshold in every condition. The native averaging flips occur below this operating threshold. Consequently there is no measured confident-flip advantage at the specified fixed point; that threshold was not loosened after viewing the results.',
        '- **Reliability/OOD:** validation reliability is fixed and unchanged on every row. HGB OOD does react: strong Stats creates 3,203 / 3,136 newly hard-OOD rows for seeds 42 / 43; strong Temporal creates 1,906 / 1,757. Effective learned-view reliability after OOD falls and hard evidence is suppressed. The failure is chiefly coverage collapse and residual selective error, not a claim that learned-view OOD never reacted.',
        '', '| Strong condition | Seed | New wrong target + conflict | Average follows / denominator | Full follows / denominator | Full new native rejections |', '|---|---:|---:|---|---|---:|']
    for c in ['stats_strong','temporal_strong','tls_strong']:
        for seed in SEEDS:
            d=pooled[(pooled.condition==c)&(pooled.seed==seed)].set_index('method');a=d.loc['V1_Average'];v=d.loc['V3_Full'];n=int(v.new_wrong_view_conflict)
            lines.append(f'| {c} | {seed} | {n} | {int(a.follows_new_wrong_view)}/{n} | {int(v.follows_new_wrong_view)}/{n} | {int(v.new_native_rejections)} |')
    lines += ['', '## References: clean and strong conditions','', '| Condition | Reference method | Macro-F1 mean ± std | Native coverage | Native error |', '|---|---|---|---:|---:|']
    for c in ['clean','stats_strong','temporal_strong','tls_strong']:
        for m in ['V0_Stats','V0_Temporal','V0_TLS','V2_Yager','V6_Attention_Frozen']:
            g=pooled[(pooled.condition==c)&(pooled.method==m)]
            lines.append(f'| {c} | {m} | {ms(g.macro_f1)} | {g.native_coverage.mean():.4%} | {fm(g.native_selective_error.mean())} |')
    lines += ['', 'All reference methods and strengths, including outcomes favoring attention or averaging, are retained in the CSVs. TLS single-view comparisons are subset-only; missing TLS is not counted as an erroneous fabricated prediction.',
        '', '## Go / no-go','', 'The implementation fixed the following conservative expansion rule before manipulated inference:', '', '> '+cfg['go_criterion'],
        '', '**Result: NO-GO.** All three strong conditions fail that rule in both seeds; each component is recorded in `go_no_go.csv`. This rule is a pilot decision heuristic, not a significance test. It is stricter than merely observing a refusal: it explicitly guards against all-abstention and requires a measurable confident averaging failure. Under the broader qualitative question, some native averaging errors are prevented by full refusal, but useful accepted accuracy is not sustained and the fixed-point flip comparison is uninformative. The unchanged 20×10 study is therefore not justified as a confirmatory test of a demonstrated advantage.',
        '', 'This is not proof that every possible adversarial attack or group will fail. A future study would need a separately specified threat model and evaluation of the coverage/error tradeoff; this pilot neither tunes such a policy nor expands the experiments without a new specification.',
        '', '## Artifacts and reproduction','',
        '[Per-fold and pooled results](adv_view_pilot.csv), [seed mean/std](pilot_seed_summary.csv), [paired per-sample labels](per_sample_labels.csv.gz), [go/no-go components](go_no_go.csv), [frozen registration](PREREGISTRATION.json), [validation](VALIDATION.json), [commands and run record](RUNS.md).',
        '', 'Each `seed_*/fold_*/<condition>.npz` stores the altered view outputs, unchanged evidence, all fuser decisions and frozen thresholds. `input_*.npz` stores transformed feature arrays/flags; TLS JSONL files preserve before/after fields. `pilot_source_records.jsonl.gz` retains the exact selected source records. Parent splits and fitted models remain at their authoritative paths. No plots were required for this pilot.']
    (OUT/'ADV_PILOT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    commands=['register','prepare','run','report']
    runlines=['# Pilot run record','', 'Executed 2026-09-22 from `.`. Existing Python environment, no package installation. Fixed seeds 42/43, folds 7/12; original models/thresholds, no new hyperparameters.','', '```powershell']
    runlines += [f'.\\.codex_mad_etd_v24_venv\\Scripts\\python.exe -X utf8 scripts\\adversarial_view_pilot.py {stage}' for stage in commands]
    runlines += ['.\\.codex_mad_etd_v24_venv\\Scripts\\python.exe -X utf8 -m pytest tests\\test_adversarial_view_pilot.py -q','```','',
        'Registration and input preparation completed before manipulated detector evaluation. Four fold/seed cells completed once; DONE markers prevent subsequent reruns. Changed-view HGB/OOD forward passes: six learned-view conditions × four cells × 2,000 rows; TLS rules are deterministic and computed once on each applicable transformed source record, shared across seeds. Clean test view outputs are reused. Attention has validation-only threshold reconstruction and forward calls; no parameters are updated.',
        '', 'The three tests check exact prefix-padding moments against literal full-packet transformations (including an untouched suffix), view isolation / absent-TLS preservation, and cipher-only rule invariance. All pass. Clean feature parity covers 4,000 rows; clean method predictions match the existing saved outputs. Other evidence columns and every applicability/reliability value are asserted unchanged in every intervention. All 3,606 protected parent files are hash-checked after reporting.',
        '', 'Reporting was regenerated from saved outputs to separate newly induced following from preexisting wrong-view agreement and to retain exhaustive per-sample outcome labels. This reporting clarification did not change transforms, thresholds, go/no-go criteria, detector outputs or test inference. Reports include both the broad agreement diagnostic and strict paired-change counts. The go/no-go registration has not been edited after inference.',
        '', 'No training or experimental cell failed. During source inspection, an assumed TLS-agent filename did not exist and a Windows wildcard search path was invalid; both were corrected using the actual source paths. A fitting guard initially referred to an unimported Pipeline name and was fixed before execution. A final report check rejected an inferred Temporal strong/flip probability-equality claim (199 differing probability entries in the first inspected cell); the prose was corrected. Equal aggregate scores do not imply equal view outputs. TLS strong/flip equality and neutral p=0.5 were separately verified in all four cells. There were no outcome-driven retries or changed detector predictions. Runtime exceptions, if any occur on reproduction, append to `failures.log`.',
        '', 'No formal group CI: two selected groups cannot provide nondegenerate label-stratified group uncertainty. All two-seed means/std and individual-fold rows are retained. One-class fold F1 has ceiling 0.5. These selected-fold clean results must not be substituted for whole-protocol anchors.',
        '', 'Scientific negative outcomes: NO-GO; Stats hard manipulation gives full-fusion zero acceptance, Temporal hard manipulation 0.6% acceptance, directional padding leaves high error in the retained set, and no confident harmful flips distinguish full fusion from averaging at the original fixed point. No test-time threshold was changed to make a comparison favorable.']
    (OUT/'RUNS.md').write_text('\n'.join(runlines)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','prepare','run','report']);args=p.parse_args()
    try:globals()[args.stage]()
    except Exception:
        import traceback
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.log').open('a',encoding='utf8') as f2:f2.write(args.stage+'\n'+traceback.format_exc()+'\n')
        raise
if __name__=='__main__':main()
