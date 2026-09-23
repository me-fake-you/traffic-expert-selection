"""Paired excluded-record-field checks and ONE fixed time-scale stress test."""
from common_new import *
import copy
sys.path.insert(0,str(R2/'experiments'))
import raw_pipeline as historical_raw

def record(raw):
    n=len(raw.lengths);total=sum(raw.lengths);out=sum(v for v,d in zip(raw.lengths,raw.directions) if d==1);mean=total/n
    return {'stats':{'packet_count':n,'total_bytes':total,'outbound_bytes':out,'inbound_bytes':total-out,
        'outbound_ratio':out/max(total,1),'mean_packet_length':mean,'packet_length_variance':max(0.,sum(v*v for v in raw.lengths)/n-mean*mean),'duration':raw.last-raw.first},
        'sequence':{'packet_lengths':raw.lengths[:64],'directions':raw.directions[:64],
                    'iats':[max(0.,b-a) for a,b in zip(raw.times[:64],raw.times[1:64])]},
        'context':{'src_ip':raw.origin[0],'src_port':raw.origin[1]},'provenance':{'source_file':raw.capture},
        'labels':{'binary':'decoy_not_scoring_truth'},'sample_id':raw.sample_hash}
def project(r):
    # Exact existing functions used by the 8/40 pipeline, not the runtime FieldAudit class.
    return np.asarray(historical_raw.env['_stats_vector'](r)+historical_raw.env['_sequence_vector'](r),dtype=np.float32)
def mutate(r,variant):
    r=copy.deepcopy(r)
    if variant=='labels_identity':r['labels']={'binary':'malicious','family':'decoy','target':-999};r['sample_id']='mutated-record-id';r['trace_id']='different-trace'
    elif variant=='context':r['context']={'src_ip':'192.0.2.1','dst_ip':'198.51.100.2','src_port':1,'dst_port':65535,'timestamp':-999}
    elif variant=='provenance_unknown':
        r['provenance']={'source_file':'decoy.pcap','capture_id':'changed','group':'changed'}
        r['unknown']={'label':12345};r['stats']['label']=999;r['sequence']['source_file']='ignored'
    return r
def all_outputs(x,f,bundle,refs,locks):
    a=bundle['temporal'].predict_proba(x[:,8:])[:,1];b=bundle['stats'].predict_proba(x[:,:8])[:,1]
    r=refs[str(f)];ra,rb=r['reliability_temporal'],r['reliability_stats'];first=a>=.5;tr=abs(a-.5)<.495
    outputs={'temporal':first.astype(int),'stats':(b>=.5).astype(int),'equal_average':((a+b)/2>=.5).astype(int),
             'fixed':np.where(tr&(abs(b-.5)*rb>.25*abs(a-.5)*ra),b>=.5,first).astype(int)}
    state={'pT':a,'pS':b,'trigger':tr}
    for kind in ['logistic','hgb']:
        q=bundle[kind].predict_proba(meta(a,b,ra,rb))[:,1];state['q_'+kind]=q
        lock=locks[(locks.outer_fold==f)&(locks.condition=='row100')&(locks.kind==kind)].iloc[0]
        outputs[kind+'_05']=learned(a,b,q,.495,.5,.5)
        outputs[kind+'_dev']=learned(a,b,q,.495,lock.selected_low,lock.selected_high)
    return outputs,state
def main():
    dest=OUT/'shortcut_checks';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'completion.json').exists()
    start=time.perf_counter();cfg,data,pv,hist=load_inputs();data['y'].setflags(write=False)
    ids={str(h):i for i,h in enumerate(data['sample_hash'])};variants=['labels_identity','context','provenance_unknown']
    lock={'created':stamp(),'command':sys.argv,'isolation_variants':variants,'population':'all original 40000 held-out segment identities, immutable split/scoring label sidecar',
          'allowed_feature_adapter':'existing explicit _stats_vector/_sequence_vector projection; full runtime FieldAudit is NOT executed by this path',
          'stress':'one predefined factor-two relative time-scale stress: duration and prefix IAT multiply by 2; lengths/directions fixed',
          'stress_hypothesis':'Frozen detector/selector outputs may be sensitive to time scale even with explicit metadata excluded',
          'stress_scope':'distribution-shift stress, NOT retrained feature-removal ablation, NOT proof of shortcut reliance or causal generalization',
          'choice_basis':'thesis timing perturbation and existing duration/IAT feature definitions, not new evaluation scores',
          'stop':'stop on missing raw records, feature mismatch, or any paired isolation failure; retain failures. No stress factor search.',
          'new_fits':0,'feature_source_sha256':sha(ROOT/'src/mad_etd/ustc_group_heldout_hybrid_w98.py')}
    dump(dest/'protocol_lock.json',lock)
    clean=np.empty_like(data['x_hybrid']);changed={k:np.empty_like(clean) for k in variants};stress=np.empty_like(clean);seen=set();positive_controls=0
    raw_files=pd.read_csv(R2/'results_r2/raw_001/raw_inventory.csv')
    for path in raw_files.raw_records:
        for raw in joblib.load(path):
            if raw.sample_hash not in ids:continue
            i=ids[raw.sample_hash];assert i not in seen;seen.add(i);r=record(raw);clean[i]=project(r)
            for v in variants:changed[v][i]=project(mutate(r,v))
            s=copy.deepcopy(r);s['stats']['duration']*=2;s['sequence']['iats']=[t*2 for t in s['sequence']['iats']];stress[i]=project(s)
            # Harness control: an allowed value must reach the projected vector.
            control=copy.deepcopy(r);control['stats']['packet_count']+=1
            positive_controls+=int(not np.array_equal(project(control),clean[i]))
    assert len(seen)==40000;assert np.allclose(clean,data['x_hybrid'],rtol=1e-6,atol=1e-6)
    assert positive_controls==40000
    for v in variants:assert np.array_equal(clean,changed[v]),v
    refs=read(P1/'e1/run_001/reference_lock.json');locks=pd.read_csv(P1/'e1/run_001/policy_lock.csv');splits=read(hist/'split_manifest.json')
    checks=[];predictions=[];metrics=[]
    with threadpool_limits(limits=1):
        for split in splits:
            f=split['outer_fold'];ix=np.flatnonzero(np.isin(data['group'],split['groups']['evaluation']));y=data['y'][ix]
            bundle={v:joblib.load(hist/f'seed_42_fold_{f}/models/full_{v}.joblib') for v in ['temporal','stats']}
            bundle.update({k:joblib.load(P1/f'e1/run_001/fold_{f}/models/row100_20260920_{k}.joblib') for k in ['logistic','hgb']})
            baseline,base_state=all_outputs(clean[ix],f,bundle,refs,locks)
            previous=pd.read_csv(P1/f'e1/run_001/fold_{f}/predictions.csv.gz').set_index('sample_hash')
            for kind in ['logistic','hgb']:
                assert np.array_equal(baseline[kind+'_05'],previous.loc[data['sample_hash'][ix],f'row100_20260920_{kind}_primary'])
            for v in variants:
                out,state=all_outputs(changed[v][ix],f,bundle,refs,locks)
                row={'fold_id':f,'variant':v,'pairs':len(ix),'allowed_inputs_identical':True}
                for k in state:row[k+'_max_abs_delta']=float(np.max(abs(state[k].astype(float)-base_state[k].astype(float))))
                row['final_class_mismatches']=sum(int((out[k]!=baseline[k]).sum()) for k in out)
                evidence={'sample_hash':data['sample_hash'][ix],'allowed_inputs_equal':np.all(clean[ix]==changed[v][ix],axis=1),
                          'method_names':np.asarray(list(out)),
                          'original_classes':np.column_stack(list(baseline.values())),
                          'mutated_classes':np.column_stack(list(out.values()))}
                for k in state:
                    evidence[k+'_original']=base_state[k];evidence[k+'_mutated']=state[k]
                np.savez_compressed(dest/f'fold_{f}_{v}_semantic_pairs.npz',**evidence)
                checks.append(row);assert row['final_class_mismatches']==0 and all(row[k+'_max_abs_delta']==0 for k in state)
            shifted,_=all_outputs(stress[ix],f,bundle,refs,locks)
            for name,p in baseline.items():
                for setting,values in [('original',p),('time_scale_2',shifted[name])]:
                    metrics.append({'fold_id':f,'method':name,'condition':setting,'n':len(ix),**metric(y,values),
                        'corrected_vs_clean':int(((p!=y)&(values==y)).sum()),'damaged_vs_clean':int(((p==y)&(values!=y)).sum())})
            frame=pd.DataFrame({'sample_hash':data['sample_hash'][ix],'fold_id':f,'group':data['group'][ix],'label':y})
            for name,p in baseline.items():frame[name+'_original']=p;frame[name+'_stress']=shifted[name]
            frame.to_csv(dest/f'fold_{f}_stress_predictions.csv.gz',index=False,compression='gzip')
            print(f'field checks fold {f}: 3 x {len(ix)} pairs; fixed 2x timing stress evaluated',flush=True)
    table(dest/'paired_semantic_checks.csv',checks);table(dest/'stress_fold_metrics.csv',metrics)
    aggregate=[]
    for (method,condition),g in pd.DataFrame(metrics).groupby(['method','condition']):
        tn,fp,fn,tp=[int(g[k].sum()) for k in ['tn','fp','fn','tp']]
        aggregate.append({'method':method,'condition':condition,'n':int(g.n.sum()),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
            'macro_f1':tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1),'corrected_vs_clean':int(g.corrected_vs_clean.sum()),'damaged_vs_clean':int(g.damaged_vs_clean.sum())})
    table(dest/'stress_aggregate.csv',aggregate)
    dump(dest/'completion.json',{'status':'COMPLETED_SCOPED_PROJECTION_AND_TIMING_STRESS','finished':stamp(),'seconds':time.perf_counter()-start,
        'original_segments':40000,'new_isolation_pairs':120000,'historical_302_not_recounted':True,'positive_projection_controls':positive_controls,
        'feature_cache_parity':True,'scoring_labels_and_roles_unchanged':True,'full_runtime_FieldAudit_executed':False,'new_fits':0,
        'sensitivity_kind':'fixed-model timing distribution stress, not training ablation','script_sha256':sha(__file__)})
if __name__=='__main__':main()
