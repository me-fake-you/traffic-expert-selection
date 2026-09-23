"""One bounded, no-fitting transfer of already selected USTC policies.

IoT-23 publisher flows are a NEW evaluation unit, never old USTC segments.
All writers are exclusive; source data, checkpoints and earlier outputs stay intact.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping
from collections import Counter
import ast, csv, hashlib, json, math, struct, sys, time, ctypes
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
REPAIR = ROOT/'output/mad_etd_icassp2027_external_repair_001'
P1 = ROOT/'output/mad_etd_icassp2027_p1_20260920/e1/run_001'
HIST = ROOT/'output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001'
DELTA = ROOT/'output/mad_etd_icassp2027_upgrade_014/analysis'
OLDRAW = ROOT/'output/mad_etd_icassp2027_upgrade_013'
FEATURE = ROOT/'src/mad_etd/ustc_group_heldout_hybrid_w98.py'
METHODS = ['Temporal', 'Stats', 'Equal', 'Fixed', 'Fixed-Dev16',
           'Logistic-0.5', 'HGB-0.5', 'Logistic-Dev16', 'HGB-Dev16',
           'Logistic-Global4', 'HGB-Global4']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def table(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def npz(p,**data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f: np.savez_compressed(f,**data)

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: yield from csv.DictReader(f)

def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')

def log(*items):
    text=' '.join(str(x) for x in items)
    print(text,flush=True)
    with (OUT/'execution.log').open('a',encoding='utf-8') as f:f.write(now()+' '+text+'\n')

# Execute only the exact five frozen feature functions and original Segment class.
# The source module's data construction / fitting entry points are never imported.
env=dict(np=np,math=math,Any=Any,Mapping=Mapping,dataclass=dataclass,field=field,__name__=__name__)
for p,names in [(FEATURE,{'_log','_stats_vector','_change_rate','_prefix_features','_sequence_vector'}),
                (OLDRAW/'scripts/raw_all.py',{'Segment'}),
                (P1.parents[1]/'scripts/run_e1.py',{'meta'})]:
    for node in ast.parse(p.read_text(encoding='utf-8')).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),env)
env['env']=env
Segment=env['Segment']; meta=env['meta']

def parts():
    return sorted([p for run in ['native_existing_001','native_new_001']
                   for p in (REPAIR/run).glob('CTU-*')],key=lambda p:p.name)

def lock():
    original=read(OLDRAW/'raw_pcap_run/protocol_lock.json')
    for path,h in original['model_hashes'].items():assert sha(Path(path))==h
    assert sha(FEATURE)==original['feature_source_sha256']
    assert sha(DELTA/'selection_records.csv')=='1a2ac710db4c7a2ecc2bdbf5ab0e4ef17d70f7dcea1e5853e5d6464320ec9f44'
    manifests=[];files={};sources=[]
    for part in parts():
        manifest=part/'development_manifest.csv';c=Counter();n=0
        for r in rows(manifest):
            n+=1
            if r['annotation_qualified']=='True':
                assert r['status']=='qualified_author_annotation'
                c['qualified']+=1;c['packets']+=int(r['packet_count'])
        manifests.append(dict(scenario=part.name,part=str(part.relative_to(REPAIR)),
                              author_rows=n,qualified=c['qualified'],qualified_packets=c['packets']))
        for name in ['development_manifest.csv','packet_locators.csv','pcap_lookup_LOCAL_ONLY.json','completion.json']:
            p=part/name;files[str(p)]=sha(p)
        for x in read(part/'pcap_lookup_LOCAL_ONLY.json'):
            assert sha(Path(x['path']))==x['sha256']
            sources.append(dict(scenario=part.name,sha256=x['sha256'],bytes=x['bytes']))
    for p in [P1/'reference_lock.json',P1/'policy_lock.csv',DELTA/'selection_records.csv',
              REPAIR/'label_contract.json',REPAIR/'ROLE_DECISION.md',REPAIR/'source_groups.csv',
              FEATURE,OLDRAW/'scripts/raw_all.py',P1.parents[1]/'scripts/run_e1.py',Path(__file__)]:files[str(p)]=sha(p)
    total=sum(x['qualified'] for x in manifests);assert total==1159418
    config=dict(created=now(),status='LOCKED_BEFORE_TARGET_MODEL_INFERENCE',question='How do USTC-selected policies transfer, unchanged, to label-qualified IoT-23 publisher flows?',
        hypotheses='Descriptive transfer, no superiority claim, no target selection and no new diagnostic rule. Positive, zero and negative comparisons all retained.',
        unit='publisher_author_flow_NOT_old_60s_segment',population=manifests,total_qualified=total,methods=METHODS,
        fitting='NONE. Five existing USTC frozen bundles, reliabilities, normalizers and source-selected thresholds. All five retained; no best-fold or ensemble selection.',
        exposure='IoT source labels/structure already inspected for repair. Newly generated model evaluation, not a blind benchmark. Shared CTU parent and related lineages retained.',
        features='Exact historical Stats8 full-unit wire-byte aggregates and Temporal40 first min(n,64) observed packets; >=1 packet; first actual packet source direction; microsecond PCAP epoch converted to decimal seconds float; no endpoint, UID, family, label or epoch feature.',
        scope='Source AND unit transfer; not controlled cross-source replication with identical segmentation; no pooling with 40k or 625523 USTC results.',
        policies='Fixed abs(b-.5)*rS > .25*abs(a-.5)*rT; trigger abs(a-.5)<.495; learned switch only triggered disagreement; Dev16 and Global4 reuse original selected values.',
        outputs='Per-flow probabilities, arbitration scores, all 11 binary policies; all 5 bundles, 7 supported scenarios, pooled confusion counts and all per-scenario metrics.',
        metrics='Binary Macro-F1 with both classes; single-class scenario Macro-F1 NA; malicious recall; false-positive rate; FP/FN; corrected/introduced errors and switches vs Temporal. Pooled metrics are descriptive. Fold ranges are not confidence intervals.',
        comparisons=['all policies versus Temporal','Logistic/HGB 0.5 and Dev16 versus Fixed and Fixed-Dev16','Dev16 versus Global4'],
        pilot='First 2048 qualified manifest rows of scenario with most qualified flows, chosen before inference; label-free timing of features and all five bundles. Conservative estimate <=7200 s, process peak <=12 GiB, disk availability checked before full work.',
        budget_seconds=7200,memory_limit_bytes=12*1024**3,native_threads=1,batch_size=16384,
        stop='Input hash, digest, packet count, feature finiteness or source integrity mismatch stops that path. No score-driven exclusions, fitting, threshold change, extra model or extra source.',
        model_hashes=original['model_hashes'],input_hashes=files,pcap_sources=sources,new_training=0,
        cost='This is shared-feature frozen inference, not an end-to-end method speed benchmark. No acceleration conclusion.')
    dump(OUT/'protocol_lock.json',config)
    log('PROTOCOL_LOCKED',total,'native flows;',len(METHODS),'policies x 5 bundles; no fitting')

def check_lock():
    config=read(OUT/'protocol_lock.json')
    amendment_path=OUT/'protocol_amendment_001.json'
    if amendment_path.exists():
        amendment=read(amendment_path)
        assert sha(OUT/'protocol_lock.json')==amendment['original_protocol_sha256']
        assert sha(OUT/'scripts/frozen_transfer_preflight_001.py')==amendment['original_script_sha256']
        config['input_hashes'][str(Path(__file__))]=amendment['revised_script_sha256']
        config['budget_seconds']=amendment['authorized_budget_seconds']
    for p,h in {**config['input_hashes'],**config['model_hashes']}.items():assert sha(Path(p))==h,Path(p).name
    return config

def build_features(part,dest,limit=None):
    started=time.perf_counter();index={};labels=[];counts=[];ids=[];digest=[]
    for r in rows(part/'development_manifest.csv'):
        if r['annotation_qualified']!='True':continue
        assert r['status']=='qualified_author_annotation'
        fid=r['sample_id'];assert fid not in index
        index[fid]=len(ids);ids.append(fid);labels.append(int(r['binary_label']))
        counts.append(int(r['packet_count']));digest.append(r['packet_digest_sha256'])
        if limit and len(ids)==limit:break
    n=len(ids);assert n
    xt=np.full((n,40),np.nan,dtype=np.float32);xs=np.full((n,8),np.nan,dtype=np.float32)
    active={};completed=np.zeros(n,dtype=bool);packet_count=0;done_count=0;calc_s=0.;read_s=0.;active_peak=0
    mapping={r['sha256']:Path(r['path']) for r in read(part/'pcap_lookup_LOCAL_ONLY.json')}
    handles={h:p.open('rb') for h,p in mapping.items()}
    for f in handles.values():
        head=f.read(24);assert head[:4]==b'\xd4\xc3\xb2\xa1' and struct.unpack('<I',head[20:24])[0]==1
    try:
        for loc in rows(part/'packet_locators.csv'):
            fid=loc['flow_id'];i=index.get(fid)
            if i is None:continue
            assert not completed[i]
            tick=time.perf_counter();f=handles[loc['pcap_sha256']];f.seek(int(loc['record_offset']))
            header=f.read(16);assert len(header)==16
            sec,micro,cap,wire=struct.unpack('<IIII',header);raw=f.read(cap)
            assert len(raw)==cap==int(loc['caplen'])==wire
            us=sec*1000000+micro;assert us==int(loc['epoch_us'])
            off=14;et=int.from_bytes(raw[12:14],'big')
            while et in (0x8100,0x88a8):et=int.from_bytes(raw[off+2:off+4],'big');off+=4
            assert et==0x0800
            ip=raw[off:];assert ip[0]>>4==4 and int.from_bytes(ip[6:8],'big')&0x3fff==0 and ip[9] in (6,17)
            ihl=(ip[0]&15)*4;src=(ip[12:16],int.from_bytes(ip[ihl:ihl+2],'big'))
            stamp=float(f'{sec}.{micro:06d}')
            if i not in active:active[i]=[Segment(fid,stamp,stamp,src,'tcp' if ip[9]==6 else 'udp'),hashlib.sha256(),us]
            seg,h,last=active[i];assert us>=last;active[i][2]=us
            h.update(struct.pack('!Q',us)+raw);seg.add(stamp,src,wire);packet_count+=1
            assert seg.n<=counts[i]
            read_s+=time.perf_counter()-tick
            if seg.n==counts[i]:
                assert h.hexdigest()==digest[i]
                tick=time.perf_counter();xt[i]=seg.features('temporal');xs[i]=seg.features('stats');calc_s+=time.perf_counter()-tick
                completed[i]=True;done_count+=1;del active[i]
                if done_count%100000==0:log('FEATURE_PROGRESS',part.name,done_count,'/',n)
            active_peak=max(active_peak,len(active))
    finally:
        for f in handles.values():f.close()
    assert completed.all() and not active and packet_count==sum(counts)
    assert np.isfinite(xt).all() and np.isfinite(xs).all()
    ids=np.asarray(ids,dtype='S64')
    npz(dest/'features.npz',sample_id=ids,temporal=xt,stats=xs,packet_count=np.asarray(counts,dtype=np.int64))
    # Stored separately and not passed into prediction. Labels originate from repaired author annotations.
    npz(dest/'labels.npz',sample_id=ids,label=np.asarray(labels,dtype=np.uint8))
    receipt=dict(scenario=part.name,flows=n,packets=packet_count,feature_dimensions=[40,8],
                 every_flow_raw_digest_checked=True,every_locator_header_and_count_checked=True,
                 feature_seconds=calc_s,packet_read_seconds=read_s,total_seconds=time.perf_counter()-started,
                 active_flow_peak=active_peak,features_sha256=sha(dest/'features.npz'),labels_sha256=sha(dest/'labels.npz'))
    dump(dest/'features_receipt.json',receipt);log('FEATURES_COMPLETE',part.name,n,round(receipt['total_seconds'],2),'s')
    return receipt

def load_bundle(f):
    refs=read(P1/'reference_lock.json')[str(f)]
    locks=[r for r in rows(P1/'policy_lock.csv') if int(r['outer_fold'])==f and r['condition']=='row100' and r['subset_seed']=='20260920']
    dev={r['kind']:(float(r['selected_low']),float(r['selected_high'])) for r in locks};assert set(dev)=={'logistic','hgb'}
    selected={r['method']:r for r in rows(DELTA/'selection_records.csv') if int(r['fold_id'])==f}
    expert={v:joblib.load(HIST/f'seed_42_fold_{f}/models/full_{v}.joblib') for v in ['temporal','stats']}
    arb={k:joblib.load(P1/f'fold_{f}/models/row100_20260920_{k}.joblib') for k in ['logistic','hgb']}
    assert expert['temporal'].n_features_in_==40 and expert['stats'].n_features_in_==8
    assert all(x.n_features_in_==11 for x in arb.values())
    return expert,arb,refs,dev,selected

def predict(xt,xs,bundle):
    expert,arb,refs,dev,selected=bundle
    a=expert['temporal'].predict_proba(xt)[:,1];b=expert['stats'].predict_proba(xs)[:,1]
    first=a>=.5;second=b>=.5;trigger=abs(a-.5)<.495;conflict=trigger&(first!=second)
    ra=refs['reliability_temporal'];rb=refs['reliability_stats']
    pred=np.tile(first[:,None],(1,len(METHODS))).astype(np.uint8)
    pred[:,1]=second;pred[:,2]=(a+b)/2>=.5
    change=conflict&(abs(b-.5)*rb>.25*abs(a-.5)*ra);pred[change,3]=second[change]
    sel=selected['Fixed-Dev16'];change=np.zeros(len(a),dtype=bool)
    for direction in (0,1):
        value=sel[f'parameter{direction}']
        if value!='no_switch':change|=conflict&(first==direction)&(abs(b-.5)*rb>float(value)*abs(a-.5)*ra)
    pred[change,4]=second[change]
    q=np.full((len(a),2),np.nan)
    for j,(kind,name) in enumerate([('logistic','Logistic'),('hgb','HGB')]):
        if conflict.any():q[conflict,j]=arb[kind].predict_proba(meta(a[conflict],b[conflict],ra,rb))[:,1]
        global_sel=selected[name+'-Global4']
        for point,low,high in [('0.5',.5,.5),('Dev16',*dev[kind]),('Global4',float(global_sel['parameter0']),float(global_sel['parameter1']))]:
            switch=conflict&(q[:,j]>=np.where(first,high,low));pred[switch,METHODS.index(name+'-'+point)]=second[switch]
    assert ((pred==0)|(pred==1)).all()
    return dict(predictions=pred,first_probability=a,second_probability=b,arbiter_scores=q,trigger=trigger)

def pilot():
    config=check_lock();p=max(config['population'],key=lambda r:r['qualified']);dest=OUT/'pilot'
    receipt=build_features(REPAIR/p['part'],dest,limit=2048)
    data=np.load(dest/'features.npz');tic=time.perf_counter();infer_s=0
    with threadpool_limits(limits=1):
        for f in range(5):
            bundle=load_bundle(f);tick=time.perf_counter();result=predict(data['temporal'],data['stats'],bundle);infer_s+=time.perf_counter()-tick
            assert len(result['predictions'])==2048
    # Two-times safety factor; reading the largest locator file already happened once.
    scale=config['total_qualified']/2048
    projected=2*(receipt['feature_seconds']+receipt['packet_read_seconds']+infer_s)*scale+receipt['total_seconds']*8
    peak=peak_bytes();passed=projected<=config['budget_seconds'] and peak<=config['memory_limit_bytes']
    status=dict(created=now(),sample_rule='first 2048 qualified rows of largest scenario; no quality scores computed',
                inference_seconds_all_five=infer_s,load_plus_inference_seconds=time.perf_counter()-tic,
                projected_seconds_with_safety_factor=projected,process_peak_bytes=peak,gate_passed=passed)
    dump(dest/'pilot_gate.json',status);log('PILOT',json.dumps(status));assert passed,'Budget gate failed; do not run full path'

def peak_bytes():
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    c=PMC();c.cb=ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
    assert c.PeakWorkingSetSize>0
    return c.PeakWorkingSetSize

def amend():
    original=read(OUT/'protocol_lock.json');old=read(OUT/'pilot/pilot_gate.json')
    assert not old['gate_passed'] and not (OUT/'features_complete.json').exists()
    data=np.load(OUT/'pilot/features.npz')
    with threadpool_limits(limits=1):
        bundles=[load_bundle(f) for f in range(5)]
        for bundle in bundles:predict(data['temporal'],data['stats'],bundle)
        peak=peak_bytes()
    amendment=dict(created=now(),reason='Repair 64-bit Windows process-memory API signature and label-file hash verification; retain failed pilot. User explicitly authorized <=3 hours before full run. No quality scores viewed.',
        authorization='User reply: 允许最多3小时，完成既定评价',authorized_budget_seconds=10800,
        original_protocol_sha256=sha(OUT/'protocol_lock.json'),original_script_sha256=sha(OUT/'scripts/frozen_transfer_preflight_001.py'),
        revised_script_sha256=sha(Path(__file__)),unchanged='population, model hashes, features, all policies, source-only chosen parameters, score definitions',
        scope='Same existing 2048 pilot rows rerun without labels/scores solely to verify memory; original conservative time estimate retained.')
    assert amendment['original_script_sha256']==original['input_hashes'][str(Path(__file__))]
    dump(OUT/'protocol_amendment_001.json',amendment)
    gate=dict(created=now(),gate_passed=old['projected_seconds_with_safety_factor']<=10800 and peak<=original['memory_limit_bytes'],
              projected_seconds_with_safety_factor=old['projected_seconds_with_safety_factor'],rechecked_process_peak_bytes=peak,quality_scores_computed=False)
    dump(OUT/'pilot/pilot_gate_amended.json',gate);log('AMENDED_PILOT_GATE',json.dumps(gate));assert gate['gate_passed']

def budget_guard():
    config=read(OUT/'protocol_lock.json');budget=10800 if (OUT/'protocol_amendment_001.json').exists() else config['budget_seconds']
    start=read(OUT/'full_run_start.json')['epoch']
    assert time.time()-start < budget,'Authorized wall budget reached; stop, retain partial outputs'
    assert peak_bytes() <= config['memory_limit_bytes'],'Memory guard reached'

def features():
    config=check_lock();gate=OUT/'pilot/pilot_gate_amended.json'
    assert read(gate if gate.exists() else OUT/'pilot/pilot_gate.json')['gate_passed']
    if not (OUT/'full_run_start.json').exists():dump(OUT/'full_run_start.json',dict(created=now(),epoch=time.time(),budget_seconds=config['budget_seconds']))
    receipts=[]
    for r in config['population']:
        budget_guard()
        if not r['qualified']:continue
        dest=OUT/'features'/r['scenario']
        if (dest/'features_receipt.json').exists():
            receipt=read(dest/'features_receipt.json');assert sha(dest/'features.npz')==receipt['features_sha256']
        else:receipt=build_features(REPAIR/r['part'],dest)
        assert receipt['flows']==r['qualified'];receipts.append(receipt)
    assert sum(r['flows'] for r in receipts)==config['total_qualified']
    dump(OUT/'features_complete.json',dict(created=now(),scenarios=receipts,process_peak_bytes=peak_bytes()))

def infer():
    config=check_lock();complete=read(OUT/'features_complete.json');receipts=[]
    with threadpool_limits(limits=1):
        bundles=[load_bundle(f) for f in range(5)]
        for r in complete['scenarios']:
            budget_guard()
            src=OUT/'features'/r['scenario'];assert sha(src/'features.npz')==r['features_sha256']
            data=np.load(src/'features.npz');xt=data['temporal'];xs=data['stats'];ids=data['sample_id']
            for f,bundle in enumerate(bundles):
                dest=OUT/'predictions'/r['scenario']/f'fold_{f}.npz';tick=time.perf_counter();result={}
                for start in range(0,len(ids),config['batch_size']):
                    budget_guard()
                    batch=predict(xt[start:start+config['batch_size']],xs[start:start+config['batch_size']],bundle)
                    for k,v in batch.items():result.setdefault(k,[]).append(v)
                result={k:np.concatenate(v) for k,v in result.items()}
                npz(dest,sample_id=ids,**result)
                receipts.append(dict(scenario=r['scenario'],bundle=f,flows=len(ids),seconds=time.perf_counter()-tick,sha256=sha(dest)))
                log('INFERENCE_COMPLETE',r['scenario'],'bundle',f,len(ids),round(receipts[-1]['seconds'],2),'s; not scored')
    assert len(receipts)==35
    dump(OUT/'predictions_complete.json',dict(created=now(),receipt=receipts,labels_used_in_inference=False,process_peak_bytes=peak_bytes()))

def metric(y,p,first):
    tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tp=int(((y==1)&(p==1)).sum())
    c=int(((p==y)&(first!=y)).sum());d=int(((p!=y)&(first==y)).sum());s=int((p!=first).sum())
    assert c+d==s
    return dict(n=len(y),benign=tn+fp,malicious=tp+fn,tn=tn,fp=fp,fn=fn,tp=tp,corrected=c,introduced=d,switches=s,net_corrected=c-d)

def finish_metric(r):
    tn,fp,fn,tp=[r[k] for k in ['tn','fp','fn','tp']]
    return {**r,'macro_f1':(.5*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))) if (tn+fp and tp+fn) else None,
            'malicious_recall':tp/(tp+fn) if tp+fn else None,'false_positive_rate':fp/(tn+fp) if tn+fp else None}

def score():
    config=check_lock();pred_complete=read(OUT/'predictions_complete.json');scores=[];pooled={};comparisons=[]
    for r in pred_complete['receipt']:
        scenario=r['scenario'];f=r['bundle'];p=OUT/'predictions'/scenario/f'fold_{f}.npz';assert sha(p)==r['sha256']
        label_path=OUT/'features'/scenario/'labels.npz'
        assert sha(label_path)==read(OUT/'features'/scenario/'features_receipt.json')['labels_sha256']
        pred=np.load(p);lab=np.load(label_path)
        assert np.array_equal(pred['sample_id'],lab['sample_id']);y=lab['label'];matrix=pred['predictions'];first=matrix[:,0]
        assert matrix.shape==(len(y),len(METHODS))
        for j,m in enumerate(METHODS):
            met=metric(y,matrix[:,j],first);scores.append(dict(scenario=scenario,bundle=f,method=m,**finish_metric(met)))
            pooled.setdefault((f,m),Counter()).update(met)
        for a,b in [('Fixed','Temporal'),('Fixed-Dev16','Fixed')]+[(k+p,b) for k in ['Logistic','HGB'] for p in ['-0.5','-Dev16'] for b in ['Fixed','Fixed-Dev16']]+[(k+'-Dev16',k+'-Global4') for k in ['Logistic','HGB']]:
            ma=metric(y,matrix[:,METHODS.index(a)],first);mb=metric(y,matrix[:,METHODS.index(b)],first)
            comparisons.append(dict(scenario=scenario,bundle=f,method=a,reference=b,delta_fp=ma['fp']-mb['fp'],delta_fn=ma['fn']-mb['fn'],different_predictions=int((matrix[:,METHODS.index(a)]!=matrix[:,METHODS.index(b)]).sum())))
    pool=[dict(scenario='ALL_qualified_author_flows',bundle=f,method=m,**finish_metric(dict(v))) for (f,m),v in pooled.items()]
    assert len(scores)==385 and len(pool)==55 and all(r['n']==config['total_qualified'] for r in pool)
    table(OUT/'results/scenario_metrics.csv',scores);table(OUT/'results/pooled_metrics.csv',pool);table(OUT/'results/scenario_contrasts.csv',comparisons)
    poolcompar=[]
    for f in range(5):
        subset={r['method']:r for r in pool if r['bundle']==f}
        for m in METHODS:
            for ref in ['Temporal','Fixed','Fixed-Dev16']:
                a,b=subset[m],subset[ref]
                poolcompar.append(dict(bundle=f,method=m,reference=ref,delta_fp=a['fp']-b['fp'],delta_fn=a['fn']-b['fn'],delta_macro_f1_pp=100*(a['macro_f1']-b['macro_f1'])))
    table(OUT/'results/pooled_contrasts.csv',poolcompar)
    dump(OUT/'completion.json',dict(created=now(),status='ALL_PREDEFINED_FROZEN_TRANSFER_RESULTS_COMPLETE',qualified_flows=config['total_qualified'],scenarios=7,bundles=5,policies=11,per_scenario_metric_rows=len(scores),pooled_metric_rows=len(pool),new_training=0,new_selection=0,protocol_sha256=sha(OUT/'protocol_lock.json'),separate_unit_not_blind=True))
    log('SCORING_COMPLETE',len(scores),'scenario rows;',len(pool),'pooled rows; all results retained')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    command=sys.argv[1]
    try:globals()[command]()
    except Exception as exc:
        log('FAIL',command,type(exc).__name__,str(exc));raise
