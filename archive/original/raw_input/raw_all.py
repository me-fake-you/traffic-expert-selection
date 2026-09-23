"""Actual raw-PCAP all-eligible execution, one serial method at a time."""
from common_new import *
import argparse,ast,math,heapq,csv,gzip,subprocess,ctypes,threading
from typing import Any,Mapping
from dataclasses import dataclass,field
from collections import Counter
from mad_etd.extract_ustc import TSHARK_FIELDS,resolve_tshark,tshark_version,_endpoint,_stream_key,_labels_for
sys.path.insert(0,str(R2/'experiments'))
from win_memory import peak_rss,Counters,psapi

METHODS=['temporal','strong_single','equal_average','fixed','logistic_05','hgb_05','logistic_dev','hgb_dev']
FEATURE_SOURCE=ROOT/'src/mad_etd/ustc_group_heldout_hybrid_w98.py'
env=dict(np=np,math=math,Any=Any,Mapping=Mapping)
for node in ast.parse(FEATURE_SOURCE.read_text(encoding='utf-8')).body:
    if isinstance(node,ast.FunctionDef) and node.name in {'_log','_stats_vector','_change_rate','_prefix_features','_sequence_vector'}:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(FEATURE_SOURCE),'exec'),env)

@dataclass
class Segment:
    identity:str;first:float;last:float;origin:tuple;transport:str
    n:int=0;total:int=0;outbound:int=0;square:int=0;version:int=0
    lengths:list=field(default_factory=list);directions:list=field(default_factory=list);times:list=field(default_factory=list)
    def add(self,t,src,length):
        d=1 if src==self.origin else -1;self.n+=1;self.total+=length;self.square+=length*length;self.outbound+=length if d==1 else 0
        self.last=t;self.version+=1
        if len(self.lengths)<64:self.lengths.append(length);self.directions.append(d);self.times.append(t)
    def features(self,view):
        if view=='temporal':
            record={'sequence':{'packet_lengths':self.lengths,'directions':self.directions,'iats':[max(0.,b-a) for a,b in zip(self.times,self.times[1:])]}}
            return np.array(env['_sequence_vector'](record),dtype=np.float32)
        mean=self.total/self.n
        record={'stats':dict(packet_count=self.n,total_bytes=self.total,outbound_bytes=self.outbound,inbound_bytes=self.total-self.outbound,
            outbound_ratio=self.outbound/max(self.total,1),mean_packet_length=mean,packet_length_variance=max(0.,self.square/self.n-mean*mean),duration=self.last-self.first)}
        return np.array(env['_stats_vector'](record),dtype=np.float32)

def parse_capture(path,relative,counts,stderr_path,child_memory):
    cmd=[resolve_tshark(),'-n','-r',str(path),'-T','fields','-E','separator=\t','-E','occurrence=f']
    for f in TSHARK_FIELDS:cmd+=['-e',f]
    active={};heap=[];segments={}
    with stderr_path.open('x',encoding='utf-8') as error:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=error,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
        stop=threading.Event()
        def monitor():
            while not stop.is_set():
                c=Counters();c.cb=ctypes.sizeof(c)
                if psapi.GetProcessMemoryInfo(int(proc._handle),ctypes.byref(c),c.cb):
                    child_memory['child_peak_bytes']=max(child_memory.get('child_peak_bytes',0),c.PeakWorkingSetSize)
                    child_memory['sampled_combined_peak_upper_bytes']=max(child_memory.get('sampled_combined_peak_upper_bytes',0),peak_rss()+c.WorkingSetSize)
                stop.wait(.2)
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        try:
            for line in proc.stdout:
                counts['frames_total']+=1
                v=line.rstrip('\n').split('\t');v+=['']*(16-len(v))
                t,s4,s6,ts,us,d4,d6,td,ud,l,protocol,tcp,udp,ip,tls,cipher=v[:16]
                if not t or not l:counts['frames_missing_time_or_length']+=1;continue
                try:stamp=float(t);length=int(l);assert math.isfinite(stamp) and length>=0
                except (ValueError,AssertionError):counts['frames_parse_failed']+=1;continue
                while heap and heap[0][0]<=stamp-60.:
                    _,key,ver=heapq.heappop(heap);raw=active.get(key)
                    if raw is None or raw.version!=ver:continue
                    del active[key];segments[key]=segments.get(key,0)+1;yield raw
                src=_endpoint(s4,s6,ts,us);dst=_endpoint(d4,d6,td,ud);key=_stream_key(src,dst,tcp,udp,ip)
                if key not in active:
                    sid=hashlib.sha256(f'{relative}|{key[0]}|{key[1]}|{stamp:.6f}|{segments.get(key,0)}'.encode()).hexdigest()[:20]
                    h=hashlib.sha256(('mad_etd_external_multiagent_v5_1:'+sid).encode()).hexdigest()
                    active[key]=Segment(h,stamp,stamp,src,key[0])
                raw=active[key];raw.add(stamp,src,length);heapq.heappush(heap,(stamp,key,raw.version));counts['frames_sessionized']+=1
            proc.stdout.close();code=proc.wait()
            if code:raise RuntimeError(f'tshark exit {code}; see {stderr_path}')
            for raw in sorted(active.values(),key=lambda x:x.first):yield raw
        finally:
            stop.set();monitor_thread.join(timeout=2)
            if proc.poll() is None:proc.terminate();proc.wait()

class Bundle:
    def __init__(self,f,hist,refs,locks):
        self.models={v:joblib.load(hist/f'seed_42_fold_{f}/models/full_{v}.joblib') for v in ['temporal','stats']}
        self.arbs={k:joblib.load(P1/f'e1/run_001/fold_{f}/models/row100_20260920_{k}.joblib') for k in ['logistic','hgb']}
        self.r=refs[str(f)];self.locks={k:locks[(locks.outer_fold==f)&(locks.condition=='row100')&(locks.kind==k)].iloc[0] for k in self.arbs}
        self.strong=read(R2/f'results_r2/a_001/fold_{f}/locked_references.json')

def infer(batch,bundle,method):
    stages={};caches=[{} for _ in batch]
    def features(ids,view):
        values=[]
        for i in ids:
            if view not in caches[i]:caches[i][view]=batch[i].features(view)
            values.append(caches[i][view])
        return np.stack(values)
    def timed(name,fn):
        start=time.perf_counter_ns();v=fn();stages[name]=time.perf_counter_ns()-start;return v
    n=len(batch);ids=np.arange(n);view='temporal';threshold=.5
    if method=='strong_single':view=bundle.strong['strong_tuned'];threshold=bundle.strong['thresholds'][view]['threshold']
    x=timed('first_features_ns',lambda:features(ids,view));a=timed('first_predict_ns',lambda:bundle.models[view].predict_proba(x)[:,1])
    pred=(a>=threshold).astype(np.uint8);trigger=np.zeros(n,dtype=bool);q=np.full(n,np.nan);b=np.full(n,np.nan)
    if method not in ['temporal','strong_single']:
        trigger=timed('trigger_ns',lambda:np.ones(n,dtype=bool) if method=='equal_average' else abs(a-.5)<.495)
        active=np.flatnonzero(trigger)
        if len(active):
            x2=timed('second_features_ns',lambda:features(active,'stats'));b[active]=timed('second_predict_ns',lambda:bundle.models['stats'].predict_proba(x2)[:,1])
            tick=time.perf_counter_ns();ra=bundle.r['reliability_temporal'];rb=bundle.r['reliability_stats']
            if method=='equal_average':pred[active]=(a[active]+b[active])/2>=.5
            elif method=='fixed':
                change=active[abs(b[active]-.5)*rb>.25*abs(a[active]-.5)*ra];pred[change]=b[change]>=.5
            else:
                kind,point=method.split('_');conflict=active[(a[active]>=.5)!=(b[active]>=.5)]
                if len(conflict):
                    q[conflict]=bundle.arbs[kind].predict_proba(meta(a[conflict],b[conflict],ra,rb))[:,1]
                    lo=hi=.5
                    if point=='dev':lo=bundle.locks[kind].selected_low;hi=bundle.locks[kind].selected_high
                    change=conflict[q[conflict]>=np.where(a[conflict]>=.5,hi,lo)];pred[change]=b[change]>=.5
            stages['arbitration_ns']=time.perf_counter_ns()-tick
        assert all(('stats' in caches[i])==bool(trigger[i]) for i in ids)
    return pred,a,b,trigger,q,stages

def initialize():
    dest=OUT/'raw_pcap_run';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'protocol_lock.json').exists():return
    cfg,data,pv,hist=load_inputs();plans=read(hist/'split_manifest.json')
    sources=pd.read_csv(R2/'cost_pcap_manifest.csv');rows=[]
    for rec in sources.to_dict('records'):
        path=Path(rec['raw_path']);h=sha(path);assert h==rec['observed_sha256']
        labels=_labels_for(Path(rec['capture']));g=('benign:'+labels['application']) if labels['binary']=='benign' else ('malware:'+labels['family'])
        folds=[s['outer_fold'] for s in plans if g in s['groups']['evaluation']];assert len(folds)==1
        f=folds[0];assert all(g not in s['groups'][r] for s in plans if s['outer_fold']==f for r in ['train','calibration','selection'])
        rows.append({'capture':rec['capture'],'raw_path':str(path),'bytes':path.stat().st_size,'sha256':h,'group':g,'label':int(labels['binary']=='malicious'),'evaluation_fold':f})
    table(dest/'source_manifest.csv',rows)
    model_paths=list((hist).glob('seed_42_fold_*/models/full_*.joblib'))+list((P1/'e1/run_001').glob('fold_*/models/row100_20260920_*.joblib'))
    lock={'created':stamp(),'source_manifest_sha256':sha(dest/'source_manifest.csv'),'methods':METHODS,'batch_size':128,'native_threads':1,
          'eligibility':'Every segment with at least one valid frame and supported TCP/UDP or numbered IP fallback; ip-unknown is emitted unsupported. No sampling/caps/score filtering.',
          'session':'historical tshark stream identity, 60 s idle, capture order, first source direction; first 64 packet prefix; negative IAT clamped as original',
          'fragmentation':'unchanged tshark default dissection; missing transport uses original IP fallback; no new defragmentation or payload execution',
          'labels':'inherited USTC capture-directory binary/application-family labels, not packet-level manual truth; no model-filled labels',
          'roles':'each capture uses the one outer model excluding its application/family from expert, reliability, arbiter and threshold selection dependencies; enlarged same-source exposed population',
          'cost':'Each method rereads every PCAP independently; one shared parser per method, raw aggregate counters common, expert feature transforms/prediction lazy. All methods use same raw segmentation.',
          'latency':'batch wall intervals from requesting raw segments to flushing output, not amortized single-request or online network latency; entire-run wall also measured',
          'memory':'native process lifetime peak plus sampled child peak; combined uses process peak plus live child working set, an upper bound not exact simultaneous RSS',
          'warmup':'up to 16 original training-role feature rows per fold, not timed; no new fitting',
          'cold':'suite checkpoint load recorded separately; file cache uncontrolled; no GPU',
          'pilot':'lexically first capture, no quality scoring, training fold 0, Temporal only. Estimate; stop full launch if projected >4 CPU-hours or memory >12 GiB',
          'model_hashes':{str(p):sha(p) for p in model_paths},'feature_source_sha256':sha(FEATURE_SOURCE),'tshark':tshark_version(),
          'strong_single':'existing R2 development-tuned single expert, original window and selection budget disclosed; no new selection',
          'full_status_requires':'all source EOFs, all segment terminal states, every eligible segment output by all planned methods; parity on eligible original40k IDs',
          'new_training':0}
    dump(dest/'protocol_lock.json',lock)

def execute(method,pilot=False):
    initialize();base=OUT/'raw_pcap_run';dest=base/('pilot' if pilot else method);dest.mkdir(exist_ok=False)
    cfg,data,pv,hist=load_inputs();refs=read(P1/'e1/run_001/reference_lock.json');locks=pd.read_csv(P1/'e1/run_001/policy_lock.csv')
    load_start=time.perf_counter();bundles={f:Bundle(f,hist,refs,locks) for f in range(5)};load_s=time.perf_counter()-load_start
    sources=pd.read_csv(base/'source_manifest.csv');sources=sources.iloc[:1] if pilot else sources
    plans=read(hist/'split_manifest.json')
    for f,bundle in bundles.items():
        ids=np.flatnonzero(np.isin(data['group'],plans[f]['groups']['train']))[:16]
        for view in bundle.models:bundle.models[view].predict_proba((data['x_stats'] if view=='stats' else data['x_hybrid'][:,8:])[ids])
        for arb in bundle.arbs.values():arb.predict_proba(meta(np.full(16,.5),np.full(16,.5),.5,.5))
    old={}
    for f in range(5):
        d=pd.read_csv(P1/f'e1/run_001/fold_{f}/predictions.csv.gz',float_precision='round_trip')
        key={'temporal':'first_only','equal_average':'static_equal','fixed':'same_trigger_fixed','logistic_05':'row100_20260920_logistic_primary','hgb_05':'row100_20260920_hgb_primary','logistic_dev':'row100_20260920_logistic_selected','hgb_dev':'row100_20260920_hgb_selected'}.get(method)
        if key:old.update(zip(d.sample_hash,d[key]))
    if method=='strong_single':
        d=pd.read_csv(R2/'results_r2/a_001/predictions.csv.gz',usecols=['sample_hash','strong_single_tuned']);old=dict(zip(d.sample_hash,d.strong_single_tuned))
    start=time.perf_counter();coverage=[];latencies=[];stages=Counter();confusion=np.zeros(4,dtype=np.int64);parity=0;seen=set();child_memory={}
    with gzip.open(dest/'predictions.csv.gz','wt',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_hash','capture','group','evaluation_fold','label','status','prediction','trigger','first_probability','second_probability','arbiter_score','packets'])
        for rec in sources.to_dict('records'):
            cap_start=time.perf_counter();counts=Counter({k:0 for k in ['frames_total','frames_missing_time_or_length','frames_parse_failed','frames_sessionized','segments_total','classified','unsupported','second_expert_rows']});f=0 if pilot else int(rec['evaluation_fold']);batch=[];tick=time.perf_counter_ns()
            def flush():
                nonlocal batch,tick,parity,confusion
                if not batch:return
                result=infer(batch,bundles[f],method);pred,a,b,tr,q,timing=result
                for key,v in timing.items():stages[key]+=v
                for i,raw in enumerate(batch):
                    assert raw.identity not in seen;seen.add(raw.identity)
                    writer.writerow([raw.identity,rec['capture'],rec['group'],f,rec['label'],'classified',int(pred[i]),int(tr[i]),a[i],b[i],q[i],raw.n])
                    if not pilot and raw.identity in old:assert int(pred[i])==int(old[raw.identity]),(raw.identity,method);parity+=1
                stream.flush();latencies.append({'capture':rec['capture'],'rows':len(batch),'raw_to_flush_batch_ns':time.perf_counter_ns()-tick})
                if not pilot:
                    y=np.full(len(batch),rec['label']);m=metric(y,pred);confusion+=np.array([m[k] for k in ['tn','fp','fn','tp']])
                counts['classified']+=len(batch);counts['second_expert_rows']+=int(tr.sum());batch=[];tick=time.perf_counter_ns()
            for raw in parse_capture(Path(rec['raw_path']),rec['capture'],counts,dest/f'parser_{len(coverage):02d}.stderr.log',child_memory):
                counts['segments_total']+=1
                if raw.transport=='ip-unknown':
                    counts['unsupported']+=1;writer.writerow([raw.identity,rec['capture'],rec['group'],f,rec['label'],'unsupported_non_ip','','','','','',raw.n]);continue
                batch.append(raw)
                if len(batch)==128:flush()
            flush();counts['source_eof']=1
            assert counts['segments_total']==counts['classified']+counts['unsupported']
            assert counts['frames_total']==counts['frames_sessionized']+counts['frames_missing_time_or_length']+counts['frames_parse_failed']
            coverage.append({**rec,**counts,'wall_seconds':time.perf_counter()-cap_start});table(dest/'coverage.csv',coverage)
            dump(dest/'progress.json',{'method':method,'captures_completed':len(coverage),'of':len(sources),'classified':sum(r['classified'] for r in coverage),'seconds':time.perf_counter()-start,'status':'RUNNING'})
            print(f'{method}: capture {len(coverage)}/{len(sources)}, {counts["classified"]} outputs, {time.perf_counter()-cap_start:.1f}s',flush=True)
    wall=time.perf_counter()-start;n=sum(r['classified'] for r in coverage);table(dest/'batch_latency.csv',latencies)
    tn,fp,fn,tp=map(int,confusion);metrics={'tn':tn,'fp':fp,'fn':fn,'tp':tp,'macro_f1':tp/max(2*tp+fp+fn,1)+tn/max(2*tn+fp+fn,1)} if not pilot else None
    result={'status':'PILOT_NO_QUALITY_SCORE' if pilot else 'RAW_PCAP_ALL_ELIGIBLE_EXECUTED_FOR_THIS_METHOD','method':method,'started':stamp(),'command':sys.argv,
            'full_source_files':len(coverage),'source_bytes':int(sources.bytes.sum()),'classified':n,'unsupported':sum(r['unsupported'] for r in coverage),'scored':0 if pilot else n,
            'wall_seconds_raw_entry_to_final_output':wall,'checkpoint_load_seconds_separate':load_s,'throughput_segments_per_second':n/wall,
            'batch_p50_ms':float(np.quantile([v['raw_to_flush_batch_ns']/1e6 for v in latencies],.5)),
            'batch_p95_ms':float(np.quantile([v['raw_to_flush_batch_ns']/1e6 for v in latencies],.95)),
            'batch_mean_ms':float(np.mean([v['raw_to_flush_batch_ns']/1e6 for v in latencies])),
            'latency_unit':'actual variable-sized (<=128) raw-entry-to-output batch; not single-request latency','feature_and_inference_stage_ns':stages,
            'process_peak_bytes':peak_rss(),**child_memory,'old40k_parity_matches':parity,'old40k_population':40000,'metrics':metrics,
            'eligible_id_set_sha256':hashlib.sha256('\n'.join(sorted(seen)).encode()).hexdigest(),'script_sha256':sha(__file__),
            'natural_population_not_rebalanced':True,'fresh_blind_test':False,'online_network_wait_included':False}
    if pilot:
        total_bytes=int(pd.read_csv(base/'source_manifest.csv').bytes.sum());projected=wall*total_bytes/max(int(sources.bytes.sum()),1)*len(METHODS)
        result.update(projected_all_methods_seconds=projected,launch_gate=projected<=4*3600 and peak_rss()<12*1024**3)
    dump(dest/'completion.json',result);print(json.dumps(result,ensure_ascii=False,default=str),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','all'],required=True);args=p.parse_args();initialize()
    if args.mode=='pilot':execute('temporal',True);return
    assert read(OUT/'raw_pcap_run/pilot/completion.json')['launch_gate']
    for method in METHODS:
        if (OUT/f'raw_pcap_run/{method}/completion.json').exists():continue
        # Separate processes keep cold-loading/peak RSS method-specific.
        cmd=[sys.executable,'-X','utf8','-s',str(Path(__file__).with_name('raw_method.py')),method]
        log=OUT/f'logs/raw_{method}.log'
        with log.open('x',encoding='utf-8') as handle:
            result=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT)
        assert result.returncode==0,f'{method} failed; see {log}'
        print(f'COMPLETED {method}',flush=True)
    records=[read(OUT/f'raw_pcap_run/{m}/completion.json') for m in METHODS]
    assert len({r['eligible_id_set_sha256'] for r in records})==1
    table(OUT/'raw_pcap_run/method_results.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))}|r['metrics'] for r in records])
    dump(OUT/'raw_pcap_run/completion.json',{'status':'RAW_PCAP_ALL_ELIGIBLE_EXECUTED','methods':METHODS,'classified_per_method':records[0]['classified'],'all_populations_identical':True,'finished':stamp()})
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
