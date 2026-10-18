"""Independent official hash + native raw row fixture gates and O_DIRECT timing."""
import argparse,ctypes as C,json,sys,time,hashlib,statistics
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];REF=ROOT.parent/'ref'
sys.path[:0]=[str(REF),str(REF/'official/inference')]
from native_store import NativeStore
from engram import EngramLayout,NgramHashState
from types import SimpleNamespace
from transformers import AutoTokenizer
class Config(C.Structure):
 _fields_=[('struct_bytes',C.c_uint32),('abi_version',C.c_uint32)]+[(n,C.c_char_p) for n in ('weight_root','index_json','hash_json','integrity_json')]+[('cache_rows',C.c_uint64)]+[(n,C.c_uint32) for n in ('queue_depth','max_batch_rows','pin_memory','reserved')]
class Stats(C.Structure):
 _fields_=[(n,C.c_uint64) for n in ('submitted_rows','cache_hits','unique_misses','disk_pages','disk_bytes','batches','cache_capacity_rows','cache_live_rows')]+[('io_ms',C.c_double),('total_ms',C.c_double)]
class API:
 def __init__(self,lib,pin=0,qd=48):
  self.lib=C.CDLL(str(lib));self.lib.ds41_engram_error.restype=C.c_char_p;self.store=C.c_void_p();w=Path.home()/'models/deepseek-v41-flash-1017/1bae0e1088993b9b82de08f59de5be5a34e42170';self.weights=w
  self.config=Config(C.sizeof(Config),1,str(w).encode(),str(REF/'metadata/model.safetensors.index.json').encode(),str(ROOT.parent/'engine/generated/engram.json').encode(),str(REF/'metadata/full-integrity.json').encode(),65536,qd,4096,pin,0)
  self.call('open',C.byref(self.config),C.byref(self.store))
 def call(self,name,*args):
  r=getattr(self.lib,'ds41_engram_'+name)(*args)
  if r:raise RuntimeError(f'{name}: {self.lib.ds41_engram_error().decode()}')
 def state(self):
  s=C.c_void_p();self.call('state_create',self.store,C.byref(s));return s
 def hash(self,s,tokens,pos=0,mask=None):
  tokens=np.asarray(tokens,dtype=np.int32);out=np.empty((len(tokens),2,24),np.int64);m=None if mask is None else np.asarray(mask,dtype=np.uint8);self.call('hash',s,C.c_void_p(tokens.ctypes.data),None if m is None else C.c_void_p(m.ctypes.data),C.c_uint64(pos),C.c_size_t(len(tokens)),C.c_void_p(out.ctypes.data));return out
 def submit(self,layer,rows):
  rows=np.ascontiguousarray(rows,dtype=np.int64).reshape(-1);t=C.c_void_p();self.call('submit',self.store,C.c_int(layer),C.c_void_p(rows.ctypes.data),C.c_size_t(len(rows)),C.byref(t));return t
 def data(self,t):
  self.call('wait',t);ptr=C.c_void_p();n=C.c_size_t();self.call('data',t,C.byref(ptr),C.byref(n));return np.ctypeslib.as_array((C.c_uint8*(n.value*264)).from_address(ptr.value)).reshape(n.value,264)
 def free(self,t):self.lib.ds41_engram_ticket_free(t)
 def stats(self):
  z=Stats();self.call('get_stats',self.store,C.byref(z));return {n:getattr(z,n) for n,_ in z._fields_}
 def close(self):self.lib.ds41_engram_close(self.store)
def main():
 p=argparse.ArgumentParser();p.add_argument('--lib',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--long',action='store_true');p.add_argument('--hash-only',action='store_true');p.add_argument('--pin',type=int,default=1);p.add_argument('--qd',type=int,default=48);a=p.parse_args();a.output.parent.mkdir(exist_ok=True,parents=True);api=API(a.lib,a.pin,a.qd)
 result=dict(schema='ds41-engram-qualification-v1',library_sha256=hashlib.sha256(Path(a.lib).read_bytes()).hexdigest(),hash_metadata_sha256=hashlib.sha256(Path(api.config.hash_json.decode()).read_bytes()).hexdigest(),config=dict(pin=a.pin,qd=a.qd,cache_rows=65536,batch_rows=4096),gates=[],timings=[])
 def emit(kind,r):result[kind].append(r);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(r),flush=True)
 config=json.loads((api.weights/'inference/config.json').read_text());config.update(max_batch_size=1,max_seq_len=160000);args=SimpleNamespace(**config);layout=EngramLayout.from_args(args);tokenizer=AutoTokenizer.from_pretrained(api.weights,local_files_only=True);official=NgramHashState(args,layout,tokenizer);rng=np.random.default_rng(417);ids=rng.integers(0,129280,10017,dtype=np.int32);mask=rng.random(len(ids))>.03;mask[:4]=[False,True,True,True]
 want=official(torch.tensor(ids,dtype=torch.int64).reshape(1,-1),0,torch.tensor(mask).reshape(1,-1)).numpy()[0];state=api.state();got=api.hash(state,ids,mask=mask);assert np.array_equal(got,want);emit('gates',dict(gate='official_hash_monolithic_dead_tokens',positions=len(ids),values=got.size,bitwise=True))
 api.call('state_truncate',state,C.c_uint64(0));parts=[];pos=0
 for count in [1,2,3,37,509,4096,1,2,3,5363]:
  count=min(count,len(ids)-pos)
  if count:parts.append(api.hash(state,ids[pos:pos+count],pos,mask[pos:pos+count]));pos+=count
 if pos<len(ids):parts.append(api.hash(state,ids[pos:],pos,mask[pos:]))
 assert np.array_equal(np.concatenate(parts),want);emit('gates',dict(gate='official_hash_chunk_history',positions=len(ids),bitwise=True))
 clone=C.c_void_p();api.call('state_clone',state,C.byref(clone));alternative=rng.integers(0,129280,17,dtype=np.int32);tail=api.hash(clone,alternative,9999);expected=official(torch.tensor(alternative,dtype=torch.int64).reshape(1,-1),9999).numpy()[0];assert np.array_equal(tail,expected);assert np.array_equal(api.hash(state,ids[-18:],9999,mask[-18:]),want[-18:]);emit('gates',dict(gate='draft_clone_rollback_replacement',positions=17,bitwise=True));api.lib.ds41_engram_state_free(clone)
 if a.hash_only:
  api.lib.ds41_engram_state_free(state);api.close();print('HASH_DONE',flush=True);return
 required=[f'layers.{l}.engram.embed.{n}' for l in (1,14) for n in ('weight','scale')];native=NativeStore(api.weights,REF/'metadata/model.safetensors.index.json',REF/'metadata/full-integrity.json',required_names=required)
 for li,layer in enumerate((1,14)):
  rows=np.concatenate((want[:8,li].reshape(-1),want[:2,li].reshape(-1),[0,int(want[...,li,:].max())]));unique=sorted(set(rows.tolist()));values=native.rows(f'layers.{layer}.engram.embed.weight',unique).view(torch.uint8).numpy();scales=native.rows(f'layers.{layer}.engram.embed.scale',unique).view(torch.uint8).numpy();raw=np.concatenate((values,scales),-1);index={r:i for i,r in enumerate(unique)};expected=raw[[index[r] for r in rows]]
  t=api.submit(layer,rows);actual=api.data(t);assert np.array_equal(actual,expected)
  if a.pin:
   raw=torch.from_numpy(actual.copy()).cuda();out=torch.empty(len(rows),256,dtype=torch.bfloat16,device='cuda')
   def dequant():
    rc=api.lib.ds41_engram_dequant(C.c_void_p(raw.data_ptr()),C.c_void_p(out.data_ptr()),C.c_int(len(rows)),C.c_void_p(torch.cuda.current_stream().cuda_stream));assert rc==0,rc
   dequant();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):dequant()
   graph.replay();want=(raw[:,:256].contiguous().view(torch.float8_e4m3fn).float().view(-1,8,32)*raw[:,256:].contiguous().view(torch.float8_e8m0fnu).float().unsqueeze(-1)).reshape(-1,256).bfloat16()
   assert torch.equal(out.view(torch.uint8),want.view(torch.uint8));emit('gates',dict(gate='native_row_dequant_graph',layer=layer,values=out.numel(),bitwise=True))
  emit('gates',dict(gate='native_row_fixture',layer=layer,rows=len(rows),bytes=actual.size,bitwise=True,fixture_sha256=hashlib.sha256(expected.tobytes()).hexdigest()));np.savez(ROOT/f'fixtures/engram-layer{layer}-rows-v1.npz',rows=rows,expected=expected);api.free(t)
  before=api.stats();t=api.submit(layer,rows);assert np.array_equal(api.data(t),expected);after=api.stats();assert after['disk_pages']==before['disk_pages'];api.free(t);emit('gates',dict(gate='cache_repeat_no_io',layer=layer,bitwise=True))
 native.close()
 # Real corpus token replay, sufficiently long to measure all requested prefill sizes.
 corpus=[]
 for line in (REF/'corpora/betterwright-agentic-v2.jsonl').open():
  corpus.extend(json.loads(line)['tokens'])
  if len(corpus)>151000:break
 corpus=np.asarray(corpus[:151000],np.int32);api.call('state_truncate',state,C.c_uint64(0));t0=time.perf_counter();replay=api.hash(state,corpus);hash_ms=(time.perf_counter()-t0)*1000
 result['prefill_source']='concatenated real agentic session token replay for I/O distribution; not a full model prefill';result['hash_151k_ms']=hash_ms
 for m in range(1,7):
  for cache in ('cold','warm'):
   vals=[];ios=[]
   for trial in range(15):
    if cache=='cold':api.call('clear_cache',api.store)
    rows=replay[1000+trial*m:1000+(trial+1)*m]
    if cache=='warm':
     warm=[api.submit(layer,rows[:,li]) for li,layer in enumerate((1,14))]
     [api.call('wait',t) for t in warm];[api.free(t) for t in warm]
    before=api.stats();t0=time.perf_counter();ts=[api.submit(layer,rows[:,li]) for li,layer in enumerate((1,14))];first=time.perf_counter();api.call('wait',ts[0]);layer1=time.perf_counter();api.call('wait',ts[1]);end=time.perf_counter();after=api.stats();vals.append([(first-t0)*1000,(layer1-t0)*1000,(end-t0)*1000]);ios.append(after['io_ms']-before['io_ms']);[api.free(t) for t in ts]
   med=np.median(vals,axis=0);emit('timings',dict(op='decode_both_layers',M=m,cache=cache,trials=len(vals),submit_ms=float(med[0]),layer1_ready_ms=float(med[1]),both_ready_ms=float(med[2]),io_ms=float(np.median(ios))))
 if a.long:
  for n in (8192,32768,65536,131072,150000):
   api.call('clear_cache',api.store);before=api.stats();t0=time.perf_counter();tickets=[api.submit(layer,replay[:n,li]) for li,layer in enumerate((1,14))];submitted=time.perf_counter();api.call('wait',tickets[0]);first=time.perf_counter();api.call('wait',tickets[1]);end=time.perf_counter();after=api.stats();[api.free(t) for t in tickets];emit('timings',dict(op='prefill_both_layers',tokens=n,requested_rows=n*48,submit_ms=(submitted-t0)*1000,layer1_ready_ms=(first-t0)*1000,both_ready_ms=(end-t0)*1000,counters={k:after[k]-before[k] for k in ('unique_misses','disk_pages','disk_bytes','batches','io_ms','total_ms')}))
 api.lib.ds41_engram_state_free(state);result['stats']=api.stats();api.close();a.output.write_text(json.dumps(result,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
