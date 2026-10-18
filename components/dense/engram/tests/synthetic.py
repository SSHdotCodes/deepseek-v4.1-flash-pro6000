"""CPU-only C ABI/direct-I/O concurrency, boundary, and failure recovery checks.
Uses tiny physical extents in temporary sparse synthetic safetensors, never native payloads.
"""
import argparse,concurrent.futures,ctypes as C,json,os,sys,tempfile,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify import API,Config,Stats,ROOT
p=argparse.ArgumentParser();p.add_argument('--lib',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();lib=C.CDLL(a.lib);lib.ds41_engram_error.restype=C.c_char_p
meta=json.loads((ROOT.parent/'engine/generated/engram.json').read_text());rng=np.random.default_rng(418);checks=[]
def check(name,**kw):checks.append(dict(gate=name,passed=True,**kw));print(json.dumps(checks[-1]),flush=True)
with tempfile.TemporaryDirectory(prefix='ds41-engram-synthetic-') as tmp:
 root=Path(tmp);index={'weight_map':{}};receipt=dict(root=str(root),complete=True,files=[]);expected={};sizes={}
 for layer in meta['layers']:
  n=layer['offsets'][-1]+layer['primes'][-1];li=layer['layer'];name=f'layer{li}.safetensors';values=n*256;scales=n*8;header={f'layers.{li}.engram.embed.weight':dict(dtype='F8_E4M3',shape=[n,256],data_offsets=[0,values]),f'layers.{li}.engram.embed.scale':dict(dtype='F8_E8M0',shape=[n,8],data_offsets=[values,values+scales])};raw=json.dumps(header,separators=(',',':')).encode();base=8+len(raw);path=root/name
  fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,base+values+scales);os.pwrite(fd,len(raw).to_bytes(8,'little')+raw,0)
  # Dense local rows hit page sharing; end rows exercise EOF short direct reads.
  rows=list(range(64))+[n-1,n-2]
  for row in rows:
   v=rng.integers(0,256,256,np.uint8).tobytes();s=rng.integers(118,128,8,np.uint8).tobytes();os.pwrite(fd,v,base+row*256);os.pwrite(fd,s,base+values+row*8);expected[li,row]=v+s
  os.fsync(fd);os.close(fd);st=path.stat();sizes[li]=st.st_size;receipt['files'].append(dict(path=name,size=st.st_size,mtime_ns=st.st_mtime_ns,ok=True))
  index['weight_map'].update({k:name for k in header})
 (root/'index.json').write_text(json.dumps(index));(root/'receipt.json').write_text(json.dumps(receipt));(root/'hash.json').write_text(json.dumps(meta))
 api=object.__new__(API);api.lib=lib;api.store=C.c_void_p();config=Config(C.sizeof(Config),1,str(root).encode(),str(root/'index.json').encode(),str(root/'hash.json').encode(),str(root/'receipt.json').encode(),16,48,8,0,0);api.call('open',C.byref(config),C.byref(api.store))
 for li in (1,14):
  n=next(l for l in meta['layers'] if l['layer']==li)['offsets'][-1]+next(l for l in meta['layers'] if l['layer']==li)['primes'][-1];rows=np.array([n-1,n-2]+list(range(64))+[n-1]*10,np.int64);t=api.submit(li,rows);got=api.data(t).tobytes();want=b''.join(expected[li,int(r)] for r in rows);assert got==want;api.free(t);check('page_crossing_EOF_order_dedup',layer=li,rows=len(rows))
 def work(i):
  li=(1,14)[i%2];rows=np.arange(i%32,i%32+8,dtype=np.int64);t=api.submit(li,rows);got=api.data(t).tobytes();want=b''.join(expected[li,int(r)] for r in rows);assert got==want;api.free(t)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(work,range(100)))
 check('concurrent_submit_wait_cache_eviction',requests=100,cache_rows=16)
 # Failed read must drain all submitted completions so the next request is sound.
 api.call('clear_cache',api.store);path=root/'layer1.safetensors';fd=os.open(path,os.O_WRONLY);os.ftruncate(fd,4096);os.close(fd);t=api.submit(1,np.arange(64,dtype=np.int64));assert lib.ds41_engram_wait(t)==-1;assert b'O_DIRECT read' in lib.ds41_engram_error();api.free(t)
 # Only row0 remained in the first page, but restore row10 and its distant scale.
 fd=os.open(path,os.O_RDWR);os.ftruncate(fd,sizes[1]);h=json.loads(os.pread(fd,int.from_bytes(os.pread(fd,8,0),'little'),8));base=8+len(json.dumps(h,separators=(',',':')).encode());n=h['layers.1.engram.embed.weight']['shape'][0];row=10;os.pwrite(fd,expected[1,row][:256],base+row*256);os.pwrite(fd,expected[1,row][256:],base+n*256+row*8);os.fsync(fd);os.close(fd);t=api.submit(1,[row]);assert api.data(t).tobytes()==expected[1,row];api.free(t);check('short_read_error_and_ring_recovery')
 state=api.state();ids=np.array([1,2,3],np.int32);before=api.hash(state,ids);bad=np.array([1,129280],np.int32);out=np.empty((2,2,24),np.int64);assert lib.ds41_engram_hash(state,C.c_void_p(bad.ctypes.data),None,C.c_uint64(1),C.c_size_t(2),C.c_void_p(out.ctypes.data))==-1;assert np.array_equal(api.hash(state,ids,0),before);check('invalid_token_atomic_state')
 lib.ds41_engram_state_free(state);api.close()
result=dict(schema='ds41-engram-synthetic-v1',library_sha256=hashlib.sha256(Path(a.lib).read_bytes()).hexdigest(),checks=checks);a.output.write_text(json.dumps(result,indent=2));print('DONE',flush=True)
