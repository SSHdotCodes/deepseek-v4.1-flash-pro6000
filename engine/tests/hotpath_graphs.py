"""Device-position replay and index scan-padding exactness. Synthetic data only."""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ops=C.CDLL(str((a.root/'lib/libengine_ops.so').resolve()))
attn=C.CDLL(str((a.root/'lib/libds41_attn.so').resolve()))
P=C.c_void_p;I=C.c_int
ops.ds41_hotpath_frequencies.argtypes=[P,P,P,P,I,I,P]
ops.ds41_hotpath_store_groups.argtypes=[P,P,P,I,I,I,P]
attn.ds41_attn_pack.argtypes=[P,P,I,I,P]
attn.ds41_attn_roundtrip.argtypes=[P,I,I,I,P]
attn.ds41_attn_index_scores.argtypes=[P,P,P,P,P,P,I,I,P]
def call(fn,*args):
  args=[P(x.data_ptr()) if isinstance(x,torch.Tensor) else x for x in args]
  rc=fn(*args,P(torch.cuda.current_stream().cuda_stream))
  if rc: raise RuntimeError((fn.__name__,rc))
def exact(x,y):
  torch.cuda.synchronize()
  if not torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)):
    raise AssertionError((tuple(x.shape),int((x!=y).sum())))
torch.manual_seed(41)
checks=0;context=8192
positions=(0,1,2,7,8,9,127,128,130,1023,1024,2047,4096,8186)
table=torch.randn((2,context,64),device='cuda')
position=torch.zeros((1,),device='cuda',dtype=torch.int64)
for m in range(1,7):
  main=torch.full((12,64),-17.,device='cuda')
  comp=torch.full((6,64),-19.,device='cuda')
  call(ops.ds41_hotpath_frequencies,table,main,comp,position,m,context)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):call(ops.ds41_hotpath_frequencies,table,main,comp,position,m,context)
  for pos in positions:
    position.fill_(pos);g.replay()
    exact(main[:m],table[0,pos:pos+m]);checks+=1
    exact(main[6:6+m],table[1,pos:pos+m]);checks+=1
    idx=torch.arange(pos,pos+m,device='cuda').sub(1).clamp_min(0)
    exact(comp[:m],table[1,idx]);checks+=1
for byte_count in (68,288,528):
 for ratio in (1,2):
  for m in range(1,7):
    raw=torch.randint(0,256,(m,byte_count),device='cuda',dtype=torch.uint8)
    dest=torch.full((context//ratio,byte_count),17,device='cuda',dtype=torch.uint8)
    call(ops.ds41_hotpath_store_groups,raw,dest,position,m,byte_count,ratio)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):call(ops.ds41_hotpath_store_groups,raw,dest,position,m,byte_count,ratio)
    for pos in positions:
      dest.fill_(17);expected=torch.full_like(dest,17)
      for r in range(m):
        if (pos+r+1)%ratio==0:expected[(pos+r+1)//ratio-1]=raw[r]
      position.fill_(pos);g.replay();exact(dest,expected);checks+=1
# For N>1 the retained indexer head-sum is independent of the scan extent.
# N=1 uses a different reduction and is kept per-row by the engine's early path.
for n in (2,3,511,512,513,1023,1024,2047):
 scan=(n+255)//256*256
 keys=torch.randn((scan,128),device='cuda',dtype=torch.bfloat16)
 packed=torch.empty((scan,68),device='cuda',dtype=torch.uint8)
 call(attn.ds41_attn_pack,keys,packed,scan,2)
 for m in range(1,7):
  q=torch.randn((m,32,128),device='cuda',dtype=torch.bfloat16)
  call(attn.ds41_attn_roundtrip,q,m*32,128,2)
  weights=torch.randn((m,32),device='cuda',dtype=torch.bfloat16)
  visible=torch.tensor([max(2,n-r) for r in range(m)],device='cuda',dtype=torch.int32)
  actual=torch.empty((m,scan),device='cuda',dtype=torch.bfloat16)
  call(attn.ds41_attn_index_scores,q,packed,weights,None,visible,actual,m,scan)
  for r in range(m):
    length=int(visible[r]);expected=torch.empty((1,length),device='cuda',dtype=torch.bfloat16)
    call(attn.ds41_attn_index_scores,q[r],packed,weights[r],None,visible[r:r+1],expected,1,length)
    exact(actual[r,:length],expected[0]);checks+=1
    assert bool(torch.isneginf(actual[r,length:]).all());checks+=1
result=dict(schema='ds41-hotpath-graph-metadata-v1',checks=checks,mismatches=0)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
