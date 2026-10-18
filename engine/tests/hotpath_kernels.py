"""Independent kernel comparisons against the previously qualified component ABIs.

Run under heavy.sh. This checks M1 reduction geometry and explicit BF16
activation boundaries, not full-model quality or workload throughput.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--ops',type=Path,required=True)
p.add_argument('--components',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ops=C.CDLL(str(a.ops.resolve()))
dense=C.CDLL(str((a.components/'lib/libds41_dense.so').resolve()))
moe=C.CDLL(str((a.components/'lib/libds41_moe.so').resolve()))
ptr=C.c_void_p; integer=C.c_int
ops.ds41_hotpath_rms.argtypes=[ptr,ptr,ptr,integer,integer,ptr]
dense.ds41_dense_rms.argtypes=[ptr,ptr,ptr,integer,integer,C.c_float,ptr]
ops.ds41_hotpath_activate.argtypes=[ptr,ptr,ptr,ptr,ptr,ptr,integer,ptr]
moe.ds41_swiglu_raw.argtypes=[ptr,ptr,ptr,ptr,integer,integer,C.c_float,ptr]
moe.ds41_quantize_mxfp8_raw.argtypes=[ptr,ptr,ptr,integer,integer,ptr]
stream=ptr(torch.cuda.current_stream().cuda_stream)
def call(fn,*args):
    args=[ptr(x.data_ptr()) if isinstance(x,torch.Tensor) else x for x in args]
    rc=fn(*args,stream)
    if rc: raise RuntimeError((fn.__name__,rc))
def exact(x,y):
    torch.cuda.synchronize()
    if not torch.equal(x.view(torch.uint8),y.view(torch.uint8)):
        raise AssertionError((x.shape,int((x.view(torch.uint8)!=y.view(torch.uint8)).sum())))
torch.manual_seed(41)
checks=0
for k in (128,512,1280,5120):
  for m in range(1,7):
    for kind in ('random','zero','wide','cancellation'):
      x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
      if kind=='zero': x.zero_(); x[:,::2]=-0.
      if kind=='wide': x*=torch.logspace(-5,5,k,device='cuda').bfloat16()
      if kind=='cancellation': x[:,::2]=x[:,1::2].neg()
      w=torch.randn((k,),device='cuda',dtype=torch.bfloat16)
      expected=torch.empty_like(x);actual=torch.empty_like(x)
      for r in range(m):
        call(dense.ds41_dense_rms,x[r],w,expected[r],1,k,1e-20)
      call(ops.ds41_hotpath_rms,x,w,actual,m,k);exact(actual,expected);checks+=1
      call(ops.ds41_hotpath_rms,x,w,x,m,k);exact(x,expected);checks+=1
for m in range(1,37):
  g=(torch.randn((36,2304),device='cuda')*15).bfloat16()
  u=(torch.randn_like(g)*15).bfloat16()
  w=torch.rand((36,),device='cuda')*1.5
  rows=torch.randperm(36,device='cuda',dtype=torch.int64)[:m].int().contiguous()
  act=torch.empty_like(g);q=torch.empty((36,2304),device='cuda',dtype=torch.uint8)
  s=torch.empty((36,72),device='cuda',dtype=torch.uint8)
  call(moe.ds41_swiglu_raw,g,u,w,act,36,2304,10.)
  call(moe.ds41_quantize_mxfp8_raw,act,q,s,36,2304)
  actual_q=torch.full_like(q,17);actual_s=torch.full_like(s,19)
  call(ops.ds41_hotpath_activate,g,u,w,rows,actual_q,actual_s,m)
  exact(actual_q[rows.long()],q[rows.long()]);exact(actual_s[rows.long()],s[rows.long()])
  unused=torch.ones((36,),device='cuda',dtype=torch.bool);unused[rows.long()]=False
  assert bool((actual_q[unused]==17).all()) and bool((actual_s[unused]==19).all())
  checks+=4
result=dict(schema='ds41-hotpath-kernels-v1',checks=checks,mismatches=0)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
