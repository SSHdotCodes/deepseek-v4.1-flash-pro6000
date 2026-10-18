"""Independent PyTorch comparisons for engine-owned TEMP kernels. Heavy lock required."""
import ctypes
import argparse
import json
from pathlib import Path
import sys
import torch

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("--ops-lib",type=Path);a=p.parse_args()
lib=ctypes.CDLL(str(a.ops_lib or root/"build/libengine_ops.so"))
ptr=ctypes.c_void_p
lib.ds41_engine_route_weights.argtypes=[ptr,ptr,ptr,ctypes.c_int,ptr]
lib.ds41_engine_engram_dequant.argtypes=[ptr,ptr,ptr,ctypes.c_int,ptr]
lib.ds41_engine_capture_mean.argtypes=[ptr,ptr,ctypes.c_int,ptr]
torch.manual_seed(314159)
torch.set_default_device("cuda")
checks=[]
for k in (3,6):
    for seed in range(128):
        scores=torch.nn.functional.softplus(torch.randn(384 if k==6 else 128)).sqrt()
        ids=scores.topk(k).indices.int()
        selected=scores[ids.long()]
        expected=selected/(selected.sum()+1e-20)*1.5
        actual=torch.empty_like(expected)
        assert lib.ds41_engine_route_weights(scores.data_ptr(),ids.data_ptr(),actual.data_ptr(),k,torch.cuda.current_stream().cuda_stream)==0
        torch.cuda.synchronize()
        mismatches=(expected.view(torch.int32)!=actual.view(torch.int32)).sum().item()
        checks.append(dict(op="route",k=k,seed=seed,mismatches=mismatches,maxabs=(expected-actual).abs().max().item()))
values=torch.arange(256,dtype=torch.int16).to(torch.uint8).repeat(24,1).contiguous()
scales=torch.randint(1,254,(24,8),dtype=torch.uint8)
# Keep finite products representable; subnormal scale zero is separately tested.
scales.fill_(127)
scales[0,:]=0
expected=(values.view(torch.float8_e4m3fn).float().reshape(24,8,32)*scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)).reshape(24,256).bfloat16()
actual=torch.empty_like(expected)
assert lib.ds41_engine_engram_dequant(values.data_ptr(),scales.data_ptr(),actual.data_ptr(),24,torch.cuda.current_stream().cuda_stream)==0
torch.cuda.synchronize()
finite=torch.isfinite(expected)
checks.append(dict(op="engram_dequant",mismatches=(expected.view(torch.int16)[finite]!=actual.view(torch.int16)[finite]).sum().item(),nan_match=torch.equal(expected.isnan(),actual.isnan())))
for seed in range(16):
    h=torch.randn(4,5120).bfloat16()
    if seed%2==0:h[2]=-h[0]
    actual=torch.empty(5120,dtype=torch.bfloat16)
    assert lib.ds41_engine_capture_mean(h.data_ptr(),actual.data_ptr(),5120,torch.cuda.current_stream().cuda_stream)==0
    expected=h.mean(dim=0)
    torch.cuda.synchronize()
    checks.append(dict(op="capture_mean",seed=seed,mismatches=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum()),maxabs=float((actual-expected).abs().max())))
for mask in range(16):
    h=torch.zeros(4,5120,dtype=torch.bfloat16)
    for row in range(4):
        if mask&(1<<row):h[row]=-0.
    actual=torch.empty(5120,dtype=torch.bfloat16)
    assert lib.ds41_engine_capture_mean(h.data_ptr(),actual.data_ptr(),5120,torch.cuda.current_stream().cuda_stream)==0
    expected=h.mean(dim=0)
    torch.cuda.synchronize()
    checks.append(dict(op="capture_signed_zero",mask=mask,mismatches=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())))
result=dict(schema="ds41-engine-ops-v1",checks=checks,failed=sum(c["mismatches"]!=0 or not c.get("nan_match",True) for c in checks))
out=root/"results/native-ops.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(checks=len(checks),failed=result["failed"],receipt=str(out))))
sys.exit(bool(result["failed"]))
