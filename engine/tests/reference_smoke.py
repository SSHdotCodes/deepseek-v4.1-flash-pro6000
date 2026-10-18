"""Small same-mode target logits from the exact reference, no reference edits.

Run under the heavy lock after full integrity verification. This is a small
bring-up fixture, never a replacement for QUALITY.md's held-out gold suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument("--weights",type=Path,required=True)
p.add_argument("--receipt",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--ids",default="0,100,200,300")
p.add_argument("--input-jsonl",type=Path)
p.add_argument("--greedy",type=int,default=0)
p.add_argument("--trace-dir",type=Path)
a=p.parse_args()
sys.path.insert(0,str(Path.home()/"ai/ds41/ref"))
from reference import Reference
ids=[int(x) for x in a.ids.split(",")]
if a.input_jsonl:
    ids=json.loads(next(a.input_jsonl.open()))["tokens"][:512]
prompt_rows=len(ids)
assert 0<=a.greedy<=256
ref=Reference(a.weights,receipt=a.receipt,max_seq_len=max(256,len(ids)+a.greedy+8))
out=[]
if a.trace_dir:
    def dump(layer,tag,tensor):
        dest=a.trace_dir/f"pos{position}"/f"layer{layer}"
        dest.mkdir(parents=True,exist_ok=True)
        (dest/f"{tag}.bf16").write_bytes(tensor.detach().bfloat16().contiguous().view(torch.int16).cpu().numpy().tobytes())
    for layer_id,layer in enumerate(ref.model.layers):
        layer.register_forward_pre_hook(lambda mod,args,l=layer_id:dump(l,"input",args[0]))
        layer.register_forward_hook(lambda mod,args,result,l=layer_id:dump(l,"output",result[0]))
        for name in ("attn","ffn"):
            module=getattr(layer,name)
            module.register_forward_pre_hook(lambda mod,args,l=layer_id,n=name:dump(l,n+"_input",args[0]))
            module.register_forward_hook(lambda mod,args,result,l=layer_id,n=name:dump(l,n+"_output",result))
for position in range(prompt_rows+a.greedy):
    if position>=prompt_rows:ids.append(int(np.argmax(out[-1])))
    token=ids[position]
    _,logits,_=ref.forward([token],position,full_logits=True,greedy=True)
    out.append(logits.float().reshape(-1).cpu().numpy().copy())
    print(f"reference-smoke position={position} top1={int(np.argmax(out[-1]))}",flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez(a.output,token_ids=np.array(ids,np.int32),logits=np.stack(out),prompt_rows=np.array(prompt_rows,np.int32),greedy_ids=np.array(ids[prompt_rows:],np.int32))
Path(str(a.output)+".json").write_text(json.dumps(dict(schema="ds41-engine-bringup-v1",execution_mode="teacher-forced-decode-T1",weight_receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),profile=ref.numerics_profile),indent=2)+"\n")
ref.close()
