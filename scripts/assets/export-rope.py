"""Export the official CUDA-generated complex RoPE cache without loading weights.

Run under the DS41 heavy lock using ref/python.sh. This imports the official
module unchanged and calls its own precompute_freqs_cis function.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser()
p.add_argument("--weights",type=Path,required=True)
p.add_argument("--rows",type=int,default=8192)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
assert 1<=a.rows<=1048576
sys.path.insert(0,str(a.weights/"inference"))
from model import precompute_freqs_cis
c=json.loads((a.weights/"inference/config.json").read_text())
a.output.parent.mkdir(parents=True,exist_ok=True)
tmp=a.output.with_suffix(".partial")
torch.set_default_device("cuda")
with tmp.open("wb") as f:
    for original,base in ((0,c["rope_theta"]),(c["original_seq_len"],c["compress_rope_theta"])):
        cis=precompute_freqs_cis(c["rope_head_dim"],a.rows,original,base,c["rope_factor"],c["beta_fast"],c["beta_slow"])
        f.write(torch.view_as_real(cis).cpu().numpy().tobytes())
tmp.replace(a.output)
meta=dict(schema="ds41-rope-v1",rows=a.rows,source_sha256=hashlib.sha256((a.weights/"inference/model.py").read_bytes()).hexdigest(),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),torch=torch.__version__,dtype="F32",layout="[2,rows,32,2] pure-SWA then compressed")
Path(str(a.output)+".json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta))
