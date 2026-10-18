"""Export CPU metadata through the unchanged official hash implementation.

No model allocation, CUDA context, or source/weights mutation. Run with the
baseline venv, while all native payload work stays under the heavy lock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument("--weights",required=True,type=Path)
p.add_argument("--output",required=True,type=Path)
a=p.parse_args()
sys.path.insert(0,str(a.weights/"inference"))
from engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers
from types import SimpleNamespace
config=json.loads((a.weights/"inference/config.json").read_text())
layout=EngramLayout.from_args(SimpleNamespace(**config))
tokenizer=AutoTokenizer.from_pretrained(a.weights,local_files_only=True)
cmap,size=build_compressed_token_map(tokenizer)
assert size==config["engram_compressed_vocab_size"]
multipliers=compute_hash_multipliers(layout.layer_ids,4,size).tolist()
meta=dict(schema="ds41-engram-hash-v1",compressed_vocab_size=size,pad_id=cmap[config["engram_pad_id"]],token_map=cmap,layers=[])
for i,layer in enumerate(layout.layer_ids):
    primes=[p for n in layout.primes[i] for p in n]
    offsets=[0]
    for prime in primes[:-1]:offsets.append(offsets[-1]+prime)
    meta["layers"].append(dict(layer=layer,primes=primes,offsets=offsets,multipliers=multipliers[i]))
meta["provenance"]={str(f.relative_to(a.weights)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.weights/"inference/engram.py",a.weights/"inference/config.json",a.weights/"tokenizer.json")}
a.output.parent.mkdir(parents=True,exist_ok=True)
tmp=a.output.with_suffix(".partial")
tmp.write_text(json.dumps(meta,separators=(",",":"))+"\n")
tmp.replace(a.output)
print(json.dumps(dict(path=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),vocab=size)))
