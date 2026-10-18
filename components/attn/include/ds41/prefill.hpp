#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace ds41 {
struct Span {int64_t begin,end;int64_t size()const{return end-begin;}};
struct LayerWork {
 int layer;
 Span input,query,window_kv,global_kv,capture;
 // Input stream includes the previous sublayer's pre_mix. Both must be retained.
};
inline Span tail(int64_t n,int64_t width){return {std::max<int64_t>(0,n-width),n};}
// An exact cold-prefill dependency cone. It preserves all 128-slot target and
// DSpark rings, last logits, and all encoder-derived global KV. Omitted decoder
// outputs have no causal path to any retained state. No bounded replay.
inline std::vector<LayerWork> ced_cone(int64_t n,int64_t logits_tail=1,bool dspark=true){
 if(n<1||logits_tail<1||logits_tail>n)throw std::invalid_argument("prefill length/output tail");
 std::vector<LayerWork> plan;
 for(int l=0;l<40;++l){
  int64_t qwidth=l<20?n:(l==39?logits_tail:std::max<int64_t>(128,logits_tail+127)+(38-l)*127);
  Span q=tail(n,qwidth);
  Span kv=l<20?Span{0,n}:tail(n,std::max<int64_t>(128,q.size()+127));
  Span in=l<=20?Span{0,n}:tail(n,kv.size());
  Span global=(l==2||l==8||l==14||l==20)?Span{0,n}:Span{n,n};
  Span capture=dspark&&l>=37?tail(n,128):Span{n,n};
  plan.push_back({l,in,q,kv,global,capture});
 }
 return plan;
}
// New suffixes are full-layer causal updates; never replay/truncate an old prefix.
inline std::vector<LayerWork> append_plan(int64_t committed,int64_t n,bool dspark=true){
 if(committed<0||n<1)throw std::invalid_argument("append length");std::vector<LayerWork> p;
 for(int l=0;l<40;++l){Span s{committed,committed+n};bool owner=l==2||l==8||l==14||l==20;p.push_back({l,s,s,s,owner?s:Span{s.end,s.end},dspark&&l>=37?s:Span{s.end,s.end}});}return p;
}
struct ScratchBudget {
 size_t bytes;
 int64_t context;
 int global_topk=512;
 // Per-query scratch: score BF16[N], topk scratch about N/16 bytes, gathered
 // BF16[640,512], Q and output BF16[64,512], selected candidate ids, and margin.
 size_t bytes_per_query()const{return size_t(context)*2+((context+255)/256)*16+516*4+size_t(128+global_topk)*512*2+2*64*512*2+16384*4+4096;}
 int chunk_rows()const{size_t q=bytes_per_query();if(bytes<q)throw std::invalid_argument("scratch budget below one query");return int(std::min<size_t>(bytes/q,4096));}
};
// Engine callback runs layer-major. Keep projections/global store in their own
// passes, then bounded query chunks. Callback owns native GEMM and expert sweep;
// this scheduler never reroutes experts or recomputes a truncated window.
template<class Engine> void run_prefill(Engine& engine,const std::vector<LayerWork>& plan,int chunk){
 if(chunk<1)throw std::invalid_argument("query chunk");
 for(const auto& w:plan){
  engine.begin_layer(w);
  engine.capture(w.capture);
  engine.project_window(w.window_kv);
  engine.project_global(w.global_kv);
  for(int64_t p=w.query.begin;p<w.query.end;p+=chunk)engine.attention(Span{p,std::min(p+chunk,w.query.end)});
  // Retain attention residual + pre_mix until the expert-major MoE sweep ends.
  engine.moe_layer_major(w.query);
  engine.end_layer(w);
 }
}
}
