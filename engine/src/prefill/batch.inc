// Included inside Native before pengram/prefill. Only FP8 projection rows share
// a launch. RMS, mHC coefficient construction, Engram gates and cuBLAS remain
// M1. Row-local HC/reduction kernels have no M-dependent reduction tree.
struct PrefillBatch {
  Device h{8*HC*H*2}, next{8*HC*H*2}, x{8*H*2}, y{8*H*2};
  Device pre{8*4*4}, att_pre{8*4*4}, ffn_pre{8*4*4}, post{8*4*4}, comb{8*16*4};
  Device f32{8*HC*H*4}, mixes{8*24*4}, rstd{8*4};
  Device qa{8*1280*2}, q{8*32768*2}, kv{8*512*2}, o{8*32768*2}, wo{8*8192*2};
  Device aq{8*32768}, as{8*32768/32}, router{8*384*4}, unbiased{8*384*4}, biased{8*384*4};
  Device sg{8*I*2},su{8*I*2},sa{8*I*2},shared{8*H*2};
};
std::unique_ptr<PrefillBatch> prefill_batch;
void plinear(const std::string& name,const void* input,void* output,int n,int k,int rows) {
  auto &b=*prefill_batch;auto &weight=w(name+".weight");
  if(weight.shape!=std::vector<uint64_t>({uint64_t(n),uint64_t(k)}))throw std::runtime_error("prefill batch linear shape");
  if(weight.dtype=="F8_E4M3"||weight.dtype=="F8_E4M3FN") {
    CK(ds41_dense_quant(input,b.aq.p,b.as.p,rows,k,stream));
    CK(ds41_dense_fp8(b.aq.p,b.as.p,weight.data.p,w(name+".scale").data.p,output,rows,n,k,2,stream));
  }else {
    size_t element=weight.dtype=="F32"?4:2;
    for(int row=0;row<rows;++row)CK(ds41_dense_cublas(blas,
        static_cast<const char*>(input)+size_t(row)*k*element,weight.data.p,
        static_cast<char*>(output)+size_t(row)*n*element,1,n,k,weight.dtype=="F32",1,stream));
  }
}
void pnorm(const std::string& name,const void* input,void* output,int k,int rows) {
  for(int row=0;row<rows;++row)CK(ds41_dense_rms(static_cast<const char*>(input)+size_t(row)*k*2,
      w(name+".weight").data.p,static_cast<char*>(output)+size_t(row)*k*2,1,k,1e-20f,stream));
}
void phc(const std::string& name,const char* kind,Device& target,int rows) {
  auto &b=*prefill_batch;
  for(int r=0;r<rows;++r)CK(ds41_dense_hc_mixes(blas,b.h.as<char>(r*HC*H*2),
      w(name+".hc_"+kind+"_fn").data.as<float>(),w(name+".hc_"+kind+"_scale").data.as<float>(),
      w(name+".hc_"+kind+"_base").data.as<float>(),target.as<float>(r*4*4),
      b.post.as<float>(r*4*4),b.comb.as<float>(r*16*4),b.f32.as<float>(r*HC*H*4),
      b.mixes.as<float>(r*24*4),b.rstd.as<float>(r*4),1,H,1e-20f,1e-6f,stream));
}
void pproject_batch(const ds41e::prefill::Layer& work,int64_t first,int rows,const char* raw) {
  auto &pf=*pref;auto &b=*prefill_batch;int l=work.layer;
  size_t base=size_t(first-pf.begin);auto p="layers."+std::to_string(l),a=p+".attn";
  pcopy(b.h.p,pf.hidden.as<char>(base*HC*H*2),size_t(rows)*HC*H*2);
  pcopy(b.pre.p,pf.previous_pre.as<char>(base*4*4),size_t(rows)*4*4);
  if(l==1||l==14) {
    for(int r=0;r<rows;++r) {
      pcopy(h.p,b.h.as<char>(r*HC*H*2),HC*H*2);
      if(raw)pengram(l,raw+size_t(r)*24*264);
      else {auto &all=pf.rowids[l==1?0:1];size_t i=(base+r)*24;engram(l,{all.begin()+i,all.begin()+i+24});}
      pcopy(b.h.as<char>(r*HC*H*2),h.p,HC*H*2);
    }
  }
  for(int r=0;r<rows;++r)if(work.capture.contains(first+r))
    capture_mean(b.h.as<char>(r*HC*H*2),pf.capture.as<char>(size_t(first+r-work.capture.begin)*3*H*2+(l-37)*H*2),H,stream);
  phc(p,"attn",b.att_pre,rows);
  CK(ds41_dense_hc_pre(b.h.p,b.pre.as<float>(),b.x.p,rows,H,stream));pnorm(p+".attn_norm",b.x.p,b.x.p,H,rows);
  bool queries=first+rows>work.query.begin;
  if(queries) {
    plinear(a+".wq_a",b.x.p,b.qa.p,1280,H,rows);pnorm(a+".q_norm",b.qa.p,b.qa.p,1280,rows);
    plinear(a+".wq_b",b.qa.p,b.q.p,32768,1280,rows);
    CK(cudaMemsetAsync(b.o.p,0,size_t(rows)*32768*2,stream));
  }
  plinear(a+".wkv",b.x.p,b.kv.p,512,H,rows);pnorm(a+".kv_norm",b.kv.p,b.kv.p,512,rows);
  for(int r=0;r<rows;++r) {
    pcopy(x.p,b.x.as<char>(r*H*2),H*2);pcopy(kv.p,b.kv.as<char>(r*512*2),512*2);
    if(work.query.contains(first+r)) {
      pcopy(qa.p,b.qa.as<char>(r*1280*2),1280*2);pcopy(q.p,b.q.as<char>(r*32768*2),32768*2);
    }
    pattention(l,first+r,false);
    if(work.query.contains(first+r))pcopy(b.o.as<char>(r*32768*2),o.p,32768*2);
  }
  if(!queries)return;
  const void *woa=woa_weight(a+".wo_a");
  for(int r=0;r<rows;++r)CK(ds41_dense_cublas(blas,b.o.as<char>(r*32768*2),
      woa,b.wo.as<char>(r*8192*2),1,1024,4096,0,8,stream));
  plinear(a+".wo_b",b.wo.p,b.y.p,H,8192,rows);
  CK(ds41_dense_hc_post(b.y.p,b.h.p,b.post.as<float>(),b.comb.as<float>(),b.next.p,rows,H,stream));
  pcopy(b.h.p,b.next.p,size_t(rows)*HC*H*2);
  phc(p,"ffn",b.ffn_pre,rows);
  CK(ds41_dense_hc_pre(b.h.p,b.att_pre.as<float>(),b.x.p,rows,H,stream));pnorm(p+".ffn_norm",b.x.p,b.x.p,H,rows);
  bf16_float(b.x.p,b.f32.as<float>(),rows*H,stream);plinear(p+".ffn.gate",b.f32.p,b.router.p,384,H,rows);
  CK(ds41_dense_router_scores(b.router.as<float>(),w(p+".ffn.gate.bias").data.as<float>(),b.unbiased.as<float>(),b.biased.as<float>(),rows,384,stream));
  for(int r=0;r<rows;++r)if(work.query.contains(first+r)) {
    size_t slot=base+r;
    ds41e::prefill::select_routes(b.biased.as<float>(r*384*4),b.unbiased.as<float>(r*384*4),
        pf.ids.as<int>(slot*6*4),pf.weights.as<float>(slot*6*4),stream);
    pcopy(pf.hidden.as<char>(slot*HC*H*2),b.h.as<char>(r*HC*H*2),HC*H*2);
    pcopy(pf.ffn_input.as<char>(slot*H*2),b.x.as<char>(r*H*2),H*2);
    pcopy(pf.ffn_post.as<char>(slot*4*4),b.post.as<char>(r*4*4),4*4);
    pcopy(pf.ffn_comb.as<char>(slot*16*4),b.comb.as<char>(r*16*4),16*4);
    pcopy(pf.previous_pre.as<char>(slot*4*4),b.ffn_pre.as<char>(r*4*4),4*4);
  }
}
void pshared_batch(int l,int64_t first,int rows) {
  auto &pf=*pref;auto &b=*prefill_batch;size_t base=size_t(first-pf.begin);
  auto p="layers."+std::to_string(l)+".ffn.shared_experts";
  pcopy(b.x.p,pf.ffn_input.as<char>(base*H*2),size_t(rows)*H*2);
  plinear(p+".w1",b.x.p,b.sg.p,I,H,rows);plinear(p+".w3",b.x.p,b.su.p,I,H,rows);
  CK(ds41::swiglu(b.sg.as<Bf16>(),b.su.as<Bf16>(),nullptr,b.sa.as<Bf16>(),rows,I,10,stream));
  plinear(p+".w2",b.sa.p,b.shared.p,H,I,rows);
  CK(ds41::reduce_experts(pf.routed.as<Bf16>(base*6*H*2),pf.ids.as<int32_t>(base*6*4),
      b.shared.as<Bf16>(),b.y.as<Bf16>(),rows,6,H,stream));
  CK(ds41_dense_hc_post(b.y.p,pf.hidden.as<char>(base*HC*H*2),pf.ffn_post.as<float>(base*4*4),
      pf.ffn_comb.as<float>(base*16*4),b.h.p,rows,H,stream));
  pcopy(pf.hidden.as<char>(base*HC*H*2),b.h.p,size_t(rows)*HC*H*2);
}
