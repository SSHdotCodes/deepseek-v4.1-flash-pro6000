// Included inside Native after prefill/batch.inc. Class-N batched prefill (--prefill-chunk R): every dense
// projection, RMS, mHC coefficient pass, compressor/indexer step, gather and sparse-attention call covers up
// to R rows of one layer. FP8 weights are dequantized exactly to BF16 and multiplied with FP8-roundtripped
// activations in one cuBLAS GEMM: the same products as the M1 FP8 kernel, summed in a different order.
// Row-local kernels (RMS, mHC) keep their M1 reduction geometry. Engram gates and captures stay per row.
struct PrefillChunk {
  int R;
  Device h, next, x, y, pre, att_pre, ffn_pre, post, comb, promoted, mixes, rstd;
  Device qa, q, kv, o, wo, aq, as, xq, wdq, f32x, router, unbiased, biased;
  Device packed, latent, ik, rawkv, rawscore, pooled, cs, iq, iw;
  Device visible, block_visible, start, top, local, gathered, attn_scores;
  Device scores, blocks, chosen, cand, cand_top, topk_scratch;
  Device sg, su, sa, shared;
  // Sizes for R rows of a prompt ending at `end`, with the layer-20 candidate width and the largest FP8 weight.
  static std::vector<size_t> sizes(int r,int64_t end,int candidate_width,size_t fp8_elements) {
    const size_t R=size_t(r),n=size_t(std::max<int64_t>(end,candidate_width)),nb=(size_t(end)+7)/8;
    const size_t topk=std::max({ds41_attn_topk_scratch(r,int(n)),ds41_attn_topk_scratch(r,int(std::max<size_t>(nb,1)))});
    return {R*HC*H*2,R*HC*H*2,R*H*2,R*H*2,R*16,R*16,R*16,R*16,R*64,R*HC*H*4,R*24*4,R*4,
        R*1280*2,R*32768*2,R*512*2,R*32768*2,R*8192*2,R*8192,R*8192/32,R*8192*2,fp8_elements*2,R*H*4,R*384*4,R*384*4,R*384*4,
        R*528,R*512*2,R*128*2,(R+1)*512*4,(R+1)*512*4,(R/2+1)*512*2,R*64*4,R*4096*2,R*32*2,
        R*4,R*4,8,R*512*4,R*640*4,R*640*512*2,R*64*640*4,
        R*n*2,R*nb*2+64,R*2048*4,R*16384*4,R*512*4,topk,
        R*I*2,R*I*2,R*I*2,R*H*2};
  }
  PrefillChunk(int r,int64_t end,int candidate_width,size_t fp8_elements):R(r) {
    auto s=sizes(r,end,candidate_width,fp8_elements);size_t i=0;
    for(Device* d:{&h,&next,&x,&y,&pre,&att_pre,&ffn_pre,&post,&comb,&promoted,&mixes,&rstd,
        &qa,&q,&kv,&o,&wo,&aq,&as,&xq,&wdq,&f32x,&router,&unbiased,&biased,
        &packed,&latent,&ik,&rawkv,&rawscore,&pooled,&cs,&iq,&iw,
        &visible,&block_visible,&start,&top,&local,&gathered,&attn_scores,
        &scores,&blocks,&chosen,&cand,&cand_top,&topk_scratch,&sg,&su,&sa,&shared})d->resize(s.at(i++));
  }
};
std::unique_ptr<PrefillChunk> prefill_chunk;
size_t prefill_fp8_elements() const {
  size_t best=0;
  for(const auto &item:weights)
    if((item.second.dtype=="F8_E4M3"||item.second.dtype=="F8_E4M3FN")&&item.second.shape.size()==2)
      best=std::max<size_t>(best,item.second.shape[0]*item.second.shape[1]);
  return best;
}
void clinear(const std::string& name,const void* input,void* output,int n,int k,int rows) {
  auto &c=*prefill_chunk;auto &weight=w(name+".weight");
  if(weight.shape!=std::vector<uint64_t>({uint64_t(n),uint64_t(k)}))throw std::runtime_error("prefill chunk linear shape: "+name);
  if(weight.dtype=="F8_E4M3"||weight.dtype=="F8_E4M3FN") {
    if(size_t(rows)*k>c.xq.bytes/2||size_t(n)*k*2>c.wdq.bytes)throw std::runtime_error("prefill chunk FP8 scratch: "+name);
    CK(ds41_dense_quant(input,c.aq.p,c.as.p,rows,k,stream));
    CK(ds41_dense_roundtrip(c.aq.p,c.as.p,c.xq.p,rows,k,stream));
    CK(ds41_dense_dequant(weight.data.p,w(name+".scale").data.p,c.wdq.p,n,k,stream));
    CK(ds41_dense_cublas(blas,c.xq.p,c.wdq.p,output,rows,n,k,0,1,stream));
  }else CK(ds41_dense_cublas(blas,input,weight.data.p,output,rows,n,k,weight.dtype=="F32",1,stream));
}
void cnorm(const std::string& name,const void* input,void* output,int k,int rows) {
  CK(ds41e::prefill::rms_rows(input,w(name+".weight").data.p,output,rows,k,1e-20f,stream));
}
void chc(const std::string& name,const char* kind,const void* hrows,float* target,float* post_out,float* comb_out,int rows) {
  auto &c=*prefill_chunk;
  CK(ds41e::prefill::hc_mixes_rows(blas,hrows,w(name+".hc_"+kind+"_fn").data.as<float>(),
      w(name+".hc_"+kind+"_scale").data.as<float>(),w(name+".hc_"+kind+"_base").data.as<float>(),
      target,post_out,comb_out,c.promoted.as<float>(),c.mixes.as<float>(),c.rstd.as<float>(),rows,H,1e-20f,1e-6f,stream));
}
void crope(void* x,int heads,int dim,const float* cos_sin,int rows,bool inverse=false) {
  CK(ds41_dense_rope(x,cos_sin,rows,heads,dim,64,inverse,stream));
}
// KV/compressor for rows [first,first+rows) and attention for query rows [qb,qb+qn), all of layer work.layer.
void cattention(const ds41e::prefill::Layer& work,int64_t first,int rows,int64_t qb,int qn) {
  auto &pf=*pref;auto &c=*prefill_chunk;int l=work.layer;
  std::string p="layers."+std::to_string(l)+".attn";
  crope(c.kv.p,1,512,freqs(l,first),rows);
  CK(ds41_attn_pack(c.kv.p,c.packed.p,rows,DS41_SWA_FP8,stream));
  CK(ds41_attn_ring_write(c.packed.p,pf.window.p,528,first,rows,int(pf.rows+128),stream));
  const int owner=l<2?-1:l<8?2:l<14?8:l<20?14:20,ratio=l<20?2:1;
  if(globals.count(l)) {
    auto &g=*globals.at(l);
    int entries=0;int64_t entry0=0,b0=first;
    if(ratio==1) {
      clinear(p+".compressor.wkv",c.x.p,c.latent.p,512,H,rows);
      entries=rows;entry0=first;
    } else {
      // Pairs (even, odd). An odd first row pairs with the previous raw row kept in the 8-slot ring.
      const int lead=first%2?1:0;b0=first-lead;
      if(lead) {
        size_t off=size_t((first-1)%8)*512*4;
        pcopy(c.rawkv.p,g.raw_kv.as<char>(off),512*4);pcopy(c.rawscore.p,g.raw_score.as<char>(off),512*4);
      }
      bf16_float(c.x.p,c.f32x.as<float>(),rows*H,stream);
      clinear(p+".compressor.wkv",c.f32x.p,c.rawkv.as<char>(size_t(lead)*512*4),512,H,rows);
      clinear(p+".compressor.wgate",c.f32x.p,c.rawscore.as<char>(size_t(lead)*512*4),512,H,rows);
      const int buffered=rows+lead;entries=buffered/2;entry0=b0/2;
      if(entries)CK(ds41_attn_pool2_prefill(c.rawkv.as<float>(),c.rawscore.as<float>(),c.latent.p,buffered,stream));
      // Keep the last raw row where the per-row path and decode expect it.
      size_t last=size_t((first+rows-1)%8)*512*4,src=size_t(buffered-1)*512*4;
      pcopy(g.raw_kv.as<char>(last),c.rawkv.as<char>(src),512*4);pcopy(g.raw_score.as<char>(last),c.rawscore.as<char>(src),512*4);
    }
    if(entries) {
      // Entry j is RoPE'd at position b0+ratio*j (pos+1-ratio of its completing row).
      const float* cs=freqs(l,b0);
      if(ratio==2) {
        CK(cudaMemcpy2DAsync(c.cs.p,64*4,cs,2*64*4,64*4,entries,cudaMemcpyDeviceToDevice,stream));
        cs=c.cs.as<float>();
      }
      cnorm(p+".compressor.norm",c.latent.p,c.latent.p,512,entries);
      clinear(p+".indexer.wk",c.latent.p,c.ik.p,128,512,entries);
      cnorm(p+".indexer.k_norm",c.ik.p,c.ik.p,128,entries);
      crope(c.ik.p,1,128,cs,entries);
      CK(ds41_attn_pack(c.ik.p,g.key.as<char>(size_t(entry0)*68),entries,DS41_INDEX_FP4,stream));
      crope(c.latent.p,1,512,cs,entries);
      CK(ds41_attn_pack(c.latent.p,g.kv.as<char>(size_t(entry0)*288),entries,DS41_GLOBAL_FP4,stream));
    }
  }
  if(!qn)return;
  const int qoff=int(qb-first);
  const bool index=l==2||l==8||l==14||l==20||l==24||l==28||l==32||l==36;
  const int n=owner<0?0:(pf.cold?int(pf.end/ratio):int((qb+qn)/ratio));
  const int k=std::min(512,n);
  std::vector<int32_t> visible(qn);
  for(int r=0;r<qn;++r)visible[r]=owner<0?0:int((qb+r+1)/ratio);
  const size_t row=size_t(qb-pf.begin);
  crope(c.q.p,64,512,freqs(l,qb),qn);
  if(index&&n) {
    auto &g=*globals.at(owner);
    const int scan=l>20?pf.candidate_width:n;
    const int32_t* key_ids=l>20?pf.candidates.as<int32_t>(size_t(qb-pf.candidate_begin)*pf.candidate_width*4):nullptr;
    clinear(p+".indexer.wq_b",c.qa.p,c.iq.p,4096,1280,qn);crope(c.iq.p,32,128,freqs(l,qb),qn);
    CK(ds41_attn_roundtrip(c.iq.p,qn*32,128,DS41_INDEX_FP4,stream));
    clinear(p+".indexer.weights_proj",c.x.as<char>(size_t(qoff)*H*2),c.iw.p,32,H,qn);scale_bf16(c.iw.p,qn*32,1.f/64,stream);
    upload(c.visible,visible.data(),size_t(qn)*4);
    CK(ds41_attn_index_scores(c.iq.p,g.key.p,c.iw.p,key_ids,c.visible.as<int32_t>(),c.scores.p,qn,scan,stream));
    if(l==20) {
      const int nb=(n+7)/8,kb=std::min(2048,nb);
      std::vector<int32_t> vb(qn);for(int r=0;r<qn;++r)vb[r]=(visible[r]+7)/8;
      CK(ds41_attn_block_scores(c.scores.p,c.visible.as<int32_t>(),c.blocks.p,qn,n,stream));
      upload(c.block_visible,vb.data(),size_t(qn)*4);
      CK(ds41_attn_topk(c.blocks.p,c.block_visible.as<int32_t>(),c.chosen.as<int32_t>(),qn,nb,kb,c.topk_scratch.p,stream));
      CK(ds41_attn_expand_blocks(c.chosen.as<int32_t>(),c.cand.as<int32_t>(),qn,kb,n,stream));
      const int64_t cb=std::max<int64_t>(qb,pf.candidate_begin),ce=qb+qn;
      if(ce>cb) {
        auto* dst=pf.candidates.as<char>(size_t(cb-pf.candidate_begin)*pf.candidate_width*4);
        CK(cudaMemsetAsync(dst,0xff,size_t(ce-cb)*pf.candidate_width*4,stream));
        CK(cudaMemcpy2DAsync(dst,size_t(pf.candidate_width)*4,c.cand.as<char>(size_t(cb-qb)*kb*8*4),size_t(kb)*8*4,
            size_t(kb)*8*4,size_t(ce-cb),cudaMemcpyDeviceToDevice,stream));
      }
    }
    if(l>20) {
      std::vector<int32_t> sv(qn,scan);upload(c.block_visible,sv.data(),size_t(qn)*4);
      CK(ds41_attn_topk(c.scores.p,c.block_visible.as<int32_t>(),c.cand_top.as<int32_t>(),qn,scan,k,c.topk_scratch.p,stream));
      CK(ds41_attn_remap_topk(c.cand_top.as<int32_t>(),key_ids,c.top.as<int32_t>(),qn,k,scan,stream));
    } else CK(ds41_attn_topk(c.scores.p,c.visible.as<int32_t>(),c.top.as<int32_t>(),qn,n,k,c.topk_scratch.p,stream));
    CK(cudaMemcpy2DAsync(pf.selected.as<char>(row*512*4),512*4,c.top.p,size_t(k)*4,size_t(k)*4,qn,cudaMemcpyDeviceToDevice,stream));
  } else if(!index&&k)
    CK(cudaMemcpy2DAsync(c.top.p,size_t(k)*4,pf.selected.as<char>(row*512*4),512*4,size_t(k)*4,qn,cudaMemcpyDeviceToDevice,stream));
  auto* g=owner<0?nullptr:globals.at(owner).get();
  const int width=pf.cold?int(std::min<int64_t>(128,pf.end)):128;
  if(pf.cold) {
    upload(c.start,&qb,8);
    CK(ds41_attn_gather_prefill_at(pf.window.p,int(pf.rows+128),g?g->kv.p:nullptr,g?g->capacity:0,
        k?c.top.as<int32_t>():nullptr,k,c.start.as<int64_t>(),qn,ratio,width,c.gathered.p,c.local.as<int32_t>(),stream));
  } else CK(ds41_attn_gather(pf.window.p,int(pf.rows+128),g?g->kv.p:nullptr,g?g->capacity:0,
      k?c.top.as<int32_t>():nullptr,k,qb,qn,ratio,c.gathered.p,c.local.as<int32_t>(),stream));
  if(!pf.cold&&owner>=0)CK(ds41_attn_sparse_split(c.q.p,c.gathered.p,w(p+".attn_sink").data.as<float>(),
      c.local.as<int32_t>(),c.o.p,c.attn_scores.as<float>(),qn,width+k,16,stream));
  else CK(ds41_attn_sparse(c.q.p,c.gathered.p,w(p+".attn_sink").data.as<float>(),c.local.as<int32_t>(),c.o.p,qn,width+k,16,stream));
  crope(c.o.p,64,512,freqs(l,qb),qn,true);CK(ds41_attn_roundtrip(c.o.p,qn,32768,DS41_SWA_FP8,stream));
}
void cproject(const ds41e::prefill::Layer& work,int64_t first,int rows,const char* raw) {
  auto &pf=*pref;auto &c=*prefill_chunk;int l=work.layer;
  size_t base=size_t(first-pf.begin);auto p="layers."+std::to_string(l),a=p+".attn";
  pcopy(c.h.p,pf.hidden.as<char>(base*HC*H*2),size_t(rows)*HC*H*2);
  pcopy(c.pre.p,pf.previous_pre.as<char>(base*4*4),size_t(rows)*4*4);
  if(l==1||l==14) {
    for(int r=0;r<rows;++r) {
      pcopy(h.p,c.h.as<char>(size_t(r)*HC*H*2),HC*H*2);
      if(raw)pengram(l,raw+size_t(r)*24*264);
      else {auto &all=pf.rowids[l==1?0:1];size_t i=(base+r)*24;engram(l,{all.begin()+i,all.begin()+i+24});}
      pcopy(c.h.as<char>(size_t(r)*HC*H*2),h.p,HC*H*2);
    }
  }
  for(int r=0;r<rows;++r)if(work.capture.contains(first+r))
    capture_mean(c.h.as<char>(size_t(r)*HC*H*2),pf.capture.as<char>(size_t(first+r-work.capture.begin)*3*H*2+(l-37)*H*2),H,stream);
  chc(p,"attn",c.h.p,c.att_pre.as<float>(),c.post.as<float>(),c.comb.as<float>(),rows);
  CK(ds41_dense_hc_pre(c.h.p,c.pre.as<float>(),c.x.p,rows,H,stream));cnorm(p+".attn_norm",c.x.p,c.x.p,H,rows);
  const int64_t qb=std::max(first,work.query.begin),qe=std::min(first+rows,work.query.end);
  const int qn=qe>qb?int(qe-qb):0,qoff=int(qb-first);
  if(qn) {
    clinear(a+".wq_a",c.x.as<char>(size_t(qoff)*H*2),c.qa.p,1280,H,qn);cnorm(a+".q_norm",c.qa.p,c.qa.p,1280,qn);
    clinear(a+".wq_b",c.qa.p,c.q.p,32768,1280,qn);
  }
  clinear(a+".wkv",c.x.p,c.kv.p,512,H,rows);cnorm(a+".kv_norm",c.kv.p,c.kv.p,512,rows);
  cattention(work,first,rows,qb,qn);
  if(!qn)return;
  CK(ds41e::prefill::wo_a_rows(blas,c.o.p,woa_weight(a+".wo_a"),c.wo.p,qn,1024,4096,8,stream));
  clinear(a+".wo_b",c.wo.p,c.y.p,H,8192,qn);
  char* hq=c.h.as<char>(size_t(qoff)*HC*H*2);
  CK(ds41_dense_hc_post(c.y.p,hq,c.post.as<float>(size_t(qoff)*4*4),c.comb.as<float>(size_t(qoff)*16*4),c.next.p,qn,H,stream));
  pcopy(hq,c.next.p,size_t(qn)*HC*H*2);
  chc(p,"ffn",hq,c.ffn_pre.as<float>(),c.post.as<float>(),c.comb.as<float>(),qn);
  CK(ds41_dense_hc_pre(hq,c.att_pre.as<float>(size_t(qoff)*4*4),c.x.p,qn,H,stream));cnorm(p+".ffn_norm",c.x.p,c.x.p,H,qn);
  bf16_float(c.x.p,c.f32x.as<float>(),qn*H,stream);clinear(p+".ffn.gate",c.f32x.p,c.router.p,384,H,qn);
  CK(ds41_dense_router_scores(c.router.as<float>(),w(p+".ffn.gate.bias").data.as<float>(),c.unbiased.as<float>(),c.biased.as<float>(),qn,384,stream));
  const size_t slot=size_t(qb-pf.begin);
  ds41e::prefill::select_routes_rows(c.biased.as<float>(),c.unbiased.as<float>(),pf.ids.as<int>(slot*6*4),pf.weights.as<float>(slot*6*4),qn,stream);
  pcopy(pf.hidden.as<char>(slot*HC*H*2),hq,size_t(qn)*HC*H*2);
  pcopy(pf.ffn_input.as<char>(slot*H*2),c.x.p,size_t(qn)*H*2);
  pcopy(pf.ffn_post.as<char>(slot*4*4),c.post.p,size_t(qn)*4*4);pcopy(pf.ffn_comb.as<char>(slot*16*4),c.comb.p,size_t(qn)*16*4);
  pcopy(pf.previous_pre.as<char>(slot*4*4),c.ffn_pre.p,size_t(qn)*4*4);
}
void cshared(int l,int64_t first,int rows) {
  auto &pf=*pref;auto &c=*prefill_chunk;size_t base=size_t(first-pf.begin);
  auto p="layers."+std::to_string(l)+".ffn.shared_experts";
  clinear(p+".w1",pf.ffn_input.as<char>(base*H*2),c.sg.p,I,H,rows);clinear(p+".w3",pf.ffn_input.as<char>(base*H*2),c.su.p,I,H,rows);
  CK(ds41::swiglu(c.sg.as<Bf16>(),c.su.as<Bf16>(),nullptr,c.sa.as<Bf16>(),rows,I,10,stream));
  clinear(p+".w2",c.sa.p,c.shared.p,H,I,rows);
  CK(ds41::reduce_experts(pf.routed.as<Bf16>(base*6*H*2),pf.ids.as<int32_t>(base*6*4),
      c.shared.as<Bf16>(),c.y.as<Bf16>(),rows,6,H,stream));
  CK(ds41_dense_hc_post(c.y.p,pf.hidden.as<char>(base*HC*H*2),pf.ffn_post.as<float>(base*4*4),
      pf.ffn_comb.as<float>(base*16*4),c.h.p,rows,H,stream));
  pcopy(pf.hidden.as<char>(base*HC*H*2),c.h.p,size_t(rows)*HC*H*2);
}
