// Included inside Native. All scratch is allocated at construction. Host route
// boundaries remain explicit; no expert/table owner is duplicated here.
void bgate(size_t offset,const void *source,size_t bytes,int rows) {
  if(!config.hotpath_gate) return;
  constexpr size_t pitch=hotpath::GateLayout::row_bytes;
  if(offset+bytes>pitch) throw std::logic_error("gate capture extent");
  CK(cudaMemcpy2DAsync(gate_slab.as<char>(offset),pitch,source,bytes,
                      bytes,rows,cudaMemcpyDeviceToDevice,stream));
}
void btrace(int64_t pos,int m,int layer,const char *tag,const Device &data,int stride) {
  if(config.trace_dir.empty()) return;
  for(int r=0;r<m;++r) {
    auto dir=std::filesystem::path(config.trace_dir)/("pos"+std::to_string(pos+r))/("layer"+std::to_string(layer));
    std::filesystem::create_directories(dir);
    std::vector<uint16_t> raw(stride);
    CK(cudaMemcpyAsync(raw.data(),data.as<char>(size_t(r)*stride*2),stride*2,cudaMemcpyDeviceToHost,stream));
    CK(cudaStreamSynchronize(stream));
    std::ofstream f(dir/(std::string(tag)+".bf16"),std::ios::binary);
    f.write(reinterpret_cast<const char*>(raw.data()),stride*2);
    if(!f) throw std::runtime_error("batched trace write");
  }
}
void blinear(const std::string &prefix, const void *in, void *out,
             int n, int k, int m, bool quant_ready=false) {
  auto &weight = w(prefix + ".weight");
  if(weight.shape != std::vector<uint64_t>({uint64_t(n),uint64_t(k)}))
    throw std::runtime_error("linear shape: "+prefix);
  if (weight.dtype == "F8_E4M3" || weight.dtype == "F8_E4M3FN") {
    if(!quant_ready) CK(ds41_dense_quant(in, aq.p, as.p, m, k, stream));
    CK(ds41_dense_fp8(aq.p, as.p, weight.data.p, w(prefix + ".scale").data.p,
                      out, m, n, k, 2, stream));
  } else {
    // Keep the accepted M1 cuBLAS geometry. Batch-dependent reductions must
    // pass class N gates before a different geometry can replace these calls.
    const size_t element = weight.dtype == "F32" ? 4 : 2;
    for (int r = 0; r < m; ++r)
      CK(ds41_dense_cublas(blas, static_cast<const char *>(in) + r*k*element,
          weight.data.p, static_cast<char *>(out) + r*n*element, 1, n, k,
          weight.dtype == "F32", 1, stream));
  }
}
void bnorm(const std::string &name, const void *in, void *out, int k, int m) {
  hotpath::rms_decode(in,w(name+".weight").data.p,out,m,k,stream);
}
void bhc(const std::string &p, const char *kind, Device &target_pre, int m) {
  for (int r = 0; r < m; ++r)
    CK(ds41_dense_hc_mixes(blas, h.as<char>(r*HC*H*2),
        w(p + ".hc_" + kind + "_fn").data.as<float>(),
        w(p + ".hc_" + kind + "_scale").data.as<float>(),
        w(p + ".hc_" + kind + "_base").data.as<float>(),
        target_pre.as<float>(r*4*4), post.as<float>(r*4*4),
        comb.as<float>(r*16*4), f32.as<float>(r*HC*H*4),
        mixes.as<float>(r*24*4), rstd.as<float>(r*4),
        1, H, 1e-20f, 1e-6f, stream));
}
void bfused_prepare(const std::string &p,const char *kind,Device &new_pre,
                    Device &incoming,const std::string &norm,int m,bool quant) {
  CK(fused->prepare(blas,h.p,w(p+".hc_"+kind+"_fn").data.as<float>(),
      w(p+".hc_"+kind+"_scale").data.as<float>(),w(p+".hc_"+kind+"_base").data.as<float>(),
      incoming.as<float>(),w(norm+".weight").data.p,new_pre.as<float>(),post.as<float>(),
      comb.as<float>(),f32.as<float>(),mixes.as<float>(),rstd.as<float>(),x.p,
      quant?nullptr:f32.as<float>(),quant?aq.p:nullptr,quant?as.p:nullptr,
      m,H,1e-20f,1e-6f,stream));
}
void brope(Device &data, int heads, int dim, int l, int64_t pos, int m,
           bool inverse = false) {
  CK(ds41_dense_rope(data.p, batch_frequencies.as<float>((l>=2?6:0)*64*4),
      m, heads, dim, 64, inverse, stream));
}
void battention(int l, int64_t pos, int m) {
  std::string p = "layers." + std::to_string(l) + ".attn";
  brope(q,64,512,l,pos,m);
  if(fused) CK(fused->rms_rope(kv.p,w(p+".kv_norm.weight").data.p,kv.p,
      batch_frequencies.as<float>((l>=2?6:0)*64*4),m,512,64,1e-20f,0,stream));
  else brope(kv,1,512,l,pos,m);
  CK(ds41_attn_pack(kv.p,packed.p,m,DS41_SWA_FP8,stream));
  CK(ds41_attn_ring_write_at(packed.p,windows[l].p,528,position_device.as<int64_t>(),m,RING,stream));
  int owner=l<2?-1:l<8?2:l<14?8:l<20?14:20;
  int ratio=l<20?2:1;
  std::array<int32_t,6> lengths{}, block_lengths{};
  for(int r=0;r<m;++r) {
    lengths[r]=owner<0?0:int((pos+r+1)/ratio);
    block_lengths[r]=(lengths[r]+7)/8;
  }
  int n=lengths[m-1], k=std::min(512,n);
  if(globals.count(l)) {
    auto &g=*globals.at(l);
    if(ratio==1) blinear(p+".compressor.wkv",x.p,latent.p,512,H,m);
    else {
      bf16_float(x.p,f32.as<float>(),m*H,stream);
      blinear(p+".compressor.wkv",f32.p,braw_kv.p,512,H,m);
      blinear(p+".compressor.wgate",f32.p,braw_score.p,512,H,m);
      CK(ds41_attn_ring_write_at(braw_kv.p,g.raw_kv.p,512*4,position_device.as<int64_t>(),m,8,stream));
      CK(ds41_attn_ring_write_at(braw_score.p,g.raw_score.p,512*4,position_device.as<int64_t>(),m,8,stream));
      CK(ds41_attn_pool2(g.raw_kv.as<float>(),g.raw_score.as<float>(),
                         latent.p,pos<8?pos:pos%8,m,8,stream));
    }
    bnorm(p+".compressor.norm",latent.p,latent.p,512,m);
    blinear(p+".indexer.wk",latent.p,ik.p,128,512,m);
    if(!fused) bnorm(p+".indexer.k_norm",ik.p,ik.p,128,m);
    const float *cf=ratio==1?batch_frequencies.as<float>(6*64*4):compressed_frequencies.as<float>();
    if(fused) CK(fused->rms_rope(ik.p,w(p+".indexer.k_norm.weight").data.p,ik.p,cf,m,128,64,1e-20f,0,stream));
    else CK(ds41_dense_rope(ik.p,cf,m,1,128,64,0,stream));
    CK(ds41_attn_pack(ik.p,packed_keys.p,m,DS41_INDEX_FP4,stream));
    hotpath::store_groups(packed_keys.p,g.key.p,position_device.as<int64_t>(),m,68,ratio,stream);
    CK(ds41_dense_rope(latent.p,cf,m,1,512,64,0,stream));
    CK(ds41_attn_pack(latent.p,packed_globals.p,m,DS41_GLOBAL_FP4,stream));
    hotpath::store_groups(packed_globals.p,g.kv.p,position_device.as<int64_t>(),m,288,ratio,stream);
  }

  bool index=l==2 || l==8 || l==14 || l==20 || l==24 || l==28 || l==32 || l==36;
  if(index && n) {
    auto &g=*globals.at(owner);
    blinear(p+".indexer.wq_b",qa.p,iq.p,4096,1280,m);
    brope(iq,32,128,l,pos,m);
    CK(ds41_attn_roundtrip(iq.p,m*32,128,DS41_INDEX_FP4,stream));
    blinear(p+".indexer.weights_proj",x.p,iw.p,32,H,m);
    scale_bf16(iw.p,m*32,1.f/64,stream);
    zc(visible.p,ratio==1?lengths1.p:lengths2.p,m*4);
    int scan=l>20?candidate_count:(n>=512?std::min(globals.at(owner)->capacity,(n+255)/256*256):n);
    if(l<=20 && lengths[0]<=1 && n>1) {
      // N=1 has a distinct reference head-sum geometry. Keep that geometry
      // for an early row even when later rows in this verify have N>1.
      for(int r=0;r<m;++r) if(lengths[r])
        CK(ds41_attn_index_scores(iq.as<char>(r*4096*2),g.key.p,iw.as<char>(r*32*2),
            nullptr,visible.as<int32_t>(r*4),scores.as<char>(size_t(r)*scan*2),
            1,lengths[r],stream));
    } else CK(ds41_attn_index_scores(iq.p,g.key.p,iw.p,
        l>20?candidate_ids.as<int32_t>():nullptr,visible.as<int32_t>(),
        scores.p,m,scan,stream));
    if(l==20) {
      int nb=(scan+7)/8, kb=std::min(2048,nb);
      CK(ds41_attn_block_scores(scores.p,visible.as<int32_t>(),blocks.p,m,scan,stream));
      hotpath::lengths_aux(visible.as<int>(),block_visible.as<int>(),m,8,-1,stream);
      CK(ds41_attn_topk(blocks.p,block_visible.as<int32_t>(),
          chosen_blocks.as<int32_t>(),m,nb,kb,top_scratch.p,stream));
      CK(ds41_attn_expand_blocks(chosen_blocks.as<int32_t>(),
          candidate_ids.as<int32_t>(),m,kb,scan,stream));
      candidate_count=kb*8;
    }
    if(l>20) {
      hotpath::lengths_aux(nullptr,block_visible.as<int>(),m,1,scan,stream);
      CK(ds41_attn_topk(scores.p,block_visible.as<int32_t>(),
          candidate_top.as<int32_t>(),m,scan,k,top_scratch.p,stream));
      CK(ds41_attn_remap_topk(candidate_top.as<int32_t>(),candidate_ids.as<int32_t>(),
          top.as<int32_t>(),m,k,scan,stream));
    } else {
      CK(ds41_attn_topk(scores.p,visible.as<int32_t>(),top.as<int32_t>(),
          m,scan,k,top_scratch.p,stream));
    }
  }
  Global *g=owner<0?nullptr:globals.at(owner).get();
  CK(ds41_attn_gather_at(windows[l].p,RING,g?g->kv.p:nullptr,g?g->capacity:0,
      k?top.as<int32_t>():nullptr,k,position_device.as<int64_t>(),m,ratio,
      gather.p,local_idx.as<int32_t>(),stream));
  // Until a trailing-padding proof covers every sparse width, early rows use
  // their own exact extent. Once all rows have k=512 the M1..6 kernel runs once.
  bool uniform=pos>0 && std::min(512,int((pos+1)/ratio))==k;
  if(owner<0) uniform=pos>0;
  if(uniform) {
    if(k) CK(ds41_attn_sparse_split(q.p,gather.p,w(p+".attn_sink").data.as<float>(),
        local_idx.as<int32_t>(),o.p,sparse_scores.as<float>(),m,128+k,16,stream));
    else CK(ds41_attn_sparse(q.p,gather.p,w(p+".attn_sink").data.as<float>(),
        local_idx.as<int32_t>(),o.p,m,128,16,stream));
  } else for(int r=0;r<m;++r) {
    int kr=owner<0?0:std::min(512,int((pos+r+1)/ratio));
    void *gr=gather.as<char>(size_t(r)*(128+k)*512*2);
    auto *ix=local_idx.as<int32_t>(size_t(r)*(128+k)*4);
    int width=128+kr;
    if(pos+r==0) {
      // Single-token official prefill at position zero, decode layout later.
      CK(ds41_attn_gather_prefill_at(windows[l].p,RING,g?g->kv.p:nullptr,
          g?g->capacity:0,kr?top.as<int32_t>():nullptr,kr,
          position_device.as<int64_t>(),1,ratio,1,gr,ix,stream));
      width=1+kr;
    }
    CK(ds41_attn_sparse(q.as<char>(r*64*512*2),gr,w(p+".attn_sink").data.as<float>(),
        ix,o.as<char>(r*64*512*2),1,width,16,stream));
  }
  brope(o,64,512,l,pos,m,true);
  CK(ds41_attn_roundtrip(o.p,m,32768,DS41_SWA_FP8,stream));
  bgraph(p+".output",m,[&]{
    if(config.compact_woa)
      CK(ds41_dense_wo_a_compact(o.p,w(p+".wo_a.weight").data.p,w(p+".wo_a.scale").data.p,wo.p,m,1024,4096,8,stream));
    else {
      const void *woa=woa_weight(p+".wo_a");
      for(int r=0;r<m;++r) CK(ds41_dense_cublas(blas,o.as<char>(r*32768*2),
          woa,wo.as<char>(r*8192*2),1,1024,4096,0,8,stream));
    }
    blinear(p+".wo_b",wo.p,y.p,H,8192,m);
  });
}
// Route scores land in a per-lane slice of the pinned step buffer: lane A at 0, lane B below the
// 4 MiB upload staging (lane A's logits copy may run while lane B still routes).
float *lane_scores() const {
  return reinterpret_cast<float *>(static_cast<char *>(host->data())+(lane_b_active?(size_t(3)<<20)+(size_t(1)<<19):0));
}
// Front half of the MoE block: queue the route-score copy (and lookahead) behind the router.
void bmoe_scores(int l,int m) {
  zc(host_alias(lane_scores()),biased.p,m*384*4);
  CK(cudaEventRecord(route_events[lane_b_active],stream));
  if(lookahead_sync && l+1<40 && !lane_b_active && (m==1 || !lookahead_audit.is_open())) {
    // Score layer l+1's router on this layer's FFN input (f32, every verify row) behind the route
    // copy. The host waits only for the route copy; the prediction is read after this layer's demand.
    CK(cudaEventRecord(lookahead_sync->start,stream));
    const int depth=lookahead_audit.is_open()?3:1;
    for(int d=1;d<=depth && l+d<40;++d) {
      const std::string next="layers."+std::to_string(l+d)+".ffn";
      bgraph(next+".lookahead",m,[&]{
        blinear(next+".gate",f32.p,lookahead_router.p,384,H,m);
        CK(ds41_dense_router_scores(lookahead_router.as<float>(),w(next+".gate.bias").data.as<float>(),
            lookahead_unbiased.as<float>(),lookahead_biased.as<float>(),m,384,stream));
      });
      zc(static_cast<float *>(lookahead_host->device_data())+(d-1)*384,lookahead_biased.p,m*384*4);
    }
    CK(cudaEventRecord(lookahead_sync->end,stream));
    lookahead_layer=l+1; lookahead_m=m;
  }
}
void bmoe(int l,int m) {
  std::string p="layers."+std::to_string(l)+".ffn";
  double t=clock_ms();
  auto *score_host=lane_scores();
  CK(cudaEventSynchronize(route_events[lane_b_active]));
  route_ms+=clock_ms()-t;
  check_moe_retirement(false);
  double plan_start=clock_ms();
  std::array<int,36> selected{};
  for(int r=0;r<m;++r) {
    std::array<int,384> order{};
    std::iota(order.begin(),order.end(),0);
    std::partial_sort(order.begin(),order.begin()+6,order.end(),[&](int a,int b){
      return score_host[r*384+a]==score_host[r*384+b]?a<b:
             score_host[r*384+a]>score_host[r*384+b];
    });
    std::copy_n(order.begin(),6,selected.begin()+r*6);
  }
  auto plan=hotpath::plan_routes(selected.data(),m,l);
  if(lookahead_audit.is_open() && m==1) {
    auto snap=experts->residency_snapshot();
    lookahead_audit<<audit_pos<<' '<<l;
    for(int j=0;j<6;++j) {
      int e=selected[j]; uint8_t b=snap[size_t(l)*384+e];
      lookahead_audit<<' '<<e<<':'<<((b&1)?0:(b&2)?1:2);
      for(auto &v:audit_pred[l]) {
        auto it=std::find(v.begin(),v.end(),e);
        lookahead_audit<<':'<<(v.empty()?-2:it==v.end()?-1:int(it-v.begin()));
      }
    }
    for(auto &v:audit_pred[l]) {
      lookahead_audit<<" p";
      for(size_t i=0;i<v.size() && i<16;++i) {
        uint8_t b=snap[size_t(l)*384+v[i]];
        lookahead_audit<<((b&1)?0:(b&2)?1:2);
      }
    }
    if(l==0 && audit_pos%32==0) {
      size_t vram=0,ram=0,both=0;
      for(auto b:snap){vram+=b==1;ram+=b==2;both+=b==3;}
      lookahead_audit<<" tiers "<<vram<<' '<<ram<<' '<<both;
    }
    lookahead_audit<<'\n';
    for(auto &v:audit_pred[l]) v.clear();
  }
  if(config.hotpath_profile) step_demands.push_back({
      {"layer",l},{"logical_ids",plan.ids},{"consumer_backend","gpu"}});

  std::vector<ds41::GemmJob> gj,uj,dj;
  for(const auto &e:plan.experts) {
    // EXL3: gathered BF16 rows in, BF16 activations (act) into the down projection.
    if(exl3) gj.push_back({e.id,e.count,bxh.as<uint8_t>(e.begin*H*2),nullptr,gate.as<Bf16>(e.begin*I*2)});
    else gj.push_back({e.id,e.count,bx.as<uint8_t>(e.begin*H),
        bxs.as<uint8_t>(e.begin*H/32),gate.as<Bf16>(e.begin*I*2)});
    uj.push_back(gj.back()); uj.back().output=up.as<Bf16>(e.begin*I*2);
    if(exl3) dj.push_back({e.id,e.count,act.as<uint8_t>(e.begin*I*2),nullptr,compact_down.as<Bf16>(e.begin*H*2)});
    else dj.push_back({e.id,e.count,eq.as<uint8_t>(e.begin*I),
        es.as<uint8_t>(e.begin*I/32),compact_down.as<Bf16>(e.begin*H*2)});
  }
  zc_begin();
  upload(ids_device,selected.data(),m*6*4);
  upload(route_source,plan.source.data(),plan.count*4);
  upload(route_slots,plan.destination.data(),plan.count*4);
  upload(jobs,gj.data(),gj.size()*sizeof(ds41::GemmJob));
  upload(up_jobs,uj.data(),uj.size()*sizeof(ds41::GemmJob));
  upload(down_jobs,dj.data(),dj.size()*sizeof(ds41::GemmJob));
  zc_end();
  for(int r=0;r<m;++r) route_weights(unbiased.as<float>(r*384*4),
      ids_device.as<int>(r*6*4),route.as<float>(r*6*4),6,stream);
  if(exl3) hotpath::gather_routes_bf16(x.p,route.as<float>(),route_source.as<int>(),
      route_slots.as<int>(),bxh.p,broute.as<float>(),plan.count,stream);
  else {
    CK(ds41::quantize_mxfp8(x.as<Bf16>(),aq.as<uint8_t>(),as.as<uint8_t>(),m,H,stream));
    hotpath::gather_routes(aq.as<uint8_t>(),as.as<uint8_t>(),route.as<float>(),
        route_source.as<int>(),route_slots.as<int>(),bx.as<uint8_t>(),
        bxs.as<uint8_t>(),broute.as<float>(),plan.count,stream);
  }
  bgate(size_t(l)*hotpath::GateLayout::LayerBytes+hotpath::GateLayout::RouteIds,
        ids_device.p,6*4,m);
  bgate(size_t(l)*hotpath::GateLayout::LayerBytes+hotpath::GateLayout::RouteWeights,
        route.p,6*4,m);
  host_plan_ms+=clock_ms()-plan_start;
  if(config.streamed_experts) {
    bmoe_streamed(p,m,plan,gj,uj,dj);
    return;
  }
  t=clock_ms();
  auto lease=experts->need_now(plan.ids,stream);
  io_ms+=clock_ms()-t;
  try {
    bgraph(p+".experts."+std::to_string(plan.ids.size()),m,[&]{
      CK(expert_gemm(jobs.as<ds41::GemmJob>(),int(gj.size()),ds41::Projection::Gate,I,H,stream));
      CK(expert_gemm(up_jobs.as<ds41::GemmJob>(),int(uj.size()),ds41::Projection::Up,I,H,stream));
      CK(ds41::swiglu(gate.as<Bf16>(),up.as<Bf16>(),broute.as<float>(),
          act.as<Bf16>(),plan.count,I,10,stream));
      if(!exl3) CK(ds41::quantize_mxfp8(act.as<Bf16>(),eq.as<uint8_t>(),es.as<uint8_t>(),
          plan.count,I,stream));
      CK(expert_gemm(down_jobs.as<ds41::GemmJob>(),int(dj.size()),ds41::Projection::Down,H,I,stream));
      hotpath::scatter_routes(compact_down.p,route_slots.as<int>(),down.p,plan.count,stream);
      bshared(p,m);
      CK(ds41::reduce_experts(down.as<Bf16>(),ids_device.as<int32_t>(),
          shared.as<Bf16>(),y.as<Bf16>(),m,6,H,stream));
    });
    retire_experts(lease);
  } catch(...) {
    CK(cudaStreamSynchronize(stream));
    if(lease.owner) experts->retire(std::move(lease),stream);
    throw;
  }
}
void bshared(const std::string &p,int m) {
  blinear(p+".shared_experts.w1",x.p,sg.p,I,H,m);
  blinear(p+".shared_experts.w3",x.p,su.p,I,H,m);
  CK(ds41::swiglu(sg.as<Bf16>(),su.as<Bf16>(),nullptr,sa.as<Bf16>(),m,I,10,stream));
  blinear(p+".shared_experts.w2",sa.p,shared.p,H,I,m);
}
template<class F> void bgraph(const std::string &key,int m,F f) {
  if(hotpath_capturing) { f(); return; }
  // Lane B owns different scratch addresses, so its graphs never share keys with lane A's.
  graph(std::string(lane_b_active?"laneB.":"")+"batch."+std::to_string(m)+"."+key,[&]{
    hotpath_capturing=true;
    try { f(); } catch(...) { hotpath_capturing=false; throw; }
    hotpath_capturing=false;
  });
}
// Previous verify completed on this stream before the next call. Old scan
// shapes can be destroyed without adding a synchronization. Width and modulo8
// variants remain reusable; rejected-prefix rollback simply recaptures a shape.
void trim_attention_graphs(int layer,int rows,int scan) {
  if(!config.graphs || !config.trace_dir.empty()) return;
  int &previous=attention_scan_bucket[rows-1][layer];
  if(previous==scan) return;
  const auto prefix=std::string(lane_b_active?"laneB.":"")+"batch."+std::to_string(rows)+".layers."+
      std::to_string(layer)+".attention."+std::to_string(previous)+".";
  for(int mod=0;mod<8;++mod)
    graph_evictions+=graphs.erase(prefix+std::to_string(mod));
  previous=scan;
}
// One verify lane: m consecutive rows starting at pos. The plain path runs a single lane;
// --verify-lanes runs row 0 (lane A) ahead of rows 1.. (lane B) on a second stream.
struct BatchLane { int64_t pos=0; int m=0; std::array<std::vector<int64_t>,2> rowids; };
struct StepMarks { double start=0,io=0,route=0,engram=0,plan=0,enqueue=0; ds41::Stats stats{}; };
StepMarks step_marks(int64_t pos) {
  lookahead_layer=-1;
  audit_pos=pos;
  for(auto &layer:audit_pred) for(auto &v:layer) v.clear();
  StepMarks k;
  k.start=clock_ms(); k.io=io_ms; k.route=route_ms; k.engram=engram_ms;
  k.plan=host_plan_ms; k.enqueue=callback_enqueue_ms;
  if(config.hotpath_profile) {
    k.stats=sample_expert_stats();
    step_demands=Json::array();
  }
  expert_event_index=0;
  if(consumer_readiness) consumer_readiness->reset();
  phase_index=0; phase_names.clear(); upload_cursor=4<<20;
  in_step=true; graph_ready=pos>0;
  return k;
}
template<class F> void lane_phase(const std::string &name,F call) {
  if(lane_b_active) call(); else phase(name,call);
}
// Per-lane prologue: positions, lengths, RoPE rows, Engram submits and the embedding.
void batch_begin(const int *tokens,int m,int64_t pos,BatchLane &lane) {
  lane.pos=pos; lane.m=m; lane.rowids={};
  upload(position_device,&pos,8);
  upload(batch_tokens,tokens,m*4);
  CK(ds41_attn_lengths(position_device.as<int64_t>(),lengths1.as<int32_t>(),
      lengths2.as<int32_t>(),m,config.max_context,stream));
  hotpath::stage_frequencies(rope.as<float>(),batch_frequencies.as<float>(),
      compressed_frequencies.as<float>(),position_device.as<int64_t>(),m,config.max_context,stream);
  for(int r=0;r<m;++r) {
    auto ids=hash.rows(tokens[r],pos+r);
    for(int e=0;e<2;++e) lane.rowids[e].insert(lane.rowids[e].end(),ids[e].begin(),ids[e].end());
  }
  if(async_engram) {
    engram_tickets[0]=async_engram->submit(1,lane.rowids[0]);
    engram_tickets[1]=async_engram->submit(14,lane.rowids[1]);
  }
  hotpath::embedding(w("embed.weight").data.p,batch_tokens.as<int>(),h.p,
      pre.as<float>(),m,stream);
}
// Front of layer l: Engram, captures, attention, router and the queued route-score copy.
void batch_front(int l,BatchLane &lane) {
  const int64_t pos=lane.pos; const int m=lane.m;
  std::string p="layers."+std::to_string(l);
  if(l==1 || l==14) engram(l,lane.rowids[l==1?0:1],m);
  btrace(pos,m,l,"input",h,HC*H);
  // Lane B's rows start at row 1 of the step's capture slab.
  if(l>=37) hotpath::capture_rows(h.p,capture_slab.as<char>(size_t(lane_b_active)*3*H*2),m,l-37,stream);
  lane_phase("attention",[&]{
    int ratio=l<20?2:1;
    int n=l<2?0:int((pos+m)/ratio);
    int scan=n>=512?std::min(config.max_context/ratio,(n+255)/256*256):n;
    auto key=p+".attention."+std::to_string(scan)+"."+std::to_string(pos%8);
    if(l==20) candidate_count=std::min(2048,(scan+7)/8)*8;
    auto attention_body=[&]{
    bgraph(p+".query",m,[&]{
      size_t base=size_t(l)*hotpath::GateLayout::LayerBytes;
      bgate(base+hotpath::GateLayout::Input,h.p,hotpath::GateLayout::hc_bytes,m);
      bgate(base+hotpath::GateLayout::IncomingPre,pre.p,16,m);
      if(fused) bfused_prepare(p,"attn",att_pre,pre,p+".attn_norm",m,true);
      else {
        bhc(p,"attn",att_pre,m);
        CK(ds41_dense_hc_pre(h.p,pre.as<float>(),x.p,m,H,stream));
        bnorm(p+".attn_norm",x.p,x.p,H,m);
      }
      auto a=p+".attn";
      blinear(a+".wq_a",x.p,qa.p,1280,H,m,bool(fused));
      if(fused) CK(fused->norm(qa.p,nullptr,w(a+".q_norm.weight").data.p,qa.p,nullptr,aq.p,as.p,m,1280,1e-20f,stream));
      else bnorm(a+".q_norm",qa.p,qa.p,1280,m);
      blinear(a+".wq_b",qa.p,q.p,32768,1280,m,bool(fused));
      blinear(a+".wkv",x.p,kv.p,512,H,m);
      if(!fused) bnorm(a+".kv_norm",kv.p,kv.p,512,m);
      bgate(base+hotpath::GateLayout::AttnPre,att_pre.p,16,m);
      bgate(base+hotpath::GateLayout::AttnPost,post.p,16,m);
      bgate(base+hotpath::GateLayout::AttnComb,comb.p,64,m);
    });
    btrace(pos,m,l,"attn_input",x,H);
    battention(l,pos,m);
    btrace(pos,m,l,"attn_output",y,H);
    CK(ds41_dense_hc_post(y.p,h.p,post.as<float>(),comb.as<float>(),next_h.p,m,H,stream));
    };
    if(pos>=1024) {
      trim_attention_graphs(l,m,scan);
      bgraph(key,m,attention_body);
    }
    else attention_body();
    std::swap(h,next_h);
  });
  bgraph(p+".route",m,[&]{
    size_t base=size_t(l)*hotpath::GateLayout::LayerBytes;
    bgate(base+hotpath::GateLayout::AfterAttn,h.p,hotpath::GateLayout::hc_bytes,m);
    if(fused) bfused_prepare(p,"ffn",ffn_pre,att_pre,p+".ffn_norm",m,false);
    else {
      bhc(p,"ffn",ffn_pre,m);
      CK(ds41_dense_hc_pre(h.p,att_pre.as<float>(),x.p,m,H,stream));
      bnorm(p+".ffn_norm",x.p,x.p,H,m);
      bf16_float(x.p,f32.as<float>(),m*H,stream);
    }
    blinear(p+".ffn.gate",f32.p,router.p,384,H,m);
    CK(ds41_dense_router_scores(router.as<float>(),w(p+".ffn.gate.bias").data.as<float>(),
        unbiased.as<float>(),biased.as<float>(),m,384,stream));
    bgate(base+hotpath::GateLayout::FfnPre,ffn_pre.p,16,m);
    bgate(base+hotpath::GateLayout::FfnPost,post.p,16,m);
    bgate(base+hotpath::GateLayout::FfnComb,comb.p,64,m);
  });
  btrace(pos,m,l,"ffn_input",x,H);
  bmoe_scores(l,m);
}
// Back of layer l: route sync, expert demand and reduction, FFN HC post.
void batch_back(int l,BatchLane &lane) {
  const int m=lane.m;
  lane_phase("moe",[&]{
    bmoe(l,m);
    btrace(lane.pos,m,l,"ffn_output",y,H);
    CK(ds41_dense_hc_post(y.p,h.p,post.as<float>(),comb.as<float>(),next_h.p,m,H,stream));
    std::swap(h,next_h); std::swap(pre,ffn_pre);
  });
}
// Final norm and head for one lane; its logits land at host row `row`.
void batch_head(BatchLane &lane,int row) {
  const int m=lane.m;
  btrace(lane.pos,m,40,"output",h,HC*H);
  lane_phase("head",[&]{ bgraph("head",m,[&]{
    bgate(hotpath::GateLayout::final_h,h.p,hotpath::GateLayout::hc_bytes,m);
    bgate(hotpath::GateLayout::final_pre,pre.p,16,m);
    if(fused) CK(fused->norm(h.p,pre.as<float>(),w("norm.weight").data.p,x.p,f32.as<float>(),nullptr,nullptr,m,H,1e-20f,stream));
    else {
      CK(ds41_dense_hc_pre(h.p,pre.as<float>(),x.p,m,H,stream));
      bnorm("norm",x.p,x.p,H,m);
      bf16_float(x.p,f32.as<float>(),m*H,stream);
    }
    if(fused_head) fused_head_logits(f32.p,logits.p,m);
    else blinear("head",f32.p,logits.p,V,H,m);
  }); });
  zc(host_alias(static_cast<float *>(host->data())+size_t(row)*V),logits.p,size_t(m)*V*4);
}
std::vector<std::vector<float>> batch_finish(int64_t pos,int m,const StepMarks &k) {
  const double start=k.start; const double io_before=k.io, route_before=k.route,
      engram_before=k.engram, plan_before=k.plan, enqueue_before=k.enqueue;
  const ds41::Stats &stats_before=k.stats;
  auto *output=static_cast<float *>(host->data());
  if(config.hotpath_gate) {
    size_t stride=hotpath::GateLayout::row_bytes;
    CK(cudaMemcpy(gate_host.data(),gate_slab.p,m*stride,cudaMemcpyDeviceToHost));
    gate_rows.resize(m);
    for(int r=0;r<m;++r)
      gate_rows[r].assign(gate_host.begin()+r*stride,gate_host.begin()+(r+1)*stride);
  }
  std::map<std::string,double> step_phases;
  for(size_t i=0;i<phase_index;++i) {
    float ms; CK(cudaEventElapsedTime(&ms,phases[i]->start,phases[i]->end));
    measured[phase_names[i]]+=ms;
    if(config.hotpath_profile) step_phases[phase_names[i]]+=ms;
  }
  double elapsed=clock_ms()-start;
  total_ms+=elapsed; in_step=false;
  if(config.hotpath_profile) {
    double compute=0;
    for(size_t i=0;i<expert_event_index;++i) {
      float ms; CK(cudaEventElapsedTime(&ms,expert_events[i]->start,expert_events[i]->end)); compute+=ms;
    }
    auto after=sample_expert_stats();
    Json step={{"begin_position",pos},{"rows",m},{"wall_ms",elapsed},
        {"expert_stream_segments_ms",compute},{"need_host_ms",io_ms-io_before},
        {"route_wait_ms",route_ms-route_before},{"engram_read_ms",engram_ms-engram_before},
        {"host_route_plan_ms",host_plan_ms-plan_before},{"callback_enqueue_ms",callback_enqueue_ms-enqueue_before},
        {"vram_hits",after.hits-stats_before.hits},{"host_hits",after.host_hits-stats_before.host_hits},
        {"nvme_misses",after.nvme_misses-stats_before.nvme_misses},
        {"h2d_bytes",after.h2d_bytes-stats_before.h2d_bytes},
        {"nvme_bytes",after.nvme_bytes-stats_before.nvme_bytes}};
    step["moe_delta"]=hotpath::moe_delta(stats_before,after,true);
    step["stream_phase_ms_including_stalls"]=step_phases;
    step["demand_calls"]=std::move(step_demands);
    if(consumer_readiness && config.streamed_experts)
      step["consumer_readiness"]=consumer_readiness->collect();
    step["cpu_only_experts"]=0;
    step["moe_demand_grouping"]="sorted unique logical-ID union per layer/verify";
    if(profile_steps.size()==4096) profile_steps.erase(profile_steps.begin());
    profile_steps.push_back(std::move(step));
  }
  std::vector<std::vector<float>> result;
  for(int r=0;r<m;++r) result.emplace_back(output+r*V,output+(r+1)*V);
  return result;
}
std::vector<std::vector<float>> batch(const std::vector<int> &tokens,int64_t pos) {
  int m=int(tokens.size());
  auto marks=step_marks(pos);
  BatchLane lane;
  batch_begin(tokens.data(),m,pos,lane);
  for(int l=0;l<40;++l) { batch_front(l,lane); batch_back(l,lane); }
  batch_head(lane,0);
  CK(cudaStreamSynchronize(stream));
  return batch_finish(pos,m,marks);
}
// Lane B runs on swapped-in spare scratch, its own stream and cuBLAS handle. Captured graphs and
// enqueued work hold raw pointers, so swapping the owning members between enqueues is safe.
void swap_lane() {
  for(auto &slot:lane_slots) std::swap(*slot.first,slot.second);
  std::swap(stream,lane_stream);
  std::swap(blas,lane_blas);
  std::swap(candidate_count,lane_candidate_count);
  std::swap(attention_scan_bucket,lane_scan_bucket);
  std::swap(engram_tickets,lane_engram_tickets);
  lane_b_active=!lane_b_active;
}
void ensure_lanes() {
  if(lane_stream) return;
  if(fused || config.hotpath_gate || !config.trace_dir.empty() || lookahead_sync)
    throw std::invalid_argument("--verify-lanes excludes --fuse-library, gate capture, tracing and lookahead");
  CK(cudaStreamCreateWithFlags(&lane_stream,cudaStreamNonBlocking));
  if(cublasCreate(&lane_blas)!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate (lane B)");
  for(auto &e:lane_att_events) CK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&lane_entry,cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&lane_exit,cudaEventDisableTiming));
  size_t total=0;
  std::vector<Device *> members{&h,&next_h,&x,&y,&f32,&pre,&att_pre,&ffn_pre,&post,&comb,
      &mixes,&rstd,&qa,&q,&kv,&latent,&ik,&iq,&iw,&packed,&gather,&local_idx,
      &o,&wo,&attn_scores,&aq,&as,&gate,&up,&act,&eq,&es,&down,&shared,&sg,&su,&sa,
      &jobs,&up_jobs,&down_jobs,&ids_device,&route,&router,&unbiased,&biased,
      &logits,&scores,&top,&visible,&position_device,&blocks,&block_visible,&chosen_blocks,
      &candidate_ids,&candidate_top,&top_scratch,&rows_raw,&scales_raw,&rows_bf,&engram_kv,
      &rows_interleaved,&braw_kv,&braw_score,&batch_tokens,&route_source,&route_slots,
      &bx,&bxs,&broute,&bxh,&compact_down,&sparse_scores,&work_jobs,&work_up_jobs,&work_down_jobs,
      &work_rows,&batch_frequencies,&compressed_frequencies,&lengths1,&lengths2,
      &packed_keys,&packed_globals};
  if(woa_scratch.bytes) members.push_back(&woa_scratch);
  for(Device *d:members) {
    if(d->mapped_host) throw std::logic_error("lane scratch must be device memory");
    Device spare; spare.resize(d->bytes); total+=d->bytes;
    lane_slots.emplace_back(d,std::move(spare));
  }
  std::cerr<<"verify lanes: lane B scratch "<<total/1048576.0<<" MiB\n";
}
// Pipelined verify: lane A (row 0) never waits for lane B; lane B's attention at layer l waits
// for lane A's front of layer l (KV ring, compressor and global-store writes). While the host
// blocks on one lane's expert demand, the GPU runs the other lane's queued work.
std::vector<std::vector<float>> batch_lanes(const std::vector<int> &tokens,int64_t pos) {
  const int m=int(tokens.size());
  ensure_lanes();
  auto marks=step_marks(pos);
  BatchLane a,b;
  CK(cudaEventRecord(lane_entry,stream));
  batch_begin(tokens.data(),1,pos,a);
  swap_lane();
  CK(cudaStreamWaitEvent(stream,lane_entry,0));
  batch_begin(tokens.data()+1,m-1,pos+1,b);
  swap_lane();
  auto front_b=[&](int l){
    swap_lane();
    CK(cudaStreamWaitEvent(stream,lane_att_events[l],0));
    batch_front(l,b);
    swap_lane();
  };
  try {
    batch_front(0,a); CK(cudaEventRecord(lane_att_events[0],stream));
    front_b(0);
    for(int l=0;l<40;++l) {
      batch_back(l,a);
      if(l+1<40) { batch_front(l+1,a); CK(cudaEventRecord(lane_att_events[l+1],stream)); }
      swap_lane(); batch_back(l,b); swap_lane();
      if(l+1<40) front_b(l+1);
    }
    batch_head(a,0);
    swap_lane(); batch_head(b,1); CK(cudaEventRecord(lane_exit,stream)); swap_lane();
  } catch(...) {
    if(lane_b_active) swap_lane();
    throw;
  }
  // Later work on the main stream (commit, rollback, drafter) orders after lane B.
  CK(cudaStreamWaitEvent(stream,lane_exit,0));
  CK(cudaStreamSynchronize(stream));
  return batch_finish(pos,m,marks);
}
