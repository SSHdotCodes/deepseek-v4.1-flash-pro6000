// Included inside Native. The decode/verify implementation remains independent.
struct PrefillData {
  size_t rows;
  int64_t begin, end;
  bool cold;
  int candidate_width;
  int64_t candidate_begin;
  int dense_batch=1;
  bool warm=true;
  ds41e::prefill::Span query{0,0};
  Device hidden, previous_pre, ffn_input, ffn_post, ffn_comb;
  Device window, selected, candidates, ids, weights, quant, scales, routed, capture;
  Device bx{256*H}, bs{256*H/32}, bw{256*4};
  Device gate{256*I*2}, up{256*I*2}, act{256*I*2},
      eq{256*I}, es{256*I/32}, down{256*H*2};
  std::array<std::vector<int64_t>,2> rowids;
  std::unique_ptr<ds41::HostBuffer> route_host;
  std::unique_ptr<ds41::HostBuffer> descriptors;
  Json layers=Json::array();
  std::array<std::unique_ptr<EngramAsync::Ticket>,2> engram;
  std::array<const char*,2> engram_raw{};
  double engram_barrier_ms=0;
  static size_t required_bytes(size_t n,int64_t b,size_t tail) {
    const int64_t end=b+int64_t(n);
    const int candidate_width=std::min<int64_t>(2048,(end+7)/8)*8;
    const int64_t candidate_begin=ds41e::prefill::plan(b,n,tail)[20].query.begin;
    return n*(size_t(H)*(4*2+2+6*2+1)+H/32+512*4+6*8+24*4)
        +(n+128)*528+size_t(end-candidate_begin)*candidate_width*4
        +std::min<size_t>(128,n)*3*H*2+(256<<20);
  }
  PrefillData(size_t n,int64_t b,bool c,size_t tail,size_t budget):rows(n),begin(b),end(b+n),cold(c),
      candidate_width(std::min<int64_t>(2048,(end+7)/8)*8),
      candidate_begin(ds41e::prefill::plan(b,n,tail)[20].query.begin) {
    // Largest arrays are explicitly bounded by new rows, never prior context.
    const size_t need=required_bytes(n,b,tail);
    if(need>budget)throw std::invalid_argument("prefill activation budget exceeded: need "+
        std::to_string(need)+" bytes; declare sufficient reserved scratch without evicting decode cache");
    size_t free,total;CK(cudaMemGetInfo(&free,&total));
    if(need>free || free-need<(64<<20))
      throw std::runtime_error("prefill reserved scratch does not fit beside the live decode cache");
    hidden.resize(n*HC*H*2);previous_pre.resize(n*4*4);
    ffn_input.resize(n*H*2);ffn_post.resize(n*4*4);ffn_comb.resize(n*16*4);
    window.resize((n+128)*528);selected.resize(n*512*4);
    candidates.resize(size_t(end-candidate_begin)*candidate_width*4);
    ids.resize(n*6*4);weights.resize(n*6*4);
    quant.resize(n*H);scales.resize(n*H/32);routed.resize(n*6*H*2);
    capture.resize(std::min<size_t>(128,n)*3*H*2);
    route_host=std::make_unique<ds41::HostBuffer>(n*6*4,true);
  }
};

// v0.4 atomically retains only resident IDs, without admission or policy
// credit. The v0.3 fallback is kept for its frozen historical gate bundle.
template<class Store> typename Store::Lease pborrow(Store& store,const std::vector<ds41::ExpertId>& ids) {
  if constexpr(Store::api_version>=4)return store.borrow_vram(ids,stream);
  else {
    std::vector<ds41::ExpertId> hits;
    for(auto id:ids)if(store.residency(id)>=0)hits.push_back(id);
    return hits.empty()?typename Store::Lease{}:store.need_now(hits,stream);
  }
}
template<class Store> std::vector<uint8_t> p_residency(Store& store) {
  if constexpr(Store::api_version>=4)return store.residency_snapshot();
  else return {};
}
template<class F> void pstream(std::vector<ds41::ExpertId> ids,F consume) {
  prefill_transfers={};prefill_vram_hits=0;
  auto lease=pborrow(*experts,ids);
  std::set<ds41::ExpertId> hits;std::vector<ds41::ExpertId> misses;
  try {
    for(const auto &a:lease.allocations) {
      if(!a.hit)throw std::runtime_error("resident-only prefill borrow admitted a miss");
      hits.insert(a.id);
    }
    prefill_vram_hits=hits.size();
    for(auto id:ids)if(!hits.count(id))misses.push_back(id);
    for(auto id:hits)consume(id,experts->table(),stream);
    if(lease.serial)experts->retire(std::move(lease),stream);
  }catch(...){cudaStreamSynchronize(stream);if(lease.serial&&lease.owner)experts->retire(std::move(lease),stream);throw;}
  if(!misses.empty()) {
    if(!prefill_stage)prefill_stage=std::make_unique<ds41e::prefill::Stager>(prefill_records,8);
    auto lookup=config.prefill_host_view;
    auto lease=std::make_shared<ds41::OffloadStore::HostLease>(experts->borrow_host(misses));
    if(!lookup)lookup=[lease](uint32_t id){
      for(const auto &v:lease->views)if(v.id==id&&v.data)
        return ds41e::prefill::HostView{v.data,v.record.bytes,lease};
      return ds41e::prefill::HostView{};
    };
    prefill_transfers=prefill_stage->stream(misses,stream,consume,lookup,[&]{p_engram_barrier();});
  }
}
void p_engram_barrier() {
  if(!async_engram)return;
  double start=clock_ms();
  for(int j=0;j<2;++j)if(!pref->engram_raw[j])
    pref->engram_raw[j]=static_cast<const char*>(pref->engram[j]->wait(pref->rows*24));
  pref->engram_barrier_ms+=clock_ms()-start;
}
void pcopy(void* dst,const void* src,size_t bytes) {
  CK(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));
}
// One row retains the accepted dense M1 geometry. Cold attention alone changes
// to the official full-N layout. Full-M cold cuBLAS equivalence is not claimed.
void pattention(int l,int64_t pos,bool output_projection=true) {
  auto &pf=*pref;
  std::string p="layers."+std::to_string(l)+".attn";
  apply_rope(q,64,512,l,pos);apply_rope(kv,1,512,l,pos);
  CK(ds41_attn_pack(kv.p,packed.p,1,DS41_SWA_FP8,stream));
  CK(ds41_attn_ring_write(packed.p,pf.window.p,528,pos,1,int(pf.rows+128),stream));
  int owner=l<2?-1:l<8?2:l<14?8:l<20?14:20;
  int ratio=l<20?2:1;
  int visible_n=owner<0?0:int((pos+1)/ratio);
  int n=owner<0?0:(pf.cold?int(pf.end/ratio):visible_n);
  int k=std::min(512,n);
  bool index=l==2||l==8||l==14||l==20||l==24||l==28||l==32||l==36;
  if(globals.count(l)) {
    auto &g=*globals.at(l);bool complete=ratio==1||pos%2==1;
    if(ratio==1)linear(p+".compressor.wkv",x.p,latent.p,512,H);
    else {
      bf16_float(x.p,f32.as<float>(),H,stream);auto off=(pos%8)*512*4;
      linear(p+".compressor.wkv",f32.p,g.raw_kv.as<char>(off),512,H);
      linear(p+".compressor.wgate",f32.p,g.raw_score.as<char>(off),512,H);
      if(complete)CK(ds41_attn_pool2(g.raw_kv.as<float>(),g.raw_score.as<float>(),latent.p,pos,1,8,stream));
    }
    if(complete) {
      norm(p+".compressor.norm",latent.p,latent.p,512);
      linear(p+".indexer.wk",latent.p,ik.p,128,512);
      norm(p+".indexer.k_norm",ik.p,ik.p,128);
      apply_rope(ik,1,128,l,pos+1-ratio);
      CK(ds41_attn_pack(ik.p,g.key.as<char>(size_t(visible_n-1)*68),1,DS41_INDEX_FP4,stream));
      apply_rope(latent,1,512,l,pos+1-ratio);
      CK(ds41_attn_pack(latent.p,g.kv.as<char>(size_t(visible_n-1)*288),1,DS41_GLOBAL_FP4,stream));
    }
  }
  if(!pf.query.contains(pos))return;
  size_t row=size_t(pos-pf.begin);
  int scan=l>20?pf.candidate_width:n;
  if(index&&n) {
    auto &g=*globals.at(owner);
    if(l>20)pcopy(candidate_ids.p,pf.candidates.as<char>(size_t(pos-pf.candidate_begin)*pf.candidate_width*4),pf.candidate_width*4);
    linear(p+".indexer.wq_b",qa.p,iq.p,4096,1280);apply_rope(iq,32,128,l,pos);
    CK(ds41_attn_roundtrip(iq.p,32,128,DS41_INDEX_FP4,stream));
    linear(p+".indexer.weights_proj",x.p,iw.p,32,H);scale_bf16(iw.p,32,1.f/64,stream);
    upload(visible,&visible_n,4);
    CK(ds41_attn_index_scores(iq.p,g.key.p,iw.p,l>20?candidate_ids.as<int32_t>():nullptr,
        visible.as<int32_t>(),scores.p,1,scan,stream));
    if(l==20) {
      int nb=(n+7)/8,kb=std::min(2048,nb),vb=(visible_n+7)/8;
      CK(ds41_attn_block_scores(scores.p,visible.as<int32_t>(),blocks.p,1,n,stream));
      upload(block_visible,&vb,4);
      CK(ds41_attn_topk(blocks.p,block_visible.as<int32_t>(),chosen_blocks.as<int32_t>(),1,nb,kb,top_scratch.p,stream));
      CK(ds41_attn_expand_blocks(chosen_blocks.as<int32_t>(),candidate_ids.as<int32_t>(),1,kb,n,stream));
      if(pos>=pf.candidate_begin) {
        auto* dst=pf.candidates.as<char>(size_t(pos-pf.candidate_begin)*pf.candidate_width*4);
        CK(cudaMemsetAsync(dst,0xff,pf.candidate_width*4,stream));
        pcopy(dst,candidate_ids.p,kb*8*4);
      }
    }
    if(l>20) {
      upload(block_visible,&scan,4);
      CK(ds41_attn_topk(scores.p,block_visible.as<int32_t>(),candidate_top.as<int32_t>(),1,scan,k,top_scratch.p,stream));
      CK(ds41_attn_remap_topk(candidate_top.as<int32_t>(),candidate_ids.as<int32_t>(),top.as<int32_t>(),1,k,scan,stream));
    }else CK(ds41_attn_topk(scores.p,visible.as<int32_t>(),top.as<int32_t>(),1,n,k,top_scratch.p,stream));
    pcopy(pf.selected.as<char>(row*512*4),top.p,k*4);
  }else if(index&&k) {
    CK(cudaMemsetAsync(top.p,0xff,k*4,stream));pcopy(pf.selected.as<char>(row*512*4),top.p,k*4);
  }else if(!index&&k)pcopy(top.p,pf.selected.as<char>(row*512*4),k*4);
  auto*g=owner<0?nullptr:globals.at(owner).get();
  int width=pf.cold?int(std::min<int64_t>(128,pf.end)):128;
  if(pf.cold) {
    upload(position_device,&pos,8);
    CK(ds41_attn_gather_prefill_at(pf.window.p,int(pf.rows+128),g?g->kv.p:nullptr,g?g->capacity:0,
        k?top.as<int32_t>():nullptr,k,position_device.as<int64_t>(),1,ratio,width,gather.p,local_idx.as<int32_t>(),stream));
  }else CK(ds41_attn_gather(pf.window.p,int(pf.rows+128),g?g->kv.p:nullptr,g?g->capacity:0,
      k?top.as<int32_t>():nullptr,k,pos,1,ratio,gather.p,local_idx.as<int32_t>(),stream));
  // Both frozen sparse implementations use the same arithmetic; keep decode
  // split dispatch for append and cold single-pass per its component contract.
  if(!pf.cold&&owner>=0)CK(ds41_attn_sparse_split(q.p,gather.p,w(p+".attn_sink").data.as<float>(),
      local_idx.as<int32_t>(),o.p,attn_scores.as<float>(),1,width+k,16,stream));
  else CK(ds41_attn_sparse(q.p,gather.p,w(p+".attn_sink").data.as<float>(),local_idx.as<int32_t>(),o.p,1,width+k,16,stream));
  apply_rope(o,64,512,l,pos,true);CK(ds41_attn_roundtrip(o.p,1,32768,DS41_SWA_FP8,stream));
  if(!output_projection)return;
  CK(ds41_dense_cublas(blas,o.p,woa_weight(p+".wo_a"),wo.p,1,1024,4096,0,8,stream));
  linear(p+".wo_b",wo.p,y.p,H,8192);
}
Json pmoe(int l,ds41e::prefill::Span span) {
  auto &pf=*pref;size_t base=size_t(span.begin-pf.begin),count=size_t(span.size());
  auto* ids=static_cast<int32_t*>(pf.route_host->data());
  double host_start=clock_ms();
  CK(cudaMemcpyAsync(ids,pf.ids.as<char>(base*6*4),count*6*4,cudaMemcpyDeviceToHost,stream));
  CK(cudaStreamSynchronize(stream));
  double boundary=clock_ms()-host_start;
  if(config.prefill_warm_rows && pf.warm && pf.rows>=config.prefill_warm_rows && base+count+config.prefill_warm_rows>pf.rows)
    for(size_t r=count-std::min(count,base+count+config.prefill_warm_rows-pf.rows);r<count;++r)
      for(int j=0;j<6;++j)++prefill_warm_counts[ds41::expert_id(l,ids[r*6+j])];
  auto groups=ds41e::prefill::group(ids,count);
  size_t chunks=0;
  Json distribution={{"1",0},{"2",0},{"3-8",0},{"9-32",0},{"33-256",0},{"257+",0}};
  for(auto &g:groups){auto n=g.slots.size();chunks+=(n+255)/256;
    const char* bucket=n==1?"1":n==2?"2":n<=8?"3-8":n<=32?"9-32":n<=256?"33-256":"257+";
    distribution[bucket]=distribution[bucket].get<size_t>()+1;
  }
  size_t descriptor_bytes=(chunks*3*sizeof(ds41::GemmJob)+63)&~size_t(63);
  pf.descriptors=std::make_unique<ds41::HostBuffer>(descriptor_bytes+count*6*4,true);
  auto* cpu_jobs=static_cast<ds41::GemmJob*>(pf.descriptors->data());
  auto* gpu_jobs=static_cast<ds41::GemmJob*>(pf.descriptors->device_data());
  auto* cpu_slots=reinterpret_cast<int32_t*>(static_cast<char*>(pf.descriptors->data())+descriptor_bytes);
  auto* gpu_slots=reinterpret_cast<int32_t*>(static_cast<char*>(pf.descriptors->device_data())+descriptor_bytes);
  struct Batch {size_t job,offset;int rows;};
  std::map<ds41::ExpertId,std::vector<Batch>> batches;
  std::vector<ds41::ExpertId> needed;
  size_t job=0,offset=0;
  for(auto &g:groups) {
    auto id=ds41::expert_id(l,g.expert);needed.push_back(id);
    for(size_t first=0;first<g.slots.size();first+=256) {
      int m=int(std::min<size_t>(256,g.slots.size()-first));
      std::copy_n(g.slots.data()+first,m,cpu_slots+offset);
      cpu_jobs[job+0]={id,m,pf.bx.as<uint8_t>(),pf.bs.as<uint8_t>(),pf.gate.as<Bf16>()};
      cpu_jobs[job+1]={id,m,pf.bx.as<uint8_t>(),pf.bs.as<uint8_t>(),pf.up.as<Bf16>()};
      cpu_jobs[job+2]={id,m,pf.eq.as<uint8_t>(),pf.es.as<uint8_t>(),pf.down.as<Bf16>()};
      batches[id].push_back({job,offset,m});job+=3;offset+=m;
    }
  }
  CK(ds41::quantize_mxfp8(pf.ffn_input.as<Bf16>(base*H*2),pf.quant.as<uint8_t>(),
      pf.scales.as<uint8_t>(),int(count),H,stream));
  auto resident_before=experts->stats();

  std::vector<std::unique_ptr<Phase>> compute;
  for(size_t i=0;i<needed.size();++i)compute.push_back(std::make_unique<Phase>());
  Phase sweep;CK(cudaEventRecord(sweep.start,stream));
  host_start=clock_ms();size_t dispatched=0;
  pstream(needed,[&](auto id,const ds41::DeviceExpert* table,auto s){
    auto &timer=*compute.at(dispatched++);CK(cudaEventRecord(timer.start,s));
    for(const auto &b:batches.at(id)) {
      ds41e::prefill::gather_routes(pf.quant.as<uint8_t>(),pf.scales.as<uint8_t>(),
          pf.weights.as<float>(base*6*4),gpu_slots+b.offset,pf.bx.as<uint8_t>(),
          pf.bs.as<uint8_t>(),pf.bw.as<float>(),b.rows,s);
      CK(ds41::grouped_mxfp4_prefill(table,gpu_jobs+b.job,1,ds41::Projection::Gate,I,H,b.rows,s));
      CK(ds41::grouped_mxfp4_prefill(table,gpu_jobs+b.job+1,1,ds41::Projection::Up,I,H,b.rows,s));
      CK(ds41::swiglu(pf.gate.as<Bf16>(),pf.up.as<Bf16>(),pf.bw.as<float>(),pf.act.as<Bf16>(),b.rows,I,10,s));
      CK(ds41::quantize_mxfp8(pf.act.as<Bf16>(),pf.eq.as<uint8_t>(),pf.es.as<uint8_t>(),b.rows,I,s));
      CK(ds41::grouped_mxfp4_prefill(table,gpu_jobs+b.job+2,1,ds41::Projection::Down,H,I,b.rows,s));
      ds41e::prefill::scatter_routes(pf.down.p,gpu_slots+b.offset,pf.routed.as<char>(base*6*H*2),b.rows,s);
      CK(cudaGetLastError());
    }
    CK(cudaEventRecord(timer.end,s));
  });
  if(dispatched!=needed.size())throw std::runtime_error("prefill expert dispatch coverage");
  double schedule=clock_ms()-host_start;
  CK(cudaEventRecord(sweep.end,stream));CK(cudaStreamSynchronize(stream));
  float sweep_ms;CK(cudaEventElapsedTime(&sweep_ms,sweep.start,sweep.end));
  double compute_ms=0;for(auto &t:compute){float ms;CK(cudaEventElapsedTime(&ms,t->start,t->end));compute_ms+=ms;}
  auto resident_after=experts->stats();auto stage_after=prefill_transfers;
  if(resident_after.evictions!=resident_before.evictions)
    throw std::runtime_error("decode cache changed during prefill; competing forecasts must be quiesced");
  return {{"tokens_per_expert_histogram",distribution},{"unique_experts",needed.size()},{"expert_chunks",chunks},{"route_boundary_ms",boundary},
      {"expert_schedule_host_ms",schedule},{"expert_sweep_stream_ms",sweep_ms},
      {"expert_compute_stream_ms",compute_ms},{"expert_stream_gap_ms",std::max(0.,double(sweep_ms)-compute_ms)},
      {"h2d_bytes",stage_after.h2d_bytes},
      {"nvme_bytes",stage_after.nvme_bytes},
      {"decode_cache_evictions",resident_after.evictions-resident_before.evictions},
      {"decode_vram_hits",resident_after.hits-resident_before.hits},
      {"borrowed_vram_experts",prefill_vram_hits},{"host_tier_hits",stage_after.host_hits},{"nvme_expert_misses",stage_after.nvme_misses},
      {"h2d_stream_ms",stage_after.h2d_stream_ms},
      {"ram_service_including_backpressure_ms",stage_after.ram_service_host_ms},
      {"nvme_service_including_backpressure_ms",stage_after.nvme_service_host_ms},
      {"staging_mode","isolated-vram-ram-nvme-v0.3"}};
}
#include "batch.inc"
#include "chunk.inc"

void pengram(int layer,const void* raw) {
  auto p="layers."+std::to_string(layer)+".engram";
  CK(cudaMemcpyAsync(rows_interleaved.p,raw,24*264,cudaMemcpyHostToDevice,stream));
  async_engram->dequant(rows_interleaved.p,rows_bf.p,24,stream);
  linear(p+".wkv",rows_bf.p,engram_kv.p,(HC+1)*H,6144);
  CK(ds41_dense_engram_gate(h.p,engram_kv.p,engram_kv.as<char>(HC*H*2),
      w(p+".q_weight").data.p,w(p+".k_weight").data.p,nullptr,next_h.p,1,H,1e-20f,stream));
  std::swap(h,next_h);
}
public:
std::vector<int> cached_prefix() const override {return prefix.ids();}
std::map<std::string,std::vector<uint8_t>> live_state() override {
  if(pending||poisoned)throw std::logic_error("state inspection requires healthy committed boundary");
  std::map<std::string,std::vector<uint8_t>> out;
  auto copy=[&](const std::string& key,const void* pointer,size_t bytes){
    auto &v=out[key];v.resize(bytes);if(bytes)CK(cudaMemcpyAsync(v.data(),pointer,bytes,cudaMemcpyDeviceToHost,stream));
  };
  int64_t first=std::max<int64_t>(0,committed-128);
  for(int l=0;l<40;++l)for(int64_t p=first;p<committed;++p)
    copy("swa/"+std::to_string(l)+"/"+std::to_string(p),windows[l].as<char>(size_t(p%RING)*528),528);
  for(auto &item:globals) {
    auto &g=*item.second;auto key="global/"+std::to_string(item.first);
    size_t rows=size_t(committed/g.ratio);copy(key+"/kv",g.kv.p,rows*288);copy(key+"/index",g.key.p,rows*68);
    if(g.ratio==2&&committed%2) {
      size_t offset=size_t((committed-1)%8)*512*4;
      copy(key+"/raw-kv-tail",g.raw_kv.as<char>(offset),512*4);
      copy(key+"/raw-score-tail",g.raw_score.as<char>(offset),512*4);
    }
  }
  CK(cudaStreamSynchronize(stream));
  const auto &history=hash.history();auto &bytes=out["engram/token-history"];
  bytes.resize(size_t(committed)*sizeof(int64_t));
  if(!bytes.empty())std::memcpy(bytes.data(),history.data(),bytes.size());
  auto &ids=prefix.ids();auto &tokens=out["session/committed-input-ids"];
  tokens.resize(ids.size()*sizeof(int));if(!tokens.empty())std::memcpy(tokens.data(),ids.data(),tokens.size());
  return out;
}
bool supports_prefill(PrefillMode mode,size_t rows) const override {
  if(!config.prefill||config.prefetch_routes||!prefetch_tickets.empty()||
      poisoned||pending||!rows||rows>size_t(config.max_context-committed)||
      (mode!=PrefillMode::Cold&&mode!=PrefillMode::Append)||
      (mode==PrefillMode::Cold?committed!=0:committed==0))return false;
  size_t need=PrefillData::required_bytes(rows,committed,1);
  if(need>config.prefill_budget_bytes)return false;
  // Opt-in (DS41_PREFILL_CHUNK_FIT=1): a segment must also fit the chunked prefill scratch, so long
  // prompts split into fast chunked segments instead of one oversized M1-path prefill. 256 MiB slack
  // plus a 128 MiB margin keep the later allocations (taken in the same order) from racing.
  if(prefill_chunk_fit&&config.prefill_chunk_rows) {
    const int64_t end=committed+int64_t(rows);
    const int width=int(std::min<int64_t>(2048,(end+7)/8)*8);
    for(auto b:PrefillChunk::sizes(int(config.prefill_chunk_rows),end,width,prefill_fp8_elements()))need+=b;
    need+=size_t(256)<<20;
  }
  size_t free,total;CK(cudaMemGetInfo(&free,&total));
  return need<=free&&free-need>=(prefill_chunk_fit?(size_t(128)<<20):(size_t(64)<<20));
}
PrefillResult prefill(const std::vector<int>& input,const PrefillOptions& options) override {
  if(!config.prefill)throw std::logic_error("native prefill requires Config::prefill opt-in");
  if(poisoned)throw std::runtime_error("native forward/store poisoned; restart ds41d");
  if((options.mode!=PrefillMode::Cold&&options.mode!=PrefillMode::Append)||pending||input.empty()||input.size()>size_t(config.max_context-committed)||
      options.logits_tail<1||options.logits_tail>input.size()||options.chunk_rows<1||options.chunk_rows>4096||options.dense_batch_rows<1||options.dense_batch_rows>8||
      (options.mode==PrefillMode::Cold&&committed!=0)||
      (options.mode==PrefillMode::Append&&committed==0))
    throw std::invalid_argument("prefill mode/state/context/chunk/tail");
  for(int id:input)if(id<0||id>=V)throw std::invalid_argument("prefill token id");
  double begin_ms=clock_ms();
  const void* initial_h=h.p;
  prefill_warm_counts.clear();
  std::vector<uint8_t> residency_before;
  try{residency_before=p_residency(*experts);}catch(...){poisoned=true;throw;}
  auto plan=ds41e::prefill::plan(committed,int64_t(input.size()),int64_t(options.logits_tail));
  // Preflight allocations before any live state mutation. No eviction or cache
  // rebuild is used to make room for scratch. A failed allocation is retryable.
  pref=std::make_unique<PrefillData>(input.size(),committed,options.mode==PrefillMode::Cold,
      options.logits_tail,config.prefill_budget_bytes);
  auto &pf=*pref;pf.dense_batch=int(options.dense_batch_rows);pf.warm=options.warm;
  prefill_chunk.reset();
  if(config.prefill_chunk_rows) {
    // Optional scratch: allocated only beside the live decode cache, otherwise the M1 path runs.
    try {
      const int R=int(config.prefill_chunk_rows);const size_t fp8=prefill_fp8_elements();
      auto sizes=PrefillChunk::sizes(R,pf.end,pf.candidate_width,fp8);
      size_t need=0;for(auto b:sizes)need+=b;
      size_t free,total;CK(cudaMemGetInfo(&free,&total));
      if(need+(size_t(256)<<20)<=free)prefill_chunk=std::make_unique<PrefillChunk>(R,pf.end,pf.candidate_width,fp8);
    }catch(const std::exception&){prefill_chunk.reset();cudaGetLastError();}
  }
  PrefillResult result;result.begin_position=committed;result.consumed_rows=input.size();
  try {
    in_step=true;graph_ready=false;
    for(size_t r=0;r<input.size();++r) {
      auto ids=hash.rows(input[r],committed+r);
      for(int j=0;j<2;++j)pf.rowids[j].insert(pf.rowids[j].end(),ids[j].begin(),ids[j].end());
    }
    // Both whole-suffix tickets deduplicate globally. They start before layer0
    // and stay alive through their last H2D. Appends hash/submit only new IDs.
    if(async_engram)for(int j=0;j<2;++j)pf.engram[j]=async_engram->submit(j==0?1:14,pf.rowids[j]);
    // Start I/O before enqueueing prompt-sized embedding work as well.
    for(size_t r=0;r<input.size();++r) {
      expand_embedding(w("embed.weight").data.as<Bf16>(size_t(input[r])*H*2),h.p,pre.as<float>(),H,stream);
      pcopy(pf.hidden.as<char>(r*HC*H*2),h.p,HC*H*2);
      pcopy(pf.previous_pre.as<char>(r*4*4),pre.p,4*4);
    }
    CK(cudaStreamSynchronize(stream));
    for(const auto &work:plan) {
      int l=work.layer;auto p="layers."+std::to_string(l);pf.query=work.query;
      upload_cursor=1<<20;
      CK(cudaMemsetAsync(pf.window.p,0,pf.window.bytes,stream));
      if(!pf.cold) {
        int64_t old=std::max<int64_t>(0,pf.begin-128);
        upload(position_device,&old,8);
        // packed scratch is one row, so copy the ring's retained tail in two
        // device spans with absolute modulo addressing and no host roundtrip.
        for(int64_t pos=old;pos<pf.begin;++pos)
          pcopy(pf.window.as<char>(size_t(pos%int64_t(pf.rows+128))*528),windows[l].as<char>(size_t(pos%RING)*528),528);

      }
      Phase project;CK(cudaEventRecord(project.start,stream));
      auto projection=work.global.size()?work.global:work.window;
      for(int64_t first=projection.begin;first<projection.end;first+=int64_t(options.chunk_rows)) {
        int64_t end=std::min(projection.end,first+int64_t(options.chunk_rows));
        upload_cursor=1<<20;
        const char* raw=nullptr;
        if((l==1||l==14)&&async_engram) {
          int j=l==1?0:1;
          if(!pf.engram_raw[j])pf.engram_raw[j]=static_cast<const char*>(pf.engram[j]->wait(pf.rows*24));
          raw=pf.engram_raw[j]+size_t(first-pf.begin)*24*264;
        }
        if(prefill_chunk) {
          const int64_t R=prefill_chunk->R;
          for(int64_t pos=first;pos<end;pos+=R)
            cproject(work,pos,int(std::min<int64_t>(R,end-pos)),raw?raw+size_t(pos-first)*24*264:nullptr);
        } else if(pf.dense_batch>1) {
          if(!prefill_batch)prefill_batch=std::make_unique<PrefillBatch>();
          for(int64_t pos=first;pos<end;pos+=pf.dense_batch)
            pproject_batch(work,pos,int(std::min<int64_t>(pf.dense_batch,end-pos)),
                raw?raw+size_t(pos-first)*24*264:nullptr);
        } else for(int64_t pos=first;pos<end;++pos) {
          size_t r=size_t(pos-pf.begin);
          pcopy(h.p,pf.hidden.as<char>(r*HC*H*2),HC*H*2);
          pcopy(pre.p,pf.previous_pre.as<char>(r*4*4),4*4);
          if(l==1||l==14) {
            if(raw)pengram(l,raw+size_t(pos-first)*24*264);
            else {auto &all=pf.rowids[l==1?0:1];engram(l,{all.begin()+r*24,all.begin()+(r+1)*24});}
          }
          if(work.capture.contains(pos))capture_mean(h.p,pf.capture.as<char>(size_t(pos-work.capture.begin)*3*H*2+(l-37)*H*2),H,stream);
          hc(p,"attn",att_pre);
          CK(ds41_dense_hc_pre(h.p,pre.as<float>(),x.p,1,H,stream));norm(p+".attn_norm",x.p,x.p,H);
          if(work.query.contains(pos)) {
            linear(p+".attn.wq_a",x.p,qa.p,1280,H);norm(p+".attn.q_norm",qa.p,qa.p,1280);
            linear(p+".attn.wq_b",qa.p,q.p,32768,1280);
          }
          linear(p+".attn.wkv",x.p,kv.p,512,H);norm(p+".attn.kv_norm",kv.p,kv.p,512);
          pattention(l,pos);
          if(!work.query.contains(pos))continue;
          CK(ds41_dense_hc_post(y.p,h.p,post.as<float>(),comb.as<float>(),next_h.p,1,H,stream));std::swap(h,next_h);
          pcopy(pf.hidden.as<char>(r*HC*H*2),h.p,HC*H*2);
          hc(p,"ffn",ffn_pre);
          CK(ds41_dense_hc_pre(h.p,att_pre.as<float>(),x.p,1,H,stream));norm(p+".ffn_norm",x.p,x.p,H);
          router_scores(l);
          ds41e::prefill::select_routes(biased.as<float>(),unbiased.as<float>(),pf.ids.as<int>(r*6*4),pf.weights.as<float>(r*6*4),stream);
          pcopy(pf.ffn_input.as<char>(r*H*2),x.p,H*2);
          pcopy(pf.ffn_post.as<char>(r*4*4),post.p,4*4);pcopy(pf.ffn_comb.as<char>(r*16*4),comb.p,16*4);
          pcopy(pf.previous_pre.as<char>(r*4*4),ffn_pre.p,4*4);
        }
        CK(cudaStreamSynchronize(stream)); // metadata and Engram ticket lifetime
      }
      CK(cudaEventRecord(project.end,stream));CK(cudaStreamSynchronize(stream));
      float project_ms;CK(cudaEventElapsedTime(&project_ms,project.start,project.end));
      Json timings=pmoe(l,work.query);
      Phase epilogue;CK(cudaEventRecord(epilogue.start,stream));
      if(prefill_chunk) {
        const int64_t R=prefill_chunk->R;
        for(int64_t pos=work.query.begin;pos<work.query.end;pos+=R)
          cshared(l,pos,int(std::min<int64_t>(R,work.query.end-pos)));
      } else if(pf.dense_batch>1) {
        for(int64_t pos=work.query.begin;pos<work.query.end;pos+=pf.dense_batch)
          pshared_batch(l,pos,int(std::min<int64_t>(pf.dense_batch,work.query.end-pos)));
      } else for(int64_t pos=work.query.begin;pos<work.query.end;++pos) {
        size_t r=size_t(pos-pf.begin);
        pcopy(x.p,pf.ffn_input.as<char>(r*H*2),H*2);
        linear(p+".ffn.shared_experts.w1",x.p,sg.p,I,H);linear(p+".ffn.shared_experts.w3",x.p,su.p,I,H);
        CK(ds41::swiglu(sg.as<Bf16>(),su.as<Bf16>(),nullptr,sa.as<Bf16>(),1,I,10,stream));
        linear(p+".ffn.shared_experts.w2",sa.p,shared.p,H,I);
        CK(ds41::reduce_experts(pf.routed.as<Bf16>(r*6*H*2),pf.ids.as<int32_t>(r*6*4),shared.as<Bf16>(),y.as<Bf16>(),1,6,H,stream));
        CK(ds41_dense_hc_post(y.p,pf.hidden.as<char>(r*HC*H*2),pf.ffn_post.as<float>(r*4*4),pf.ffn_comb.as<float>(r*16*4),next_h.p,1,H,stream));
        pcopy(pf.hidden.as<char>(r*HC*H*2),next_h.p,HC*H*2);
      }
      int64_t tail=std::max<int64_t>(0,pf.end-128);
      for(int64_t pos=tail;pos<pf.end;++pos)
        pcopy(windows[l].as<char>(size_t(pos%RING)*528),pf.window.as<char>(size_t(pos%int64_t(pf.rows+128))*528),528);
      CK(cudaEventRecord(epilogue.end,stream));CK(cudaStreamSynchronize(stream));
      float epilogue_ms;CK(cudaEventElapsedTime(&epilogue_ms,epilogue.start,epilogue.end));
      timings["engram_prestage_barrier_ms_cumulative"]=pf.engram_barrier_ms;
      timings["layer"]=l;timings["query_rows"]=work.query.size();timings["project_route_stream_ms"]=project_ms;
      timings["shared_reduce_hc_stream_ms"]=epilogue_ms;pf.layers.push_back(timings);
      std::cerr<<"prefill layer="<<l<<" rows="<<work.query.size()<<" project_ms="<<project_ms
          <<" expert_ms="<<timings["expert_sweep_stream_ms"]<<" epilogue_ms="<<epilogue_ms<<"\n";
    }
    for(size_t r=input.size()-options.logits_tail;r<input.size();++r) {
      pcopy(h.p,pf.hidden.as<char>(r*HC*H*2),HC*H*2);pcopy(pre.p,pf.previous_pre.as<char>(r*4*4),4*4);
      CK(ds41_dense_hc_pre(h.p,pre.as<float>(),x.p,1,H,stream));norm("norm",x.p,x.p,H);
      bf16_float(x.p,f32.as<float>(),H,stream);if(fused_head)fused_head_logits(f32.p,logits.p,1);else linear("head",f32.p,logits.p,V,H);
      result.logits.emplace_back(V);
      CK(cudaMemcpyAsync(result.logits.back().data(),logits.p,V*4,cudaMemcpyDeviceToHost,stream));
      CK(cudaStreamSynchronize(stream));
    }
    if(!residency_before.empty()&&p_residency(*experts)!=residency_before)
      throw std::runtime_error("prefill changed decode tier membership; quiesce competing admissions");
    prefix.commit(input,input.size());
    committed=pf.end;
    size_t capture_count=std::min<size_t>(128,pf.rows);
    for(size_t first=0;first<capture_count;first+=6) {
      capture_rows=int(std::min<size_t>(6,capture_count-first));capture_committed=capture_rows;
      capture_begin=pf.end-capture_count+first;
      pcopy(capture_slab.p,pf.capture.as<char>(first*3*H*2),capture_rows*3*H*2);
      if(capture_sink){auto view=captures();int rc=capture_sink(capture_user,&view);if(rc)throw std::runtime_error("prefill capture sink failed");}
      CK(cudaStreamSynchronize(stream));
    }
    // Decode graphs capture these scratch addresses. The clipped cone can run
    // an odd number of attention rows, unlike the full40-layer token schedule.
    if(h.p!=initial_h)std::swap(h,next_h);
    result.layers=pf.layers;in_step=false;
    const size_t chunk_rows=prefill_chunk?size_t(prefill_chunk->R):0;
    prefill_chunk.reset();pref.reset();result.elapsed_ms=clock_ms()-begin_ms;total_ms+=result.elapsed_ms;
    prefill_summary={{"mode",options.mode==PrefillMode::Cold?"cold-full-N-M1-dense-candidate":"append-decode-layout"},
        {"rows",input.size()},{"chunk_rows",options.chunk_rows},{"dense_batch_rows",options.dense_batch_rows},{"batched_chunk_rows",chunk_rows},{"cache_reused_tokens",result.begin_position},
        {"elapsed_ms",result.elapsed_ms},{"cache_retention_checked",!residency_before.empty()},{"layers",result.layers}};
    if(!prefill_warm_counts.empty()) {
      // Residency only (after the tier-membership check): warm RAM with the prompt tail's experts.
      std::vector<std::pair<int,ds41::ExpertId>> order;
      for(auto [id,count]:prefill_warm_counts)order.push_back({-count,id});
      std::sort(order.begin(),order.end());
      std::vector<ds41::ExpertId> ids;for(auto [count,id]:order)ids.push_back(id);
      double warm_start=clock_ms();
      size_t read=experts->warm_host(ids,size_t(config.prefill_warm_gib*double(size_t(1)<<30)));
      prefill_summary["warm_host"]={{"candidates",ids.size()},{"read",read},{"ms",clock_ms()-warm_start}};
      prefill_warm_counts.clear();
    }
    return result;
  }catch(...){cudaStreamSynchronize(stream);prefill_chunk.reset();poisoned=true;in_step=false;throw;}
}
private:
