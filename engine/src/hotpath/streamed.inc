// v0.2 callbacks execute on the store worker. All per-batch scratch is owned by
// Native and reused only through its ordered consumer stream. Callbacks never
// reenter the store or wait for GPU work.
void bmoe_streamed(const std::string &p,int m,const hotpath::RoutePlan &plan,
                  const std::vector<ds41::GemmJob> &gj,
                  const std::vector<ds41::GemmJob> &uj,
                  const std::vector<ds41::GemmJob> &dj) {
  bgraph(p+".shared",m,[&]{bshared(p,m);});
  const int layer=int(plan.ids.front()/384);
  const bool readiness=consumer_readiness && !lane_b_active;
  if(readiness) consumer_readiness->shared_done(layer,stream);
  double t=clock_ms();
  // Lane B demands skip the width-one drift contract (like DSpark's); LFU/tiers still see them.
  auto lease=experts->need_streamed_batches(plan.ids,stream,
      [&](const std::vector<ds41::ExpertId> &ready,cudaStream_t consumer){
    double enqueue_start=clock_ms();
    if(consumer!=stream || ready.empty()) throw std::logic_error("stream callback contract");
    if(readiness) consumer_readiness->arrived(layer,stream);
    std::array<ds41::GemmJob,36> gates{},ups{},downs{};
    std::array<int,36> rows{}; int count=0,n=int(ready.size());
    for(int i=0;i<n;++i) {
      auto it=std::lower_bound(plan.ids.begin(),plan.ids.end(),ready[i]);
      if(it==plan.ids.end() || *it!=ready[i]) throw std::logic_error("foreign ready expert");
      size_t ix=size_t(it-plan.ids.begin());
      gates[i]=gj[ix]; ups[i]=uj[ix]; downs[i]=dj[ix];
      const auto &e=plan.experts[ix];
      for(int j=0;j<e.count;++j) rows[count++]=e.begin+j;
    }
    zc_begin();
    upload(work_jobs,gates.data(),n*sizeof(ds41::GemmJob));
    upload(work_up_jobs,ups.data(),n*sizeof(ds41::GemmJob));
    upload(work_down_jobs,downs.data(),n*sizeof(ds41::GemmJob));
    upload(work_rows,rows.data(),count*4);
    zc_end();
    ++stream_callbacks;
    if(n>1) ++stream_hit_batches;
    // Node launch geometry depends only on counts. Actual logical IDs, token
    // membership and physical slots are device metadata refreshed before replay.
    Phase *event=nullptr;
    if(config.hotpath_profile && !lane_b_active) {
      if(expert_event_index==expert_events.size()) throw std::logic_error("expert profile event capacity");
      event=expert_events[expert_event_index++].get();
      CK(cudaEventRecord(event->start,stream));
    }
    // The device job descriptors carry every layer/expert/output address.
    // Reuse this body across layers; dimensions and scratch addresses are fixed.
    bgraph("experts.stream."+std::to_string(n)+"."+std::to_string(count),m,[&]{
      CK(expert_gemm(work_jobs.as<ds41::GemmJob>(),n,ds41::Projection::Gate,I,H,stream));
      CK(expert_gemm(work_up_jobs.as<ds41::GemmJob>(),n,ds41::Projection::Up,I,H,stream));
      if(exl3) hotpath::activate_routes_bf16(gate.p,up.p,broute.as<float>(),work_rows.as<int>(),
          act.p,count,stream);
      else hotpath::activate_routes(gate.p,up.p,broute.as<float>(),work_rows.as<int>(),
          eq.as<uint8_t>(),es.as<uint8_t>(),count,stream);
      CK(expert_gemm(work_down_jobs.as<ds41::GemmJob>(),n,ds41::Projection::Down,H,I,stream));
    });
    if(event) {
      CK(cudaEventRecord(event->end,stream));
      consumer_readiness->expert_done(layer,event->end);
    }
    callback_enqueue_ms+=clock_ms()-enqueue_start;
  },!lane_b_active);
  io_ms+=clock_ms()-t;
  try {
    // Logical IDs and scatter maps are refreshed before each ordered launch.
    bgraph("experts.reduce",m,[&]{
      hotpath::scatter_routes(compact_down.p,route_slots.as<int>(),down.p,plan.count,stream);
      CK(ds41::reduce_experts(down.as<Bf16>(),ids_device.as<int32_t>(),
          shared.as<Bf16>(),y.as<Bf16>(),m,6,H,stream));
    });
    retire_experts(lease);
  } catch(...) {
    CK(cudaStreamSynchronize(stream));
    if(lease.owner) experts->retire(std::move(lease),stream);
    throw;
  }
  previous_routes[layer]=plan.ids;
  if(lookahead_layer==layer+1 && !lane_b_active) {
    lookahead_layer=-1;
    CK(cudaEventSynchronize(lookahead_sync->end));
    const auto *scores=static_cast<const float *>(lookahead_host->data());
    auto rank=[](const float *score,int k){
      std::array<int,384> order{};
      std::iota(order.begin(),order.end(),0);
      std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](int a,int b){
        return score[a]==score[b]?a<b:score[a]>score[b];
      });
      return std::vector<int>(order.begin(),order.begin()+k);
    };
    if(lookahead_audit.is_open())
      for(int d=1;d<=3 && layer+d<40;++d) audit_pred[layer+d][d-1]=rank(scores+(d-1)*384,32);
    if(!lookahead_dryrun) {
      // Every row's top-k, best score first across rows; at most lookahead_n candidates. The store
      // drops VRAM-resident and NVMe-only ones and uploads the RAM-resident rest behind this demand.
      std::vector<std::pair<float,int>> cand;
      for(int r=0;r<lookahead_m;++r)
        for(int e:rank(scores+r*384,lookahead_k)) cand.push_back({scores[r*384+e],e});
      std::stable_sort(cand.begin(),cand.end(),[](const auto &a,const auto &b){return a.first>b.first;});
      std::vector<ds41::ExpertId> predicted; std::set<int> seen;
      for(auto &c:cand) if(seen.insert(c.second).second && int(predicted.size())<lookahead_n)
        predicted.push_back(ds41::ExpertId((layer+1)*384+c.second));
      prefetch_ram_experts(predicted);
    }
  }
  // Predictions affect transfers only. They never choose target routing or
  // admission. Queued prefetch is bounded by the store's one-expert chunk.
  if(config.prefetch_routes && layer+1<40 && !previous_routes[layer+1].empty()) {
    prefetch_experts(previous_routes[layer+1],PrefetchIntent::NextLayer);
  }
}
