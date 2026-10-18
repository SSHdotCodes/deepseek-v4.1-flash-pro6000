#include "ds41/engine.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <iostream>
using namespace ds41e;
static void equal(const std::vector<float>& a,const std::vector<float>& b,const char* context) {
  size_t mismatch=0;for(size_t i=0;i<a.size()&&i<b.size();++i)mismatch+=std::memcmp(&a[i],&b[i],4)!=0;
  if(a.size()!=b.size()||mismatch)throw std::runtime_error(std::string(context)+": logits differ: "+std::to_string(mismatch));
}
using State=std::map<std::string,std::vector<uint8_t>>;
static void equal(const State&a,const State&b){
  if(a==b)return;for(auto &item:a)if(!b.count(item.first)||b.at(item.first)!=item.second)
    throw std::runtime_error("live state byte mismatch: "+item.first);
  throw std::runtime_error("live state keys mismatch");
}
struct Captures {
  std::map<int64_t,std::vector<uint8_t>> values;
  static int ingest(void* user,const ds41_target_capture_v1* view){
    auto &self=*static_cast<Captures*>(user);
    for(int r=0;r<view->committed_rows;++r){auto&v=self.values[view->begin_position+r];v.resize(30720);
      if(cudaMemcpyAsync(v.data(),static_cast<const char*>(view->hidden_bf16)+r*view->row_stride_bytes,v.size(),cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(view->cuda_stream)))return 1;
    }
    return int(cudaStreamSynchronize(static_cast<cudaStream_t>(view->cuda_stream)));
  }
};
int main(int argc,char**argv){
 try {
  if(argc<8||argc>9)throw std::runtime_error("native_prefill WEIGHTS RECEIPT ENGRAM ROPE EXPERTS SLOTS ENGRAM_LIB [ROWS]");
  Config c;c.weights=argv[1];c.receipt=argv[2];c.engram=argv[3];c.rope=argv[4];c.experts=argv[5];
  c.expert_slots=std::stoull(argv[6]);c.engram_library=argv[7];c.prefill=true;c.graphs=true;c.allow_unqualified=true;c.max_context=8192;
  int n=argc==9?std::stoi(argv[8]):13;
  std::vector<int> ids(n+1);for(int i=0;i<=n;++i)ids[i]=i?100+i*17:0;
  c.host_bytes=size_t(4)<<30;
  auto placement=std::filesystem::temp_directory_path()/("ds41-prefill-placement-"+std::to_string(getpid())+".placement");
  {std::ofstream f(placement);f<<"# DS41PLACEMENT01\npools soft\n";
   for(int layer=0;layer<40;++layer)for(int expert=0;expert<4;++expert)f<<"host "<<layer*384+expert<<'\n';
   if(!f)throw std::runtime_error("cannot write native prefill placement fixture");}
  c.placement=placement.string();
  auto model=make_native(c);std::filesystem::remove(placement);Captures capture;model->set_capture_sink(Captures::ingest,&capture);
  std::vector<std::vector<float>> sequential;State baseline;
  for(int i=0;i<=n;++i){sequential.push_back(model->verify({ids[i]})[0]);model->commit(1);if(i==n-1)baseline=model->live_state();}
  auto wanted_captures=capture.values;capture.values.clear();model->reset();
  for(int i=0;i<2;++i){equal(model->verify({ids[i]})[0],sequential[i],"seed");model->commit(1);}
  PrefillOptions o;o.mode=PrefillMode::Append;o.logits_tail=n-2;o.chunk_rows=5;o.dense_batch_rows=8;
  auto append=model->prefill({ids.begin()+2,ids.begin()+n},o);
  for(int i=2;i<n;++i)equal(append.logits[i-2],sequential[i],"append X3b component");
  equal(baseline,model->live_state());
  auto expected_captures=std::map<int64_t,std::vector<uint8_t>>(wanted_captures.begin(),wanted_captures.lower_bound(n));
  for(int i=2;i<std::max(2,n-128);++i)expected_captures.erase(i);
  if(capture.values!=expected_captures)throw std::runtime_error("append captures differ");
  equal(model->verify({ids[n]})[0],sequential[n],"post-append decode");model->commit(1);
  auto append_stats=model->counters();
  uint64_t host_hits=0;for(auto &layer:append.layers)host_hits+=layer.value("host_tier_hits",uint64_t(0));
  // Same cold mode, different scheduling chunks, including odd CR2 tails.
  std::vector<std::vector<float>> cold;State cold_state;std::map<int64_t,std::vector<uint8_t>> cold_captures;
  Json cold_runs=Json::array();std::vector<float> cold_next;
  for(auto policy:std::vector<std::pair<int,int>>{{1,1},{5,8},{509,8},{509,2},{509,3},{509,4},{509,5},{509,6},{509,7}}) {
    int chunk=policy.first;o.dense_batch_rows=policy.second;
    model->reset();capture.values.clear();o.mode=PrefillMode::Cold;o.chunk_rows=chunk;o.logits_tail=n;
    auto r=model->prefill({ids.begin(),ids.begin()+n},o);
    if(cold.empty()){cold=r.logits;cold_state=model->live_state();cold_captures=capture.values;}
    else {for(int i=0;i<n;++i)equal(r.logits[i],cold[i],"cold X3 chunk scheduling");equal(cold_state,model->live_state());if(capture.values!=cold_captures)throw std::runtime_error("cold captures differ");}
    for(auto &layer:r.layers)host_hits+=layer.value("host_tier_hits",uint64_t(0));
    cold_runs.push_back({{"chunk_rows",chunk},{"dense_batch_rows",o.dense_batch_rows},{"layers",r.layers}});
    auto next=model->verify({ids[n]})[0];model->commit(1);
    if(cold_next.empty())cold_next=next;else equal(cold_next,next,"cold post-prefill graph decode");
  }
  model->reset();capture.values.clear();o.mode=PrefillMode::Cold;o.logits_tail=1;o.chunk_rows=5;
  auto cone=model->prefill({ids.begin(),ids.begin()+n},o);
  equal(cone.logits[0],cold.back(),"cold CED tail");equal(cold_state,model->live_state());
  if(capture.values!=cold_captures)throw std::runtime_error("CED captures differ");
  equal(model->verify({ids[n]})[0],cold_next,"CED post-prefill cached graph decode");model->commit(1);
  // Single-row cold uses exactly the accepted position0 prefill geometry.
  model->reset();o.logits_tail=1;equal(model->prefill({ids[0]},o).logits[0],sequential[0],"cold single row");
  if(!host_hits)throw std::runtime_error("host fixture did not exercise borrowed RAM");
  std::cout<<Json({{"schema","ds41-prefill-native-gate-v1"},{"rows",n},{"append_full_logits_bitwise",true},
      {"append_live_state_bytes_bitwise",true},{"append_captures_bitwise",true},{"post_append_decode_bitwise",true},
      {"cold_chunk_invariance",true},{"dense_batches_1_to_8_bitwise",true},{"cold_single_row",true},{"cold_cone_state_bitwise",true},{"cold_full_M_gold","PENDING"},
      {"host_tier_hits",host_hits},{"append_stats",append_stats},{"cold_runs",cold_runs}}).dump()<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
