#include "ds41/prefill/stager.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
namespace ds41e::prefill {
namespace {
void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
double ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Event {
 cudaEvent_t e=nullptr;
 Event(){ck(cudaEventCreate(&e));}
 ~Event(){if(e)cudaEventDestroy(e);}
 Event(const Event&)=delete;Event& operator=(const Event&)=delete;
};
}
struct Stager::Impl {
 size_t slots,record_bytes=0;
 int device=0;
 std::map<ds41::ExpertId,ds41::Record> records;
 ds41::DirectReader reader{64};
 std::unique_ptr<ds41::HostBuffer> staging,table;
 void* weights=nullptr;
 cudaStream_t ram=nullptr,disk=nullptr;
 std::vector<std::unique_ptr<Event>> consumed,dma_done;
 std::vector<unsigned char> used;
 std::mutex enqueue;
 std::exception_ptr poison;
 Impl(std::vector<ds41::Record> input,size_t n):slots(n),used(n,false) {
  if(n<2||input.empty())throw std::invalid_argument("isolated staging needs at least two slots");
  for(auto &r:input){if(!r.bytes||r.bytes>SIZE_MAX-4095)throw std::invalid_argument("stager record size");
   for(auto v:{r.gate,r.gate_scale,r.up,r.up_scale,r.down,r.down_scale})if(v.offset>r.bytes||v.bytes>r.bytes-v.offset)throw std::invalid_argument("stager view bounds");
   if(!records.emplace(r.id,r).second||r.id>=ds41::expert_count)throw std::invalid_argument("stager expert ID");record_bytes=std::max(record_bytes,(r.bytes+4095)&~size_t(4095));}
  try {
   ck(cudaGetDevice(&device));ck(cudaMalloc(&weights,slots*record_bytes));
   ck(cudaStreamCreateWithFlags(&ram,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&disk,cudaStreamNonBlocking));
   staging=std::make_unique<ds41::HostBuffer>(slots*record_bytes,true);
   table=std::make_unique<ds41::HostBuffer>(ds41::expert_count*sizeof(ds41::DeviceExpert),true);
   std::fill_n(static_cast<ds41::DeviceExpert*>(table->data()),ds41::expert_count,ds41::DeviceExpert{});
   for(size_t i=0;i<slots;++i){consumed.push_back(std::make_unique<Event>());dma_done.push_back(std::make_unique<Event>());}
  }catch(...){clear();throw;}
 }
 void clear()noexcept {
  if(ram)cudaStreamSynchronize(ram);if(disk)cudaStreamSynchronize(disk);
  // Last consumers must finish before weight slots and mapped descriptors die.
  for(size_t i=0;i<consumed.size();++i)if(used[i])cudaEventSynchronize(consumed[i]->e);
  consumed.clear();dma_done.clear();table.reset();staging.reset();
  if(weights)cudaFree(weights);weights=nullptr;
  if(ram)cudaStreamDestroy(ram);if(disk)cudaStreamDestroy(disk);ram=disk=nullptr;
 }
 ~Impl(){clear();}
 void upload(ds41::ExpertId id,const void* data,size_t slot,cudaStream_t copy,
             cudaStream_t consumer,const Ready& ready,Event& begin,Event& end) {
  if(used[slot])ck(cudaEventSynchronize(consumed[slot]->e));
  auto &r=records.at(id);auto* base=static_cast<uint8_t*>(weights)+slot*record_bytes;
  // Each logical ID appears once in a sweep; this mapped descriptor stays
  // stable until all callbacks in that sweep finish. No pageable H2D metadata.
  auto* views=static_cast<ds41::DeviceExpert*>(table->data());
  views[id]={base+r.gate.offset,base+r.gate_scale.offset,base+r.up.offset,base+r.up_scale.offset,
      base+r.down.offset,base+r.down_scale.offset,nullptr,0,int32_t(slot)};
  ck(cudaEventRecord(begin.e,copy));
  ck(cudaMemcpyAsync(base,data,r.bytes,cudaMemcpyHostToDevice,copy));
  ck(cudaEventRecord(end.e,copy));
  {std::lock_guard<std::mutex> lock(enqueue);
   ck(cudaStreamWaitEvent(consumer,end.e,0));
   ready(id,static_cast<ds41::DeviceExpert*>(table->device_data()),consumer);
   ck(cudaEventRecord(consumed[slot]->e,consumer));used[slot]=true;
  }
 }
};
Stager::Stager(std::vector<ds41::Record> records,size_t slots):impl(new Impl(std::move(records),slots)){}
Stager::~Stager()=default;
size_t Stager::device_bytes() const noexcept{return impl->slots*impl->record_bytes;}
TransferStats Stager::stream(const std::vector<ds41::ExpertId>& ids,cudaStream_t consumer,
    const Ready& ready,const HostLookup& host,const std::function<void()>& before_disk) {
 auto &p=*impl;if(p.poison)std::rethrow_exception(p.poison);
 if(!consumer||!ready)throw std::invalid_argument("stager consumer/callback");
 // A later sweep may reuse a logical table entry in a different slot.
 // Drain the preceding sweep before republishing any mapped descriptor.
 for(size_t i=0;i<p.slots;++i)if(p.used[i])ck(cudaEventSynchronize(p.consumed[i]->e));
 double start=ms();TransferStats stats;
 std::vector<std::pair<ds41::ExpertId,HostView>> ram;
 std::vector<const ds41::Record*> disk;
 std::set<ds41::ExpertId> seen;
 for(auto id:ids){
  if(!seen.insert(id).second)throw std::invalid_argument("stager duplicate expert");
  auto &r=p.records.at(id);auto view=host?host(id):HostView{};
  if(view.data){if(view.bytes<r.bytes||!view.keepalive)throw std::invalid_argument("unsafe prefill host view");
   cudaPointerAttributes attributes{};ck(cudaPointerGetAttributes(&attributes,view.data));
   if(attributes.type!=cudaMemoryTypeHost)throw std::invalid_argument("prefill host view must be pinned host memory");
   ram.emplace_back(id,std::move(view));++stats.host_hits;}
  else {disk.push_back(&r);++stats.nvme_misses;for(auto &e:r.extents)stats.nvme_bytes+=e.bytes;}
  stats.h2d_bytes+=r.bytes;
 }
 if(ids.empty())return stats;
 size_t ram_slots=ram.empty()?0:disk.empty()?p.slots:std::max<size_t>(1,p.slots/2);
 size_t disk_slots=p.slots-ram_slots;
 std::vector<std::unique_ptr<Event>> begins,ends;
 for(size_t i=0;i<ids.size();++i){begins.push_back(std::make_unique<Event>());ends.push_back(std::make_unique<Event>());}
 std::future<void> ram_job;
 try {
  if(!ram.empty())ram_job=std::async(std::launch::async,[&]{
   double ram_start=ms();ck(cudaSetDevice(p.device));
   for(size_t i=0;i<ram.size();++i)p.upload(ram[i].first,ram[i].second.data,i%ram_slots,p.ram,consumer,ready,*begins[i],*ends[i]);
   ck(cudaStreamSynchronize(p.ram)); // borrowed host leases end only after DMA
   stats.ram_service_host_ms=ms()-ram_start;
  });
  if(!disk.empty()) {
   if(before_disk)before_disk();
   double io_start=ms();
   std::vector<void*> ring;
   for(size_t i=0;i<disk_slots;++i)ring.push_back(static_cast<char*>(p.staging->data())+i*p.record_bytes);
   p.reader.stream(disk,ring,[&](size_t i,size_t buffer){
    p.upload(disk[i]->id,ring[buffer],ram_slots+buffer,p.disk,consumer,ready,*begins[ram.size()+i],*ends[ram.size()+i]);
    ck(cudaEventRecord(p.dma_done[buffer]->e,p.disk));
   },[&](size_t buffer){ck(cudaEventSynchronize(p.dma_done[buffer]->e));});
   ck(cudaStreamSynchronize(p.disk));
   stats.nvme_service_host_ms=ms()-io_start;
  }
  if(ram_job.valid())ram_job.get();
  for(size_t i=0;i<ids.size();++i){float elapsed;ck(cudaEventElapsedTime(&elapsed,begins[i]->e,ends[i]->e));stats.h2d_stream_ms+=elapsed;}
  stats.host_service_ms=ms()-start;
  return stats;
 }catch(...){
  p.poison=std::current_exception();
  if(ram_job.valid())try{ram_job.get();}catch(...){}
  cudaStreamSynchronize(p.ram);cudaStreamSynchronize(p.disk);cudaStreamSynchronize(consumer);
  std::rethrow_exception(p.poison);
 }
}
}
