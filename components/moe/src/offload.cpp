#include "ds41/offload.hpp"
#include "ds41/gds_reader.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ds41 {
namespace {
void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
using Clock = std::chrono::steady_clock;
uint64_t ns(Clock::duration d) { return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count(); }
}
struct OffloadStore::Impl {
    Config cfg;
    CachePolicy policy_options;
    bool placed=false;
    std::unique_ptr<DriftPolicy> drift;
    std::ofstream policy_audit;
    std::vector<SlotPool::Allocation> audit_ram;
    uint64_t audit_call=0;
    bool chunk_upload=false;
    // Opt-in exclusive RAM tier (DS41_MOE_DEMOTE=1). Decode demand keeps `demote_reserve` VRAM
    // slots free: policy victims drain into them, and a victim not already in RAM is copied
    // back (D2H on its own stream) into a RAM slot held until the copy lands. NVMe misses in
    // decode are staged instead of admitted to RAM; RAM evicts VRAM-resident duplicates first.
    bool demote=false;
    size_t demote_reserve=16;
    bool demote_eager=false;
    cudaStream_t demote_stream=nullptr;
    struct Drain { size_t slot; ExpertId id; cudaEvent_t event; std::vector<SlotPool::Allocation> held; };
    std::deque<Drain> draining;
    std::unordered_map<ExpertId,cudaEvent_t> demoting;
    // Opt-in GPUDirect Storage (DS41_MOE_GDS=1): decode NVMe misses read straight into their VRAM
    // slot via cuFile P2P, without RAM admission or a host bounce. Prefill/prefetch keep the reader.
    std::unique_ptr<GdsReader> gds;
    static bool flag(const char* name) {
        const char* value=std::getenv(name);if(!value || std::string(value)=="0")return false;
        if(std::string(value)!="1")throw std::invalid_argument(std::string(name)+" must be 0 or 1");
        return true;
    }
    void apply_options() {
        pool.fast_minima(policy_options.fast_minima);
        if(host_pool)host_pool->fast_minima(policy_options.fast_minima);
        if(policy_options.decay256 || policy_options.drift_0_50) {
            pool.decay_interval(256);if(host_pool)host_pool->decay_interval(256);
        }
    }
    SlotPool pool;
    std::unique_ptr<SlotPool> host_pool;
    DirectReader reader;
    std::unordered_map<ExpertId,Record> records;
    std::unordered_map<ExpertId,size_t> host;
    std::shared_ptr<HostBuffer> host_arena;
    std::unique_ptr<HostBuffer> staging, entries, generations;
    size_t slot_bytes=0, host_bytes=0;
    uint8_t* vram=nullptr;
    DeviceExpert* table=nullptr;
    uint32_t* ready=nullptr;
    std::vector<cudaEvent_t> slot_copy,slot_mapping,staging_copy;
    std::vector<bool> known_ready;
    cudaStream_t copy=nullptr, mapping=nullptr;
    cudaEvent_t speculative_done=nullptr;
    bool speculative_pending=false;
    cudaEvent_t copy_idle=nullptr;
    struct HostRetirement {std::vector<std::shared_ptr<const void>> tokens;cudaEvent_t event;};
    std::deque<HostRetirement> host_retired;
    std::set<ExpertId> host_prefetched;
    Clock::time_point refill_next{};
    int device=0;
    struct Retirement { std::vector<SlotPool::Allocation> allocations; cudaEvent_t event; bool consumer=false; };
    std::deque<Retirement> retired;
    std::map<uint64_t,std::vector<SlotPool::Allocation>> live;
    std::set<ExpertId> prefetched;
    uint64_t serial=0;
    std::atomic<uint64_t> ticket_serial{0};
    Stats counters;
    std::exception_ptr poison;
    std::mutex error_mutex;
    std::exception_ptr error(){std::lock_guard<std::mutex> lock(error_mutex);return poison;}
    void fail(std::exception_ptr e)noexcept{if(!e)return;std::lock_guard<std::mutex> lock(error_mutex);if(!poison)poison=e;}
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> demand;
    std::array<std::deque<std::function<void()>>,3> speculative;
    bool has_speculative()const {for(const auto& q:speculative)if(!q.empty())return true;return false;}
    bool stopping=false;
    std::thread worker;

    Impl(std::vector<Record> input,Config c) : cfg(c),pool(c.slots,c.recent_window,c.decay_interval,c.distinct_lfu),reader(c.io_depth) {
        if ((cfg.io_chunk_bytes && cfg.io_chunk_bytes%4096) || !cfg.staging_slots || !cfg.prefetch_batch || input.empty()) throw std::invalid_argument("empty records/staging/prefetch batch");
        for(auto& r:input) {
            if(r.id>=expert_count || !r.bytes || !records.emplace(r.id,r).second) throw std::invalid_argument("record id/bytes");
            const TensorView views[]={r.gate,r.gate_scale,r.up,r.up_scale,r.down,r.down_scale};
            for(auto v:views) if(v.offset>r.bytes || v.bytes>r.bytes-v.offset) throw std::invalid_argument("tensor outside record");
            slot_bytes=std::max(slot_bytes,(r.bytes+4095)/4096*4096);
        }
        if(!std::isfinite(cfg.refill_gbps) || cfg.refill_gbps<=0)throw std::invalid_argument("refill rate");
        if(cfg.host_budget_bytes/slot_bytes)host_pool=std::make_unique<SlotPool>(cfg.host_budget_bytes/slot_bytes,cfg.host_recent_window,cfg.host_decay_interval,cfg.distinct_lfu);
        policy_options={flag("DS41_MOE_DECAY256"),flag("DS41_MOE_DRIFT_0_50"),flag("DS41_MOE_FAST_MINIMA")};
        // Opt-in chunk-level NVMe->H2D pipelining: upload each O_DIRECT chunk as it lands; publish the slot
        // (readiness/events/notify) only after the record's last chunk. Same bytes, earlier copies.
        chunk_upload=flag("DS41_MOE_CHUNK_UPLOAD");
        if(const char* kb=std::getenv("DS41_MOE_IO_CHUNK_KB")) {
            size_t bytes=size_t(std::stoull(kb))*1024;
            if(!bytes || bytes%4096)throw std::invalid_argument("DS41_MOE_IO_CHUNK_KB must be a positive multiple of 4");
            cfg.io_chunk_bytes=bytes;
        }
        if(chunk_upload && !cfg.io_chunk_bytes)cfg.io_chunk_bytes=size_t(4)<<20;
        demote=flag("DS41_MOE_DEMOTE");demote_eager=flag("DS41_MOE_DEMOTE_EAGER");
        if(const char* r=std::getenv("DS41_MOE_DEMOTE_RESERVE"))demote_reserve=std::stoul(r);
        if(demote && (!host_pool || !cfg.dynamic_host || !demote_reserve || demote_reserve>=cfg.slots))
            throw std::invalid_argument("DS41_MOE_DEMOTE requires a dynamic host tier and 0 < reserve < slots");
        if(demote){pool.peer(host_pool.get(),false);host_pool->peer(&pool,true);host_pool->exclusive(true);}
        if(policy_options.drift_0_50 && !flag("DS41_MOE_TOKEN_MAJOR_WIDTH1"))
            throw std::invalid_argument("drift requires explicit DS41_MOE_TOKEN_MAJOR_WIDTH1=1 contract");
        apply_options();
        if(const char* path=std::getenv("DS41_MOE_POLICY_AUDIT")) {
            policy_audit.open(path);if(!policy_audit)throw std::runtime_error("policy audit open");
        }
        struct Guard { Impl* p;bool active=true;~Guard(){if(active)p->clear_device();} } guard{this};
        ck(cudaGetDevice(&device));
        ck(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
        ck(cudaStreamCreateWithFlags(&mapping,cudaStreamNonBlocking));
        if(demote)ck(cudaStreamCreateWithFlags(&demote_stream,cudaStreamNonBlocking));
        ck(cudaEventCreateWithFlags(&speculative_done,cudaEventDisableTiming));
        ck(cudaEventCreateWithFlags(&copy_idle,cudaEventDisableTiming));
        ck(cudaMalloc(&vram,slot_bytes*cfg.slots));
        if(flag("DS41_MOE_GDS")) {
            unsigned threads=8;if(const char* t=std::getenv("DS41_MOE_GDS_THREADS"))threads=unsigned(std::stoul(t));
            gds=std::make_unique<GdsReader>(vram,slot_bytes*cfg.slots,cfg.io_chunk_bytes?cfg.io_chunk_bytes:size_t(2)<<20,threads);
        }
        ck(cudaMalloc(&table,expert_count*sizeof(DeviceExpert)));
        ck(cudaMemset(table,0,expert_count*sizeof(DeviceExpert)));
        ck(cudaMalloc(&ready,cfg.slots*sizeof(uint32_t)));ck(cudaMemset(ready,0,cfg.slots*sizeof(uint32_t)));
        slot_copy.resize(cfg.slots,nullptr);slot_mapping.resize(cfg.slots,nullptr);known_ready.resize(cfg.slots,false);
        for(size_t s=0;s<cfg.slots;++s){ck(cudaEventCreateWithFlags(&slot_copy[s],cudaEventDisableTiming));ck(cudaEventCreateWithFlags(&slot_mapping[s],cudaEventDisableTiming));}
        staging=std::make_unique<HostBuffer>(slot_bytes*cfg.staging_slots,true);
        staging_copy.resize(cfg.staging_slots,nullptr);
        for(auto& e:staging_copy)ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
        entries=std::make_unique<HostBuffer>(expert_count*sizeof(DeviceExpert),true);
        generations=std::make_unique<HostBuffer>(cfg.slots*sizeof(uint32_t),true);
        worker=std::thread([this]{
            cudaError_t set_device=cudaSetDevice(device);
            if(set_device!=cudaSuccess) fail(std::make_exception_ptr(std::runtime_error(cudaGetErrorString(set_device))));
            for(;;){
                std::function<void()> f;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock,[this]{return stopping || !demand.empty() || has_speculative();});
                    if(stopping && demand.empty() && !has_speculative()) break;
                    // Bound speculative DMA, not just CPU admission. Without
                    // this, RAM forecasts can enqueue many copies before the
                    // next demand arrives, defeating queue priorities.
                    if(demand.empty() && speculative_pending){
                        auto rc=cudaEventQuery(speculative_done);
                        if(rc==cudaErrorNotReady){
                            cv.wait_for(lock,std::chrono::microseconds(25),[this]{return stopping || !demand.empty();});
                            continue; // a demand wakes immediately and overtakes
                        }
                        if(rc!=cudaSuccess)fail(std::make_exception_ptr(std::runtime_error(cudaGetErrorString(rc))));
                        speculative_pending=false;
                    }
                    if(!demand.empty()){f=std::move(demand.front());demand.pop_front();}
                    else for(int p=2;p>=0;--p)if(!speculative[p].empty()){f=std::move(speculative[p].front());speculative[p].pop_front();break;}
                }
                f();
            }
        });
        guard.active=false;
    }
    ~Impl() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}cv.notify_one();
        if(worker.joinable())worker.join();
        // The caller must retire graph consumers before destruction. Waiting
        // here also prevents unregistering host sources still used by DMA.
        for(auto& r:retired){cudaEventSynchronize(r.event);cudaEventDestroy(r.event);}
        if(!live.empty() || error())cudaDeviceSynchronize();
        for(auto& r:host_retired){cudaEventSynchronize(r.event);cudaEventDestroy(r.event);}
        for(auto& d:draining){cudaEventSynchronize(d.event);cudaEventDestroy(d.event);}
        clear_device();
    }
    void clear_device() noexcept {
        gds.reset();
        if(copy)cudaStreamSynchronize(copy);
        if(mapping)cudaStreamSynchronize(mapping);
        if(demote_stream){cudaStreamSynchronize(demote_stream);cudaStreamDestroy(demote_stream);demote_stream=nullptr;}
        if(speculative_done)cudaEventDestroy(speculative_done);
        if(copy_idle)cudaEventDestroy(copy_idle);
        copy_idle=nullptr;
        speculative_done=nullptr;
        for(auto e:staging_copy)if(e)cudaEventDestroy(e);
        staging_copy.clear();
        for(auto e:slot_copy)if(e)cudaEventDestroy(e);
        for(auto e:slot_mapping)if(e)cudaEventDestroy(e);
        slot_copy.clear();slot_mapping.clear();
        if(vram)cudaFree(vram);
        if(table)cudaFree(table);
        if(ready)cudaFree(ready);
        if(copy)cudaStreamDestroy(copy);
        if(mapping)cudaStreamDestroy(mapping);
        vram=nullptr;table=nullptr;ready=nullptr;copy=nullptr;mapping=nullptr;
    }
    template<class F> auto submit(F f,int priority=-1) {
        using R=decltype(f());
        auto task=std::make_shared<std::packaged_task<R()>>([this,f=std::move(f),priority]() mutable -> R {
            auto e=error();if(e && priority<0)std::rethrow_exception(e);
            if(!e)collect(false);
            return f();
        });
        auto future=task->get_future();
        {std::lock_guard<std::mutex> lock(mutex);if(stopping)throw std::runtime_error("store shutting down");(priority<0?demand:speculative.at(priority)).emplace_back([task]{(*task)();});}
        cv.notify_one();return future;
    }
    void collect(bool wait) {
        for(auto it=draining.begin();it!=draining.end();) {
            auto rc=cudaEventQuery(it->event);if(rc==cudaErrorNotReady){++it;continue;}ck(rc);
            pool.drained(it->slot);if(!it->held.empty())host_pool->release(it->held);
            auto d=demoting.find(it->id);if(d!=demoting.end() && d->second==it->event)demoting.erase(d);
            ck(cudaEventDestroy(it->event));it=draining.erase(it);
        }
        for(auto it=host_retired.begin();it!=host_retired.end();){auto rc=cudaEventQuery(it->event);if(rc==cudaErrorNotReady){++it;continue;}ck(rc);ck(cudaEventDestroy(it->event));it=host_retired.erase(it);}
        for(auto it=retired.begin();it!=retired.end();) {
            auto rc=wait?cudaEventSynchronize(it->event):cudaEventQuery(it->event);
            if(rc==cudaErrorNotReady){++it;continue;}ck(rc);
            if(it->consumer)for(auto a:it->allocations)known_ready[a.slot]=true;
            pool.release(it->allocations);ck(cudaEventDestroy(it->event));it=retired.erase(it);
            if(wait)break;
        }
    }
    Lease load(const std::vector<ExpertId>& ids,cudaStream_t consumer,bool is_prefetch,const ReadyCallback& callback={},const ReadyBatchCallback& batch_callback={},bool allow_host_admission=true,bool drift_demand=true) {
        auto start=Clock::now();
        for(auto id:ids) if(!records.count(id))throw std::out_of_range("expert has no storage record");
        // pin_vram() uses load(false, allow_host_admission=false) to retain its
        // legacy accounting/startup behavior. It is not a processed target token.
        const bool policy_demand=!is_prefetch && allow_host_admission;
        if(drift && policy_demand && drift_demand)drift->before(ids);
        std::unique_ptr<Stats> audit_before;
        if(policy_audit.is_open()){audit_ram.clear();audit_before=std::make_unique<Stats>(counters);}
        std::vector<SlotPool::Allocation> allocations;
        for(;;)try {allocations=pool.acquire(ids,is_prefetch);break;}
        catch(const std::runtime_error&) {
            if(retired.empty())throw;
            collect(true);
        }
        if(drift && policy_demand && drift_demand)drift->after(ids);
        std::vector<SlotPool::Allocation> disk;
        std::vector<std::shared_ptr<const void>> host_holds;
        if(host_pool){
            if(!is_prefetch){std::vector<ExpertId> misses;for(auto a:allocations)if(!a.hit || demote)misses.push_back(a.id);host_pool->touch(misses);}
            // Protect every host hit before admitting any cold record.
            for(auto a:allocations)if(!a.hit && host.count(a.id))host_holds.push_back(host_pool->retain(a.id));
        }
        auto* views=static_cast<DeviceExpert*>(entries->data());
        auto* gens=static_cast<uint32_t*>(generations->data());
        try {
            for(auto a:allocations) {
                auto& layer=counters.layers[expert_layer(a.id)];
                if(!is_prefetch){++counters.requests;++layer.requests;}
                else ++counters.prefetch_requests;
                if(a.hit){if(!is_prefetch){++counters.hits;++layer.hits;++counters.vram.hits;if(prefetched.erase(a.id))++counters.prefetch_used;}
                    continue;}
                ++counters.vram.promotions;if(!is_prefetch)++counters.vram.misses;
                if(a.evicted>=0){++counters.evictions;++counters.vram.evictions;++counters.direct_evictions;if(prefetched.erase(ExpertId(a.evicted))){++counters.prefetch_wasted;counters.vram.wasted_bytes+=records.at(ExpertId(a.evicted)).bytes;}}
                auto& r=records.at(a.id);
                known_ready[a.slot]=false;
                uint32_t generation=++gens[a.slot];if(!generation)generation=++gens[a.slot];
                auto* base=vram+a.slot*slot_bytes;
                auto& v=views[a.id];
                v={base+r.gate.offset,base+r.gate_scale.offset,base+r.up.offset,base+r.up_scale.offset,base+r.down.offset,base+r.down_scale.offset,
                   cfg.ready_flags?ready+a.slot:nullptr,generation,int32_t(a.slot)};
                // Publish maps independently. Event-mode consumers wait on copy,
                // flag-mode consumers wait on mapping and then poll DMA flags.
                ck(cudaMemcpyAsync(table+a.id,views+a.id,sizeof(DeviceExpert),cudaMemcpyHostToDevice,mapping));
                ck(cudaEventRecord(slot_mapping[a.slot],mapping));
                auto h=host.find(a.id);
                if(h==host.end()){disk.push_back(a);if(!is_prefetch){++counters.nvme_misses;++layer.nvme_misses;++counters.ram.misses;}}
                else if(!is_prefetch){++counters.host_hits;++layer.host_hits;++counters.ram.hits;host_prefetched.erase(a.id);}
                if(is_prefetch)prefetched.insert(a.id);
            }
            // Submit resident work first so a cold expert cannot head-of-line
            // block cache hits. Each miss is published independently afterwards.
            bool streaming=bool(callback)||bool(batch_callback);
            auto notify=[&](const SlotPool::Allocation& a){order(a,consumer);if(callback)callback(a.id,consumer);if(batch_callback)batch_callback({a.id},consumer);};
            if(batch_callback){std::vector<ExpertId> hits;for(auto a:allocations)if(a.hit){order(a,consumer);hits.push_back(a.id);}if(!hits.empty())batch_callback(hits,consumer);}
            else if(callback)for(auto a:allocations)if(a.hit)notify(a);
            for(auto a:allocations)if(!a.hit) {
                auto h=host.find(a.id);if(h==host.end())continue;
                if(auto d=demoting.find(a.id);d!=demoting.end())ck(cudaStreamWaitEvent(copy,d->second,0));
                upload(a,static_cast<char*>(host_arena->data())+h->second,records.at(a.id).bytes,is_prefetch);
                if(streaming)notify(a);
            }
            if(gds && batch_callback && !is_prefetch && !disk.empty()) {
                std::vector<const Record*> direct;std::vector<size_t> offsets;
                for(auto a:disk) {
                    auto& r=records.at(a.id);direct.push_back(&r);offsets.push_back(a.slot*slot_bytes);
                    for(auto& e:r.extents){counters.nvme_bytes+=e.bytes;counters.layers[expert_layer(a.id)].nvme_bytes+=e.bytes;}
                }
                auto io_start=Clock::now();
                gds->read(direct,offsets,[&](size_t j){
                    // Payload already landed by DMA; publish on the (otherwise tiny) mapping stream so the
                    // consumer does not queue behind unrelated host uploads.
                    auto a=disk[j];
                    if(cfg.ready_flags)ck(cudaMemcpyAsync(ready+a.slot,static_cast<uint32_t*>(generations->data())+a.slot,sizeof(uint32_t),cudaMemcpyHostToDevice,mapping));
                    ck(cudaEventRecord(slot_copy[a.slot],mapping));
                    if(streaming)notify(a);
                });
                counters.io_ns+=ns(Clock::now()-io_start);
                disk.clear();
            }
            std::vector<const Record*> refs;std::vector<void*> ring,fixed;
            for(auto a:disk) {
                auto& r=records.at(a.id);refs.push_back(&r);
                void* destination=nullptr;
                if(cfg.dynamic_host && allow_host_admission && !is_prefetch && host_pool && !((demote || gds) && batch_callback)){
                    auto slot=admit_host(a.id,false);
                    if(slot>=0){destination=static_cast<char*>(host_arena->data())+size_t(slot)*slot_bytes;host_holds.push_back(host_pool->retain(a.id));}
                }
                fixed.push_back(destination);
                for(auto& e:r.extents){counters.nvme_bytes+=e.bytes;counters.layers[expert_layer(a.id)].nvme_bytes+=e.bytes;if(is_prefetch)counters.prefetch_nvme_bytes+=e.bytes;}
            }
            for(size_t j=0;j<std::min(cfg.staging_slots,disk.size());++j)ring.push_back(static_cast<char*>(staging->data())+j*slot_bytes);
            auto io_start=Clock::now();
            auto record_ready=[&](size_t j,size_t b){
                auto a=disk[j];upload(a,fixed[j]?fixed[j]:ring[b],refs[j]->bytes,is_prefetch,!chunk_upload);
                if(!fixed[j])ck(cudaEventRecord(staging_copy[b],copy));
                if(streaming)notify(a);
            };
            auto ring_recycle=[&](size_t b){auto t=Clock::now();ck(cudaEventSynchronize(staging_copy[b]));counters.staging_wait_ns+=ns(Clock::now()-t);};
            if(refs.empty()){}
            else if(chunk_upload) {
                reader.stream_mixed_chunked(refs,ring,fixed,record_ready,[&](size_t j,size_t b,uint64_t dst,uint32_t bytes){
                    const size_t record_bytes=refs[j]->bytes;
                    if(dst>=record_bytes)return;
                    const size_t n=std::min<size_t>(bytes,record_bytes-dst);
                    char* src=static_cast<char*>(fixed[j]?fixed[j]:ring[b]);
                    ck(cudaMemcpyAsync(vram+disk[j].slot*slot_bytes+dst,src+dst,n,cudaMemcpyHostToDevice,copy));
                },ring_recycle,cfg.io_chunk_bytes);
            } else reader.stream_mixed(refs,ring,fixed,record_ready,ring_recycle,cfg.io_chunk_bytes);
            counters.io_ns+=ns(Clock::now()-io_start);
            // A later worker call may evict RAM while these async DMA reads
            // still execute. Retain all source slots through a copy event.
            if(!host_holds.empty()){cudaEvent_t done;ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaEventRecord(done,copy));host_retired.push_back({std::move(host_holds),done});}
            if(!is_prefetch && !streaming)for(auto a:allocations)order(a,consumer);
            if(demote && batch_callback && !is_prefetch)top_up(ids);
            // Keep prefetch leases until DMA finishes; demand ownership is
            // transferred to the caller and released only after its consumer.
            uint64_t id=++serial;
            if(is_prefetch){cudaEvent_t done;ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaEventRecord(done,copy));retired.push_back({allocations,done,false});ck(cudaEventRecord(speculative_done,copy));speculative_pending=true;}else live.emplace(id,allocations);
            auto elapsed=ns(Clock::now()-start);counters.need_host_ns+=elapsed;
            if(!is_prefetch){std::set<unsigned> layers;for(auto a:allocations)layers.insert(expert_layer(a.id));for(auto l:layers)counters.layers[l].stall_ns+=elapsed;}
            if(policy_audit.is_open() && policy_demand) {
                policy_audit<<audit_call++/40<<' '<<expert_layer(ids.front())<<" V";
                for(auto a:allocations)policy_audit<<' '<<a.id<<':'<<a.slot<<':'<<a.hit<<':'<<a.evicted;
                policy_audit<<" R";
                for(auto a:audit_ram)policy_audit<<' '<<a.id<<':'<<a.slot<<':'<<a.hit<<':'<<a.evicted;
                policy_audit<<" phase="<<(drift && drift->active())<<" age="<<(drift?drift->completed_tokens():0);
                const auto& b=*audit_before;
                policy_audit<<" counts="<<(counters.requests-b.requests)<<','<<(counters.hits-b.hits)<<','
                  <<(counters.host_hits-b.host_hits)<<','<<(counters.nvme_misses-b.nvme_misses)<<','
                  <<(counters.vram.promotions-b.vram.promotions)<<','<<(counters.vram.evictions-b.vram.evictions)<<','
                  <<(counters.ram.promotions-b.ram.promotions)<<','<<(counters.ram.evictions-b.ram.evictions)<<','
                  <<(counters.ram_admission_blocked-b.ram_admission_blocked)<<'\n';
                policy_audit.flush();
            }
            return {id,std::move(allocations),reinterpret_cast<uintptr_t>(this)};
        }catch(...){
            // A partially failed IO/copy may have overwritten a previous slot.
            // Poison rather than returning a plausible stale cache hit.
            fail(std::current_exception());ck(cudaStreamSynchronize(copy));throw;
        }
    }
    size_t warm_host(const std::vector<ExpertId>& ids,size_t budget){
        if(!cfg.dynamic_host || !host_pool)return 0;
        // Every batch admission stays held until its read completes, so none can evict another.
        budget=std::min(budget,cfg.host_budget_bytes/4);
        std::vector<const Record*> refs;std::vector<void*> fixed;std::set<int> slots;std::vector<SlotPool::Allocation> held;
        for(auto id:ids){
            const auto& record=records.at(id);
            if(host.count(id) || pool.residency(id)>=0)continue;
            size_t bytes=0;for(const auto& e:record.extents)bytes+=e.bytes;
            if(bytes>budget)break;
            int slot=admit_host(id,true,&held);
            if(slot<0)break;
            if(!slots.insert(slot).second){host_pool->release(held);auto e=std::make_exception_ptr(std::logic_error("warm_host reused a slot within one batch"));fail(e);std::rethrow_exception(e);}
            refs.push_back(&record);fixed.push_back(static_cast<char*>(host_arena->data())+size_t(slot)*slot_bytes);
            budget-=bytes;counters.refill_bytes+=bytes;counters.nvme_bytes+=bytes;counters.layers[expert_layer(id)].nvme_bytes+=bytes;
        }
        if(refs.empty()){if(!held.empty())host_pool->release(held);return 0;}
        try{reader.stream_mixed(refs,{},fixed,{},{},cfg.io_chunk_bytes);}
        catch(...){host_pool->release(held);fail(std::current_exception());throw;}
        host_pool->release(held);
        return refs.size();
    }
    void top_up(const std::vector<ExpertId>& ids) {
        std::unordered_set<ExpertId> wanted(ids.begin(),ids.end());
        // Copy-backs share host DRAM bandwidth with H2D demand: start them only after this
        // demand's uploads, i.e. in the compute window that follows (DS41_MOE_DEMOTE_EAGER=1 skips).
        bool waited=demote_eager;
        for(auto v:pool.reserve(demote_reserve,wanted)) {
            auto x=ExpertId(v.evicted);
            ++counters.evictions;++counters.vram.evictions;
            if(prefetched.erase(x)){++counters.prefetch_wasted;counters.vram.wasted_bytes+=records.at(x).bytes;}
            if(host.count(x)){++counters.demote_resident;pool.drained(v.slot);continue;}
            std::vector<SlotPool::Allocation> held;
            int hs=admit_host(x,false,&held);
            if(hs<0){pool.drained(v.slot);continue;}
            // The slot's last upload must land before its bytes are copied back.
            auto rc=cudaEventQuery(slot_copy[v.slot]);
            if(rc==cudaErrorNotReady)ck(cudaStreamWaitEvent(demote_stream,slot_copy[v.slot],0));else ck(rc);
            if(!waited){ck(cudaStreamWaitEvent(demote_stream,copy_idle,0));waited=true;}
            ck(cudaMemcpyAsync(static_cast<char*>(host_arena->data())+size_t(hs)*slot_bytes,vram+v.slot*slot_bytes,
                records.at(x).bytes,cudaMemcpyDeviceToHost,demote_stream));
            cudaEvent_t done;ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaEventRecord(done,demote_stream));
            demoting[x]=done;draining.push_back({v.slot,x,done,std::move(held)});
            ++counters.demotions;
        }
    }
    int admit_host(ExpertId id,bool speculative,std::vector<SlotPool::Allocation>* hold=nullptr){
        if(!host_pool)return -1;
        std::vector<SlotPool::Allocation> a;
        try{a=host_pool->acquire({id},true);}catch(const std::runtime_error&){++counters.ram_admission_blocked;return -1;}
        if(policy_audit.is_open())audit_ram.insert(audit_ram.end(),a.begin(),a.end());
        auto v=a.front();
        if(!v.hit){
            if(!host_arena)host_arena=std::make_shared<HostBuffer>(cfg.host_budget_bytes,true);
            if(v.evicted>=0){host.erase(ExpertId(v.evicted));++counters.ram.evictions;if(host_prefetched.erase(ExpertId(v.evicted)))counters.ram.wasted_bytes+=records.at(ExpertId(v.evicted)).bytes;}
            host[id]=v.slot*slot_bytes;++counters.ram.promotions;
            if(speculative)host_prefetched.insert(id);
        }
        if(hold)hold->insert(hold->end(),a.begin(),a.end());else host_pool->release(a);
        return int(v.slot);
    }
    void order(const SlotPool::Allocation& a,cudaStream_t consumer) {
        if(known_ready[a.slot])return;
        auto map=slot_mapping[a.slot];auto rc=cudaEventQuery(map);
        if(rc==cudaErrorNotReady)ck(cudaStreamWaitEvent(consumer,map,0));else ck(rc);
        if(!cfg.ready_flags){rc=cudaEventQuery(slot_copy[a.slot]);if(rc==cudaErrorNotReady)ck(cudaStreamWaitEvent(consumer,slot_copy[a.slot],0));else ck(rc);}
    }
    void upload(const SlotPool::Allocation& a,void* src,size_t bytes,bool is_prefetch,bool copy_bytes=true) {
        size_t chunk=cfg.h2d_chunk_bytes?cfg.h2d_chunk_bytes:bytes;
        if(copy_bytes)for(size_t off=0;off<bytes;off+=chunk)ck(cudaMemcpyAsync(vram+a.slot*slot_bytes+off,static_cast<char*>(src)+off,std::min(chunk,bytes-off),cudaMemcpyHostToDevice,copy));
        counters.h2d_bytes+=bytes;counters.layers[expert_layer(a.id)].h2d_bytes+=bytes;
        if(is_prefetch)counters.prefetch_h2d_bytes+=bytes;
        if(cfg.ready_flags)ck(cudaMemcpyAsync(ready+a.slot,static_cast<uint32_t*>(generations->data())+a.slot,sizeof(uint32_t),cudaMemcpyHostToDevice,copy));
        ck(cudaEventRecord(slot_copy[a.slot],copy));
        ck(cudaEventRecord(copy_idle,copy));
    }
};
OffloadStore::OffloadStore(std::vector<Record> r,Config c):impl_(new Impl(std::move(r),c)){}
OffloadStore::~OffloadStore()=default;
OffloadStore::Lease OffloadStore::need_now(std::vector<ExpertId> ids,cudaStream_t stream) {
    return impl_->submit([this,ids=std::move(ids),stream]{return impl_->load(ids,stream,false);}).get();
}
OffloadStore::Lease OffloadStore::need_streamed(std::vector<ExpertId> ids,cudaStream_t stream,ReadyCallback ready) {
    if(!ready)throw std::invalid_argument("empty ready callback");
    return impl_->submit([this,ids=std::move(ids),stream,ready=std::move(ready)]{return impl_->load(ids,stream,false,ready);}).get();
}
OffloadStore::Lease OffloadStore::need_streamed_batches(std::vector<ExpertId> ids,cudaStream_t stream,ReadyBatchCallback ready,bool drift_demand) {
    if(!ready)throw std::invalid_argument("empty ready batch callback");
    return impl_->submit([this,ids=std::move(ids),stream,ready=std::move(ready),drift_demand]{return impl_->load(ids,stream,false,{},ready,true,drift_demand);}).get();
}
void OffloadStore::stream_prefill(std::vector<ExpertId> ids,cudaStream_t stream,ReadyCallback ready,size_t window) {
    if(!ready)throw std::invalid_argument("empty prefill callback");
    std::set<ExpertId> seen;std::vector<ExpertId> unique;
    for(auto id:ids)if(seen.insert(id).second)unique.push_back(id);
    if(!window)window=std::min(impl_->cfg.staging_slots,impl_->cfg.slots);
    if(!window || window>impl_->cfg.slots)throw std::invalid_argument("prefill ring window");
    for(size_t first=0;first<unique.size();first+=window){
        size_t end=std::min(first+window,unique.size());
        auto lease=need_streamed({unique.begin()+first,unique.begin()+end},stream,ready);
        retire(std::move(lease),stream);
    }
}
OffloadStore::PrefetchTicket OffloadStore::prefetch(std::vector<ExpertId> ids,PrefetchPriority priority){return prefetch_impl(std::move(ids),priority,false);}
OffloadStore::PrefetchTicket OffloadStore::prefetch_ram(std::vector<ExpertId> ids,PrefetchPriority priority){return prefetch_impl(std::move(ids),priority,true);}
OffloadStore::PrefetchTicket OffloadStore::prefetch_impl(std::vector<ExpertId> ids,PrefetchPriority priority,bool ram_only) {
    if(unsigned(priority)>2)throw std::invalid_argument("prefetch priority");
    struct State {
        std::promise<PrefetchResult> promise;
        PrefetchResult result;
        std::exception_ptr error;
        std::shared_ptr<std::atomic<bool>> cancelled=std::make_shared<std::atomic<bool>>(false);
    };
    check_error();
    auto state=std::make_shared<State>();state->result.requested=ids.size();
    PrefetchTicket ticket{++impl_->ticket_serial,state->promise.get_future().share(),state->cancelled};
    if(ids.empty()){state->promise.set_value(state->result);return ticket;}
    for(size_t first=0;first<ids.size();first+=impl_->cfg.prefetch_batch) {
        size_t end=std::min(ids.size(),first+impl_->cfg.prefetch_batch);bool last=end==ids.size();
        std::vector<ExpertId> chunk(ids.begin()+first,ids.begin()+end);
        impl_->submit([this,chunk=std::move(chunk),state,last,ram_only]()mutable{
            if(state->cancelled->load() || state->error){state->result.cancelled+=chunk.size();impl_->counters.prefetch_cancelled+=chunk.size();}
            else try{
                if(ram_only){auto end=std::remove_if(chunk.begin(),chunk.end(),[this](ExpertId id){impl_->records.at(id);return !impl_->host.count(id) && impl_->pool.residency(id)<0;});size_t n=chunk.end()-end;chunk.erase(end,chunk.end());state->result.cancelled+=n;impl_->counters.prefetch_cancelled+=n;}
                if(auto e=impl_->error())std::rethrow_exception(e);if(!chunk.empty())impl_->load(chunk,nullptr,true);state->result.submitted+=chunk.size();}catch(...){state->error=std::current_exception();}
            if(last){if(state->error)state->promise.set_exception(state->error);else state->promise.set_value(state->result);}
        },int(priority));
    }
    return ticket;
}
OffloadStore::PrefetchTicket OffloadStore::refill_host(std::vector<ExpertId> ids,uint64_t idle_ns,size_t byte_budget){
    check_error();
    struct State{std::promise<PrefetchResult> promise;PrefetchResult result;std::exception_ptr error;size_t budget;Clock::time_point deadline;std::shared_ptr<std::atomic<bool>> cancelled=std::make_shared<std::atomic<bool>>(false);};
    auto state=std::make_shared<State>();state->budget=byte_budget;state->result.requested=ids.size();
    state->deadline=Clock::now()+std::chrono::nanoseconds(std::min<uint64_t>(idle_ns,60'000'000'000ULL));
    PrefetchTicket ticket{++impl_->ticket_serial,state->promise.get_future().share(),state->cancelled};
    if(ids.empty()){state->promise.set_value(state->result);return ticket;}
    for(size_t j=0;j<ids.size();++j){auto id=ids[j];bool last=j+1==ids.size();
        impl_->submit([this,id,last,state]{
            bool skip=state->cancelled->load() || bool(state->error) || !impl_->cfg.dynamic_host || !impl_->host_pool;
            try{
                if(auto e=impl_->error())std::rethrow_exception(e);
                const auto& record=impl_->records.at(id);size_t bytes=0;for(const auto& e:record.extents)bytes+=e.bytes;
                auto now=Clock::now();auto cost=std::chrono::nanoseconds(uint64_t(std::ceil(bytes/impl_->cfg.refill_gbps))+250'000);
                {std::lock_guard<std::mutex> lock(impl_->mutex);skip=skip || !impl_->demand.empty();}
                auto rc=cudaEventQuery(impl_->copy_idle);if(rc!=cudaSuccess && rc!=cudaErrorNotReady)ck(rc);
                skip=skip || rc==cudaErrorNotReady || impl_->host.count(id) || bytes>state->budget || now<impl_->refill_next || now+cost>state->deadline;
                if(!skip){
                    int slot=impl_->admit_host(id,true);if(slot<0)skip=true;
                    else {
                        try{impl_->reader.read({&record},{static_cast<char*>(impl_->host_arena->data())+size_t(slot)*impl_->slot_bytes});}
                        catch(...){impl_->fail(std::current_exception());throw;}
                        state->budget-=bytes;impl_->counters.refill_bytes+=bytes;impl_->counters.nvme_bytes+=bytes;
                        impl_->counters.layers[expert_layer(id)].nvme_bytes+=bytes;
                        impl_->refill_next=now+cost;++state->result.submitted;
                    }
                }
            }catch(...){state->error=std::current_exception();skip=true;}
            if(skip){++state->result.cancelled;++impl_->counters.refill_skipped;}
            if(last){if(state->error)state->promise.set_exception(state->error);else state->promise.set_value(state->result);}
        },int(PrefetchPriority::Session));
    }
    return ticket;
}
size_t OffloadStore::warm_host(std::vector<ExpertId> ids,size_t byte_budget){
    check_error();
    return impl_->submit([this,ids=std::move(ids),byte_budget]{return impl_->warm_host(ids,byte_budget);}).get();
}
void OffloadStore::cancel(const PrefetchTicket& ticket)noexcept {if(ticket.cancelled)ticket.cancelled->store(true);}
std::future<void> OffloadStore::prefetch(std::vector<ExpertId> ids) {
    auto ticket=prefetch(std::move(ids),PrefetchPriority::NextLayer);
    return impl_->submit([ticket]{ticket.completion.get();},int(PrefetchPriority::NextLayer));
}
void OffloadStore::retire(Lease lease,cudaStream_t stream) {retire_async(std::move(lease),stream).get();}
std::future<void> OffloadStore::retire_async(Lease lease,cudaStream_t stream) {
    if(lease.owner!=reinterpret_cast<uintptr_t>(impl_.get()))throw std::logic_error("lease belongs to another store");
    // Record the event on the calling thread after graph submission, then
    // register it with the scheduler before accepting a later need call.
    cudaEvent_t done;ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaEventRecord(done,stream));
    auto accepted=std::make_shared<bool>(false);
    auto guard=std::shared_ptr<CUevent_st>(done,[accepted](cudaEvent_t e){if(!*accepted)cudaEventDestroy(e);});
    return impl_->submit([this,lease=std::move(lease),done,guard,accepted]{
        auto it=impl_->live.find(lease.serial);
        if(it==impl_->live.end()){throw std::logic_error("lease already retired or from another store");}
        impl_->retired.push_back({std::move(it->second),done,true});*accepted=true;impl_->live.erase(it);
    });
}
void OffloadStore::pin_vram(std::vector<ExpertId> ids) {
    impl_->submit([this,ids=std::move(ids)]{
        auto lease=impl_->load(ids,nullptr,false,{}, {},false);ck(cudaStreamSynchronize(impl_->copy));
        ck(cudaStreamSynchronize(impl_->mapping));
        for(auto a:lease.allocations){impl_->pool.pin(a.id);impl_->known_ready[a.slot]=true;}
        impl_->pool.release(lease.allocations);impl_->live.erase(lease.serial);
    }).get();
}
void OffloadStore::pin_host(std::vector<ExpertId> ids) {
    impl_->submit([this,ids=std::move(ids)]{
        std::set<ExpertId> distinct(ids.begin(),ids.end());
        for(auto id:distinct)impl_->records.at(id);
        std::vector<ExpertId> pinned;
        for(auto id:distinct){
            bool hit=impl_->host.count(id);int slot=impl_->admit_host(id,false);
            if(slot<0)throw std::runtime_error("pinned host budget exceeded");
            impl_->host_pool->pin(id);pinned.push_back(id);
            if(!hit)try{impl_->reader.read({&impl_->records.at(id)},{static_cast<char*>(impl_->host_arena->data())+size_t(slot)*impl_->slot_bytes});}catch(...){impl_->fail(std::current_exception());throw;}
        }
    }).get();
}
OffloadStore::HostLease OffloadStore::borrow_host(std::vector<ExpertId> ids) {
    return impl_->submit([this,ids=std::move(ids)]{
        HostLease lease;
        std::set<ExpertId> seen;
        for(auto id:ids) {
            if(!seen.insert(id).second)continue;
            const auto& record=impl_->records.at(id); // validate nonresident IDs too
            HostView view;view.id=id;view.record=record;
            auto h=impl_->host.find(id);
            if(h!=impl_->host.end()) {
                if(auto d=impl_->demoting.find(id);d!=impl_->demoting.end())ck(cudaEventSynchronize(d->second));
                lease.backing_=impl_->host_arena;
                lease.retention_.push_back(impl_->host_pool->retain(id));
                view.data=static_cast<const uint8_t*>(lease.backing_->data())+h->second;
                view.device_data=static_cast<const uint8_t*>(lease.backing_->device_data())+h->second;
            }
            lease.views.push_back(std::move(view));
        }
        return lease;
    }).get();
}
void OffloadStore::cache_policy(CachePolicy options) {
    impl_->submit([this,options]{
        if(impl_->placed || !impl_->live.empty() || !impl_->retired.empty())throw std::logic_error("cache policy requires pre-placement store");
        impl_->policy_options=options;impl_->apply_options();
    }).get();
}
void OffloadStore::place(const Placement& input) {
    Placement p=input;
    if(impl_->policy_options.decay256 || impl_->policy_options.drift_0_50)p.decay_interval=256;
    // Placement is a startup operation; fixed allocations cannot grow here.
    impl_->submit([this,p]{
        if(!impl_->live.empty() || !impl_->retired.empty())throw std::logic_error("placement requires quiescent startup store");
        if(p.vram_pins.size()>impl_->cfg.slots || p.host_residents.size()>impl_->cfg.host_budget_bytes/impl_->slot_bytes)
            throw std::runtime_error("placement exceeds configured tier capacity");
        std::set<ExpertId> selected;std::array<size_t,43> pins{};size_t planned=p.vram_pins.size();
        for(auto n:p.dynamic_slots_per_layer)planned+=n;
        if(planned>impl_->cfg.slots)throw std::runtime_error("placement dynamic/pin slot budget exceeded");
        for(auto id:p.vram_pins){impl_->records.at(id);if(!selected.insert(id).second)throw std::invalid_argument("duplicate placement ID");++pins[expert_layer(id)];}
        for(auto id:p.host_residents){impl_->records.at(id);if(!selected.insert(id).second)throw std::invalid_argument("overlapping placement ID");}
        for(unsigned l=0;l<43;++l)if(p.dynamic_slots_per_layer[l]>(l<40?384u:128u)-pins[l])throw std::invalid_argument("layer dynamic budget exceeds experts");
        if(impl_->placed && impl_->policy_options.drift_0_50)throw std::logic_error("drift placement already initialized");
        if(impl_->policy_options.drift_0_50 && (!impl_->host_pool || !impl_->cfg.dynamic_host))
            throw std::invalid_argument("drift requires dynamic host pool");
        impl_->pool.policy(p);impl_->apply_options();
        if(impl_->policy_options.drift_0_50) {
            impl_->drift=std::make_unique<DriftPolicy>(p,impl_->pool,*impl_->host_pool);
        }
        impl_->placed=true;
        if(impl_->host_pool && p.hard_layer_budgets){Placement h;h.hard_layer_budgets=true;h.dynamic_slots_per_layer=p.host_slots_per_layer;h.recent_window=impl_->cfg.host_recent_window;h.decay_interval=impl_->cfg.host_decay_interval;h.distinct_lfu=impl_->cfg.distinct_lfu;impl_->host_pool->policy(h);}
    }).get();
    pin_host(p.host_residents);
    if(impl_->cfg.dynamic_host)impl_->submit([this,p]{for(auto id:p.host_residents)impl_->host_pool->unpin(id);}).get();
    pin_vram(p.vram_pins);
}
OffloadStore::Lease OffloadStore::borrow_vram(std::vector<ExpertId> ids,cudaStream_t consumer){
    return impl_->submit([this,ids=std::move(ids),consumer]{
        std::vector<ExpertId> hits;for(auto id:ids){impl_->records.at(id);if(impl_->pool.residency(id)>=0)hits.push_back(id);}
        auto allocations=impl_->pool.borrow(hits);
        try{for(auto a:allocations)impl_->order(a,consumer);auto serial=++impl_->serial;impl_->live.emplace(serial,allocations);return Lease{serial,std::move(allocations),reinterpret_cast<uintptr_t>(impl_.get())};}
        catch(...){impl_->pool.release(allocations);throw;}
    }).get();
}
void OffloadStore::reset_history(){impl_->submit([this]{impl_->pool.reset_history();if(impl_->host_pool)impl_->host_pool->reset_history();}).get();}
void OffloadStore::report_error(std::exception_ptr error)noexcept{impl_->fail(error);}
void OffloadStore::check_error(){if(auto e=impl_->error())std::rethrow_exception(e);}
std::vector<uint8_t> OffloadStore::residency_snapshot(){return impl_->submit([this]{std::vector<uint8_t> result(expert_count);for(auto [id,record]:impl_->records)result[id]=(impl_->pool.residency(id)>=0?1:0)|(impl_->host.count(id)?2:0);return result;}).get();}
const DeviceExpert* OffloadStore::table()const{return impl_->table;}
int OffloadStore::residency(ExpertId id){return impl_->submit([this,id]{return impl_->pool.residency(id);}).get();}
Stats OffloadStore::stats(){return impl_->submit([this]{return impl_->counters;}).get();}
} // namespace ds41
