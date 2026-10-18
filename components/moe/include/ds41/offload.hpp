#pragma once
#include "ds41/store.hpp"
#include <cuda_runtime.h>
#include <future>
#include <atomic>

namespace ds41 {
class OffloadStore {
public:
    static constexpr unsigned api_version=4;
    struct Config {
        size_t slots = 64, staging_slots = 8;
        size_t prefetch_batch = 1; // demand can overtake queued speculative chunks
        size_t host_budget_bytes = 0;
        unsigned io_depth = 64;
        bool ready_flags = false; // default event ordering, same device table ABI
        unsigned recent_window = 6, decay_interval = 32;
        size_t io_chunk_bytes = 0, h2d_chunk_bytes = 0; // 0: complete native extent/record
        bool distinct_lfu = true;
        bool dynamic_host = false; // placement host IDs become evictable seeds
        unsigned host_recent_window = 6, host_decay_interval = 32;
        // Explicit idle-window refills only; conservative vs measured 14.17 GB/s.
        double refill_gbps = 8.0;
    };
    struct CachePolicy {
        bool decay256=false, drift_0_50=false, fast_minima=false;
    };
    // Before place(), only. Existing Config and api_version=4 are unchanged.
    void cache_policy(CachePolicy options);
    struct Lease {
        uint64_t serial = 0;
        std::vector<SlotPool::Allocation> allocations;
        uintptr_t owner = 0;
    };
    // Callbacks run on the store worker and may only enqueue CUDA work. Use
    // logical expert IDs to select output buffers; never reduce in callback order.
    using ReadyCallback = std::function<void(ExpertId,cudaStream_t)>;
    using ReadyBatchCallback = std::function<void(const std::vector<ExpertId>&,cudaStream_t)>;
    enum class PrefetchPriority : unsigned { Session=0, Draft=1, NextLayer=2 };
    struct PrefetchResult { uint64_t requested=0, submitted=0, cancelled=0; };
    struct PrefetchTicket {
        uint64_t id=0;
        std::shared_future<PrefetchResult> completion;
        std::shared_ptr<std::atomic<bool>> cancelled;
    };
    struct HostView {
        ExpertId id=0;
        Record record; // original read plan and six native tensor offsets
        const uint8_t* data=nullptr; // null means not host-resident
        const uint8_t* device_data=nullptr; // explicit mapped CUDA alias
    };
    struct HostLease {
        std::vector<HostView> views;
    private:
        // Shared slot tokens prevent overwrite until the last lease copy dies.
        std::shared_ptr<const HostBuffer> backing_;
        std::vector<std::shared_ptr<const void>> retention_;
        friend class OffloadStore;
    };
    OffloadStore(std::vector<Record> records, Config config);
    ~OffloadStore();
    OffloadStore(const OffloadStore&) = delete;
    OffloadStore& operator=(const OffloadStore&) = delete;
    Lease need_now(std::vector<ExpertId> ids, cudaStream_t consumer);
    Lease need_streamed(std::vector<ExpertId> ids,cudaStream_t consumer,ReadyCallback ready);
    // Coalesces resident hits into one callback; cold completions are singletons.
    // drift=false: a demand outside the width-one token-major traversal (e.g. a pipelined verify's
    // second lane); it updates LFU/tiers but not the drift controller.
    Lease need_streamed_batches(std::vector<ExpertId> ids,cudaStream_t consumer,ReadyBatchCallback ready,bool drift=true);
    // Fixed-capacity prefill ring. Each callback must enqueue all consumers of
    // that expert. Scratch/output storage remains caller-owned until stream end.
    void stream_prefill(std::vector<ExpertId> ids,cudaStream_t consumer,ReadyCallback ready,size_t window=0);
    std::future<void> prefetch(std::vector<ExpertId> ids);
    PrefetchTicket prefetch(std::vector<ExpertId> ids,PrefetchPriority priority);
    PrefetchTicket prefetch_ram(std::vector<ExpertId> ids,PrefetchPriority priority=PrefetchPriority::NextLayer);
    // Caller grants an actual link-idle window. One record per admission,
    // no H2D, no policy credit; skip if copies/demand or the deadline intervene.
    PrefetchTicket refill_host(std::vector<ExpertId> ids,uint64_t idle_ns,size_t byte_budget);
    // Bulk NVMe->RAM admission for a caller-granted idle point (e.g. after prompt prefill): records
    // already in VRAM or RAM are skipped; the rest are admitted in order until the byte budget, a
    // blocked admission or a slot reused within the batch, then read in parallel. No H2D, no policy
    // credit. Runs as one demand-priority worker task; returns the number of records read.
    size_t warm_host(std::vector<ExpertId> ids,size_t byte_budget);
    void cancel(const PrefetchTicket& ticket) noexcept;
    void retire(Lease lease, cudaStream_t consumer);
    std::future<void> retire_async(Lease lease,cudaStream_t consumer);
    void pin_vram(std::vector<ExpertId> ids);
    void pin_host(std::vector<ExpertId> ids);
    // No tier allocation, IO, DMA, VRAM admission, policy updates or eviction.
    // Keep the lease until CPU work and every asynchronous reader have finished.
    HostLease borrow_host(std::vector<ExpertId> ids);
    Lease borrow_vram(std::vector<ExpertId> ids,cudaStream_t consumer);
    void reset_history(); // preserve cache contents, clear conversation LFU history
    void report_error(std::exception_ptr error) noexcept;
    void check_error();
    // bit 0: VRAM, bit 1: RAM. Advisory only; admission rechecks on the worker.
    std::vector<uint8_t> residency_snapshot();
    void place(const Placement& placement); // startup, before graph/consumer submission
    const DeviceExpert* table() const;
    int residency(ExpertId id);
    Stats stats();
private:
    PrefetchTicket prefetch_impl(std::vector<ExpertId> ids,PrefetchPriority priority,bool ram_only);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ds41
