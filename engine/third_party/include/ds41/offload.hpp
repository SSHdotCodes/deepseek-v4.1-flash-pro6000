#pragma once
#include "ds41/store.hpp"
#include <cuda_runtime.h>
#include <future>

namespace ds41 {
class OffloadStore {
public:
    struct Config {
        size_t slots = 64, staging_slots = 8;
        size_t prefetch_batch = 2; // demand can overtake queued speculative chunks
        size_t host_budget_bytes = 0;
        unsigned io_depth = 64;
        bool ready_flags = false; // default event ordering, same device table ABI
        unsigned recent_window = 6, decay_interval = 32;
    };
    struct Lease {
        uint64_t serial = 0;
        std::vector<SlotPool::Allocation> allocations;
        uintptr_t owner = 0;
    };
    OffloadStore(std::vector<Record> records, Config config);
    ~OffloadStore();
    OffloadStore(const OffloadStore&) = delete;
    OffloadStore& operator=(const OffloadStore&) = delete;
    Lease need_now(std::vector<ExpertId> ids, cudaStream_t consumer);
    std::future<void> prefetch(std::vector<ExpertId> ids);
    void retire(Lease lease, cudaStream_t consumer);
    void pin_vram(std::vector<ExpertId> ids);
    void pin_host(std::vector<ExpertId> ids);
    const DeviceExpert* table() const;
    int residency(ExpertId id);
    Stats stats();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ds41
