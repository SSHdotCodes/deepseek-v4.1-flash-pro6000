#pragma once
#include <cstddef>
#include <atomic>
#include <deque>
#include <unordered_set>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ds41 {
// Flatten target layers 0..39, then DSpark layers 40..42 (128 experts each).
using ExpertId = uint32_t;
constexpr ExpertId expert_id(unsigned layer, unsigned expert) {
    return layer < 40 ? layer * 384 + expert : 40 * 384 + (layer - 40) * 128 + expert;
}
constexpr size_t expert_count = 40 * 384 + 3 * 128;
constexpr unsigned expert_layer(ExpertId e) { return e<40*384?e/384:40+(e-40*384)/128; }
struct TensorView { uint64_t offset = 0, bytes = 0; };
// A record is already kernel ready. Offsets are relative to its host/device base.
// Native shards use two aligned extents; banks use one. Neither changes bytes.
struct Extent {
    std::string path;
    uint64_t file_offset = 0, bytes = 0, dst_offset = 0;
    // Last native tensors can end at a nonaligned EOF. O_DIRECT length remains
    // aligned, but a short final read is valid if ALL payload bytes arrived.
    uint64_t required_bytes = 0; // zero means bytes
    uint64_t expected_file_bytes = 0; // reject a still-downloading shard
};
struct Record {
    ExpertId id = 0;
    size_t bytes = 0;
    TensorView gate, gate_scale, up, up_scale, down, down_scale;
    std::vector<Extent> extents;
};
struct Placement {
    std::vector<ExpertId> vram_pins,host_residents;
    std::array<size_t,43> dynamic_slots_per_layer{}; // soft targets, pinned slots excluded
    unsigned recent_window=6,decay_interval=32;
    bool distinct_lfu=true;
    bool hard_layer_budgets=false;
    std::array<size_t,43> host_slots_per_layer{}; // total host capacity, including seeds
};
Placement load_placement(const std::string& path);
std::vector<Record> load_records(const std::string& manifest_tsv);
struct DeviceExpert {
    const uint8_t *gate = nullptr, *gate_scale = nullptr;
    const uint8_t *up = nullptr, *up_scale = nullptr;
    const uint8_t *down = nullptr, *down_scale = nullptr;
    const uint32_t *ready = nullptr;
    uint32_t generation = 0;
    int32_t slot = -1;
};
#define DS41_MOE_STATS_DEMOTE 1
struct Stats {
    uint64_t requests = 0, hits = 0, host_hits = 0, nvme_misses = 0;
    uint64_t h2d_bytes = 0, nvme_bytes = 0, evictions = 0, prefetch_used = 0;
    uint64_t route_wait_ns = 0, need_host_ns = 0, io_ns = 0;
    struct Layer {
        uint64_t requests=0, hits=0, host_hits=0, nvme_misses=0;
        uint64_t h2d_bytes=0, nvme_bytes=0, stall_ns=0;
    };
    std::array<Layer,43> layers{};
    uint64_t prefetch_requests=0, prefetch_cancelled=0, prefetch_wasted=0;
    uint64_t prefetch_h2d_bytes=0, prefetch_nvme_bytes=0, staging_wait_ns=0;
    struct Tier {
        uint64_t hits=0, misses=0, promotions=0, evictions=0, wasted_bytes=0;
    };
    Tier vram, ram;
    uint64_t ram_admission_blocked=0, refill_bytes=0, refill_skipped=0;
    // DS41_MOE_DEMOTE: VRAM victims copied back to RAM, victims already in RAM, demand evictions
    // that took a victim's slot directly (reserve empty, no demotion).
    uint64_t demotions=0, demote_resident=0, direct_evictions=0;
};

// CPU-only, deterministic allocation policy. A lease prevents eviction until
// compute has finished, not merely until the transfer completed.
class SlotPool {
public:
    explicit SlotPool(size_t slots, unsigned recent_window = 6, unsigned decay_interval = 32, bool distinct_lfu = true);
    void policy(const Placement& placement);
    void touch(const std::vector<ExpertId>& ids);
    void reset_history(); // residency, pins and drift session state survive
    // Opt-in metadata only. All calls except token destruction are worker-serialized.
    void fast_minima(bool enabled);
    void decay_interval(unsigned interval);
    void peer(const SlotPool* pool, bool host);
    void switch_target_layer(const Placement& placement, unsigned layer);
    struct Allocation { ExpertId id; size_t slot; bool hit; int64_t evicted = -1; };
    std::vector<Allocation> acquire(const std::vector<ExpertId>& ids, bool prefetch = false);
    std::vector<Allocation> borrow(const std::vector<ExpertId>& ids); // hits only, no policy updates
    void release(const std::vector<Allocation>& lease);
    void pin(ExpertId id);
    void unpin(ExpertId id);
    int residency(ExpertId id) const;
    // The token protects this slot's current contents, including after pool
    // destruction. Token release is thread-safe and never calls the pool.
    std::shared_ptr<const void> retain(ExpertId id) const;
    // Opt-in exclusive tiers. reserve() evicts policy victims into draining slots until
    // empty+draining >= target and returns them (evicted = id). A draining slot has no
    // residency and is not allocatable until drained(slot). Soft (non-hard) pools only.
    std::vector<Allocation> reserve(size_t target, const std::unordered_set<ExpertId>& wanted);
    void drained(size_t slot);
    // Host pool: experts resident in the VRAM peer score 0, so duplicates are evicted first.
    void exclusive(bool enabled);
private:
    struct Slot { int64_t id = -1; uint32_t leases = 0; bool pinned = false; bool draining = false; uint64_t stamp = 0;
        std::shared_ptr<const void> retention;
    };
    std::vector<Slot> slots_;
    std::vector<int32_t> residency_;
    std::vector<uint32_t> frequency_;
    std::vector<uint64_t> last_use_;
    unsigned recent_, decay_;
    bool distinct_=true, budgets_=false;
    bool hard_=false;
    std::vector<unsigned> slot_layers_;
    std::array<size_t,43> layer_budgets_{};
    uint64_t clock_ = 0, stamp_ = 0;
    // MiMo policy clocks are per layer. Forty global layer calls must not
    // trigger forty times as much LFU decay or shorten a six-step recent window.
    std::array<uint64_t,43> layer_clocks_{};
    bool fast_=false, host_=false, exclusive_=false;
    int scan_victim(const std::unordered_set<ExpertId>& wanted, const std::array<size_t,43>& counts, int layer) const;
    const SlotPool* peer_=nullptr;
    std::array<bool,43> late_{}, dirty_{};
    std::array<int,43> best_any_{}, best_old_{};
    std::array<uint64_t,43> epochs_{}, peer_epochs_{}, seen_retention_epochs_{};
    using RetentionEpochs = std::array<std::atomic<uint64_t>,43>;
    std::shared_ptr<RetentionEpochs> retention_epochs_;
    uint64_t score(ExpertId id) const;
    bool less_slot(size_t a, size_t b) const;
    int fast_victim(const std::unordered_set<ExpertId>& wanted,
                    const std::array<size_t,43>& counts);
};

// Width-one, token-major policy controller. Before/after are paired once per
// successful target layer demand in traversal order 0..39; DSpark is ignored.
// A fresh controller is a fresh session; reset_history deliberately does not reset it.
class DriftPolicy {
public:
    DriftPolicy(const Placement& placement, SlotPool& vram, SlotPool& ram);
    void before(const std::vector<ExpertId>& ids);
    void after(const std::vector<ExpertId>& ids);
    bool active() const { return active_; }
    uint64_t completed_tokens() const { return age_; }
private:
    Placement placement_;
    SlotPool &vram_, &ram_;
    std::array<bool,expert_count> pins_{};
    std::deque<std::pair<uint64_t,uint64_t>> window_;
    uint64_t hits_=0, requests_=0, token_hits_=0, token_requests_=0, age_=0;
    unsigned next_layer_=0;
    bool active_=false, pending_=false;
};

// Anonymous, 2 MiB aligned MADV_HUGEPAGE memory. CUDA builds additionally use
// cudaHostRegister(..., Mapped) so it supports async DMA and UVA kernels.
class HostBuffer {
public:
    explicit HostBuffer(size_t bytes, bool cuda_register = false);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    void* data() const { return data_; }
    void* device_data() const { return device_; }
    size_t size() const { return bytes_; }
private:
    void *data_ = nullptr, *device_ = nullptr;
    size_t bytes_ = 0;
    bool registered_ = false;
};

// Single submitting thread, many requests in flight. No buffered fallback.
// The caller provides page-aligned registered staging and gives demand batches
// priority over speculative batches before submission; in-flight IO is not cancelled.
class DirectReader {
public:
    explicit DirectReader(unsigned queue_depth = 64);
    ~DirectReader();
    void read(const std::vector<const Record*>& records, const std::vector<void*>& destinations);
    // Bounded ring: ready(record_index, buffer_index) runs once, after every
    // extent of a record completed. recycle(buffer_index) must wait for its
    // previous asynchronous reader (e.g. H2D) before the next O_DIRECT write.
    // Callbacks execute on the submitting thread. Exceptions drain submitted IO.
    void stream(const std::vector<const Record*>& records, const std::vector<void*>& buffers,
                const std::function<void(size_t,size_t)>& ready,
                const std::function<void(size_t)>& recycle = {}, size_t chunk_bytes = 0);
    // A nonnull fixed[r] reads record r directly into a retained cache slot;
    // null records use the bounded staging ring. Callback buffer indices for
    // fixed records are buffers.size()+r. Ring recycle only sees ring indices.
    void stream_mixed(const std::vector<const Record*>& records, const std::vector<void*>& buffers,
                const std::vector<void*>& fixed,
                const std::function<void(size_t,size_t)>& ready,
                const std::function<void(size_t)>& recycle = {}, size_t chunk_bytes = 0);
    // As stream_mixed, plus chunk(record, buffer, dst_offset, bytes) once per successfully read chunk
    // (completion order, after CQ consumption is published), always before ready(record, buffer).
    void stream_mixed_chunked(const std::vector<const Record*>& records, const std::vector<void*>& buffers,
                const std::vector<void*>& fixed,
                const std::function<void(size_t,size_t)>& ready,
                const std::function<void(size_t,size_t,uint64_t,uint32_t)>& chunk,
                const std::function<void(size_t)>& recycle = {}, size_t chunk_bytes = 0);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ds41
