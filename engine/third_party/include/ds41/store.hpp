#pragma once
#include <cstddef>
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
std::vector<Record> load_records(const std::string& manifest_tsv);
struct DeviceExpert {
    const uint8_t *gate = nullptr, *gate_scale = nullptr;
    const uint8_t *up = nullptr, *up_scale = nullptr;
    const uint8_t *down = nullptr, *down_scale = nullptr;
    const uint32_t *ready = nullptr;
    uint32_t generation = 0;
    int32_t slot = -1;
};
struct Stats {
    uint64_t requests = 0, hits = 0, host_hits = 0, nvme_misses = 0;
    uint64_t h2d_bytes = 0, nvme_bytes = 0, evictions = 0, prefetch_used = 0;
    uint64_t route_wait_ns = 0, need_host_ns = 0, io_ns = 0;
};

// CPU-only, deterministic allocation policy. A lease prevents eviction until
// compute has finished, not merely until the transfer completed.
class SlotPool {
public:
    explicit SlotPool(size_t slots, unsigned recent_window = 6, unsigned decay_interval = 32);
    struct Allocation { ExpertId id; size_t slot; bool hit; int64_t evicted = -1; };
    std::vector<Allocation> acquire(const std::vector<ExpertId>& ids, bool prefetch = false);
    void release(const std::vector<Allocation>& lease);
    void pin(ExpertId id);
    void unpin(ExpertId id);
    int residency(ExpertId id) const;
private:
    struct Slot { int64_t id = -1; uint32_t leases = 0; bool pinned = false; uint64_t stamp = 0; };
    std::vector<Slot> slots_;
    std::vector<uint32_t> frequency_;
    std::vector<uint64_t> last_use_;
    unsigned recent_, decay_;
    uint64_t clock_ = 0, stamp_ = 0;
    // MiMo policy clocks are per layer. Forty global layer calls must not
    // trigger forty times as much LFU decay or shorten a six-step recent window.
    std::array<uint64_t,43> layer_clocks_{};
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ds41
