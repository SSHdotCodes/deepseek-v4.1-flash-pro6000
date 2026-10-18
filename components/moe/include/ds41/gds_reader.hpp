#pragma once
// Opt-in GPUDirect Storage reads (DS41_MOE_GDS=1): NVMe extents land directly in a registered
// VRAM arena through cuFile (PCI P2PDMA), bypassing host DRAM. libcufile is loaded at runtime.
#include "ds41/store.hpp"
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ds41 {
class GdsReader {
public:
    // Registers `bytes` of device memory at `base` for P2P. Throws if cuFile or P2P is unavailable.
    GdsReader(void* base, size_t bytes, size_t piece_bytes, unsigned threads);
    ~GdsReader();
    GdsReader(const GdsReader&) = delete;
    GdsReader& operator=(const GdsReader&) = delete;
    // Reads every extent of records[j] to base+offsets[j]+extent.dst_offset. ready(j) runs on the
    // calling thread, in record order, once all of record j's payload arrived. Errors throw after
    // every submitted piece finished.
    void read(const std::vector<const Record*>& records, const std::vector<size_t>& offsets,
              const std::function<void(size_t)>& ready);
    bool p2p() const { return p2p_; }
private:
    struct Piece { size_t record; const Extent* extent; uint64_t file_offset, bytes, required; size_t device_offset; };
    void* lib_ = nullptr;
    void* base_ = nullptr;
    size_t bytes_ = 0, piece_ = 0;
    bool p2p_ = false;
    struct Api;
    Api* api_ = nullptr;
    struct File { int fd = -1; void* handle = nullptr; };
    std::map<std::string, File> files_;
    std::mutex mutex_;
    std::condition_variable work_cv_, done_cv_;
    std::vector<Piece> queue_;
    size_t next_ = 0;
    std::vector<size_t> remaining_;
    std::string error_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
    void* handle(const std::string& path);
    void worker();
};
} // namespace ds41
