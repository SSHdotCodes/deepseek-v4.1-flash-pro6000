#pragma once
#include "ds41/prefill/tiers.hpp"
#include "ds41/store.hpp"
#include <cuda_runtime.h>
namespace ds41e::prefill {
struct TransferStats {
  uint64_t host_hits=0,nvme_misses=0,h2d_bytes=0,nvme_bytes=0;
  double h2d_stream_ms=0,host_service_ms=0;
  double ram_service_host_ms=0,nvme_service_host_ms=0;
};
// Fixed isolated VRAM and pinned disk ring. It never admits to the decode pool.
// At most one stream() call at a time. Callbacks only enqueue GPU operations,
// never call this stager or the originating store. They are serialized so one
// consumer scratch slab can safely be reused between complete expert jobs.
class Stager {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  using Ready=std::function<void(ds41::ExpertId,const ds41::DeviceExpert*,cudaStream_t)>;
  Stager(std::vector<ds41::Record>,size_t slots=8);
  ~Stager();
  size_t device_bytes() const noexcept; // immutable payload allocation census
  TransferStats stream(const std::vector<ds41::ExpertId>&,cudaStream_t,
      const Ready&,const HostLookup& = {},const std::function<void()>& before_disk = {});
};
}
