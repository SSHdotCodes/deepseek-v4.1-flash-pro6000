#pragma once
#include <array>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ds41e::hotpath {
// Profile only. The caller completes the consumer stream before collect/reset.
// Callback entry is after the store's mapping/payload wait commands, so these
// intervals include worker/host submission gaps as well as GPU readiness waits.
// They are not isolated DMA durations or a substitute for a core wait observer.
class ConsumerReadiness {
  static void check(cudaError_t rc) {
    if(rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
  }
  struct Event {
    cudaEvent_t value=nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { if(value) cudaEventDestroy(value); }
    Event(const Event&)=delete;
    Event& operator=(const Event&)=delete;
  };
  struct Gap { int layer=0;cudaEvent_t before=nullptr,after=nullptr; };
  std::array<Event,40> shared_;
  std::array<Event,1440> arrival_;
  std::array<Gap,1440> gaps_{};
  std::array<cudaEvent_t,40> previous_{};
  size_t count_=0;
public:
  void reset() { count_=0;previous_.fill(nullptr); }
  void shared_done(int layer,cudaStream_t stream) {
    check(cudaEventRecord(shared_.at(layer).value,stream));
    previous_.at(layer)=shared_[layer].value;
  }
  // Exactly one callback writer, ordered by need_streamed_batches completion.
  // No allocation, query, synchronization or store reentry in either hook.
  void arrived(int layer,cudaStream_t stream) {
    if(count_==gaps_.size() || !previous_.at(layer))
      throw std::logic_error("consumer readiness event capacity/order");
    const auto end=arrival_[count_].value;
    check(cudaEventRecord(end,stream));
    gaps_[count_++]={layer,previous_[layer],end};
  }
  void expert_done(int layer,cudaEvent_t end) { previous_.at(layer)=end; }
  nlohmann::json collect() const {
    using J=nlohmann::json;
    std::array<double,40> elapsed{};
    std::array<size_t,40> batches{};
    double total=0;
    for(size_t i=0;i<count_;++i) {
      float ms=0;
      check(cudaEventElapsedTime(&ms,gaps_[i].before,gaps_[i].after));
      total+=ms;elapsed[gaps_[i].layer]+=ms;++batches[gaps_[i].layer];
    }
    J layers=J::array();
    for(size_t layer=0;layer<40;++layer)
      layers.push_back({{"layer",layer},{"gap_ms",elapsed[layer]},{"batches",batches[layer]}});
    return {{"gap_ms",total},{"batches",count_},{"layers",std::move(layers)},
        {"clock","CUDA events on the actual consumer stream"},
        {"scope","after shared/previous expert end marker to next ready callback entry; includes mapping/payload waits and worker/host submission gaps; excludes callback metadata upload"},
        {"isolated_gpu_transfer_wait_ms",nullptr},
        {"lifetime","startup event pool, reused only after verify stream completion; demand lease retained through ordered reduction and retirement"}};
  }
};
} // namespace ds41e::hotpath
