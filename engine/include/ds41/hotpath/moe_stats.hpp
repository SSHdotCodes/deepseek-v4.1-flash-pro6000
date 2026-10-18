#pragma once
#include "ds41/offload.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#ifndef DS41_MOE_SOURCE_API
#define DS41_MOE_SOURCE_API 2
#endif
static_assert(ds41::OffloadStore::api_version==DS41_MOE_SOURCE_API,
              "MoE headers and the explicitly selected source API must match");
namespace ds41e::hotpath {
inline uint64_t counter_delta(uint64_t before,uint64_t after,const char *field) {
  if(after<before) throw std::runtime_error(std::string("MoE counter epoch changed: ")+field);
  return after-before;
}
inline nlohmann::json moe_delta(const ds41::Stats &before,const ds41::Stats &after,
                                bool include_layers=false) {
  using J=nlohmann::json;
  J out={{"source_api",DS41_MOE_SOURCE_API}};
#define DELTA(field) out[#field]=counter_delta(before.field,after.field,#field)
  DELTA(requests);DELTA(hits);DELTA(host_hits);DELTA(nvme_misses);
  DELTA(h2d_bytes);DELTA(nvme_bytes);DELTA(evictions);DELTA(prefetch_used);
  DELTA(need_host_ns);DELTA(io_ns);DELTA(staging_wait_ns);
  DELTA(prefetch_requests);DELTA(prefetch_cancelled);DELTA(prefetch_wasted);
  DELTA(prefetch_h2d_bytes);DELTA(prefetch_nvme_bytes);
#if DS41_MOE_SOURCE_API>=4
  DELTA(ram_admission_blocked);DELTA(refill_bytes);DELTA(refill_skipped);
#ifdef DS41_MOE_STATS_DEMOTE
  DELTA(demotions);DELTA(demote_resident);DELTA(direct_evictions);
#endif
  for(int tier=0;tier<2;++tier) {
    const auto &a=tier?after.ram:after.vram,&b=tier?before.ram:before.vram;
    J value;
#define TIER(field) value[#field]=counter_delta(b.field,a.field,#field)
    TIER(hits);TIER(misses);TIER(promotions);TIER(evictions);TIER(wasted_bytes);
#undef TIER
    out[tier?"ram":"vram"]=std::move(value);
  }
#endif
#undef DELTA
  if(include_layers) {
    out["layers"]=J::array();
    for(size_t i=0;i<after.layers.size();++i) {
      const auto &a=after.layers[i],&b=before.layers[i];J layer={{"layer",i}};
#define LAYER(field) layer[#field]=counter_delta(b.field,a.field,#field)
      LAYER(requests);LAYER(hits);LAYER(host_hits);LAYER(nvme_misses);
      LAYER(h2d_bytes);LAYER(nvme_bytes);LAYER(stall_ns);
#undef LAYER
      out["layers"].push_back(std::move(layer));
    }
    out["layer_stall_scope"]="host scheduler; multi-layer attribution overlaps";
  }
  // Callback entry is after the core's readiness waits. A before-wait observer
  // is required to measure these with consumer events. Never substitute host ns.
  out["gpu_transfer_wait_ms"]=nullptr;
  out["gpu_transfer_wait_status"]="observer unavailable in pinned interface";
  return out;
}
} // namespace ds41e::hotpath
