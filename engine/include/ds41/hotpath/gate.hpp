#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ds41e::hotpath {
// Diagnostic-only byte layout. Copies observe existing device values and never
// feed the target. The optional slab is allocated once, before graph capture.
struct GateLayout {
  static constexpr size_t hc_bytes=4*5120*2, layers=40;
  enum Offset : size_t {
    Input=0, IncomingPre=Input+hc_bytes,
    AttnPre=IncomingPre+16, AttnPost=AttnPre+16, AttnComb=AttnPost+16,
    AfterAttn=AttnComb+64, FfnPre=AfterAttn+hc_bytes,
    FfnPost=FfnPre+16, FfnComb=FfnPost+16,
    RouteIds=FfnComb+64, RouteWeights=RouteIds+6*4,
    LayerBytes=RouteWeights+6*4
  };
  static constexpr size_t final_h=layers*LayerBytes;
  static constexpr size_t final_pre=final_h+hc_bytes;
  static constexpr size_t row_bytes=final_pre+16;
  static std::string describe(size_t byte) {
    if(byte>=final_pre) return "final.pre";
    if(byte>=final_h) return "final.hc";
    size_t layer=byte/LayerBytes,offset=byte%LayerBytes;
    const size_t ends[]={IncomingPre,AttnPre,AttnPost,AttnComb,AfterAttn,
                         FfnPre,FfnPost,FfnComb,RouteIds,RouteWeights,LayerBytes};
    const char *names[]={"input.hc","incoming.pre","attn.pre","attn.post",
                         "attn.comb","after_attn.hc","ffn.pre","ffn.post",
                         "ffn.comb","route.ids","route.weights"};
    for(size_t i=0;i<11;++i) if(offset<ends[i])
      return "layer."+std::to_string(layer)+"."+names[i];
    return "invalid";
  }
};
static_assert(GateLayout::row_bytes<4*1024*1024,"bounded diagnostic slab");
// Test-only interface keeps diagnostics out of the serving/target C ABIs.
struct GateEvidence {
  virtual ~GateEvidence()=default;
  virtual const std::vector<std::vector<uint8_t>> &hotpath_gate_rows() const=0;
};
} // namespace ds41e::hotpath
