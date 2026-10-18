#include "ds41/hotpath/moe_stats.hpp"
#include <cassert>
int main() {
  ds41::Stats before{},after{};
  before.requests=100;before.hits=80;before.host_hits=15;before.nvme_misses=5;
  after=before;after.requests+=10;after.hits+=7;after.host_hits+=2;after.nvme_misses+=1;
  before.h2d_bytes=123;after.h2d_bytes=123+3*18804736;
  before.layers[3].requests=80;after.layers[3].requests=90;
  auto d=ds41e::hotpath::moe_delta(before,after,true);
  assert(d["requests"]==10 && d["hits"]==7 && d["host_hits"]==2 && d["nvme_misses"]==1);
  assert(d["h2d_bytes"]==3*18804736 && d["layers"][3]["requests"]==10);
  assert(d["gpu_transfer_wait_ms"].is_null());
#if DS41_MOE_SOURCE_API>=4
  before.vram.promotions=1000;after.vram.promotions=1003;
  before.ram.hits=100;after.ram.hits=102;
  after.ram_admission_blocked=1;
  d=ds41e::hotpath::moe_delta(before,after);
  assert(d["vram"]["promotions"]==3 && d["ram"]["hits"]==2);
  assert(d["ram_admission_blocked"]==1);
#endif
  after.requests=99;bool rejected=false;
  try { (void)ds41e::hotpath::moe_delta(before,after); }
  catch(const std::runtime_error &) { rejected=true; }
  assert(rejected);
}
