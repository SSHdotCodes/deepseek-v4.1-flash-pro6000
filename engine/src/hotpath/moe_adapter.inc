// Native members. Source API selection is explicit and checked by moe_stats.hpp.
void retire_experts(ds41::OffloadStore::Lease &lease) {
#if DS41_MOE_SOURCE_API>=4
  if(config.async_retire) {
    auto completion=experts->retire_async(std::move(lease),stream);
    lease={};retirement_futures.push_back(std::move(completion));
    return;
  }
#endif
  experts->retire(std::move(lease),stream);lease={};
}
void check_moe_retirement(bool drain) {
#if DS41_MOE_SOURCE_API>=4
  experts->check_error();
  while(retirement_observed<retirement_futures.size()) {
    auto &f=retirement_futures[retirement_observed];
    if(!drain && f.wait_for(std::chrono::seconds(0))!=std::future_status::ready) break;
    f.get();++retirement_observed;
  }
  if(drain) { retirement_futures.clear();retirement_observed=0; }
  experts->check_error();
#else
  (void)drain;
#endif
}

ds41::Stats sample_expert_stats() const {
  auto result=experts->stats();
  last_expert_stats=result;expert_stats_available=true;
  return result;
}
