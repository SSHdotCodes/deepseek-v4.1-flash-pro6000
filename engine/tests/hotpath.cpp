#include "ds41/engine.hpp"
#include "ds41/hotpath/gate.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace ds41e;
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main(int argc,char **argv) {
  try {
    if(argc<7) throw std::invalid_argument("test_hotpath WEIGHTS RECEIPT ENGRAM ROPE EXPERTS SLOTS [positions] [graphs]");
    Config c;
    c.weights=argv[1]; c.receipt=argv[2]; c.engram=argv[3];
    c.rope=argv[4]; c.experts=argv[5]; c.expert_slots=std::stoull(argv[6]);
    c.max_context=8192; c.allow_unqualified=true;
    int n=argc>7?std::stoi(argv[7]):18;
    std::vector<int> ids(n+40);
    ids[0]=0;
    for(size_t i=1;i<ids.size();++i) ids[i]=100+int(i)*17;
    std::string token_source="synthetic";
    for(int i=8;i<argc;++i) {
      auto flag=std::string(argv[i]);
      if(flag.rfind("aux=",0)==0) c.auxiliary=flag.substr(4);
      if(flag.rfind("host-gib=",0)==0) c.host_bytes=size_t(std::stod(flag.substr(9))*1073741824.0);
      if(flag.rfind("tokens=",0)==0) {
        token_source=flag.substr(7);
        std::ifstream f(token_source); Json fixture; f>>fixture;
        ids=(fixture.is_array()?fixture:fixture.at("tokens")).get<std::vector<int>>();
        if(ids.size()<size_t(n+40)) throw std::invalid_argument("token fixture needs positions+40 rows");
        ids.resize(n+40);
      }
    }
    auto baseline=make_native(c);
    std::vector<std::vector<float>> gold;
    for(int id:ids) { gold.push_back(baseline->verify({id})[0]); baseline->commit(1); }
    auto base_stats=baseline->counters(); baseline.reset();
    c.batched_verify=true;
    bool perturb_suffix=false;
    std::string result_tag="r2";
    for(int i=8;i<argc;++i) {
      auto flag=std::string(argv[i]);
      if(flag=="graphs") c.graphs=true;
      else if(flag=="stream") c.streamed_experts=true;
      else if(flag=="prefetch") c.prefetch_routes=true;
      else if(flag=="profile") c.hotpath_profile=true;
      else if(flag=="dynamic-host") c.dynamic_host=true;
      else if(flag=="async-retire") c.async_retire=true;
      else if(flag.rfind("host-gib=",0)==0) {}
      else if(flag=="perturb") { perturb_suffix=true;c.hotpath_gate=true; }
      else if(flag.rfind("tag=",0)==0) result_tag=flag.substr(4);
      else if(flag.rfind("engram=",0)==0) c.engram_library=flag.substr(7);
      else if(flag.rfind("fuse=",0)==0) c.fuse_library=flag.substr(5);
      else if(flag.rfind("aux=",0)==0 || flag.rfind("tokens=",0)==0) {}
      else throw std::invalid_argument("unknown gate flag");
    }
    auto model=make_native(c);
    auto *evidence=dynamic_cast<hotpath::GateEvidence*>(model.get());
    if(perturb_suffix && !evidence) throw std::runtime_error("missing native gate evidence");
    Json results=Json::array(); int failures=0;
    for(int width=1;width<=6;++width) {
      model->reset(); int checks=0, causal_checks=0; uint64_t bad=0; double maxabs=0;
      int replaced_cases=0,permuted_cases=0,distinct_permutations=0;
      uint64_t state_bad=0,state_checked=0,causal_logit_bad=0;
      Json first_state=nullptr,first_causal_logit=nullptr;
      Json first=nullptr;
      auto compare=[&](const std::vector<float> &a,int position){
        ++checks;
        if(a.size()!=gold[position].size()) throw std::runtime_error("logit extent");
        for(size_t i=0;i<a.size();++i) {
          if(std::memcmp(&a[i],&gold[position][i],sizeof(float))) {
            ++bad;
            maxabs=std::max(maxabs,double(std::abs(a[i]-gold[position][i])));
            if(first.is_null()) first={{"position",position},{"vocab_id",i},
                {"actual",a[i]},{"expected",gold[position][i]}};
          }
        }
      };
      double start=now();
      for(int i=0;i<n;) {
        int count=std::min(width,n-i);
        auto out=model->verify(std::vector<int>(ids.begin()+i,ids.begin()+i+count));
        for(int r=0;r<count;++r) compare(out[r],i+r);
        model->commit(count); i+=count;
      }
      // Every possible accepted prefix, with rejected rows overwritten by
      // a six-row retry. The committed anchor advances only by accepted rows.
      int anchor=n;
      for(int accepted=0;accepted<=width;++accepted) {
        auto out=model->verify(std::vector<int>(ids.begin()+anchor,ids.begin()+anchor+width));
        for(int r=0;r<width;++r) compare(out[r],anchor+r);
        // Unreached draft contents may change expert grouping and scheduling,
        // but must never change the distribution of an earlier reached row.
        // Retry the original rows afterward to cover overwritten rejected state.
        if(perturb_suffix && accepted>0 && accepted<width) {
          auto original_state=evidence->hotpath_gate_rows();
          if(original_state.size()!=size_t(width)) throw std::runtime_error("gate row extent");
          for(int mode=0;mode<2;++mode) {
            if(mode==1 && width-accepted<2) continue;
            model->commit(0);
            std::vector<int> alternate(ids.begin()+anchor,ids.begin()+anchor+width);
            if(mode==0) {
              ++replaced_cases;
              for(int r=accepted;r<width;++r)
                alternate[r]=(alternate[r]+7000+131*width+29*r)%model->vocab();
            } else {
              ++permuted_cases;
              std::reverse(alternate.begin()+accepted,alternate.end());
              distinct_permutations+=!std::equal(alternate.begin(),alternate.end(),ids.begin()+anchor);
            }
            auto changed=model->verify(alternate);
            const auto &changed_state=evidence->hotpath_gate_rows();
            if(changed_state.size()!=size_t(width)) throw std::runtime_error("changed gate row extent");
            for(int r=0;r<accepted;++r) {
              compare(changed[r],anchor+r);++causal_checks;
              for(size_t i=0;i<out[r].size();++i) if(std::memcmp(&out[r][i],&changed[r][i],4)) {
                ++causal_logit_bad;
                if(first_causal_logit.is_null()) first_causal_logit={{"position",anchor+r},
                  {"vocab_id",i},{"mode",mode?"permute":"replace"},
                  {"expected",out[r][i]},{"actual",changed[r][i]}};
              }
              const auto &before=original_state[r],&after=changed_state[r];
              if(before.size()!=hotpath::GateLayout::row_bytes || after.size()!=before.size())
                throw std::runtime_error("gate byte extent");
              state_checked+=before.size();
              if(std::memcmp(before.data(),after.data(),before.size()))
                for(size_t i=0;i<before.size();++i) if(before[i]!=after[i]) {
                  ++state_bad;
                  if(first_state.is_null()) first_state={{"position",anchor+r},{"byte",i},
                    {"field",hotpath::GateLayout::describe(i)},{"mode",mode?"permute":"replace"},
                    {"expected",before[i]},{"actual",after[i]}};
                }
            }
          }
          model->commit(0);
          auto restored=model->verify(std::vector<int>(ids.begin()+anchor,ids.begin()+anchor+width));
          for(int r=0;r<width;++r) compare(restored[r],anchor+r);
        }
        model->commit(accepted); anchor+=accepted;
        auto retry=model->verify(std::vector<int>(ids.begin()+anchor,ids.begin()+anchor+6));
        for(int r=0;r<6;++r) compare(retry[r],anchor+r);
        model->commit(0);
      }
      auto stats=model->counters();
      if(stats.contains("step_profile") && !stats["step_profile"].empty()) {
        std::string path="results/hotpath-step-"+result_tag+"-width"+std::to_string(width)+".json";
        std::ofstream f(path);f<<stats["step_profile"].dump();
        if(!f) throw std::runtime_error("profile write");
        stats.erase("step_profile");stats["step_profile_path"]=path;
      }
      results.push_back({{"width",width},{"checks",checks},{"causal_checks",causal_checks},{"mismatched_logits",bad},
          {"causal_mismatched_logits",causal_logit_bad},{"first_causal_logit_difference",first_causal_logit},
          {"state_bytes_checked",state_checked},{"state_mismatched_bytes",state_bad},
          {"first_state_difference",first_state},{"replaced_cases",replaced_cases},
          {"permuted_cases",permuted_cases},{"distinct_permutations",distinct_permutations},
          {"maxabs",maxabs},{"first_difference",first},{"elapsed_seconds",now()-start},
          {"counters",stats}});
      std::cout<<results.back().dump()<<std::endl;
      failures+=bad!=0 || state_bad!=0 || causal_logit_bad!=0;
    }
    std::cout<<Json({{"schema","ds41-hotpath-native-v2"},{"baseline",base_stats},
        {"positions",n},{"token_source",token_source},{"perturbed_suffix",perturb_suffix},
        {"state_layout","ds41-hotpath-state-v1"},{"state_row_bytes",hotpath::GateLayout::row_bytes},
        {"results",results},{"failed_widths",failures}}).dump()<<std::endl;
    return failures?1:0;
  } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl; return 2; }
}
