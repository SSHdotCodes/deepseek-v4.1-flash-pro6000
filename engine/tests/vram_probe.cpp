#include "ds41/engine.hpp"
#include "ds41/hotpath/gate.hpp"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <cstring>
using namespace ds41e;
static void checkpoint(const std::string &name){auto f=(void(*)(const char*))dlsym(RTLD_DEFAULT,"ds41_vram_checkpoint");if(f)f(name.c_str());}
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("probe WEIGHTS RECEIPT ENGRAM ROPE EXPERTS AUX TOKENS OUT");
 Config c;c.weights=argv[1];c.receipt=argv[2];c.engram=argv[3];c.rope=argv[4];c.experts=argv[5];c.auxiliary=argv[6];
 c.report_vram=true;c.expert_slots=384;c.max_context=8192;c.allow_unqualified=true;c.batched_verify=true;c.streamed_experts=true;c.graphs=true;c.hotpath_gate=true;
 Json fixture;{std::ifstream f(argv[7]);f>>fixture;}auto ids=fixture.get<std::vector<int>>();if(ids.size()>64)ids.resize(64);
 Json result={{"schema","ds41-vram-native-exactness-v1"},{"class","E0"},{"tokens",ids},{"cases",Json::array()}};
 cudaFree(nullptr);checkpoint("context_only");
 std::vector<std::vector<float>> gold;std::vector<std::vector<uint8_t>> state;
 auto base=make_native(c);checkpoint("base_loaded");result["base_loaded"]=base->counters();
 auto*e=dynamic_cast<hotpath::GateEvidence*>(base.get());
 for(int id:ids){gold.push_back(base->verify({id})[0]);state.push_back(e->hotpath_gate_rows().at(0));base->commit(1);}
 checkpoint("base_warmed");result["base_warmed"]=base->counters();base.reset();checkpoint("after_base_destroy");
 for(int flags=1;flags<=5;++flags){c.host_embedding=flags&1;c.reuse_woa=flags&2;c.cuda_stack_bytes=flags&4?1024:8192;auto model=make_native(c);checkpoint("candidate_"+std::to_string(flags)+"_loaded");
  auto *evidence=dynamic_cast<hotpath::GateEvidence*>(model.get());Json one={{"flags",flags},{"loaded",model->counters()},{"widths",Json::array()}};
  uint64_t bad=0,state_bad=0,values=0,state_values=0;
  for(int width=1;width<=6;++width){model->reset();for(size_t i=0;i<ids.size();){int n=std::min<size_t>(width,ids.size()-i);auto out=model->verify(std::vector<int>(ids.begin()+i,ids.begin()+i+n));auto&states=evidence->hotpath_gate_rows();
   for(int r=0;r<n;++r){values+=out[r].size();for(size_t k=0;k<out[r].size();++k)bad+=std::memcmp(&out[r][k],&gold[i+r][k],4)!=0;
    if(states[r].size()!=state[i+r].size())throw std::runtime_error("state extent");state_values+=states[r].size();for(size_t k=0;k<states[r].size();++k)state_bad+=states[r][k]!=state[i+r][k];}
   // Rejected rows can reuse both the shared dequant buffer and mapped tensor.
   model->commit(n);i+=n;}
   one["widths"].push_back(width);
  }
  one["logit_values"]=values;one["logit_mismatches"]=bad;one["state_bytes"]=state_values;one["state_mismatches"]=state_bad;one["pass"]=bad==0&&state_bad==0;
  checkpoint("candidate_"+std::to_string(flags)+"_warmed");one["warmed"]=model->counters();result["cases"].push_back(one);
  std::ofstream(argv[8])<<result.dump(2)<<"\n";std::cout<<"case "<<flags<<" logits="<<values<<" mismatches="<<bad<<" state="<<state_values<<" state_mismatches="<<state_bad<<std::endl;
  if(bad||state_bad)return 1;
 }
 // Additional DSpark dense residency census. No drafting or probability changes.
 c.host_embedding=false;c.reuse_woa=false;c.cuda_stack_bytes=8192;c.load_mtp=true;
 auto mtp=make_native(c);checkpoint("mtp_loaded");result["mtp_loaded"]=mtp->counters();mtp.reset();
 result["pass"]=true;std::ofstream(argv[8])<<result.dump(2)<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
