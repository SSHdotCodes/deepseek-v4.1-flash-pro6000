#include "ds41/decode_backend.hpp"
#include <dlfcn.h>
#include <stdexcept>
namespace ds41 {
static_assert(sizeof(DeviceExpert)==64 && sizeof(GemmJob)==32,"xk ABI v1 layout");
DecodeBackend::DecodeBackend()=default;
DecodeBackend::DecodeBackend(const Config& c){
 if(!c.enable_xk)return;
 if(c.xk_library.empty())throw std::invalid_argument("enabled xk backend requires an explicit frozen library path");
 module_=dlopen(c.xk_library.c_str(),RTLD_NOW|RTLD_LOCAL);
 if(!module_)throw std::runtime_error(std::string("xk dlopen: ")+dlerror());
 try{
  auto version=reinterpret_cast<int(*)()>(dlsym(module_,"ds41_xk_projection_abi_version"));
  grouped_=reinterpret_cast<Grouped>(dlsym(module_,"ds41_xk_grouped_v1"));
  if(!version || version()!=1 || !grouped_)throw std::runtime_error("incompatible xk projection ABI");
 }catch(...){dlclose(module_);module_=nullptr;grouped_=nullptr;throw;}
}
DecodeBackend::~DecodeBackend(){if(module_)dlclose(module_);}
cudaError_t DecodeBackend::launch(const DeviceExpert* t,const GemmJob* jobs,int count,Projection p,int n,int k,cudaStream_t stream)const{
 if(grouped_ && count>0 && count<=48 && n>0 && n%8==0 && (k==5120 || k==2304))return cudaError_t(grouped_(t,jobs,count,int(p),n,k,stream));
 return grouped_mxfp4_native_parallel(t,jobs,count,p,n,k,stream);
}
}
