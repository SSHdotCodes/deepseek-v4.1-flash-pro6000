#pragma once
#include "ds41/fuse.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>
namespace ds41e::hotpath {
// Optional C ABI module; loaded once before any capture and kept alive until
// every captured consumer has been destroyed. No caller uses a second cuBLAS
// handle or changes the target's M1 numerical profile.
struct FusedOps {
  void *library=nullptr;
  decltype(&ds41_fuse_norm_quant_decode) norm=nullptr;
  decltype(&ds41_fuse_hc_prepare) prepare=nullptr;
  decltype(&ds41_fuse_rms_rope_decode) rms_rope=nullptr;
  template<class T> T symbol(const char *name) {
    auto result=reinterpret_cast<T>(dlsym(library,name));
    if(!result) throw std::runtime_error(std::string("missing fusion symbol: ")+name);
    return result;
  }
  explicit FusedOps(const std::string &path) {
    library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!library) throw std::runtime_error(std::string("fusion library: ")+dlerror());
    try {
      norm=symbol<decltype(norm)>("ds41_fuse_norm_quant_decode");
      prepare=symbol<decltype(prepare)>("ds41_fuse_hc_prepare");
      rms_rope=symbol<decltype(rms_rope)>("ds41_fuse_rms_rope_decode");
    } catch(...) { dlclose(library);library=nullptr;throw; }
  }
  ~FusedOps() { if(library) dlclose(library); }
  FusedOps(const FusedOps &)=delete;
  FusedOps &operator=(const FusedOps &)=delete;
};
}
