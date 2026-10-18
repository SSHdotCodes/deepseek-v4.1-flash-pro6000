#include "ds41/blas_runtime.hpp"
#include "ds41/native_store.hpp"
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
namespace ds41e {
nlohmann::json native_blas_runtime() {
#ifdef DS41_BLAS_PIN_ACTIVE
  static const nlohmann::json identity = [] {
    nlohmann::json result = {{"enforced", true}};
    struct Entry { const char *name, *symbol, *sha; };
    for (const auto &entry : {
        Entry{"cublas", "cublasCreate_v2", DS41_CUBLAS_SHA256},
        Entry{"cublasLt", "cublasLtGetVersion", DS41_CUBLASLT_SHA256},
        Entry{"cudart", "cudaRuntimeGetVersion", DS41_CUDART_SHA256}}) {
      void *address = dlsym(RTLD_DEFAULT, entry.symbol);
      Dl_info info{};
      if (!address || !dladdr(address, &info) || !info.dli_fname)
        throw std::runtime_error(std::string("native runtime symbol missing: ") + entry.symbol);
      const auto path = std::filesystem::canonical(info.dli_fname).string();
      const auto digest = sha256_file(path);
      if (digest != entry.sha)
        throw std::runtime_error(std::string("native runtime family mismatch: ") +
            entry.name + " at " + path + "; use the selected pinned BLAS overlay");
      result[entry.name] = {{"path", path}, {"sha256", digest}};
    }
    return result;
  }();
  return identity;
#else
  return {{"enforced", false}};
#endif
}
}
