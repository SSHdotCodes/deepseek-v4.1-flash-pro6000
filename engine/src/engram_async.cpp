#include "ds41/engram_async.hpp"
#include "ds41/engram.h"
#include <dlfcn.h>
#include <stdexcept>
namespace ds41e {
struct EngramAsync::Impl {
  void *library = nullptr;
  ds41_engram_store *store = nullptr;
  decltype(&ds41_engram_open) open;
  decltype(&ds41_engram_close) close;
  decltype(&ds41_engram_submit) submit;
  decltype(&ds41_engram_wait) wait;
  decltype(&ds41_engram_data) data;
  decltype(&ds41_engram_ticket_free) free;
  decltype(&ds41_engram_dequant) dequant;
  decltype(&ds41_engram_get_stats) stats;
  decltype(&ds41_engram_error) error;
  template <class T> T symbol(const char *name) {
    dlerror();
    auto p = dlsym(library, name);
    const char *e = dlerror();
    if (e || !p)
      throw std::runtime_error(std::string("Engram symbol ") + name + ": " +
                               (e ? e : "null"));
    return reinterpret_cast<T>(p);
  }
  explicit Impl(const std::string &path) {
    library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(std::string("Engram dlopen: ") + dlerror());
    try {
#define SYMBOL(name) name = symbol<decltype(name)>("ds41_engram_" #name)
      SYMBOL(open);
      SYMBOL(close);
      SYMBOL(submit);
      SYMBOL(wait);
      SYMBOL(data);
      free = symbol<decltype(free)>("ds41_engram_ticket_free");
      SYMBOL(dequant);
      stats = symbol<decltype(stats)>("ds41_engram_get_stats");
      SYMBOL(error);
#undef SYMBOL
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }
  ~Impl() {
    if (store)
      close(store);
    if (library)
      dlclose(library);
  }
  void check(int rc) const {
    if (rc)
      throw std::runtime_error(std::string("Engram store: ") + error());
  }
};
EngramAsync::EngramAsync(const std::string &library, const Config &c)
    : impl(std::make_shared<Impl>(library)) {
  const std::string index = c.weights + "/model.safetensors.index.json";
  ds41_engram_config cfg{};
  cfg.struct_bytes = sizeof(cfg);
  cfg.abi_version = DS41_ENGRAM_ABI_VERSION;
  cfg.weight_root = c.weights.c_str();
  cfg.index_json = index.c_str();
  cfg.hash_json = c.engram.c_str();
  cfg.integrity_json = c.receipt.c_str();
  cfg.cache_rows = 65536;
  cfg.queue_depth = 48;
  cfg.max_batch_rows = 4096;
  cfg.pin_memory = 1;
  impl->check(impl->open(&cfg, &impl->store));
}
EngramAsync::Ticket::Ticket(std::shared_ptr<Impl> p, void *t)
    : owner(std::move(p)), ticket(t) {}
EngramAsync::Ticket::~Ticket() {
  owner->free(static_cast<ds41_engram_ticket *>(ticket));
}
const void *EngramAsync::Ticket::wait(size_t expected_rows) {
  auto t = static_cast<ds41_engram_ticket *>(ticket);
  owner->check(owner->wait(t));
  const void *data = nullptr;
  size_t count = 0;
  owner->check(owner->data(t, &data, &count));
  if (!data || count != expected_rows)
    throw std::runtime_error("Engram ticket shape mismatch");
  return data;
}
std::unique_ptr<EngramAsync::Ticket>
EngramAsync::submit(int layer, const std::vector<int64_t> &rows) {
  ds41_engram_ticket *ticket = nullptr;
  impl->check(
      impl->submit(impl->store, layer, rows.data(), rows.size(), &ticket));
  return std::unique_ptr<Ticket>(new Ticket(impl, ticket));
}
void EngramAsync::dequant(const void *raw, void *bf16, int rows, void *stream) {
  impl->check(impl->dequant(raw, bf16, rows, stream));
}
Json EngramAsync::stats() const {
  ds41_engram_stats s{};
  impl->check(impl->stats(impl->store, &s));
  return {{"submitted_rows", s.submitted_rows},
          {"cache_hits", s.cache_hits},
          {"unique_misses", s.unique_misses},
          {"disk_pages", s.disk_pages},
          {"disk_bytes", s.disk_bytes},
          {"batches", s.batches},
          {"cache_capacity_rows", s.cache_capacity_rows},
          {"cache_live_rows", s.cache_live_rows},
          {"io_ms", s.io_ms},
          {"total_ms", s.total_ms}};
}
} // namespace ds41e
