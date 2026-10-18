#pragma once
#include "ds41/engine.hpp"
#include <memory>
namespace ds41e {
// Optional additive host-store consumer. The library owns hashing/IO/cache;
// this class only keeps its C ABI, tickets and dynamic-library lifetime intact.
class EngramAsync {
  struct Impl;
  std::shared_ptr<Impl> impl;

public:
  class Ticket {
    friend class EngramAsync;
    std::shared_ptr<Impl> owner;
    void *ticket = nullptr;
    Ticket(std::shared_ptr<Impl>, void *);

  public:
    ~Ticket();
    const void *wait(size_t expected_rows);
  };
  EngramAsync(const std::string &library, const Config &);
  std::unique_ptr<Ticket> submit(int layer, const std::vector<int64_t> &rows);
  void dequant(const void *raw_device, void *bf16, int rows, void *stream);
  Json stats() const;
};
} // namespace ds41e
