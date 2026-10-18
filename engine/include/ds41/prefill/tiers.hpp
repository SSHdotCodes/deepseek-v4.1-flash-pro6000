#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
namespace ds41e::prefill {
// Borrowed pinned native record. Empty data means a host-tier miss. keepalive
// must prevent relocation/unregister until the stager completes its H2D reads.
struct HostView {
  const void* data = nullptr;
  size_t bytes = 0;
  std::shared_ptr<void> keepalive;
};
using HostLookup=std::function<HostView(uint32_t)>;
}
