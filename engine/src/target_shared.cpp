#include "ds41/engine.hpp"
#include <cstring>
using namespace ds41e;
extern "C" int ds41_target_query(void *target, uint32_t kind, uint32_t version,
                                 void *output, size_t bytes) {
  if (!target || !output)
    return -1;
  auto *model = static_cast<Forward *>(target);
  try {
    switch (kind) {
    case DS41_TARGET_WEIGHTS: {
      if (version != 2 || bytes != sizeof(ds41_target_weights_v2))
        return -1;
      auto view = model->shared_weights();
      if (view.abi_version != 2 || view.struct_bytes != sizeof(view))
        return -2;
      std::memcpy(output, &view, sizeof(view));
      return 0;
    }
    case DS41_TARGET_SERVICES: {
      if (version != 1 || bytes != sizeof(ds41_target_services_v1))
        return -1;
      auto view = model->services();
      if (view.abi_version != 1)
        return -2;
      std::memcpy(output, &view, sizeof(view));
      return 0;
    }
    case DS41_TARGET_CAPTURES: {
      if (version != 1 || bytes != sizeof(ds41_target_capture_v1))
        return -1;
      auto view = model->captures();
      if (view.abi_version != 1)
        return -2;
      std::memcpy(output, &view, sizeof(view));
      return 0;
    }
    default:
      return -1;
    }
  } catch (...) {
    return -2;
  }
}
extern "C" int ds41_target_tensor(void *target, const char *name,
                                  uint32_t version, void *output,
                                  size_t bytes) {
  if (!target || !name || !output || version != 1 ||
      bytes != sizeof(ds41_target_tensor_v1))
    return -1;
  try {
    auto view = static_cast<Forward *>(target)->resident_tensor(name);
    if (view.abi_version != 1 || view.struct_bytes != sizeof(view) ||
        !view.data)
      return -2;
    std::memcpy(output, &view, sizeof(view));
    return 0;
  } catch (...) {
    return -2;
  }
}
extern "C" int ds41_target_visit_tensors(void *target, const char *prefix,
                                         uint32_t version,
                                         ds41_target_tensor_visitor visitor,
                                         void *user) {
  if (!target || !prefix || !visitor || version != 1)
    return -1;
  try {
    auto *model = static_cast<Forward *>(target);
    for (const auto &name : model->resident_names(prefix)) {
      auto view = model->resident_tensor(name);
      if (view.abi_version != 1 || view.struct_bytes != sizeof(view) ||
          !view.data)
        return -2;
      int rc = visitor(user, name.c_str(), &view);
      if (rc)
        return rc;
    }
    return 0;
  } catch (...) {
    return -2;
  }
}
