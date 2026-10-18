#pragma once
#include <nlohmann/json.hpp>
namespace ds41e {
// Pure loader/file identity check. It does not create a CUDA/BLAS context.
nlohmann::json native_blas_runtime();
}
