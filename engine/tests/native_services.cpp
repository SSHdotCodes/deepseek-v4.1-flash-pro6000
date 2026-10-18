#include "ds41/engine.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace ds41e;
static void ck(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
struct Buffer {
  void *p = nullptr;
  explicit Buffer(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
  ~Buffer() { cudaFree(p); }
  void upload(const void *data, size_t bytes, cudaStream_t s) {
    ck(cudaMemcpyAsync(p, data, bytes, cudaMemcpyHostToDevice, s));
  }
};
static std::vector<char> tensor(const std::string &path, const char *name) {
  std::ifstream f(path, std::ios::binary);
  uint64_t header_bytes = 0;
  f.read(reinterpret_cast<char *>(&header_bytes), 8);
  if (!f || header_bytes > 16 * 1024 * 1024)
    throw std::runtime_error("invalid fixture header");
  std::string header(header_bytes, '\0');
  f.read(header.data(), header.size());
  auto entry = Json::parse(header).at(name);
  auto offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
  if (offsets.size() != 2 || offsets[1] < offsets[0])
    throw std::runtime_error("invalid fixture tensor");
  std::vector<char> data(offsets[1] - offsets[0]);
  f.seekg(8 + header_bytes + offsets[0]);
  f.read(data.data(), data.size());
  if (!f)
    throw std::runtime_error("short fixture tensor");
  return data;
}
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument("test_native_services WEIGHTS RECEIPT ENGRAM "
                                  "ROPE EXPERTS FIXTURE_DIRECTORY");
    Config c;
    c.weights = argv[1];
    c.receipt = argv[2];
    c.engram = argv[3];
    c.rope = argv[4];
    c.experts = argv[5];
    c.max_context = 8192;
    c.expert_slots = 64;
    c.allow_unqualified = true;
    c.load_mtp = true;
    auto model = make_native(c);
    auto shared = model->shared_weights();
    auto services = model->services();
    ds41_target_weights_v2 query{};
    if (ds41_target_query(model.get(), DS41_TARGET_WEIGHTS, 2, &query,
                          sizeof(query)) ||
        query.embedding_bf16 != shared.embedding_bf16 ||
        query.head_f32 != shared.head_f32 ||
        query.blas_handle != shared.blas_handle ||
        query.rope_f32 != shared.rope_f32)
      throw std::runtime_error("target C query does not match internal views");
    if (shared.abi_version != 2 || shared.struct_bytes != sizeof(shared) ||
        !shared.embedding_bf16 || !shared.head_f32 || !shared.cuda_stream ||
        !shared.blas_handle || services.abi_version != 1)
      throw std::runtime_error("missing shared native services");
    ds41_target_tensor_v1 named{};
    for (const auto &entry : {std::pair<const char *, const void *>{
                                  "embed.weight", shared.embedding_bf16},
                              {"head.weight", shared.head_f32},
                              {"rope", shared.rope_f32}}) {
      if (ds41_target_tensor(model.get(), entry.first, 1, &named,
                             sizeof(named)) ||
          named.data != entry.second)
        throw std::runtime_error(
            "shared tensor binding is duplicated or unavailable");
    }
    struct Names {
      int count = 0;
      int stages[3]{};
    } names;
    auto visitor = [](void *raw, const char *name,
                      const ds41_target_tensor_v1 *view) -> int {
      auto &n = *static_cast<Names *>(raw);
      std::string key(name);
      if (view->abi_version != 1 || view->struct_bytes != sizeof(*view) ||
          !view->data || view->rows < 1 || view->columns < 1 ||
          key.find(".experts.") != std::string::npos)
        return -1;
      if ((key.find(".ffn.gate.weight") != std::string::npos ||
           key == "mtp.2.markov_head.head.weight" ||
           key == "mtp.2.confidence_head.proj.weight") &&
          view->dtype != DS41_TARGET_F32)
        return -2;
      if (key.find(".attn.wo_a.weight") != std::string::npos &&
          view->dtype != DS41_TARGET_BF16)
        return -3;
      if (key.compare(0, 4, "mtp.") || key[4] < '0' || key[4] > '2')
        return -4;
      ++n.count;
      ++n.stages[key[4] - '0'];
      return 0;
    };
    if (ds41_target_visit_tensors(model.get(), "mtp.", 1, visitor, &names) ||
        names.count < 60 || !names.stages[0] || !names.stages[1] ||
        !names.stages[2])
      throw std::runtime_error(
          "MTP tensor enumeration/dtype qualification failed");
    auto stream = static_cast<cudaStream_t>(shared.cuda_stream);
    Buffer x(8 * 5120 * 2), ids(8 * 3 * 4), routes(8 * 3 * 4),
        out(8 * 3 * 5120 * 2);
    int values = 0, calls = 0;
    int known_ids[] = {0, 127, 64};
    const char *files[] = {"layer40-expert000.safetensors",
                           "layer41-expert127.safetensors",
                           "layer42-expert064.safetensors"};
    for (int stage = 0; stage < 3; ++stage) {
      auto file = std::string(argv[6]) + "/" + files[stage];
      auto input = tensor(file, "m33_x");
      auto route = tensor(file, "m33_route_weight");
      auto expected = tensor(file, "m33_output");
      if (input.size() != 33 * 5120 * 2 || route.size() != 33 * 4 ||
          expected.size() != 33 * 5120 * 2)
        throw std::runtime_error("official fixture shape mismatch");
      for (int rows : {1, 3, 5, 8}) {
        std::vector<int32_t> selected(rows * 3);
        std::vector<float> weights(rows * 3);
        for (int r = 0; r < rows; ++r)
          for (int k = 0; k < 3; ++k) {
            selected[r * 3 + k] =
                (known_ids[stage] + (k + 3 - r % 3) % 3) % 128;
            std::memcpy(&weights[r * 3 + k], route.data() + r * 4, 4);
          }
        x.upload(input.data(), rows * 5120 * 2, stream);
        ids.upload(selected.data(), rows * 3 * 4, stream);
        routes.upload(weights.data(), rows * 3 * 4, stream);
        int rc = services.experts(
            services.user, stage, x.p, static_cast<int32_t *>(ids.p),
            static_cast<float *>(routes.p), out.p, rows, shared.cuda_stream);
        if (rc)
          throw std::runtime_error(services.error(services.user));
        std::vector<char> actual(rows * 3 * 5120 * 2);
        ck(cudaMemcpyAsync(actual.data(), out.p, actual.size(),
                           cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        for (int r = 0; r < rows; ++r) {
          size_t slot = r * 3 + r % 3;
          if (std::memcmp(actual.data() + slot * 5120 * 2,
                          expected.data() + r * 5120 * 2, 5120 * 2))
            throw std::runtime_error(
                "routed adapter differs from official BF16");
          values += 5120;
        }
        ++calls;
      }
    }
    // C errors must propagate and prevent any later model work on this
    // instance.
    if (!services.experts(services.user, 3, x.p, static_cast<int32_t *>(ids.p),
                          static_cast<float *>(routes.p), out.p, 1,
                          shared.cuda_stream) ||
        !*services.error(services.user))
      throw std::runtime_error("callback did not reject invalid stage");
    bool poisoned = false;
    try {
      model->verify({0});
    } catch (const std::exception &) {
      poisoned = true;
    }
    if (!poisoned)
      throw std::runtime_error("failed adapter allowed further model work");
    std::cout << Json({{"schema", "ds41-engine-spark-services-v2"},
                       {"calls", calls},
                       {"mtp_named_tensors", names.count},
                       {"official_bf16_values", values},
                       {"mismatches", 0},
                       {"fatal_error_propagation", true},
                       {"execution_scope",
                        "correctness-only; timings excluded"}})
                     .dump()
              << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
