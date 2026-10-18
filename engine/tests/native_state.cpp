#include "ds41/engine.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
using namespace ds41e;
static void ck(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
static std::vector<uint16_t> capture(Forward &model) {
  auto view = model.captures();
  if (view.abi_version != 1 || view.columns != 15360 || view.rows < 1 ||
      view.rows > 6 || view.row_stride_bytes != 30720 || !view.hidden_bf16 ||
      !view.cuda_stream)
    throw std::runtime_error("capture descriptor invalid");
  std::vector<uint16_t> data(view.rows * 15360);
  ck(cudaMemcpyAsync(data.data(), view.hidden_bf16, data.size() * 2,
                     cudaMemcpyDeviceToHost,
                     static_cast<cudaStream_t>(view.cuda_stream)));
  ck(cudaStreamSynchronize(static_cast<cudaStream_t>(view.cuda_stream)));
  return data;
}
struct Sink {
  int64_t position = 0;
  int checks = 0;
  const std::vector<std::vector<uint16_t>> *baseline;
  static int ingest(void *user, const ds41_target_capture_v1 *view) {
    auto &self = *static_cast<Sink *>(user);
    if (view->begin_position != self.position || view->committed_rows < 1 ||
        view->committed_rows > view->rows)
      return 1;
    std::vector<uint16_t> actual(view->committed_rows * 15360);
    if (cudaMemcpyAsync(actual.data(), view->hidden_bf16, actual.size() * 2,
                        cudaMemcpyDeviceToHost,
                        static_cast<cudaStream_t>(view->cuda_stream)) ||
        cudaStreamSynchronize(static_cast<cudaStream_t>(view->cuda_stream)))
      return 2;
    for (int row = 0; row < view->committed_rows; ++row) {
      const auto &expected = self.baseline->at(self.position + row);
      if (std::memcmp(actual.data() + row * 15360, expected.data(), 30720))
        return 3;
      ++self.checks;
    }
    self.position += view->committed_rows;
    return 0;
  }
};
static void equal(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() ||
      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
    throw std::runtime_error("state/rollback bit mismatch");
}
int main(int argc, char **argv) {
  try {
    if (argc < 7 || argc > 9)
      throw std::invalid_argument("test_native_state WEIGHTS RECEIPT ENGRAM "
                                  "ROPE EXPERTS SLOTS [graphs] [ENGRAM_LIB]");
    Config c;
    c.weights = argv[1];
    c.receipt = argv[2];
    c.engram = argv[3];
    c.rope = argv[4];
    c.experts = argv[5];
    c.expert_slots = std::stoull(argv[6]);
    c.max_context = 8192;
    c.allow_unqualified = true;
    auto model = make_native(c);
    const auto shared = model->shared_weights();
    if (shared.abi_version != 2 || shared.struct_bytes != sizeof(shared) ||
        shared.vocab_size != 129280 || shared.hidden_size != 5120 ||
        shared.embedding_stride_elements != 5120 ||
        shared.head_stride_elements != 5120 || !shared.embedding_bf16 ||
        !shared.head_f32 || !shared.cuda_stream || !shared.blas_handle)
      throw std::runtime_error("shared native weight descriptor invalid");
    std::vector<int> ids(142);
    ids[0] = 0;
    for (size_t i = 1; i < ids.size(); ++i)
      ids[i] = int(100 + i * 17);
    std::vector<std::vector<float>> baseline;
    std::vector<std::vector<uint16_t>> baseline_captures;
    for (int id : ids) {
      baseline.push_back(model->verify({id})[0]);
      baseline_captures.push_back(capture(*model));
      model->commit(1);
      if (baseline.size() % 32 == 0)
        std::cerr << "baseline positions=" << baseline.size() << "\n";
    }
    if (argc >= 8) {
      model.reset();
      for (int i = 7; i < argc; ++i)
        if (std::string(argv[i]) == "graphs")
          c.graphs = true;
        else
          c.engram_library = argv[i];
      model = make_native(c);
    }
    Sink sink{0, 0, &baseline_captures};
    model->set_capture_sink(Sink::ingest, &sink);
    int checks = 0;
    for (int width = 1; width <= 6; ++width) {
      model->reset();
      sink.position = 0;
      for (int i = 0; i < 130;) {
        int n = std::min(6, 130 - i);
        std::vector<int> block(ids.begin() + i, ids.begin() + i + n);
        auto out = model->verify(block);
        for (int j = 0; j < n; ++j) {
          equal(out[j], baseline[i + j]);
          ++checks;
        }
        model->commit(n);
        i += n;
      }
      std::vector<int> block(ids.begin() + 130, ids.begin() + 130 + width);
      auto out = model->verify(block);
      for (int j = 0; j < width; ++j) {
        equal(out[j], baseline[130 + j]);
        ++checks;
      }
      int accepted = width / 2;
      auto services = model->services();
      if (services.abi_version != 1 || !services.experts ||
          !services.rollback || !services.error)
        throw std::runtime_error("target service descriptor invalid");
      if (services.rollback(services.user, 130 + accepted,
                            model->shared_weights().cuda_stream))
        throw std::runtime_error(services.error(services.user));
      if (sink.position != 130 + accepted)
        throw std::runtime_error("rejected capture was ingested");
      model->rollback(130 + accepted); // Idempotent at the committed boundary.
      std::vector<int> overwritten(ids.begin() + 130 + accepted,
                                   ids.begin() + 136 + accepted);
      auto repeat = model->verify(overwritten);
      for (int j = 0; j < 6; ++j) {
        equal(repeat[j], baseline[130 + accepted + j]);
        ++checks;
      }
      model->commit(6);
      std::cerr << "width=" << width << " logit_checks=" << checks
                << " capture_checks=" << sink.checks << "\n";
    }
    std::cout << Json({{"schema", "ds41-engine-state-v1"},
                       {"bitwise_checks", checks},
                       {"mismatches", 0},
                       {"accepted_capture_checks", sink.checks},
                       {"boundary", 130},
                       {"stats", model->counters()}})
                     .dump()
              << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
