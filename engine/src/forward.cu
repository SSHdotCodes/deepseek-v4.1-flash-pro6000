#include "ds41/attention.h"
#include "ds41/blas_runtime.hpp"
#include "ds41/hotpath/kernels.cuh"
#include "ds41/hotpath/route_plan.hpp"
#include "ds41/hotpath/fused.hpp"
#include "ds41/hotpath/gate.hpp"
#include "ds41/hotpath/moe_stats.hpp"
#include "ds41/hotpath/readiness.cuh"
#include <array>
#include "ds41/dense.h"
#include "ds41/engine.hpp"
#include "ds41/engram_async.hpp"
#include "ds41/kernels.cuh"
#include "ds41/native_store.hpp"
#include "ds41/offload.hpp"
#include "ds41/ops.cuh"
#include "ds41/prefill/plan.hpp"
#include "ds41/prefill/kernels.cuh"
#include "ds41/prefill/stager.hpp"
#include "ds41/vision.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <cublas_v2.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
namespace ds41e {
namespace {
// Zero-copy byte copy on the SMs. The GPU has one copy engine per direction, so small per-layer
// metadata uploads and score/logit readbacks issued as cudaMemcpyAsync queue FIFO behind 18.8 MB
// expert uploads (H2D) and RAM demotions (D2H). Mapped pinned memory avoids both engines.
// Batched zero-copy: block i copies segment i; the segment table itself lives in mapped pinned memory.
struct ZcSeg { const unsigned char *src; unsigned char *dst; unsigned long long n; };
__global__ void zc_multi(const ZcSeg *__restrict__ segs, int count) {
  for (int s = blockIdx.x; s < count; s += gridDim.x) {
    const ZcSeg g = segs[s];
    for (size_t i = size_t(threadIdx.x) * 16; i < g.n; i += size_t(blockDim.x) * 16) {
      if (i + 16 <= g.n && !(reinterpret_cast<uintptr_t>(g.src + i) & 15) && !(reinterpret_cast<uintptr_t>(g.dst + i) & 15))
        *reinterpret_cast<uint4 *>(g.dst + i) = *reinterpret_cast<const uint4 *>(g.src + i);
      else
        for (size_t j = i; j < g.n && j < i + 16; ++j) g.dst[j] = g.src[j];
    }
  }
}
__global__ void zc_copy(const unsigned char *__restrict__ src, unsigned char *__restrict__ dst, size_t n) {
  const size_t stride = size_t(gridDim.x) * blockDim.x * 16;
  for (size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16; i < n; i += stride) {
    if (i + 16 <= n && !(reinterpret_cast<uintptr_t>(src + i) & 15) && !(reinterpret_cast<uintptr_t>(dst + i) & 15))
      *reinterpret_cast<uint4 *>(dst + i) = *reinterpret_cast<const uint4 *>(src + i);
    else
      for (size_t j = i; j < n && j < i + 16; ++j) dst[j] = src[j];
  }
}
constexpr int H = 5120, I = 2304, V = 129280, HC = 4, RING = 134;
void ck(int status, const char *op) {
  if (status)
    throw std::runtime_error(std::string(op) + ": " +
                             (status < 10000
                                  ? cudaGetErrorString(cudaError_t(status))
                                  : std::to_string(status)));
}
#define CK(op) ck(int(op), #op)
double clock_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
void vram_checkpoint(const char *label) {
  auto f=reinterpret_cast<void(*)(const char*)>(dlsym(RTLD_DEFAULT,"ds41_vram_checkpoint"));
  if (f) f(label);
}
struct Device {
  void *p = nullptr, *mapped_host = nullptr;
  size_t bytes = 0;
  Device() = default;
  explicit Device(size_t n) { resize(n); }
  ~Device() { release(); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&b) noexcept : p(b.p), mapped_host(b.mapped_host), bytes(b.bytes) {
    b.p = b.mapped_host = nullptr; b.bytes = 0;
  }
  Device &operator=(Device &&b) noexcept {
    if (this != &b) {
      release(); p=b.p; mapped_host=b.mapped_host; bytes=b.bytes;
      b.p=b.mapped_host=nullptr; b.bytes=0;
    }
    return *this;
  }
  void release() noexcept {
    if (mapped_host) cudaFreeHost(mapped_host);
    else if (p) cudaFree(p);
    p=mapped_host=nullptr; bytes=0;
  }
  void resize(size_t n) {
    release();
    if (n) { CK(cudaMalloc(&p,n)); bytes=n; }
  }
  void resize_mapped(size_t n) {
    release();
    if (n) {
      CK(cudaHostAlloc(&mapped_host,n,cudaHostAllocMapped));
      auto rc=cudaHostGetDevicePointer(&p,mapped_host,0);
      if (rc!=cudaSuccess) { release(); CK(rc); }
      bytes=n;
    }
  }
  template <class T = void> T *as(size_t offset = 0) const {
    return reinterpret_cast<T *>(static_cast<char *>(p) + offset);
  }
};
struct Weight {
  Device data;
  std::vector<uint64_t> shape;
  std::string dtype;
};
struct Global {
  Device kv, key, raw_kv, raw_score;
  int ratio, capacity;
  Global(int max, int r)
      : kv(size_t(max / r) * 288), key(size_t(max / r) * 68),
        raw_kv(8 * 512 * 4), raw_score(8 * 512 * 4), ratio(r),
        capacity(max / r) {}
};
struct Phase {
  cudaEvent_t start = nullptr, end = nullptr;
  explicit Phase() {
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&end));
  }
  ~Phase() {
    if (start)
      cudaEventDestroy(start);
    if (end)
      cudaEventDestroy(end);
  }
};
struct Graph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ~Graph() {
    if (exec)
      cudaGraphExecDestroy(exec);
    if (graph)
      cudaGraphDestroy(graph);
  }
};
Store verified_store(const Config &c) {
  std::cerr << "native runtime " << native_blas_runtime().dump() << '\n';
  Store source(c.weights);
  source.verify_receipt(c.receipt);
  std::map<std::string, std::string> role_paths = {
      {"weight_index", c.weights + "/model.safetensors.index.json"},
      {"config", c.weights + "/config.json"},
      {"inference_config", c.weights + "/inference/config.json"},
      {"tokenizer", c.weights + "/tokenizer.json"},
      {"official_model", c.weights + "/inference/model.py"},
      {"official_kernel", c.weights + "/inference/kernel.py"},
      {"official_engram", c.weights + "/inference/engram.py"},
      {"official_generate", c.weights + "/inference/generate.py"},
      {"engram_export", c.engram},
      {"rope_export", c.rope},
      {"rope_metadata", c.rope + ".json"},
      {"expert_manifest", c.experts}};
  if (!c.engram_library.empty())
    role_paths.emplace("engram_library", c.engram_library);
  auto auxiliary = authenticate_files(
      c.auxiliary.empty() ? DS41_DEFAULT_AUXILIARY : c.auxiliary,
      DS41_AUX_SHA256, role_paths);
  if (auxiliary.at("weight_root") != c.weights ||
      auxiliary.at("revision") != "1bae0e1088993b9b82de08f59de5be5a34e42170")
    throw std::runtime_error("auxiliary checkpoint identity mismatch");
  auto engram = read_json(c.engram);
  for (const auto &entry : {std::pair<const char *, const char *>{
                                "inference/engram.py", "official_engram"},
                            {"inference/config.json", "inference_config"},
                            {"tokenizer.json", "tokenizer"}})
    if (engram.at("provenance").at(entry.first) !=
        auxiliary.at("files").at(entry.second).at("sha256"))
      throw std::runtime_error("Engram export provenance mismatch");
  auto rope = read_json(c.rope + ".json");
  auto rows = rope.at("rows").get<int64_t>();
  if (rope.at("schema") != "ds41-rope-v1" || rows < c.max_context ||
      rows > 1048576 || rope.at("dtype") != "F32" ||
      rope.at("layout") != "[2,rows,32,2] pure-SWA then compressed" ||
      rope.at("source_sha256") !=
          auxiliary.at("files").at("official_model").at("sha256") ||
      rope.at("sha256") !=
          auxiliary.at("files").at("rope_export").at("sha256") ||
      std::filesystem::file_size(c.rope) != uint64_t(rows) * 2 * 64 * 4)
    throw std::runtime_error("RoPE export provenance/geometry mismatch");
  if (!c.allow_unqualified)
    throw std::runtime_error(
        "M1 native forward is unqualified; full-model gold gate pending (use "
        "--allow-unqualified for development only)");
  return source;
}
class Native final : public Forward, public hotpath::GateEvidence {
  Config config;
  Store source;
  EngramHash hash;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  std::map<std::string, Weight> weights;
  std::unique_ptr<ds41::OffloadStore> experts;
  std::unique_ptr<hotpath::FusedOps> fused;
  std::unique_ptr<ds41::DirectReader> reader;
  std::unique_ptr<ds41::HostBuffer> pages, host;
  std::unique_ptr<EngramAsync> async_engram;
  std::array<std::unique_ptr<EngramAsync::Ticket>, 2> engram_tickets;
  Json initial_engram_stats, host_cache_policy;
  Json prefill_summary=Json::object();
  std::map<int, std::unique_ptr<Global>> globals;
  std::vector<Device> windows;
  std::vector<std::unique_ptr<Phase>> phases;
  std::map<std::string, std::unique_ptr<Graph>> graphs;
  uint64_t graph_replays = 0, initial_replays = 0;
  uint64_t graph_captures = 0, initial_captures = 0;
  uint64_t graph_evictions = 0, initial_evictions = 0;
  std::array<std::array<int,40>,6> attention_scan_bucket{};
  bool graph_ready = false;
  Device h{HC * H * 2}, next_h{HC * H * 2}, x{H * 2}, y{H * 2}, f32{HC * H * 4};
  Device pre{4 * 4}, att_pre{4 * 4}, ffn_pre{4 * 4}, post{4 * 4}, comb{16 * 4},
      mixes{24 * 4}, rstd{4};
  Device qa{1280 * 2}, q{64 * 512 * 2}, kv{512 * 2}, latent{512 * 2},
      ik{128 * 2}, iq{32 * 128 * 2}, iw{32 * 2};
  Device packed{528}, gather{640 * 512 * 2}, local_idx{640 * 4},
      o{64 * 512 * 2}, wo{8192 * 2}, attn_scores{64 * 640 * 4};
  Device aq{32768}, as{1024}, gate{6 * I * 2}, up{6 * I * 2}, act{6 * I * 2},
      eq{6 * I}, es{6 * I / 32}, down{6 * H * 2}, shared{H * 2};
  Device sg{I * 2}, su{I * 2}, sa{I * 2}, jobs{6 * sizeof(ds41::GemmJob)},
      up_jobs{6 * sizeof(ds41::GemmJob)}, down_jobs{6 * sizeof(ds41::GemmJob)},
      ids_device{6 * 4}, route{6 * 4};
  Device router{384 * 4}, unbiased{384 * 4}, biased{384 * 4}, logits{V * 4};
  Device scores, top{512 * 4}, visible{4}, position_device{8}, blocks,
      block_visible{4}, chosen_blocks{2048 * 4}, candidate_ids{16384 * 4},
      candidate_top{512 * 4}, top_scratch;
  Device rows_raw{24 * 256}, scales_raw{24 * 8}, rows_bf{24 * 256 * 2},
      engram_kv{(HC + 1) * H * 2}, rows_interleaved{24 * 264};
  Device braw_kv{6*512*4}, braw_score{6*512*4}, batch_tokens{6*4};
  Device route_source{36*4}, route_slots{36*4}, bx{36*H}, bxs{36*H/32}, broute{36*4};
  Device compact_down{36*H*2}, sparse_scores{6*64*640*4};
  Device work_jobs{36*sizeof(ds41::GemmJob)}, work_up_jobs{36*sizeof(ds41::GemmJob)},
      work_down_jobs{36*sizeof(ds41::GemmJob)}, work_rows{36*4};
  std::array<std::vector<ds41::ExpertId>,40> previous_routes;
  std::vector<ds41::OffloadStore::PrefetchTicket> prefetch_tickets;
  // --prefetch-lookahead scratch. lookahead_sync->start: route scores on host; ->end: predicted
  // scores for lookahead_layer on host.
  Device lookahead_router{8 * 384 * 4}, lookahead_unbiased{8 * 384 * 4}, lookahead_biased{8 * 384 * 4};
  int lookahead_m = 1, lookahead_n = 4;
  std::unique_ptr<ds41::HostBuffer> spark_host;
  const bool prefill_chunk_fit = [] { const char *v = std::getenv("DS41_PREFILL_CHUNK_FIT"); return v && std::string(v) == "1"; }();
  std::unique_ptr<ds41::HostBuffer> lookahead_host;
  std::unique_ptr<Phase> lookahead_sync;
  int lookahead_layer = -1, lookahead_k = 6;
  // DS41_LOOKAHEAD_AUDIT=path: per decode layer (m==1), each routed expert's tier at demand
  // (0 VRAM, 1 RAM, 2 NVMe) and its rank in the router lookahead made 1..3 layers earlier.
  // DS41_LOOKAHEAD_DRYRUN=1 scores without prefetching. Diagnostics only.
  std::ofstream lookahead_audit;
  bool lookahead_dryrun = false;
  int64_t audit_pos = 0;
  std::array<std::array<std::vector<int>, 3>, 40> audit_pred;
  // --verify-lanes (batch_lanes): lane B's spare scratch, stream, cuBLAS handle and step state are
  // swapped into the members while its work is enqueued; route_events[lane] marks score copies.
  bool lane_b_active = false;
  cudaStream_t lane_stream = nullptr;
  cublasHandle_t lane_blas = nullptr;
  std::vector<std::pair<Device *, Device>> lane_slots;
  int lane_candidate_count = 0;
  std::array<std::array<int,40>,6> lane_scan_bucket{};
  std::array<std::unique_ptr<EngramAsync::Ticket>, 2> lane_engram_tickets;
  std::array<cudaEvent_t, 2> route_events{};
  std::array<cudaEvent_t, 40> lane_att_events{};
  cudaEvent_t lane_entry = nullptr, lane_exit = nullptr;
  using FusedHead = int (*)(const float *, const void *, float *, int, int, int, cudaStream_t);
  void *fused_head_handle = nullptr;
  FusedHead fused_head = nullptr;
  // Target logits for m FP32 rows from the native BF16 head (--fused-head only).
  void fused_head_logits(const void *in, void *out, int m) {
    CK(fused_head(static_cast<const float *>(in), w("head.weight").data.p, static_cast<float *>(out), m, V, H, stream));
  }
  std::map<ds41::ExpertId, int> prefill_warm_counts;
  std::vector<std::future<void>> retirement_futures;
  size_t retirement_observed=0;
  uint64_t stream_callbacks=0, stream_hit_batches=0;
  Device batch_frequencies{12*64*4}, compressed_frequencies{6*64*4},
      lengths1{6*4}, lengths2{6*4}, packed_keys{6*68}, packed_globals{6*288};
  bool hotpath_capturing=false;
  std::vector<std::unique_ptr<Phase>> expert_events;
  size_t expert_event_index=0;
  std::unique_ptr<hotpath::ConsumerReadiness> consumer_readiness;
  double host_plan_ms=0, callback_enqueue_ms=0;
  Json profile_steps=Json::array(), step_demands=Json::array();
  Device gate_slab;
  std::vector<uint8_t> gate_host;
  std::vector<std::vector<uint8_t>> gate_rows;
  Device rope, woa_scratch;
  // Image prompts: the vision tower (built at the first image) and the IMAGE_START, IMAGE_NEW_LINE and
  // IMAGE_END embeddings, in that order.
  std::unique_ptr<VisionEncoder> vision;
  Device image_delimiters;
  Device capture_slab{6 * 3 * H * 2};
  int capture_slot = 0, capture_rows = 0, capture_committed = 0;
  int64_t capture_begin = 0;
  ds41_target_capture_sink capture_sink = nullptr;
  void *capture_user = nullptr;
  std::string callback_error;
  Device spark_aq, spark_as, spark_gate, spark_up, spark_act, spark_eq,
      spark_es, spark_jobs, spark_up_jobs, spark_down_jobs;
  int64_t committed = 0;
  int pending = 0;
  ds41e::prefill::Prefix prefix;
  std::vector<int> pending_ids;
  int shared_topk = 0, candidate_count = 0, eos_id = 1;
  ds41::Stats initial_stats{};
  mutable ds41::Stats last_expert_stats{};
  mutable bool expert_stats_available=false;
  double io_ms = 0, route_ms = 0, engram_ms = 0, total_ms = 0;
  size_t phase_index = 0;
  size_t upload_cursor = 1 << 20;
  bool in_step = false;
  bool poisoned = false;
  struct PrefillData;
  std::unique_ptr<PrefillData> pref;
  std::unique_ptr<ds41e::prefill::Stager> prefill_stage;
  ds41e::prefill::TransferStats prefill_transfers;
  size_t prefill_vram_hits=0;
  std::vector<ds41::Record> prefill_records;
  std::map<std::string, double> measured;

#include "prefill/native.inc"
public:
  const std::vector<std::vector<uint8_t>> &hotpath_gate_rows() const override {
    return gate_rows;
  }
  explicit Native(const Config &c)
      : config(c), source(verified_store(c)), hash(c.engram) {
    // Every payload read, CUDA allocation, and host pin is after the full
    // receipt. This first engine is an N candidate until full fidelity gates
    // establish B0.
#if DS41_MOE_SOURCE_API<4
    if(c.dynamic_host || c.async_retire || c.reset_cache_history || c.host_recent_window!=-1 || c.host_decay_interval!=-1)
      throw std::invalid_argument("dynamic host/policy/history/async retirement require a jointly pinned MoE API4 build");
#else
    if(c.async_retire) retirement_futures.reserve(240);
#endif
    if (c.expert_slots < 6 || c.expert_slots > 5000)
      throw std::invalid_argument("expert slots must be 6..5000");
    eos_id =
        read_json(c.weights + "/config.json").at("eos_token_id").get<int>();
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (auto &e : route_events) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    vram_checkpoint("native_members_before_cublas");
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasCreate");
    vram_checkpoint("native_after_cublas");
    if (c.cuda_stack_bytes) {
      if (c.cuda_stack_bytes > 65536) throw std::invalid_argument("CUDA stack request exceeds bounded candidate limit");
      CK(cudaDeviceSetLimit(cudaLimitStackSize, c.cuda_stack_bytes));
    }
    vram_checkpoint("native_after_stack_limit");
    if(c.hotpath_gate) {
      gate_slab.resize(6*hotpath::GateLayout::row_bytes);
      gate_host.resize(6*hotpath::GateLayout::row_bytes);
    }
    // Six-row verify arena. The same addresses serve every width and graph.
    for (Device *d : {&h,&next_h,&x,&y,&f32,&pre,&att_pre,&ffn_pre,&post,&comb,
         &mixes,&rstd,&qa,&q,&kv,&latent,&ik,&iq,&iw,&packed,&gather,&local_idx,
         &o,&wo,&aq,&as,&gate,&up,&act,&eq,&es,&down,&shared,&sg,&su,&sa,
         &jobs,&up_jobs,&down_jobs,&ids_device,&route,&router,&unbiased,&biased,
         &logits,&top,&visible,&block_visible,&chosen_blocks,&candidate_ids,
         &candidate_top,&rows_raw,&scales_raw,&rows_bf,&engram_kv,&rows_interleaved})
      d->resize(d->bytes * 6);
    host = std::make_unique<ds41::HostBuffer>(16 << 20, true);
    pages = std::make_unique<ds41::HostBuffer>(1 << 20, true);
    if (!c.fused_head_library.empty()) {
      // Spark drafts with its compact BF16 head over this same native matrix (head_kind 1).
      fused_head_handle = dlopen(c.fused_head_library.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!fused_head_handle) throw std::runtime_error(std::string("fused head: ") + dlerror());
      fused_head = reinterpret_cast<FusedHead>(dlsym(fused_head_handle, "ds41_fuse_head_decode"));
      if (!fused_head) throw std::runtime_error("fused head: missing ds41_fuse_head_decode");
    }
    if (c.prefetch_lookahead) {
      if (!c.streamed_experts)
        throw std::invalid_argument("--prefetch-lookahead requires --streamed-experts");
      if (const char *k = std::getenv("DS41_LOOKAHEAD_K")) lookahead_k = std::stoi(k);
      if (lookahead_k < 1 || lookahead_k > 32)
        throw std::invalid_argument("DS41_LOOKAHEAD_K must be 1..32");
      lookahead_host = std::make_unique<ds41::HostBuffer>(8 * 384 * 4, true);
      if (const char *n = std::getenv("DS41_LOOKAHEAD_N")) lookahead_n = std::stoi(n);
      if (lookahead_n < 1 || lookahead_n > 48) throw std::invalid_argument("DS41_LOOKAHEAD_N must be 1..48");
      if (const char *path = std::getenv("DS41_LOOKAHEAD_AUDIT")) {
        lookahead_audit.open(path);
        if (!lookahead_audit) throw std::runtime_error("lookahead audit open");
      }
      if (const char *dry = std::getenv("DS41_LOOKAHEAD_DRYRUN")) lookahead_dryrun = std::string(dry) == "1";
      lookahead_sync = std::make_unique<Phase>();
    }
    reader = std::make_unique<ds41::DirectReader>(64);
    load_weights();
    if (fused_head) {
      const auto &head = w("head.weight");
      if (head.dtype != "BF16" || head.shape != std::vector<uint64_t>({uint64_t(V), uint64_t(H)}))
        throw std::runtime_error("--fused-head requires the native BF16 [vocab, hidden] head");
    }
    vram_checkpoint("native_after_weights");
    load_rope();
    vram_checkpoint("native_after_rope");
    if(!c.fuse_library.empty()) fused=std::make_unique<hotpath::FusedOps>(c.fuse_library);
    if (!c.engram_library.empty())
      async_engram = std::make_unique<EngramAsync>(c.engram_library, c);
    for (int l = 0; l < 40; ++l)
      windows.emplace_back(RING * 528);
    for (int l : {2, 8, 14, 20})
      globals.emplace(
          l, std::make_unique<Global>(config.max_context, l == 20 ? 1 : 2));
    scores.resize(size_t(std::max(16384, config.max_context + 8)) * 2 * 6);
    blocks.resize(size_t((config.max_context + 7) / 8) * 2 * 6);
    top_scratch.resize(
        ds41_attn_topk_scratch(6, std::max(16384, config.max_context + 8)));
    vram_checkpoint("native_after_kv");
    auto records = ds41::load_records(c.experts);
    if (records.size() != ds41::expert_count)
      throw std::runtime_error(
          "expert manifest must include all 15744 logical ids");
    // The manifest is not trusted to select different weights: every tensor
    // view must land on the indexed native tensor in this verified checkpoint.
    validate_records(records);
    if(c.prefill)prefill_records=records;
    ds41::OffloadStore::Config ec;
    ec.slots = c.expert_slots;
    ec.host_budget_bytes = c.host_bytes;
    ec.staging_slots = 8;
    ec.io_depth = 64;
#if DS41_MOE_SOURCE_API>=4
    ec.dynamic_host=c.dynamic_host;
    if(c.host_recent_window < -1 || c.host_decay_interval < -1)
      throw std::invalid_argument("invalid host cache policy");
    if(c.host_recent_window>=0) ec.host_recent_window=unsigned(c.host_recent_window);
    if(c.host_decay_interval>=0) ec.host_decay_interval=unsigned(c.host_decay_interval);
    host_cache_policy={{"recent_window",ec.host_recent_window},{"decay_interval",ec.host_decay_interval}};
#endif
    experts = std::make_unique<ds41::OffloadStore>(std::move(records), ec);
    vram_checkpoint("native_after_store");
    if (!c.placement.empty())
      experts->place(ds41::load_placement(c.placement));
#if DS41_MOE_SOURCE_API>=4
    if(c.reset_cache_history) experts->reset_history();
#endif
    for (int j = 0; j < 200; ++j)
      phases.push_back(std::make_unique<Phase>());
    if(c.hotpath_profile) {
      for(int i=0;i<1440;++i) expert_events.push_back(std::make_unique<Phase>());
      consumer_readiness=std::make_unique<hotpath::ConsumerReadiness>();
    }
    reset();
    std::cerr << "ds41d native M1 loaded; UNQUALIFIED; expert slots="
              << c.expert_slots << "\n";
  }
  ~Native() override {
    if (stream)
      cudaStreamSynchronize(stream);
    if (lane_stream)
      cudaStreamSynchronize(lane_stream);
    for (auto &ticket : engram_tickets)
      ticket.reset();
    for (auto &ticket : lane_engram_tickets)
      ticket.reset();
    for(auto &f:retirement_futures) if(f.valid()) {
      try { f.get(); } catch(...) { /* verify already poisons/reports errors */ }
    }
    retirement_futures.clear();
    async_engram.reset();
    graphs.clear(); // Captured consumers must die before the store/table.
    fused.reset();
    experts.reset();
    reader.reset();
    pages.reset();
    host.reset();
    if (blas)
      cublasDestroy(blas);
    if (lane_blas)
      cublasDestroy(lane_blas);
    lane_slots.clear();
    for (auto e : lane_att_events) if (e) cudaEventDestroy(e);
    for (auto e : route_events) if (e) cudaEventDestroy(e);
    if (lane_entry) cudaEventDestroy(lane_entry);
    if (lane_exit) cudaEventDestroy(lane_exit);
    if (lane_stream)
      cudaStreamDestroy(lane_stream);
    if (stream)
      cudaStreamDestroy(stream);
  }
  int vocab() const override { return V; }
  int eos() const override { return eos_id; }
  ds41_target_weights_v2 shared_weights() const override {
    return {2,
            sizeof(ds41_target_weights_v2),
            V,
            H,
            H,
            H,
            weights.at("embed.weight").data.p,
            weights.at("head.weight").data.as<float>(),
            stream,
            blas,
            rope.as<float>(),
            config.max_context};
  }
  ds41_target_services_v1 services() override {
    return {1, this,
            [](void *user, int stage, const void *input, const int32_t *ids,
               const float *weights, void *output, int rows, void *s) -> int {
              auto *self = static_cast<Native *>(user);
              return self->service_call([&] {
                self->routed_spark(stage, input, ids, weights, output, rows, s);
              });
            },
            [](void *user, int64_t position, void *s) -> int {
              auto *self = static_cast<Native *>(user);
              return self->service_call([&] {
                if (s != self->stream)
                  throw std::invalid_argument(
                      "target rollback stream mismatch");
                self->rollback(position);
              });
            },
            [](void *user) -> const char * {
              return static_cast<Native *>(user)->callback_error.c_str();
            }};
  }
  ds41_target_tensor_v1
  resident_tensor(const std::string &name) const override {
    ds41_target_tensor_v1 view{};
    view.abi_version = 1;
    view.struct_bytes = sizeof(view);
    if (name == "rope") {
      view.dtype = DS41_TARGET_F32;
      view.rows = config.max_context;
      view.columns = 64;
      view.data = rope.p;
      return view;
    }
    auto found = weights.find(name);
    if (found == weights.end())
      return {};
    const auto &t = found->second;
    if (t.shape.empty() || t.shape.size() > 2)
      throw std::runtime_error("resident tensor rank unsupported: " + name);
    view.rows = int64_t(t.shape[0]);
    view.columns = t.shape.size() == 2 ? int64_t(t.shape[1]) : 1;
    view.dtype =
        t.dtype == "BF16"                                  ? DS41_TARGET_BF16
        : t.dtype == "F32"                                 ? DS41_TARGET_F32
        : (t.dtype == "F8_E4M3" || t.dtype == "F8_E4M3FN") ? DS41_TARGET_FP8
        : (t.dtype == "U8" || t.dtype == "F8_E8M0" || t.dtype == "F8_E8M0FNU")
            ? DS41_TARGET_U8
            : -1;
    if (view.dtype < 0)
      throw std::runtime_error("resident tensor dtype unsupported: " + name);
    view.data = t.data.p;
    return view;
  }
  std::vector<std::string>
  resident_names(const std::string &prefix) const override {
    std::vector<std::string> names;
    for (const auto &t : weights)
      if (t.first.compare(0, prefix.size(), prefix) == 0)
        names.push_back(t.first);
    if (std::string("rope").compare(0, prefix.size(), prefix) == 0)
      names.push_back("rope");
    return names;
  }
  ds41_target_capture_v1 captures() const override {
    return {1,
            capture_rows,
            3 * H,
            capture_committed,
            capture_begin,
            size_t(3 * H * 2),
            capture_slab.p,
            stream};
  }
  void set_capture_sink(ds41_target_capture_sink sink, void *user) override {
    capture_sink = sink;
    capture_user = user;
  }
  void rollback(int64_t position) override {
    if (!pending) {
      if (position != committed)
        throw std::invalid_argument("rollback beyond pending verify is unsafe");
      return;
    }
    if (position < committed || position > committed + pending)
      throw std::invalid_argument("rollback outside pending verify");
    commit(size_t(position - committed));
  }
  uint64_t prefetch_experts(const std::vector<uint32_t> &ids, PrefetchIntent intent) override {
    for(auto id:ids) if(id>=ds41::expert_count) throw std::invalid_argument("prefetch logical expert id");
    try {
      for(auto it=prefetch_tickets.begin();it!=prefetch_tickets.end();) {
        if(it->completion.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
          it->completion.get(); it=prefetch_tickets.erase(it);
        } else ++it;
      }
      auto priority=intent==PrefetchIntent::NextLayer?ds41::OffloadStore::PrefetchPriority::NextLayer:
          intent==PrefetchIntent::Draft?ds41::OffloadStore::PrefetchPriority::Draft:
          ds41::OffloadStore::PrefetchPriority::Session;
      auto ticket=experts->prefetch(ids,priority);
      auto id=ticket.id; prefetch_tickets.push_back(std::move(ticket)); return id;
    } catch(...) { poisoned=true; throw; }
  }
  // RAM-resident subset only: a wrong guess costs one host-to-device copy, never an NVMe read.
  void prefetch_ram_experts(const std::vector<ds41::ExpertId> &ids) {
    try {
      for(auto it=prefetch_tickets.begin();it!=prefetch_tickets.end();) {
        if(it->completion.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
          it->completion.get(); it=prefetch_tickets.erase(it);
        } else ++it;
      }
      prefetch_tickets.push_back(experts->prefetch_ram(ids,ds41::OffloadStore::PrefetchPriority::NextLayer));
    } catch(...) { poisoned=true; throw; }
  }
  void cancel_prefetch(uint64_t id) override {
    for(const auto &ticket:prefetch_tickets) if(ticket.id==id) experts->cancel(ticket);
  }
  void begin_request() override {
    if (poisoned)
      throw std::runtime_error("native forward/store poisoned; restart ds41d "
                               "before further model work");
    if (pending)
      throw std::logic_error("begin request during active verify");
    // Lookahead prefetch is per token. Settle it so served append prefill, which
    // refuses outstanding prefetch, stays available for the next prompt.
    if (lookahead_sync) {
      for (auto &ticket : prefetch_tickets) {
        experts->cancel(ticket);
        ticket.completion.wait();
      }
      prefetch_tickets.clear();
    }
    io_ms = route_ms = engram_ms = total_ms = 0;
    measured.clear();prefill_summary=Json::object();
    host_plan_ms=callback_enqueue_ms=0; profile_steps=Json::array();
    stream_callbacks=stream_hit_batches=0;
    initial_stats = sample_expert_stats();
    if (async_engram)
      initial_engram_stats = async_engram->stats();
    initial_replays = graph_replays;
    initial_captures = graph_captures;
    initial_evictions = graph_evictions;
  }
  struct NativeSnapshot : Snapshot {
    std::vector<Device> windows;
    std::map<int, std::pair<Device, Device>> raw;
    std::vector<int> prefix_ids;
  };
  std::shared_ptr<Snapshot> snapshot() override {
    if (poisoned || pending)
      throw std::logic_error("snapshot requires a committed, healthy state");
    auto s = std::make_shared<NativeSnapshot>();
    s->position = committed;
    for (auto &w : windows) {
      s->windows.emplace_back(w.bytes);
      CK(cudaMemcpyAsync(s->windows.back().p, w.p, w.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    for (auto &[l, g] : globals) {
      auto &r = s->raw[l];
      r.first.resize(g->raw_kv.bytes); r.second.resize(g->raw_score.bytes);
      CK(cudaMemcpyAsync(r.first.p, g->raw_kv.p, g->raw_kv.bytes, cudaMemcpyDeviceToDevice, stream));
      CK(cudaMemcpyAsync(r.second.p, g->raw_score.p, g->raw_score.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    s->prefix_ids = prefix.ids();
    CK(cudaStreamSynchronize(stream));
    return s;
  }
  void restore(const Snapshot &base) override {
    auto *s = dynamic_cast<const NativeSnapshot *>(&base);
    if (!s || s->windows.size() != windows.size() || s->position > committed ||
        int64_t(s->prefix_ids.size()) != s->position)
      throw std::invalid_argument("restore: incompatible snapshot");
    begin_request();
    for (size_t i = 0; i < windows.size(); ++i)
      CK(cudaMemcpyAsync(windows[i].p, s->windows[i].p, windows[i].bytes, cudaMemcpyDeviceToDevice, stream));
    for (auto &[l, g] : globals) {
      auto &r = s->raw.at(l);
      CK(cudaMemcpyAsync(g->raw_kv.p, r.first.p, g->raw_kv.bytes, cudaMemcpyDeviceToDevice, stream));
      CK(cudaMemcpyAsync(g->raw_score.p, r.second.p, g->raw_score.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    committed = s->position;
    gate_rows.clear();
    for (auto &ticket : prefetch_tickets) experts->cancel(ticket);
    prefetch_tickets.clear();
    for (auto &routes : previous_routes) routes.clear();
    hash.truncate(size_t(s->position));
    prefix.reset(); prefix.commit(s->prefix_ids, s->prefix_ids.size()); pending_ids.clear();
    capture_rows = capture_committed = 0; capture_begin = 0;
    CK(cudaStreamSynchronize(stream));
  }
  void reset() override {
    begin_request();
    committed = 0;
    gate_rows.clear();
    for(auto &ticket:prefetch_tickets) experts->cancel(ticket);
    prefetch_tickets.clear();
    for(auto &routes:previous_routes) routes.clear();
    hash.reset();
    prefix.reset();pending_ids.clear();
    capture_rows = capture_committed = 0;
    capture_begin = 0;
    for (auto &w : windows)
      CK(cudaMemsetAsync(w.p, 0, w.bytes, stream));
    // Position-addressed global bytes need no clearing. Visibility masks and
    // completed compression groups prevent any future record from being read.
    CK(cudaStreamSynchronize(stream));
  }
  std::vector<std::vector<float>>
  verify(const std::vector<int> &input) override {
    if (poisoned)
      throw std::runtime_error("native forward/store poisoned; restart ds41d");
    if (pending || input.empty() || input.size() > 6 ||
        committed + int64_t(input.size()) > config.max_context)
      throw std::invalid_argument("verify state/width/context");
    for (int t : input)
      if (t < 0 || t >= V)
        throw std::invalid_argument("input id");
    pending = int(input.size());
    pending_ids = input;
    capture_begin = committed;
    capture_rows = pending;
    capture_committed = 0;
    std::vector<std::vector<float>> out;
    const double verify_begin=config.hotpath_profile?clock_ms():0;
    try {
      check_moe_retirement(false);
      if (config.batched_verify)
        out = config.verify_lanes && input.size() >= 2 ? batch_lanes(input, committed) : batch(input, committed);
      else for (size_t i = 0; i < input.size(); ++i) {
        capture_slot = int(i);
        out.push_back(one(input[i], committed + i));
      }
      const double retire_begin=config.hotpath_profile?clock_ms():0;
      check_moe_retirement(true);
      if(config.hotpath_profile && config.batched_verify && !profile_steps.empty()) {
        auto &step=profile_steps.back();
        step["verify_wall_ms"]=clock_ms()-verify_begin;
        step["retirement_registration_host_ms"]=clock_ms()-retire_begin;
      }
    } catch (...) {
      poisoned = true;
      throw;
    }
    return out;
  }
  void commit(size_t accepted) override {
    if (poisoned)
      throw std::runtime_error("native forward/store poisoned; restart ds41d");
    if (!pending || accepted > size_t(pending))
      throw std::invalid_argument("accepted input prefix");
    prefix.commit(pending_ids,accepted);pending_ids.clear();
    committed += accepted;
    capture_committed = int(accepted);
    pending = 0;
    if (accepted && capture_sink) {
      try {
        auto view = captures();
        int rc = capture_sink(capture_user, &view);
        if (rc)
          throw std::runtime_error("DSpark capture sink failed: " +
                                   std::to_string(rc));
      } catch (...) {
        poisoned = true;
        throw;
      }
    }
  }
  DemandCounters demand_counters() const override {
    const auto s = sample_expert_stats();
    if(s.host_hits < initial_stats.host_hits ||
       s.nvme_misses < initial_stats.nvme_misses)
      throw std::runtime_error("target tier counter moved backwards");
    return {double(s.host_hits - initial_stats.host_hits),
            double(s.nvme_misses - initial_stats.nvme_misses)};
  }
  Json counters() const override {
    ds41::Stats s{};bool stale=false;std::string stats_error;
    try { s=sample_expert_stats(); }
    catch(const std::exception &e) {
      if(!expert_stats_available) throw;
      s=last_expert_stats;stale=true;stats_error=e.what();
    }
    Json j = {{"engine_mode", config.batched_verify ? "native-layer-major-verify-UNQUALIFIED" : "native-sequential-M1-UNQUALIFIED"},
              {"committed", committed},
              {"cache_epoch", prefix.epoch()},
              {"cache_committed_tokens", prefix.size()},
              {"expert_vram_hits", s.hits - initial_stats.hits},
              {"expert_host_hits", s.host_hits - initial_stats.host_hits},
              {"expert_nvme_misses", s.nvme_misses - initial_stats.nvme_misses},
              {"expert_h2d_bytes", s.h2d_bytes - initial_stats.h2d_bytes},
              {"expert_nvme_bytes", s.nvme_bytes - initial_stats.nvme_bytes},
              {"need_host_ms", io_ms},
              {"route_wait_ms", route_ms},
              {"engram_read_ms", engram_ms},
              {"forward_ms", total_ms},
              {"stream_phase_ms_including_stalls", measured}};
    {
      // DSpark drafter (MTP layers 40..42) share of the store traffic.
      uint64_t req=0,hit=0,host=0,nvme=0;
      for(int l=40;l<43;++l){
        const auto &a=s.layers[l], &b=initial_stats.layers[l];
        req+=a.requests-b.requests; hit+=a.hits-b.hits; host+=a.host_hits-b.host_hits; nvme+=a.nvme_misses-b.nvme_misses;
      }
      j["mtp_expert_requests"]=req; j["mtp_expert_vram_hits"]=hit;
      j["mtp_expert_host_hits"]=host; j["mtp_expert_nvme_misses"]=nvme;
    }
    j["fuse_library"]=config.fuse_library;
    j["moe_source_api"]=DS41_MOE_SOURCE_API;
    j["moe_stats_snapshot"]=stale?"last successful snapshot after error":"fresh";
    if(stale) j["moe_stats_error"]=stats_error;
    j["cpu_only_experts"]=0; // CPU backend is not integrated in this adapter.
    j["expert_kernel"]="grouped_mxfp4_native_parallel";
    j["xk_enabled"]=false; // Cache-only adapter never constructs DecodeBackend/XK.
    j["moe_demand_grouping"]="sorted unique logical-ID union per layer/verify";
    j["dynamic_host"]=config.dynamic_host;j["async_retire"]=config.async_retire;
    j["host_cache_policy"]=host_cache_policy;
    j["cache_history_initialization"]=config.reset_cache_history?
        "reset_history after placement; contents and pins retained":"placement seed histories retained";
    j["moe_delta"]=hotpath::moe_delta(initial_stats,s);
    j["hotpath_profile"]=config.hotpath_profile;
    j["step_profile"]=profile_steps;
    j["host_route_plan_ms"]=host_plan_ms;
    j["callback_enqueue_ms"]=callback_enqueue_ms;
    Json layer_stats=Json::array();
    for(int l=0;l<43;++l) {
      const auto &a=s.layers[l], &b=initial_stats.layers[l];
      layer_stats.push_back({{"layer",l},{"requests",a.requests-b.requests},
          {"vram_hits",a.hits-b.hits},{"host_hits",a.host_hits-b.host_hits},
          {"nvme_misses",a.nvme_misses-b.nvme_misses},{"stall_ns",a.stall_ns-b.stall_ns},
          {"h2d_bytes",a.h2d_bytes-b.h2d_bytes},{"nvme_bytes",a.nvme_bytes-b.nvme_bytes}});
    }
    j["expert_layers"]=layer_stats;
    j["streamed_experts"]=config.streamed_experts;
    j["stream_callbacks"]=stream_callbacks;
    j["stream_hit_batches"]=stream_hit_batches;
    j["prefetch_previous_routes"]=config.prefetch_routes;
    j["graphs_requested"] = config.graphs;
    j["prefill"] = prefill_summary;
    if (!config.hotpath_profile)
      j["prefill"].erase("layers");
    j["auxiliary_receipt_sha256"] = DS41_AUX_SHA256;
    j["mtp_loaded"] = config.load_mtp;
    uint64_t resident_bytes = 0, mtp_bytes = 0;
    for (const auto &item : weights) {
      resident_bytes += item.second.data.bytes;
      if (item.first.compare(0, 4, "mtp.") == 0)
        mtp_bytes += item.second.data.bytes;
    }
    j["resident_weight_bytes"] = resident_bytes;
    if (config.report_vram) j["vram_inventory"] = vram_inventory();
    j["mtp_resident_bytes"] = mtp_bytes;
    j["graph_count"] = graphs.size();
    j["graph_captures"] = graph_captures - initial_captures;
    j["graph_evictions"] = graph_evictions - initial_evictions;
    j["graph_replays"] = graph_replays - initial_replays;
    if (async_engram) {
      auto stats = async_engram->stats();
      for (const char *key : {"submitted_rows", "cache_hits", "unique_misses",
                              "disk_pages", "disk_bytes", "batches"})
        stats[key] = stats[key].get<uint64_t>() -
                     initial_engram_stats.at(key).get<uint64_t>();
      for (const char *key : {"io_ms", "total_ms"})
        stats[key] = stats[key].get<double>() -
                     initial_engram_stats.at(key).get<double>();
      j["engram_store"] = std::move(stats);
    }
    return j;
  }

  Json vram_inventory() const {
    Json j={{"schema","ds41-vram-census-v1"},{"max_context",config.max_context},
       {"host_embedding",config.host_embedding},{"reuse_woa",config.reuse_woa}};
    j["weights"]=Json::array();
    uint64_t weight_gpu=0,weight_host=0,arena=0,kv_bytes=0;
    for (const auto &entry:weights) {
      const auto &w=entry.second;
      bool mapped=w.data.mapped_host!=nullptr;
      j["weights"].push_back({{"name",entry.first},{"bytes",w.data.bytes},
          {"dtype",w.dtype},{"location",mapped?"mapped_pinned_host":"device"}});
      (mapped?weight_host:weight_gpu)+=w.data.bytes;
    }
    j["arenas"]=Json::object();
    for (auto entry : std::initializer_list<std::pair<const char*,const Device*>>{{"h",&h},{"next_h",&next_h},{"x",&x},{"y",&y},{"f32",&f32},{"pre",&pre},{"att_pre",&att_pre},{"ffn_pre",&ffn_pre},{"post",&post},{"comb",&comb},{"mixes",&mixes},{"rstd",&rstd},{"qa",&qa},{"q",&q},{"kv",&kv},{"latent",&latent},{"ik",&ik},{"iq",&iq},{"iw",&iw},{"packed",&packed},{"gather",&gather},{"local_idx",&local_idx},{"o",&o},{"wo",&wo},{"attn_scores",&attn_scores},{"aq",&aq},{"as",&as},{"gate",&gate},{"up",&up},{"act",&act},{"eq",&eq},{"es",&es},{"down",&down},{"shared",&shared},{"sg",&sg},{"su",&su},{"sa",&sa},{"jobs",&jobs},{"up_jobs",&up_jobs},{"down_jobs",&down_jobs},{"ids_device",&ids_device},{"route",&route},{"router",&router},{"unbiased",&unbiased},{"biased",&biased},{"logits",&logits},{"scores",&scores},{"top",&top},{"visible",&visible},{"position_device",&position_device},{"blocks",&blocks},{"block_visible",&block_visible},{"chosen_blocks",&chosen_blocks},{"candidate_ids",&candidate_ids},{"candidate_top",&candidate_top},{"top_scratch",&top_scratch},{"rows_raw",&rows_raw},{"scales_raw",&scales_raw},{"rows_bf",&rows_bf},{"engram_kv",&engram_kv},{"rows_interleaved",&rows_interleaved},{"braw_kv",&braw_kv},{"braw_score",&braw_score},{"batch_tokens",&batch_tokens},{"route_source",&route_source},{"route_slots",&route_slots},{"bx",&bx},{"bxs",&bxs},{"broute",&broute},{"compact_down",&compact_down},{"sparse_scores",&sparse_scores},{"work_jobs",&work_jobs},{"work_up_jobs",&work_up_jobs},{"work_down_jobs",&work_down_jobs},{"work_rows",&work_rows},{"batch_frequencies",&batch_frequencies},{"compressed_frequencies",&compressed_frequencies},{"lengths1",&lengths1},{"lengths2",&lengths2},{"packed_keys",&packed_keys},{"packed_globals",&packed_globals},{"gate_slab",&gate_slab},{"rope",&rope},{"woa_scratch",&woa_scratch},{"capture_slab",&capture_slab},{"spark_aq",&spark_aq},{"spark_as",&spark_as},{"spark_gate",&spark_gate},{"spark_up",&spark_up},{"spark_act",&spark_act},{"spark_eq",&spark_eq},{"spark_es",&spark_es},{"spark_jobs",&spark_jobs},{"spark_up_jobs",&spark_up_jobs},{"spark_down_jobs",&spark_down_jobs}}) {
      j["arenas"][entry.first]=entry.second->bytes; arena+=entry.second->bytes;
    }
    j["attention"]=Json::object();
    uint64_t window=0;
    for (const auto &w:windows) window+=w.bytes;
    j["attention"]["swa_windows"]=window; kv_bytes+=window;
    for (const auto &entry:globals) {
      const auto &g=*entry.second; uint64_t n=g.kv.bytes+g.key.bytes+g.raw_kv.bytes+g.raw_score.bytes;
      j["attention"][std::to_string(entry.first)]={{"kv",g.kv.bytes},{"indexer_key",g.key.bytes},
          {"raw_kv",g.raw_kv.bytes},{"raw_score",g.raw_score.bytes}};
      kv_bytes+=n;
    }
    j["weight_device_bytes"]=weight_gpu; j["weight_mapped_host_bytes"]=weight_host;
    j["arena_device_bytes"]=arena; j["attention_device_bytes"]=kv_bytes;
    uint64_t prefill_bytes=0;
    j["prefill_arenas"]=Json::object();
    if (pref) for (auto entry : std::initializer_list<std::pair<const char*,const Device*>>{{"pref.hidden",&pref->hidden},{"pref.previous_pre",&pref->previous_pre},{"pref.ffn_input",&pref->ffn_input},{"pref.ffn_post",&pref->ffn_post},{"pref.ffn_comb",&pref->ffn_comb},{"pref.window",&pref->window},{"pref.selected",&pref->selected},{"pref.candidates",&pref->candidates},{"pref.ids",&pref->ids},{"pref.weights",&pref->weights},{"pref.quant",&pref->quant},{"pref.scales",&pref->scales},{"pref.routed",&pref->routed},{"pref.capture",&pref->capture},{"pref.bx",&pref->bx},{"pref.bs",&pref->bs},{"pref.bw",&pref->bw},{"pref.gate",&pref->gate},{"pref.up",&pref->up},{"pref.act",&pref->act},{"pref.eq",&pref->eq},{"pref.es",&pref->es},{"pref.down",&pref->down}}) {
      j["prefill_arenas"][entry.first]=entry.second->bytes;prefill_bytes+=entry.second->bytes;
    }
    if (prefill_batch) for (auto entry : std::initializer_list<std::pair<const char*,const Device*>>{{"prefill_batch.h",&prefill_batch->h},{"prefill_batch.next",&prefill_batch->next},{"prefill_batch.x",&prefill_batch->x},{"prefill_batch.y",&prefill_batch->y},{"prefill_batch.pre",&prefill_batch->pre},{"prefill_batch.att_pre",&prefill_batch->att_pre},{"prefill_batch.ffn_pre",&prefill_batch->ffn_pre},{"prefill_batch.post",&prefill_batch->post},{"prefill_batch.comb",&prefill_batch->comb},{"prefill_batch.f32",&prefill_batch->f32},{"prefill_batch.mixes",&prefill_batch->mixes},{"prefill_batch.rstd",&prefill_batch->rstd},{"prefill_batch.qa",&prefill_batch->qa},{"prefill_batch.q",&prefill_batch->q},{"prefill_batch.kv",&prefill_batch->kv},{"prefill_batch.o",&prefill_batch->o},{"prefill_batch.wo",&prefill_batch->wo},{"prefill_batch.aq",&prefill_batch->aq},{"prefill_batch.as",&prefill_batch->as},{"prefill_batch.router",&prefill_batch->router},{"prefill_batch.unbiased",&prefill_batch->unbiased},{"prefill_batch.biased",&prefill_batch->biased},{"prefill_batch.sg",&prefill_batch->sg},{"prefill_batch.su",&prefill_batch->su},{"prefill_batch.sa",&prefill_batch->sa},{"prefill_batch.shared",&prefill_batch->shared}}) {
      j["prefill_arenas"][entry.first]=entry.second->bytes;prefill_bytes+=entry.second->bytes;
    }
    uint64_t prefill_stage_bytes=prefill_stage?prefill_stage->device_bytes():0;
    j["prefill_arena_device_bytes"]=prefill_bytes;
    j["prefill_staging_device_bytes"]=prefill_stage_bytes;
    j["target_device_requested_bytes"]=weight_gpu+arena+kv_bytes+prefill_bytes+prefill_stage_bytes;
    j["expert_slot_bytes"]=18808832; j["expert_slots"]=config.expert_slots;
    j["expert_payload_device_bytes"]=uint64_t(config.expert_slots)*18808832;
    j["moe_staging_pinned_host_bytes"]=uint64_t(8)*18808832;
    j["engine_staging_pinned_host_bytes"]=(16+1)*(1<<20);
    j["graphs"]=graphs.size();
    size_t stack=0;CK(cudaDeviceGetLimit(&stack,cudaLimitStackSize));
    j["cuda_stack_bytes_per_thread"]=stack;
    size_t free=0,total=0; CK(cudaMemGetInfo(&free,&total));
    j["cuda_device_total_bytes"]=total;j["cuda_device_free_bytes"]=free;
    return j;
  }

private:
  template <class F> int service_call(F operation) noexcept {
    try {
      if (poisoned)
        throw std::runtime_error("native forward/store poisoned");
      operation();
      callback_error.clear();
      return 0;
    } catch (const std::exception &e) {
      callback_error = e.what();
    } catch (...) {
      callback_error = "unknown target callback failure";
    }
    poisoned = true;
    return -1;
  }
  void routed_spark(int stage, const void *input, const int32_t *ids,
                    const float *route_weights, void *output, int rows,
                    void *supplied_stream) {
    if (supplied_stream != stream || stage < 0 || stage >= 3 || rows < 1 ||
        rows > 8 || !input || !ids || !route_weights || !output)
      throw std::invalid_argument("DSpark routed adapter arguments/stream");
    if (pending)
      throw std::logic_error(
          "DSpark callback during target verify transaction");
    constexpr int max_rows = 8, max_jobs = 3 * max_rows;
    if (!spark_aq.p) {
      spark_aq.resize(max_rows * H);
      spark_as.resize(max_rows * H / 32);
      spark_gate.resize(max_jobs * I * 2);
      spark_up.resize(max_jobs * I * 2);
      spark_act.resize(max_jobs * I * 2);
      spark_eq.resize(max_jobs * I);
      spark_es.resize(max_jobs * I / 32);
      spark_jobs.resize(max_jobs * sizeof(ds41::GemmJob));
      spark_up_jobs.resize(max_jobs * sizeof(ds41::GemmJob));
      spark_down_jobs.resize(max_jobs * sizeof(ds41::GemmJob));
    }
    // One deliberate route boundary, then unique logical IDs under one lease.
    auto selected = static_cast<int32_t *>(host->data());
    double t = clock_ms();
    CK(cudaMemcpyAsync(selected, ids, rows * 3 * sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    route_ms += clock_ms() - t;
    std::vector<ds41::ExpertId> logical(rows * 3), need;
    for (int r = 0; r < rows; ++r) {
      std::set<int32_t> unique_row;
      for (int k = 0; k < 3; ++k) {
        int id = selected[r * 3 + k];
        if (id < 0 || id >= 128 || !unique_row.insert(id).second)
          throw std::invalid_argument("invalid/duplicate DSpark route");
        logical[r * 3 + k] = ds41::expert_id(40 + stage, id);
      }
    }
    need = logical;
    std::sort(need.begin(), need.end());
    need.erase(std::unique(need.begin(), need.end()), need.end());
    if (need.size() > config.expert_slots)
      throw std::runtime_error(
          "DSpark unique demand exceeds expert slot budget");
    t = clock_ms();
    auto lease = experts->need_now(need, stream);
    io_ms += clock_ms() - t;
    try {
      int count = rows * 3;
      std::vector<ds41::GemmJob> job(count);
      for (int j = 0; j < count; ++j)
        job[j] = {logical[j], 1, spark_aq.as<uint8_t>((j / 3) * H),
                  spark_as.as<uint8_t>((j / 3) * H / 32),
                  spark_gate.as<Bf16>(j * I * 2)};
      // Job descriptors go through a pinned mapped buffer and a zero-copy kernel: a pageable H2D would
      // synchronize the stream before each copy. The buffer is rewritten only after the next route
      // readback, i.e. after these descriptors were consumed in stream order.
      constexpr size_t slab = size_t(max_jobs) * sizeof(ds41::GemmJob);
      if (!spark_host) spark_host = std::make_unique<ds41::HostBuffer>(3 * slab, true);
      auto *hj = static_cast<ds41::GemmJob *>(spark_host->data());
      auto *dj = static_cast<char *>(spark_host->device_data());
      std::copy(job.begin(), job.end(), hj);
      for (int j = 0; j < count; ++j) {
        hj[max_jobs + j] = job[j];
        hj[max_jobs + j].output = spark_up.as<Bf16>(j * I * 2);
        hj[2 * max_jobs + j] = {logical[j], 1, spark_eq.as<uint8_t>(j * I),
                                spark_es.as<uint8_t>(j * I / 32),
                                static_cast<Bf16 *>(output) + j * H};
      }
      zc(spark_jobs.p, dj, count * sizeof(ds41::GemmJob));
      zc(spark_up_jobs.p, dj + slab, count * sizeof(ds41::GemmJob));
      zc(spark_down_jobs.p, dj + 2 * slab, count * sizeof(ds41::GemmJob));
      CK(ds41::quantize_mxfp8(static_cast<const Bf16 *>(input),
                              spark_aq.as<uint8_t>(), spark_as.as<uint8_t>(),
                              rows, H, stream));
      CK(ds41::grouped_mxfp4_native_parallel(
          experts->table(), spark_jobs.as<ds41::GemmJob>(), count,
          ds41::Projection::Gate, I, H, stream));
      CK(ds41::grouped_mxfp4_native_parallel(
          experts->table(), spark_up_jobs.as<ds41::GemmJob>(), count,
          ds41::Projection::Up, I, H, stream));
      CK(ds41::swiglu(spark_gate.as<Bf16>(), spark_up.as<Bf16>(), route_weights,
                      spark_act.as<Bf16>(), count, I, 10, stream));
      CK(ds41::quantize_mxfp8(spark_act.as<Bf16>(), spark_eq.as<uint8_t>(),
                              spark_es.as<uint8_t>(), count, I, stream));
      CK(ds41::grouped_mxfp4_native_parallel(
          experts->table(), spark_down_jobs.as<ds41::GemmJob>(), count,
          ds41::Projection::Down, H, I, stream));
      // The store may recycle slots only after the down consumers complete.
      experts->retire(std::move(lease), stream);
    } catch (...) {
      cudaStreamSynchronize(stream);
      if (lease.owner)
        experts->retire(std::move(lease), stream);
      throw;
    }
  }
  template <class F> void graph(const std::string &key, F body) {
    if (!config.graphs || !graph_ready || !config.trace_dir.empty()) {
      body();
      return;
    }
    auto it = graphs.find(key);
    if (it != graphs.end()) {
      CK(cudaGraphLaunch(it->second->exec, stream));
      ++graph_replays;
      return;
    }
    auto recorded = std::make_unique<Graph>();
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {
      cudaStreamEndCapture(stream, &recorded->graph);
      throw;
    }
    CK(cudaStreamEndCapture(stream, &recorded->graph));
    CK(cudaGraphInstantiate(&recorded->exec, recorded->graph, 0));
    CK(cudaGraphLaunch(recorded->exec, stream));
    ++graph_replays;
    graphs.emplace(key, std::move(recorded));
    ++graph_captures;
  }
  void dump(int64_t pos, int layer, const char *tag, const Device &data,
            size_t elements) {
    if (config.trace_dir.empty())
      return;
    auto dir = std::filesystem::path(config.trace_dir) /
               ("pos" + std::to_string(pos)) /
               ("layer" + std::to_string(layer));
    std::filesystem::create_directories(dir);
    std::vector<uint16_t> raw(elements);
    CK(cudaMemcpyAsync(raw.data(), data.p, elements * 2, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    std::ofstream f(dir / (std::string(tag) + ".bf16"), std::ios::binary);
    f.write(reinterpret_cast<const char *>(raw.data()), raw.size() * 2);
    if (!f)
      throw std::runtime_error("failed diagnostic activation dump");
  }
  Weight &w(const std::string &n) {
    auto it = weights.find(n);
    if (it == weights.end())
      throw std::runtime_error("missing resident tensor: " + n);
    return it->second;
  }
  void upload(Device &d, const void *p, size_t n) {
    if (n > d.bytes)
      throw std::runtime_error("device upload overflow");
    if (in_step) {
      size_t aligned = (n + 63) & ~size_t(63);
      if (upload_cursor + aligned > host->size())
        throw std::runtime_error("per-step pinned metadata capacity");
      void *owned = static_cast<char *>(host->data()) + upload_cursor;
      std::memcpy(owned, p, n);
      p = owned;
      upload_cursor += aligned;
    }
    if (in_step && zc_batching) zc_pending.push_back({static_cast<const unsigned char *>(host_alias(p)), static_cast<unsigned char *>(d.p), n});
    else if (in_step) zc(d.p, host_alias(p), n);
    else CK(cudaMemcpyAsync(d.p, p, n, cudaMemcpyHostToDevice, stream));
  }
  // Group consecutive in-step uploads into one zero-copy launch (zc_begin ... zc_end). Only for sites
  // whose consumers are enqueued after zc_end().
  bool zc_batching = false;
  std::vector<ZcSeg> zc_pending;
  void zc_begin() { zc_batching = in_step; zc_pending.clear(); }
  void zc_end() {
    zc_batching = false;
    if (zc_pending.empty()) return;
    if (zc_pending.size() == 1) { zc(zc_pending[0].dst, zc_pending[0].src, zc_pending[0].n); zc_pending.clear(); return; }
    const size_t bytes = zc_pending.size() * sizeof(ZcSeg), aligned = (bytes + 63) & ~size_t(63);
    if (upload_cursor + aligned > host->size()) throw std::runtime_error("per-step pinned metadata capacity");
    void *table = static_cast<char *>(host->data()) + upload_cursor;
    std::memcpy(table, zc_pending.data(), bytes);
    upload_cursor += aligned;
    zc_multi<<<int(zc_pending.size()), 128, 0, stream>>>(static_cast<const ZcSeg *>(host_alias(table)), int(zc_pending.size()));
    CK(cudaGetLastError());
    zc_pending.clear();
  }
  // Device alias of an address inside the mapped pinned step buffer.
  void *host_alias(const void *p) const {
    return static_cast<char *>(host->device_data()) + (static_cast<const char *>(p) - static_cast<const char *>(host->data()));
  }
  void zc(void *dst, const void *src, size_t n) {
    if (!n) return;
    const int threads = 256;
    const int blocks = int(std::min<size_t>((n + 16 * threads - 1) / (16 * threads), 512));
    zc_copy<<<blocks, threads, 0, stream>>>(static_cast<const unsigned char *>(src), static_cast<unsigned char *>(dst), n);
    CK(cudaGetLastError());
  }
  template <class F> void phase(const std::string &name, F call) {
    if (phase_index >= phases.size())
      throw std::logic_error("phase capacity");
    auto &p = *phases[phase_index++];
    CK(cudaEventRecord(p.start, stream));
    call();
    CK(cudaEventRecord(p.end, stream));
    phase_names.push_back(name);
  }
  std::vector<std::string> phase_names;
  void load_weights() {
    size_t sum = 0;
    for (auto item : source.tensors()) {
      const auto &t = item.second;
      auto n = t.name;
      if (n.find(".experts.") != std::string::npos ||
          n.find(".engram.embed.") != std::string::npos ||
          (n.compare(0, 4, "mtp.") == 0 && !config.load_mtp) ||
          n.compare(0, 7, "vision.") == 0 || n.compare(0, 8, "aligner.") == 0 ||
          n.compare(0, 6, "image_") == 0)
        continue;
      bool promote =
          (n == "head.weight" && config.fused_head_library.empty()) ||
          n.find(".ffn.gate.weight") != std::string::npos ||
          n.find(".attn.compressor.wgate.weight") != std::string::npos ||
          n == "mtp.2.markov_head.head.weight" ||
          n == "mtp.2.confidence_head.proj.weight";
      if (n.find(".attn.compressor.wkv.weight") != std::string::npos &&
          n.compare(0, 10, "layers.20.") != 0)
        promote = true;
      auto bytes = source.read(t);
      Weight weight;
      weight.shape = t.shape;
      weight.dtype = t.dtype;
      if (n == "embed.weight" && config.host_embedding) {
        // Native BF16 bytes; GPU gather and Spark keep the same tensor view.
        weight.data.resize_mapped(t.bytes);
        std::memcpy(weight.data.mapped_host,bytes.data(),bytes.size());
      } else {
        weight.data.resize(t.bytes);
        upload(weight.data, bytes.data(), bytes.size());
      }
      CK(cudaStreamSynchronize(stream));
      if (promote && t.dtype == "BF16") {
        Device converted(t.bytes * 2);
        bf16_float(weight.data.p, converted.as<float>(), int(t.bytes / 2),
                   stream);
        weight.data = std::move(converted);
        weight.dtype = "F32";
      }
      sum += weight.data.bytes;
      weights.emplace(n, std::move(weight));
    }
    for (int l = 0; l < (config.load_mtp ? 43 : 40); ++l) {
      std::string p = (l < 40 ? "layers." + std::to_string(l)
                              : "mtp." + std::to_string(l - 40)) +
                      ".attn.wo_a.";
      auto &a = w(p + "weight");
      auto &s = w(p + "scale");
      if (a.dtype != "F8_E4M3" && a.dtype != "F8_E4M3FN")
        throw std::runtime_error(
            "wo_a must retain native FP8 source before exact BF16 dequant");
      if (config.reuse_woa && l < 40) {
        // Keep the authenticated FP8 source and original scales. Dequantize
        // one layer into a stream-ordered shared BF16 buffer before cuBLAS.
        if (!woa_scratch.bytes) woa_scratch.resize(a.shape[0]*a.shape[1]*2);
        if (woa_scratch.bytes != a.shape[0]*a.shape[1]*2)
          throw std::runtime_error("shared wo_a shape mismatch");
        continue;
      }
      Device dequant(a.shape[0] * a.shape[1] * 2);
      CK(ds41_dense_dequant(a.data.p, s.data.p, dequant.p, int(a.shape[0]),
                            int(a.shape[1]), stream));
      a.data = std::move(dequant);
      a.dtype = "BF16";
    }
    CK(cudaStreamSynchronize(stream));
    std::cerr << "resident source bytes " << sum
              << " (wo_a BF16 promotion additional)\n";
  }
  void load_rope() {
    auto meta = read_json(config.rope + ".json");
    if (meta.at("schema") != "ds41-rope-v1" ||
        meta.at("rows").get<int>() < config.max_context)
      throw std::runtime_error("RoPE metadata/context mismatch");
    size_t file_rows = meta.at("rows").get<size_t>();
    std::ifstream f(config.rope, std::ios::binary);
    if (!f)
      throw std::runtime_error("missing RoPE table");
    rope.resize(size_t(config.max_context) * 64 * 4 * 2);
    std::vector<float> part(size_t(config.max_context) * 64);
    for (int i = 0; i < 2; ++i) {
      f.seekg(i * file_rows * 64 * 4);
      f.read(reinterpret_cast<char *>(part.data()), part.size() * 4);
      if (!f)
        throw std::runtime_error("short RoPE table");
      CK(cudaMemcpyAsync(rope.as<char>(i * part.size() * 4), part.data(),
                         part.size() * 4, cudaMemcpyHostToDevice, stream));
      CK(cudaStreamSynchronize(stream));
    }
  }
  const float *freqs(int l, int64_t pos) const {
    return rope.as<float>((size_t(l >= 2) * config.max_context + pos) * 64 * 4);
  }
  const void *woa_weight(const std::string &prefix) {
    auto &weight=w(prefix+".weight");
    if (!config.reuse_woa || prefix.compare(0,7,"layers.") != 0)
      return weight.data.p;
    // This call is captured inside the layer output graph, so every replay
    // refreshes scratch on the same stream before its unchanged cuBLAS call.
    CK(ds41_dense_dequant(weight.data.p,w(prefix+".scale").data.p,
        woa_scratch.p,int(weight.shape[0]),int(weight.shape[1]),stream));
    return woa_scratch.p;
  }
  void linear(const std::string &prefix, const void *in, void *out, int n,
              int k) {
    auto &weight = w(prefix + ".weight");
    if (weight.shape != std::vector<uint64_t>({uint64_t(n), uint64_t(k)}))
      throw std::runtime_error("linear shape: " + prefix);
    if (weight.dtype == "F8_E4M3" || weight.dtype == "F8_E4M3FN") {
      CK(ds41_dense_quant(in, aq.p, as.p, 1, k, stream));
      CK(ds41_dense_fp8(aq.p, as.p, weight.data.p, w(prefix + ".scale").data.p,
                        out, 1, n, k, 2, stream));
    } else {
      CK(ds41_dense_cublas(blas, in, weight.data.p, out, 1, n, k,
                           weight.dtype == "F32", 1, stream));
    }
  }
  void norm(const std::string &name, const void *in, void *out, int k) {
    CK(ds41_dense_rms(in, w(name + ".weight").data.p, out, 1, k, 1e-20f,
                      stream));
  }
  void hc(const std::string &prefix, const std::string &kind,
          Device &target_pre) {
    CK(ds41_dense_hc_mixes(
        blas, h.p, w(prefix + ".hc_" + kind + "_fn").data.as<float>(),
        w(prefix + ".hc_" + kind + "_scale").data.as<float>(),
        w(prefix + ".hc_" + kind + "_base").data.as<float>(),
        target_pre.as<float>(), post.as<float>(), comb.as<float>(),
        f32.as<float>(), mixes.as<float>(), rstd.as<float>(), 1, H, 1e-20f,
        1e-6f, stream));
  }
  void apply_rope(Device &d, int heads, int dim, int l, int64_t pos,
                  bool inverse = false) {
    CK(ds41_dense_rope(d.p, freqs(l, pos), 1, heads, dim, 64, inverse, stream));
  }
  void engram(int layer, const std::vector<int64_t> &rowids, int m = 1) {
    double t = clock_ms();
    std::string prefix = "layers." + std::to_string(layer) + ".engram";
    if (async_engram) {
      const void *raw = engram_tickets[layer == 1 ? 0 : 1]->wait(rowids.size());
      upload(rows_interleaved, raw, rowids.size() * 264);
      async_engram->dequant(rows_interleaved.p, rows_bf.p, int(rowids.size()), stream);
      engram_ms += clock_ms() - t;
    } else {
      const auto &values = source.tensor(prefix + ".embed.weight");
      const auto &scales = source.tensor(prefix + ".embed.scale");
      std::map<std::pair<std::string, uint64_t>, size_t> page_index;
      std::vector<ds41::Record> requests;
      std::vector<void *> destinations;
      auto fetch = [&](const Tensor &tensor, int64_t row, size_t rowbytes) {
        if (row < 0 || uint64_t(row) >= tensor.shape[0])
          throw std::runtime_error("Engram row OOB");
        uint64_t address = tensor.offset + uint64_t(row) * rowbytes;
        for (uint64_t p = address & ~4095ull; p < address + rowbytes;
             p += 4096) {
          auto key = std::make_pair(tensor.path, p);
          if (page_index.count(key))
            continue;
          size_t index = page_index.size();
          page_index[key] = index;
          ds41::Record rec;
          rec.id = uint32_t(index);
          rec.bytes = 4096;
          ds41::Extent e{tensor.path,
                         p,
                         4096,
                         0,
                         std::min<uint64_t>(4096, tensor.file_bytes - p),
                         tensor.file_bytes};
          rec.extents.push_back(e);
          requests.push_back(std::move(rec));
        }
      };
      for (auto row : rowids) {
        fetch(values, row, 256);
        fetch(scales, row, 8);
      }
      if (requests.size() * 4096 > pages->size())
        throw std::runtime_error("Engram staging capacity");
      std::vector<const ds41::Record *> ptrs;
      for (size_t i = 0; i < requests.size(); ++i) {
        ptrs.push_back(&requests[i]);
        destinations.push_back(static_cast<char *>(pages->data()) + i * 4096);
      }
      reader->read(ptrs, destinations);
      auto gather_row = [&](const Tensor &tensor, int64_t row, size_t stride,
                            uint8_t *out) {
        uint64_t a = tensor.offset + uint64_t(row) * stride;
        for (size_t i = 0; i < stride; ++i) {
          uint64_t offset = a + i;
          auto ix = page_index.at({tensor.path, offset & ~4095ull});
          out[i] = static_cast<uint8_t *>(
              pages->data())[ix * 4096 + (offset & 4095)];
        }
      };
      auto raw = static_cast<uint8_t *>(host->data());
      for (size_t i = 0; i < rowids.size(); ++i) {
        gather_row(values, rowids[i], 256, raw + i * 256);
        gather_row(scales, rowids[i], 8, raw + rowids.size() * 256 + i * 8);
      }
      upload(rows_raw, raw, rowids.size() * 256);
      upload(scales_raw, raw + rowids.size() * 256, rowids.size() * 8);
      engram_ms += clock_ms() - t;
      engram_dequant(rows_raw.as<uint8_t>(), scales_raw.as<uint8_t>(),
                     rows_bf.p, int(rowids.size()), stream);
    }
    blinear(prefix + ".wkv", rows_bf.p, engram_kv.p, (HC + 1) * H, 6144, m);
    for (int r = 0; r < m; ++r)
    CK(ds41_dense_engram_gate(h.as<char>(r*HC*H*2), engram_kv.as<char>(r*(HC+1)*H*2), engram_kv.as<char>((r*(HC+1)+HC)*H*2),
                              w(prefix + ".q_weight").data.p,
                              w(prefix + ".k_weight").data.p, nullptr, next_h.as<char>(r*HC*H*2),
                              1, H, 1e-20f, stream));
    std::swap(h, next_h);
  }
  void attention(int l, int64_t pos) {
    std::string p = "layers." + std::to_string(l) + ".attn";
    apply_rope(q, 64, 512, l, pos);
    apply_rope(kv, 1, 512, l, pos);
    CK(ds41_attn_pack(kv.p, packed.p, 1, DS41_SWA_FP8, stream));
    CK(ds41_attn_ring_write(packed.p, windows[l].p, 528, pos, 1, RING, stream));
    int owner = l < 2 ? -1 : l < 8 ? 2 : l < 14 ? 8 : l < 20 ? 14 : 20;
    int ratio = l < 20 ? 2 : 1;
    int n = owner < 0 ? 0 : int((pos + 1) / ratio);
    bool owns = globals.count(l);
    bool index = l == 2 || l == 8 || l == 14 || l == 20 || l == 24 || l == 28 ||
                 l == 32 || l == 36;
    if (owns) {
      auto &g = *globals.at(l);
      bool complete = ratio == 1 || (pos % 2 == 1);
      if (ratio == 1)
        linear(p + ".compressor.wkv", x.p, latent.p, 512, H);
      else {
        bf16_float(x.p, f32.as<float>(), H, stream);
        auto off = (pos % 8) * 512 * 4;
        linear(p + ".compressor.wkv", f32.p, g.raw_kv.as<char>(off), 512, H);
        linear(p + ".compressor.wgate", f32.p, g.raw_score.as<char>(off), 512,
               H);
        if (complete)
          CK(ds41_attn_pool2(g.raw_kv.as<float>(), g.raw_score.as<float>(),
                             latent.p, pos, 1, 8, stream));
      }
      if (complete) {
        norm(p + ".compressor.norm", latent.p, latent.p, 512);
        linear(p + ".indexer.wk", latent.p, ik.p, 128, 512);
        norm(p + ".indexer.k_norm", ik.p, ik.p, 128);
        apply_rope(ik, 1, 128, l, pos + 1 - ratio);
        CK(ds41_attn_pack(ik.p, g.key.as<char>(size_t(n - 1) * 68), 1,
                          DS41_INDEX_FP4, stream));
        apply_rope(latent, 1, 512, l, pos + 1 - ratio);
        CK(ds41_attn_pack(latent.p, g.kv.as<char>(size_t(n - 1) * 288), 1,
                          DS41_GLOBAL_FP4, stream));
      }
    }
    int k = std::min(512, n);
    if (index && n) {
      auto &g = *globals.at(owner);
      linear(p + ".indexer.wq_b", qa.p, iq.p, 4096, 1280);
      apply_rope(iq, 32, 128, l, pos);
      CK(ds41_attn_roundtrip(iq.p, 32, 128, DS41_INDEX_FP4, stream));
      linear(p + ".indexer.weights_proj", x.p, iw.p, 32, H);
      scale_bf16(iw.p, 32, 1.f / 64, stream);
      upload(visible, &n, 4);
      int nscan = l > 20 ? candidate_count : n;
      const int32_t *keys = l > 20 ? candidate_ids.as<int32_t>() : nullptr;
      CK(ds41_attn_index_scores(iq.p, g.key.p, iw.p, keys,
                                visible.as<int32_t>(), scores.p, 1, nscan,
                                stream));
      if (l == 20) {
        int nb = (n + 7) / 8, kb = std::min(2048, nb);
        CK(ds41_attn_block_scores(scores.p, visible.as<int32_t>(), blocks.p, 1,
                                  n, stream));
        upload(block_visible, &nb, 4);
        CK(ds41_attn_topk(blocks.p, block_visible.as<int32_t>(),
                          chosen_blocks.as<int32_t>(), 1, nb, kb, top_scratch.p,
                          stream));
        CK(ds41_attn_expand_blocks(chosen_blocks.as<int32_t>(),
                                   candidate_ids.as<int32_t>(), 1, kb, n,
                                   stream));
        candidate_count = kb * 8;
      }
      if (l > 20) {
        int scan_visible = nscan;
        upload(block_visible, &scan_visible, 4);
        CK(ds41_attn_topk(scores.p, block_visible.as<int32_t>(),
                          candidate_top.as<int32_t>(), 1, nscan, k,
                          top_scratch.p, stream));
        CK(ds41_attn_remap_topk(candidate_top.as<int32_t>(),
                                candidate_ids.as<int32_t>(), top.as<int32_t>(),
                                1, k, nscan, stream));
      } else
        CK(ds41_attn_topk(scores.p, visible.as<int32_t>(), top.as<int32_t>(), 1,
                          n, k, top_scratch.p, stream));
      shared_topk = k;
    } else if (index)
      shared_topk = 0;
    if (owner < 0)
      k = 0;
    else if (shared_topk != k)
      throw std::runtime_error("shared index width mismatch");
    Global *g = owner < 0 ? nullptr : globals.at(owner).get();
    if (pos == 0) {
      upload(position_device, &pos, 8);
      CK(ds41_attn_gather_prefill_at(
          windows[l].p, RING, g ? g->kv.p : nullptr, g ? g->capacity : 0,
          k ? top.as<int32_t>() : nullptr, k, position_device.as<int64_t>(), 1,
          ratio, 1, gather.p, local_idx.as<int32_t>(), stream));
    } else
      CK(ds41_attn_gather(windows[l].p, RING, g ? g->kv.p : nullptr,
                          g ? g->capacity : 0, k ? top.as<int32_t>() : nullptr,
                          k, pos, 1, ratio, gather.p, local_idx.as<int32_t>(),
                          stream));
    int columns = (pos == 0 ? 1 : 128) + k;
    if (owner >= 0 && pos > 0)
      CK(ds41_attn_sparse_split(
          q.p, gather.p, w(p + ".attn_sink").data.as<float>(),
          local_idx.as<int32_t>(), o.p, attn_scores.as<float>(), 1, columns, 16,
          stream));
    else
      CK(ds41_attn_sparse(q.p, gather.p, w(p + ".attn_sink").data.as<float>(),
                          local_idx.as<int32_t>(), o.p, 1, columns, 16,
                          stream));
    apply_rope(o, 64, 512, l, pos, true);
    CK(ds41_attn_roundtrip(o.p, 1, 32768, DS41_SWA_FP8, stream));
    graph(p + ".output", [&] {
      CK(ds41_dense_cublas(blas, o.p, woa_weight(p + ".wo_a"), wo.p, 1,
                           1024, 4096, 0, 8, stream));
      linear(p + ".wo_b", wo.p, y.p, H, 8192);
    });
  }
  // image: an image-span row, which selects experts with the vision bias (official Gate.bias_vl).
  void router_scores(int l, bool image = false) {
    std::string p = "layers." + std::to_string(l) + ".ffn";
    bf16_float(x.p, f32.as<float>(), H, stream);
    linear(p + ".gate", f32.p, router.p, 384, H);
    CK(ds41_dense_router_scores(
        router.as<float>(), w(p + (image ? ".gate.bias_vl" : ".gate.bias")).data.as<float>(),
        unbiased.as<float>(), biased.as<float>(), 1, 384, stream));
  }
  void ensure_vision() {
    if (vision) return;
    image_delimiters.resize(3 * H * 2);
    int i = 0;
    for (const char *name : {"image_start", "image_newline", "image_end"}) {
      const auto &t = source.tensor(name);
      if (t.dtype != "BF16" || t.shape != std::vector<uint64_t>({uint64_t(H)}))
        throw std::runtime_error(std::string("image delimiter dtype/shape: ") + name);
      auto bytes = source.read(t);
      CK(cudaMemcpyAsync(image_delimiters.as<char>(size_t(i++) * H * 2), bytes.data(), bytes.size(),
                         cudaMemcpyHostToDevice, stream));
      CK(cudaStreamSynchronize(stream));
    }
    vision = std::make_unique<VisionEncoder>(source, stream);
    std::cerr << "ds41d vision tower loaded\n";
  }
  void moe(int l) {
    std::string p = "layers." + std::to_string(l) + ".ffn";
    double t = clock_ms();
    auto score_host = static_cast<float *>(host->data());
    CK(cudaMemcpyAsync(score_host, biased.p, 384 * 4, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    route_ms += clock_ms() - t;
    check_moe_retirement(false);
    std::vector<int> selected(384);
    std::iota(selected.begin(), selected.end(), 0);
    std::partial_sort(selected.begin(), selected.begin() + 6, selected.end(),
                      [&](int a, int b) {
                        return score_host[a] == score_host[b]
                                   ? a < b
                                   : score_host[a] > score_host[b];
                      });
    selected.resize(6);
    upload(ids_device, selected.data(), 6 * 4);
    std::vector<ds41::ExpertId> need;
    for (int e : selected)
      need.push_back(ds41::expert_id(l, e));
    t = clock_ms();
    auto lease = experts->need_now(need, stream);
    io_ms += clock_ms() - t;
    try {
      std::vector<ds41::GemmJob> job(6);
      for (int j = 0; j < 6; ++j)
        job[j] = {need[j], 1, aq.as<uint8_t>(), as.as<uint8_t>(),
                  gate.as<Bf16>(j * I * 2)};
      upload(jobs, job.data(), job.size() * sizeof(ds41::GemmJob));
      for (int j = 0; j < 6; ++j)
        job[j].output = up.as<Bf16>(j * I * 2);
      upload(up_jobs, job.data(), job.size() * sizeof(ds41::GemmJob));
      for (int j = 0; j < 6; ++j)
        job[j] = {need[j], 1, eq.as<uint8_t>(j * I), es.as<uint8_t>(j * I / 32),
                  down.as<Bf16>(j * H * 2)};
      upload(down_jobs, job.data(), job.size() * sizeof(ds41::GemmJob));
      graph(p + ".experts", [&] {
        route_weights(unbiased.as<float>(), ids_device.as<int>(),
                      route.as<float>(), 6, stream);
        CK(ds41::quantize_mxfp8(x.as<Bf16>(), aq.as<uint8_t>(),
                                as.as<uint8_t>(), 1, H, stream));
        CK(ds41::grouped_mxfp4_native_parallel(
            experts->table(), jobs.as<ds41::GemmJob>(), 6,
            ds41::Projection::Gate, I, H, stream));
        CK(ds41::grouped_mxfp4_native_parallel(
            experts->table(), up_jobs.as<ds41::GemmJob>(), 6,
            ds41::Projection::Up, I, H, stream));
        CK(ds41::swiglu(gate.as<Bf16>(), up.as<Bf16>(), route.as<float>(),
                        act.as<Bf16>(), 6, I, 10, stream));
        CK(ds41::quantize_mxfp8(act.as<Bf16>(), eq.as<uint8_t>(),
                                es.as<uint8_t>(), 6, I, stream));
        CK(ds41::grouped_mxfp4_native_parallel(
            experts->table(), down_jobs.as<ds41::GemmJob>(), 6,
            ds41::Projection::Down, H, I, stream));
        linear(p + ".shared_experts.w1", x.p, sg.p, I, H);
        linear(p + ".shared_experts.w3", x.p, su.p, I, H);
        CK(ds41::swiglu(sg.as<Bf16>(), su.as<Bf16>(), nullptr, sa.as<Bf16>(), 1,
                        I, 10, stream));
        linear(p + ".shared_experts.w2", sa.p, shared.p, H, I);
        CK(ds41::reduce_experts(down.as<Bf16>(), ids_device.as<int32_t>(),
                                shared.as<Bf16>(), y.as<Bf16>(), 1, 6, H,
                                stream));
      });
      retire_experts(lease);
    } catch (...) {
      CK(cudaStreamSynchronize(stream));
      if (lease.owner)
        experts->retire(std::move(lease), stream);
      throw;
    }
  }
  std::vector<float> one(int token, int64_t pos) {
    double start = clock_ms();
    phase_index = 0;
    phase_names.clear();
    upload_cursor = 1 << 20;
    in_step = true;
    graph_ready = pos > 0;
    auto rowids = hash.rows(token, pos);
    if (async_engram) {
      // The previous one() synchronized this stream before its tickets can
      // be recycled. Keep both current tickets through all H2D consumers,
      // including exception cleanup in the Native destructor.
      engram_tickets[0] = async_engram->submit(1, rowids[0]);
      engram_tickets[1] = async_engram->submit(14, rowids[1]);
    }
    expand_embedding(w("embed.weight").data.as<Bf16>(size_t(token) * H * 2),
                     h.p, pre.as<float>(), H, stream);
    for (int l = 0; l < 40; ++l) {
      std::string p = "layers." + std::to_string(l);
      if (l == 1 || l == 14)
        engram(l, rowids[l == 1 ? 0 : 1]);
      dump(pos, l, "input", h, HC * H);
      if (l >= 37)
        capture_mean(
            h.p,
            capture_slab.as<char>((size_t(capture_slot) * 3 + l - 37) * H * 2),
            H, stream);
      phase("attention", [&] {
        graph(p + ".query", [&] {
          hc(p, "attn", att_pre);
          CK(ds41_dense_hc_pre(h.p, pre.as<float>(), x.p, 1, H, stream));
          norm(p + ".attn_norm", x.p, x.p, H);
          std::string a = p + ".attn";
          linear(a + ".wq_a", x.p, qa.p, 1280, H);
          norm(a + ".q_norm", qa.p, qa.p, 1280);
          linear(a + ".wq_b", qa.p, q.p, 32768, 1280);
          linear(a + ".wkv", x.p, kv.p, 512, H);
          norm(a + ".kv_norm", kv.p, kv.p, 512);
        });
        dump(pos, l, "attn_input", x, H);
        attention(l, pos);
        dump(pos, l, "attn_output", y, H);
        CK(ds41_dense_hc_post(y.p, h.p, post.as<float>(), comb.as<float>(),
                              next_h.p, 1, H, stream));
        std::swap(h, next_h);
      });
      phase("moe", [&] {
        graph(p + ".route", [&] {
          hc(p, "ffn", ffn_pre);
          CK(ds41_dense_hc_pre(h.p, att_pre.as<float>(), x.p, 1, H, stream));
          norm(p + ".ffn_norm", x.p, x.p, H);
          router_scores(l);
        });
        dump(pos, l, "ffn_input", x, H);
        moe(l);
        dump(pos, l, "ffn_output", y, H);
        CK(ds41_dense_hc_post(y.p, h.p, post.as<float>(), comb.as<float>(),
                              next_h.p, 1, H, stream));
        std::swap(h, next_h);
        std::swap(pre, ffn_pre);
      });
      dump(pos, l, "output", h, HC * H);
    }
    phase("head", [&] {
      CK(ds41_dense_hc_pre(h.p, pre.as<float>(), x.p, 1, H, stream));
      norm("norm", x.p, x.p, H);
      bf16_float(x.p, f32.as<float>(), H, stream);
      if (fused_head) fused_head_logits(f32.p, logits.p, 1);
      else linear("head", f32.p, logits.p, V, H);
    });
    auto output = static_cast<float *>(host->data());
    CK(cudaMemcpyAsync(output, logits.p, V * 4, cudaMemcpyDeviceToHost,
                       stream));
    CK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < phase_index; ++i) {
      float elapsed;
      CK(cudaEventElapsedTime(&elapsed, phases[i]->start, phases[i]->end));
      measured[phase_names[i]] += elapsed;
    }
    total_ms += clock_ms() - start;
    in_step = false;
    return {output, output + V};
  }
  #include "hotpath/moe_adapter.inc"
  #include "hotpath/batch.inc"
  #include "hotpath/streamed.inc"
  void validate_records(const std::vector<ds41::Record> &records) {
    std::set<unsigned> seen;
    for (auto &r : records) {
      std::vector<std::pair<uint64_t, uint64_t>> spans;
      for (const auto &extent : r.extents) {
        uint64_t required =
            extent.required_bytes ? extent.required_bytes : extent.bytes;
        if (required > extent.bytes ||
            extent.file_offset > extent.expected_file_bytes ||
            required > extent.expected_file_bytes - extent.file_offset)
          throw std::runtime_error(
              "invalid required expert span/file coverage");
        spans.emplace_back(extent.dst_offset, extent.bytes);
      }
      validate_spans(r.bytes, std::move(spans));
      std::vector<bool> used(r.extents.size());
      if (r.id >= ds41::expert_count || !seen.insert(r.id).second)
        throw std::runtime_error("expert manifest duplicate/id");
      unsigned layer = r.id < 15360 ? r.id / 384 : 40 + (r.id - 15360) / 128;
      unsigned expert = r.id < 15360 ? r.id % 384 : (r.id - 15360) % 128;
      std::string p = (layer < 40 ? "layers." + std::to_string(layer)
                                  : "mtp." + std::to_string(layer - 40)) +
                      ".ffn.experts." + std::to_string(expert);
      std::vector<std::pair<std::string, ds41::TensorView>> views = {
          {"w1.weight", r.gate}, {"w1.scale", r.gate_scale},
          {"w3.weight", r.up},   {"w3.scale", r.up_scale},
          {"w2.weight", r.down}, {"w2.scale", r.down_scale}};
      for (auto item : views) {
        auto &t = source.tensor(p + "." + item.first);
        if (item.second.bytes != t.bytes)
          throw std::runtime_error("expert view byte mismatch");
        bool match = false;
        for (size_t index = 0; index < r.extents.size(); ++index) {
          const auto &e = r.extents[index];
          uint64_t coverage = e.required_bytes ? e.required_bytes : e.bytes;
          if (item.second.offset >= e.dst_offset &&
              item.second.offset - e.dst_offset <= coverage &&
              item.second.bytes <=
                  coverage - (item.second.offset - e.dst_offset) &&
              e.path == t.path && e.expected_file_bytes == t.file_bytes &&
              e.file_offset + item.second.offset - e.dst_offset == t.offset)
            match = used[index] = true;
        }
        if (!match)
          throw std::runtime_error("expert view does not match verified native "
                                   "tensor/required extent coverage: " +
                                   p);
      }
      if (std::find(used.begin(), used.end(), false) != used.end())
        throw std::runtime_error("extraneous expert read span");
    }
  }
};
} // namespace
std::unique_ptr<Forward> make_native(const Config &c) {
  return std::make_unique<Native>(c);
}
} // namespace ds41e
