#include "ds41/engine.hpp"
#include "ds41/native_store.hpp"
#include <iostream>
#include <stdexcept>
using namespace ds41e;
int main(int argc, char **argv) {
  try {
    std::string command = argc > 1 ? argv[1] : "help", socket, replay;
    Config c;
    bool headers = false;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--headers-only") {
        headers = true;
        continue;
      }
      if (a == "--vram-inventory") { c.report_vram = true; continue; }
      if (a == "--host-embedding") { c.host_embedding = true; continue; }
      if (a == "--reuse-woa") { c.reuse_woa = true; continue; }
      if (a == "--verify-lanes") { c.verify_lanes = true; continue; }
      if (a == "--compact-woa") { c.compact_woa = c.reuse_woa = true; continue; }
      if (a == "--allow-unqualified") {
        c.allow_unqualified = true;
        continue;
      }
      if (a == "--dynamic-host") { c.dynamic_host=true;continue; }
      if (a == "--reset-cache-history") { c.reset_cache_history=true;continue; }
      if (a == "--async-retire") { c.async_retire=true;continue; }
      if (a == "--hotpath-profile") {
        c.hotpath_profile = true; continue;
      }
      if (a == "--streamed-experts") {
        c.streamed_experts = true; c.batched_verify = true;
        continue;
      }
      if (a == "--prefetch-routes") {
        c.prefetch_routes = true;
        continue;
      }
      if (a == "--prefetch-lookahead") {
        c.prefetch_lookahead = true;
        continue;
      }
      if (a == "--batched-verify") {
        c.batched_verify = true;
        continue;
      }
      if (a == "--graphs") {
        c.graphs = true;
        continue;
      }
      if (a == "--load-mtp") {
        c.load_mtp = true;
        continue;
      }
      if (a == "--prefix-any-conversation") { c.prefix_any_conversation = true; continue; }
      if (a == "--prefix-cache") {
        c.prefix_cache = true;
        continue;
      }
      if (a == "--prefill") {
        c.prefill = true;
        continue;
      }
      if (a == "--spark") {
        c.spark = c.load_mtp = true;
        continue;
      }
      if (a == "--spark-policy-v2") {
        c.spark_policy_v2 = c.spark = c.load_mtp = true;
        continue;
      }
      if (i + 1 >= argc)
        throw std::invalid_argument("missing option value: " + a);
      std::string v = argv[++i];
      if (a == "--weights")
        c.weights = v;
      else if (a == "--receipt")
        c.receipt = v;
      else if (a == "--auxiliary")
        c.auxiliary = v;
      else if (a == "--engram")
        c.engram = v;
      else if (a == "--engram-lib")
        c.engram_library = v;
      else if (a == "--rope")
        c.rope = v;
      else if (a == "--experts")
        c.experts = v;
      else if (a == "--fused-head")
        c.fused_head_library = v;
      else if (a == "--fuse-library") {
        c.fuse_library = v; c.batched_verify = true;
      }
      else if (a == "--placement")
        c.placement = v;
      else if (a == "--trace-dir")
        c.trace_dir = v;
      else if (a == "--socket")
        socket = v;
      else if (a == "--replay-test")
        replay = v;
      else if (a == "--max-context")
        c.max_context = std::stoi(v);
      else if (a == "--prefill-warm-rows")
        c.prefill_warm_rows = std::stoull(v);
      else if (a == "--prefill-warm-gib") {
        c.prefill_warm_gib = std::stod(v);
        if(!(c.prefill_warm_gib > 0)) throw std::invalid_argument("prefill warm GiB must be positive");
      }
      else if (a == "--prefill-chunk") {
        c.prefill_chunk_rows = std::stoull(v);
        if(c.prefill_chunk_rows < 8 || c.prefill_chunk_rows > 1024) throw std::invalid_argument("prefill chunk must be 8..1024");
      }
      else if (a == "--prefill-dense-batch") {
        c.prefill_dense_batch = std::stoull(v);
        if(c.prefill_dense_batch < 1 || c.prefill_dense_batch > 8) throw std::invalid_argument("prefill dense batch must be 1..8");
      }
      else if (a == "--cuda-stack-bytes")
        c.cuda_stack_bytes = std::stoull(v);
      else if (a == "--expert-slots")
        c.expert_slots = std::stoull(v);
      else if (a == "--host-recent") {
        c.host_recent_window = std::stoi(v);
        if(c.host_recent_window < 0) throw std::invalid_argument("host recent must be nonnegative");
      }
      else if (a == "--host-decay") {
        c.host_decay_interval = std::stoi(v);
        if(c.host_decay_interval < 0) throw std::invalid_argument("host decay must be nonnegative");
      }
      else if (a == "--host-gib")
        c.host_bytes = size_t(std::stod(v) * 1073741824.0);
      else if (a == "--spark-width")
        c.spark_width = std::stoi(v);
      else
        throw std::invalid_argument("unknown option: " + a);
    }
    if (command == "inspect") {
      Store store(c.weights, !headers);
      if (!c.receipt.empty())
        store.verify_receipt(c.receipt);
      std::cout << store.inventory().dump(2) << "\n";
      return 0;
    }
    if (command == "serve") {
      if (socket.empty() || c.max_context < 1 || c.max_context > 1048576)
        throw std::invalid_argument("socket and valid max-context required");
      std::unique_ptr<Forward> model;
      if (!replay.empty())
        model = make_replay(replay);
#ifdef DS41_CUDA
      else
        model = make_native(c);
#else
      else
        throw std::runtime_error("this build has no CUDA forward; "
                                 "--replay-test is for protocol tests only");
#endif
#ifdef DS41_CUDA
      if (c.spark)
        model = make_spark_target(std::move(model), c.spark_width, c.spark_policy_v2);
#endif
      serve(socket, std::move(model), c.max_context, c.prefix_cache, c.prefix_any_conversation,
            c.prefill_dense_batch);
      return 0;
    }
    std::cout
        << "ds41d inspect --weights DIR [--headers-only] [--receipt FILE]\n"
        << "ds41d serve --socket PATH --weights DIR --receipt FILE --engram "
           "FILE --experts TSV [--expert-slots N] [--host-gib N] [--placement "
           "FILE] [--allow-unqualified]\n"
        << "protocol tests: ds41d serve --socket PATH --replay-test FILE\n";
    return command == "help" ? 0 : 2;
  } catch (const std::exception &e) {
    std::cerr << "ds41d: " << e.what() << "\n";
    return 1;
  }
}
