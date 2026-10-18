#include "ds41/engine.hpp"
#include "ds41/native_store.hpp"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace ds41e;
template <class F> void fails(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  assert(failed);
}
int main() {
  Rng r{0};
  // Independent Philox known answer with counter[3]=0x53500000 v2 domain.
  assert(r.word(0, 0, Purpose::Target, 0) == 0xe0481467u);
  assert(r.word(0, 0, Purpose::Target, 1) == 0x8ddfd609u);
  assert(r.word(0, 0, Purpose::Target, 2) == 0xb08e446fu);
  assert(r.word(0, 0, Purpose::Target, 3) == 0x1852b5abu);
  assert(r.uniform(0, 0, Purpose::Target) == 0.8760998513862246);
  assert(r.word(0, 0, Purpose::Target) != r.word(0, 0, Purpose::Draft));
  assert(r.word(1, 0, Purpose::Target) != r.word(0, 0, Purpose::Target));
  assert(r.word(0, 1, Purpose::Target) != r.word(0, 0, Purpose::Target));
  fails([&] { r.uniform(0, 0, Purpose::Target, 8589934592ULL); });
  fails([&] { r.word(0, 6, Purpose::Target); });
  AttemptCounter nonce(UINT64_MAX - 1);
  assert(nonce.take() == UINT64_MAX - 1 && nonce.take() == UINT64_MAX);
  fails([&] { nonce.take(); });
  auto p = probabilities({0, 0, 0, 0}, 1, 1);
  assert(p[0] == .25);
  assert(probabilities({2, 1, 0}, 0, 1) == Distribution({1, 0, 0}));
  assert(probabilities({0, 0, 0, 0}, 1, .5) == Distribution({.5, .5, 0, 0}));
  assert(probabilities({0, 1e-6f}, 1e-6, 1) ==
         probabilities({0, 1e-6f}, 1e-5, 1));
  fails([] { probabilities({NAN}, 1, 1); });
  fails([] { probabilities({0}, -1, 1); });
  fails([&] { sample({0, 0}, r, 0, 0, Purpose::Target); });
  std::vector<std::pair<Distribution, Distribution>> pairs = {
      {{.1, .2, .3, .4}, {.4, .3, .2, .1}},
      {{.2, .3, .5, 0}, {0, 0, 0, 1}},
      {{.25, .25, .25, .25}, {.25, .25, .25, .25}},
      {{1e-30, .5, .25, .25 - 1e-30}, {.5, 1e-30, .25, .25 - 1e-30}}};
  const int trials = 200000;
  const std::vector<Distribution> greedy_rows{
      {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};
  for (uint64_t seed : {0, 1, 42, 99}) {
    auto g = reject_sample({2, 0}, {}, greedy_rows, Rng{seed}, 100, 0);
    assert(g.emitted == std::vector<int>({2, 1}) && g.accepted == 1 &&
           g.accepted_input_rows == 2 && g.rejected);
    auto accepted = reject_sample({2}, {}, {greedy_rows[0], greedy_rows[1]},
                                  Rng{seed}, 100, 0);
    assert(accepted.emitted == std::vector<int>({2, 1}) && !accepted.rejected &&
           accepted.accepted_input_rows == 2);
  }
  uint64_t round = 0;
  for (auto pair : pairs) {
    std::vector<int> counts(4);
    for (int i = 0; i < trials; ++i) {
      uint64_t pos = round++;
      int draft = sample(pair.second, r, pos, 0, Purpose::Draft);
      auto v = reject_sample({draft}, {pair.second}, {pair.first, pair.first},
                             r, pos, 1);
      ++counts.at(v.emitted.at(0));
    }
    for (int j = 0; j < 4; ++j) {
      double expected = trials * pair.first[j],
             sigma = std::sqrt(expected * (1 - pair.first[j]));
      assert(std::abs(counts[j] - expected) < 6 * sigma + 2);
    }
  }
  auto base = std::filesystem::temp_directory_path() /
              ("ds41-core-" + std::to_string(getpid()));
  std::filesystem::create_directory(base);
  auto write = [&](const std::string &p, const std::string &s) {
    std::ofstream f(base / p, std::ios::binary);
    f << s;
  };
  write("aux.bin", "abc");
  auto payload_sha = sha256_file((base / "aux.bin").string());
  assert(payload_sha ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  write("aux.json",
        Json({{"schema", "ds41-auxiliary-v1"},
              {"files",
               {{"rope_export", {{"bytes", 3}, {"sha256", payload_sha}}}}}})
            .dump());
  auto receipt_sha = sha256_file((base / "aux.json").string());
  std::map<std::string, std::string> roles{
      {"rope_export", (base / "aux.bin").string()}};
  authenticate_files((base / "aux.json").string(), receipt_sha, roles);
  write("aux.bin", "abd");
  fails([&] {
    authenticate_files((base / "aux.json").string(), receipt_sha, roles);
  });
  write("aux.bin", "abc");
  write("aux.json", "{}");
  fails([&] {
    authenticate_files((base / "aux.json").string(), receipt_sha, roles);
  });
  validate_spans(8192, {{0, 4096}, {4096, 4096}});
  fails([] { validate_spans(8192, {{0, 4096}, {2048, 4096}}); });
  fails([] { validate_spans(8192, {{4096, 8192}}); });
  fails([] { validate_spans(8192, {{UINT64_MAX, 4096}}); });
  write("replay.json", Json({{"schema", "ds41-replay-test-v1"},
                             {"logits", {{1, 0}}},
                             {"eos", 1}})
                           .dump());
  auto replay = make_replay((base / "replay.json").string());
  ds41_target_weights_v2 shared;
  std::memset(&shared, 0x5a, sizeof(shared));
  auto canary = shared;
  assert(ds41_target_query(replay.get(), DS41_TARGET_WEIGHTS, 1, &shared,
                           sizeof(shared)) == -1);
  assert(!std::memcmp(&shared, &canary, sizeof(shared)));
  assert(ds41_target_query(replay.get(), DS41_TARGET_WEIGHTS, 2, &shared,
                           sizeof(shared) - 1) == -1);
  assert(!std::memcmp(&shared, &canary, sizeof(shared)));
  assert(ds41_target_query(replay.get(), DS41_TARGET_WEIGHTS, 2, &shared,
                           sizeof(shared)) == -2);
  assert(!std::memcmp(&shared, &canary, sizeof(shared)));
  Json header = {
      {"model.embed.weight",
       {{"dtype", "BF16"}, {"shape", {2, 2}}, {"data_offsets", {0, 8}}}}};
  auto raw = header.dump();
  std::string file(8, '\0');
  for (int i = 0; i < 8; ++i)
    file[i] = char(uint64_t(raw.size()) >> (8 * i));
  file += raw;
  file += std::string("abcdefgh", 8);
  write("tiny.safetensors", file);
  write("model.safetensors.index.json",
        Json({{"weight_map", {{"model.embed.weight", "tiny.safetensors"}}}})
            .dump());
  Store store(base.string());
  auto t = store.tensor("embed.weight");
  assert(t.bytes == 8 && t.shape == std::vector<uint64_t>({2, 2}));
  assert(store.read(t) ==
         std::vector<uint8_t>({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}));
  assert(store.inventory().at("resident_bytes") == 8);
  fails([&] { store.verify_receipt((base / "missing").string()); });
  write("tiny.safetensors", file.substr(0, file.size() - 1));
  fails([&] { Store incomplete(base.string()); });
  Store metadata(base.string(), false);
  fails([&] { metadata.read(metadata.tensor("embed.weight")); });
  write("model.safetensors.index.json",
        Json({{"weight_map", {{"model.embed.weight", "../bad"}}}}).dump());
  fails([&] { Store unsafe(base.string()); });
  std::filesystem::remove_all(base);
  std::cout << "core: Philox known-answer, 800000 lossless first-token draws, "
               "loader/corruption checks passed\n";
}
