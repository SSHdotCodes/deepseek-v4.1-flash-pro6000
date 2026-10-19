#include "ds41/engine.hpp"
#include "ds41/native_store.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <poll.h>
#include <random>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
namespace ds41e {
namespace {
volatile std::sig_atomic_t stopping = 0;
void stop_signal(int) { stopping = 1; }
double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
int integer(const Json &j, const char *key, int def, int lo, int hi) {
  if (!j.contains(key))
    return def;
  auto v = j.at(key);
  if (!v.is_number_integer())
    throw std::invalid_argument(std::string(key) + " must be an integer");
  auto n = v.get<int64_t>();
  if (n < lo || n > hi)
    throw std::invalid_argument(std::string(key) + " out of bounds");
  return int(n);
}
std::vector<int> token_ids(const Json &j, int vocab) {
  if (!j.is_array() || j.empty() || j.size() > 1048576)
    throw std::invalid_argument("token_ids must be nonempty (max 1M)");
  std::vector<int> ids;
  for (auto id : j) {
    if (!id.is_number_integer())
      throw std::invalid_argument("token id must be an integer");
    auto i = id.get<int64_t>();
    if (i < 0 || i >= vocab)
      throw std::invalid_argument("token id outside vocabulary");
    ids.push_back(int(i));
  }
  return ids;
}
std::vector<uint8_t> base64_decode(const std::string &text) {
  static const auto table = [] {
    std::array<int16_t, 256> t;
    t.fill(-1);
    const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[uint8_t(alphabet[i])] = int16_t(i);
    return t;
  }();
  std::vector<uint8_t> out;
  out.reserve(text.size() / 4 * 3);
  uint32_t buffer = 0;
  int bits = 0;
  size_t padding = 0;
  for (char ch : text) {
    if (ch == '=') { ++padding; continue; }
    const int v = table[uint8_t(ch)];
    if (v < 0 || padding) throw std::invalid_argument("image pixels must be base64");
    buffer = (buffer << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(uint8_t(buffer >> bits));
    }
  }
  if (padding > 2 || (text.size() % 4))
    throw std::invalid_argument("image pixels must be padded base64");
  return out;
}
// The prompt's images: geometry and resized RGB pixels from the API (see ImageSpan), checked against the
// token ids. Spans are sorted, disjoint, inside the prompt and hold the image token id at every position.
std::vector<ImageSpan> image_spans(const Json &op, const std::vector<int> &ids) {
  std::vector<ImageSpan> spans;
  if (!op.contains("images") || op.at("images").is_null())
    return spans;
  const auto &list = op.at("images");
  if (!list.is_array() || list.size() > 64)
    throw std::invalid_argument("images must be an array of at most 64 entries");
  int64_t previous_end = 0;
  for (const auto &j : list) {
    if (!j.is_object())
      throw std::invalid_argument("image entries must be objects");
    for (const char *key : {"start", "n_vit_h", "n_vit_w", "n_llm_h", "n_llm_w", "pixels"})
      if (!j.contains(key))
        throw std::invalid_argument(std::string("image entry needs ") + key);
    ImageSpan img;
    if (!j.at("start").is_number_integer())
      throw std::invalid_argument("image start must be an integer");
    img.start = j.at("start").get<int64_t>();
    img.n_vit_h = integer(j, "n_vit_h", 0, 1, 1 << 14);
    img.n_vit_w = integer(j, "n_vit_w", 0, 1, 1 << 14);
    img.n_llm_h = integer(j, "n_llm_h", 0, 1, ImageSpan::max_tokens);
    img.n_llm_w = integer(j, "n_llm_w", 0, 1, ImageSpan::max_tokens);
    if (img.n_llm_h != (img.n_vit_h + 2) / 3 || img.n_llm_w != (img.n_vit_w + 2) / 3 ||
        img.tokens() > ImageSpan::max_tokens)
      throw std::invalid_argument("image grid does not match the vision geometry");
    if (!j.at("pixels").is_string())
      throw std::invalid_argument("image pixels must be a base64 string");
    img.pixels = base64_decode(j.at("pixels").get<std::string>());
    if (img.pixels.size() != size_t(img.n_vit_h) * 14 * img.n_vit_w * 14 * 3)
      throw std::invalid_argument("image pixels must be n_vit_h*14 rows of n_vit_w*14 RGB pixels");
    if (img.start < previous_end || img.end() > int64_t(ids.size()))
      throw std::invalid_argument("image spans must be sorted, disjoint and inside the prompt");
    for (int64_t pos = img.start; pos < img.end(); ++pos)
      if (ids[pos] != ImageSpan::token_id)
        throw std::invalid_argument("image span positions must hold the image token id");
    const std::string geometry = std::to_string(img.n_vit_h) + "x" + std::to_string(img.n_vit_w) + ":";
    img.digest = sha256_bytes(geometry.data(), geometry.size()) + sha256_bytes(img.pixels.data(), img.pixels.size());
    previous_end = img.end();
    spans.push_back(std::move(img));
  }
  return spans;
}
// Prefix-cache identity per position: the token id, or for image positions the image digest and offset, so
// two different images with the same geometry (identical token ids) never share cached state.
std::vector<uint64_t> position_keys(const std::vector<int> &ids, const std::vector<ImageSpan> &images) {
  std::vector<uint64_t> keys(ids.begin(), ids.end());
  for (const auto &img : images) {
    const uint64_t base = std::stoull(img.digest.substr(0, 15), nullptr, 16) ^
                          std::stoull(img.digest.substr(64, 15), nullptr, 16);
    for (int64_t pos = img.start; pos < img.end(); ++pos)
      keys[pos] = (uint64_t(1) << 63) | ((base + uint64_t(pos - img.start) * 0x9E3779B97F4A7C15ull) >> 1);
  }
  return keys;
}
bool send_json(int fd, const Json &event,
               const std::function<bool()> &cancelled = {}) {
  auto raw = event.dump() + "\n";
  size_t pos = 0;
  const bool terminal = event.value("type", std::string()) == "finish";
  double deadline = now() + 2.0;
  while (pos < raw.size()) {
    if (!terminal && (stopping || (cancelled && cancelled()))) {
      if (!pos)
        return false;
      deadline = std::min(deadline, now() + .25);
    }
    if (now() >= deadline) {
      // Never append a finish to an incomplete JSON frame. A stalled peer
      // becomes a transport error, and cannot retain the model lock forever.
      shutdown(fd, SHUT_RDWR);
      return false;
    }
    pollfd p{fd, POLLOUT, 0};
    int ready = poll(&p, 1, 100);
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready < 0 || p.revents & (POLLERR | POLLHUP | POLLNVAL))
      return false;
    if (!ready)
      continue;
    auto n = send(fd, raw.data() + pos, raw.size() - pos, MSG_NOSIGNAL);
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
      continue;
    if (n <= 0)
      return false;
    pos += n;
  }
  return true;
}
double log_normalizer(const std::vector<float> &logits) {
  double mx = logits.at(argmax(logits)), sum = 0;
  for (float value : logits)
    sum += std::exp(double(value) - mx);
  return mx + std::log(sum);
}
std::string logits_digest(const std::vector<float> &logits) {
  static_assert(sizeof(float) == 4, "FP32 logit fingerprint required");
  const uint32_t one = 1;
  if (*reinterpret_cast<const uint8_t *>(&one) == 1)
    return sha256_bytes(logits.data(), logits.size() * sizeof(float));
  std::vector<uint8_t> bytes(logits.size() * 4);
  for (size_t i = 0; i < logits.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &logits[i], 4);
    for (int k = 0; k < 4; ++k)
      bytes[4 * i + k] = uint8_t(bits >> (8 * k));
  }
  return sha256_bytes(bytes.data(), bytes.size());
}
std::vector<int> verify_widths(const Json &op) {
  if (!op.contains("verify_width"))
    return {1};
  auto widths = op.at("verify_width");
  if (!widths.is_array())
    widths = Json::array({widths});
  if (widths.empty() || widths.size() > 64)
    throw std::invalid_argument("verify_width pattern must have 1..64 entries");
  std::vector<int> out;
  for (const auto &width : widths) {
    if (!width.is_number_integer() || width.get<int64_t>() < 1 ||
        width.get<int64_t>() > 6)
      throw std::invalid_argument("verify_width entries must be integers 1..6");
    out.push_back(width.get<int>());
  }
  return out;
}
// Read no more than one bounded frame at a time, preserving pipelined controls.
bool read_line(int fd, std::string &pending, std::string &line) {
  for (;;) {
    auto end = pending.find('\n');
    if (end != std::string::npos) {
      line = pending.substr(0, end);
      pending.erase(0, end + 1);
      return true;
    }
    if (stopping)
      return false;
    if (pending.size() > (256u << 20))
      throw std::invalid_argument("request frame exceeds 256 MiB");
    pollfd p{fd, POLLIN, 0};
    int r = poll(&p, 1, 100);
    if (r < 0 && errno == EINTR)
      continue;
    if (r < 0 || p.revents & (POLLERR | POLLNVAL))
      return false;
    if (!r)
      continue;
    char buf[16384];
    auto n = recv(fd, buf, sizeof(buf), 0);
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
      continue;
    if (n <= 0)
      return false;
    pending.append(buf, n);
  }
}
bool disconnected(int fd) {
  char c;
  int n = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  return n == 0 ||
         (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
}
struct Live {
  std::atomic<bool> cancel{false};
};
class Service {
  std::unique_ptr<Forward> model;
  std::mutex serial, registry_mutex;
  std::map<std::string, std::shared_ptr<Live>> live;
  int context;
  bool prefix_cache, prefix_any = false;
  size_t prefill_dense_batch = 1;
  std::atomic<uint64_t> sequence{0};
  // One live native state. Only consumed inputs are recorded here; a sampled
  // correction/bonus that has not been fed back is never reusable KV.
  struct Prefix {
    std::string conversation;
    std::vector<int> inputs;
    std::vector<uint64_t> keys; // position_keys of inputs: what reuse compares
    std::vector<float> logits;
    std::shared_ptr<Forward::Snapshot> boundary;
    size_t boundary_len = 0;
    std::vector<float> boundary_logits;
    bool same_conversation(const std::string &id, bool any) const {
      return any || (!id.empty() && id == conversation);
    }
    bool matches(const std::string &id, const std::vector<uint64_t> &prompt, bool any) const {
      return same_conversation(id, any) && !keys.empty() &&
             keys.size() <= prompt.size() &&
             std::equal(keys.begin(), keys.end(), prompt.begin());
    }
  } prefix;

public:
  std::atomic<int> connections{0};
  Service(std::unique_ptr<Forward> m, int c, bool reuse, bool any = false, size_t dense_batch = 1)
      : model(std::move(m)), context(c), prefix_cache(reuse), prefix_any(any),
        prefill_dense_batch(dense_batch) {}
  void handle(int fd) {
    std::string id, pending, line;
    std::shared_ptr<Live> state;
    int prompt = 0, completed = 0;
    bool finished = false, registered = false;
    std::unique_lock<std::mutex> run(serial, std::defer_lock);
    auto finish = [&](std::string reason, Json error = nullptr,
                      Json stats = Json::object()) {
      if (finished)
        return;
      finished = true;
      Json usage = {{"prompt_tokens", prompt}, {"completion_tokens", completed}};
      for (const auto &field : {
               std::pair<const char *, const char *>{"reused_prompt_tokens", "cached_prompt_tokens"},
               {"prefill_ms", "prefill_ms"}, {"generation_ms", "decode_ms"},
               {"drafted", "draft_tokens"}, {"accepted", "accepted_draft_tokens"}})
        if (stats.contains(field.first))
          usage[field.second] = stats.at(field.first);
      send_json(
          fd, {{"type", "finish"},
               {"finish_reason", reason},
               {"usage", usage},
               {"error", error},
               {"stats", stats}});
    };
    try {
      while (read_line(fd, pending, line)) {
        auto op = Json::parse(line);
        auto verb = op.value("op", std::string());
        if (verb == "cancel") {
          bool found = false;
          {
            std::lock_guard<std::mutex> lock(registry_mutex);
            auto it = live.find(op.value("request_id", std::string()));
            if (it != live.end()) {
              it->second->cancel = true;
              found = true;
            }
          }
          if (!send_json(fd, {{"ok", found}}))
            break;
          continue;
        }
        if (verb != "submit" && verb != "score")
          throw std::invalid_argument("unknown op");
        auto ids = token_ids(op.at("token_ids"), model->vocab());
        prompt = int(ids.size());
        if (prompt > context)
          throw std::invalid_argument("prompt exceeds max_context");
        const auto images = image_spans(op, ids);
        const auto keys = position_keys(ids, images);
        if (!images.empty() && verb == "score")
          throw std::invalid_argument("unsupported: score requests with images");
        const std::string draft_mode = op.value("draft", std::string("auto"));
        if (draft_mode != "off" && draft_mode != "auto" && draft_mode != "on")
          throw std::invalid_argument(
              "unsupported: draft mode must be off/auto/on");
        if (draft_mode == "on" && !model->drafter())
          throw std::invalid_argument(
              "unsupported: native drafting is not enabled in this build");
        std::string conversation;
        if (op.contains("conversation_id") &&
            !op.at("conversation_id").is_null()) {
          conversation = op.at("conversation_id").get<std::string>();
          if (conversation.size() > 4096)
            throw std::invalid_argument("conversation_id exceeds 4096 bytes");
        }
        Json s = op.value("sampling", Json::object());
        if (!s.is_object())
          throw std::invalid_argument("sampling must be an object");
        Sampling sampling;
        sampling.temperature = s.value("temperature", 1.0);
        sampling.top_p = s.value("top_p", 1.0);
        if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
            !std::isfinite(sampling.top_p) || sampling.top_p <= 0 ||
            sampling.top_p > 1)
          throw std::invalid_argument("invalid temperature/top_p");
        sampling.max_tokens = integer(s, "max_tokens", 4096, 0, 1048576);
        sampling.logprobs =
            integer(s, "logprobs", 0, 0, std::min(1024, model->vocab()));
        if (s.contains("seed") && !s.at("seed").is_null()) {
          if (!s.at("seed").is_number_integer() ||
              (!s.at("seed").is_number_unsigned() &&
               s.at("seed").get<int64_t>() < 0))
            throw std::invalid_argument("seed must be a uint64");
          sampling.seed = s.at("seed").get<uint64_t>();
        } else {
          std::random_device random;
          sampling.seed = (uint64_t(random()) << 32) ^ random();
        }
        if (s.contains("stop_token_ids")) {
          if (!s.at("stop_token_ids").is_array())
            throw std::invalid_argument("stop_token_ids must be an array");
          for (auto x : s.at("stop_token_ids")) {
            if (!x.is_number_integer() || x.get<int64_t>() < 0 ||
                x.get<int64_t>() >= model->vocab())
              throw std::invalid_argument("invalid stop token");
            sampling.stop.push_back(x.get<int>());
          }
        }
        if (op.contains("request_id") && !op.at("request_id").is_null())
          id = op.at("request_id").get<std::string>();
        if (id.empty())
          id = "ds41-" + std::to_string(++sequence);
        if (id.size() > 256)
          throw std::invalid_argument("request_id too long");
        state = std::make_shared<Live>();
        {
          std::lock_guard<std::mutex> lock(registry_mutex);
          if (live.size() >= 8)
            throw std::runtime_error("engine queue full");
          if (!live.emplace(id, state).second) {
            state.reset();
            throw std::invalid_argument("request_id already live");
          }
          registered = true;
        }
        auto cancelled = [&] {
          return stopping || state->cancel.load() || disconnected(fd);
        };
        while (!run.try_lock()) {
          if (cancelled())
            break;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (cancelled()) {
          finish("cancelled");
          break;
        }
        double started = now(), first = 0, last = 0, prefill_end = 0;
        std::vector<float> logits;
        if (verb == "score") {
          prefix = {};
          model->reset();
          int from = integer(op, "score_from", 0, 0, prompt - 1),
              top = integer(op, "top", 64, 1, std::min(1024, model->vocab()));
          auto mode = op.value("mode", std::string("verify"));
          if (mode != "verify" && mode != "prefill")
            throw std::invalid_argument("unknown score mode");
          auto widths = verify_widths(op);
          if (mode != "verify")
            throw std::invalid_argument(
                "unsupported: cold prefill score is not implemented");
          if (op.value("return_routes", false))
            throw std::invalid_argument(
                "unsupported: route events are not implemented");
          size_t cycle = 0;
          for (int begin = 0; begin < prompt && !cancelled();) {
            int count =
                std::min(widths[cycle++ % widths.size()], prompt - begin);
            std::vector<int> chunk(ids.begin() + begin,
                                   ids.begin() + begin + count);
            auto rows = model->verify(chunk);
            if (rows.size() != size_t(count))
              throw std::runtime_error(
                  "target verify returned wrong row count");
            model->commit(count);
            bool sent = true;
            for (int row = 0; row < count && !cancelled(); ++row) {
              int i = begin + row;
              if (i < from)
                continue;
              auto &l = rows[row];
              double normalizer = log_normalizer(l);
              auto top_ids = top_order(l, top);
              Json logp = Json::array();
              for (int t : top_ids)
                logp.push_back(double(l[t]) - normalizer);
              Json event = {{"type", "score"},
                            {"position", i},
                            {"top1", argmax(l)},
                            {"top_id", top_ids},
                            {"top_logprob", logp},
                            {"logsumexp", normalizer},
                            {"logits_sha256", logits_digest(l)}};
              if (op.value("return_logits", false))
                event["logits"] = l;
              if (i + 1 < prompt) {
                event["next_token_id"] = ids[i + 1];
                event["next_logprob"] = double(l[ids[i + 1]]) - normalizer;
              }
              if (!send_json(fd, event, cancelled)) {
                sent = false;
                break;
              }
            }
            begin += count;
            if (!sent)
              break;
          }
          auto stats = model->counters();
          stats["prefill_mode"] = "verify";
          stats["verify_width"] = widths;
          finish(cancelled() ? "cancelled" : "stop", nullptr, stats);
          break;
        }
        if (sampling.max_tokens == 0) {
          finish("length");
          break;
        }
        size_t reused = prefix_cache && prefix.matches(conversation, keys, prefix_any)
                            ? prefix.inputs.size()
                            : 0;
        bool rewound = false;
        if (!reused && prefix_cache && prefix.boundary && prefix.boundary_len &&
            prefix.same_conversation(conversation, prefix_any) &&
            prefix.boundary_len <= ids.size() && prefix.boundary_len <= prefix.inputs.size() &&
            std::equal(prefix.keys.begin(), prefix.keys.begin() + prefix.boundary_len, keys.begin())) {
          try {
            model->restore(*prefix.boundary);
            prefix.inputs.resize(prefix.boundary_len);
            prefix.keys.resize(prefix.boundary_len);
            prefix.logits = prefix.boundary_logits;
            reused = prefix.boundary_len;
            rewound = true;
          } catch (const std::exception &) {
            reused = 0;
          }
        }
        if (reused) {
          if (!rewound) model->begin_request();
          logits = prefix.logits;
        } else {
          prefix = {};
          model->reset();
        }
        prefix.conversation = conversation;
        model->begin_prompt(prompt, reused != 0);
        auto consume = [&](int token) {
          auto next = model->verify({token}).at(0);
          model->commit(1);
          prefix.keys.push_back(keys[prefix.inputs.size()]);
          prefix.inputs.push_back(token);
          prefix.logits = next;
          logits = std::move(next);
        };
        // Prefill the suffix in the largest segments the model accepts (one activation arena must fit
        // beside the live decode cache): the first in the prompt's mode, the rest as appends. Prompts or
        // remainders the model cannot prefill use the sequential consume path.
        size_t next = reused;
        while (next < ids.size() && !cancelled()) {
          const PrefillMode mode = next ? PrefillMode::Append : PrefillMode::Cold;
          size_t rows = ids.size() - next;
          if (!model->supports_prefill(mode, rows)) {
            size_t lo = 0, hi = rows;  // largest supported multiple of 256 below rows
            while (hi - lo > 256) {
              size_t mid = (lo + (hi - lo) / 2) / 256 * 256;
              if (mid <= lo) mid = lo + 256;
              if (model->supports_prefill(mode, mid)) lo = mid; else hi = mid;
            }
            rows = lo;
          }
          if (!rows)
            break;
          PrefillOptions options;
          options.mode = mode;
          options.dense_batch_rows = prefill_dense_batch;
          options.warm = next + rows == ids.size();
          options.images = images.empty() ? nullptr : &images;
          std::vector<int> segment(ids.begin() + next, ids.begin() + next + rows);
          auto result = model->prefill(segment, options);
          if (result.begin_position != int64_t(next) ||
              result.consumed_rows != rows || result.logits.size() != 1 ||
              result.logits.back().size() != size_t(model->vocab()))
            throw std::runtime_error("prefill returned inconsistent committed rows");
          prefix.inputs.insert(prefix.inputs.end(), segment.begin(), segment.end());
          prefix.keys.insert(prefix.keys.end(), keys.begin() + next, keys.begin() + next + rows);
          prefix.logits = result.logits.back();
          logits = std::move(result.logits.back());
          next += rows;
        }
        // The sequential path has no image embeddings: an image the prefill could not take is an error.
        for (const auto &img : images)
          if (img.end() > int64_t(next) && !cancelled())
            throw std::runtime_error("image prompt needs a prefill segment, and it did not fit beside "
                                     "the expert cache");
        for (size_t i = next; i < ids.size(); ++i) {
          if (cancelled())
            break;
          consume(ids[i]);
        }
        const size_t consumed_prompt = prefix.inputs.size() - reused;
        const bool prompt_complete = prefix.inputs.size() == ids.size();
        model->end_prompt(prompt_complete);
        if (!prompt_complete) {
          prefix = {};
        } else if (prefix_cache) {
          try {
            prefix.boundary = model->snapshot();
            prefix.boundary_len = prefix.boundary ? prefix.inputs.size() : 0;
            prefix.boundary_logits = prefix.boundary ? prefix.logits : std::vector<float>{};
          } catch (const std::exception &) {
            prefix.boundary.reset(); prefix.boundary_len = 0;
          }
        }
        prefill_end = now();
        const auto prefill_counters = model->counters();
        Rng rng{sampling.seed};
        AttemptCounter attempts;
        DraftProvider *drafter =
            draft_mode == "off" ? nullptr : model->drafter();
        std::string reason = "length";
        uint64_t cycles = 0, drafted = 0, accepted = 0, verified_accepted = 0;
        double draft_ms = 0, verify_ms = 0;
        auto emit = [&](int token, const std::vector<float> &row) {
          if (cancelled()) {
            reason = "cancelled";
            return false;
          }
          if (token == model->eos() ||
              std::find(sampling.stop.begin(), sampling.stop.end(), token) !=
                  sampling.stop.end()) {
            reason = "stop";
            return false;
          }
          Json event = {{"type", "token"}, {"token_id", token}};
          if (sampling.logprobs) {
            const double normalizer = log_normalizer(row);
            event["logsumexp"] = normalizer;
            event["logits_sha256"] = logits_digest(row);
            event["logprob"] = double(row[token]) - normalizer;
            event["top_logprobs"] = Json::array();
            for (int t : top_order(row, sampling.logprobs))
              event["top_logprobs"].push_back(
                  {{"token_id", t}, {"logprob", double(row[t]) - normalizer}});
          }
          if (!send_json(fd, event, cancelled)) {
            reason = "cancelled";
            return false;
          }
          last = now();
          if (!first)
            first = last;
          ++completed;
          return true;
        };
        if (!cancelled()) {
          // Initial target anchor has its own nonce, including greedy/stops.
          const uint64_t nonce = attempts.take();
          ++cycles;
          int anchor = sampling.temperature == 0
                           ? argmax(logits)
                           : sample(probabilities(logits, sampling.temperature,
                                                  sampling.top_p),
                                    rng, nonce, 0, Purpose::Target);
          bool active = emit(anchor, logits);
          while (active && completed < sampling.max_tokens &&
                 prefix.inputs.size() < size_t(context)) {
            if (cancelled()) {
              reason = "cancelled";
              break;
            }
            const int maximum = std::min({6, sampling.max_tokens - completed,
                                          context - int(prefix.inputs.size())});
            // Burn the nonce before width selection, including failed attempts.
            const uint64_t nonce = attempts.take();
            ++cycles;
            SpeculativeStep step(*model, drafter, anchor,
                                 int64_t(prefix.inputs.size()), sampling,
                                 uint64_t(prompt + completed), nonce, maximum);
            const auto &result = step.result();
            drafted += step.inputs().size() - 1;
            verified_accepted += result.accepted;
            draft_ms += step.observation().draft_ms;
            verify_ms += step.observation().verify_ms;
            size_t delivered = 0;
            for (size_t k = 0; k < result.emitted.size(); ++k) {
              if (!emit(result.emitted[k], step.rows()[k])) {
                active = false;
                break;
              }
              ++delivered;
            }
            if (cancelled() && completed < sampling.max_tokens) {
              reason = "cancelled";
              active = false;
            }
            const bool terminal = !active || completed == sampling.max_tokens;
            const size_t consumed = step.finish(delivered, terminal,
                                                 reason == "cancelled");
            accepted += std::min(delivered, result.accepted);
            prefix.inputs.insert(prefix.inputs.end(), step.inputs().begin(),
                                 step.inputs().begin() + consumed);
            prefix.keys.insert(prefix.keys.end(), step.inputs().begin(),
                               step.inputs().begin() + consumed);
            prefix.logits = step.rows()[consumed - 1];
            anchor = result.emitted.back();
          }
        } else
          reason = "cancelled";
        auto stats = model->counters();
        stats["seed"] = sampling.seed;
        stats["sampling_profile"] = "spark-host-fp64-temperature-floor-1e-5";
        stats["rng_protocol"] = "spark-v2-monotone-attempt-slot-53bit";
        stats["score_profile"] = "fp64-logsumexp-from-fp32-logits";
        stats["prefill_ms"] = (prefill_end - started) * 1000;
        stats["prefill_mode"] = "verify";
        stats["reused_prompt_tokens"] = reused;
        stats["prefix_cache_requested"] = prefix_cache;
        stats["prefill_input_tokens"] = consumed_prompt;
        stats["prefill_tier_counters"] =
            tier_counters(Json::object(), prefill_counters);
        stats["decode_tier_counters"] = tier_counters(prefill_counters, stats);
        stats["ttft_ms"] = first ? (first - started) * 1000 : 0;
        stats["decode_ms"] = first ? (last - first) * 1000 : 0;
        // The public usage rate divides all completion tokens by this span,
        // including the initial anchor. The steady-state metric below instead
        // divides subsequent output intervals by first-to-last elapsed time.
        stats["generation_ms"] = first ? (last - prefill_end) * 1000 : 0;
        stats["first_token_time"] = first;
        stats["last_token_time"] = last;
        stats["t_first_token"] = first;
        stats["t_last_token"] = last;
        stats["cycles"] = cycles;
        stats["draft_mode"] = drafter ? "native-candidate" : "off";
        stats["draft_ms"] = draft_ms;
        stats["target_verify_ms"] = verify_ms;
        stats["decode_tps"] = completed > 1 && last > first
                                  ? (completed - 1) / (last - first)
                                  : 0;
        stats["drafted"] = drafted;
        stats["accepted"] = accepted;
        stats["verified_accepted"] = verified_accepted;
        stats["acceptance"] =
            drafted ? double(accepted) / double(drafted) : 0.0;
        finish(reason, nullptr, stats);
        break;
      }
    } catch (const std::exception &e) {
      // Error paths cannot promise a reusable native state. Poison checks in
      // the target still decide whether a subsequent reset is legal.
      if (run.owns_lock())
        prefix = {};
      finish("error", e.what());
    }
    if (registered) {
      std::lock_guard<std::mutex> lock(registry_mutex);
      live.erase(id);
    }
    close(fd);
    --connections;
  }
  template <class T>
  static std::vector<int> top_order(const std::vector<T> &p, int n) {
    std::vector<int> ids(p.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(
        ids.begin(), ids.begin() + n, ids.end(),
        [&](int a, int b) { return p[a] == p[b] ? a < b : p[a] > p[b]; });
    ids.resize(n);
    return ids;
  }
  static Json tier_counters(const Json &before, const Json &after) {
    Json out = Json::object();
    for (const char *key :
         {"expert_vram_hits", "expert_host_hits", "expert_nvme_misses",
          "expert_h2d_bytes", "expert_nvme_bytes", "mtp_expert_requests",
          "mtp_expert_vram_hits", "mtp_expert_host_hits", "mtp_expert_nvme_misses"}) {
      if (!after.contains(key))
        continue;
      uint64_t a = after.at(key).get<uint64_t>(),
               b = before.value(key, uint64_t(0));
      if (a < b)
        throw std::runtime_error("request tier counter moved backwards");
      out[key] = a - b;
    }
    for (const char *key :
         {"need_host_ms", "route_wait_ms", "engram_read_ms", "forward_ms"})
      if (after.contains(key))
        out[key] = after.at(key).get<double>() - before.value(key, 0.0);
    return out;
  }
};
class Replay final : public Forward {
  std::vector<std::vector<float>> rows;
  size_t position = 0, pending = 0;
  int eos_;
  int delay;
  size_t calls = 0;

public:
  explicit Replay(const std::string &path) {
    auto j = read_json(path);
    if (j.at("schema") != "ds41-replay-test-v1")
      throw std::invalid_argument("explicit test replay schema required");
    rows = j.at("logits").get<std::vector<std::vector<float>>>();
    if (rows.empty() || rows[0].empty())
      throw std::invalid_argument("empty replay");
    for (auto r : rows)
      if (r.size() != rows[0].size())
        throw std::invalid_argument("replay shape");
    eos_ = j.value("eos", -1);
    delay = j.value("delay_ms", 0);
  }
  int vocab() const override { return int(rows[0].size()); }
  int eos() const override { return eos_; }
  void begin_request() override { calls = 0; }
  void reset() override {
    begin_request();
    position = 0;
    pending = 0;
  }
  std::vector<std::vector<float>> verify(const std::vector<int> &ids) override {
    if (pending || ids.empty())
      throw std::logic_error("active/empty transaction");
    pending = ids.size();
    ++calls;
    if (delay)
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    std::vector<std::vector<float>> out;
    for (size_t i = 0; i < ids.size(); ++i)
      out.push_back(rows[(position + i) % rows.size()]);
    return out;
  }
  void commit(size_t n) override {
    if (!pending || n > pending)
      throw std::logic_error("invalid commit");
    position += n;
    pending = 0;
  }
  Json counters() const override {
    return {{"engine_mode", "REPLAY_TEST_ONLY"},
            {"committed", position},
            {"verify_calls", calls}};
  }
};
} // namespace
std::unique_ptr<Forward> make_replay(const std::string &path) {
  return std::make_unique<Replay>(path);
}
void serve(const std::string &path, std::unique_ptr<Forward> model,
           int max_context, bool prefix_cache, bool prefix_any_conversation,
           size_t prefill_dense_batch) {
  if (path.size() >= sizeof(sockaddr_un::sun_path))
    throw std::invalid_argument("Unix socket path too long");
  int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener < 0)
    throw std::runtime_error("socket failed");
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  // Refuse to replace any preexisting socket/file: another process may own it.
  if (bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) {
    close(listener);
    throw std::runtime_error(
        "socket bind failed (remove only your own stale socket): " +
        std::string(strerror(errno)));
  }
  chmod(path.c_str(), 0600);
  if (listen(listener, 64)) {
    close(listener);
    unlink(path.c_str());
    throw std::runtime_error("listen failed");
  }
  stopping = 0;
  std::signal(SIGTERM, stop_signal);
  std::signal(SIGINT, stop_signal);
  std::signal(SIGPIPE, SIG_IGN);
  Service service(std::move(model), max_context, prefix_cache, prefix_any_conversation,
                  prefill_dense_batch);
  struct Worker {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<Worker> workers;
  std::cerr << "ds41d listening " << path << "\n";
  while (!stopping) {
    for (auto i = workers.begin(); i != workers.end();)
      if (i->done->load()) {
        i->thread.join();
        i = workers.erase(i);
      } else
        ++i;
    pollfd p{listener, POLLIN, 0};
    if (poll(&p, 1, 100) <= 0)
      continue;
    int client =
        accept4(listener, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (client < 0)
      continue;
    if (service.connections >= 64) {
      close(client);
      continue;
    }
    ++service.connections;
    auto done = std::make_shared<std::atomic<bool>>(false);
    workers.push_back({std::thread([&service, client, done] {
                         service.handle(client);
                         *done = true;
                       }),
                       done});
  }
  close(listener);
  for (auto &w : workers)
    w.thread.join();
  unlink(path.c_str());
}
} // namespace ds41e
