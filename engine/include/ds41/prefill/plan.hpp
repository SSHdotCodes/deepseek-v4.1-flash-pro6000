#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace ds41e::prefill {
struct Span {
  int64_t begin, end;
  int64_t size() const { return end - begin; }
  bool contains(int64_t p) const { return p >= begin && p < end; }
};
struct Layer {
  int layer;
  Span input, query, window, global, capture;
};
// Clipping applies only to new rows. Predecessors come from retained live rings.
inline std::vector<Layer> plan(int64_t begin, int64_t rows,
                               int64_t logits_tail = 1) {
  if (begin < 0 || rows < 1 || logits_tail < 1 || logits_tail > rows ||
      begin > std::numeric_limits<int64_t>::max() - rows)
    throw std::invalid_argument("prefill extent/tail");
  int64_t end = begin + rows;
  auto tail = [&](int64_t width) { return Span{std::max(begin, end-width), end}; };
  std::vector<Layer> out;
  for (int l=0;l<40;++l) {
    auto q = tail(l<20 ? rows : l==39 ? logits_tail :
        std::max<int64_t>(128,logits_tail+127)+(38-l)*127);
    auto kv = l<20 ? Span{begin,end} : tail(std::max<int64_t>(128,q.size()+127));
    bool owner=l==2||l==8||l==14||l==20;
    out.push_back({l,l<=20?Span{begin,end}:kv,q,kv,
        owner?Span{begin,end}:Span{end,end},l>=37?tail(128):Span{end,end}});
  }
  return out;
}
struct ExpertRows { int expert; std::vector<int32_t> slots; };
// Destination slot is token*6+route_rank. IO order never becomes sum order.
inline std::vector<ExpertRows> group(const int32_t *ids, size_t rows) {
  if (!ids || !rows || rows>size_t(INT32_MAX/6))
    throw std::invalid_argument("prefill routes");
  std::array<std::vector<int32_t>,384> buckets;
  for(size_t r=0;r<rows;++r) {
    std::array<bool,384> seen{};
    for(int s=0;s<6;++s) {
      int e=ids[r*6+s];
      if(e<0||e>=384||seen[e])throw std::invalid_argument("invalid/duplicate route");
      seen[e]=true;buckets[e].push_back(int32_t(r*6+s));
    }
  }
  std::vector<ExpertRows> out;
  for(int e=0;e<384;++e)if(!buckets[e].empty())out.push_back({e,std::move(buckets[e])});
  return out;
}
// Token identity, not conversation_id, authorizes reuse. A sampled token does
// not enter the retained prefix until the target has consumed and committed it.
class Prefix {
  std::vector<int> ids_;
  uint64_t epoch_=0;
public:
  enum class Match { Exact, Append, Rebuild };
  Match match(const std::vector<int>& full) const {
    if(full.size()<ids_.size()||!std::equal(ids_.begin(),ids_.end(),full.begin()))
      return Match::Rebuild;
    return full.size()==ids_.size()?Match::Exact:Match::Append;
  }
  void commit(const std::vector<int>& input,size_t accepted) {
    if(accepted>input.size())throw std::invalid_argument("prefix accepted rows");
    ids_.insert(ids_.end(),input.begin(),input.begin()+accepted);
  }
  void reset(){ids_.clear();++epoch_;}
  const std::vector<int>& ids()const{return ids_;}
  size_t size()const{return ids_.size();}
  uint64_t epoch()const{return epoch_;}
};
}
