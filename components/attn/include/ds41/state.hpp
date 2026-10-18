#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
namespace ds41 {
// Only these four layers own global records. Aliases are logical ids, never copies.
inline constexpr std::array<int,4> kv_owners{2,8,14,20};
inline int kv_source(int l){if(l<0||l>=40)return -1;if(l<2)return -1;if(l<8)return 2;if(l<14)return 8;if(l<20)return 14;return 20;}
inline int index_source(int l){if(l<0||l>=40)return -1;if(l<20)return kv_source(l);if(l<24)return 20;if(l<28)return 24;if(l<32)return 28;if(l<36)return 32;return 36;}
inline int compression(int l){return l<2||l>=40?0:l<20?2:1;}
struct VerifyState {
 int64_t committed=0;
 int64_t pending_start=-1;
 int pending_width=0;
 static constexpr int swa_capacity=134;
 static constexpr int compressor_capacity=8;
 void begin(int width){if(pending_width||width<1||width>6)throw std::invalid_argument("verify transaction width/state");pending_start=committed;pending_width=width;}
 void finish(int accepted_input_rows){
  if(!pending_width||accepted_input_rows<0||accepted_input_rows>pending_width)throw std::invalid_argument("accepted prefix");
  committed=pending_start+accepted_input_rows;pending_start=-1;pending_width=0;
 }
 int64_t visible(int ratio,int query_row=0)const{
  if(ratio<1)throw std::invalid_argument("compression ratio");
  if(pending_width){if(query_row<0||query_row>=pending_width)throw std::out_of_range("query row");return (pending_start+query_row+1)/ratio;}
  return committed/ratio;
 }
 void reset(){if(pending_width)throw std::logic_error("reset active transaction");committed=0;}
};
}
