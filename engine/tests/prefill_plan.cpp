#include "ds41/prefill/plan.hpp"
#include <cassert>
#include <iostream>
using namespace ds41e::prefill;
int main(){
  for(int64_t b:{0,1,2048,32768,100000})for(int64_t n:{1,127,128,509,600,4096,150000}) {
    auto p=plan(b,n);assert(p.size()==40);
    for(int l=0;l<40;++l){auto&w=p[l];assert(w.query.begin>=b&&w.query.end==b+n);
      assert(w.window.begin<=w.query.begin&&w.input.begin<=w.window.begin);
      if(l<20)assert(w.query.size()==n);
      if(l>20)assert(p[l-1].query.begin<=w.input.begin);
      if(l>=37)assert(w.input.begin<=w.capture.begin);
    }
    assert(p.back().query.size()==1);
    auto all=plan(b,n,n);for(auto&w:all)assert(w.query.size()==n);
    if(n>2541)assert(p[20].query.size()==2414&&p[20].window.size()==2541);
  }
  int32_t ids[]={383,0,1,5,7,9,9,7,5,1,0,383};
  auto g=group(ids,2);assert(g.size()==6&&g.front().expert==0&&g.back().expert==383);
  std::vector<int> visits(12);for(auto&e:g)for(auto slot:e.slots)++visits[slot];
  for(auto n:visits)assert(n==1);
  ids[1]=383;bool rejected=false;try{group(ids,2);}catch(...){rejected=true;}assert(rejected);
  Prefix prefix;prefix.commit({0,2,3},2);assert(prefix.match({0,2})==Prefix::Match::Exact);
  assert(prefix.match({0,2,3})==Prefix::Match::Append);
  assert(prefix.match({0,3,3})==Prefix::Match::Rebuild);
  assert(prefix.match({0})==Prefix::Match::Rebuild);
  prefix.commit({3,4,5},1);assert(prefix.size()==3);prefix.reset();assert(prefix.epoch()==1);
  std::cout<<"prefill cone, append clipping, route coverage and exact prefix identity pass\n";
}
