#include "ds41/hotpath/route_plan.hpp"
#include <cassert>
#include <random>
#include <set>
int main() {
  using namespace ds41e::hotpath;
  std::mt19937 rng(41);
  for(int rows=1;rows<=6;++rows) for(int trial=0;trial<1000;++trial) {
    int selected[36];
    for(int r=0;r<rows;++r) {
      std::set<int> seen;
      for(int k=0;k<6;++k) {
        int id;
        do { id=int(rng()%384); } while(!seen.insert(id).second);
        selected[r*6+k]=id;
      }
    }
    auto plan=plan_routes(selected,rows,39);
    std::set<int> slots;
    int offset=0; uint32_t prior=0;
    for(auto e:plan.experts) {
      assert(e.begin==offset && e.count<=rows && e.id>prior);
      prior=e.id;
      for(int i=0;i<e.count;++i) {
        int j=e.begin+i;
        assert(slots.insert(plan.destination[j]).second);
        assert(plan.source[j]==plan.destination[j]/6);
        assert(e.id==uint32_t(39*384+selected[plan.destination[j]]));
      }
      offset+=e.count;
    }
    assert(offset==rows*6 && slots.size()==size_t(rows*6));
  }
  int duplicate[6]={0,1,2,3,4,4}; bool rejected=false;
  try { plan_routes(duplicate,1,0); } catch(const std::invalid_argument &) { rejected=true; }
  assert(rejected);
}
