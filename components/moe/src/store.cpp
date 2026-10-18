#include "ds41/store.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <tuple>
#include <stdexcept>
#include <unordered_set>
#include <sys/mman.h>
#ifdef DS41_CUDA
#include <cuda_runtime.h>
#endif

namespace ds41 {
static unsigned layer_of(ExpertId e){return e<40*384?e/384:40+(e-40*384)/128;}
SlotPool::SlotPool(size_t slots, unsigned recent, unsigned decay, bool distinct)
    : slots_(slots), residency_(expert_count,-1), frequency_(expert_count), last_use_(expert_count), recent_(recent), decay_(decay),distinct_(distinct) {
    retention_epochs_=std::make_shared<RetentionEpochs>();
    for(auto& e:*retention_epochs_)e.store(0);
    dirty_.fill(true);
    if (!slots) throw std::invalid_argument("empty slot pool");
}
void SlotPool::policy(const Placement& p) {
    dirty_.fill(true);
    recent_=p.recent_window;decay_=p.decay_interval;distinct_=p.distinct_lfu;
    layer_budgets_=p.dynamic_slots_per_layer;
    hard_=p.hard_layer_budgets;
    slot_layers_.clear();
    if(hard_){
        std::array<size_t,43> pins{};for(auto id:p.vram_pins)++pins.at(expert_layer(id));
        for(unsigned l=0;l<43;++l)for(size_t j=0;j<pins[l]+layer_budgets_[l];++j)slot_layers_.push_back(l);
        if(slot_layers_.size()!=slots_.size())throw std::invalid_argument("hard layer capacities must cover every slot");
        for(size_t j=0;j<slots_.size();++j)if(slots_[j].id>=0 && expert_layer(slots_[j].id)!=slot_layers_[j])throw std::logic_error("hard policy requires compatible empty/startup pool");
    }
    budgets_=std::any_of(layer_budgets_.begin(),layer_budgets_.end(),[](size_t n){return n!=0;});
}
void SlotPool::reset_history(){dirty_.fill(true);std::fill(frequency_.begin(),frequency_.end(),0);std::fill(last_use_.begin(),last_use_.end(),0);layer_clocks_.fill(0);clock_=0;}
void SlotPool::touch(const std::vector<ExpertId>& ids){
    std::vector<ExpertId> unique;std::unordered_set<ExpertId> seen;
    for(auto e:ids){if(e>=expert_count)throw std::out_of_range("expert id");if(seen.insert(e).second)unique.push_back(e);}
        ++clock_;
        std::array<bool,43> layers{};
        for(auto e:unique)layers[layer_of(e)]=true;
        for(unsigned layer=0;layer<layers.size();++layer)if(layers[layer]){
            dirty_[layer]=true;
            auto clock=++layer_clocks_[layer];
            unsigned decay=late_[layer]?(host_?128u:512u):decay_;
            if(decay && clock%decay==0){
                unsigned first=layer<40?layer*384:40*384+(layer-40)*128;
                unsigned end=first+(layer<40?384:128);
                for(unsigned e=first;e<end;++e)frequency_[e]/=2;
            }
        }
        for (auto e : (distinct_?unique:ids)) { if (frequency_[e] != UINT32_MAX) ++frequency_[e]; last_use_[e] = layer_clocks_[layer_of(e)]; }
 }
std::shared_ptr<const void> SlotPool::retain(ExpertId id)const{
    int s=residency(id);if(s<0)throw std::logic_error("retain requires residency");
    if(!fast_)return slots_[s].retention;
    // The guard follows the last shared_ptr copy, including after pool destruction.
    auto epochs=retention_epochs_;auto layer=layer_of(id);
    auto token=slots_[s].retention;
    auto guard=std::shared_ptr<const void>(token.get(),[token,epochs,layer](const void*) mutable {
        token.reset();(*epochs)[layer].fetch_add(1,std::memory_order_release);
    });
    (*epochs)[layer].fetch_add(1,std::memory_order_release);
    return guard;
}
std::vector<SlotPool::Allocation> SlotPool::acquire(const std::vector<ExpertId>& input, bool prefetch) {
    std::vector<ExpertId> ids;
    std::unordered_set<ExpertId> wanted;
    for (auto e : input) {
        if (e >= expert_count) throw std::out_of_range("expert id");
        if (wanted.insert(e).second) ids.push_back(e);
    }
    // Transactional capacity check: reject before changing state.
    size_t available = 0;
    for (const auto& s : slots_) {
        if (s.draining) continue;
        if (s.id < 0 || wanted.count(s.id) || (!s.pinned && !s.leases && s.retention.use_count()<=1)) ++available;
    }
    if(hard_){
        std::array<size_t,43> need{},avail{};
        for(auto e:ids)++need[expert_layer(e)];
        for(size_t j=0;j<slots_.size();++j){const auto& s=slots_[j];if(s.draining)continue;if(s.id<0 || wanted.count(s.id) || (!s.pinned && !s.leases && s.retention.use_count()<=1))++avail[slot_layers_[j]];}
        for(unsigned l=0;l<43;++l)if(need[l]>avail[l])throw std::runtime_error("need set exceeds hard layer capacity");
    }
    if (ids.size() > available) throw std::runtime_error("need set exceeds unleased slot capacity; split the batch");
    for(auto e:ids)dirty_[layer_of(e)]=true;
    if(!prefetch)touch(input);
    std::array<size_t,43> layer_counts{};
    for(const auto& s:slots_)if(s.id>=0 && !s.pinned)++layer_counts[layer_of(s.id)];
    std::vector<Allocation> result;
    for (auto e : ids) {
        int slot = residency(e);
        bool hit = slot >= 0;
        int64_t evicted=-1;
        if (!hit) {
            for (size_t s = 0; s < slots_.size(); ++s) if (slots_[s].id < 0 && !slots_[s].draining && (!hard_ || slot_layers_[s]==layer_of(e))) { slot = s; break; }
            if (slot < 0) {
                if(fast_ && distinct_ && !hard_)slot=fast_victim(wanted,layer_counts);
                else slot=scan_victim(wanted,layer_counts,hard_?int(layer_of(e)):-1);
            }
            if (slot < 0) throw std::logic_error("capacity precheck disagreed");
            evicted=slots_[slot].id;
            if(evicted>=0){dirty_[layer_of(evicted)]=true;++epochs_[layer_of(evicted)];--layer_counts[layer_of(evicted)];residency_[evicted]=-1;}
            dirty_[layer_of(e)]=true;++epochs_[layer_of(e)];
            ++layer_counts[layer_of(e)];residency_[e]=slot;slots_[slot].id=e;slots_[slot].retention=std::make_shared<const ExpertId>(e);
        }
        auto& s = slots_[slot]; dirty_[layer_of(e)]=true; ++s.leases; s.stamp = ++stamp_;
        result.push_back({e, size_t(slot), hit,evicted});
    }
    return result;
}
int SlotPool::scan_victim(const std::unordered_set<ExpertId>& wanted,const std::array<size_t,43>& layer_counts,int layer)const {
    int slot=-1;
    for (unsigned pass = 0; pass != 2 && slot < 0; ++pass) {
        for (size_t s = 0; s < slots_.size(); ++s) {
            const auto& c = slots_[s];
            if (c.id < 0) continue;
            if (c.pinned || c.leases || c.retention.use_count()>1 || wanted.count(c.id) || (layer>=0 && slot_layers_[s]!=unsigned(layer))) continue;
            if (!pass && recent_ && last_use_[c.id] && last_use_[c.id] + recent_ >= layer_clocks_[layer_of(c.id)]) continue;
            bool over=budgets_ && layer_counts[layer_of(c.id)]>layer_budgets_[layer_of(c.id)];
            bool prior_over=slot>=0 && budgets_ && layer_counts[layer_of(slots_[slot].id)]>layer_budgets_[layer_of(slots_[slot].id)];
            if (slot<0 || (over && !prior_over) || (over==prior_over &&
                less_slot(s,size_t(slot))))slot=s;
        }
    }
    return slot;
}
std::vector<SlotPool::Allocation> SlotPool::reserve(size_t target,const std::unordered_set<ExpertId>& wanted) {
    std::vector<Allocation> out;
    if(hard_)throw std::logic_error("reserve requires soft layer budgets");
    size_t idle=0;
    for(const auto& s:slots_)if(s.id<0)++idle; // empty or draining
    if(idle>=target)return out;
    std::array<size_t,43> layer_counts{};
    for(const auto& s:slots_)if(s.id>=0 && !s.pinned)++layer_counts[layer_of(s.id)];
    while(idle<target) {
        int slot=fast_ && distinct_?fast_victim(wanted,layer_counts):scan_victim(wanted,layer_counts,-1);
        if(slot<0)break;
        auto e=ExpertId(slots_[slot].id);
        dirty_[layer_of(e)]=true;++epochs_[layer_of(e)];--layer_counts[layer_of(e)];residency_[e]=-1;
        slots_[slot].id=-1;slots_[slot].draining=true;slots_[slot].retention.reset();
        out.push_back({e,size_t(slot),false,int64_t(e)});
        ++idle;
    }
    return out;
}
void SlotPool::drained(size_t slot) {
    if(slot>=slots_.size() || !slots_[slot].draining || slots_[slot].id>=0)throw std::logic_error("drained requires a draining slot");
    slots_[slot].draining=false;
}
void SlotPool::exclusive(bool enabled) { exclusive_=enabled;dirty_.fill(true); }
std::vector<SlotPool::Allocation> SlotPool::borrow(const std::vector<ExpertId>& ids){
    for(auto id:ids)if(id>=expert_count)throw std::out_of_range("expert id");
    std::unordered_set<ExpertId> seen;std::vector<Allocation> result;
    for(auto id:ids)if(seen.insert(id).second){int slot=residency(id);if(slot>=0){result.push_back({id,size_t(slot),true,-1});}}
    for(auto a:result){++slots_[a.slot].leases;dirty_[layer_of(a.id)]=true;}
    return result;
}
void SlotPool::release(const std::vector<Allocation>& lease) {
    for (const auto& a : lease) {
        if (a.slot >= slots_.size() || slots_[a.slot].id != a.id || !slots_[a.slot].leases)
            throw std::logic_error("invalid or duplicate lease release");
    }
    for (const auto& a : lease) {--slots_[a.slot].leases;dirty_[layer_of(a.id)]=true;}
}
void SlotPool::pin(ExpertId id) { int s = residency(id); if (s < 0) throw std::logic_error("pin requires resident expert"); slots_[s].pinned = true; dirty_[layer_of(id)]=true; }
void SlotPool::unpin(ExpertId id) { int s = residency(id); if (s >= 0) {slots_[s].pinned = false;dirty_[layer_of(id)]=true;} }
int SlotPool::residency(ExpertId id) const {return id<expert_count?residency_[id]:-1;}
void SlotPool::fast_minima(bool enabled) { fast_=enabled;dirty_.fill(true); }
void SlotPool::decay_interval(unsigned interval) { decay_=interval;dirty_.fill(true); }
void SlotPool::peer(const SlotPool* pool,bool host) { peer_=pool;host_=host;dirty_.fill(true); }
void SlotPool::switch_target_layer(const Placement& p,unsigned l) {
    if(l>=40 || late_[l])return;
    late_[l]=true;dirty_[l]=true;
    if(!host_)for(auto id:p.vram_pins)if(layer_of(id)==l){unpin(id);++layer_budgets_[l];}
    budgets_=std::any_of(layer_budgets_.begin(),layer_budgets_.end(),[](size_t n){return n!=0;});
}
uint64_t SlotPool::score(ExpertId id)const {
    uint64_t f=frequency_[id];
    if(exclusive_ && host_ && peer_ && peer_->residency_[id]>=0)return 0;
    if(late_[layer_of(id)] && peer_) {
        if(host_){if(peer_->residency_[id]>=0)f/=2;}
        else if(peer_->residency_[id]<0)f*=2;
    }
    return f;
}
bool SlotPool::less_slot(size_t a,size_t b)const {
    return std::make_tuple(score(slots_[a].id),slots_[a].stamp,a)<
           std::make_tuple(score(slots_[b].id),slots_[b].stamp,b);
}
int SlotPool::fast_victim(const std::unordered_set<ExpertId>& wanted,const std::array<size_t,43>& counts) {
    for(unsigned l=0;l<43;++l) {
        auto retention=(*retention_epochs_)[l].load(std::memory_order_acquire);
        bool peer_change=(late_[l] || exclusive_) && peer_ && peer_epochs_[l]!=peer_->epochs_[l];
        if(!dirty_[l] && !peer_change && retention==seen_retention_epochs_[l])continue;
        best_any_[l]=best_old_[l]=-1;
        unsigned first=expert_id(l,0),end=first+(l<40?384:128);
        for(unsigned e=first;e<end;++e) {
            int j=residency_[e];if(j<0)continue;const auto& c=slots_[j];
            if(c.pinned || c.leases || c.retention.use_count()>1 || wanted.count(e))continue;
            if(best_any_[l]<0 || less_slot(j,best_any_[l]))best_any_[l]=j;
            if(recent_ && last_use_[e] && last_use_[e]+recent_>=layer_clocks_[l])continue;
            if(best_old_[l]<0 || less_slot(j,best_old_[l]))best_old_[l]=j;
        }
        dirty_[l]=false;seen_retention_epochs_[l]=retention;
        if(peer_)peer_epochs_[l]=peer_->epochs_[l];
    }
    for(unsigned pass=0;pass<2;++pass) {
        int best=-1;bool best_over=false;
        for(unsigned l=0;l<43;++l) {
            int j=pass?best_any_[l]:best_old_[l];if(j<0)continue;
            bool over=budgets_ && counts[l]>layer_budgets_[l];
            if(best<0 || (over && !best_over) || (over==best_over && less_slot(j,best))){best=j;best_over=over;}
        }
        if(best>=0)return best;
    }
    return -1;
}
DriftPolicy::DriftPolicy(const Placement& p,SlotPool& v,SlotPool& h):placement_(p),vram_(v),ram_(h) {
    if(p.hard_layer_budgets || !p.distinct_lfu || p.recent_window!=2 || p.decay_interval!=256)
        throw std::invalid_argument("drift-0-50 requires soft recent2 distinct decay256");
    for(auto e:p.vram_pins)pins_.at(e)=true;
    vram_.peer(&ram_,false);ram_.peer(&vram_,true);
}
void DriftPolicy::before(const std::vector<ExpertId>& ids) {
    if(ids.empty())throw std::invalid_argument("empty drift demand");
    unsigned l=expert_layer(ids.front());
    for(auto e:ids)if(e>=expert_count || expert_layer(e)!=l)throw std::invalid_argument("drift demand must represent one layer");
    if(l>=40)return;
    if(pending_ || l!=next_layer_)throw std::invalid_argument("drift requires complete width-one token-major traversal");
    if(l==0 && !active_ && window_.size()==128 && requests_ && 100*hits_<50*requests_) {
        for(unsigned layer=0;layer<40;++layer){vram_.switch_target_layer(placement_,layer);ram_.switch_target_layer(placement_,layer);}
        active_=true;
    }
    pending_=true;
}
void DriftPolicy::after(const std::vector<ExpertId>& ids) {
    unsigned l=expert_layer(ids.front());if(l>=40)return;
    if(!pending_ || l!=next_layer_)throw std::logic_error("unpaired drift completion");
    std::unordered_set<ExpertId> seen;
    for(auto e:ids)if(seen.insert(e).second){++token_requests_;if(pins_[e])++token_hits_;}
    pending_=false;
    if(++next_layer_==40) {
        next_layer_=0;++age_;
        window_.push_back({token_hits_,token_requests_});hits_+=token_hits_;requests_+=token_requests_;
        token_hits_=token_requests_=0;
        if(window_.size()>128){hits_-=window_.front().first;requests_-=window_.front().second;window_.pop_front();}
    }
}
HostBuffer::HostBuffer(size_t bytes, bool reg) : registered_(reg) {
    constexpr size_t huge = 2 * 1024 * 1024;
    if (!bytes || bytes > std::numeric_limits<size_t>::max() - 2 * huge) throw std::invalid_argument("host buffer size");
    bytes_ = (bytes + huge - 1) / huge * huge;
    void* raw = mmap(nullptr, bytes_ + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error(std::strerror(errno));
    auto base = reinterpret_cast<uintptr_t>(raw);
    auto aligned = (base + huge - 1) / huge * huge;
    if (aligned > base) munmap(raw, aligned - base);
    size_t tail = base + bytes_ + huge - aligned - bytes_;
    if (tail) munmap(reinterpret_cast<void*>(aligned + bytes_), tail);
    data_ = reinterpret_cast<void*>(aligned);
    madvise(data_, bytes_, MADV_HUGEPAGE);
    // Fault before registration, while THP can still coalesce. Report actual
    // AnonHugePages externally; MADV_HUGEPAGE is a request, not a guarantee.
    std::memset(data_, 0, bytes_);
    device_ = data_;
    if (reg) {
#ifdef DS41_CUDA
        auto rc = cudaHostRegister(data_, bytes_, cudaHostRegisterMapped);
        if (rc == cudaSuccess) rc = cudaHostGetDevicePointer(&device_, data_, 0);
        if (rc != cudaSuccess) { cudaHostUnregister(data_); munmap(data_, bytes_); data_ = nullptr; throw std::runtime_error(cudaGetErrorString(rc)); }
#else
        munmap(data_, bytes_); data_ = nullptr;
        throw std::runtime_error("CUDA registration requested from CPU-only build");
#endif
    }
}
HostBuffer::~HostBuffer() {
#ifdef DS41_CUDA
    if (registered_ && data_) cudaHostUnregister(data_);
#endif
    if (data_) munmap(data_, bytes_);
}
} // namespace ds41
