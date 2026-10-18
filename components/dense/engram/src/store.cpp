#include "ds41/engram.h"
#include "direct.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static thread_local std::string last_error;
template<class F> static int guard(F fn){try{fn();return 0;}catch(const std::exception&e){last_error=e.what();return -1;}catch(...){last_error="unknown Engram error";return -1;}}
static void require(bool ok,const char*w){if(!ok)throw std::invalid_argument(w);}
static Json load_json(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("cannot open JSON: "+p);return Json::parse(f);}
struct Memory {
 void*p=nullptr;size_t bytes=0;bool pinned=false;
 Memory()=default;Memory(size_t n,bool pin){allocate(n,pin);}
 void allocate(size_t n,bool pin){bytes=n;pinned=pin;if(!n)return;if(pin){auto r=cudaHostAlloc(&p,n,cudaHostAllocPortable);if(r!=cudaSuccess)throw std::runtime_error(std::string("cudaHostAlloc: ")+cudaGetErrorString(r));}else if(posix_memalign(&p,4096,n))throw std::bad_alloc();}
 ~Memory(){if(p){if(pinned)cudaFreeHost(p);else free(p);}} Memory(const Memory&)=delete;
};
struct LayerHash {int id;std::array<uint64_t,4> mul;std::array<int64_t,24> primes,offsets;};
struct HashConfig {std::vector<int64_t> map;int64_t pad;std::array<LayerHash,2> layers;};
struct Tensor {int fd=-1;uint64_t base=0,rows=0,stride=0,file_bytes=0;};
struct ds41_engram_state {std::shared_ptr<HashConfig> config;std::vector<int64_t> history;};
struct ds41_engram_ticket {
 int layer;std::vector<int64_t> rows;Memory output;std::mutex mutex;std::condition_variable cv;bool done=false;std::string error;
 ds41_engram_ticket(int l,const int64_t*r,size_t n,bool pin):layer(l),rows(r,r+n),output(n*264,pin){}
};
struct ds41_engram_store {
 std::shared_ptr<HashConfig> hash=std::make_shared<HashConfig>();std::array<std::array<Tensor,2>,2> tensors;
 std::vector<int> files;engram::Ring ring;unsigned qd,batch;bool pin;uint64_t capacity,live=0;Memory cache,stage;
 std::unordered_map<uint64_t,std::pair<size_t,std::list<uint64_t>::iterator>> entries;std::list<uint64_t> lru;
 std::mutex mutex;std::condition_variable cv;std::deque<ds41_engram_ticket*> queue;bool stopping=false,active=false;std::thread worker;ds41_engram_stats stats{};
 explicit ds41_engram_store(const ds41_engram_config&c):ring(c.queue_depth?c.queue_depth:48),qd(c.queue_depth?c.queue_depth:48),batch(c.max_batch_rows?c.max_batch_rows:4096),pin(c.pin_memory!=0),capacity(c.cache_rows),cache(c.cache_rows*264,pin),stage(size_t(batch)*4*4096,false){
  try{
   auto meta=load_json(c.hash_json);require(meta.at("schema")=="ds41-engram-hash-v1","hash schema");hash->map=meta.at("token_map").get<std::vector<int64_t>>();hash->pad=meta.at("pad_id");require(hash->map.size()==129280&&meta.at("compressed_vocab_size")==99092&&meta.at("layers").size()==2,"hash vocabulary/layout");
   for(auto v:hash->map){require(v>=0&&v<99092,"invalid compressed token");}require(hash->pad>=0&&hash->pad<99092,"invalid pad");
   for(int l=0;l<2;++l){auto m=meta.at("layers").at(l);auto&h=hash->layers[l];h.id=m.at("layer");require(h.id==(l?14:1),"Engram layer order");h.mul=m.at("multipliers").get<std::array<uint64_t,4>>();h.primes=m.at("primes").get<std::array<int64_t,24>>();h.offsets=m.at("offsets").get<std::array<int64_t,24>>();int64_t off=0;for(int k=0;k<24;++k){require(h.primes[k]>0&&h.offsets[k]==off,"invalid prime layout");off+=h.primes[k];}for(auto v:h.mul)require(v&&v<=uint64_t(INT64_MAX)/99092,"invalid hash multiplier");}
   auto index=load_json(c.index_json).at("weight_map"),receipt=load_json(c.integrity_json);require(receipt.at("complete").get<bool>(),"incomplete integrity receipt");require(receipt.at("root")==std::string(c.weight_root),"receipt weight root mismatch");std::unordered_map<std::string,Json> verified,headers;std::unordered_map<std::string,int> fds;std::unordered_map<std::string,uint64_t> starts,sizes;
   for(auto f:receipt.at("files"))if(f.value("ok",false))verified[f.at("path")]=f;
   for(int l=0;l<2;++l)for(int s=0;s<2;++s){std::string wanted="model.layers."+std::to_string(hash->layers[l].id)+".engram.embed."+(s?"weight_scale_inv":"weight");
    // Index uses the unchanged HF names. Also accept normalized exports when explicitly indexed.
    if(!index.contains(wanted)){std::string alt="layers."+std::to_string(hash->layers[l].id)+".engram.embed."+(s?"scale":"weight");require(index.contains(alt),"Engram tensor missing from index");wanted=alt;}
    std::string shard=index.at(wanted),path=std::string(c.weight_root)+"/"+shard;require(shard.find("/")==std::string::npos&&shard!="..","invalid shard path");
    if(!fds.count(shard)){int fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC|O_NOFOLLOW);if(fd<0)engram::sysfail("open Engram O_DIRECT");files.push_back(fd);struct stat st{};if(fstat(fd,&st))engram::sysfail("fstat Engram");require(verified.count(shard),"unverified Engram shard");auto v=verified.at(shard);uint64_t ns=uint64_t(st.st_mtim.tv_sec)*1000000000+st.st_mtim.tv_nsec;require(v.at("size").get<uint64_t>()==uint64_t(st.st_size)&&v.at("mtime_ns").get<uint64_t>()==ns,"stale Engram integrity receipt");
     int f=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(f<0)engram::sysfail("open safetensors header");uint64_t n=0;ssize_t got=pread(f,&n,8,0);if(got!=8||!n||n>(128ull<<20)){close(f);throw std::runtime_error("bad safetensors header length");}std::string raw(n,'\0');size_t done=0;while(done<n){auto r=pread(f,raw.data()+done,n-done,8+done);if(r<0&&errno==EINTR)continue;if(r<=0){close(f);throw std::runtime_error("short safetensors header");}done+=r;}close(f);headers[shard]=Json::parse(raw);starts[shard]=8+n;fds[shard]=fd;sizes[shard]=st.st_size;
    }
    auto d=headers.at(shard).at(wanted);auto shape=d.at("shape").get<std::vector<uint64_t>>();require(shape.size()==2&&shape[1]==uint64_t(s?8:256),"Engram row shape");std::string dtype=d.at("dtype");require(s?(dtype=="F8_E8M0"||dtype=="F8_E8M0FNU"):(dtype=="F8_E4M3"||dtype=="F8_E4M3FN"),"Engram dtype");auto offsets=d.at("data_offsets").get<std::array<uint64_t,2>>();uint64_t bytes=shape[0]*shape[1];require(offsets[1]>=offsets[0]&&offsets[1]-offsets[0]==bytes&&starts[shard]+offsets[1]<=sizes[shard],"Engram tensor bounds");
    tensors[l][s]={fds[shard],starts[shard]+offsets[0],shape[0],shape[1],sizes[shard]};require(shape[0]==uint64_t(hash->layers[l].offsets[23]+hash->layers[l].primes[23]),"Engram rows vs layout");
   }
   stats.cache_capacity_rows=capacity;worker=std::thread([this]{run();});
  }catch(...){for(int f:files)close(f);throw;}
 }
 ~ds41_engram_store(){{std::lock_guard<std::mutex>g(mutex);stopping=true;}cv.notify_one();if(worker.joinable())worker.join();for(int f:files)close(f);}
 bool cached(uint64_t key,uint8_t*out){auto i=entries.find(key);if(i==entries.end())return false;memcpy(out,(char*)cache.p+i->second.first*264,264);lru.splice(lru.begin(),lru,i->second.second);return true;}
 void insert(uint64_t key,const uint8_t*row){if(!capacity)return;size_t slot;if(live<capacity){slot=live++;}else{auto old=lru.back();slot=entries.at(old).first;lru.pop_back();entries.erase(old);}memcpy((char*)cache.p+slot*264,row,264);lru.push_front(key);entries.emplace(key,std::make_pair(slot,lru.begin()));}
 void fetch(ds41_engram_ticket&t){auto begin=Clock::now();int li=t.layer==1?0:1;auto*out=(uint8_t*)t.output.p;std::vector<uint32_t> owner(t.rows.size());std::vector<std::pair<int64_t,uint32_t>> misses;std::unordered_map<int64_t,uint32_t> first;first.reserve(t.rows.size());ds41_engram_stats delta{};delta.submitted_rows=t.rows.size();
  for(size_t i=0;i<t.rows.size();++i){auto[it,added]=first.emplace(t.rows[i],uint32_t(i));owner[i]=it->second;if(!added)continue;uint64_t key=(uint64_t(li)<<63)|uint64_t(t.rows[i]);if(cached(key,out+i*264))++delta.cache_hits;else misses.emplace_back(t.rows[i],uint32_t(i));}
  delta.unique_misses=misses.size();std::sort(misses.begin(),misses.end());
  for(size_t start=0;start<misses.size();start+=batch){size_t count=std::min(size_t(batch),misses.size()-start);std::vector<engram::Page> pages;std::unordered_map<uint64_t,size_t> page_index;pages.reserve(count*3);page_index.reserve(count*3);
   auto page_key=[](int s,uint64_t off){return (uint64_t(s)<<63)|off;};
   for(size_t j=0;j<count;++j)for(int s=0;s<2;++s){auto d=tensors[li][s];uint64_t off=d.base+uint64_t(misses[start+j].first)*d.stride,last=off+d.stride-1;for(uint64_t p=off&~4095ull;p<=last;p+=4096){auto[it,added]=page_index.emplace(page_key(s,p),pages.size());if(added)pages.push_back({d.fd,p,uint32_t(std::min(uint64_t(4096),d.file_bytes-p))});}}
   require(pages.size()*4096<=stage.bytes,"Engram staging overflow");auto io=Clock::now();ring.read(pages,stage.p,qd);delta.io_ms+=elapsed(io);delta.disk_pages+=pages.size();delta.disk_bytes+=pages.size()*4096;++delta.batches;
   for(size_t j=0;j<count;++j){auto[row,index]=misses[start+j];auto*dst=out+size_t(index)*264;for(int s=0;s<2;++s){auto d=tensors[li][s];uint64_t off=d.base+uint64_t(row)*d.stride;size_t copied=0;while(copied<d.stride){uint64_t p=off&~4095ull;size_t n=std::min(d.stride-copied,4096-(off&4095));memcpy(dst+(s?256:0)+copied,(char*)stage.p+page_index.at(page_key(s,p))*4096+(off&4095),n);copied+=n;off+=n;}}insert((uint64_t(li)<<63)|uint64_t(row),dst);}
  }
  for(size_t i=0;i<t.rows.size();++i)if(owner[i]!=i)memcpy(out+i*264,out+size_t(owner[i])*264,264);
  delta.total_ms=elapsed(begin);{std::lock_guard<std::mutex>g(mutex);stats.submitted_rows+=delta.submitted_rows;stats.cache_hits+=delta.cache_hits;stats.unique_misses+=delta.unique_misses;stats.disk_pages+=delta.disk_pages;stats.disk_bytes+=delta.disk_bytes;stats.batches+=delta.batches;stats.io_ms+=delta.io_ms;stats.total_ms+=delta.total_ms;stats.cache_live_rows=live;}
 }
 void run(){for(;;){ds41_engram_ticket*t;{std::unique_lock<std::mutex>g(mutex);cv.wait(g,[&]{return stopping||!queue.empty();});if(queue.empty()&&stopping)return;t=queue.front();queue.pop_front();active=true;}std::string e;try{fetch(*t);}catch(const std::exception&x){e=x.what();}catch(...){e="unknown worker exception";}{std::lock_guard<std::mutex>g(mutex);active=false;}{std::lock_guard<std::mutex>g(t->mutex);t->error=e;t->done=true;t->cv.notify_all();}}}
};
extern "C" const char*ds41_engram_error(){return last_error.c_str();}
extern "C" int ds41_engram_open(const ds41_engram_config*c,ds41_engram_store**out){return guard([&]{require(c&&out,"null open args");*out=nullptr;require(c->struct_bytes==sizeof(*c)&&c->abi_version==1,"Engram ABI version");require(c->weight_root&&c->index_json&&c->hash_json&&c->integrity_json,"null paths");require(c->queue_depth<=4096&&c->max_batch_rows<=65536&&c->cache_rows<=16777216&&c->pin_memory<=1&&!c->reserved,"Engram config bounds");*out=new ds41_engram_store(*c);});}
extern "C" void ds41_engram_close(ds41_engram_store*s){delete s;}
extern "C" int ds41_engram_get_stats(ds41_engram_store*s,ds41_engram_stats*out){return guard([&]{require(s&&out,"stats args");std::lock_guard<std::mutex>g(s->mutex);*out=s->stats;});}
extern "C" int ds41_engram_clear_cache(ds41_engram_store*s){return guard([&]{require(s,"null store");std::lock_guard<std::mutex>g(s->mutex);require(!s->active&&s->queue.empty(),"clear_cache requires idle store");s->entries.clear();s->lru.clear();s->live=0;s->stats.cache_live_rows=0;});}
extern "C" int ds41_engram_state_create(ds41_engram_store*s,ds41_engram_state**out){return guard([&]{require(s&&out,"state args");*out=new ds41_engram_state{s->hash,{}};});}
extern "C" int ds41_engram_state_clone(const ds41_engram_state*s,ds41_engram_state**out){return guard([&]{require(s&&out,"clone args");*out=new ds41_engram_state(*s);});}
extern "C" void ds41_engram_state_free(ds41_engram_state*s){delete s;}
extern "C" int ds41_engram_state_truncate(ds41_engram_state*s,uint64_t pos){return guard([&]{require(s&&pos<=s->history.size(),"truncate position");s->history.resize(pos);});}
extern "C" int ds41_engram_hash(ds41_engram_state*s,const int32_t*tokens,const uint8_t*mask,uint64_t pos,size_t count,int64_t*out){return guard([&]{require(s&&pos<=s->history.size()&&count<=16777216&&pos<=SIZE_MAX-count,"hash position/size");require(!count||(tokens&&out),"hash buffers");for(size_t i=0;i<count;++i)require(tokens[i]>=0&&tokens[i]<int(s->config->map.size()),"token id");s->history.resize(pos+count);for(size_t i=0;i<count;++i){size_t at=pos+i;s->history[at]=(mask&&!mask[i])?-1:s->config->map[tokens[i]];uint64_t v[4];bool blocked=false;for(int k=0;k<4;++k){blocked=blocked||at<size_t(k)||(at>=size_t(k)&&s->history[at-k]==-1);v[k]=blocked?s->config->pad:s->history[at-k];}for(int l=0;l<2;++l){auto h=s->config->layers[l];uint64_t rolling=v[0]*h.mul[0];for(int k=1;k<4;++k){rolling^=v[k]*h.mul[k];for(int a=0;a<8;++a){int col=(k-1)*8+a;out[(i*2+l)*24+col]=int64_t(rolling%uint64_t(h.primes[col]))+h.offsets[col];}}}}});}
extern "C" int ds41_engram_submit(ds41_engram_store*s,int layer,const int64_t*rows,size_t count,ds41_engram_ticket**out){return guard([&]{require(s&&out&&(layer==1||layer==14)&&count<=16777216,"submit args");*out=nullptr;require(!count||rows,"null rows");int l=layer==1?0:1;for(size_t i=0;i<count;++i)require(rows[i]>=0&&uint64_t(rows[i])<s->tensors[l][0].rows,"row id");int64_t empty=0;std::unique_ptr<ds41_engram_ticket>t(new ds41_engram_ticket(layer,rows?rows:&empty,count,s->pin));{std::lock_guard<std::mutex>g(s->mutex);require(!s->stopping,"store closing");s->queue.push_back(t.get());}*out=t.release();s->cv.notify_one();});}
extern "C" int ds41_engram_ready(ds41_engram_ticket*t){if(!t){last_error="null ticket";return -1;}std::lock_guard<std::mutex>g(t->mutex);if(!t->done)return 0;if(!t->error.empty()){last_error=t->error;return -1;}return 1;}
extern "C" int ds41_engram_wait(ds41_engram_ticket*t){return guard([&]{require(t,"null ticket");std::unique_lock<std::mutex>g(t->mutex);t->cv.wait(g,[&]{return t->done;});if(!t->error.empty())throw std::runtime_error(t->error);});}
extern "C" int ds41_engram_data(ds41_engram_ticket*t,const void**data,size_t*count){return guard([&]{require(t&&data&&count,"data args");std::lock_guard<std::mutex>g(t->mutex);require(t->done,"ticket pending");if(!t->error.empty())throw std::runtime_error(t->error);*data=t->output.p;*count=t->rows.size();});}
extern "C" void ds41_engram_ticket_free(ds41_engram_ticket*t){if(t){ds41_engram_wait(t);delete t;}}
extern "C" int ds41_engram_copy(ds41_engram_ticket*t,void*fp8,void*scales){return guard([&]{require(t,"copy ticket");std::lock_guard<std::mutex>g(t->mutex);require(t->done,"ticket pending");if(!t->error.empty())throw std::runtime_error(t->error);require(t->rows.empty()||(fp8&&scales),"copy buffers");for(size_t i=0;i<t->rows.size();++i){memcpy((char*)fp8+i*256,(char*)t->output.p+i*264,256);memcpy((char*)scales+i*8,(char*)t->output.p+i*264+256,8);}});}
