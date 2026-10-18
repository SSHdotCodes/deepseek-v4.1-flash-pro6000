#include "ds41/store.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace ds41 {
static void fail(const char* what) { throw std::runtime_error(std::string(what) + ": " + std::strerror(errno)); }
struct DirectReader::Impl {
    int fd = -1;
    io_uring_params p{};
    void *sq = MAP_FAILED, *cq = MAP_FAILED;
    io_uring_sqe* sqes = reinterpret_cast<io_uring_sqe*>(MAP_FAILED);
    size_t sq_bytes = 0, cq_bytes = 0;
    unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr, *array = nullptr;
    unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;
    std::unordered_map<std::string, int> files;
    ~Impl() {
        for (const auto& [_, f] : files) close(f);
        if (sqes != MAP_FAILED) munmap(sqes, p.sq_entries * sizeof(io_uring_sqe));
        if (cq != MAP_FAILED && cq != sq) munmap(cq, cq_bytes);
        if (sq != MAP_FAILED) munmap(sq, sq_bytes);
        if (fd >= 0) close(fd);
    }
};
DirectReader::DirectReader(unsigned depth) : impl_(new Impl) {
    if (!depth || depth > 4096) throw std::invalid_argument("io queue depth");
    auto& i = *impl_;
    i.fd = syscall(__NR_io_uring_setup, depth, &i.p);
    if (i.fd < 0) fail("io_uring_setup");
    i.sq_bytes = i.p.sq_off.array + i.p.sq_entries * sizeof(unsigned);
    i.cq_bytes = i.p.cq_off.cqes + i.p.cq_entries * sizeof(io_uring_cqe);
    bool single = i.p.features & IORING_FEAT_SINGLE_MMAP;
    if (single) i.sq_bytes = std::max(i.sq_bytes, i.cq_bytes);
    i.sq = mmap(nullptr, i.sq_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, i.fd, IORING_OFF_SQ_RING);
    if (i.sq == MAP_FAILED) fail("mmap sq");
    i.cq = single ? i.sq : mmap(nullptr, i.cq_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, i.fd, IORING_OFF_CQ_RING);
    if (i.cq == MAP_FAILED) fail("mmap cq");
    i.sqes = static_cast<io_uring_sqe*>(mmap(nullptr, i.p.sq_entries*sizeof(io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED, i.fd, IORING_OFF_SQES));
    if (i.sqes == MAP_FAILED) fail("mmap sqes");
    auto sq = static_cast<char*>(i.sq); auto cq = static_cast<char*>(i.cq);
    i.sq_head = reinterpret_cast<unsigned*>(sq + i.p.sq_off.head); i.sq_tail = reinterpret_cast<unsigned*>(sq + i.p.sq_off.tail);
    i.sq_mask = reinterpret_cast<unsigned*>(sq + i.p.sq_off.ring_mask); i.array = reinterpret_cast<unsigned*>(sq + i.p.sq_off.array);
    i.cq_head = reinterpret_cast<unsigned*>(cq + i.p.cq_off.head); i.cq_tail = reinterpret_cast<unsigned*>(cq + i.p.cq_off.tail);
    i.cq_mask = reinterpret_cast<unsigned*>(cq + i.p.cq_off.ring_mask); i.cqes = reinterpret_cast<io_uring_cqe*>(cq + i.p.cq_off.cqes);
}
DirectReader::~DirectReader() = default;
void DirectReader::read(const std::vector<const Record*>& records, const std::vector<void*>& destinations) {
    if(records.size()!=destinations.size())throw std::invalid_argument("read destinations");
    stream(records,destinations,{});
}
void DirectReader::stream(const std::vector<const Record*>& records,const std::vector<void*>& buffers,
                          const std::function<void(size_t,size_t)>& ready,
                          const std::function<void(size_t)>& recycle,size_t chunk_bytes) {
    stream_mixed(records,buffers,{},ready,recycle,chunk_bytes);
}
void DirectReader::stream_mixed(const std::vector<const Record*>& records,const std::vector<void*>& buffers,
                          const std::vector<void*>& fixed,
                          const std::function<void(size_t,size_t)>& ready,
                          const std::function<void(size_t)>& recycle,size_t chunk_bytes) {
    stream_mixed_chunked(records,buffers,fixed,ready,{},recycle,chunk_bytes);
}
void DirectReader::stream_mixed_chunked(const std::vector<const Record*>& records,const std::vector<void*>& buffers,
                          const std::vector<void*>& fixed,
                          const std::function<void(size_t,size_t)>& ready,
                          const std::function<void(size_t,size_t,uint64_t,uint32_t)>& chunk,
                          const std::function<void(size_t)>& recycle,size_t chunk_bytes) {
    if(records.empty())return;
    if((!fixed.empty() && fixed.size()!=records.size()) || (buffers.empty() && (fixed.empty() || std::any_of(fixed.begin(),fixed.end(),[](void* p){return !p;}))) || (chunk_bytes && (chunk_bytes%4096 || chunk_bytes>INT32_MAX)))
        throw std::invalid_argument("direct read ring/chunk size");
    for(auto ptr:buffers)if(reinterpret_cast<uintptr_t>(ptr)%4096)throw std::invalid_argument("unaligned ring buffer");
    for(auto ptr:fixed)if(reinterpret_cast<uintptr_t>(ptr)%4096)throw std::invalid_argument("unaligned fixed buffer");
    auto base=[&](size_t r,size_t b){return b<buffers.size()?buffers[b]:fixed[r];};
    struct Request { int fd; uint64_t offset,dst_offset; uint32_t bytes,required; size_t record; };
    std::vector<Request> work;
    std::vector<std::vector<size_t>> requests(records.size());
    auto& i=*impl_;
    // Validate the complete plan before issuing writes into caller-owned memory.
    for(size_t r=0;r<records.size();++r) {
        if(!records[r] || records[r]->extents.empty())throw std::invalid_argument("record without extents");
        for(const auto& e:records[r]->extents) {
            if(e.file_offset%4096 || e.bytes%4096 || e.dst_offset%4096 ||
               e.dst_offset>records[r]->bytes || e.bytes>records[r]->bytes-e.dst_offset ||
               !e.bytes || e.bytes>INT32_MAX || e.required_bytes>e.bytes)
                throw std::invalid_argument("unaligned/oversized O_DIRECT extent");
            auto found=i.files.find(e.path);
            if(found==i.files.end()) {
                int fd=open(e.path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
                if(fd<0)fail("open O_DIRECT");
                struct stat st{};
                if(fstat(fd,&st)<0){close(fd);fail("fstat expert shard");}
                if(e.expected_file_bytes && uint64_t(st.st_size)!=e.expected_file_bytes){close(fd);throw std::runtime_error("expert shard still downloading or wrong size: "+e.path);}
                found=i.files.emplace(e.path,fd).first;
            }
            uint64_t required=e.required_bytes?e.required_bytes:e.bytes;
            for(uint64_t off=0;off<e.bytes;) {
                uint64_t bytes=std::min(e.bytes-off,chunk_bytes?uint64_t(chunk_bytes):e.bytes);
                uint64_t need=required>off?std::min(bytes,required-off):0;
                requests[r].push_back(work.size());
                work.push_back({found->second,e.file_offset+off,e.dst_offset+off,uint32_t(bytes),uint32_t(need),r});
                off+=bytes;
            }
        }
    }
    std::vector<size_t> remaining(records.size()),buffer_of(records.size());
    std::deque<size_t> queued,free;
    for(size_t b=0;b<buffers.size();++b)free.push_back(b);
    size_t next=0,submitted=0,completed=0;
    int error=0;std::exception_ptr callback_error;
    while(true) {
        while(!error && !callback_error && next<records.size() && ((!fixed.empty() && fixed[next]) || !free.empty())) {
            bool retained=!fixed.empty() && fixed[next];
            size_t b=retained?buffers.size()+next:free.front();if(!retained)free.pop_front();
            try {if(!retained && recycle)recycle(b);}catch(...){callback_error=std::current_exception();break;}
            buffer_of[next]=b;remaining[next]=requests[next].size();
            for(auto w:requests[next])queued.push_back(w);
            ++next;
        }
        unsigned head=__atomic_load_n(i.sq_head,__ATOMIC_ACQUIRE),tail=*i.sq_tail,count=0;
        while(!error && !callback_error && !queued.empty() && tail-head<i.p.sq_entries && submitted-completed<i.p.cq_entries) {
            auto index=queued.front();queued.pop_front();const auto& w=work[index];
            unsigned slot=tail&*i.sq_mask;auto& e=i.sqes[slot];std::memset(&e,0,sizeof(e));
            e.opcode=IORING_OP_READ;e.fd=w.fd;e.off=w.offset;
            e.addr=reinterpret_cast<uint64_t>(static_cast<char*>(base(w.record,buffer_of[w.record]))+w.dst_offset);
            e.len=w.bytes;e.user_data=index;i.array[slot]=slot;++tail;++submitted;++count;
        }
        __atomic_store_n(i.sq_tail,tail,__ATOMIC_RELEASE);
        for(unsigned pending=count;pending;) {
            int rc=syscall(__NR_io_uring_enter,i.fd,pending,0,0,nullptr,0);
            if(rc<0 && errno==EINTR)continue;
            if(rc<=0)fail("io_uring_submit");
            pending-=rc;
        }
        if(completed==submitted)break;
        unsigned ch=*i.cq_head,ct=__atomic_load_n(i.cq_tail,__ATOMIC_ACQUIRE);
        if(ch==ct) {
            int rc=syscall(__NR_io_uring_enter,i.fd,0,1,IORING_ENTER_GETEVENTS,nullptr,0);
            if(rc<0 && errno!=EINTR)fail("io_uring_wait");
            continue;
        }
        std::vector<size_t> done,landed;
        while(ch!=ct) {
            const auto& c=i.cqes[ch&*i.cq_mask];const auto& w=work.at(c.user_data);
            if(c.res<0 || uint32_t(c.res)<w.required || uint32_t(c.res)>w.bytes)error=c.res<0?-c.res:EIO;
            else if(chunk)landed.push_back(c.user_data);
            if(!--remaining[w.record])done.push_back(w.record);
            ++ch;++completed;
        }
        __atomic_store_n(i.cq_head,ch,__ATOMIC_RELEASE);
        for(auto index:landed) {
            const auto& w=work[index];
            if(!error && !callback_error)try {chunk(w.record,buffer_of[w.record],w.dst_offset,w.bytes);}catch(...){callback_error=std::current_exception();}
        }
        // Publish CQ consumption before user code. A throwing callback must not
        // leave stale completions in the ring on the next call.
        for(auto r:done) {
            size_t b=buffer_of[r];
            if(!error && !callback_error)try {if(ready)ready(r,b);}catch(...){callback_error=std::current_exception();}
            if(b<buffers.size())free.push_back(b);
        }
    }
    if(error){errno=error;fail("io_uring read (no buffered fallback)");}
    if(callback_error)std::rethrow_exception(callback_error);
    if(completed!=work.size())throw std::runtime_error("incomplete direct read");
}
} // namespace ds41
