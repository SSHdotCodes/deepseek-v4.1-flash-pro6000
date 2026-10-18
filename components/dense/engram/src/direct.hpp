#pragma once
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <linux/io_uring.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
namespace engram {
inline void sysfail(const char* w){throw std::runtime_error(std::string(w)+": "+strerror(errno));}
struct Page { int fd; uint64_t offset; uint32_t required; };
class Ring {
 int fd=-1;io_uring_params p{};void *sq=MAP_FAILED,*cq=MAP_FAILED;io_uring_sqe* es=(io_uring_sqe*)MAP_FAILED;
 size_t sb=0,cb=0;unsigned *sh=nullptr,*st=nullptr,*sm=nullptr,*arr=nullptr,*ch=nullptr,*ct=nullptr,*cm=nullptr;io_uring_cqe* ce=nullptr;
 void cleanup(){if(es!=MAP_FAILED)munmap(es,p.sq_entries*sizeof(*es));if(cq!=MAP_FAILED&&cq!=sq)munmap(cq,cb);if(sq!=MAP_FAILED)munmap(sq,sb);if(fd>=0)close(fd);}
 public:
 explicit Ring(unsigned depth){try{
  fd=syscall(__NR_io_uring_setup,depth,&p);if(fd<0)sysfail("io_uring_setup");sb=p.sq_off.array+p.sq_entries*4;cb=p.cq_off.cqes+p.cq_entries*sizeof(*ce);
  if(p.features&IORING_FEAT_SINGLE_MMAP)sb=std::max(sb,cb);
  sq=mmap(nullptr,sb,PROT_READ|PROT_WRITE,MAP_SHARED,fd,IORING_OFF_SQ_RING);if(sq==MAP_FAILED)sysfail("sq mmap");
  cq=(p.features&IORING_FEAT_SINGLE_MMAP)?sq:mmap(nullptr,cb,PROT_READ|PROT_WRITE,MAP_SHARED,fd,IORING_OFF_CQ_RING);if(cq==MAP_FAILED)sysfail("cq mmap");
  es=(io_uring_sqe*)mmap(nullptr,p.sq_entries*sizeof(*es),PROT_READ|PROT_WRITE,MAP_SHARED,fd,IORING_OFF_SQES);if(es==MAP_FAILED)sysfail("sqe mmap");
  auto a=(char*)sq,b=(char*)cq;sh=(unsigned*)(a+p.sq_off.head);st=(unsigned*)(a+p.sq_off.tail);sm=(unsigned*)(a+p.sq_off.ring_mask);arr=(unsigned*)(a+p.sq_off.array);ch=(unsigned*)(b+p.cq_off.head);ct=(unsigned*)(b+p.cq_off.tail);cm=(unsigned*)(b+p.cq_off.ring_mask);ce=(io_uring_cqe*)(b+p.cq_off.cqes);
 }catch(...){cleanup();throw;}}
 ~Ring(){cleanup();} Ring(const Ring&)=delete;
 // Queue is deliberately limited to configured QD even when kernel rounds SQ size up.
 void read(const std::vector<Page>& work,void* data,unsigned qd){
  size_t next=0,done=0;int error=0;
  while(done<work.size()){
   unsigned tail=*st,head=__atomic_load_n(sh,__ATOMIC_ACQUIRE),count=0;
   while(!error&&next<work.size()&&next-done<qd&&tail-head<p.sq_entries){unsigned slot=tail&*sm;auto&e=es[slot];memset(&e,0,sizeof(e));e.opcode=IORING_OP_READ;e.fd=work[next].fd;e.off=work[next].offset;e.addr=(uint64_t)((char*)data+next*4096);e.len=4096;e.user_data=next;arr[slot]=slot;++tail;++count;++next;}
   __atomic_store_n(st,tail,__ATOMIC_RELEASE);
   for(unsigned pending=count;pending;){int r=syscall(__NR_io_uring_enter,fd,pending,0,0,nullptr,0);if(r<0&&errno==EINTR)continue;if(r<=0)sysfail("io_uring_submit");pending-=r;}
   if(done==next)break;
   unsigned h=*ch,t=__atomic_load_n(ct,__ATOMIC_ACQUIRE);
   if(h==t){int r=syscall(__NR_io_uring_enter,fd,0,1,IORING_ENTER_GETEVENTS,nullptr,0);if(r<0&&errno!=EINTR)sysfail("io_uring_wait");continue;}
   while(h!=t){auto c=ce[h&*cm];if(c.user_data>=work.size())throw std::runtime_error("bad CQ user_data");if(c.res<0||uint32_t(c.res)<work[c.user_data].required||c.res>4096)error=c.res<0?-c.res:EIO;++done;++h;}
   __atomic_store_n(ch,h,__ATOMIC_RELEASE);
  }
  if(error){errno=error;sysfail("O_DIRECT read (no fallback)");}if(done!=work.size())throw std::runtime_error("incomplete direct batch");
 }
};
}
