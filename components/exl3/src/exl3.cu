#include <ds41/exl3.cuh>
#include "device.cuh"
namespace ds41 {
namespace exl3_detail {
// Fixed 8-row tile, 128 columns, four contiguous K partitions. M never changes
// geometry, K order, rounding, or reduction order. No inter-CTA communication.
template<int K>
__global__ __launch_bounds__(512,1) void decode(const DeviceExpert* table,const GemmJob* jobs,int projection,int N) {
    GemmJob job=jobs[blockIdx.y];
    if(job.tokens<1||job.tokens>8){asm volatile("trap;");return;}
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection); const half* sv=su+K;
    extern __shared__ __align__(16) unsigned char storage[];
    half* A=reinterpret_cast<half*>(storage);
    float* red=reinterpret_cast<float*>(A+(8*K<2048?2048:8*K));
    int warp=threadIdx.x/32,lane=threadIdx.x%32,wn=warp%4,wk=warp/4;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    for(int idx=warp;idx<job.tokens*(K/128);idx+=16) {
        int r=idx/(K/128),c=(idx%(K/128))*128;
        rotate_input(x+r*K+c,su+c,A+r*K+c,lane);
    }
    __syncthreads();
    constexpr int SLICES=K/16,CHUNK=(SLICES+3)/4,PF=4,WNT=2;
    const int start=wk*CHUNK, len=min(CHUNK,SLICES-start);
    const int r=lane/4; bool active=r<job.tokens;
    const uint32_t* A2=reinterpret_cast<const uint32_t*>(A);
    uint32_t pf[PF][WNT];
    auto load=[&](int i,int t){
        size_t off=(size_t(start+i)*(N/16)+blockIdx.x*8+wn*2+t)*24+lane;
        return lane<24?__ldcs(B+off):0u;
    };
    #pragma unroll
    for(int d=0;d<PF;++d)if(d<len){
        #pragma unroll
        for(int t=0;t<WNT;++t)pf[d][t]=load(d,t);
    }
    float acc[WNT][2][4]={};
    for(int base=0;base<len;base+=PF){
        #pragma unroll
        for(int d=0;d<PF;++d){
            int i=base+d;if(i>=len)break;
            uint32_t bw[WNT];
            #pragma unroll
            for(int t=0;t<WNT;++t){bw[t]=pf[d][t];if(i+PF<len)pf[d][t]=load(i+PF,t);}
            int col=(start+i)*8+(lane&3);
            uint32_t a[4]={active?A2[r*(K/2)+col]:0u,0u,active?A2[r*(K/2)+col+4]:0u,0u};
            #pragma unroll
            for(int t=0;t<WNT;++t){Pair f0,f1;dq3(bw[t],f0,f1,lane);mma(acc[t][0],a,f0);mma(acc[t][1],a,f1);}
        }
    }
    if(active){
        #pragma unroll
        for(int t=0;t<WNT;++t){
            #pragma unroll
            for(int f=0;f<2;++f){
                int c=wn*32+t*16+f*8+2*(lane&3);
                red[(wk*8+r)*128+c]=acc[t][f][0];red[(wk*8+r)*128+c+1]=acc[t][f][1];
            }
        }
    }
    __syncthreads();
    // Reuse the input buffer for the reduced tile. Each element has one writer.
    float* out=reinterpret_cast<float*>(A);
    for(int i=threadIdx.x;i<job.tokens*128;i+=512){
        float sum=0;
        #pragma unroll
        for(int w=0;w<4;++w)sum+=red[w*8*128+i];
        out[i]=sum;
    }
    __syncthreads();
    if(warp<job.tokens)rotate_output(out+warp*128,sv+blockIdx.x*128,job.output+warp*N+blockIdx.x*128,lane);
}
// Prefill: 64 rows x 128 cols. Eight warps own 16 cols each, reuse trellis
// fragments across four m16 tiles. K128 staging avoids an M*K scratch buffer.
// The 136-half stride distributes eight MMA rows over all shared banks.
__global__ __launch_bounds__(256,1) void prefill(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,int K,int max_tokens) {
    GemmJob job=jobs[blockIdx.y];
    if(job.tokens<1||job.tokens>max_tokens){asm volatile("trap;");return;}
    int m0=blockIdx.z*64;if(m0>=job.tokens)return;
    DeviceExpert e=table[job.expert];wait_ready(e);
    const uint32_t* B=weights(e,projection);
    const half* su=scales(e,projection);const half* sv=su+K;
    __shared__ __align__(16) half A[64*136];
    __shared__ float out[64*128];
    int warp=threadIdx.x/32,lane=threadIdx.x%32,r=lane/4;
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(job.x);
    float acc[4][2][4]={};
    for(int k0=0;k0<K;k0+=128){
        for(int row=warp;row<64;row+=8)if(m0+row<job.tokens)
            rotate_input(x+(m0+row)*K+k0,su+k0,A+row*136,lane);
        __syncthreads();
        #pragma unroll
        for(int ki=0;ki<8;++ki){
            const uint32_t* b=B+(size_t((k0/16)+ki)*(N/16)+blockIdx.x*8+warp)*24;
            uint32_t word=lane<24?__ldcs(b+lane):0;
            Pair f0,f1;dq3(word,f0,f1,lane);
            #pragma unroll
            for(int mi=0;mi<4;++mi){
                uint32_t a[4]={};
                #pragma unroll
                for(int hr=0;hr<2;++hr){int row=mi*16+r+hr*8;
                    if(m0+row<job.tokens){
                        const uint32_t* p=reinterpret_cast<const uint32_t*>(A+row*136)+ki*8+(lane&3);
                        a[hr]=p[0];a[hr+2]=p[4];
                    }
                }
                mma(acc[mi][0],a,f0);mma(acc[mi][1],a,f1);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int mi=0;mi<4;++mi){
        #pragma unroll
        for(int hr=0;hr<2;++hr){
            int row=mi*16+r+hr*8;
            #pragma unroll
            for(int f=0;f<2;++f){int col=warp*16+f*8+(lane&3)*2;
                out[row*128+col]=acc[mi][f][hr*2];out[row*128+col+1]=acc[mi][f][hr*2+1];
            }
        }
    }
    __syncthreads();
    for(int row=warp;row<64;row+=8)if(m0+row<job.tokens)
        rotate_output(out+row*128,sv+blockIdx.x*128,job.output+(m0+row)*N+blockIdx.x*128,lane);
}
}
namespace {
bool valid(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k){
    return count>=0&&count<=65535&&int(p)>=0&&int(p)<=2&&n>0&&n%128==0&&k>0&&k%128==0&&k<=5120&&(count==0||(t&&j));
}
template<int K> cudaError_t launch_decode(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,cudaStream_t s){
    constexpr int bytes=(8*K*2<4096?4096:8*K*2)+4*8*128*4;
    // Function attributes are host metadata operations, with no stream work,
    // allocation or synchronization; legal during capture as well as replay.
    auto err=cudaFuncSetAttribute(exl3_detail::decode<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess)return err;
    exl3_detail::decode<K><<<dim3(n/128,count),512,bytes,s>>>(t,j,int(p),n);
    return cudaGetLastError();
}
}
cudaError_t grouped_exl3(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,cudaStream_t s){
    if(!valid(t,j,count,p,n,k))return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(k==5120)return launch_decode<5120>(t,j,count,p,n,s);
    if(k==2304)return launch_decode<2304>(t,j,count,p,n,s);
    if(k==256)return launch_decode<256>(t,j,count,p,n,s);
    if(k==128)return launch_decode<128>(t,j,count,p,n,s);
    return cudaErrorNotSupported;
}
cudaError_t grouped_exl3_prefill(const DeviceExpert* t,const GemmJob* j,int count,Projection p,int n,int k,int max_tokens,cudaStream_t s){
    if(!valid(t,j,count,p,n,k)||max_tokens<1||max_tokens>256)return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    exl3_detail::prefill<<<dim3(n/128,count,(max_tokens+63)/64),256,0,s>>>(t,j,int(p),n,k,max_tokens);
    return cudaGetLastError();
}
}
