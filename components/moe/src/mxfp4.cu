#include "ds41/kernels.cuh"
#include <cuda_fp8.h>

namespace ds41 {
__device__ __forceinline__ float e8m0(uint8_t x) {
    return x == 0 ? __uint_as_float(0x00400000) : x == 255 ? __uint_as_float(0x7fffffff) : __uint_as_float(uint32_t(x) << 23);
}
__global__ void quant_kernel(const __nv_bfloat16* x, uint8_t* q, uint8_t* s, int groups) {
    int g = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (g >= groups) return;
    float value = __bfloat162float(x[g * 32 + lane]);
    float amax = fabsf(value);
#pragma unroll
    for (int d=16; d; d/=2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
    amax = fmaxf(amax, 1e-4f);
    // Match fast_round_scale: first rounded FP32 amax * (1/448), then bit ceil.
    uint32_t b = __float_as_uint(__fmul_rn(amax, 1.0f/448.0f));
    unsigned exponent = (b >> 23) + ((b & 0x7fffff) != 0);
    float scale = __uint_as_float(exponent << 23);
    q[g*32+lane] = __nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,value/scale)), __NV_SATFINITE, __NV_E4M3);
    if (!lane) s[g] = uint8_t(exponent);
}
// Four packed E2M1 values -> four exact E4M3 values. No floating conversion.
__device__ __forceinline__ uint32_t expand4(uint16_t x) {
    uint32_t magnitude=__byte_perm(0x3c383000u,0x4c484440u,x&0x7777);
    uint32_t sign=((x&0x0008)<<4)|((x&0x0080)<<8)|((x&0x0800)<<12)|((x&0x8000)<<16);
    return magnitude|sign;
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
       : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void wait_ready(const DeviceExpert& e) {
    if (!e.ready) return;
    uint32_t value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(e.ready) : "memory");
        if (value != e.generation) __nanosleep(64);
    } while (value != e.generation);
}
// Four warps, each handles 16 output channels and all <=8 tokens. We transpose
// the mathematical GEMM so weights are MMA A and tokens MMA B; no K permutation.
template<bool Direct=false>
__global__ void __launch_bounds__(128) gemm_kernel(const DeviceExpert* table, const GemmJob* jobs, int projection, int N, int K,DeviceExpert raw_e={},GemmJob raw_job={}) {
    GemmJob job = Direct?raw_job:jobs[blockIdx.y];
    DeviceExpert e = Direct?raw_e:table[job.expert];
    wait_ready(e);
    const uint8_t* w = projection == 0 ? e.gate : projection == 1 ? e.up : e.down;
    const uint8_t* ws = projection == 0 ? e.gate_scale : projection == 1 ? e.up_scale : e.down_scale;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int g = lane >> 2, t = lane & 3;
    int row = blockIdx.x * 64 + warp * 16 + g;
    int ng = K / 32;
    float acc[4] = {0,0,0,0};
    for (int group=0; group<ng; ++group) {
        uint32_t a[4] = {0,0,0,0}, b[2] = {0,0};
#pragma unroll
        for (int hr=0; hr<2; ++hr) {
            if (row+hr*8 < N) {
                const auto* src = w + (size_t)(row+hr*8)*(K/2) + group*16 + t*2;
                a[hr] = expand4(*reinterpret_cast<const uint16_t*>(src));
                a[hr+2] = expand4(*reinterpret_cast<const uint16_t*>(src+8));
            }
        }
        if (g < job.tokens) {
            const auto* x = job.x + (size_t)g*K + group*32 + t*4;
            b[0] = *reinterpret_cast<const uint32_t*>(x);
            b[1] = *reinterpret_cast<const uint32_t*>(x+16);
        }
        float c[4] = {0,0,0,0};
        mma(c,a,b);
#pragma unroll
        for (int hr=0; hr<2; ++hr) {
            float sw = row+hr*8 < N ? e8m0(ws[(size_t)(row+hr*8)*ng+group]) : 0.f;
#pragma unroll
            for (int cc=0; cc<2; ++cc) {
                int token = t*2+cc;
                float sx = token < job.tokens ? e8m0(job.x_scale[(size_t)token*ng+group]) : 0.f;
                // Force the reference's two multiplications and serial addition.
                acc[hr*2+cc] = __fadd_rn(acc[hr*2+cc], __fmul_rn(__fmul_rn(c[hr*2+cc], sx), sw));
            }
        }
    }
#pragma unroll
    for (int hr=0; hr<2; ++hr) {
#pragma unroll
        for (int cc=0; cc<2; ++cc) {
            int token=t*2+cc;
            if (row+hr*8 < N && token < job.tokens) job.output[(size_t)token*N+row+hr*8] = __float2bfloat16_rn(acc[hr*2+cc]);
        }
    }
}
__device__ __forceinline__ void copy8(void* smem,const void* gmem) {
    uint32_t addr=uint32_t(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"::"r"(addr),"l"(gmem));
}
__device__ __forceinline__ uint32_t pad_fp4(uint16_t packed){
    // Duplicate each packed byte, then select alternating low/high nibbles.
    // This is a bit permutation into the MMA b8 operand, never a conversion.
    uint32_t v=__byte_perm(uint32_t(packed),0,0x1100);
    return ((v<<2)&0x003c003c)|((v>>2)&0x3c003c00);
}
__device__ __forceinline__ void mma_native(float (&c)[4],const uint32_t (&a)[4],const uint32_t (&b)[2],uint32_t sa,uint32_t sb){
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}
#include "prefill.cuh"
template<bool Direct=false>
__global__ void __launch_bounds__(128) gemm_native(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,int K,DeviceExpert raw_e={},GemmJob raw_job={}){
    GemmJob job=Direct?raw_job:jobs[blockIdx.y];DeviceExpert e=Direct?raw_e:table[job.expert];wait_ready(e);
    const uint8_t* w=projection==0?e.gate:projection==1?e.up:e.down;
    const uint8_t* ws=projection==0?e.gate_scale:projection==1?e.up_scale:e.down_scale;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3;
    int n0=blockIdx.x*32+warp*8,ng=K/32;
    float acc0=0,acc1=0;
    for(int group=0;group<ng;++group){
        uint32_t a[4]={0,0,0,0},b[2]={0,0};
        if(g<job.tokens){const uint8_t* x=job.x+(size_t)g*K+group*32+t*4;a[0]=*reinterpret_cast<const uint32_t*>(x);a[2]=*reinterpret_cast<const uint32_t*>(x+16);}
        if(n0+g<N){const uint8_t* v=w+(size_t)(n0+g)*(K/2)+group*16+t*2;b[0]=pad_fp4(*reinterpret_cast<const uint16_t*>(v));b[1]=pad_fp4(*reinterpret_cast<const uint16_t*>(v+8));}
        uint32_t sa=(g<job.tokens && !(t&1))?job.x_scale[(size_t)g*ng+group]:127;
        uint32_t sb=n0+g<N?ws[(size_t)(n0+g)*ng+group]:127;
        float c[4]={0,0,0,0};mma_native(c,a,b,sa,sb);
        // Zero per-group MMA accumulator keeps the reference group boundary.
        acc0=__fadd_rn(acc0,c[0]);acc1=__fadd_rn(acc1,c[1]);
    }
    if(g<job.tokens){if(n0+t*2<N)job.output[(size_t)g*N+n0+t*2]=__float2bfloat16_rn(acc0);if(n0+t*2+1<N)job.output[(size_t)g*N+n0+t*2+1]=__float2bfloat16_rn(acc1);}
}
// Compute four K segments in parallel, retain EVERY scaled 32-wide dot in
// registers, then pass the accumulator in K order. Unlike split-K partial sums,
// this preserves the serial reference addition order exactly. No FP32 scratch
// traffic and no atomic operations. Four warps cooperate on 16 output channels.
template<int K,bool Direct=false>
__global__ void __launch_bounds__(128,1) gemm_parallel(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,DeviceExpert raw_e={},GemmJob raw_job={}){
    constexpr int ng=K/32,G=(ng+3)/4,Groups=8,Stride=144,WarpStage=16*Stride,Stage=4*WarpStage;
    __shared__ __align__(16) uint8_t smem[2*Stage];
    __shared__ float carry[4][128];
    GemmJob job=Direct?raw_job:jobs[blockIdx.y];DeviceExpert e=Direct?raw_e:table[job.expert];wait_ready(e);
    const uint8_t* w=projection==0?e.gate:projection==1?e.up:e.down;
    const uint8_t* ws=projection==0?e.gate_scale:projection==1?e.up_scale:e.down_scale;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3,n0=blockIdx.x*16,row=n0+g;
    float dots[G][4];
#pragma unroll
    for(int j=0;j<G;++j)
#pragma unroll
        for(int c=0;c<4;++c)dots[j][c]=0;
    auto load=[&](int tile,int stage){
#pragma unroll
        for(int j=0;j<8;++j){int index=lane+j*32,r=index/16,piece=index%16,local=tile*Groups+piece/2,global=warp*G+local;
            if(n0+r<N && local<G && global<ng)copy8(smem+stage*Stage+warp*WarpStage+r*Stride+piece*8,w+(size_t)(n0+r)*(K/2)+global*16+(piece%2)*8);}
        asm volatile("cp.async.commit_group;");
    };
    load(0,0);
#pragma unroll
    for(int tile=0;tile<(G+Groups-1)/Groups;++tile){
        int stage=tile&1;
        if(tile+1<(G+Groups-1)/Groups){load(tile+1,stage^1);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
#pragma unroll
        for(int u=0;u<Groups;++u){
            int local=tile*Groups+u,group=warp*G+local;
            if(local<G && group<ng){
                uint32_t a[4]={0,0,0,0},b[2]={0,0};
#pragma unroll
                for(int hr=0;hr<2;++hr)if(row+hr*8<N){const uint8_t*src=smem+stage*Stage+warp*WarpStage+(g+hr*8)*Stride+u*16+t*2;a[hr]=expand4(*reinterpret_cast<const uint16_t*>(src));a[hr+2]=expand4(*reinterpret_cast<const uint16_t*>(src+8));}
                if(g<job.tokens){const uint8_t*x=job.x+(size_t)g*K+group*32+t*4;b[0]=*reinterpret_cast<const uint32_t*>(x);b[1]=*reinterpret_cast<const uint32_t*>(x+16);}
                float c[4]={0,0,0,0};mma(c,a,b);
#pragma unroll
                for(int hr=0;hr<2;++hr){float sw=row+hr*8<N?e8m0(ws[(size_t)(row+hr*8)*ng+group]):0.f;
#pragma unroll
                    for(int cc=0;cc<2;++cc){int token=t*2+cc;float sx=token<job.tokens?e8m0(job.x_scale[(size_t)token*ng+group]):0.f;dots[local][hr*2+cc]=__fmul_rn(__fmul_rn(c[hr*2+cc],sx),sw);}}
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int turn=0;turn<4;++turn){
        if(warp==turn){float acc[4]={0,0,0,0};
            if(turn)
#pragma unroll
                for(int c=0;c<4;++c)acc[c]=carry[turn-1][lane*4+c];
#pragma unroll
            for(int j=0;j<G;++j)if(turn*G+j<ng)
#pragma unroll
                for(int c=0;c<4;++c)acc[c]=__fadd_rn(acc[c],dots[j][c]);
#pragma unroll
            for(int c=0;c<4;++c)carry[turn][lane*4+c]=acc[c];
            if(turn==3)
#pragma unroll
                for(int hr=0;hr<2;++hr)
#pragma unroll
                    for(int cc=0;cc<2;++cc){int token=t*2+cc;if(row+hr*8<N && token<job.tokens)job.output[(size_t)token*N+row+hr*8]=__float2bfloat16_rn(acc[hr*2+cc]);}
        }
        __syncthreads();
    }
}
template<int K,int Warps=4,bool Direct=false>
__global__ void __launch_bounds__(Warps*32,1) gemm_native_parallel(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,DeviceExpert raw_e={},GemmJob raw_job={}){
    constexpr int ng=K/32,G=(ng+Warps-1)/Warps;
    __shared__ float carry[Warps][64];
    GemmJob job=Direct?raw_job:jobs[blockIdx.y];DeviceExpert e=Direct?raw_e:table[job.expert];wait_ready(e);
    const uint8_t* w=projection==0?e.gate:projection==1?e.up:e.down;
    const uint8_t* ws=projection==0?e.gate_scale:projection==1?e.up_scale:e.down_scale;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3,n0=blockIdx.x*8;
    float dots[G][2];
#pragma unroll
    for(int j=0;j<G;++j){
        int group=warp*G+j;dots[j][0]=dots[j][1]=0;
        if(group<ng){
            uint32_t a[4]={0,0,0,0},b[2]={0,0};
            if(g<job.tokens){const uint8_t*x=job.x+(size_t)g*K+group*32+t*4;a[0]=*reinterpret_cast<const uint32_t*>(x);a[2]=*reinterpret_cast<const uint32_t*>(x+16);}
            if(n0+g<N){const uint8_t*v=w+(size_t)(n0+g)*(K/2)+group*16+t*2;b[0]=pad_fp4(*reinterpret_cast<const uint16_t*>(v));b[1]=pad_fp4(*reinterpret_cast<const uint16_t*>(v+8));}
            uint32_t sa=(g<job.tokens && !(t&1))?job.x_scale[(size_t)g*ng+group]:127;
            uint32_t sb=n0+g<N?ws[(size_t)(n0+g)*ng+group]:127;
            float c[4]={0,0,0,0};mma_native(c,a,b,sa,sb);dots[j][0]=c[0];dots[j][1]=c[1];
        }
    }
    __syncthreads();
#pragma unroll
    for(int turn=0;turn<Warps;++turn){
        if(warp==turn){float acc0=turn?carry[turn-1][lane*2]:0.f,acc1=turn?carry[turn-1][lane*2+1]:0.f;
#pragma unroll
            for(int j=0;j<G;++j)if(turn*G+j<ng){acc0=__fadd_rn(acc0,dots[j][0]);acc1=__fadd_rn(acc1,dots[j][1]);}
            carry[turn][lane*2]=acc0;carry[turn][lane*2+1]=acc1;
            if(turn==Warps-1 && g<job.tokens){if(n0+t*2<N)job.output[(size_t)g*N+n0+t*2]=__float2bfloat16_rn(acc0);if(n0+t*2+1<N)job.output[(size_t)g*N+n0+t*2+1]=__float2bfloat16_rn(acc1);}
        }
        __syncthreads();
    }
}
__global__ void __launch_bounds__(128) gemm_native_tiled(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,int K){
    constexpr int Groups=8,Bytes=128,Stride=144,Stage=32*Stride;
    __shared__ __align__(16) uint8_t smem[2*Stage];
    GemmJob job=jobs[blockIdx.y];DeviceExpert e=table[job.expert];wait_ready(e);
    const uint8_t*w=projection==0?e.gate:projection==1?e.up:e.down;
    const uint8_t*ws=projection==0?e.gate_scale:projection==1?e.up_scale:e.down_scale;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3,n0=blockIdx.x*32,ng=K/32;
    float acc0=0,acc1=0;
    auto load=[&](int tile,int stage){
#pragma unroll
        for(int j=0;j<4;++j){int index=threadIdx.x+j*128,r=index/16,piece=index%16;
            if(n0+r<N)copy8(smem+stage*Stage+r*Stride+piece*8,w+(size_t)(n0+r)*(K/2)+tile*Bytes+piece*8);}
        asm volatile("cp.async.commit_group;");
    };
    int tiles=ng/Groups;load(0,0);
    for(int tile=0;tile<tiles;++tile){
        int stage=tile&1;
        if(tile+1<tiles){load(tile+1,stage^1);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
#pragma unroll
        for(int u=0;u<Groups;++u){
            int group=tile*Groups+u;uint32_t a[4]={0,0,0,0},b[2]={0,0};
            if(g<job.tokens){const uint8_t*x=job.x+(size_t)g*K+group*32+t*4;a[0]=*reinterpret_cast<const uint32_t*>(x);a[2]=*reinterpret_cast<const uint32_t*>(x+16);}
            if(n0+warp*8+g<N){const uint8_t*v=smem+stage*Stage+(warp*8+g)*Stride+u*16+t*2;b[0]=pad_fp4(*reinterpret_cast<const uint16_t*>(v));b[1]=pad_fp4(*reinterpret_cast<const uint16_t*>(v+8));}
            uint32_t sa=(g<job.tokens && !(t&1))?job.x_scale[(size_t)g*ng+group]:127;
            uint32_t sb=n0+warp*8+g<N?ws[(size_t)(n0+warp*8+g)*ng+group]:127;
            float c[4]={0,0,0,0};mma_native(c,a,b,sa,sb);acc0=__fadd_rn(acc0,c[0]);acc1=__fadd_rn(acc1,c[1]);
        }
        __syncthreads();
    }
    if(g<job.tokens){if(n0+warp*8+t*2<N)job.output[(size_t)g*N+n0+warp*8+t*2]=__float2bfloat16_rn(acc0);if(n0+warp*8+t*2+1<N)job.output[(size_t)g*N+n0+warp*8+t*2+1]=__float2bfloat16_rn(acc1);}
}
// Tiled native weights: coalesced 8-byte loads support the 8-byte (not 16-byte)
// alignment of the unmodified safetensors weight plane. Double buffering hides
// global memory latency without changing any arithmetic/reduction boundary.
template<bool Direct=false>
__global__ void __launch_bounds__(128) gemm_tiled(const DeviceExpert* table,const GemmJob* jobs,int projection,int N,int K,DeviceExpert raw_e={},GemmJob raw_job={}) {
    constexpr int Groups=8, Bytes=Groups*16, Stride=Bytes+16, Stage=64*Stride;
    __shared__ __align__(16) uint8_t smem[2*Stage];
    GemmJob job=Direct?raw_job:jobs[blockIdx.y];DeviceExpert e=Direct?raw_e:table[job.expert];wait_ready(e);
    const uint8_t* w=projection==0?e.gate:projection==1?e.up:e.down;
    const uint8_t* ws=projection==0?e.gate_scale:projection==1?e.up_scale:e.down_scale;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3;
    int n0=blockIdx.x*64,row=n0+warp*16+g,ng=K/32;
    float acc[4]={0,0,0,0};
    auto load=[&](int tile,int stage){
#pragma unroll
        for(int j=0;j<8;++j){
            int index=threadIdx.x+j*128, r=index/(Bytes/8), piece=index%(Bytes/8);
            if(n0+r<N && tile*Groups*32<K)copy8(smem+stage*Stage+r*Stride+piece*8,w+(size_t)(n0+r)*(K/2)+tile*Bytes+piece*8);
        }
        asm volatile("cp.async.commit_group;");
    };
    int tiles=(ng+Groups-1)/Groups;
    load(0,0);
    for(int tile=0;tile<tiles;++tile){
        int stage=tile&1;
        if(tile+1<tiles){load(tile+1,stage^1);asm volatile("cp.async.wait_group 1;");}
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
#pragma unroll
        for(int u=0;u<Groups;++u){
            int group=tile*Groups+u;
            if(group>=ng)break;
            uint32_t a[4]={0,0,0,0},b[2]={0,0};
#pragma unroll
            for(int hr=0;hr<2;++hr){
                if(row+hr*8<N){
                    const uint8_t* src=smem+stage*Stage+(warp*16+g+hr*8)*Stride+u*16+t*2;
                    a[hr]=expand4(*reinterpret_cast<const uint16_t*>(src));
                    a[hr+2]=expand4(*reinterpret_cast<const uint16_t*>(src+8));
                }
            }
            if(g<job.tokens){const uint8_t* x=job.x+(size_t)g*K+group*32+t*4;b[0]=*reinterpret_cast<const uint32_t*>(x);b[1]=*reinterpret_cast<const uint32_t*>(x+16);}
            float c[4]={0,0,0,0};mma(c,a,b);
#pragma unroll
            for(int hr=0;hr<2;++hr){
                float sw=row+hr*8<N?e8m0(ws[(size_t)(row+hr*8)*ng+group]):0.f;
#pragma unroll
                for(int cc=0;cc<2;++cc){int token=t*2+cc;float sx=token<job.tokens?e8m0(job.x_scale[(size_t)token*ng+group]):0.f;acc[hr*2+cc]=__fadd_rn(acc[hr*2+cc],__fmul_rn(__fmul_rn(c[hr*2+cc],sx),sw));}
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int hr=0;hr<2;++hr)
#pragma unroll
        for(int cc=0;cc<2;++cc){int token=t*2+cc;if(row+hr*8<N && token<job.tokens)job.output[(size_t)token*N+row+hr*8]=__float2bfloat16_rn(acc[hr*2+cc]);}
}
__global__ void swiglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, const float* weight,
                              __nv_bfloat16* out, int rows, int cols, float limit) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= rows*cols) return;
    float g=__bfloat162float(gate[i]), u=__bfloat162float(up[i]);
    if (limit > 0) { g=fminf(g,limit); u=fminf(limit,fmaxf(-limit,u)); }
    float h=__fmul_rn(g / (1.f+expf(-g)),u);
    if (weight) h=__fmul_rn(weight[i/cols],h);
    out[i]=__float2bfloat16_rn(h);
}
__global__ void reduce_kernel(const __nv_bfloat16* routed,const int32_t* ids,const __nv_bfloat16* shared,__nv_bfloat16* out,int rows,int topk,int hidden){
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=rows*hidden)return;
    int token=index/hidden,h=index%hidden,last=-1;float acc=0;
    for(int i=0;i<topk;++i){int chosen=-1,smallest=0x7fffffff;for(int j=0;j<topk;++j){int id=ids[token*topk+j];if(id>last && id<smallest){chosen=j;smallest=id;}}
        if(chosen<0){asm volatile("trap;");return;} // duplicate/invalid ids must never silently drop an expert
        acc=__fadd_rn(acc,__bfloat162float(routed[((size_t)token*topk+chosen)*hidden+h]));last=smallest;}
    if(shared)acc=__fadd_rn(acc,__bfloat162float(shared[index]));out[index]=__float2bfloat16_rn(acc);
}
cudaError_t quantize_mxfp8(const __nv_bfloat16* x, uint8_t* q, uint8_t* s, int rows, int cols, cudaStream_t stream) {
    if (rows<=0 || cols<=0 || cols%32) return cudaErrorInvalidValue;
    quant_kernel<<<(rows*(cols/32)+3)/4,128,0,stream>>>(x,q,s,rows*(cols/32));
    return cudaGetLastError();
}
cudaError_t grouped_mxfp4(const DeviceExpert* table, const GemmJob* jobs, int count, Projection p, int n, int k, cudaStream_t stream) {
    if (count<=0 || n<=0 || k<=0 || k%32 || int(p)<0 || int(p)>2) return cudaErrorInvalidValue;
    if(k%256==0)gemm_tiled<false><<<dim3((n+63)/64,count),128,0,stream>>>(table,jobs,int(p),n,k);
    else gemm_kernel<false><<<dim3((n+63)/64,count),128,0,stream>>>(table,jobs,int(p),n,k);
    return cudaGetLastError();
}
cudaError_t swiglu(const __nv_bfloat16* g,const __nv_bfloat16* u,const float* w,__nv_bfloat16* out,int rows,int cols,float limit,cudaStream_t stream) {
    if(rows<=0 || cols<=0) return cudaErrorInvalidValue;
    swiglu_kernel<<<(rows*cols+255)/256,256,0,stream>>>(g,u,w,out,rows,cols,limit);
    return cudaGetLastError();
}
cudaError_t grouped_mxfp4_native(const DeviceExpert* table,const GemmJob* jobs,int count,Projection p,int n,int k,cudaStream_t stream){
    if(count<1 || n<1 || k<1 || k%32)return cudaErrorInvalidValue;
    gemm_native<false><<<dim3((n+31)/32,count),128,0,stream>>>(table,jobs,int(p),n,k);return cudaGetLastError();
}
cudaError_t grouped_mxfp4_parallel(const DeviceExpert* table,const GemmJob* jobs,int count,Projection p,int n,int k,cudaStream_t stream){
    if(count<1 || n<1)return cudaErrorInvalidValue;
    if(k==5120)gemm_parallel<5120><<<dim3((n+15)/16,count),128,0,stream>>>(table,jobs,int(p),n);
    else if(k==2304)gemm_parallel<2304><<<dim3((n+15)/16,count),128,0,stream>>>(table,jobs,int(p),n);
    else if(k==256)gemm_parallel<256><<<dim3((n+15)/16,count),128,0,stream>>>(table,jobs,int(p),n);
    else return grouped_mxfp4(table,jobs,count,p,n,k,stream);
    return cudaGetLastError();
}
cudaError_t grouped_mxfp4_native_parallel(const DeviceExpert* table,const GemmJob* jobs,int count,Projection p,int n,int k,cudaStream_t stream){
    if(count<1 || n<1)return cudaErrorInvalidValue;
    if(k==5120)gemm_native_parallel<5120,8><<<dim3((n+7)/8,count),256,0,stream>>>(table,jobs,int(p),n);
    else if(k==2304)gemm_native_parallel<2304><<<dim3((n+7)/8,count),128,0,stream>>>(table,jobs,int(p),n);
    else if(k==256)gemm_native_parallel<256><<<dim3((n+7)/8,count),128,0,stream>>>(table,jobs,int(p),n);
    else return grouped_mxfp4_native(table,jobs,count,p,n,k,stream);
    return cudaGetLastError();
}
cudaError_t grouped_mxfp4_native_tiled(const DeviceExpert* table,const GemmJob* jobs,int count,Projection p,int n,int k,cudaStream_t stream){
    if(k%256)return grouped_mxfp4_native(table,jobs,count,p,n,k,stream);
    if(count<1 || n<1 || k<1)return cudaErrorInvalidValue;
    gemm_native_tiled<<<dim3((n+31)/32,count),128,0,stream>>>(table,jobs,int(p),n,k);return cudaGetLastError();
}
cudaError_t grouped_mxfp4_prefill(const DeviceExpert*table,const GemmJob*jobs,int count,Projection p,int n,int k,int max_tokens,cudaStream_t stream){
    if(!table || !jobs || count<1 || n<1 || k<256 || k%256 || max_tokens<1 || max_tokens>256 || int(p)<0 || int(p)>2)return cudaErrorInvalidValue;
    gemm_prefill<false><<<dim3((n+63)/64,count,(max_tokens+31)/32),128,0,stream>>>(table,jobs,int(p),n,k,{}, {},max_tokens);return cudaGetLastError();
}
cudaError_t reduce_experts(const __nv_bfloat16*r,const int32_t*ids,const __nv_bfloat16*shared,__nv_bfloat16*out,int rows,int topk,int hidden,cudaStream_t stream){
    if(rows<1 || topk<1 || topk>8 || hidden<1)return cudaErrorInvalidValue;
    reduce_kernel<<<(rows*hidden+255)/256,256,0,stream>>>(r,ids,shared,out,rows,topk,hidden);return cudaGetLastError();
}
} // namespace ds41

extern "C" int ds41_mxfp4_raw(const uint8_t* w,const uint8_t* ws,const uint8_t* x,const uint8_t* xs,void* out,int m,int n,int k,void* stream){
    if(m<1 || n<1 || k<1 || k%32 || !w || !ws || !x || !xs || !out)return int(cudaErrorInvalidValue);
    ds41::DeviceExpert e;e.gate=w;e.gate_scale=ws;
    for(int first=0;first<m;first+=8){
        ds41::GemmJob job{0,m-first<8?m-first:8,x+(size_t)first*k,xs+(size_t)first*(k/32),static_cast<__nv_bfloat16*>(out)+(size_t)first*n};
        if(k%256==0)ds41::gemm_tiled<true><<<dim3((n+63)/64),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,k,e,job);
        else ds41::gemm_kernel<true><<<dim3((n+63)/64),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,k,e,job);
    }
    return int(cudaGetLastError());
}
extern "C" int ds41_quantize_mxfp8_raw(const void* x,uint8_t*q,uint8_t*s,int rows,int cols,void* stream){
    return int(ds41::quantize_mxfp8(static_cast<const __nv_bfloat16*>(x),q,s,rows,cols,static_cast<cudaStream_t>(stream)));
}
extern "C" int ds41_mxfp4_native_raw(const uint8_t*w,const uint8_t*ws,const uint8_t*x,const uint8_t*xs,void*out,int m,int n,int k,void*stream){
    if(m<1 || n<1 || k<1 || k%32 || !w || !ws || !x || !xs || !out)return int(cudaErrorInvalidValue);
    ds41::DeviceExpert e;e.gate=w;e.gate_scale=ws;
    for(int first=0;first<m;first+=8){ds41::GemmJob job{0,m-first<8?m-first:8,x+(size_t)first*k,xs+(size_t)first*(k/32),static_cast<__nv_bfloat16*>(out)+(size_t)first*n};ds41::gemm_native<true><<<dim3((n+31)/32),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,k,e,job);}
    return int(cudaGetLastError());
}
extern "C" int ds41_mxfp4_parallel_raw(const uint8_t*w,const uint8_t*ws,const uint8_t*x,const uint8_t*xs,void*out,int m,int n,int k,void*stream){
    if(m<1 || n<1 || !w || !ws || !x || !xs || !out)return int(cudaErrorInvalidValue);
    if(k!=5120 && k!=2304 && k!=256)return ds41_mxfp4_raw(w,ws,x,xs,out,m,n,k,stream);
    ds41::DeviceExpert e;e.gate=w;e.gate_scale=ws;
    for(int first=0;first<m;first+=8){ds41::GemmJob job{0,m-first<8?m-first:8,x+(size_t)first*k,xs+(size_t)first*(k/32),static_cast<__nv_bfloat16*>(out)+(size_t)first*n};
        if(k==5120)ds41::gemm_parallel<5120,true><<<dim3((n+15)/16),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,e,job);
        else if(k==2304)ds41::gemm_parallel<2304,true><<<dim3((n+15)/16),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,e,job);
        else ds41::gemm_parallel<256,true><<<dim3((n+15)/16),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,e,job);
    }return int(cudaGetLastError());
}
extern "C" int ds41_swiglu_raw(const void*g,const void*u,const float*w,void*out,int rows,int cols,float limit,void*stream){
    if(!g || !u || !out)return int(cudaErrorInvalidValue);
    return int(ds41::swiglu(static_cast<const __nv_bfloat16*>(g),static_cast<const __nv_bfloat16*>(u),w,static_cast<__nv_bfloat16*>(out),rows,cols,limit,static_cast<cudaStream_t>(stream)));
}
extern "C" int ds41_mxfp4_native_parallel_raw(const uint8_t*w,const uint8_t*ws,const uint8_t*x,const uint8_t*xs,void*out,int m,int n,int k,void*stream){
    if(m<1 || n<1 || !w || !ws || !x || !xs || !out)return int(cudaErrorInvalidValue);
    if(k!=5120 && k!=2304 && k!=256)return ds41_mxfp4_native_raw(w,ws,x,xs,out,m,n,k,stream);
    ds41::DeviceExpert e;e.gate=w;e.gate_scale=ws;
    for(int first=0;first<m;first+=8){ds41::GemmJob job{0,m-first<8?m-first:8,x+(size_t)first*k,xs+(size_t)first*(k/32),static_cast<__nv_bfloat16*>(out)+(size_t)first*n};
        if(k==5120)ds41::gemm_native_parallel<5120,8,true><<<dim3((n+7)/8),256,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,e,job);
        else if(k==2304)ds41::gemm_native_parallel<2304,4,true><<<dim3((n+7)/8),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,e,job);
        else ds41::gemm_native_parallel<256,4,true><<<dim3((n+7)/8),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,e,job);
    }return int(cudaGetLastError());
}

extern "C" int ds41_mxfp4_prefill_raw(const uint8_t*w,const uint8_t*ws,const uint8_t*x,const uint8_t*xs,void*out,int m,int n,int k,void*stream){
    if(m<1 || m>256 || n<1 || k<256 || k%256 || !w || !ws || !x || !xs || !out)return int(cudaErrorInvalidValue);
    ds41::DeviceExpert e;e.gate=w;e.gate_scale=ws;
    ds41::GemmJob job{0,m,x,xs,static_cast<__nv_bfloat16*>(out)};
    ds41::gemm_prefill<true><<<dim3((n+63)/64,1,(m+31)/32),128,0,static_cast<cudaStream_t>(stream)>>>(nullptr,nullptr,0,n,k,e,job);
    return int(cudaGetLastError());
}
