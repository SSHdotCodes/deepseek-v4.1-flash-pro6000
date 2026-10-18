// Included in mxfp4.cu after the exact FP8 MMA/FP4 expansion helpers.
// Native scale metadata follows NVIDIA PTX ISA Block Scaling for mma.sync:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma-block-scaling
// Selector 0 uses quad threads 0/1 for A rows g/g+8 and thread 0 for B row g.
// A CTA reuses each native weight tile across 32 token rows. K groups are
// ALWAYS added serially, with a fresh zero MMA accumulator for every k32.
template<bool Direct=false,bool Native=true>
__global__ void __launch_bounds__(128) gemm_prefill(const DeviceExpert* table,const GemmJob* jobs,
        int projection,int N,int K,DeviceExpert raw_e={},GemmJob raw_job={},int max_tokens=256) {
    constexpr int MT=32,NT=64,Groups=8,WS=144,XS=272;
    constexpr int WBytes=NT*WS,XBytes=MT*XS,ScaleOffset=WBytes+XBytes;
    constexpr int Stage=ScaleOffset+(NT+MT)*Groups;
    __shared__ __align__(16) uint8_t smem[2*Stage];
    GemmJob job=Direct?raw_job:jobs[blockIdx.y];
    int m0=blockIdx.z*MT,n0=blockIdx.x*NT;
    if(job.tokens<1 || job.tokens>max_tokens || job.tokens>256){asm volatile("trap;");return;}
    if(m0>=job.tokens)return;
    DeviceExpert e=Direct?raw_e:table[job.expert];wait_ready(e);
    const uint8_t*w=projection==0?e.gate:projection==1?e.up:e.down;
    const uint8_t*ws=projection==0?e.gate_scale:projection==1?e.up_scale:e.down_scale;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,t=lane&3,ng=K/32;
    int wm=(warp/2)*16,wn=(warp%2)*32;
    float acc[4][4]={};
    auto load=[&](int tile,int stage){
        uint8_t*base=smem+stage*Stage;
        for(int index=threadIdx.x;index<NT*16;index+=128){
            int row=index/16,piece=index%16;
            if(n0+row<N)copy8(base+row*WS+piece*8,w+(size_t)(n0+row)*(K/2)+tile*128+piece*8);
        }
        for(int index=threadIdx.x;index<MT*32;index+=128){
            int row=index/32,piece=index%32;
            if(m0+row<job.tokens)copy8(base+WBytes+row*XS+piece*8,job.x+(size_t)(m0+row)*K+tile*256+piece*8);
        }
        if(threadIdx.x<NT && n0+int(threadIdx.x)<N)
            copy8(base+ScaleOffset+threadIdx.x*Groups,ws+(size_t)(n0+threadIdx.x)*ng+tile*Groups);
        if(threadIdx.x>=NT && threadIdx.x<NT+MT && m0+int(threadIdx.x)-NT<job.tokens)
            copy8(base+ScaleOffset+threadIdx.x*Groups,job.x_scale+(size_t)(m0+threadIdx.x-NT)*ng+tile*Groups);
        asm volatile("cp.async.commit_group;");
    };
    int tiles=ng/Groups;load(0,0);
    for(int tile=0;tile<tiles;++tile){
        int stage=tile&1;
        if(tile+1<tiles){load(tile+1,stage^1);asm volatile("cp.async.wait_group 1;");}
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const uint8_t*base=smem+stage*Stage;
#pragma unroll
        for(int u=0;u<Groups;++u){
            uint32_t a[4]={};
#pragma unroll
            for(int hr=0;hr<2;++hr)if(m0+wm+g+hr*8<job.tokens){
                const uint8_t*x=base+WBytes+(wm+g+hr*8)*XS+u*32+t*4;
                a[hr]=*reinterpret_cast<const uint32_t*>(x);a[hr+2]=*reinterpret_cast<const uint32_t*>(x+16);
            }
#pragma unroll
            for(int j=0;j<4;++j){
                uint32_t b[2]={};
                if(n0+wn+j*8+g<N){
                    const uint8_t*v=base+(wn+j*8+g)*WS+u*16+t*2;
                    if constexpr(Native){b[0]=pad_fp4(*reinterpret_cast<const uint16_t*>(v));b[1]=pad_fp4(*reinterpret_cast<const uint16_t*>(v+8));}
                    else {b[0]=expand4(*reinterpret_cast<const uint16_t*>(v));b[1]=expand4(*reinterpret_cast<const uint16_t*>(v+8));}
                }
                float dot[4]={};
                if constexpr(Native){
                    int ar=wm+g+((t&1)?8:0);
                    uint32_t sa=m0+ar<job.tokens?base[ScaleOffset+(NT+ar)*Groups+u]:127;
                    uint32_t sb=n0+wn+j*8+g<N?base[ScaleOffset+(wn+j*8+g)*Groups+u]:127;
                    mma_native(dot,a,b,sa,sb);
#pragma unroll
                    for(int c=0;c<4;++c)acc[j][c]=__fadd_rn(acc[j][c],dot[c]);
                }else {mma(dot,a,b);
#pragma unroll
                for(int hr=0;hr<2;++hr){
                    float sx=m0+wm+g+hr*8<job.tokens?e8m0(base[ScaleOffset+(NT+wm+g+hr*8)*Groups+u]):0.f;
#pragma unroll
                    for(int cc=0;cc<2;++cc){
                        int col=wn+j*8+t*2+cc;
                        float sw=n0+col<N?e8m0(base[ScaleOffset+col*Groups+u]):0.f;
                        int c=hr*2+cc;
                        acc[j][c]=__fadd_rn(acc[j][c],__fmul_rn(__fmul_rn(dot[c],sx),sw));
                    }
                }}
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int j=0;j<4;++j)
#pragma unroll
        for(int hr=0;hr<2;++hr)
#pragma unroll
            for(int cc=0;cc<2;++cc){
                int m=m0+wm+g+hr*8,n=n0+wn+j*8+t*2+cc;
                if(m<job.tokens && n<N)job.output[(size_t)m*N+n]=__float2bfloat16_rn(acc[j][hr*2+cc]);
            }
}
