#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace ds41::exl3_detail {
// Trellis extraction and mul1 codebook adapted from exllamav3 v1.6.3 (MIT).
// Copyright (c) 2025 Turboderp. See vendor/EXLLAMAV3_LICENSE.
struct Pair { uint32_t v[2]; };
__device__ __forceinline__ uint32_t pair(float x, float y) {
    half2 v=__floats2half2_rn(x,y); return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t decode_pair(uint32_t x, uint32_t y) {
    x *= 0x83DCD12Du; y *= 0x83DCD12Du;
    uint32_t sx=__dp4a(x,0x01010101u,0x6400u), sy=__dp4a(y,0x01010101u,0x6400u);
    half2 f=__halves2half2(__ushort_as_half(sx),__ushort_as_half(sy));
    half2 v=__hfma2(f,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931))); return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void dq3(uint32_t word, Pair& f0, Pair& f1, int lane) {
    const int b1=(lane*8+257)*3, b2=b1+21;
    const int i0=(b1-16)/32, i2=(b2-1)/32, s=(i2+1)*32-b2;
    uint32_t a=__shfl_sync(0xffffffff,word,i0%24), b=__shfl_sync(0xffffffff,word,i2%24);
    uint64_t joined=(uint64_t(a)<<32)|b;
    uint32_t w7=joined>>s, w3=joined>>(s+12);
    f0.v[0]=decode_pair((w3>>9)&65535,(w3>>6)&65535);
    f0.v[1]=decode_pair((w3>>3)&65535,w3&65535);
    f1.v[0]=decode_pair((w7>>9)&65535,(w7>>6)&65535);
    f1.v[1]=decode_pair((w7>>3)&65535,w7&65535);
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], Pair b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b.v[0]),"r"(b.v[1]));
}
__device__ __forceinline__ void had128(float& a,float& b,float& c,float& d,int lane) {
    float s0=a+b,d0=a-b,s1=c+d,d1=c-d;
    a=s0+s1; b=d0+d1; c=s0-s1; d=d0-d1;
    #pragma unroll
    for(int mask=1;mask<32;mask*=2) {
        float pa=__shfl_xor_sync(0xffffffff,a,mask),pb=__shfl_xor_sync(0xffffffff,b,mask);
        float pc=__shfl_xor_sync(0xffffffff,c,mask),pd=__shfl_xor_sync(0xffffffff,d,mask);
        float sign=lane&mask?-1.f:1.f;
        a=sign*a+pa; b=sign*b+pb; c=sign*c+pc; d=sign*d+pd;
    }
    a*=0.088388347648f;b*=0.088388347648f;c*=0.088388347648f;d*=0.088388347648f;
}
__device__ __forceinline__ void wait_ready(const DeviceExpert& e) {
    if(!e.ready)return;
    uint32_t value;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(e.ready):"memory");
        if(value!=e.generation)__nanosleep(64);
    }while(value!=e.generation);
}
__device__ __forceinline__ const uint32_t* weights(const DeviceExpert& e,int p) {
    return reinterpret_cast<const uint32_t*>(p==0?e.gate:p==1?e.up:e.down);
}
__device__ __forceinline__ const half* scales(const DeviceExpert& e,int p) {
    return reinterpret_cast<const half*>(p==0?e.gate_scale:p==1?e.up_scale:e.down_scale);
}
__device__ __forceinline__ void rotate_input(const __nv_bfloat16* x,const half* su,half* out,int lane) {
    int c=lane*4;
    float a=__bfloat162float(x[c])*__half2float(su[c]);
    float b=__bfloat162float(x[c+1])*__half2float(su[c+1]);
    float d=__bfloat162float(x[c+2])*__half2float(su[c+2]);
    float e=__bfloat162float(x[c+3])*__half2float(su[c+3]);
    had128(a,b,d,e,lane);
    reinterpret_cast<uint2*>(out)[lane]=make_uint2(pair(a,b),pair(d,e));
}
__device__ __forceinline__ void rotate_output(const float* x,const half* sv,__nv_bfloat16* out,int lane) {
    float4 v=reinterpret_cast<const float4*>(x)[lane];
    float a=v.x,b=v.y,c=v.z,d=v.w;
    had128(a,b,c,d,lane);
    out[lane*4]=__float2bfloat16_rn(a*__half2float(sv[lane*4]));
    out[lane*4+1]=__float2bfloat16_rn(b*__half2float(sv[lane*4+1]));
    out[lane*4+2]=__float2bfloat16_rn(c*__half2float(sv[lane*4+2]));
    out[lane*4+3]=__float2bfloat16_rn(d*__half2float(sv[lane*4+3]));
}
}
