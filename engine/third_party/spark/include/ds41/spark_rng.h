#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define SPARK_HD __host__ __device__
#else
#define SPARK_HD
#endif
SPARK_HD inline uint32_t spark_word(uint64_t seed,uint64_t pos,uint32_t purpose,uint64_t draw) {
    uint32_t c[4]={uint32_t(pos),uint32_t(pos>>32),uint32_t(draw/4),purpose};
    uint32_t k0=uint32_t(seed),k1=uint32_t(seed>>32);
    for(int r=0;r<10;++r){
        uint64_t a=uint64_t(0xD2511F53u)*c[0],b=uint64_t(0xCD9E8D57u)*c[2];
        uint32_t z0=uint32_t(b>>32)^c[1]^k0,z1=uint32_t(b),z2=uint32_t(a>>32)^c[3]^k1,z3=uint32_t(a);
        c[0]=z0;c[1]=z1;c[2]=z2;c[3]=z3;k0+=0x9E3779B9u;k1+=0xBB67AE85u;
    } return c[draw%4];
}
SPARK_HD inline double spark_uniform(uint64_t seed,uint64_t pos,uint32_t purpose,uint64_t draw=0){
    return (double(spark_word(seed,pos,purpose,draw))+.5)/4294967296.0;
}

/* RNG v2: each transaction has a fresh monotone step. Slot is local to that
 * transaction, including its bonus. The domain tag separates v1 counters. */
SPARK_HD inline uint32_t spark_word_v2(uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,uint64_t draw) {
    uint32_t c[4]={uint32_t(step),uint32_t(step>>32),uint32_t(draw/4),0x53500000u|(purpose<<3)|slot};
    uint32_t k0=uint32_t(seed),k1=uint32_t(seed>>32);
    for(int r=0;r<10;++r){
        uint64_t a=uint64_t(0xD2511F53u)*c[0],b=uint64_t(0xCD9E8D57u)*c[2];
        uint32_t z0=uint32_t(b>>32)^c[1]^k0,z1=uint32_t(b),z2=uint32_t(a>>32)^c[3]^k1,z3=uint32_t(a);
        c[0]=z0;c[1]=z1;c[2]=z2;c[3]=z3;k0+=0x9E3779B9u;k1+=0xBB67AE85u;
    } return c[draw%4];
}
SPARK_HD inline double spark_uniform_v2(uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,uint64_t draw=0){
    uint64_t a=spark_word_v2(seed,step,slot,purpose,draw*2)>>5;
    uint64_t b=spark_word_v2(seed,step,slot,purpose,draw*2+1)>>6;
    return double((a<<26)|b)*0x1p-53;
}
