#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void sparse_attn_kernel__kernel(const float* __restrict__ attn_sink, const bfloat16_t* __restrict__ kv, bfloat16_t* __restrict__ o, const bfloat16_t* __restrict__ q, const int* __restrict__ topk_idxs, int b, int m, int n, int topk);
extern "C" __global__ void __launch_bounds__(256, 1) sparse_attn_kernel__kernel(const float* __restrict__ attn_sink, const bfloat16_t* __restrict__ kv, bfloat16_t* __restrict__ o, const bfloat16_t* __restrict__ q, const int* __restrict__ topk_idxs, int b, int m, int n, int topk) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* kv_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* acc_s_cast = ((void*)((char*)buf_dyn_shmem + 81920));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 83968));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 84992));
  float acc_o[32];
  float sum_exp[2];
  float scores_max[2];
  int idxs[2];
  float acc_s[4];
  float scores_max_prev[2];
  float scores_max_clear[2];
  float scores_scale[2];
  float scores_sum[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(sum_exp + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -0x1.93e5939a08ceap+99f/*-1.000000e+30*/;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(((bfloat16_t*)q_shared) + ((((((((((int)threadIdx.x) & 63) >> 3) * 1024) + (i_1 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(q + ((((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)m)) * (int64_t)8192)) + (((int64_t)i_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
  }
  for (int t = 0; t < ((topk + 63) >> 6); ++t) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      int condval;
      if ((((((t * 64) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + i_2) < topk)) {
        condval = topk_idxs[((((((((int64_t)t) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)topk)) * ((int64_t)m))) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))) + ((int64_t)i_2))];
      } else {
        condval = -1;
      }
      idxs[i_2] = condval;
    }
    __syncthreads();
    if (((((int)threadIdx.x) & 31) >> 2) == 0) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 128; ++i_3) {
        bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_1;
        if ((((idxs[(i_3 >> 6)] != -1) && (0 <= idxs[(i_3 >> 6)])) && (idxs[(i_3 >> 6)] < n))) {
          condval_1 = *(uint4*)(kv + (((((int64_t)idxs[(((int64_t)i_3) >> (int64_t)6)]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)n)) * (int64_t)512)) + ((((int64_t)i_3) & (int64_t)63) * (int64_t)8)));
        } else {
          condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
        }
        *(uint4*)(((bfloat16_t*)kv_shared) + (((((((((i_3 & 63) >> 3) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + ((i_3 >> 6) * 64)) + (((((i_3 & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_3 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + ((((i_3 >> 6) + (i_3 & 1)) & 1) * 8))) = condval_1;
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float condval_2;
      if ((idxs[(i_4 & 1)] != -1)) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = -CUDART_INF_F;
      }
      acc_s[i_4] = condval_2;
    }
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[4];
      __syncthreads();
      for (int ki = 0; ki < 32; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_shared)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((bfloat16_t*)kv_shared)[(((((((ki >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      acc_s[i_5] = (acc_s[i_5] * 0x1.6a09e667f3bcdp-5f/*4.419417e-02*/);
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_max_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        scores_max_clear[i_6] = max(scores_max_clear[i_6], acc_s[((i_6 * 2) + rv)]);
      }
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6], (&(((float*)workspace_1)[0])));
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_max_clear[i_6]);
      scores_max[i_6] = max(scores_max[i_6], scores_max_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_scale[i_7] = expf((scores_max_prev[i_7] - scores_max[i_7]));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      acc_s[i_8] = expf((acc_s[i_8] - scores_max[(i_8 >> 1)]));
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_sum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_9] = (scores_sum[i_9] + acc_s[((i_9 * 2) + rv_1)]);
      }
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(scores_sum[i_9], (&(((float*)workspace)[0])));
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      sum_exp[i_10] = ((sum_exp[i_10] * scores_scale[i_10]) + scores_sum[i_10]);
    }
    tl::ptx_stmatrix_m8n8_x2((&(((bfloat16_t*)acc_s_cast)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc_s[0]), ((bfloat16_t)acc_s[1])), __pack_half2(((bfloat16_t)acc_s[2]), ((bfloat16_t)acc_s[3])));
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      acc_o[i_11] = (acc_o[i_11] * scores_scale[((i_11 & 3) >> 1)]);
    }
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[32];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)acc_s_cast)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
        for (int i_12 = 0; i_12 < 4; ++i_12) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)kv_shared)[(((((((int)threadIdx.x) >> 5) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_12 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    sum_exp[i_13] = (sum_exp[i_13] + expf((attn_sink[((i_13 * 8) + ((((int)threadIdx.x) & 31) >> 2))] - scores_max[i_13])));
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    acc_o[i_14] = (acc_o[i_14] / sum_exp[((i_14 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)o_shared)[(((((((int)threadIdx.x) & 15) * 512) + ((((int)threadIdx.x) >> 5) * 64)) + (i_15 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((bfloat16_t)acc_o[(i_15 * 8)]), ((bfloat16_t)acc_o[((i_15 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_15 * 8) + 2)]), ((bfloat16_t)acc_o[((i_15 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_15 * 8) + 4)]), ((bfloat16_t)acc_o[((i_15 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_15 * 8) + 6)]), ((bfloat16_t)acc_o[((i_15 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 4; ++i_16) {
    *(uint4*)(o + ((((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)m)) * (int64_t)8192)) + (((int64_t)i_16) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((bfloat16_t*)o_shared) + ((i_16 * 2048) + (((int)threadIdx.x) * 8)));
  }
}

