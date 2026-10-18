#ifndef DS41_SPARK_H
#define DS41_SPARK_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define DS41_SPARK_ABI 1
#define DS41_SPARK_RNG_VERSION 2
#define DS41_SPARK_VOCAB 129280
typedef struct ds41_spark ds41_spark;
enum ds41_spark_dtype { DS41_SPARK_BF16=0, DS41_SPARK_F32=1, DS41_SPARK_FP8=2, DS41_SPARK_U8=3 };
/* All tensor views are device memory. FP8 scales bind separately as .scale.
 * Native wo_a is dequantized exactly to BF16 by the caller; head, gate and
 * confidence weights are promoted exactly to F32 as in the official loader. */
typedef struct { const void *data; int dtype; int64_t rows, cols; } ds41_spark_tensor;
/* Routed expert core only: x BF16[M,5120], ids int32[M,3], weights F32[M,3],
 * output BF16[M,3,5120], in topk-slot order. stage=0..2 means logical40..42.
 * Use sol-moe native W4A8 kernels, routing before down, and caller stream.
 * Callback may synchronize for offload; must not retain scratch after return. */
typedef int (*ds41_spark_experts_fn)(void*, int stage, const void *x,
    const int32_t *ids, const float *weights, void *output, int rows, void *stream);
/* Optional strict router tie oracle. Same contract as torch.topk(k=3,
 * sorted=True). Default mirrors CUDA strict-winner/threshold-tie gather and
 * shape-dependent bitonic sort; qualification is tied to the recorded Torch build. */
typedef int (*ds41_spark_topk_fn)(void*, const float *scores, int32_t *ids, int rows, void *stream);
typedef int (*ds41_spark_rollback_fn)(void*, int64_t committed_inputs, void *stream);
typedef struct {
    uint32_t abi_version;
    void *stream, *blas_handle, *user;
    ds41_spark_experts_fn experts; /* NULL => use bound native expert tensors */
    ds41_spark_topk_fn topk;
    ds41_spark_rollback_fn rollback;
} ds41_spark_config;
/* Creation allocates bounded scratch and rings (~12 MiB); weights stay borrowed.
 * Caller serializes access, owns the stream/BLAS handle and all bound tensors. */
int ds41_spark_create(const ds41_spark_config*, ds41_spark**);
void ds41_spark_destroy(ds41_spark*);
const char *ds41_spark_error(void);
int ds41_spark_bind(ds41_spark*, const char *official_name, ds41_spark_tensor);
/* Ingest committed target captures [rows,15360] BF16 at absolute positions.
 * Calls must be contiguous after reset. Only committed captures may be passed.
 * rows1..8. For long seed use seed_projected, preserving official full-M norm.
 * The projection result can be exported by project to build a frozen fixture. */
int ds41_spark_project(ds41_spark*, const void *captures, void *main_x, int rows);
int ds41_spark_ingest(ds41_spark*, const void *captures, int64_t start, int rows);
/* Cold seed: captures are the last min(prompt_rows,128) BF16[rows,15360]
 * target capture rows, retaining original full-prompt RMS launch geometry.
 * Resets rings and sets committed length to prompt_rows. */
int ds41_spark_seed(ds41_spark*, const void *last_captures, int64_t prompt_rows);
/* Checked cached-append installation (frozen append-tail-m1-v1 ABI). Caller has consumed the complete
 * target append; captures are the last min(128,end-old) accepted input rows at true positions, BF16
 * [rows,15360], live through completion on the bound stream. Exact M1 project/KV/norm/RoPE per row. */
typedef struct ds41_spark_append_tail_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    int64_t previous_committed;
    int64_t committed_end;
    int64_t capture_begin;
    int32_t rows;
    int32_t columns;
    size_t row_stride_bytes;
    const void *captures_bf16;
    void *cuda_stream;
} ds41_spark_append_tail_v1;
int ds41_spark_restore_append_tail_m1(ds41_spark*, const ds41_spark_append_tail_v1*);
/* main_x BF16[rows,5120] already main_norm(main_proj(captures)). Used for
 * cold prefill seeds with the reference full-prompt RMS reduction geometry. */
int ds41_spark_seed_projected(ds41_spark*, const void *main_x, int64_t start, int rows);
int ds41_spark_reset(ds41_spark*);
/* Draft all five block positions regardless of admitted verification width.
 * anchor occupies position committed_inputs. U F32[5,V], hd BF16[5,5120].
 * No Markov samples or target state are changed. Output pointers are borrowed
 * internal buffers, valid until next forward/reset/destroy. */
int ds41_spark_forward(ds41_spark*, int32_t anchor, const float **U, const void **hd);
/* Markov stage k0..4, previous=anchor or actually sampled preceding draft.
 * Returns U[k]+Markov(previous) F32[V] and raw confidence F32[1] on device. */
int ds41_spark_markov(ds41_spark*, int k, int32_t previous, const float **logits, const float **confidence);
/* Native routed expert primitive for offload adapters. Six views in order
 * w1.weight,w1.scale,w3.weight,w3.scale,w2.weight,w2.scale. x[M,H], route[M],
 * out[M,H]. Separate scratch, so usable inside the experts callback. */
int ds41_spark_expert_raw(ds41_spark*, const void *const weights[6], const void *x,
    const float *route, void *out, int rows);
int ds41_spark_topk3_gpu(const float *scores, int32_t *ids, int rows, void *stream);
/* Calls external target-attention rollback. Draft rings cannot rewind: they
 * hold committed rows only. committed_inputs must equal ingested length. */
int ds41_spark_rollback(ds41_spark*, int64_t committed_inputs);
/* Direct adapter for sol-attn's ds41::VerifyState (state.hpp). Configure as
 * rollback callback when user points to that state, or call from your adapter. */
int ds41_spark_attn_rollback(void *verify_state, int64_t committed_inputs, void *stream);
int64_t ds41_spark_committed(const ds41_spark*);
/* Fixture inspection only: names main,ring0,ring1,ring2,hidden,pre,U.
 * No copy/sync. main is the retained128-row cold projection or latest M1. */
int ds41_spark_debug_view(ds41_spark*,const char*,const void**,size_t*);

enum ds41_spark_purpose { DS41_SPARK_TARGET=0, DS41_SPARK_DRAFT=1, DS41_SPARK_ACCEPT=2, DS41_SPARK_RESIDUAL=3 };
/* draw<=4*UINT32_MAX+3,purpose0..3. Invalid word returns0+last_error;
 * invalid uniform returnsNaN+last_error, never aliases a valid counter. */
uint32_t ds41_spark_rng_word(uint64_t seed, uint64_t position, uint32_t purpose, uint64_t draw);
double ds41_spark_uniform(uint64_t seed, uint64_t position, uint32_t purpose, uint64_t draw);
/* Host oracle and production-safe fallback. p[(drafts+1),V], q[drafts,V]
 * are actual normalized probability measures used by the sampler. A caller
 * transforming logits must retain that exact q, including sampled Markov bias.
 * Returns0 on success; result accepted_input_rows=accepted_drafts+1.
 * Greedy compares target argmax, ignoring q (may be NULL). */
typedef struct {
    int32_t tokens[6]; int emitted, accepted_drafts, accepted_input_rows, rejected;
} ds41_spark_result;
int ds41_spark_sample(const double *p, int vocab, uint64_t seed, uint64_t position, uint32_t purpose, int32_t *token);
int ds41_spark_verify(const int32_t *draft, const double *q, const double *p,
    int drafts, int vocab, int greedy, uint64_t seed, uint64_t first_output_position, ds41_spark_result*);
/* Canonical host logits transform; FP64 exp/normalization with exact top-p
 * threshold inclusion and lower-id tie order. This is a distinct sampling
 * profile, not a claim of bit-identical torch FP32 softmax probabilities. */
int ds41_spark_probabilities(const float*, double*, int vocab, double temperature, double top_p);
/* CUDA samplers consume the actual FP32 measures, normalized consistently in
 * FP64 for draw/accept/residual. result/device token buffers are caller-owned. */
int ds41_spark_sample_gpu(const float*, int vocab, uint64_t seed, uint64_t pos, uint32_t purpose, int32_t*, void *stream);
int ds41_spark_verify_gpu(const int32_t*, const float *q, const float *p,
    int drafts, int vocab, int greedy, uint64_t seed, uint64_t pos, ds41_spark_result*, void *stream);

/* Required RNG v2 protocol for production speculation: increment step for every
 * attempt, including failed/aborted attempts. Never rewind or reuse step.
 * Choose width before sampling any proposal. Slot0..5 is within this step.
 * Use identical seed/step/slot for sampling q and verifying that saved q.
 * Same seed and width sequence reproduce text; cross-width text is not promised.
 * Uniform uses 53 random bits in [0,1); draw<=8589934591. v1 APIs above remain
 * legacy test oracles and must not drive an adaptive speculative transaction. */
uint32_t ds41_spark_rng_word_v2(uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,uint64_t draw);
double ds41_spark_uniform_v2(uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,uint64_t draw);
int ds41_spark_sample_v2(const double*,int vocab,uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,int32_t*);
int ds41_spark_verify_v2(const int32_t*,const double*q,const double*p,int drafts,int vocab,int greedy,uint64_t seed,uint64_t step,ds41_spark_result*);
int ds41_spark_sample_gpu_v2(const float*,int vocab,uint64_t seed,uint64_t step,uint32_t slot,uint32_t purpose,int32_t*,void*stream);
int ds41_spark_verify_gpu_v2(const int32_t*,const float*q,const float*p,int drafts,int vocab,int greedy,uint64_t seed,uint64_t step,ds41_spark_result*,void*stream);

typedef struct {
    int max_width, min_samples; double ewma, prior_acceptance;
    double draft_ms[6], base_verify_ms[6];
    double host_gbps, nvme_gbps, expert_bytes, margin;
} ds41_spark_policy_config;
typedef struct {
    uint64_t steps[6], reached[5], accepted[5];
    double acceptance[5], host_misses[6], nvme_misses[6], measured_ms[6];
} ds41_spark_policy;
void ds41_spark_policy_default(ds41_spark_policy_config*);
void ds41_spark_policy_reset(ds41_spark_policy*, const ds41_spark_policy_config*);
/* Complete a prior transaction before observing it. Censored unreached
 * positions never count as rejections. Misses are previous step totals;
 * host_misses counts all H2D experts, including the NVMe subset. */
int ds41_spark_policy_observe(ds41_spark_policy*, const ds41_spark_policy_config*,
    int width, int accepted_drafts, double verify_ms, double host_misses, double nvme_misses);
/* Width includes anchor. Inputs deliberately exclude proposed identities,
 * current routes, current confidence conditioned on future sampled tokens. */
/* Default exploration: every16th completed-step boundary, width>=3 capped
 * by max_width. choose_ex exposes period(0 disables), width and output budget.
 * A zero remaining budget returns0; otherwise caller must finish at first stop
 * and never ingest beyond it. Observe the original verifier accepted count. */
int ds41_spark_policy_choose_ex(const ds41_spark_policy*,const ds41_spark_policy_config*,int exploration_period,int exploration_width,int remaining_tokens);
int ds41_spark_policy_choose(const ds41_spark_policy*, const ds41_spark_policy_config*);
#ifdef __cplusplus
}
#endif
#endif
