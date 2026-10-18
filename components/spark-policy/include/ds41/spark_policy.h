#ifndef DS41_SPARK_POLICY_V2_H
#define DS41_SPARK_POLICY_V2_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define DS41_SPARK_POLICY_VERSION 2
#define DS41_SPARK_POLICY_CELLS 192
enum ds41_spark_mode { DS41_SPARK_MODE_UNKNOWN=0, DS41_SPARK_MODE_CHAT=1,
    DS41_SPARK_MODE_THINKING=2, DS41_SPARK_MODE_AGENTIC=3 };
enum ds41_spark_token_kind { DS41_SPARK_KIND_UNKNOWN=0, DS41_SPARK_KIND_PROSE=1,
    DS41_SPARK_KIND_CODE=2, DS41_SPARK_KIND_TOOL=3 };
typedef struct {
    uint32_t size, version;
    int32_t max_width, qualified_mask, exploration_period, exploration_width, min_cost_samples;
    int32_t min_kind_samples, min_confidence_samples;
    double prior_strength, cell_strength, margin, min_preflight_gain, confidence_cut[2];
    double verify_ms[6], draft_ms[6], other_ms[6], prior_survival[4][5];
} ds41_spark_policy_config_v2;
typedef struct { uint64_t reached[5], accepted[5]; } ds41_spark_policy_counts_v2;
typedef struct {
    uint32_t size, version;
    uint64_t generation, completed, last_nonce;
    int32_t has_nonce;
    ds41_spark_policy_counts_v2 mode[12], kind[48], cell[DS41_SPARK_POLICY_CELLS];
    uint64_t cost_samples[6];
    double verify_ms[6], draft_ms[6], other_ms[6], host_misses[6], nvme_misses[6];
} ds41_spark_policy_state_v2;
/* All fields are known before any current draft is sampled. known_anchor is
 * the already emitted target token. token_kind comes only from observed text.
 * confidence_available means markov(0,known_anchor) raw confidence, never a
 * later head conditioned on d1...d5. paid_draft_ms includes sunk forward/q0
 * costs, even if the final width is1. Each attempt uses a fresh nonce. */
typedef struct {
    uint32_t size, version;
    uint64_t nonce, committed_inputs;
    int32_t known_anchor, remaining_tokens, max_width, qualified_mask;
    int32_t mode, token_kind, confidence_available;
    double confidence_raw, temperature, top_p, paid_draft_ms;
} ds41_spark_policy_features_v2;
typedef struct {
    uint32_t size, version;
    uint64_t nonce, generation;
    int32_t width, mode, cell, exploration;
    double paid_draft_ms;
    double predicted_emitted[6], predicted_ms[6], predicted_tps[6];
    double predicted_host_misses[6], predicted_nvme_misses[6];
} ds41_spark_policy_decision_v2;
typedef struct {
    uint32_t size, version;
    int32_t width, accepted_drafts, tested_drafts, timing_valid;
    double verify_ms, draft_ms, other_ms, host_misses, nvme_misses;
} ds41_spark_policy_feedback_v2;
void ds41_spark_policy_default_v2(ds41_spark_policy_config_v2*);
int ds41_spark_policy_reset_v2(ds41_spark_policy_state_v2*,const ds41_spark_policy_config_v2*);
/* Optional frozen calibration, before the first runtime attempt. No costs or
 * nonce/generation are consumed. Cells use actual q0 confidence and observed
 * prefix labels; calibration is not held-out validation. */
int ds41_spark_policy_seed_v2(ds41_spark_policy_state_v2*,const ds41_spark_policy_config_v2*,
    const ds41_spark_policy_features_v2*,int width,int accepted_drafts,int tested_drafts);
/* Pure decision: call once with no confidence for preflight. Width1 may skip
 * forward. Otherwise run forward/q0 ONCE, then optionally call again with its
 * raw confidence and measured paid_draft_ms. Freeze before the first draw.
 * No current proposal/target/route/residency buffer is accepted by this API. */
int ds41_spark_policy_decide_v2(const ds41_spark_policy_state_v2*,const ds41_spark_policy_config_v2*,
    const ds41_spark_policy_features_v2*,ds41_spark_policy_decision_v2*);
/* Observe only after transaction completion. tested_drafts is accepted or
 * accepted+1: stop/cap-truncated untested rows remain censored. Times are
 * disjoint: target verify, all paid draft work, and preflight/commit/ingest/
 * rollback/other. Miss deltas are target-only, with host INCLUDING NVMe. */
int ds41_spark_policy_observe_v2(ds41_spark_policy_state_v2*,const ds41_spark_policy_config_v2*,
    const ds41_spark_policy_decision_v2*,const ds41_spark_policy_feedback_v2*);
/* Aborts retire nonce/generation without treating unverified rows as failures. */
int ds41_spark_policy_abort_v2(ds41_spark_policy_state_v2*,uint64_t nonce);
#ifdef __cplusplus
}
#endif
#endif
