#pragma once
#include "ds41/spark.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ds41_spark_detail {
// Host-only checked plan, shared by the production enqueue path and CPU state
// gates. No caller can use this helper to weaken legacy ingest's cursor check.
struct AppendTailPlan {
    int64_t previous, begin, end;
    int rows;
    uintptr_t captures;
    size_t stride;
    const void *row(int i) const {
        return reinterpret_cast<const void *>(captures + size_t(i) * stride);
    }
};
inline AppendTailPlan checked_append_tail(const ds41_spark_append_tail_v1 *d,
        int64_t committed, void *stream, int64_t rope_rows, bool active, bool poisoned) {
    if (poisoned) throw std::runtime_error("append restore: poisoned state; reset required");
    if (!d || d->abi_version != 1 || d->struct_bytes != sizeof(*d) || active ||
        committed < 1 || d->previous_committed != committed ||
        d->committed_end <= committed || d->committed_end > rope_rows ||
        d->columns != 15360 || d->row_stride_bytes != 30720 ||
        !d->captures_bf16 || d->cuda_stream != stream)
        throw std::invalid_argument("append restore: descriptor/frontier/stream");
    const int rows = int(std::min<int64_t>(128, d->committed_end - committed));
    const auto base = reinterpret_cast<uintptr_t>(d->captures_bf16);
    const size_t bytes = size_t(rows) * 30720;
    if (d->rows != rows || d->capture_begin != d->committed_end - rows ||
        base % 2 || base > std::numeric_limits<uintptr_t>::max() - (bytes - 1))
        throw std::invalid_argument("append restore: incomplete tail/alignment/extent");
    return {committed, d->capture_begin, d->committed_end, rows, base, 30720};
}
template<class EnqueueRow>
void apply_append_tail(const AppendTailPlan &p, int64_t &committed,
                       bool &drafted, bool &poisoned, EnqueueRow enqueue) {
    if (poisoned || committed != p.previous)
        throw std::invalid_argument("append restore: stale private frontier");
    // The private frontier jump publishes once, after the entire tail is
    // enqueued. Every row still uses the unchanged M1 projection/write path.
    try {
        for (int i = 0; i < p.rows; ++i) enqueue(p.row(i), p.begin + i);
    } catch (...) {
        poisoned = true;
        drafted = false;
        throw;
    }
    committed = p.end;
    drafted = false;
}
} // namespace ds41_spark_detail
