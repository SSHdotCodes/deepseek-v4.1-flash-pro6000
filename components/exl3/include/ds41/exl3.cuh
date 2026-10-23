#pragma once
#include <ds41/kernels.cuh>

namespace ds41 {
// K=3, mul1 EXL3. DeviceExpert weights are [K/16,N/16,48] uint16;
// each scale pointer holds suh[K] followed immediately by svh[N], both FP16.
// GemmJob::x is BF16 [tokens,K], output BF16 [tokens,N]. x_scale is ignored.
// Device-resident jobs/table are reread on every invocation and graph replay.
// Dimensions must be positive multiples of 128; k <= 5120. No scratch needed.
// Decode supports k=128,256,2304,5120; prefill supports any such multiple.
// Required DS41 dimensions: gate/up k=5120,n=2304; down k=2304,n=5120.
// Trellis addresses need 4-byte alignment, BF16/FP16 addresses 2-byte alignment.
// Decode tokens: 1..8. Prefill tokens: 1..max_tokens, max_tokens <= 256.
cudaError_t grouped_exl3(const DeviceExpert* table, const GemmJob* jobs, int count,
                         Projection projection, int n, int k, cudaStream_t stream);
cudaError_t grouped_exl3_prefill(const DeviceExpert* table, const GemmJob* jobs, int count,
                         Projection projection, int n, int k, int max_tokens, cudaStream_t stream);
}
