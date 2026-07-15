// Fused CUDA sampling kernel for wLLM.
//
// One kernel launch replaces the whole per-token sampling pipeline
// (repetition penalty -> temperature -> top-k -> top-p -> categorical draw),
// which in the torch implementation is ~15 kernel launches per decode step.
// On Windows (WDDM) launch/submission overhead dominates decode-path CPU
// time (see documentation/PERFORMANCE_NOTES.md), so fusing the pipeline is
// worth far more here than the arithmetic it saves.
//
// Deliberately torch-free: the extension takes raw device pointers and the
// CUDA stream handle from Python (tensor.data_ptr(), stream.cuda_stream).
// This sidesteps the toolchain constraint that torch is built with CUDA 12.8
// while the installed toolkit is 13.2 — torch.utils.cpp_extension refuses
// that mismatch, but a standalone module statically linked against its own
// cudart only needs the driver, and driver >= r580 covers both.
//
// Semantics match winllm/sampling/ops.py:
//   - repetition penalty: v>0 ? v/p : v*p on the (unique) generated ids
//   - temperature: softmax over logits/T
//   - top-k: keep logits >= (k-th largest); ties at the threshold survive,
//     exactly like ops.py's masked_fill_(logits < threshold)
//   - top-p: nucleus over the softmax of the top-k survivors; a token is
//     kept iff the probability mass strictly above it is < top_p. This is
//     torch's sorted-cumsum rule except that value ties at the boundary are
//     all kept (torch keeps a sort-order-dependent subset of them).
//   - temperature == 0: penalized argmax (first index wins ties).
// All math is fp32 regardless of the logits dtype. Thresholds are found by
// an exact MSB radix select on the float bit patterns (positive floats
// compare like their uint32 representations), so top-k is bit-exact and
// top-p is exact up to fp32 summation order.
//
// The caller's logits are never mutated; each row is staged into a caller
// provided fp32 scratch buffer of shape [batch, 2, vocab] (penalized logits
// + probabilities). One block per row; RNG is Philox
// (seed, subsequence=row, offset=call counter), independent of torch's
// generators — which is why seeded requests must stay on the torch path.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <cstdint>

#include "winllm_sampling.h"

namespace {

constexpr int BLOCK = 256;  // must equal the radix histogram bin count

__device__ __forceinline__ float load_logit(const void* logits, long long idx, int dtype) {
    switch (dtype) {
        case WLS_DTYPE_F16: return __half2float(static_cast<const __half*>(logits)[idx]);
        case WLS_DTYPE_BF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(logits)[idx]);
        default: return static_cast<const float*>(logits)[idx];
    }
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// Round through the logits dtype. The torch pipeline computes the
// repetition penalty in the tensor's own dtype, so penalized values must
// lose precision the same way or temp-0 output diverges from the torch
// path on near-ties (observed in practice on fp16).
__device__ __forceinline__ float round_to_dtype(float v, int dtype) {
    switch (dtype) {
        case WLS_DTYPE_F16: return __half2float(__float2half(v));
        case WLS_DTYPE_BF16: return __bfloat162float(__float2bfloat16(v));
        default: return v;
    }
}

// Tree reduction over the block; every thread returns the result.
__device__ float block_max(float v, float* sf) {
    sf[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sf[threadIdx.x] = fmaxf(sf[threadIdx.x], sf[threadIdx.x + s]);
        __syncthreads();
    }
    float r = sf[0];
    __syncthreads();
    return r;
}

__device__ float block_sum(float v, float* sf) {
    sf[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sf[threadIdx.x] += sf[threadIdx.x + s];
        __syncthreads();
    }
    float r = sf[0];
    __syncthreads();
    return r;
}

// Argmax over the fp32 row; first index wins ties (deterministic, matching
// how the wrapper's tests pin greedy behavior).
__device__ long long row_argmax(const float* y, int vocab, float* sf, int* si) {
    float best = neg_inf();
    int best_i = vocab;  // sentinel: larger than any real index
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = y[i];
        if (v > best || (v == best && i < best_i)) { best = v; best_i = i; }
    }
    sf[threadIdx.x] = best;
    si[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = sf[threadIdx.x + s];
            int oi = si[threadIdx.x + s];
            if (ov > sf[threadIdx.x] || (ov == sf[threadIdx.x] && oi < si[threadIdx.x])) {
                sf[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    int r = si[0];
    __syncthreads();
    return r >= vocab ? 0 : r;  // all-(-inf)/NaN row: fall back to token 0
}

// Exact k-th largest probability, returned as float bits. MSB-first radix
// select over the uint32 view of the (non-negative) probabilities: 4 passes
// of a 256-bin count histogram, narrowing one byte per pass.
__device__ unsigned int radix_kth_largest(
    const float* p, int vocab, int k, int* hist, int* s_digit, int* s_rem) {
    unsigned long long prefix = 0;
    int k_rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[threadIdx.x] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < vocab; i += BLOCK) {
            unsigned int b = __float_as_uint(p[i]);
            if ((static_cast<unsigned long long>(b) >> (shift + 8)) == prefix)
                atomicAdd(&hist[(b >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, d = 255;
            for (; d > 0; --d) {
                if (acc + hist[d] >= k_rem) break;
                acc += hist[d];
            }
            *s_digit = d;
            *s_rem = k_rem - acc;
        }
        __syncthreads();
        prefix = (prefix << 8) | static_cast<unsigned int>(*s_digit);
        k_rem = *s_rem;
        __syncthreads();
    }
    return static_cast<unsigned int>(prefix);
}

// Nucleus boundary value (as float bits) among the top-k survivors
// (bits >= floor_bits): the largest value v such that the mass strictly
// above v is still < target. Same radix walk as above but on mass
// histograms. If fp rounding makes the total mass fall short of target the
// walk bottoms out at digit 0, which errs toward keeping more tokens.
__device__ unsigned int radix_nucleus_boundary(
    const float* p, int vocab, unsigned int floor_bits, float target,
    float* mass, int* s_digit, float* s_rem) {
    unsigned long long prefix = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        mass[threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = threadIdx.x; i < vocab; i += BLOCK) {
            unsigned int b = __float_as_uint(p[i]);
            if (b >= floor_bits &&
                (static_cast<unsigned long long>(b) >> (shift + 8)) == prefix)
                atomicAdd(&mass[(b >> shift) & 0xFF], p[i]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float acc = 0.0f;
            int d = 255;
            for (; d > 0; --d) {
                if (acc + mass[d] >= target) break;
                acc += mass[d];
            }
            *s_digit = d;
            *s_rem = fmaxf(target - acc, 0.0f);
        }
        __syncthreads();
        prefix = (prefix << 8) | static_cast<unsigned int>(*s_digit);
        target = *s_rem;
        __syncthreads();
    }
    return static_cast<unsigned int>(prefix);
}

__global__ void fused_sample_kernel(
    const void* __restrict__ logits, int dtype, long long row_stride, int vocab,
    const int* __restrict__ pen_ids, int n_pen, float penalty,
    float temperature, int top_k, float top_p,
    float* __restrict__ scratch, long long* __restrict__ out,
    unsigned long long seed, unsigned long long offset) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float* y = scratch + static_cast<long long>(row) * 2 * vocab;  // penalized logits
    float* p = y + vocab;                                          // unnormalized probs
    const long long base = static_cast<long long>(row) * row_stride;

    __shared__ float sf[BLOCK];
    __shared__ int si[BLOCK];
    __shared__ int s_digit;
    __shared__ int s_irem;
    __shared__ float s_frem;
    __shared__ float s_u;
    __shared__ int s_pick;

    // Stage the row as fp32 and apply the repetition penalty. Ids are unique
    // (the wrapper dedups), so each logit is penalized at most once.
    for (int i = tid; i < vocab; i += BLOCK) y[i] = load_logit(logits, base + i, dtype);
    __syncthreads();
    if (n_pen > 0 && penalty != 1.0f) {
        for (int j = tid; j < n_pen; j += BLOCK) {
            int id = pen_ids[j];
            if (id >= 0 && id < vocab) {
                float v = y[id];
                y[id] = round_to_dtype(v > 0.0f ? v / penalty : v * penalty, dtype);
            }
        }
        __syncthreads();
    }

    if (temperature <= 0.0f) {  // greedy row
        if (tid == 0) out[row] = 0;  // overwritten below; placate the compiler
        long long a = row_argmax(y, vocab, sf, si);
        if (tid == 0) out[row] = a;
        return;
    }

    float m = neg_inf();
    for (int i = tid; i < vocab; i += BLOCK) m = fmaxf(m, y[i]);
    m = block_max(m, sf);
    if (!isfinite(m)) {  // all -inf (fully masked) or NaN: match torch's argmax fallback
        long long a = row_argmax(y, vocab, sf, si);
        if (tid == 0) out[row] = a;
        return;
    }

    const float inv_t = 1.0f / temperature;
    float local = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        float e = expf((y[i] - m) * inv_t);
        p[i] = e;
        local += e;
    }
    float total = block_sum(local, sf);

    unsigned int tau = 0;  // keep every token with prob bits >= tau
    float kept_mass = total;
    if (top_k > 0 && top_k < vocab) {
        tau = radix_kth_largest(p, vocab, top_k, si, &s_digit, &s_irem);
        local = 0.0f;
        for (int i = tid; i < vocab; i += BLOCK)
            if (__float_as_uint(p[i]) >= tau) local += p[i];
        kept_mass = block_sum(local, sf);
    }
    if (top_p < 1.0f) {
        unsigned int tau_p = radix_nucleus_boundary(
            p, vocab, tau, top_p * kept_mass, sf, &s_digit, &s_frem);
        if (tau_p > tau) tau = tau_p;
    }

    // Categorical draw over the kept set. Enumeration order is
    // thread-contiguous (thread t owns elements t, t+BLOCK, ...) — any fixed
    // order samples the same distribution. The per-thread partials and the
    // scan total are the boundaries AND the walk sums, computed in the same
    // order, so the chosen interval always contains the draw exactly.
    float part = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK)
        if (__float_as_uint(p[i]) >= tau) part += p[i];
    sf[tid] = part;
    __syncthreads();
    for (int off = 1; off < BLOCK; off <<= 1) {
        float t = tid >= off ? sf[tid - off] : 0.0f;
        __syncthreads();
        sf[tid] += t;
        __syncthreads();
    }
    float z = sf[BLOCK - 1];
    float excl = tid == 0 ? 0.0f : sf[tid - 1];
    float incl = sf[tid];
    if (!(z > 0.0f) || !isfinite(z)) {
        __syncthreads();
        long long a = row_argmax(y, vocab, sf, si);
        if (tid == 0) out[row] = a;
        return;
    }

    if (tid == 0) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, static_cast<unsigned long long>(row), offset, &st);
        s_u = curand_uniform(&st) * z;  // in (0, z]
        s_pick = -1;
    }
    __syncthreads();
    const float u = s_u;
    if (part > 0.0f && u > excl && u <= incl) {  // exactly one thread claims
        float acc = excl;
        int pick = -1, last = -1;
        for (int i = tid; i < vocab; i += BLOCK) {
            if (__float_as_uint(p[i]) >= tau && p[i] > 0.0f) {
                last = i;
                acc += p[i];
                if (acc >= u) { pick = i; break; }
            }
        }
        s_pick = pick >= 0 ? pick : last;
    }
    __syncthreads();
    if (tid == 0) out[row] = s_pick >= 0 ? s_pick : row_argmax(y, vocab, sf, si);
}

__global__ void apply_bitmask_kernel(
    void* __restrict__ logits, int dtype, long long row_stride, int vocab,
    const int* __restrict__ mask, long long mask_row_words) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    if ((mask[row * mask_row_words + (i >> 5)] >> (i & 31)) & 1) return;
    const long long idx = static_cast<long long>(row) * row_stride + i;
    switch (dtype) {
        case WLS_DTYPE_F16:
            static_cast<__half*>(logits)[idx] = __float2half(__int_as_float(0xff800000));
            break;
        case WLS_DTYPE_BF16:
            static_cast<__nv_bfloat16*>(logits)[idx] = __float2bfloat16(__int_as_float(0xff800000));
            break;
        default:
            static_cast<float*>(logits)[idx] = __int_as_float(0xff800000);
            break;
    }
}

}  // namespace

extern "C" {

int wls_sample(
    uint64_t logits, int dtype, long long row_stride, int batch, int vocab,
    uint64_t pen_ids, int n_pen, float penalty,
    float temperature, int top_k, float top_p,
    uint64_t scratch, uint64_t out,
    unsigned long long seed, unsigned long long offset, uint64_t stream) {
    fused_sample_kernel<<<batch, BLOCK, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const void*>(logits), dtype, row_stride, vocab,
        reinterpret_cast<const int*>(pen_ids), n_pen, penalty,
        temperature, top_k, top_p,
        reinterpret_cast<float*>(scratch), reinterpret_cast<long long*>(out),
        seed, offset);
    return static_cast<int>(cudaPeekAtLastError());
}

int wls_apply_bitmask(
    uint64_t logits, int dtype, long long row_stride, int batch, int vocab,
    uint64_t mask, long long mask_row_words, uint64_t stream) {
    dim3 grid((vocab + BLOCK - 1) / BLOCK, batch);
    apply_bitmask_kernel<<<grid, BLOCK, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<void*>(logits), dtype, row_stride, vocab,
        reinterpret_cast<const int*>(mask), mask_row_words);
    return static_cast<int>(cudaPeekAtLastError());
}

const char* wls_error_name(int err) {
    return cudaGetErrorName(static_cast<cudaError_t>(err));
}

}  // extern "C"
