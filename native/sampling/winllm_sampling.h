// C interface between the nvcc-compiled kernels (winllm_sampling.cu) and
// the MSVC-compiled pybind11 binding (binding.cpp). Torch-free on purpose;
// see winllm_sampling.cu for why.
#pragma once

#include <cstdint>

#define WLS_DTYPE_F16 0
#define WLS_DTYPE_F32 1
#define WLS_DTYPE_BF16 2

extern "C" {

// Fused repetition-penalty/temperature/top-k/top-p/categorical-draw over
// [batch, vocab] logits. All pointers are raw device addresses; `scratch`
// must hold batch * 2 * vocab fp32 values, `out` batch int64 values,
// `pen_ids` n_pen unique int32 token ids (0/0 to disable). Returns a
// cudaError_t from the launch (0 == success); the kernel itself is async
// on `stream`.
int wls_sample(
    uint64_t logits, int dtype, long long row_stride, int batch, int vocab,
    uint64_t pen_ids, int n_pen, float penalty,
    float temperature, int top_k, float top_p,
    uint64_t scratch, uint64_t out,
    unsigned long long seed, unsigned long long offset, uint64_t stream);

// Set logits to -inf wherever the packed bitmask (int32 words, bit i of
// word i/32, 1 = allowed — xgrammar's layout) has a zero bit. In place.
int wls_apply_bitmask(
    uint64_t logits, int dtype, long long row_stride, int batch, int vocab,
    uint64_t mask, long long mask_row_words, uint64_t stream);

const char* wls_error_name(int err);

}  // extern "C"
