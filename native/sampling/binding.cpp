// pybind11 binding for the fused CUDA sampling kernels. Compiled by MSVC
// (no CUDA headers here); the kernels live in winllm_sampling.cu, compiled
// by nvcc and linked in as an object file. See winllm_sampling.cu for the
// design notes and why this module is torch-free.
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "winllm_sampling.h"

namespace py = pybind11;

namespace {

void check(int rc) {
    if (rc != 0)
        throw std::runtime_error(std::string("CUDA error: ") + wls_error_name(rc));
}

}  // namespace

PYBIND11_MODULE(winllm_sampling, m) {
    m.doc() =
        "Fused CUDA sampling for wLLM (optional accelerator; wLLM falls back "
        "to the pure-torch pipeline without it)";

    m.def(
        "sample",
        [](uint64_t logits, int dtype, long long row_stride, int batch, int vocab,
           uint64_t pen_ids, int n_pen, float penalty,
           float temperature, int top_k, float top_p,
           uint64_t scratch, uint64_t out,
           unsigned long long seed, unsigned long long offset, uint64_t stream) {
            check(wls_sample(logits, dtype, row_stride, batch, vocab,
                             pen_ids, n_pen, penalty, temperature, top_k, top_p,
                             scratch, out, seed, offset, stream));
        },
        py::arg("logits"), py::arg("dtype"), py::arg("row_stride"), py::arg("batch"),
        py::arg("vocab"), py::arg("pen_ids"), py::arg("n_pen"), py::arg("penalty"),
        py::arg("temperature"), py::arg("top_k"), py::arg("top_p"),
        py::arg("scratch"), py::arg("out"), py::arg("seed"), py::arg("offset"),
        py::arg("stream"),
        "Fused penalty/temperature/top-k/top-p/draw; async on `stream`.");

    m.def(
        "apply_bitmask",
        [](uint64_t logits, int dtype, long long row_stride, int batch, int vocab,
           uint64_t mask, long long mask_row_words, uint64_t stream) {
            check(wls_apply_bitmask(logits, dtype, row_stride, batch, vocab,
                                    mask, mask_row_words, stream));
        },
        py::arg("logits"), py::arg("dtype"), py::arg("row_stride"), py::arg("batch"),
        py::arg("vocab"), py::arg("mask"), py::arg("mask_row_words"), py::arg("stream"),
        "Apply an xgrammar-layout packed token bitmask to logits in place.");

    m.attr("DTYPE_F16") = WLS_DTYPE_F16;
    m.attr("DTYPE_F32") = WLS_DTYPE_F32;
    m.attr("DTYPE_BF16") = WLS_DTYPE_BF16;
}
