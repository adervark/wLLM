// winllm_suffix.cpp — native SuffixCache for wLLM's SuffixDecoding.
//
// Mirrors winllm/inference/suffix_cache.py exactly; see that module for the
// algorithm rationale (positions-by-bigram index with backward extension,
// the paper's alpha*p speculation-length rule). The Python class remains the
// fallback and the reference implementation for the parity tests in
// tests/test_suffix_decoding.py — any behavior change must land in BOTH.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Token = std::int64_t;

struct PairHash {
    std::size_t operator()(const std::pair<Token, Token>& p) const noexcept {
        auto h = static_cast<std::uint64_t>(p.first) * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<std::uint64_t>(p.second) + 0x9E3779B97F4A7C15ULL +
             (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

// Token history and bigram index for one request.
struct RequestHistory {
    std::vector<Token> tokens;
    // (token_a, token_b) -> positions of token_a where that bigram starts,
    // ascending (positions are only ever appended).
    std::unordered_map<std::pair<Token, Token>, std::vector<std::size_t>, PairHash>
        bigrams;
};

class SuffixCache {
public:
    SuffixCache(double alpha, long long max_spec, long long max_back_ext,
                std::size_t max_requests, std::size_t max_occurrences)
        : alpha_(alpha),
          max_spec_(max_spec),
          max_back_ext_(max_back_ext),
          max_requests_(max_requests),
          max_occurrences_(max_occurrences) {}

    void sync(const std::string& request_id, const py::sequence& tokens) {
        RequestHistory& h = touch(request_id);
        const auto n = static_cast<std::size_t>(py::len(tokens));
        const std::size_t have = h.tokens.size();
        if (n <= have) return;

        // Only the unseen tail crosses the Python/C++ boundary: converting
        // the full list every step would make sync O(n^2) over a generation.
        h.tokens.reserve(n);
        for (std::size_t i = have; i < n; ++i)
            h.tokens.push_back(tokens[i].cast<Token>());

        // Index every bigram that ends inside the new region
        for (std::size_t pos = (have > 0 ? have - 1 : 0); pos + 1 < n; ++pos)
            h.bigrams[{h.tokens[pos], h.tokens[pos + 1]}].push_back(pos);
    }

    void evict(const std::string& request_id) {
        auto it = index_.find(request_id);
        if (it == index_.end()) return;
        order_.erase(it->second);
        index_.erase(it);
    }

    bool has_request(const std::string& request_id) const {
        return index_.count(request_id) != 0;
    }

    std::vector<Token> propose(const std::string& request_id) const {
        auto it = index_.find(request_id);
        if (it == index_.end()) return {};
        const RequestHistory& h = it->second->second;
        const std::vector<Token>& tokens = h.tokens;
        const std::size_t end = tokens.size();
        if (end < 3) return {};

        auto big = h.bigrams.find({tokens[end - 2], tokens[end - 1]});
        if (big == h.bigrams.end()) return {};

        // Exclude the trailing bigram itself (starts at end-2) and keep only
        // the most recent max_occurrences positions; ascending order makes
        // both a suffix selection.
        const std::vector<std::size_t>& positions = big->second;
        std::size_t hi = positions.size();
        while (hi > 0 && positions[hi - 1] >= end - 2) --hi;
        if (hi == 0) return {};
        const std::size_t lo = hi > max_occurrences_ ? hi - max_occurrences_ : 0;

        // Longest backward extension wins; most recent occurrence breaks ties
        long long best_pos = -1, best_ext = -1;
        const long long suffix_prev = static_cast<long long>(end) - 3;
        for (std::size_t k = lo; k < hi; ++k) {
            const auto pos = static_cast<long long>(positions[k]);
            long long ext = 0;
            while (ext < max_back_ext_ && pos - 1 - ext >= 0 &&
                   suffix_prev - ext >= 0 &&
                   tokens[static_cast<std::size_t>(pos - 1 - ext)] ==
                       tokens[static_cast<std::size_t>(suffix_prev - ext)])
                ++ext;
            if (ext >= best_ext) {  // >= so later (more recent) positions win ties
                best_pos = pos;
                best_ext = ext;
            }
        }

        const long long pattern_len = 2 + best_ext;
        long long budget =
            static_cast<long long>(alpha_ * static_cast<double>(pattern_len));
        budget = std::min(budget, max_spec_);
        if (budget <= 0) return {};

        const auto start = static_cast<std::size_t>(best_pos) + 2;
        const std::size_t stop =
            std::min(start + static_cast<std::size_t>(budget), end);
        if (start >= stop) return {};
        return std::vector<Token>(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                                  tokens.begin() + static_cast<std::ptrdiff_t>(stop));
    }

private:
    // Get-or-create with LRU semantics matching the Python OrderedDict:
    // touch moves to the recent end; overflow evicts the oldest entry.
    RequestHistory& touch(const std::string& request_id) {
        auto it = index_.find(request_id);
        if (it != index_.end()) {
            order_.splice(order_.end(), order_, it->second);
            return it->second->second;
        }
        order_.emplace_back(request_id, RequestHistory{});
        index_[request_id] = std::prev(order_.end());
        if (order_.size() > max_requests_) {
            index_.erase(order_.front().first);
            order_.pop_front();
        }
        return order_.back().second;
    }

    double alpha_;
    long long max_spec_;
    long long max_back_ext_;
    std::size_t max_requests_;
    std::size_t max_occurrences_;

    std::list<std::pair<std::string, RequestHistory>> order_;
    std::unordered_map<std::string,
                       std::list<std::pair<std::string, RequestHistory>>::iterator>
        index_;
};

}  // namespace

PYBIND11_MODULE(winllm_suffix, m) {
    m.doc() = "Native (C++) suffix cache for wLLM SuffixDecoding";

    py::class_<SuffixCache>(m, "SuffixCache")
        .def(py::init<double, long long, long long, std::size_t, std::size_t>(),
             py::arg("alpha") = 2.0, py::arg("max_spec") = 16,
             py::arg("max_back_ext") = 64, py::arg("max_requests") = 64,
             py::arg("max_occurrences") = 32)
        .def("sync", &SuffixCache::sync, py::arg("request_id"), py::arg("tokens"))
        .def("propose", &SuffixCache::propose, py::arg("request_id"))
        .def("evict", &SuffixCache::evict, py::arg("request_id"))
        .def("has_request", &SuffixCache::has_request, py::arg("request_id"));
}
