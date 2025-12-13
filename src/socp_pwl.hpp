// socp_pwl.hpp
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace socp {

// f(u) = u^{3/2} 的 PWL 上界：
// 對每個區間 [b_j, b_{j+1}] 用 chord（弦）
//   f(u) <= a_j * u + c_j
class PwlUpperEnvelope {
public:
    PwlUpperEnvelope() = default;

    // bpts: 單調遞增，且 >= 0
    // 例：{0.0, 0.002, 0.005, 0.01, 0.02, 0.04}
    inline void build(const std::vector<double>& bpts)
    {
        a_.clear();
        c_.clear();
        if (bpts.size() < 2) return;

        // 確保非負 & 單調
        for (size_t i = 1; i < bpts.size(); ++i) {
            if (bpts[i] < bpts[i-1]) {
                // 不 throw，避免 header-only 帶來例外 ABI 問題
                std::cerr << "[PWL] bpts not monotone increasing\n";
                return;
            }
        }

        for (size_t j = 0; j + 1 < bpts.size(); ++j) {
            const double x1 = std::max(0.0, bpts[j]);
            const double x2 = std::max(0.0, bpts[j+1]);

            const double y1 = std::pow(x1, 1.5);
            const double y2 = std::pow(x2, 1.5);

            const double dx = std::max(1e-12, x2 - x1);
            const double slope = (y2 - y1) / dx;
            const double intercept = y1 - slope * x1;

            a_.push_back(slope);
            c_.push_back(intercept);
        }
    }

    inline int segments() const {
        return static_cast<int>(a_.size());
    }

    inline const std::vector<double>& a() const { return a_; } // slope
    inline const std::vector<double>& c() const { return c_; } // intercept

private:
    std::vector<double> a_; // slopes
    std::vector<double> c_; // intercepts
};

} // namespace socp
