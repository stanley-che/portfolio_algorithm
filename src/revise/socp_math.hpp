// socp_math.hpp
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>

namespace socp {

struct Math {
    static inline double safe_div(double a, double b) noexcept {
        return (std::abs(b) < 1e-12) ? 0.0 : (a / b);
    }

    static inline Eigen::VectorXd constant(double v, int n){
        return Eigen::VectorXd::Constant(n, v);
    }

    static inline double l1_norm(const Eigen::Ref<const Eigen::VectorXd>& x) noexcept {
        return x.cwiseAbs().sum();
    }

    static inline double l2_norm(const Eigen::Ref<const Eigen::VectorXd>& x) noexcept {
        return std::sqrt(x.squaredNorm());
    }
};

} // namespace socp
