// socp_debug.hpp
#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace socp {

class DebugReporter {
public:
    explicit DebugReporter(bool on=true): on_(on) {}

    inline void check_inputs(const Eigen::VectorXd& rhat,
                             const Eigen::MatrixXd& L,
                             const Eigen::VectorXd& w0) const
    {
        if (!on_) return;

        auto chk = [&](const char* n, bool ok){
            std::cerr << (ok ? "[OK]  " : "[BAD] ") << n << "\n";
        };

        chk("rhat finite", !has_nan(rhat));
        chk("L finite",    !has_nan(L));
        chk("w0 finite",   !has_nan(w0));

        chk("L is square", (L.rows() == L.cols()));
        chk("dims match: rhat.size == w0.size", (rhat.size() == w0.size()));
        chk("dims match: L.cols == rhat.size", (L.cols() == rhat.size()));

        print_head("rhat", rhat, 5);
        print_head("w0",   w0,   5);

        if (L.size() > 0) {
            Eigen::VectorXd d = L.diagonal();
            print_head("L.diag", d, 5);
        }
    }

    inline void quick_feas_sanity(const Eigen::VectorXd& wmax,
                                  const std::map<int,double>& group_cap,
                                  const Eigen::VectorXi& group,
                                  double tau_max,
                                  const Eigen::VectorXd& W_turn,
                                  double budget) const
    {
        if (!on_) return;

        const double sum_wmax = wmax.sum();
        std::cerr << "[SANITY] sum(wmax)=" << sum_wmax
                  << " (should >= budget=" << budget << ")\n";

        // group cap sanity：若同群 wmax 總和 < cap，則 cap 不會造成 infeasible；
        // 但若 cap < 可達到的最小需求，可能 infeasible（此處僅列出 wmax 上界資訊）
        if (!group_cap.empty() && group.size() == wmax.size()) {
            std::map<int, double> sumBy;
            std::map<int, int> cntBy;

            for (int i = 0; i < group.size(); ++i) {
                int g = group(i);
                if (g >= 0) {
                    sumBy[g] += wmax(i);
                    cntBy[g] += 1;
                }
            }

            for (auto &kv : group_cap) {
                int g = kv.first;
                double cap = kv.second;
                double sumW = sumBy[g];
                std::cerr << "[SANITY] group " << g
                          << " cap=" << cap
                          << " sum(wmax in g)=" << sumW
                          << " (#" << cntBy[g] << ")\n";
            }
        }

        // turnover 粗略上界：tau_max / max(W_turn)
        double maxW = 1.0;
        if (W_turn.size() == wmax.size() && W_turn.size() > 0) {
            maxW = W_turn.maxCoeff();
            if (!std::isfinite(maxW) || maxW <= 0) maxW = 1.0;
        }
        const double rough = tau_max / std::max(1e-12, maxW);

        std::cerr << "[SANITY] tau_max=" << tau_max
                  << ", max(W_turn)=" << maxW
                  << ", per-name Δw upper bound (rough)=" << rough
                  << "\n";
    }

    inline void log_obj(double ret, double fee, double impact, double obj) const
    {
        if (!on_) return;
        std::cerr << "[OBJ] ret=" << ret
                  << " fee=" << fee
                  << " impact=" << impact
                  << " obj=" << obj
                  << "\n";
    }

private:
    bool on_ = true;

    static inline bool has_nan(const Eigen::VectorXd& v){
        return !(v.array().isFinite().all());
    }
    static inline bool has_nan(const Eigen::MatrixXd& M){
        return !(M.array().isFinite().all());
    }

    static inline void print_head(const char* name, const Eigen::VectorXd& v, int k){
        std::cerr << name << " [N=" << v.size() << "] head:";
        int n = std::min<int>(k, v.size());
        for (int i = 0; i < n; ++i) std::cerr << " " << v(i);
        std::cerr << "\n";
    }
};

} // namespace socp
