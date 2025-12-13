// socp_solver.hpp  (header-only)
#pragma once
#include <optional>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <Eigen/Dense>

#include "socp_types.hpp"
#include "socp_config.hpp"
#include "socp_debug.hpp"
#include "socp_pwl.hpp"
#include "socp_math.hpp"
#include "socp_scs_backend.hpp"

namespace socp {

class SocpSolver {
public:
    explicit SocpSolver(SolverConfig cfg = {})
        : cfg_(std::move(cfg))
        , dbg_(cfg_.debug_on)
        , backend_(cfg_) {}

    // solve once
    std::optional<PortfolioCandidate> solve_once(const SocpProblem& pb_in) {
        // 0) debug relax (feasibility probe)
        SocpProblem pb = relaxed_copy_if_enabled(pb_in);

        // 1) fill defaults + sanity
        fill_defaults(pb);

        const int N = (int)pb.rhat_enh.size();
        if (N <= 0) return std::nullopt;
        if (pb.L.rows() != N || pb.L.cols() != N) return std::nullopt;
        if (pb.w0.size() != N) return std::nullopt;

        // 2) compute components
        const Eigen::VectorXd wmax = compute_wmax(pb);
        const Eigen::VectorXd K    = compute_impact_K(pb);

        // quick debug prints/sanity
        dbg_.check_inputs(pb.rhat_enh, pb.L, pb.w0);
        dbg_.quick_feas_sanity(wmax, pb.group_cap, pb.group, pb.tau_max, pb.W_turn, pb.budget_long_only);

        // 3) build blocks
        Eigen::MatrixXd Aeq; Eigen::VectorXd beq;
        build_eq(pb, Aeq, beq);

        PwlUpperEnvelope pwl;
        pwl.build(pb.pwl.b);

        Eigen::MatrixXd G; Eigen::VectorXd h;
        build_ineq_linear(pb, wmax, pwl, G, h);

        std::vector<Eigen::MatrixXd> socG;
        std::vector<Eigen::VectorXd> soch;
        build_soc_blocks(pb, socG, soch);

        // 4) build objective c
        // var ordering: [w (N), u_buy (N), u_sell (N), t (N)] => total 4N
        const int nW = N, nB = N, nS = N, nT = N;
        const int nvar = nW + nB + nS + nT;

        Eigen::VectorXd c = Eigen::VectorXd::Zero(nvar);
        // maximize rhat^T w  => minimize -rhat^T w
        c.segment(0, N) = -pb.rhat_enh;

        // linear fees: P0 * fee_buy * u_buy + P0 * fee_sell * u_sell
        c.segment(nW, N).array()      += pb.P0 * pb.fee_buy;
        c.segment(nW + nB, N).array() += pb.P0 * pb.fee_sell;

        // impact term: sum_i K_i * t_i  (t is variable segment)
        c.segment(nW + nB + nS, N) = K;

        // 5) solve with backend
        auto res = backend_.solve(Aeq, beq, G, h, socG, soch, c);
        if (!res.ok) return std::nullopt;

        // 6) pack solution
        if (res.x.size() != nvar) return std::nullopt;
        PortfolioCandidate out = pack_solution(pb, res.x, K);
        return out;
    }

private:
    SolverConfig cfg_;
    DebugReporter dbg_;
    ScsBackend backend_;

private:
    // ---------- helpers ----------
    static inline double clamp01(double x){ return std::clamp(x, 0.0, 1.0); }

    SocpProblem relaxed_copy_if_enabled(const SocpProblem& pb) const {
        if (!cfg_.debug_relax) return pb;

        // feasibility probe: make it easy to be feasible
        SocpProblem pr = pb;
        const int N = (int)pb.rhat_enh.size();

        pr.sigma_star = 1e6;
        pr.tau_max    = 10.0;
        pr.turn_norm  = TurnoverNorm::L1;
        pr.group_cap.clear();

        // hard cap set to 100% each name
        pr.wmax_base = Math::constant(1.0, N);

        // trivial PWL
        pr.pwl.b = {0.0, 0.05};

        // no impact
        pr.gamma_init = 0.0;

        return pr;
    }

    void fill_defaults(SocpProblem& pb) const {
        const int N = (int)pb.rhat_enh.size();

        // defaults for pwl
        if (pb.pwl.b.empty()) {
            pb.pwl.b = {0.0, 0.002, 0.005, 0.010, 0.020, 0.040};
        }

        // wmax_base default
        if (pb.wmax_base.size() != N) {
            pb.wmax_base = Math::constant(1.0, N);
        }

        // W_turn default
        if (pb.W_turn.size() != N) {
            pb.W_turn = Math::constant(1.0, N);
        }

        // group default
        if (pb.group.size() != N) {
            pb.group = Eigen::VectorXi::Constant(N, -1);
        }

        // market vectors default safety (avoid size mismatch crash)
        auto ensureN = [&](Eigen::VectorXd& v, double fill){
            if (v.size() != N) v = Math::constant(fill, N);
        };
        ensureN(pb.ADV,      1.0);
        ensureN(pb.sigma_GK, 0.0);
        ensureN(pb.Imb,      0.0);
        ensureN(pb.BF,       0.0);
        ensureN(pb.BIAS,     0.0);
        ensureN(pb.TS,       1.0);

        // numeric
        pb.eps = std::max(pb.eps, 1e-12);

        // budget sanity
        if (!(pb.budget_long_only > 0.0)) pb.budget_long_only = 1.0;

        // sigma/tau sanity
        pb.sigma_star = std::max(0.0, pb.sigma_star);
        pb.tau_max    = std::max(0.0, pb.tau_max);
    }

    Eigen::VectorXd compute_wmax(const SocpProblem& pb) const {
        const int N = (int)pb.rhat_enh.size();
        Eigen::VectorXd wmax = pb.wmax_base;

        // BF relax and BIAS shrink
        for (int i = 0; i < N; ++i) {
            const double relax  = 1.0 + pb.a_b * pb.BF(i);
            const double shrink = 1.0 - pb.a_bias * std::max(0.0, std::abs(pb.BIAS(i)) - pb.theta_bias);
            wmax(i) = std::max(0.0, wmax(i) * relax * shrink);
        }

        // ADV cap: w_i <= kappa_ADV * ADV_i / P0
        if (pb.kappa_ADV > 0.0) {
            for (int i = 0; i < N; ++i) {
                const double cap = pb.kappa_ADV * Math::safe_div(pb.ADV(i), std::max(1e-12, pb.P0));
                wmax(i) = std::min(wmax(i), std::max(0.0, cap));
            }
        }

        // TS cap: w_i <= kappa_TS * TS_i / median(TS)
        if (pb.kappa_TS > 0.0) {
            std::vector<double> ts(N);
            for (int i = 0; i < N; ++i) ts[i] = pb.TS(i);
            std::nth_element(ts.begin(), ts.begin() + N/2, ts.end());
            const double medTS = std::max(1e-12, ts[N/2]);
            for (int i = 0; i < N; ++i) {
                const double cap = pb.kappa_TS * Math::safe_div(pb.TS(i), medTS);
                wmax(i) = std::min(wmax(i), std::max(0.0, cap));
            }
        }

        // last safety: clamp [0,1] (long-only typical)
        for (int i = 0; i < N; ++i) wmax(i) = std::clamp(wmax(i), 0.0, 1.0);

        return wmax;
    }

    Eigen::VectorXd compute_impact_K(const SocpProblem& pb) const {
        // objective uses: sum_i K_i * t_i
        // K_i = gamma_init * sqrt(P0^3 / ADV_i) * sigma_GK(i) * (1 + beta_imb*|Imb|)
        const int N = (int)pb.rhat_enh.size();
        Eigen::VectorXd K = Eigen::VectorXd::Zero(N);

        for (int i = 0; i < N; ++i) {
            const double adv = std::max(1e-12, pb.ADV(i));
            const double g   = pb.sigma_GK(i) * (1.0 + pb.beta_imb_impact * std::abs(pb.Imb(i)));
            const double scale = std::sqrt((pb.P0*pb.P0*pb.P0) / adv);
            K(i) = pb.gamma_init * scale * g;
            if (!std::isfinite(K(i))) K(i) = 0.0;
        }
        return K;
    }

    void build_eq(const SocpProblem& pb,
                  Eigen::MatrixXd& Aeq, Eigen::VectorXd& beq) const
    {
        const int N = (int)pb.rhat_enh.size();
        const int nvar = 4*N;

        // rows: (1) budget sum(w)=budget, (2) N rows: w - u_buy + u_sell = w0
        Aeq = Eigen::MatrixXd::Zero(1 + N, nvar);
        beq = Eigen::VectorXd::Zero(1 + N);

        // sum w = budget
        Aeq.block(0, 0, 1, N).setOnes();
        beq(0) = pb.budget_long_only;

        // w - u_buy + u_sell = w0
        Aeq.block(1, 0,   N, N).setIdentity();          // +w
        Aeq.block(1, N,   N, N).diagonal().array() -= 1.0; // -u_buy
        Aeq.block(1, 2*N, N, N).diagonal().array() += 1.0; // +u_sell
        beq.segment(1, N) = pb.w0;
    }

    void build_ineq_linear(const SocpProblem& pb,
                           const Eigen::VectorXd& wmax,
                           const PwlUpperEnvelope& pwl,
                           Eigen::MatrixXd& G, Eigen::VectorXd& h) const
    {
        const int N = (int)pb.rhat_enh.size();
        const int nvar = 4*N;

        std::vector<Eigen::RowVectorXd> rows;
        std::vector<double> rhs;
        rows.reserve(10*N);

        auto add_le = [&](const Eigen::RowVectorXd& g, double hv){
            rows.push_back(g);
            rhs.push_back(hv);
        };

        // nonneg: w >= 0, u_buy >= 0, u_sell >= 0  ==>  -var <= 0
        for (int i = 0; i < N; ++i) {
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            r(i) = -1.0; add_le(r, 0.0);
        }
        for (int i = 0; i < N; ++i) {
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            r(N + i) = -1.0; add_le(r, 0.0);
        }
        for (int i = 0; i < N; ++i) {
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            r(2*N + i) = -1.0; add_le(r, 0.0);
        }

        // w <= wmax
        for (int i = 0; i < N; ++i) {
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            r(i) = 1.0; add_le(r, wmax(i));
        }

        // group caps: sum_{i in g} w_i <= cap
        for (const auto& kv : pb.group_cap) {
            const int gId = kv.first;
            const double cap = kv.second;
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            for (int i = 0; i < N; ++i) if (pb.group(i) == gId) r(i) = 1.0;
            add_le(r, cap);
        }

        // turnover L1: sum_i W_turn(i) (u_buy+u_sell) <= tau_max
        if (pb.turn_norm == TurnoverNorm::L1) {
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            for (int i = 0; i < N; ++i) {
                r(N + i)     = pb.W_turn(i);
                r(2*N + i)   = pb.W_turn(i);
            }
            add_le(r, pb.tau_max);
        }

        // PWL impact constraints:
        // t_i >= a_j*(u_buy_i + u_sell_i) + c_j
        // => a_j*u_buy + a_j*u_sell - t_i <= -c_j
        const auto& a = pwl.a();
        const auto& c = pwl.c();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < (int)a.size(); ++j) {
                Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
                r(N + i)   =  a[j];
                r(2*N + i) =  a[j];
                r(3*N + i) = -1.0;      // -t
                add_le(r, -c[j]);
            }
        }

        // finalize dense G,h
        G = Eigen::MatrixXd::Zero((int)rows.size(), nvar);
        h = Eigen::VectorXd::Zero((int)rhs.size());
        for (int k = 0; k < (int)rows.size(); ++k) {
            G.row(k) = rows[k];
            h(k) = rhs[k];
        }
    }

    void build_soc_blocks(const SocpProblem& pb,
                          std::vector<Eigen::MatrixXd>& socG,
                          std::vector<Eigen::VectorXd>& soch) const
    {
        socG.clear(); soch.clear();

        const int N = (int)pb.rhat_enh.size();
        const int nvar = 4*N;

        // (1) risk SOC: || L^T w ||_2 <= m * sigma_star
        // SCS style expects: [h0; h1..] - [G0; G1..] x in SOC
        // We build as: (h - Gx) in SOC, with:
        // h0 = m*sigma_star, h1..=0
        // G block: rows(1..N, w) = -L^T  => h - Gx => [m*sigma;  -(-L^T)w ] = [m*sigma; L^T w]
        Eigen::MatrixXd Gq1 = Eigen::MatrixXd::Zero(N + 1, nvar);
        Eigen::VectorXd hq1 = Eigen::VectorXd::Zero(N + 1);
        hq1(0) = std::max(0.0, pb.m * pb.sigma_star);
        Gq1.block(1, 0, N, N) = -pb.L.transpose();

        socG.push_back(Gq1);
        soch.push_back(hq1);

        // (2) turnover L2 SOC (optional):
        // || diag(W_turn) (u_buy + u_sell) ||_2 <= tau_max
        if (pb.turn_norm == TurnoverNorm::L2) {
            Eigen::MatrixXd Gq2 = Eigen::MatrixXd::Zero(N + 1, nvar);
            Eigen::VectorXd hq2 = Eigen::VectorXd::Zero(N + 1);
            hq2(0) = std::max(0.0, pb.tau_max);

            for (int i = 0; i < N; ++i) {
                Gq2(1 + i, N + i)   = -pb.W_turn(i);
                Gq2(1 + i, 2*N + i) = -pb.W_turn(i);
            }
            socG.push_back(Gq2);
            soch.push_back(hq2);
        }
    }

    PortfolioCandidate pack_solution(const SocpProblem& pb,
                                     const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& K) const
    {
        const int N = (int)pb.rhat_enh.size();
        const int nvar = 4*N;
        (void)nvar;

        const int nW = N, nB = N, nS = N, nT = N;

        PortfolioCandidate out;
        out.w = x.segment(0, N);

        const Eigen::VectorXd ub = x.segment(nW, N);
        const Eigen::VectorXd us = x.segment(nW + nB, N);
        const Eigen::VectorXd tt = x.segment(nW + nB + nS, N);
        const Eigen::VectorXd uabs = ub + us;

        out.ret_part = pb.rhat_enh.dot(out.w);

        // linear costs (ratio style: TWD/P0 if you want, but we keep absolute then normalize in obj if needed)
        out.lin_cost = pb.P0 * (pb.fee_buy * ub.sum() + pb.fee_sell * us.sum());

        // impact cost: sum_i K_i * t_i
        out.imp_cost = (K.array() * tt.array()).sum();

        // objective consistent with original: ret - costs
        out.obj = out.ret_part - out.lin_cost - out.imp_cost;

        // risk
        out.risk = (pb.L.transpose() * out.w).norm();

        // turnover
        if (pb.turn_norm == TurnoverNorm::L1) {
            out.turnover = uabs.sum();
        } else {
            out.turnover = (pb.W_turn.array() * uabs.array()).matrix().norm();
        }

        out.sigma_star = pb.sigma_star;
        out.tau_max    = pb.tau_max;
        out.impact_scale = 1.0; // sweep 才會填（base 版未知）

        dbg_.log_obj(out.ret_part, out.lin_cost, out.imp_cost, out.obj);
        return out;
    }
};

} // namespace socp
