// socp_scs_backend.hpp
#pragma once
#include <Eigen/Dense>
#include <optional>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "socp_config.hpp"

// SCS C API（header-only 但不暴露在 public API）
extern "C" {
#include <scs/scs.h>
}

namespace socp {

// ------------------------- Result -------------------------
struct ScsSolveResult {
    bool ok = false;
    bool inaccurate = false;
    std::string status;
    int iter = 0;
    double res_pri = 0, res_dual = 0, res_infeas = 0, res_unbdd = 0;
    Eigen::VectorXd x; // primal
};

// ------------------------- Backend -------------------------
class ScsBackend {
public:
    explicit ScsBackend(SolverConfig cfg) : cfg_(std::move(cfg))  {}

    // minimize c^T x  s.t.
    //   [Aeq] x = [beq]              (z-cone / equality)
    //   [ G ] x <= [ h ]             (l-cone / nonnegative slack)
    //   For each SOC k:
    //     soc_h[k] - soc_G[k] x \in SOC
    //
    // 這裡的參數是「原始形式」(Aeq,beq,G,h,SOC blocks)，backend 會轉成 SCS 需要的
    //  Ax + s = b, s in K（K = z + l + q...）
    ScsSolveResult solve(const Eigen::MatrixXd& Aeq,
                         const Eigen::VectorXd& beq,
                         const Eigen::MatrixXd& G,
                         const Eigen::VectorXd& h,
                         const std::vector<Eigen::MatrixXd>& soc_G_list,
                         const std::vector<Eigen::VectorXd>& soc_h_list,
                         const Eigen::VectorXd& c)
    {
        validate_dims_or_throw(Aeq, beq, G, h, soc_G_list, soc_h_list, c);

        const int n = (int)c.size();

        // --- Build A (vstack) ---
        CSCImpl Aeq_c = CSCImpl::to_csc(Aeq);
        CSCImpl G_c   = CSCImpl::to_csc(G);

        std::vector<CSCImpl> soc_c;
        soc_c.reserve(soc_G_list.size());
        for (const auto& M : soc_G_list) soc_c.push_back(CSCImpl::to_csc(M));

        std::vector<CSCImpl> blocks;
        blocks.reserve(2 + soc_c.size());
        blocks.push_back(Aeq_c);
        blocks.push_back(G_c);
        for (auto& s : soc_c) blocks.push_back(s);

        CSCImpl A_all = CSCImpl::vstack(blocks);

        // --- Build b = [beq; h; soc_h...] ---
        std::vector<Eigen::VectorXd> rhs;
        rhs.reserve(2 + soc_h_list.size());
        rhs.push_back(beq);
        rhs.push_back(h);
        for (auto& v : soc_h_list) rhs.push_back(v);
        std::vector<scs_float> b_all = CSCImpl::vstack_vec(rhs);

        // --- Build c ---
        std::vector<scs_float> c_all((size_t)n);
        for (int i = 0; i < n; ++i) c_all[(size_t)i] = (scs_float)c(i);

        // --- Cone K ---
        ScsCone k; std::memset(&k, 0, sizeof(k));
        k.z = (scs_int)Aeq.rows();
        k.l = (scs_int)G.rows();

        std::vector<scs_int> qdims;
        qdims.reserve(soc_G_list.size());
        for (const auto& M : soc_G_list) qdims.push_back((scs_int)M.rows());
        k.qsize = (scs_int)qdims.size();
        k.q = qdims.empty() ? nullptr : qdims.data();

        // --- Final dimension sanity: cone rows must equal A.m ---
        scs_int m_cone = k.z + k.l;
        for (auto q : qdims) m_cone += q;
        if (m_cone != (scs_int)A_all.m) {
            throw std::runtime_error("[ScsBackend] cone rows != A.m (internal bug)");
        }
        if ((int)b_all.size() != (int)A_all.m) {
            throw std::runtime_error("[ScsBackend] b size != A.m (internal bug)");
        }

        // --- SCS data matrix ---
        ScsMatrix A;
        A.m = (scs_int)A_all.m;
        A.n = (scs_int)A_all.n;
        A.p = A_all.Ap.data();
        A.i = A_all.Ai.data();
        A.x = A_all.Ax.data();

        ScsData d; std::memset(&d, 0, sizeof(d));
        d.m = (scs_int)A_all.m;
        d.n = (scs_int)n;
        d.A = &A;
        d.P = nullptr;
        d.b = b_all.data();
        d.c = c_all.data();

        // --- Settings ---
        ScsSettings stg; scs_set_default_settings(&stg);
        stg.verbose        = cfg_.verbose ? 1 : 0;
        stg.max_iters      = (scs_int)cfg_.max_iters;
        stg.eps_abs        = (scs_float)cfg_.eps_abs;
        stg.eps_rel        = (scs_float)cfg_.eps_rel;
        stg.time_limit_secs= (scs_float)cfg_.time_limit_secs;
        stg.acceleration_lookback = (scs_int)cfg_.accel_lookback;
        stg.normalize      = cfg_.normalize ? 1 : 0;
        stg.scale          = (scs_float)cfg_.scale;

        // --- Solve ---
        ScsSolution sol; std::memset(&sol, 0, sizeof(sol));
        ScsInfo     info; std::memset(&info, 0, sizeof(info));

        scs_int ret = scs(&d, &k, &stg, &sol, &info);

        ScsSolveResult out;
        out.status = info.status ? std::string(info.status) : std::string("");
        out.iter = (int)info.iter;
        out.res_pri = info.res_pri;
        out.res_dual = info.res_dual;
        out.res_infeas = info.res_infeas;
        out.res_unbdd = info.res_unbdd_a;

        const bool solved_ok = (ret == SCS_SOLVED);
        const bool solved_inacc = (ret == SCS_SOLVED_INACCURATE);

        out.ok = (solved_ok || solved_inacc);
        out.inaccurate = solved_inacc;

        if (out.ok && sol.x) {
            out.x = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < n; ++i) out.x(i) = (double)sol.x[i];
        } else {
            out.x = Eigen::VectorXd(); // empty
        }

        CSCImpl::free_solution_compat(sol);
        return out;
    }

private:
    SolverConfig cfg_;

    // ===================== private backend impl =====================
    struct CSCImpl {
        std::vector<scs_int>   Ap;  // col pointers (n+1)
        std::vector<scs_int>   Ai;  // row indices
        std::vector<scs_float> Ax;  // values
        int m = 0, n = 0;

        static CSCImpl to_csc(const Eigen::MatrixXd& M) {
            CSCImpl C;
            C.m = (int)M.rows();
            C.n = (int)M.cols();
            C.Ap.resize((size_t)C.n + 1);
            C.Ap[0] = 0;

            for (int j = 0; j < C.n; ++j) {
                for (int i = 0; i < C.m; ++i) {
                    const double v = M(i, j);
                    if (std::abs(v) > 1e-14) {
                        C.Ai.push_back((scs_int)i);
                        C.Ax.push_back((scs_float)v);
                    }
                }
                C.Ap[(size_t)j + 1] = (scs_int)C.Ai.size();
            }
            return C;
        }

        // stack blocks vertically, same #cols
        static CSCImpl vstack(const std::vector<CSCImpl>& blocks) {
            if (blocks.empty()) return CSCImpl{};
            const int n = blocks.front().n;
            for (auto& B : blocks) {
                if (B.n != n) throw std::runtime_error("[ScsBackend] vstack: col mismatch");
            }

            CSCImpl out;
            out.n = n;
            out.m = 0;
            for (auto& B : blocks) out.m += B.m;

            out.Ap.resize((size_t)out.n + 1);
            out.Ap[0] = 0;

            scs_int nnz = 0;
            for (int j = 0; j < out.n; ++j) {
                scs_int row_off = 0;
                for (auto& B : blocks) {
                    const scs_int p0 = B.Ap[(size_t)j];
                    const scs_int p1 = B.Ap[(size_t)j + 1];
                    for (scs_int k = p0; k < p1; ++k) {
                        out.Ai.push_back(B.Ai[(size_t)k] + row_off);
                        out.Ax.push_back(B.Ax[(size_t)k]);
                        ++nnz;
                    }
                    row_off += (scs_int)B.m;
                }
                out.Ap[(size_t)j + 1] = nnz;
            }
            return out;
        }

        static std::vector<scs_float> vstack_vec(const std::vector<Eigen::VectorXd>& vs) {
            size_t m = 0;
            for (auto& v : vs) m += (size_t)v.size();
            std::vector<scs_float> out(m);

            size_t off = 0;
            for (auto& v : vs) {
                for (int i = 0; i < v.size(); ++i) out[off++] = (scs_float)v(i);
            }
            return out;
        }

        static void free_solution_compat(ScsSolution& sol) {
            // SCS 新舊版本 free 名稱不一樣，我們直接用 std::free 安全處理
            if (sol.x) { std::free(sol.x); sol.x = nullptr; }
            if (sol.y) { std::free(sol.y); sol.y = nullptr; }
            if (sol.s) { std::free(sol.s); sol.s = nullptr; }
        }
    };

    static void validate_dims_or_throw(const Eigen::MatrixXd& Aeq,
                                      const Eigen::VectorXd& beq,
                                      const Eigen::MatrixXd& G,
                                      const Eigen::VectorXd& h,
                                      const std::vector<Eigen::MatrixXd>& soc_G_list,
                                      const std::vector<Eigen::VectorXd>& soc_h_list,
                                      const Eigen::VectorXd& c)
    {
        const int n = (int)c.size();
        if (n <= 0) throw std::runtime_error("[ScsBackend] c.size() must > 0");

        // Aeq x = beq
        if (Aeq.cols() != n) throw std::runtime_error("[ScsBackend] Aeq.cols != n");
        if (Aeq.rows() != beq.size()) throw std::runtime_error("[ScsBackend] Aeq.rows != beq.size");

        // G x <= h
        if (G.cols() != n) throw std::runtime_error("[ScsBackend] G.cols != n");
        if (G.rows() != h.size()) throw std::runtime_error("[ScsBackend] G.rows != h.size");

        // SOC blocks
        if (soc_G_list.size() != soc_h_list.size())
            throw std::runtime_error("[ScsBackend] soc_G_list.size != soc_h_list.size");

        for (size_t k = 0; k < soc_G_list.size(); ++k) {
            const auto& Gi = soc_G_list[k];
            const auto& hi = soc_h_list[k];
            if (Gi.cols() != n) throw std::runtime_error("[ScsBackend] soc_G.cols != n");
            if (Gi.rows() != hi.size()) throw std::runtime_error("[ScsBackend] soc_G.rows != soc_h.size");
            if (Gi.rows() < 2) throw std::runtime_error("[ScsBackend] SOC dim must be >= 2 (t + vec)");
        }
    }
};

} // namespace socp
