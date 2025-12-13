// socp_generator.hpp（header-only）
// 掃參數 + 去重 + 排序 topK
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "socp_types.hpp"
#include "socp_config.hpp"
#include "socp_solver.hpp"
#include "socp_math.hpp"

namespace socp {

class SocpCandidateGenerator {
public:
    explicit SocpCandidateGenerator(SolverConfig cfg = {})
        : cfg_(std::move(cfg))
        , solver_(cfg_) {}

    std::vector<PortfolioCandidate> generate(const SocpProblem& base,
                                             const SocpSweep& sweep)
    {
        // ---- 預估容量：sigma * tau * impact ----
        const size_t cap =
            sweep.sigma_star_list.size() * sweep.tau_list.size() * sweep.impact_scale.size();

        std::vector<PortfolioCandidate> all;
        all.reserve(cap);

        // ---- OMP threads ----
#ifdef _OPENMP
        if (cfg_.omp_threads > 0) omp_set_num_threads(cfg_.omp_threads);
#endif

        // ---- 掃參數（每組一個 solve_once）----
        // 注意：solver_ 是有狀態物件（含 backend_），因此在 OMP 下要為每個 thread 建自己的 solver
#ifdef _OPENMP
        #pragma omp parallel
        {
            SocpSolver local_solver(cfg_);
            std::vector<PortfolioCandidate> local;
            local.reserve(64);

            #pragma omp for collapse(3) nowait
            for (int is = 0; is < (int)sweep.sigma_star_list.size(); ++is)
            for (int it = 0; it < (int)sweep.tau_list.size(); ++it)
            for (int ik = 0; ik < (int)sweep.impact_scale.size(); ++ik)
            {
                SocpProblem pb = base;
                pb.sigma_star  = sweep.sigma_star_list[is];
                pb.tau_max     = sweep.tau_list[it];

                // impact scale：乘在 gamma_init
                pb.gamma_init  = base.gamma_init * sweep.impact_scale[ik];

                if (auto cand = local_solver.solve_once(pb)) {
                    cand->sigma_star    = pb.sigma_star;
                    cand->tau_max       = pb.tau_max;
                    cand->impact_scale  = sweep.impact_scale[ik];
                    local.push_back(*cand);
                }
            }

            #pragma omp critical
            {
                all.insert(all.end(), local.begin(), local.end());
            }
        }
#else
        for (size_t is = 0; is < sweep.sigma_star_list.size(); ++is)
        for (size_t it = 0; it < sweep.tau_list.size(); ++it)
        for (size_t ik = 0; ik < sweep.impact_scale.size(); ++ik)
        {
            SocpProblem pb = base;
            pb.sigma_star  = sweep.sigma_star_list[is];
            pb.tau_max     = sweep.tau_list[it];
            pb.gamma_init  = base.gamma_init * sweep.impact_scale[ik];

            if (auto cand = solver_.solve_once(pb)) {
                cand->sigma_star   = pb.sigma_star;
                cand->tau_max      = pb.tau_max;
                cand->impact_scale = sweep.impact_scale[ik];
                all.push_back(*cand);
            }
        }
#endif

        // ---- 去重（L1 radius）----
        std::vector<PortfolioCandidate> uniq;
        uniq.reserve(all.size());

        for (auto& c : all) {
            bool dup = false;
            for (auto& e : uniq) {
                if (is_dup_l1(e.w, c.w, sweep.dedup_l1_radius)) {
                    dup = true;
                    break;
                }
            }
            if (!dup) uniq.push_back(std::move(c));
        }

        // ---- 排序 + 截斷 topK ----
        sort_and_truncate(uniq, sweep.k_keep);
        return uniq;
    }

private:
    SolverConfig cfg_;
    SocpSolver solver_;

private:
    bool is_dup_l1(const Eigen::VectorXd& a,
                   const Eigen::VectorXd& b,
                   double radius) const
    {
        if (a.size() != b.size()) return false;
        const double d = (a - b).cwiseAbs().sum();
        return d < radius;
    }

    void sort_and_truncate(std::vector<PortfolioCandidate>& v, int k_keep) const
    {
        std::sort(v.begin(), v.end(),
                  [](const PortfolioCandidate& x, const PortfolioCandidate& y){
                      return x.obj > y.obj; // obj 大的在前
                  });
        if (k_keep > 0 && (int)v.size() > k_keep) v.resize((size_t)k_keep);
    }
};

} // namespace socp
