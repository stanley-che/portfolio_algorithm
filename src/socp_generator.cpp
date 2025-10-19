// socp_generator.cpp  —— ECOS -> SCS 版本
#include "socp_generator.h"

extern "C" {
#include <scs/scs.h>   // SCS C API
}
#include <omp.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
// ---- 兼容 v2 / v3 的 free 名稱 ----
#if defined(SCS_VERSION_MAJOR)
  #if SCS_VERSION_MAJOR >= 3
    #define SCS_FREE_SOLUTION scs_free_solution
  #else
    #define SCS_FREE_SOLUTION scs_free_sol
  #endif
#else
  // 沒定義就假設 v3
  #define SCS_FREE_SOLUTION scs_free_solution
#endif
// ===== Debug switches =====
#ifndef SCS_DEBUG_ON
#define SCS_DEBUG_ON 1   // 改成 0 可關閉所有 debug 輸出
#endif
#ifndef SCS_DEBUG_RELAX
#define SCS_DEBUG_RELAX 0 // 設 1 做「可行性探測」(極度放寬約束)
#endif

static inline void scs_free_solution_compat(ScsSolution* s) {
    if (!s) return;
    free(s->x); s->x = nullptr;
    free(s->y); s->y = nullptr;
    free(s->s); s->s = nullptr;
}
// -----------------------------------
static bool has_nan(const Eigen::VectorXd& v){ return !(v.array().isFinite().all()); }
static bool has_nan(const Eigen::MatrixXd& M){ return !(M.array().isFinite().all()); }

static void print_vec(const char* name, const Eigen::VectorXd& v, int k=5){
#if SCS_DEBUG_ON
    std::cerr << name << " [N="<<v.size()<<"] head:";
    int n = std::min<int>(k, v.size());
    for(int i=0;i<n;++i) std::cerr<<" "<<v(i);
    std::cerr<<"\n";
#endif
}

static void check_inputs(const SocpProblem& pb){
#if SCS_DEBUG_ON
    auto chk=[&](const char* n, bool b){ if(b) std::cerr<<"[OK] "<<n<<"\n"; else std::cerr<<"[BAD] "<<n<<"\n"; };
    chk("rhat_enh finite", !has_nan(pb.rhat_enh));
    chk("L finite",        !has_nan(pb.L));
    chk("w0 finite",       !has_nan(pb.w0));
    chk("ADV finite",      !has_nan(pb.ADV));
    chk("sigma_GK finite", !has_nan(pb.sigma_GK));
    chk("Imb finite",      !has_nan(pb.Imb));
    chk("BF finite",       !has_nan(pb.BF));
    chk("BIAS finite",     !has_nan(pb.BIAS));
    chk("TS finite",       !has_nan(pb.TS));
#endif
}

static void quick_feas_sanity(const Eigen::VectorXd& wmax,
                              const std::map<int,double>& group_cap,
                              const Eigen::VectorXi& group,
                              double tau_max, const Eigen::VectorXd& W_turn,
                              double budget){
#if SCS_DEBUG_ON
    double wmax_sum = wmax.sum();
    std::cerr << "[SANITY] sum(wmax)="<< wmax_sum <<" (should >= "<<budget<<")\n";
    // group caps
    if(!group_cap.empty()){
        std::map<int,double> sumBy;
        std::map<int,int> cntBy;
        for(int i=0;i<group.size();++i) if(group(i)>=0){
            sumBy[group(i)] += wmax(i);
            cntBy[group(i)]++;
        }
        for(auto &kv: group_cap){
            int g = kv.first; double cap=kv.second;
            double sumW = sumBy[g];
            std::cerr<<"[SANITY] group "<<g<<" cap="<<cap<<" sum(wmax in g)="<<sumW<<" (#"<<cntBy[g]<<")\n";
        }
    }
    // turnover loose lower bound: if W_turn >= 0, any single name move bounded by tau_max / max(W_turn)
    double maxW = (W_turn.size()? W_turn.maxCoeff():1.0);
    std::cerr << "[SANITY] tau_max="<<tau_max<<", max(W_turn)="<<maxW
              <<", per-name Δw upper bound (rough)="<<(tau_max/std::max(1e-12,maxW))<<"\n";
#endif
}

// ---------------- 小工具 ----------------
static inline double sqr(double x){ return x*x; }
static inline double safe_div(double a,double b){ return (std::abs(b)<1e-12)?0.0:a/b; }
static Eigen::VectorXd vee(double v, int n){ return Eigen::VectorXd::Constant(n, v); }
static double l1_norm(const Eigen::VectorXd& x){ return x.cwiseAbs().sum(); }
static double l2_norm(const Eigen::VectorXd& x){ return std::sqrt(x.dot(x)); }

// 以 chord(弦) 建 PWL 上界：對 f(u)=u^{3/2}，b[k] 單調遞增，回傳 a[k], c[k] 使 t >= a u + c
static void build_pwl_upper(const std::vector<double>& b,
                            std::vector<double>& a, std::vector<double>& c)
{
    int M=(int)b.size();
    a.clear(); c.clear();
    for(int j=0;j<M-1;++j){
        double x1=b[j], x2=b[j+1];
        double y1=std::pow(std::max(0.0,x1),1.5);
        double y2=std::pow(std::max(0.0,x2),1.5);
        double slope = (y2 - y1) / std::max(1e-12, (x2 - x1));
        double intercept = y1 - slope * x1;
        a.push_back(slope); c.push_back(intercept);
    }

}

// 轉出 CSC（一般 double / int 版本；SCS 會在 setup 時取這些指標）
struct CSC {
    std::vector<scs_int>   Ap, Ai;   // column pointers, row indices
    std::vector<scs_float> Ax;       // values
    scs_int m = 0, n = 0;
};

static CSC to_csc_scs(const Eigen::MatrixXd& M){
    CSC C; C.m = (scs_int)M.rows(); C.n = (scs_int)M.cols();
    C.Ap.resize(C.n + 1); C.Ap[0] = 0;
    for (scs_int j = 0; j < C.n; ++j){
        for (scs_int i = 0; i < C.m; ++i){
            double v = M((int)i, (int)j);
            if (std::abs(v) > 1e-14){
                C.Ai.push_back(i);
                C.Ax.push_back((scs_float)v);
            }
        }
        C.Ap[j+1] = (scs_int)C.Ai.size();
    }
    return C;
}

// vstack blocks that have the same number of columns.
// Row indices are shifted by the running row offset.
static CSC vstack_csc(const std::vector<CSC>& blocks){
    CSC out;
    if (blocks.empty()) return out;
    out.n = blocks.front().n;
    out.m = 0;
    for (auto& B : blocks){ out.m += B.m; }

    out.Ap.resize(out.n + 1);
    out.Ap[0] = 0;
    scs_int nnz = 0;

    // For each column, append all blocks' entries with row offsets
    for (scs_int j = 0; j < out.n; ++j){
        scs_int row_off = 0;
        for (auto& B : blocks){
            for (scs_int k = B.Ap[j]; k < B.Ap[j+1]; ++k){
                out.Ai.push_back(B.Ai[k] + row_off);
                out.Ax.push_back(B.Ax[k]);
                ++nnz;
            }
            row_off += B.m;
        }
        out.Ap[j+1] = nnz;
    }
    return out;
}

// helper to stack RHS vectors: [beq; h; hq1; hq2]
static std::vector<scs_float> vstack_vec(
    const std::vector<Eigen::VectorXd>& vs)
{
    size_t m = 0; for (auto& v : vs) m += (size_t)v.size();
    std::vector<scs_float> out(m);
    size_t off = 0;
    for (auto& v : vs){
        for (int i = 0; i < v.size(); ++i) out[off++] = (scs_float)v(i);
    }
    return out;
}

// ---------------- 求解一次（用 SCS） ----------------
std::optional<PortfolioCandidate> solve_socp_once(const SocpProblem& pb){
    // ---- 調試輸出 / 可行性探測 ----
#if SCS_DEBUG_ON
    std::cerr << "[SCS] solve_socp_once: N="<<pb.rhat_enh.size()
              << " sigma*="<<pb.sigma_star<<" tau="<<pb.tau_max
              << " gamma="<<pb.gamma_init<<"\n";
    check_inputs(pb);
#endif

#if SCS_DEBUG_RELAX
    SocpProblem pr = pb;                       // relaxed copy
    pr.sigma_star = 1e6;
    pr.tau_max    = 10.0;
    pr.turn_norm  = TurnoverNorm::L1;
    pr.group_cap.clear();
    pr.wmax_base  = vee(1.0, (int)pb.rhat_enh.size());
    pr.pwl.b      = {0.0, 0.05};
    pr.gamma_init = 0.0;
    const SocpProblem& P = pr;
#else
    const SocpProblem& P = pb;
#endif

    // ---- 基本尺寸與切片 ----
    const int N = (int)P.rhat_enh.size();
    const int nW=N, nB=N, nS=N, nT=N, nvar=nW+nB+nS+nT;
    const Eigen::VectorXd W_turn = (P.W_turn.size()==N) ? P.W_turn : Eigen::VectorXd::Ones(N);

    // ---- 目標 ----
    Eigen::VectorXd c = Eigen::VectorXd::Zero(nvar);
    c.segment(0, N)            = -P.rhat_enh;
    c.segment(nW, N).array()   += P.P0 * P.fee_buy;
    c.segment(nW+nB, N).array()+= P.P0 * P.fee_sell;

    // impact 係數
    Eigen::VectorXd K = Eigen::VectorXd::Zero(N);
    for (int i=0;i<N;++i){
        const double g = P.sigma_GK(i) * (1.0 + P.beta_imb_impact*std::abs(P.Imb(i)));
        const double scale = std::sqrt( (P.P0*P.P0*P.P0) / std::max(1e-12, P.ADV(i)) );
        K(i) = P.gamma_init * scale * g;
    }
    c.segment(nW+nB+nS, N) += K;

    // ---- 等式約束 ----
    Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(1 + N, nvar);
    Eigen::VectorXd beq = Eigen::VectorXd::Zero(1 + N);
    Aeq.block(0, 0, 1, N).setOnes();      beq(0)   = P.budget_long_only;
    Aeq.block(1, 0, N, N).setIdentity();  beq.segment(1, N) = P.w0;
    Aeq.block(1, nW,     N, N).diagonal().array() -= 1.0;  // -u_buy
    Aeq.block(1, nW+nB,  N, N).diagonal().array() += 1.0;  // +u_sell

    // ---- 線性不等式 Gx ≤ h ----
    std::vector<Eigen::RowVectorXd> Grow;
    std::vector<double>             hrow;
    auto add_le = [&](const Eigen::RowVectorXd& g, double hv){ Grow.push_back(g); hrow.push_back(hv); };

    // 非負
    {
        Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
        for(int i=0;i<N;++i){ r.setZero(); r(i) = -1.0;         add_le(r, 0.0); }
        for(int i=0;i<N;++i){ r.setZero(); r(nW+i) = -1.0;      add_le(r, 0.0); }
        for(int i=0;i<N;++i){ r.setZero(); r(nW+nB+i) = -1.0;   add_le(r, 0.0); }
    }

    // 個別上限 w ≤ wmax
    Eigen::VectorXd wmax = Eigen::VectorXd::Constant(N, 1.0);
    if (P.wmax_base.size()==N) wmax = P.wmax_base;
    for(int i=0;i<N;++i){
        const double relax  = 1.0 + P.a_b    * P.BF(i);
        const double shrink = 1.0 - P.a_bias * std::max(0.0, std::abs(P.BIAS(i)) - P.theta_bias);
        wmax(i) = std::max(0.0, wmax(i) * relax * shrink);
    }
    if (P.kappa_ADV > 0.0){
        for(int i=0;i<N;++i) wmax(i) = std::min(wmax(i), P.kappa_ADV * safe_div(P.ADV(i), P.P0));
    }
    if (P.kappa_TS > 0.0){
        std::vector<double> a; a.reserve(N);
        for(int i=0;i<N;++i) a.push_back(P.TS(i));
        std::nth_element(a.begin(), a.begin()+N/2, a.end());
        const double medTS = a[N/2];
        for(int i=0;i<N;++i) wmax(i) = std::min(wmax(i), P.kappa_TS * safe_div(P.TS(i), std::max(1e-12,medTS)));
    }
#if SCS_DEBUG_ON
    quick_feas_sanity(wmax, P.group_cap, P.group, P.tau_max, W_turn, P.budget_long_only);
#endif
    for(int i=0;i<N;++i){ Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar); r(i)=1.0; add_le(r, wmax(i)); }

    // 群組 cap
    for (auto& kv : P.group_cap){
        const int gId = kv.first; const double cap = kv.second;
        Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
        for(int i=0;i<N;++i) if (P.group(i)==gId) r(i) = 1.0;
        add_le(r, cap);
    }

    // L1 turnover
    if (P.turn_norm == TurnoverNorm::L1) {
        Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
        for (int i = 0; i < N; ++i) {
            r(nW + i)      = W_turn(i);
            r(nW + nB + i) = W_turn(i);
        }
        add_le(r, P.tau_max);
    }

    // PWL 衝擊
    std::vector<double> slopes, inters;
    const std::vector<double> bpts = P.pwl.b.empty()
        ? std::vector<double>{0.0, 0.002, 0.005, 0.010, 0.020, 0.040}
        : P.pwl.b;
    build_pwl_upper(bpts, slopes, inters);
    for (int i=0;i<N;++i){
        for (int j=0;j<(int)slopes.size();++j){
            Eigen::RowVectorXd r = Eigen::RowVectorXd::Zero(nvar);
            r(nW + i)              =  slopes[j];
            r(nW + nB + i)         =  slopes[j];
            r(nW + nB + nS + i)    = -1.0;
            add_le(r, -inters[j]);
        }
    }

    // 線性塊
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero((int)Grow.size(), nvar);
    Eigen::VectorXd h = Eigen::VectorXd::Zero((int)hrow.size());
    for (int k=0;k<(int)Grow.size();++k){ G.row(k) = Grow[k]; h(k) = hrow[k]; }

    // ---- SOC 塊 ----
    // 風險 SOC
    Eigen::MatrixXd Gq1 = Eigen::MatrixXd::Zero(N+1, nvar);
    Eigen::VectorXd hq1 = Eigen::VectorXd::Zero(N+1);
    hq1(0) = std::max(0.0, P.m * P.sigma_star);
    Gq1.block(1, 0, N, N) = -P.L.transpose();

    // L2 turnover（可選）
    Eigen::MatrixXd Gq2; Eigen::VectorXd hq2;
    const bool use_turn_l2 = (P.turn_norm == TurnoverNorm::L2);
    if (use_turn_l2) {
        Gq2 = Eigen::MatrixXd::Zero(N+1, nvar);
        hq2 = Eigen::VectorXd::Zero(N+1);
        hq2(0) = std::max(0.0, P.tau_max);
        for (int i=0;i<N;++i){
            Gq2(1+i, nW + i)      = -W_turn(i);
            Gq2(1+i, nW + nB + i) = -W_turn(i);
        }
    }

    // ---- vstack 到 SCS 的 A、b ----
    CSC Aeq_c = to_csc_scs(Aeq);
    CSC G_c   = to_csc_scs(G);
    CSC Gq1_c = to_csc_scs(Gq1);
    CSC Gq2_c;
    std::vector<CSC> blocks{Aeq_c, G_c, Gq1_c};
    if (use_turn_l2){ Gq2_c = to_csc_scs(Gq2); blocks.push_back(Gq2_c); }
    CSC A_csc = vstack_csc(blocks);

    std::vector<Eigen::VectorXd> rhs_vecs = {beq, h, hq1};
    if (use_turn_l2) rhs_vecs.push_back(hq2);
    std::vector<scs_float> bvec = vstack_vec(rhs_vecs);

    std::vector<scs_float> cvec(nvar);
    for (int j = 0; j < nvar; ++j) cvec[j] = (scs_float)c(j);

    // ---- SCS Data/Cone ----
    ScsMatrix A;
    A.m = (scs_int)A_csc.m;  A.n = (scs_int)A_csc.n;
    A.p = const_cast<scs_int*>(   A_csc.Ap.data() );
    A.i = const_cast<scs_int*>(   A_csc.Ai.data() );
    A.x = const_cast<scs_float*>( A_csc.Ax.data() );

    ScsData d; std::memset(&d, 0, sizeof(d));
    ScsCone k; std::memset(&k, 0, sizeof(k));
    d.m = (scs_int)A_csc.m; d.n = (scs_int)nvar; d.A = &A; d.P = nullptr;
    d.b = bvec.data(); d.c = cvec.data();

    std::vector<scs_int> qdims; qdims.push_back((scs_int)(N+1));
    if (use_turn_l2) qdims.push_back((scs_int)(N+1));
    k.z = (scs_int)Aeq.rows();  k.l = (scs_int)G.rows();
    k.qsize = (scs_int)qdims.size(); k.q = qdims.data();

    // ---- 尺寸一致性檢查 ----
    scs_int m_cone = k.z + k.l; for (auto q:qdims) m_cone += q;
    if (m_cone != (scs_int)A_csc.m) {
        std::cerr << "[ERR] Cone rows "<<(int)m_cone<<" != A.m "<<(int)A_csc.m
                  << "  z="<<(int)k.z<<" l="<<(int)k.l<<" q=";
        for(auto q:qdims) std::cerr<<(int)q<<" "; std::cerr<<"\n";
        return std::nullopt;
    }
    if ((int)bvec.size() != (int)A_csc.m){
        std::cerr << "[ERR] b size "<<bvec.size()<<" != m "<<A_csc.m<<"\n";
        return std::nullopt;
    }
    if ((int)cvec.size() != nvar){
        std::cerr << "[ERR] c size "<<cvec.size()<<" != nvar "<<nvar<<"\n";
        return std::nullopt;
    }

    // ---- SCS settings ----
    ScsSettings stg; scs_set_default_settings(&stg);
	stg.eps_abs = 1e-12;           // 放鬆一點
	stg.eps_rel = 1e-12;
	stg.max_iters = 80000;         // 或加迭代上限
	stg.time_limit_secs = 50.0;    // 加時間上限，避免卡住
	stg.verbose = 1;


    // ---- solve ----
    ScsSolution sol; std::memset(&sol, 0, sizeof(sol));
    ScsInfo     info; std::memset(&info, 0, sizeof(info));
    scs_int ret = scs(&d, &k, &stg, &sol, &info);

    if (!(ret == SCS_SOLVED || ret == SCS_SOLVED_INACCURATE)) {
        std::cerr << "[SCS] status="<<info.status
                  << " it="<<info.iter
                  << " pri="<<info.res_pri
                  << " dua="<<info.res_dual
                  << " infeas="<<info.res_infeas
                  << " unbdd="<<info.res_unbdd_a << "\n";
        if (sol.x) scs_free_solution_compat(&sol);
        return std::nullopt;
    }

    // ---- 讀解 ----
    Eigen::VectorXd x(nvar);
    for (int j=0;j<nvar;++j) x(j) = sol.x[j];
    scs_free_solution_compat(&sol);

    PortfolioCandidate out;
    out.w = x.segment(0,N);
    const Eigen::VectorXd ub   = x.segment(nW,     N);
    const Eigen::VectorXd us   = x.segment(nW+nB,  N);
    const Eigen::VectorXd uabs = ub + us;

    out.ret_part = P.rhat_enh.dot(out.w);
    out.lin_cost = P.P0 * (P.fee_buy * ub.sum() + P.fee_sell * us.sum());
    out.imp_cost = (K.array() * x.segment(nW+nB+nS, N).array()).sum();
    out.obj      = out.ret_part - out.lin_cost - out.imp_cost;
    out.risk     = (P.L.transpose() * out.w).norm();
    out.turnover = (P.turn_norm==TurnoverNorm::L1)
                 ? uabs.sum()
                 : (W_turn.array() * uabs.array()).matrix().norm();
    out.sigma_star = P.sigma_star;
    out.tau_max    = P.tau_max;
	std::cerr << "[OBJ] ret=" << out.ret_part
          << " fee=" << out.lin_cost
          << " impact=" << out.imp_cost
          << " obj=" << out.obj << "\n";

    return out;
}




std::vector<PortfolioCandidate> generate_socp_candidates(
    const SocpProblem& base, const SocpSweep& sweep)
{
    // ... 你的預設 pwl、wmax_base 保留 ...

    std::vector<PortfolioCandidate> all; 
    // 預估容量，避免反覆 realloc（粗估）
	// 預設 PWL 斷點
    SocpProblem pb = base;
    if (pb.pwl.b.empty()){
        pb.pwl.b = {0.0, 0.002, 0.005, 0.010, 0.020, 0.040}; // |Δw| (2bp~4%)
    }
    // 若沒提供 wmax_base，給 100% cap
    if (pb.wmax_base.size()!=base.rhat_enh.size()){
        pb.wmax_base = vee(1.0, (int)base.rhat_enh.size());
    }
    all.reserve(sweep.sigma_star_list.size() * sweep.tau_list.size() * sweep.impact_scale.size());

    #pragma omp parallel
    {
        std::vector<PortfolioCandidate> local; 
        // 本執行緒的暫存結果，避免 push_back 互搶

        #pragma omp for collapse(3) nowait
        for (int is = 0; is < (int)sweep.sigma_star_list.size(); ++is)
        for (int it = 0; it < (int)sweep.tau_list.size(); ++it)
        for (int ik = 0; ik < (int)sweep.impact_scale.size(); ++ik)
        {
            SocpProblem pb = base;
            pb.sigma_star  = sweep.sigma_star_list[is];
            pb.tau_max     = sweep.tau_list[it];
            pb.gamma_init  = base.gamma_init * sweep.impact_scale[ik];

            if (auto cand = solve_socp_once(pb)) {
                local.push_back(*cand);
            }
        }

        // 合併到全域
        #pragma omp critical
        {
            all.insert(all.end(), local.begin(), local.end());
        }
    }

    // 去重 + 排序 + 截斷（保留你的原邏輯）
    std::vector<PortfolioCandidate> res;
    for (auto &c : all) {
        bool dup = false;
        for (auto &e : res) if ((e.w - c.w).cwiseAbs().sum() < sweep.dedup_l1_radius) { dup = true; break; }
        if (!dup) res.push_back(c);
    }
    std::sort(res.begin(), res.end(), [](auto&a, auto&b){ return a.obj > b.obj; });
    if ((int)res.size() > sweep.k_keep) res.resize(sweep.k_keep);
    return res;
}

