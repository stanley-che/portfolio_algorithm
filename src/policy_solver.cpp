// policy_solver.cpp — 使用類別成員 idx(s,a)，修正 HiGHS addVars/addCols 相容
#include "policy_solver.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#ifdef USE_HIGHS
  #include "Highs.h"
  #ifndef kHighsInf
  #define kHighsInf 1e30
  #endif
  #if defined(HIGHS_VERSION_MAJOR) && defined(HIGHS_VERSION_MINOR)
    constexpr bool HIGHS_HAS_ADDVARS =
        (HIGHS_VERSION_MAJOR > 1) ||
        (HIGHS_VERSION_MAJOR == 1 && HIGHS_VERSION_MINOR >= 5);
  #else
    constexpr bool HIGHS_HAS_ADDVARS = true;
  #endif
#endif

using std::vector; using std::pair; using std::string;

static inline double safe_div(double a, double b){ return (std::abs(b)<1e-12)?0.0:a/b; }

PolicyLPSolver::PolicyLPSolver(const PolicySamples& smp, const Transitions& trans, const PolicyLPConfig& cfg)
: S_(smp), T_(trans), C_(cfg) {
    if (S_.S<=0 || S_.A<=0 || S_.M<=0) throw std::runtime_error("Invalid sample tensor shape");
    if (T_.S != S_.S || T_.A != S_.A)  throw std::runtime_error("Transition dims mismatch");
    if ((int)T_.P.size()!=S_.S) throw std::runtime_error("Transition P dim[0] invalid");
}
PolicyLPSolver::~PolicyLPSolver() = default;

vector<pair<int,int>> PolicyLPSolver::choose_base_pairs() const {
    vector<pair<int,int>> pairs;
    if (C_.topK_by_mean<=0 && C_.max_pairs<=0) {
        pairs.reserve(S_.S * S_.A);
        for (int i=0;i<S_.S;++i)
            for (int a=0;a<S_.A;++a)
                pairs.emplace_back(i,a);
        return pairs;
    }
    int K = std::max(1, C_.topK_by_mean);
    std::mt19937 rng(42);
    for (int i=0;i<S_.S;++i){
        vector<pair<double,int>> score; score.reserve(S_.A);
        for (int a=0;a<S_.A;++a){
            const auto& v = S_.samples[i][a];
            double mu = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            score.push_back({mu, a});
        }
        std::partial_sort(score.begin(), score.begin()+std::min(K,(int)score.size()), score.end(),
                          [](auto&x,auto&y){ return x.first>y.first; });
        for (int k=0;k<std::min(K,(int)score.size());++k)
            pairs.emplace_back(i, score[k].second);
    }
    if (C_.max_pairs>0 && (int)pairs.size()<C_.max_pairs){
        vector<pair<int,int>> pool;
        pool.reserve(S_.S*S_.A);
        for (int i=0;i<S_.S;++i)
            for (int a=0;a<S_.A;++a)
                pool.emplace_back(i,a);
        std::shuffle(pool.begin(), pool.end(), rng);
        for (auto &p: pool){
            if ((int)pairs.size()>=C_.max_pairs) break;
            if (std::find(pairs.begin(), pairs.end(), p)==pairs.end())
                pairs.push_back(p);
        }
    }
    return pairs;
}

void PolicyLPSolver::build_lp_matrices(Eigen::MatrixXd& Aeq, Eigen::VectorXd& beq,
                                       Eigen::MatrixXd& Gge, Eigen::VectorXd& hge,
                                       Eigen::VectorXd& c,
                                       vector<string>& var_names) const
{
    const int S = S_.S, A = S_.A, M = S_.M;
    const int X = S*A;
    const int Z = X;
    const int NV = X + 1;

    var_names.clear();
    var_names.reserve(NV);
    for (int s=0;s<S;++s) for (int a=0;a<A;++a)
        var_names.push_back("x_"+std::to_string(s)+"_"+std::to_string(a));
    var_names.push_back("z");

    c = Eigen::VectorXd::Zero(NV);
    c(Z) = 1.0;

    Aeq = Eigen::MatrixXd::Zero(S + 1, NV);
    beq = Eigen::VectorXd::Zero(S + 1);

    // Flow constraints
    for (int j=0;j<S;++j){
        for (int a=0;a<A;++a) Aeq(j, idx(j,a)) += 1.0;
        for (int i=0;i<S;++i){
            for (int a=0;a<A;++a){
                double pij = T_.P[i][a][j];
                Aeq(j, idx(i,a)) -= pij;
            }
        }
        beq(j) = 0.0;
    }
    // Normalization
    for (int i=0;i<S;++i) for (int a=0;a<A;++a) Aeq(S, idx(i,a)) = 1.0;
    beq(S) = 1.0;

    // CVaR constraints
    auto base_pairs = choose_base_pairs();
    const int K = (int)base_pairs.size();

    Gge = Eigen::MatrixXd::Zero(K, NV);
    hge = Eigen::VectorXd::Zero(K);

    const double coef = 1.0 / (1.0 - C_.alpha);
    for (int k=0;k<K;++k){
        int i0 = base_pairs[k].first;
        int a0 = base_pairs[k].second;

        double c_base = 0.0;
        for (int ell=0; ell<M; ++ell) c_base += S_.samples[i0][a0][ell];
        c_base /= (double)M;

        for (int i=0;i<S;++i){
            for (int a=0;a<A;++a){
                double d = 0.0;
                for (int ell=0; ell<M; ++ell) {
                    double ri = S_.samples[i][a][ell];
                    double r0 = S_.samples[i0][a0][ell];
                    double diff = ri - r0;
                    if (diff > 0.0) d += diff;
                }
                d = coef * d / (double)M;
                double coeff = c_base + d;
                Gge(k, idx(i,a)) = coeff;
            }
        }
        Gge(k, Z) = -1.0;
        hge(k) = 0.0;
    }
}

void PolicyLPSolver::write_lp_file(const std::string& path) const {
    Eigen::MatrixXd Aeq, Gge; Eigen::VectorXd beq, hge, c;
    vector<string> names;
    build_lp_matrices(Aeq, beq, Gge, hge, c, names);

    const int NV = (int)names.size();
    const int NE = (int)Aeq.rows();
    const int NG = (int)Gge.rows();

    std::ofstream lp(path);
    if (!lp.is_open()) throw std::runtime_error("Cannot open LP file for write: " + path);

    lp << "Maximize\n obj: ";
    bool first = true;
    for (int j=0;j<NV;++j){
        double cj = c(j);
        if (std::abs(cj) < 1e-14) continue;
        if (!first) lp << (cj>=0 ? " + " : " - ");
        if (first && cj<0) lp << "- ";
        lp << std::fixed << std::setprecision(12) << std::abs(cj) << " " << names[j];
        first = false;
    }
    lp << "\nSubject To\n";

    for (int i=0;i<NG;++i){
        lp << " cvar_"<<i<<": ";
        bool f=true;
        for (int j=0;j<NV;++j){
            double a = -Gge(i,j);
            if (std::abs(a) < 1e-14) continue;
            if (!f) lp << (a>=0 ? " + " : " - ");
            if (f && a<0) lp << "- ";
            lp << std::fixed << std::setprecision(12) << std::abs(a) << " " << names[j];
            f=false;
        }
        lp << " <= " << std::fixed << std::setprecision(12) << -hge(i) << "\n";
    }

    for (int i=0;i<NE;++i){
        lp << " flow_"<<i<<": ";
        bool f=true;
        for (int j=0;j<NV;++j){
            double a = Aeq(i,j);
            if (std::abs(a) < 1e-14) continue;
            if (!f) lp << (a>=0 ? " + " : " - ");
            if (f && a<0) lp << "- ";
            lp << std::fixed << std::setprecision(12) << std::abs(a) << " " << names[j];
            f=false;
        }
        lp << " = " << std::fixed << std::setprecision(12) << beq(i) << "\n";
    }

    lp << "Bounds\n";
    for (int j=0;j<NV-1;++j) lp << "  0 <= " << names[j] << "\n";
    lp << "End\n";
    lp.close();
    if (C_.verbose) std::cerr << "[policy] LP written to " << path << "\n";
}

PolicySolution PolicyLPSolver::solve() {
    PolicySolution ans; ans.solved=false;
    Eigen::MatrixXd Aeq, Gge; Eigen::VectorXd beq, hge, c;
    vector<string> names;
    build_lp_matrices(Aeq, beq, Gge, hge, c, names);

#ifndef USE_HIGHS
    write_lp_file("policy_lp.lp");
    if (C_.verbose) {
        std::cerr << "[policy] Define USE_HIGHS and link HiGHS to solve in-process.\n";
        std::cerr << "[policy] Example: highs -m policy_lp.lp\n";
    }
    return ans;
#else
    const int NV = (int)names.size();
    const int NE = (int)Aeq.rows();
    const int NG = (int)Gge.rows();

    Highs highs;
	highs.setOptionValue("output_flag", C_.verbose);
	highs.setOptionValue("log_to_console", C_.verbose);
	highs.setOptionValue("presolve", "on");

	// ★ 這三種寫法擇一（視你編好的 HiGHS 版本而定）：
	// 1) 新版常用
	highs.setOptionValue("objective_sense", "maximize");
	// 2) 有些版本用這個 key
	highs.setOptionValue("objective_sense", "max");
	// 3) C++ enum 介面（若可用）
	#ifdef HIGHS_HAVE_OBJ_SENSE_ENUM
	highs.changeObjectiveSense(ObjSense::kMaximize);
	#endif


    std::vector<double> lb(NV, -kHighsInf), ub(NV,  kHighsInf), cost(NV);
    for (int j=0;j<NV-1;++j) lb[j] = 0.0;
    for (int j=0;j<NV;++j)   cost[j] = c(j);

  #if HIGHS_HAS_ADDVARS
    HighsStatus st_add = highs.addVars((HighsInt)NV, cost.data(), lb.data(), ub.data());
  #else
    HighsStatus st_add = highs.addCols((HighsInt)NV, cost.data(), lb.data(), ub.data(),
                                       0, nullptr, nullptr, nullptr);
  #endif
    if (st_add != HighsStatus::kOk && st_add != HighsStatus::kWarning) {
        if (C_.verbose) std::cerr << "[policy] addVars/addCols failed\n";
        return ans;
    }

    for (int i=0;i<NE;++i){
        std::vector<HighsInt> idxv; idxv.reserve(NV);
        std::vector<double>   valv; valv.reserve(NV);
        for (int j=0;j<NV;++j){
            double a = Aeq(i,j);
            if (std::abs(a) > 1e-14) { idxv.push_back((HighsInt)j); valv.push_back(a); }
        }
        highs.addRow(beq(i), beq(i), (HighsInt)idxv.size(), idxv.data(), valv.data());
    }
    for (int i=0;i<NG;++i){
        std::vector<HighsInt> idxv; idxv.reserve(NV);
        std::vector<double>   valv; valv.reserve(NV);
        for (int j=0;j<NV;++j){
            double a = -Gge(i,j);
            if (std::abs(a) > 1e-14) { idxv.push_back((HighsInt)j); valv.push_back(a); }
        }
        highs.addRow(-kHighsInf, -hge(i), (HighsInt)idxv.size(), idxv.data(), valv.data());
    }

    HighsStatus st = highs.run();
    if (st != HighsStatus::kOk && st != HighsStatus::kWarning) {
        if (C_.verbose) std::cerr << "[policy] HiGHS run() failed.\n";
        return ans;
    }
    auto model_status = highs.getModelStatus();
    if (model_status != HighsModelStatus::kOptimal &&
        model_status != HighsModelStatus::kObjectiveBound){
        if (C_.verbose) std::cerr << "[policy] Model not optimal: " << (int)model_status << "\n";
        return ans;
    }

    HighsSolution sol = highs.getSolution();
    Eigen::VectorXd xv = Eigen::VectorXd::Zero(NV);
    for (int j=0;j<NV;++j) xv(j) = sol.col_value[j];

    ans.solved   = true;
    ans.objective= highs.getInfo().objective_function_value;
    ans.x        = Eigen::MatrixXd::Zero(S_.S, S_.A);
    for (int s=0;s<S_.S;++s) for (int a=0;a<S_.A;++a) ans.x(s,a) = xv(idx(s,a));
    return ans;
#endif
}

