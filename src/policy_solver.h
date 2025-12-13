//policy_solver.h
#pragma once
#include <vector>
#include <string>
#include <optional>
#include <Eigen/Dense>

// ---------- Data container ----------
// samples[s][a][ell]  : 對於 state s、動作 a 的第 ell 個淨報酬樣本 (含費稅衝擊)
struct PolicySamples {
    int S = 0;                    // #states
    int A = 0;                    // #actions (候選投組)
    int M = 0;                    // samples per (s,a)
    std::vector<std::string> state_names;
    std::vector<std::string> action_names;
    std::vector<std::vector<std::vector<double>>> samples; // S x A x M
};

// P[j|i,a]：S x A x S；若與 a 無關，可把每個 a 都放同一份
struct Transitions {
    int S = 0, A = 0;
    std::vector<std::vector<std::vector<double>>> P; // S x A x S
};

// ---------- Solver options ----------
struct PolicyLPConfig {
    double alpha = 0.95;                // CVaR level
    // 只對一部分 (i0,a0) 生成約束（加速）。若 =0 代表用全部；否則取每個 state 前 topK_by_mean動作 + 隨機補齊至 max_pairs。
    int topK_by_mean = 0;               // 每個 state 選 top-K mean 做 (i0,a0)
    int max_pairs    = 0;               // 全域上限（0=不限）
    bool verbose = true;
};

// 結果（含策略）
struct PolicySolution {
    bool solved = false;
    double objective = 0.0;             // 最佳 z
    Eigen::MatrixXd x;                  // S x A (occupation measures)
    // 建議 policy：π(a|s) = x(s,a)/Σ_a x(s,a)
    Eigen::MatrixXd policy() const {
        Eigen::MatrixXd pi = x;
        for (int s=0; s<pi.rows(); ++s) {
            double denom = pi.row(s).sum();
            if (denom > 0) pi.row(s) /= denom;
        }
        return pi;
    }
};

// ---------- API ----------
class PolicyLPSolver {
public:
    PolicyLPSolver(const PolicySamples& smp, const Transitions& trans, const PolicyLPConfig& cfg);
    ~PolicyLPSolver();
    // 產生並求解 LP；若未定義 USE_HIGHS，會把模型輸出為 policy_lp.lp，並回傳 false（等待你用外部解器）
    PolicySolution solve();

    // 也可只輸出 .lp 檔（CPLEX/GLPK/HiGHS 均可讀）
    void write_lp_file(const std::string& path) const;

private:
    PolicySamples S_;
    Transitions T_;
    PolicyLPConfig C_;

    // 內部：建構 LP 矩陣（標準形式：max z; Ax = b, Gx >= h, x>=0）
    // 我們把所有 x(s,a) 連成向量；最後一個變數是 z
    void build_lp_matrices(Eigen::MatrixXd& Aeq, Eigen::VectorXd& beq,
                           Eigen::MatrixXd& Gge, Eigen::VectorXd& hge,
                           Eigen::VectorXd& c,
                           std::vector<std::string>& var_names) const;

    // 產生 (i0,a0) 基準對（減量）
    std::vector<std::pair<int,int>> choose_base_pairs() const;

    // 幫助：變數索引
    inline int idx(int s, int a) const { return s * S_.A + a; }
    inline int nvar() const { return S_.S * S_.A + 1; } // + z
};

