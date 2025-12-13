#pragma once
// Discrete Simulated Annealing (DSA) in share/lot space
// Implements §6 Execution Layer: objective, moves, fast ΔC, cooling, black-swan modes.

#include <Eigen/Dense>
#include <map>
#include <vector>
#include <optional>
#include <cstdint>
#include <random> 
namespace dsa {

// ----------- User config (inputs) -----------
struct Config {
    // Universe
    double                P0 = 1.0;          // portfolio notional (TWD)
    Eigen::VectorXd       price;             // N, TWD/share
    Eigen::VectorXi       lot;               // N, shares per lot (e.g., 1000)
    Eigen::VectorXi       n0;                // N, current holding (in lots). Cost baseline.
    // Signals & risk
    Eigen::VectorXd       r_enh;             // N, enhanced forecast per weight
    Eigen::MatrixXd       Sigma_tilde;       // N×N covariance
    Eigen::MatrixXd       L;                 // N×N optional (if empty -> computed via LLT)
    double                m           = 1.0; // black-swan scaling
    double                sigma_star  = 0.01;// risk cap in ||L^T w|| ≤ m*sigma_star

    // Soft penalties in objective: C = rᵀw − λ/2 wᵀΣw − Tc/P0 − λ_B (∑w − 1)²
    double                lambda_risk   = 0.0;   // set >0 to add quadratic risk penalty
    double                lambdaB_start = 0.1;
    double                lambdaB_max   = 10.0;

    // Caps
    Eigen::VectorXd       wmax;                 // N, 0..1 (empty = no cap)
    std::map<int,double>  group_cap;            // group id -> cap
    Eigen::VectorXi       group_id;             // N (empty = no group cap)

    // Features used by costs / priorities
    Eigen::VectorXd       ADV;                  // N, 30D traded value (TWD)
    Eigen::VectorXd       sigma_GK;             // N
    Eigen::VectorXd       Imb;                  // N
    Eigen::VectorXd       BF;                   // N
    Eigen::VectorXd       BIAS;                 // N
    Eigen::VectorXd       TurnoverShare;        // N

    // Trading costs (weight-based; Δw is fraction of P0)
    double                fee_buy   = 0.001425; // include tax in sell-side if desired
    double                fee_sell  = 0.004425;
    double                gamma_init= 4e-4;     // impact γ
    double                beta_imb_impact = 0.2;// impact scaled by |Imb|

    // Annealing schedule
    int                   moves_per_T   = 1500; // 1000–2000
    double                cool          = 0.97; // geometric cooling
    double                T_floor       = 1e-4;
    int                   stagnate_reheat_T = 5;// reheat if no gain for L temperatures
    double                initT_eta     = 1.5;  // T0 = η * median(|ΔC|)
    int                   max_T_steps   = 40;   // max temperatures

    // Black-swan
    bool                  reduce_only       = false; // disable buys (Δw_i>0) if true
    bool                  pre_scale_to_risk = true;  // quickly scale down to meet risk
    double                risk_quick_scale  = 0.9;   // ρ<1 factor for quick scaling

    // Late-phase odd-lot
    bool                  enable_odd_lot_phase = true;
    double                odd_lot_start_T       = 5e-4; // allow ±1 share when T < this

    // RNG
    uint32_t              seed = 12345;
};

// ----------- Result -----------
struct Result {
    Eigen::VectorXi n_best;    // best lots
    Eigen::VectorXd w_best;    // best weights
    double          C_best = -1e100;
    double          risk_best = 0.0;   // ||L^T w||
    double          budget_violation = 0.0; // |∑w−1|
    int             iters = 0;         // proposals
    int             accepts = 0;       // accepted
};

// ----------- Executor -----------
class Executor {
public:
    explicit Executor(const Config& cfg);

    // 可選給一個「目標權重」作為起點（例如 SOCP 輸出）；留空則從 n0 開始。
    Result run(const std::optional<Eigen::VectorXd>& w_seed = std::nullopt);

private:
    // ===== helpers =====
    void   build_cholesky();
    void   initialize(const std::optional<Eigen::VectorXd>& w_seed);
    void   quick_scale_to_risk();
    bool   hard_violations_after_delta(const std::vector<int>& idxs,
                                       const Eigen::VectorXd& delta_w) const;

    // single proposal generators (return index set S and δw on S)
    void   propose_buy_lot(int& i, Eigen::VectorXd& dw);
    void   propose_sell_lot(int& i, Eigen::VectorXd& dw);
    void   propose_swap_pair(int& i, int& j, Eigen::VectorXd& dw);
    bool   propose_odd_lot(int& i, Eigen::VectorXd& dw); // ±1 share

    // feature-driven priorities
    void   refresh_sampling_weights();

    // incremental ΔC using cached q=Σw and trade deltas vs baseline w_hold
    double delta_objective(const std::vector<int>& idxs,
                           const Eigen::VectorXd& dw) const;

    // accept: commit n,w, caches
    void   accept_move(const std::vector<int>& idxs,
                       const Eigen::VectorXd& dw, double dC);

    // recompute totals
    double compute_objective_full() const;
    double current_risk() const;

private:
    // config & derived
    Config C_;
    int    N_ = 0;

    // state (current)
    Eigen::VectorXi n_;             // lots
    Eigen::VectorXd w_;             // weights
    Eigen::VectorXd w_hold_;        // baseline = from n0
    Eigen::VectorXd unit_w_lot_;    // lot_i * price_i / P0
    Eigen::VectorXd unit_w_share_;  // price_i / P0
    Eigen::VectorXd q_;             // Σ w (for fast Δrisk)
    double          s_sum_ = 0.0;   // ∑ w
    double          lambdaB_ = 0.0; // budget penalty λ_B (ramps)

    // caches for cost vs baseline (Δw = w - w_hold)
    Eigen::VectorXd dW_;            // Δw
    Eigen::VectorXd buy_pos_;       // max(Δw,0)
    Eigen::VectorXd sell_pos_;      // max(-Δw,0)
    Eigen::VectorXd abs_pos_;       // |Δw|
    Eigen::VectorXd K_imp_;         // impact coefficient per name
    double          lin_cost_ = 0.0;
    double          imp_cost_ = 0.0;

    // risk
    Eigen::MatrixXd L_;
    Eigen::MatrixXd Sigma_;         // alias of Σ̃

    // priorities
    Eigen::VectorXd buy_score_;
    Eigen::VectorXd sell_score_;
    std::vector<int> buy_pool_;
    std::vector<int> sell_pool_;
    std::vector<std::pair<int,int>> swap_pairs_;

    // rng
    mutable std::mt19937 rng_;
};

} // namespace dsa

