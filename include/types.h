// types.h          // Core data structures (Vector, Matrix, Portfolio)
#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <Eigen/Dense>

// --- Eigen Type Aliases for Convenience ---
using Vector  = Eigen::VectorXd;
using Matrix  = Eigen::MatrixXd;
using VectorI = Eigen::VectorXi;

// --- Basic enums ---
enum class OrderSide { BUY, SELL };
enum class LotType  { BoardLot = 0, OddLot = 1 };

// --- Core Data Structures ---

/**
 * @brief Static metadata for a single stock.
 */
/*struct StockInfo {
    std::string sid;       // stock code
    int         lot_size     = 1000; // e.g., 1000 shares per board lot (TW)
    int         odd_lot_unit = 1;    // e.g., 1 share for odd-lot trading
    std::string group;               // industry/sector label
};
*/
/**
 * @brief A trade order prepared for execution.
 */
struct Order {
    std::string sid;
    OrderSide   side           = OrderSide::BUY;
    int         shares         = 0;
    LotType     lot_type       = LotType::BoardLot; // 0=board lot, 1=odd lot
    double      estimated_notional = 0.0;  // TWD
    double      estimated_cost     = 0.0;  // fees + tax + impact (TWD)
};

/**
 * @brief A portfolio snapshot (either weight space or lot space).
 * If a field未使用，可保持為空或 -1。
 */
struct Portfolio {
    std::vector<std::string> sids;  // universe order
    Vector   w;                     // weights (sum ≈ 1), size = sids.size()
    VectorI  lots;                  // lots aligned to sids; set -1 if unknown
};

// --- Parameter Structs for Different Modules ---

/**
 * @brief Parameters for a single SOCP optimization run.
 */
struct SOCPParams {
    double m_black_swan = 1.0;     // m in ||L^T w|| ≤ m * sigma_target
    double sigma_target = 0.01;    // risk cap (σ*)
    double max_turnover = 0.20;    // τ_max (fraction of notional per rebalance)
    std::map<std::string, double> group_caps; // group -> cap on sum of weights
};

/**
 * @brief Parameters for the MDP/CVaR Policy Solver.
 */
struct PolicySolverParams {
    double cvar_alpha = 0.95;  // right-tail CVaR level
    // optional reducers for constraint generation:
    int    topK_by_mean = 0;   // per-state keep top-K actions by mean return (0 = all)
    int    max_pairs    = 0;   // global cap on (state,action) base pairs (0 = unlimited)
};

/**
 * @brief Taiwan-style transaction cost model parameters.
 */
struct CostModelParams {
    double fee_rate_buy   = 0.001425; // broker fee (buy side)
    double fee_rate_sell  = 0.004425; // broker + tax (sell side)
    double min_fee        = 20.0;     // exchange minimum per order (TWD)
    double gamma_impact   = 4e-4;     // impact coefficient γ
    double beta_imb_impact= 0.0;      // scaling with |Imb|
};

/**
 * @brief Parameters for the Discrete Simulated Annealing schedule.
 */
struct DSAParams {
    double initial_temp          = 1.0;
    double min_temp              = 1e-4;
    double cooling_rate          = 0.97;
    int    moves_per_temp        = 1500;
    double budget_penalty_start  = 0.1;  // λ_B start
    double budget_penalty_end    = 10.0; // λ_B max
    bool   reduce_only           = false; // disable buys if true
};

#endif // TYPES_H

