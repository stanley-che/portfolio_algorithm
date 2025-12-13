//socp_types.hpp
#pragma once
#include <Eigen/Dense>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace socp {

enum class TurnoverNorm { L1, L2 };

struct PWL {
    std::vector<double> b; // breakpoints
};

struct SocpProblem {
    // required
    Eigen::VectorXd rhat_enh;  // N
    Eigen::MatrixXd L;         // NxN
    Eigen::VectorXd w0;        // N
    Eigen::VectorXd W_turn;    // N

    // market/liquidity inputs
    Eigen::VectorXd ADV;       // N
    Eigen::VectorXd sigma_GK;  // N
    Eigen::VectorXd Imb;       // N
    Eigen::VectorXd BF;        // N
    Eigen::VectorXd BIAS;      // N
    Eigen::VectorXd TS;        // N
    Eigen::VectorXi group;     // N

    // constraints
    double m = 1.0;
    double sigma_star = 0.01;
    double tau_max = 0.20;
    TurnoverNorm turn_norm = TurnoverNorm::L1;
    double budget_long_only = 1.0;

    // caps
    Eigen::VectorXd wmax_base;
    double a_b = 0.0;
    double a_bias = 0.0;
    double theta_bias = 0.10;
    double kappa_ADV = 0.0;
    double kappa_TS  = 0.0;
    std::map<int,double> group_cap;

    // cost params (TWD)
    double P0 = 1.0;
    double fee_buy  = 0.001425;
    double fee_sell = 0.001425;
    double beta_imb_impact = 0.0;
    double gamma_init = 4e-4;

    PWL pwl;
    double eps = 1e-9;
};

struct SocpSweep {
    std::vector<double> sigma_star_list {0.006,0.008,0.010,0.012};
    std::vector<double> tau_list        {0.10,0.15,0.20,0.25};
    std::vector<double> impact_scale    {0.7, 1.0, 1.3};

    double dedup_l1_radius = 0.02;
    int    k_keep = 30;
};

struct PortfolioCandidate {
    Eigen::VectorXd w;              // N
    double obj = 0.0;

    double ret_part = 0.0;
    double lin_cost = 0.0;
    double imp_cost = 0.0;

    double risk = 0.0;
    double turnover = 0.0;

    double sigma_star = 0.0;
    double tau_max = 0.0;
    double impact_scale = 1.0;
};

} // namespace socp
