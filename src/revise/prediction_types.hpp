//prediction_types.hpp
// prediction_types.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <optional>

namespace prediction {

// ---------- 3.1 Target ----------
enum class TargetType { OO, OC, CC }; // 開-開 / 開-收 / 收-收

// ---------- 3.2 模型類型 ----------
enum class ModelType { Ridge, Lasso, GBDT /*預留*/ };

struct TrainConfig {
    int    window = 252;
    double half_life = 60.0;
    ModelType model = ModelType::Ridge;
    double lambda = 1.0;
    // Lasso
    int    lasso_max_iter = 500;
    double lasso_tol      = 1e-7;
};

struct CalibrationConfig {
    double q50    = 0.003;
    double q75    = 0.010;
    double z_clip = 3.0;
};

struct EnhancementConfig {
    double theta_b    = 0.10;
    double theta_imb  = 0.10;
    double theta_ts   = 0.10;
    double kappa_bias = 1.00;
    double theta_bias = 0.10;
};

struct PredictionOutput {
    Eigen::VectorXd rhat_raw;  // (N)
    Eigen::VectorXd rhat_cal;  // (N)
    Eigen::VectorXd rhat_enh;  // (N)
    Eigen::VectorXd beta;      // (P)；GBDT 時可留空
};

} // namespace prediction
