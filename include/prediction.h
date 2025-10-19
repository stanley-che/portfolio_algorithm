//prediction.h
#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <optional>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "data_loader.h"
#include "process.h"  // 只用它的橫斷面標準化工具
#include "types.h" 
using namespace std;
// ---------- 3.1 Target ----------
enum class TargetType { OO, OC, CC }; // 規格：開-開 / 開-收 / 收-收

// ---------- 3.2 模型類型 ----------
enum class ModelType { Ridge, Lasso, GBDT /*接口預留*/ };

struct TrainConfig {
    int    window = 252;        // W：回看 6~12 個月
    double half_life = 60.0;    // 時間權重半衰期
    ModelType model = ModelType::Ridge;
    double lambda = 1.0;        // Ridge/Lasso 正則化強度
    // Lasso（座標下降法）參數
    int    lasso_max_iter = 500;
    double lasso_tol      = 1e-7;
};

// ---------- 3.3 校準 + 增強 ----------
struct CalibrationConfig {
    double q50  = 0.003;   // 0.3%
    double q75  = 0.010;   // 1.0%
    double z_clip = 3.0;   // 橫斷面剪裁
};

struct EnhancementConfig {
    double theta_b   = 0.10;  // BF 權重（建議 0.1~0.5）
    double theta_imb = 0.10;  // Imb 權重（0.05~0.3）
    double theta_ts  = 0.10;  // TurnoverShare 權重（0.05~0.3）
    double kappa_bias= 1.00;  // |BIAS| 罰則倍率（0.5~1.5）
    double theta_bias= 0.10;  // |BIAS| 容忍門檻（10%）
};

struct PredictionOutput {
    Eigen::VectorXd rhat_raw;   // β^T x 或外部模型分數（N）
    Eigen::VectorXd rhat_cal;   // 量化校準後（N）
    Eigen::VectorXd rhat_enh;   // 增強後（N）
    Eigen::VectorXd beta;       // 線性模型係數（P），GBDT 時為空
};

// ------------- API -------------
// 訓練於 [t-W, t-1]，在 t 當天做預測（全橫斷面）
PredictionOutput predict_day(
    const DataLoader& dl,
    int t,
    TargetType target,                 // OO / OC / CC
    const TrainConfig&   tr,           // 3.2
    const process::ProcessingConfig& px, // 橫斷面標準化設定
    const CalibrationConfig&  cal,     // 3.3 校準
    const EnhancementConfig&  enh      // 3.3 增強
);

// 方便：把特徵向量（N×P）組出來（已做逐日橫斷面標準化）
Eigen::MatrixXd build_feature_matrix_at(
    const DataLoader& dl, int t, const process::ProcessingConfig& px);

// 取「目標報酬」向量 r_{i,s->s+1} 依 target 定義（與 day s 的特徵對齊）
Eigen::VectorXd target_return_between(
    const DataLoader& dl, int s, TargetType target);

