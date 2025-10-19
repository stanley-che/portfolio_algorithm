//process.h
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <optional>
#include "data_loader.h"

// ==================== Process API ====================
namespace process {

// 2.6：把 (T×N) 特徵做逐日橫斷面標準化（回傳同形狀）
enum class XSectionStandardize { ZScoreClip, QuantileMap };

struct ProcessingConfig {
    XSectionStandardize mode = XSectionStandardize::ZScoreClip;
    double clip = 3.0;            // z-score clip 到 [-clip, clip]
    double winsor_p = 0.01;       // 若用 QuantileMap，可一併做 1%/99% 邊界
    bool fill_missing_with_zero = true; // 缺值填 0（或之後改成產業中位數）
};
Eigen::MatrixXd xsection_standardize(const Eigen::MatrixXd& X, const ProcessingConfig& cfg);

// 2.3：建立流動性加權 Σ̃ 與 L
enum class SigmaSource { GK, Std30 };

struct RiskConfig {
    double corr_half_life = 60;   // 相關係數 EWMA 半衰期
    SigmaSource sigma_source = SigmaSource::GK; // GK 或 30D 標準差
    int std_window = 30;          // 30D 標準差窗口
    double c_liq = 0.0;           // 流動性上調強度 c_ℓ ≥ 0
    double eps = 1e-6;            // TurnoverShare 防除零
};

struct RiskOutput {
    Eigen::MatrixXd Sigma_tilde;  // Σ̃
    Eigen::MatrixXd L;            // Cholesky: L L^T = Σ̃
    Eigen::VectorXd sigma_i;      // 每資產 σ_i
    Eigen::VectorXd Dliq_i;       // 每資產 D_liq,i（無量綱）
};

// 2.3：建立流動性加權 Σ̃ 與 L
RiskOutput build_liquidity_scaled_risk(
    const DataLoader& dl,          // 需 feat_TurnoverShare / VAL / GKVol
    int t,                         // 以第 t 天為基準
    const RiskConfig& cfg);


// 2.4：估算交易成本；A_i 為「單邊金額（TWD）」；side_i: +1=買 / -1=賣 / 0=不交易
// =============== 2.4 Cost Model（TW fee + tax + impact） ===========
struct CostConfig {
    double fee_rate = 0.001425; // 券商手續費
    double tax_rate = 0.003;    // 證交稅（僅賣出）
    double min_fee = 20.0;      // 最低手續費（單邊）
    int adv_window = 30;        // ADV 以 30 日 VAL 均值
    double gamma_init = 4e-4;   // 衝擊係數 γ_i 初值
    double beta_imb   = 0.0;    // 衝擊與 |Imb| 的聯動係數
};

struct CostBreakdown {
    Eigen::VectorXd fee_buy;
    Eigen::VectorXd fee_sell;
    Eigen::VectorXd tax_sell;
    Eigen::VectorXd impact;
    double total() const {
        return fee_buy.sum() + fee_sell.sum() + tax_sell.sum() + impact.sum();
    }
};
CostBreakdown estimate_costs(
    const DataLoader& dl, int t,
    const Eigen::VectorXd& A,
    const Eigen::VectorXi& side,
    const CostConfig& cfg);

// 2.5：計算黑天鵝縮放 m（用橫斷面指標的中位數近似市場狀態）
// =============== 2.5 Black-Swan Scaling m ==========================
struct BlackSwanConfig {
    // 觸發條件（可依需求調整）
    double m_default = 1.0;
    double m_gap_bad = 0.7;         // 市場跳空太差
    double m_vol_high = 0.8;        // 市場GK波動比均值偏高
    double m_imb_bad = 0.8;         // 市場賣壓偏強
    double gap_threshold = -0.03;   // median(Gap) < -3%
    double vol_ratio_th = 1.5;      // median(GK)/median_30d > 1.5
    double imb_median_th = -0.2;    // median(Imb) < -0.2
    double m_event = 0.5;           // 外部事件（人工開關）
};

struct BlackSwanOutput {
    double m = 1.0;                 // 0.5 ~ 1.5
};
BlackSwanOutput black_swan_scale(const DataLoader& dl, int t, const BlackSwanConfig& cfg, bool external_event=false);

// 2.6：Stage-1 原始/校準/增強預測
// 需要：特徵矩陣（當天第 t 橫斷面），可選：回溯期真實報酬（用於線性β）

// =============== 2.6 Calibrated Forecast for Stage 1 ===============
struct ForecastConfig {
    // 回歸設定（線性走步加權；GBDT 留接口）
    int lookback = 252;             // 迴歸視窗
    double half_life = 60.0;        // 權重半衰期
    bool use_linear = true;         // true=線性β^T x；false=外部模型 f_θ(x)

    // 校準與增強
    double z_clip = 3.0;            // 橫斷面 z-score clip
    double theta_b   = 0.1;         // BF 權重區間建議 [0.1, 0.5]
    double theta_imb = 0.1;         // Imb 權重區間建議 [0.05, 0.3]
    double theta_ts  = 0.1;         // TurnoverShare 權重建議 [0.05, 0.3]
    double kappa_bias = 1.0;        // 罰則係數 κ_bias ∈ [0.5, 1.5]
    double theta_bias = 0.10;       // |BIAS| 的容忍門檻（10%）

    // 風險目標
    double sigma_target = 0.01;     // 目標投組波動（以 ||L^T w|| ≤ m σ_target）
};

struct ForecastOutput {
    Eigen::VectorXd rhat_raw;   // β^T x 或 fθ(x)
    Eigen::VectorXd rhat_cal;   // 橫斷面 z/quantile 校準
    Eigen::VectorXd rhat_enh;   // 加上 BF/Imb/TS，扣 BIAS 罰則
};

ForecastOutput stage1_forecast(
    const DataLoader& dl, int t,
    const ProcessingConfig& px_cfg,
    const ForecastConfig& fc,
    const std::optional<Eigen::MatrixXd>& next_day_returns = std::nullopt // (T×N)，若給就做線性回歸
);

} // namespace process

