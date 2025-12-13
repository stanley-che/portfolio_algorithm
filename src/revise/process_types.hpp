//process_types.hpp
#pragma once
#include <Eigen/Dense>
#include <optional>
#include <string>

#include "data_loader.hpp"   // 你的 DataLoader

namespace process {

// ---------- 2.6 XSection Standardize ----------
enum class XSectionStandardize { ZScoreClip, QuantileMap };

struct ProcessingConfig {
    XSectionStandardize mode = XSectionStandardize::ZScoreClip;
    double clip = 3.0;
    double winsor_p = 0.01;
    bool fill_missing_with_zero = true;
};

// ---------- 2.3 Risk ----------
enum class SigmaSource { GK, Std30 };

struct RiskConfig {
    double corr_half_life = 60;
    SigmaSource sigma_source = SigmaSource::GK;
    int std_window = 30;
    double c_liq = 0.0;
    double eps = 1e-6;
};

struct RiskOutput {
    Eigen::MatrixXd Sigma_tilde;
    Eigen::MatrixXd L;
    Eigen::VectorXd sigma_i;
    Eigen::VectorXd Dliq_i;
};

// ---------- 2.4 Cost ----------
struct CostConfig {
    double fee_rate = 0.001425;
    double tax_rate = 0.003;
    double min_fee = 20.0;
    int adv_window = 30;
    double gamma_init = 4e-4;
    double beta_imb   = 0.0;
};

struct CostBreakdown {
    Eigen::VectorXd fee_buy, fee_sell, tax_sell, impact;
    double total() const { return fee_buy.sum()+fee_sell.sum()+tax_sell.sum()+impact.sum(); }
};

// ---------- 2.5 Black Swan ----------
struct BlackSwanConfig {
    double m_default = 1.0;
    double m_gap_bad = 0.7;
    double m_vol_high = 0.8;
    double m_imb_bad = 0.8;
    double gap_threshold = -0.03;
    double vol_ratio_th  = 1.5;
    double imb_median_th = -0.2;
    double m_event = 0.5;
};
struct BlackSwanOutput { double m = 1.0; };

// ---------- 2.6 Forecast ----------
struct ForecastConfig {
    int lookback = 252;
    double half_life = 60.0;
    bool use_linear = true;

    double z_clip = 3.0;
    double theta_b    = 0.1;
    double theta_imb  = 0.1;
    double theta_ts   = 0.1;
    double kappa_bias = 1.0;
    double theta_bias = 0.10;

    double sigma_target = 0.01;
};

struct ForecastOutput {
    Eigen::VectorXd rhat_raw;
    Eigen::VectorXd rhat_cal;
    Eigen::VectorXd rhat_enh;
};

// ---------- Export ----------
struct DumpExcelConfig {
    std::string xlsx_path = "advance_parameter.xlsx";
    std::string out_dir_csv = ".";
    bool prefer_xlsx = true; // 若沒 USE_XLSX 則自動 fallback CSV
};

} // namespace process
