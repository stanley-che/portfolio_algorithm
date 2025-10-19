#pragma once
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <optional>

// 前置：你前面已經有的模組
#include "data_loader.h"
#include "process.h"      // 取 RiskOutput(L, Sigma_tilde)
#include "prediction.h"   // 取 r̂_enh（或你也可以自己給）

// ============ 參數結構 ============

// 轉手率約束型態
enum class TurnoverNorm { L1, L2 };

// PWL 逼近設定：f(u)=u^{3/2} 的上界線段 (bj, fj) with fj = bj^{3/2}
struct PWL {
    std::vector<double> b;  // 斷點（weight 變化 |Δw|，例如 {0,0.002,0.005,0.01,0.02,0.04}）
};

// 問題設定（一次求解）
struct SocpProblem {
    // 必要輸入
    Eigen::VectorXd rhat_enh;   // N
    Eigen::MatrixXd L;          // N×N, risk Cholesky (L L^T = Σ̃)
    Eigen::VectorXd w0;         // N
    Eigen::VectorXd W_turn;     // N, 轉手率權重（通常 = 1 或 = 1/lot 等）

    // 市場/流動性資料（成本/上限）
    Eigen::VectorXd ADV;        // N, 30D VAL 均值（以 TWD）
    Eigen::VectorXd sigma_GK;   // N, GK 當日波動
    Eigen::VectorXd Imb;        // N, 當日 imbalance（|Imb| 用於 impact scale）
    Eigen::VectorXd BF;         // N, 券商淨買超強度（上限放寬用）
    Eigen::VectorXd BIAS;       // N, 價格偏離（上限收縮用）
    Eigen::VectorXd TS;         // N, TurnoverShare（用於上限）
    Eigen::VectorXi group;      // N, 群組 id（-1 表未指定）

    // 基本約束
    double m = 1.0;             // black-swan 縮放
    double sigma_star = 0.01;    // 風險門檻（將乘上 m）
    double tau_max = 0.20;       // 轉手率上限（權重 L1 或 L2）
    TurnoverNorm turn_norm = TurnoverNorm::L1;
    double budget_long_only = 1.0;    // sum w = 1, w >= 0

    // 權重上限
    Eigen::VectorXd wmax_base;   // N, 基礎上限（可為 0，會自動以常數補 100%）
    double a_b = 0.0;            // BF 放寬係數
    double a_bias = 0.0;         // BIAS 收縮係數
    double theta_bias = 0.10;
    double kappa_ADV = 0.0;      // 若 >0, w_i ≤ kappa_ADV * ADV_i / P0
    double kappa_TS  = 0.0;      // 若 >0, w_i ≤ kappa_TS * TS_i / median(TS)

    // 群組上限（group id -> cap）
    std::map<int,double> group_cap;

    // 成本（TWD 基準；P0 會把 |Δw| 轉為金額）
    double P0 = 1.0;             // 投資總額（TWD）
    double fee_buy  = 0.001425;  // 券商費率（買）
    double fee_sell = 0.001425;  // 券商費率（賣）；若要含 0.003 證交稅，請把它加到這裡
    double beta_imb_impact = 0.0;// 衝擊與 |Imb| 的聯動
    double gamma_init = 4e-4;    // 衝擊係數 γ_i 的常數起點；真正係數會 *sigma_GK / sqrt(ADV)

    // PWL 斷點
    PWL pwl;

    // 數值
    double eps = 1e-9;
};

// 多組掃參數
struct SocpSweep {
    std::vector<double> sigma_star_list {0.006,0.008,0.010,0.012};
    std::vector<double> tau_list        {0.10,0.15,0.20,0.25};
    std::vector<double> impact_scale    {0.7, 1.0, 1.3};  // 乘在 gamma 上
    // 去重
    double dedup_l1_radius = 0.02;      // 2% ℓ1 半徑
    int     k_keep = 30;                // 典型保留個數
};

struct PortfolioCandidate {
    Eigen::VectorXd w;              // N
    double obj = 0.0;               // r - costs（以 TWD/P0 比率）
    double ret_part = 0.0;
    double lin_cost = 0.0;
    double imp_cost = 0.0;
    double risk = 0.0;              // ∥Lᵀ w∥₂
    double turnover = 0.0;          // L1 or L2
    double sigma_star = 0.0;
    double tau_max = 0.0;
    double impact_scale = 1.0;
};

// 以 ECOS 解一次 SOCP
// 回傳空 optional 表示求解失敗（不可行或數值發散）
std::optional<PortfolioCandidate> solve_socp_once(const SocpProblem& pb);

// 掃參數格、去重，產生候選投組
std::vector<PortfolioCandidate> generate_socp_candidates(
    const SocpProblem& base,
    const SocpSweep& sweep);

