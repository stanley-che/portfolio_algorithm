#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
// --- Main project headers ---
// These should be in your include path
#include "data_loader.h"
#include "process.h"
/*#include "prediction.h"
#include "socp_generator.h"
#include "policy_solver.h"
#include "dsa_executor.h"
#include "types.h"*/ // Assuming you have a consolidated types.h

/**
 * @brief Prints a summary of a vector for quick inspection.
 */

void print_vector_summary(const Eigen::VectorXd& vec, const std::string& name) {
    if (vec.size() == 0) {
        std::cout << name << " (size 0)" << std::endl;
        return;
    }
    std::cout << name << " (size " << vec.size() << "): "
              << "Mean=" << vec.mean()
              << ", StdDev=" << std::sqrt((vec.array() - vec.mean()).square().sum() / (vec.size() - 1))
              << ", Min=" << vec.minCoeff()
              << ", Max=" << vec.maxCoeff() << std::endl;
}
void print_vector_debug(const Eigen::VectorXd& v, const std::string& name, int head=5) {
    std::cout << name << " (size=" << v.size() << "): ";
    for (int i=0; i<std::min(head, (int)v.size()); ++i) {
        std::cout << v(i) << " ";
    }
    std::cout << "\n";
}

void print_matrix_debug(const Eigen::MatrixXd& M, const std::string& name, int head=5) {
    std::cout << name << " (" << M.rows() << "x" << M.cols() << ")\n";
    if (M.rows() == 0 || M.cols() == 0) return;

    // 印第 0 行
    std::cout << "  Row[0] head: ";
    for (int j=0; j<std::min(head, (int)M.cols()); ++j) {
        std::cout << M(0,j) << " ";
    }
    std::cout << "\n";

    // 印第 t 行或最後一行
    int row_idx = std::min(5, (int)M.rows()-1);
    std::cout << "  Row[" << row_idx << "] head: ";
    for (int j=0; j<std::min(head, (int)M.cols()); ++j) {
        std::cout << M(row_idx,j) << " ";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "--- Two-Stage Portfolio Optimization Engine ---" << std::endl;
    std::cout << "---         Taiwan Equities Strategy        ---" << std::endl << std::endl;

    // =========================================================================
    // 0. Configuration Setup (Defaults from Section 8)
    // =========================================================================
    std::cout << "[0] Setting up configurations..." << std::endl;
    //data loader
    DataLoader loader("/mnt/e/backup_portfolio_dsa/portfolio_dsa/cpp_core/src");           // 原本是 "."
    #ifdef DATA_LOADER_TESTING
        if (!loader.TEST_readDailyPricesCsv("daily_60d.csv")) 
    #else
        if (!loader.loadDailyPanelAndBuildFeatures()) 
    #endif
    // 2) 讀檔 + 特徵（內部會讀 daily_60d.csv、brokers.csv、meta.csv）
    if (!loader.loadDailyPanelAndBuildFeatures()) {
        std::cerr << "[main] loadDailyPanelAndBuildFeatures failed.\n";
        return 1;
    }
    // 3) 用「前天交易日」
    const auto &dates = loader.dates();
    if (dates.size() < 2) {
        std::cerr << "[main] not enough dates.\n";
        return 1;
    }
    int t = (int)dates.size() - 2;  // 前天
    std::cout << "[main] using date = " << dates[t] << " (前天)\n";
    // ---------- 測 Risk ----------
    process::RiskConfig rcfg;
    rcfg.corr_half_life = 30;   // 相關性半衰期
    rcfg.sigma_source   = process::SigmaSource::GK; // 用 GK 波動
    rcfg.std_window     = 30;   // 若改用 std
    rcfg.c_liq          = 0.6;  // 流動性縮放強度
    rcfg.eps            = 1e-8;

    auto rsk = process::build_liquidity_scaled_risk(loader, t, rcfg);
    std::cout << "[TB] Sigma_tilde shape = " << rsk.Sigma_tilde.rows() << " x " << rsk.Sigma_tilde.cols() << "\n";
    std::cout << "[TB] sigma_i (first 5): ";
    for (int i=0;i<std::min<int>(5, rsk.sigma_i.size()); ++i) std::cout << std::setprecision(4) << rsk.sigma_i(i) << " ";
    std::cout << "\n";

    // ---------- 測 Stage-1 Forecast ----------
    process::ProcessingConfig pxcfg;
    pxcfg.mode = process::XSectionStandardize::ZScoreClip;
    //pxcfg.clip = 3.0;
    pxcfg.fill_missing_with_zero = true;

    process::ForecastConfig fcfg;
    fcfg.use_linear = true;     // 用線性 WLS
    fcfg.lookback   = 60;
    fcfg.half_life  = 20.0;     // WLS 半衰期（你的 stage1 會用）
std::cout << "\n[Debug] Parameters for stage1_forecast:\n";

// --- 日期與索引 ---
std::cout << "t = " << t 
          << ", date[t] = " << loader.dates()[t] 
          << ", dates.size() = " << loader.dates().size() << "\n";

// --- 股票 universe ---
std::cout << "Universe size (N) = " << loader.symbols().size() << "\n";
std::cout << "First 5 symbols: ";
for (int i=0; i<std::min<int>(5, loader.symbols().size()); ++i) {
    std::cout << loader.symbols()[i] << " ";
}
std::cout << "\n";

// --- ProcessingConfig ---
std::cout << "ProcessingConfig: mode=" 
          << (pxcfg.mode==process::XSectionStandardize::ZScoreClip ? "ZScoreClip" : "QuantileMap")
          << ", clip=" << pxcfg.clip
          << ", fill_missing_with_zero=" << pxcfg.fill_missing_with_zero
          << "\n";

// --- ForecastConfig ---
std::cout << "ForecastConfig: use_linear=" << fcfg.use_linear
          << ", lookback=" << fcfg.lookback
          << ", half_life=" << fcfg.half_life
          << "\n";

// --- 特徵矩陣的行數檢查 ---
std::cout << "Feature matrix sizes (rows x cols):\n";
std::cout << "  Gap="          << loader.feat_Gap().rows()          << "x" << loader.feat_Gap().cols() << "\n";
std::cout << "  Mom5="         << loader.feat_Mom5().rows()         << "x" << loader.feat_Mom5().cols() << "\n";
std::cout << "  Mom10="        << loader.feat_Mom10().rows()        << "x" << loader.feat_Mom10().cols() << "\n";
std::cout << "  Mom20="        << loader.feat_Mom20().rows()        << "x" << loader.feat_Mom20().cols() << "\n";
std::cout << "  BIAS="         << loader.feat_BIAS().rows()         << "x" << loader.feat_BIAS().cols() << "\n";
std::cout << "  BrokerStrength=" << loader.feat_BrokerStrength().rows() << "x" << loader.feat_BrokerStrength().cols() << "\n";
std::cout << "  Liquidity="    << loader.feat_Liquidity().rows()    << "x" << loader.feat_Liquidity().cols() << "\n";
std::cout << "  TurnoverShare="<< loader.feat_TurnoverShare().rows()<< "x" << loader.feat_TurnoverShare().cols() << "\n";
std::cout << "  Imbalance="    << loader.feat_Imbalance().rows()    << "x" << loader.feat_Imbalance().cols() << "\n";
std::cout << "  GKVol="        << loader.feat_GKVol().rows()        << "x" << loader.feat_GKVol().cols() << "\n";
std::cout << "  C (prices)="   << loader.C().rows()                 << "x" << loader.C().cols() << "\n";
std::cout << "\n[Debug] Stage1 Forecast Parameters and Data\n";
std::cout << "t = " << t << ", date = " << loader.dates()[t] 
          << ", total dates = " << loader.dates().size() << "\n";

std::cout << "Universe size = " << loader.symbols().size() << "\n";
std::cout << "First 5 symbols: ";
for (int i=0; i<std::min(5, (int)loader.symbols().size()); ++i)
    std::cout << loader.symbols()[i] << " ";
std::cout << "\n";

// ProcessingConfig
std::cout << "pxcfg: mode=" 
          << (pxcfg.mode==process::XSectionStandardize::ZScoreClip ? "ZScoreClip" : "QuantileMap")
          << ", clip=" << pxcfg.clip
          << ", fill_missing=" << pxcfg.fill_missing_with_zero << "\n";

// ForecastConfig
std::cout << "fcfg: use_linear=" << fcfg.use_linear
          << ", lookback=" << fcfg.lookback
          << ", half_life=" << fcfg.half_life << "\n";
std::cout << "\n[Debug] Stage1 Forecast Parameters and Data\n";
std::cout << "t = " << t << ", date = " << loader.dates()[t] 
          << ", total dates = " << loader.dates().size() << "\n";

std::cout << "Universe size = " << loader.symbols().size() << "\n";
std::cout << "First 5 symbols: ";
for (int i=0; i<std::min(5, (int)loader.symbols().size()); ++i)
    std::cout << loader.symbols()[i] << " ";
std::cout << "\n";

// ProcessingConfig
std::cout << "pxcfg: mode=" 
          << (pxcfg.mode==process::XSectionStandardize::ZScoreClip ? "ZScoreClip" : "QuantileMap")
          << ", clip=" << pxcfg.clip
          << ", fill_missing=" << pxcfg.fill_missing_with_zero << "\n";

// ForecastConfig
std::cout << "fcfg: use_linear=" << fcfg.use_linear
          << ", lookback=" << fcfg.lookback
          << ", half_life=" << fcfg.half_life << "\n";

// Features summary
print_matrix_debug(loader.feat_Gap(), "feat_Gap");
print_matrix_debug(loader.feat_Mom5(), "feat_Mom5");
print_matrix_debug(loader.feat_Mom10(), "feat_Mom10");
print_matrix_debug(loader.feat_Mom20(), "feat_Mom20");
print_matrix_debug(loader.feat_BIAS(), "feat_BIAS");
print_matrix_debug(loader.feat_BrokerStrength(), "feat_BrokerStrength");
print_matrix_debug(loader.feat_Liquidity(), "feat_Liquidity");
print_matrix_debug(loader.feat_TurnoverShare(), "feat_TurnoverShare");
print_matrix_debug(loader.feat_Imbalance(), "feat_Imbalance");
print_matrix_debug(loader.feat_GKVol(), "feat_GKVol");
print_matrix_debug(loader.C(), "ClosePrice (C)");



    //problem
    auto fo = process::stage1_forecast(loader, t, pxcfg, fcfg, std::nullopt);
    std::cout << "[TB] rhat_raw size = " << fo.rhat_raw.size() << "\n";
    if (fo.rhat_raw.size()>0) {
        double mean=0; int m=0;
        for (int i=0;i<fo.rhat_raw.size(); ++i) if (std::isfinite(fo.rhat_raw(i))) { mean+=fo.rhat_raw(i); ++m; }
        std::cout << "[TB] rhat_raw mean ≈ " << (m>0? mean/m : 0.0) << "\n";
        std::cout << "[TB] rhat_raw head: ";
  		std::cout << "[TB] process module OK."<< std::endl;
std::cout << "[TB] process module OK."<< std::endl;
        for (int i=0;i<std::min<int>(5, fo.rhat_raw.size()); ++i) std::cout << fo.rhat_raw(i) << " ";
        std::cout << "\n";
    }

    std::cout << "[TB] process module OK."<< std::endl;
    std::cout << "[TB] process module OK."<< std::endl;
    // Prediction
    /*TrainConfig train_cfg;
std::cout << "1.\n";
    train_cfg.half_life = 60.0;
std::cout << "2.\n";
    train_cfg.model = ModelType::Ridge;
std::cout << "3.\n";
    CalibrationConfig cal_cfg;
    EnhancementConfig enh_cfg;

    //blackswang 
    process::BlackSwanConfig bs_cfg;
	bs_cfg.m_default     = 1.0;
	bs_cfg.m_gap_bad     = 0.8;
	bs_cfg.m_vol_high    = 0.7;
	bs_cfg.m_imb_bad     = 0.9;
	bs_cfg.m_event       = 0.6;
	bs_cfg.gap_threshold = 0.01;
	bs_cfg.vol_ratio_th  = 1.5;
    bs_cfg.imb_median_th = 0.2;

    // SOCP
    SocpSweep sweep_cfg;
    sweep_cfg.k_keep = 20; // Target ~20 candidates

    // Policy
    PolicyLPConfig policy_cfg;
    policy_cfg.alpha = 0.95;

    // DSA
    dsa::Config dsa_cfg;
    dsa_cfg.P0 = 10'000'000.0; // Example: 10M TWD notional
    dsa_cfg.cool = 0.97;
    dsa_cfg.moves_per_T = 1500;
    dsa_cfg.lambdaB_start = 0.1;
    dsa_cfg.lambdaB_max = 10.0;


    // =========================================================================
    // 1. ETL -> Features -> Forecast -> Risk Matrix (Workflow Step 1)
    // =========================================================================
    std::cout << "\n[1] Loading data and building features..." << std::endl;

 
    const std::string today_str = loader.dates()[t];
    const int N = loader.symbols().size();

    std::cout << "Today's date: " << today_str << ", Universe size: " << N << std::endl;

    // --- Build Risk Matrix & Black-Swan m ---
    std::cout << "Building liquidity-scaled risk matrix..." << std::endl;
    process::RiskOutput risk = process::build_liquidity_scaled_risk(loader, t, rcfg);

    std::cout << "Calculating black-swan scaling factor 'm'..." << std::endl;
    process::BlackSwanOutput bs = process::black_swan_scale(loader, t, bs_cfg, false);
    std::cout << "Black-swan factor m = " << bs.m << std::endl;

    // --- Generate Enhanced Forecast ---
    std::cout << "Generating enhanced return forecast for Stage 1..." << std::endl;
    PredictionOutput prediction = predict_day(loader, t, TargetType::OC, train_cfg, pxcfg, cal_cfg, enh_cfg);
    print_vector_summary(prediction.rhat_enh, "r_hat_enhanced");


    // =========================================================================
    // 2. SOCP Sweep -> Generate K Candidates (Workflow Step 2)
    // =========================================================================
    std::cout << "\n[2] Generating candidate portfolios via SOCP sweep..." << std::endl;
    
    SocpProblem base_problem;
    base_problem.rhat_enh = prediction.rhat_enh;
    base_problem.L = risk.L;
    base_problem.w0 = Eigen::VectorXd::Zero(N); // Assume starting from cash; could be loaded
    base_problem.P0 = dsa_cfg.P0;
    
    // Populate problem with data from loader for costs and caps
    base_problem.ADV = loader.feat_Liquidity().row(t).transpose(); // Simplified, should be proper ADV
    base_problem.sigma_GK = loader.feat_GKVol().row(t).transpose();
    base_problem.Imb = loader.feat_Imbalance().row(t).transpose();
    base_problem.BF = loader.feat_BrokerStrength().row(t).transpose();
    base_problem.BIAS = loader.feat_BIAS().row(t).transpose();
    base_problem.TS = loader.feat_TurnoverShare().row(t).transpose();

    std::vector<PortfolioCandidate> candidates = generate_socp_candidates(base_problem, sweep_cfg);
    if (candidates.empty()) {
        std::cerr << "Fatal Error: SOCP candidate generation failed. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Generated " << candidates.size() << " candidate portfolios." << std::endl;


    // =========================================================================
    // 3. Build MDP -> Solve Dual LP -> Get Policy (Workflow Steps 3 & 4)
    // =========================================================================
    std::cout << "\n[3] Solving for optimal policy via MDP Dual LP..." << std::endl;

    // --- Placeholder MDP Data ---
    // In a real system, this would be constructed from historical backtest results (rewards)
    // and market regime data (states, transitions).
    PolicySamples mdp_samples;
    mdp_samples.S = 3; // e.g., {Volatile, Bullish, Bearish}
    mdp_samples.A = candidates.size();
    mdp_samples.M = 100; // 100 historical reward samples per (s,a) pair
    mdp_samples.samples.resize(mdp_samples.S, std::vector<std::vector<double>>(mdp_samples.A, std::vector<double>(mdp_samples.M, 0.0)));
    // Fill with dummy data: higher mean for better candidates
    for(int s=0; s<mdp_samples.S; ++s) {
        for (int a=0; a<mdp_samples.A; ++a) {
            for(int m=0; m<mdp_samples.M; ++m) {
                 mdp_samples.samples[s][a][m] = (candidates[a].ret_part - candidates[a].lin_cost - candidates[a].imp_cost) + (rand() % 1000 / 10000.0 - 0.005);
            }
        }
    }
    Transitions mdp_transitions;
    mdp_transitions.S = mdp_samples.S;
    mdp_transitions.A = mdp_samples.A;
    mdp_transitions.P.resize(mdp_transitions.S, std::vector<std::vector<double>>(mdp_transitions.A, std::vector<double>(mdp_transitions.S, 1.0/mdp_transitions.S))); // Uniform transitions
    // --- End Placeholder ---

    PolicyLPSolver policy_solver(mdp_samples, mdp_transitions, policy_cfg);
    PolicySolution solution = policy_solver.solve();

    if (!solution.solved) {
        std::cerr << "Warning: Policy LP solver failed. Falling back to a conservative portfolio." << std::endl;
        // Fallback logic: choose the candidate with the best Sharpe ratio or lowest risk.
        // For simplicity, we'll just exit.
        return 1;
    }
    Eigen::MatrixXd policy = solution.policy();
    std::cout << "Policy LP solved. Optimal policy matrix (S x A) computed." << std::endl;


    // =========================================================================
    // 4. Observe State -> Select Portfolio -> DSA (Workflow Step 5)
    // =========================================================================
    std::cout << "\n[4] Selecting portfolio and executing with Discrete Simulated Annealing..." << std::endl;

    // --- Observe current state and select portfolio ---
    int current_state = 0; // Placeholder: assume we are in state 0 ("Volatile")
    Eigen::VectorXd::Index best_action_idx;
    policy.row(current_state).maxCoeff(&best_action_idx);
    const PortfolioCandidate& chosen_candidate = candidates[best_action_idx];

    std::cout << "Current market state: " << current_state << ". Selected candidate #" << best_action_idx
              << " with risk=" << chosen_candidate.risk * 100 << "% and turnover=" << chosen_candidate.turnover * 100 << "%." << std::endl;
    
    // --- Setup DSA ---
    dsa_cfg.price = loader.C().row(t).transpose(); // Today's close prices
    dsa_cfg.lot = Eigen::VectorXi::Constant(N, 1000); // Assume all are 1000 shares/lot
    dsa_cfg.n0 = Eigen::VectorXi::Zero(N); // From cash
    dsa_cfg.r_enh = prediction.rhat_enh;
    dsa_cfg.Sigma_tilde = risk.Sigma_tilde;
    dsa_cfg.L = risk.L;
    dsa_cfg.m = bs.m;
    dsa_cfg.sigma_star = chosen_candidate.sigma_star;
    dsa_cfg.ADV = base_problem.ADV;
    dsa_cfg.sigma_GK = base_problem.sigma_GK;
    dsa_cfg.Imb = base_problem.Imb;
    dsa_cfg.BF = base_problem.BF;
    dsa_cfg.BIAS = base_problem.BIAS;
    dsa_cfg.TurnoverShare = base_problem.TS;

    // --- Run DSA ---
    dsa::Executor dsa_executor(dsa_cfg);
    dsa::Result dsa_result = dsa_executor.run(chosen_candidate.w);

    std::cout << "DSA finished. Best objective found: " << dsa_result.C_best << std::endl;
    print_vector_summary(dsa_result.w_best, "Final Weights");
    
    // =========================================================================
    // 5. Generate Orders & Report (Workflow Step 6)
    // =========================================================================
    std::cout << "\n[5] Final Portfolio and Orders:" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Final Portfolio Risk (||L^T w||): " << dsa_result.risk_best * 100 << "%" << std::endl;
    std::cout << "Budget Violation (|sum(w)-1|): " << dsa_result.budget_violation << std::endl;
    
    std::cout << "\n--- Generated Orders ---" << std::endl;
    int orders_count = 0;
    for (int i = 0; i < N; ++i) {
        int lot_diff = dsa_result.n_best(i) - dsa_cfg.n0(i);
        if (lot_diff != 0) {
            orders_count++;
            std::cout << (lot_diff > 0 ? "BUY ": "SELL")
                      << "\t" << loader.symbols()[i]
                      << "\t" << std::abs(lot_diff) << " lots"
                      << " (" << std::abs(lot_diff) * dsa_cfg.lot(i) << " shares)"
                      << "\t@ price " << dsa_cfg.price(i)
                      << std::endl;
        }
    }

    if (orders_count == 0) {
        std::cout << "No trades needed." << std::endl;
    }
*/
    std::cout << "\n--- End of Daily Workflow ---" << std::endl;

    return 0;
}