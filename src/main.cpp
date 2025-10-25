// main.cpp (SOCP 後立即輸出 Excel / CSV)

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include "data_loader.h"
#include "process.h"
#include "prediction.h"
#include "socp_generator.h"
#include "policy_solver.h"
#include "dsa_executor.h"
#include "types.h"

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

// ---- tiny helpers -----------------------------------------------------------
static void print_vector_summary(const Eigen::VectorXd& v, const std::string& name) {
    if (v.size() == 0) { std::cout << name << " (size 0)\n"; return; }
    double mean = v.mean();
    double var  = (v.array() - mean).square().sum() / std::max(1, (int)v.size() - 1);
    std::cout << name << " (n=" << v.size()
              << ") mean=" << mean
              << ", sd="  << std::sqrt(var)
              << ", min=" << v.minCoeff()
              << ", max=" << v.maxCoeff() << "\n";
}

int main() {
    std::cout << "--- Two-Stage Portfolio Optimization (TW Equities) ---\n\n";

    // CPU/threads 設定
    Eigen::setNbThreads(1);
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    // 輸出目錄（存在則忽略）
    ::mkdir("./advance_parameter_csv", 0755);

    // =========================================================================
    // 0) ETL
    // =========================================================================
    std::cout << "[0] Loading data & building features...\n";
    DataLoader loader("/mnt/c/backup_portfolio_dsa/portfolio_dsa/cpp_core/src");
    if (!loader.loadDailyPanelAndBuildFeatures()) {
        std::cerr << "[FATAL] loadDailyPanelAndBuildFeatures failed.\n";
        return 1;
    }
    const auto& dates = loader.dates();
    if (dates.size() < 2) { std::cerr << "[FATAL] Not enough dates.\n"; return 1; }
    int t = (int)dates.size() - 2;              // 使用「前天」資料
    const int N = (int)loader.symbols().size();
    std::cout << "[main] use date = " << dates[t] << ", N = " << N << "\n";

    // =========================================================================
    // 1) Configs
    // =========================================================================
    process::RiskConfig rcfg;
    rcfg.corr_half_life = 30;
    rcfg.sigma_source   = process::SigmaSource::GK;
    rcfg.std_window     = 30;
    rcfg.c_liq          = 0.6;
    rcfg.eps            = 1e-8;

    process::ProcessingConfig pxcfg;
    pxcfg.mode = process::XSectionStandardize::ZScoreClip;
    pxcfg.fill_missing_with_zero = true;

    process::ForecastConfig fcfg;
    fcfg.use_linear = true;
    fcfg.lookback   = 60;
    fcfg.half_life  = 20.0;

    process::BlackSwanConfig bs_cfg;
    bs_cfg.m_default     = 1.0;
    bs_cfg.m_gap_bad     = 0.8;
    bs_cfg.m_vol_high    = 0.7;
    bs_cfg.m_imb_bad     = 0.9;
    bs_cfg.m_event       = 0.6;
    bs_cfg.gap_threshold = 0.01;
    bs_cfg.vol_ratio_th  = 1.5;
    bs_cfg.imb_median_th = 0.2;

    process::CostConfig cost_cfg;  // 使用預設（可依市場修改）

    // 其他模組設定
    TrainConfig train_cfg; train_cfg.half_life = 60.0; train_cfg.model = ModelType::Ridge;
    CalibrationConfig cal_cfg;
    EnhancementConfig enh_cfg;

    // =========================================================================
    // 2) Risk & Black-Swan m & Stage-1 Forecast
    // =========================================================================
    std::cout << "[1] Build risk & stage-1 forecast...\n";
    process::RiskOutput risk = process::build_liquidity_scaled_risk(loader, t, rcfg);
    std::cout << "  Sigma_tilde: " << risk.Sigma_tilde.rows() << "x" << risk.Sigma_tilde.cols() << "\n";

    process::BlackSwanOutput bs = process::black_swan_scale(loader, t, bs_cfg, /*external_event=*/false);
    std::cout << "  Black-swan m = " << bs.m << "\n";

    auto fo = process::stage1_forecast(loader, t, pxcfg, fcfg, std::nullopt);
    print_vector_summary(fo.rhat_raw, "rhat_raw");

    // =========================================================================
    // 3) Enhanced forecast (pipeline 的版本)
    // =========================================================================
    std::cout << "[2] Generate enhanced forecast (predict_day)...\n";
    PredictionOutput prediction = predict_day(loader, t, TargetType::OC, train_cfg, pxcfg, cal_cfg, enh_cfg);
    print_vector_summary(prediction.rhat_enh, "rhat_enhanced");

    // =========================================================================
    // 4) SOCP → 候選組合
    // =========================================================================
    std::cout << "[3] SOCP sweep (generate candidates)...\n";
    SocpSweep sweep_cfg; sweep_cfg.k_keep = 200;

    SocpProblem base_problem;
    base_problem.rhat_enh = prediction.rhat_enh;
    base_problem.L        = risk.L;
    base_problem.w0       = Eigen::VectorXd::Zero(N);
    base_problem.P0       = 10'000.0;  // 用在報表/DSA 的基準資金
    // 這裡的幾個欄位只作為 sweep/DSA 的額外資訊（實務上請以正確 ADV/成本欄位填入）
    base_problem.ADV      = loader.feat_Liquidity().row(t).transpose();
    base_problem.sigma_GK = loader.feat_GKVol().row(t).transpose();
    base_problem.Imb      = loader.feat_Imbalance().row(t).transpose();
    base_problem.BF       = loader.feat_BrokerStrength().row(t).transpose();
    base_problem.BIAS     = loader.feat_BIAS().row(t).transpose();
    base_problem.TS       = loader.feat_TurnoverShare().row(t).transpose();

    std::vector<PortfolioCandidate> candidates = generate_socp_candidates(base_problem, sweep_cfg);
    if (candidates.empty()) { std::cerr << "[FATAL] SOCP candidate generation failed.\n"; return 1; }
    std::cout << "  candidates = " << candidates.size() << "\n";

    // =========================================================================
    // 5) SOCP 之後、DSA 之前：輸出 Excel / CSV 報表
    //    選「淨期望 = ret_part - lin_cost - imp_cost」最大者
    // =========================================================================
    int best_idx = 0;
    double best_score = -1e100;
    for (int i = 0; i < (int)candidates.size(); ++i) {
        double score = candidates[i].ret_part - candidates[i].lin_cost - candidates[i].imp_cost;
        if (score > best_score) { best_score = score; best_idx = i; }
    }
    const PortfolioCandidate& chosen_socp = candidates[best_idx];
    std::cout << "[Report] Chosen SOCP cand #" << best_idx
              << " (net=" << best_score << ", risk=" << chosen_socp.risk*100
              << "%, turnover=" << chosen_socp.turnover*100 << "%)\n";

    // 由 SOCP 權重產生報表所需的 A / side
    Eigen::VectorXd A_socp  = base_problem.P0 * chosen_socp.w.cwiseAbs(); // 單邊金額
    Eigen::VectorXi side_socp = Eigen::VectorXi::Zero(N);
    for (int i = 0; i < N; ++i) {
        side_socp(i) = (chosen_socp.w(i) > 1e-12) ? +1
                     : (chosen_socp.w(i) < -1e-12) ? -1 : 0;
    }

    // 匯出 Excel/CSV（trade plan）
    std::string out_path =
#ifdef USE_XLSX
        "./advance_parameter_csv/trade_plan.xlsx";
#else
        "./advance_parameter_csv/trade_plan.csv";
#endif

    process::dump_trade_plan_excel(out_path, loader, /*t=*/t,
                                   A_socp, side_socp, base_problem.P0);
    std::cout << "[Report] Trade plan written to " << out_path << "\n";

    // 匯出分析明細（xlsx 或 CSV 後備）
#ifdef USE_XLSX
    process::dump_to_excel("./advance_parameter_csv/advance_parameter.xlsx",
                           loader, t, pxcfg, rcfg, cost_cfg, bs_cfg, fcfg,
                           A_socp, side_socp, std::nullopt);
    std::cout << "[Report] Wrote advance_parameter.xlsx (SOCP)\n";
#else
    process::dump_to_excel_csv_fallback("./advance_parameter_csv",
                           loader, t, pxcfg, rcfg, cost_cfg, bs_cfg, fcfg,
                           A_socp, side_socp, std::nullopt);
    std::cout << "[Report] Wrote CSVs under ./advance_parameter_csv (SOCP)\n";
#endif

    // =========================================================================
    // 6) （可選）MDP → DSA
    // =========================================================================
    std::cout << "[4] Solve policy via MDP Dual LP...\n";
    PolicySamples mdp_samples;
    mdp_samples.S = 3;  // e.g. {Volatile, Bullish, Bearish}
    mdp_samples.A = (int)candidates.size();
    mdp_samples.M = 100;
    mdp_samples.samples.resize(mdp_samples.S,
        std::vector<std::vector<double>>(mdp_samples.A, std::vector<double>(mdp_samples.M, 0.0)));
    for (int s=0; s<mdp_samples.S; ++s)
        for (int a=0; a<mdp_samples.A; ++a)
            for (int m=0; m<mdp_samples.M; ++m)
                mdp_samples.samples[s][a][m] =
                    (candidates[a].ret_part - candidates[a].lin_cost - candidates[a].imp_cost)
                    + (rand() % 1000 / 10000000.0 - 0.005);

    Transitions trans; trans.S = mdp_samples.S; trans.A = mdp_samples.A;
    trans.P.resize(trans.S, std::vector<std::vector<double>>(trans.A, std::vector<double>(trans.S, 1.0/trans.S)));
    PolicyLPConfig plc; plc.alpha = 0.95;
    PolicyLPSolver solver(mdp_samples, trans, plc);
    PolicySolution sol = solver.solve();
    if (!sol.solved) { std::cerr << "[WARN] Policy LP failed.\n"; return 1; }
    Eigen::MatrixXd policy = sol.policy();

    // =========================================================================
    // 7) 觀測狀態 → 選 action → DSA
    // =========================================================================
    std::cout << "[5] Pick candidate & run DSA...\n";
    int current_state = 0;
    Eigen::VectorXd::Index best_a;
    policy.row(current_state).maxCoeff(&best_a);
    const PortfolioCandidate& chosen = candidates[best_a];
    std::cout << "  choose a = " << best_a
              << ", risk=" << chosen.risk*100 << "%, turnover=" << chosen.turnover*100 << "%\n";

    dsa::Config dsa_cfg;
    dsa_cfg.P0          = base_problem.P0;
    dsa_cfg.cool        = 0.97;
    dsa_cfg.moves_per_T = 1500;
    dsa_cfg.lambdaB_start = 0.1;
    dsa_cfg.lambdaB_max   = 10.0;

    dsa_cfg.price   = loader.C().row(t).transpose();
    dsa_cfg.lot     = Eigen::VectorXi::Constant(N, 1000);
    dsa_cfg.n0      = Eigen::VectorXi::Zero(N);
    dsa_cfg.r_enh   = prediction.rhat_enh;
    dsa_cfg.Sigma_tilde = risk.Sigma_tilde;
    dsa_cfg.L       = risk.L;
    dsa_cfg.m       = bs.m;
    dsa_cfg.sigma_star = chosen.sigma_star;
    dsa_cfg.ADV        = loader.feat_Liquidity().row(t).transpose();
    dsa_cfg.sigma_GK   = loader.feat_GKVol().row(t).transpose();
    dsa_cfg.Imb        = loader.feat_Imbalance().row(t).transpose();
    dsa_cfg.BF         = loader.feat_BrokerStrength().row(t).transpose();
    dsa_cfg.BIAS       = loader.feat_BIAS().row(t).transpose();
    dsa_cfg.TurnoverShare = loader.feat_TurnoverShare().row(t).transpose();

    dsa::Executor ex(dsa_cfg);
    dsa::Result   dr = ex.run(chosen.w);

    std::cout << "  DSA best objective: " << dr.C_best << "\n";
    std::cout << "  Final risk (||L^T w||): " << dr.risk_best*100 << "%\n";
    std::cout << "  Budget violation: " << dr.budget_violation << "\n";

    // =========================================================================
    // 8) 列印訂單（簡化）
    // =========================================================================
    std::cout << "\n--- Orders ---\n";
    Eigen::VectorXi dn = dr.n_best - dsa_cfg.n0;
    int orders = 0;
    for (int i=0; i<N; ++i) {
        int lot_diff = dn(i);
        if (lot_diff == 0) continue;
        ++orders;
        std::cout << (lot_diff>0 ? "BUY " : "SELL") << "\t"
                  << loader.symbols()[i] << "\t"
                  << std::abs(lot_diff) << " lots"
                  << " (" << std::abs(lot_diff) * dsa_cfg.lot(i) << " sh)"
                  << "\t@ " << dsa_cfg.price(i) << "\n";
    }
    if (orders == 0) std::cout << "No trades.\n";

    std::cout << "\n--- End of Daily Workflow ---\n";
    return 0;
}

