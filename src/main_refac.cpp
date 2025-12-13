// main_refac.cpp
/*
g++ -std=c++17 -O2 main_refac.cpp -o run \
  -I/usr/include/eigen3 \
  -I$HOME/scs/include \
  -L$HOME/scs/build/lib \
  -lcurl -lscs -lm -lpthread \
  -fopenmp
LD_LIBRARY_PATH=$HOME/scs/build/lib:$LD_LIBRARY_PATH ./run
*/
#include <curl/curl.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <optional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "twse_quotes.hpp"
#include "twse_meta.hpp"
#include "broker.hpp"
#include "data_loader_all.hpp"
#include "process_all.hpp"
#include "prediction_all.hpp"
#include "socp_all.hpp"

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

static int pick_best_candidate_net(const std::vector<socp::PortfolioCandidate>& cands) {
    int best = 0;
    double best_score = -1e100;
    for (int i = 0; i < (int)cands.size(); ++i) {
        double net = cands[i].ret_part - cands[i].lin_cost - cands[i].imp_cost;
        if (net > best_score) { best_score = net; best = i; }
    }
    return best;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::cout << "--- Daily Workflow (TWSE ETL -> Features -> Predict -> SOCP -> Report) ---\n";

    // ===================== threads =====================
    Eigen::setNbThreads(1);
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    // ===================== paths =====================
    const std::string daily = "daily_60d.csv";
    const std::string base_path = ".";
    std::filesystem::create_directories("./OUT");
    std::filesystem::create_directories("./advance_parameter_csv");

    // ===================== 0) ETL =====================
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // 1) 抓 TWSE daily quotes -> daily_60d.csv
    {
        twse::DumpOptions opt;
        opt.days = 60;
        opt.throttle_ms = 120;
        opt.months_back = 4;

        twse::TwseQuoteDumper dumper(daily);
        if (!dumper.dump_recent_days(opt)) {
            std::cerr << "failed to fetch TWSE quotes -> " << daily << "\n";
            curl_global_cleanup();
            return 1;
        }
        std::cout << "Done. Wrote " << daily << "\n";
    }

    // 2) 由 daily_60d.csv 產 meta.csv
    {
        meta::TwseMetaBuilder b;
        meta::MetaOptions opt;
        if (!b.build_meta_csv_from_daily(daily, "meta.csv", opt)) {
            std::cerr << "failed to build meta.csv from " << daily << "\n";
            curl_global_cleanup();
            return 1;
        }
        std::cout << "Done. Wrote meta.csv\n";
    }

    // 3) 用 daily_60d.csv 抓 brokers.csv
    {
        BrokerFetcher fetcher(daily);
        if (!fetcher.run()) {
            std::cerr << "failed to fetch brokers from T86 using " << daily << "\n";
            curl_global_cleanup();
            return 1;
        }
        std::cout << "Done. Wrote brokers.csv\n";
    }

    curl_global_cleanup();

    // ===================== 1) Load & build features =====================
    dlx::DataLoader dl(base_path);
    dl.config().winsorize_p = 0.02;

    if (!dl.loadDailyPanelAndBuildFeatures()) {
        std::cerr << "DataLoader build features failed.\n";
        return 1;
    }

    const int T = (int)dl.C().rows();
    const int N = (int)dl.symbols().size();
    if (T < 3) { std::cerr << "[FATAL] Not enough dates.\n"; return 1; }

    // ✅ 用「前天」資料（跟你舊 main 一樣）
    const int t = std::max(0, T - 2);

    std::cout << "[main] T=" << T << ", use t=" << t << ", N=" << N << "\n";
    std::cout << "feat_Mom10 shape: " << dl.feat_Mom10().rows()
              << " x " << dl.feat_Mom10().cols() << "\n";

    // ===================== 2) Configs =====================
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

    process::CostConfig cost_cfg;  // default

    // prediction configs
    prediction::TrainConfig tr;
    tr.half_life = 60.0;
    tr.model = prediction::ModelType::Ridge;

    prediction::CalibrationConfig cal_cfg;
    prediction::EnhancementConfig enh_cfg;

    // ===================== 3) Risk & Black Swan & Stage1 forecast =====================
    std::cout << "[1] Build risk & black-swan...\n";

    // ✅ FIX: 用 engine member functions（不要用 process::xxx free function）
    process::ProcessEngineForecast eng(dl);

    auto risk = eng.build_liquidity_scaled_risk(t, rcfg);
    std::cout << "  Sigma_tilde: " << risk.Sigma_tilde.rows() << "x" << risk.Sigma_tilde.cols() << "\n";

    auto bs = eng.black_swan_scale(t, bs_cfg, /*external_event=*/false);
    std::cout << "  Black-swan m = " << bs.m << "\n";

    auto fo = eng.stage1_forecast(t, pxcfg, fcfg, std::nullopt);
    print_vector_summary(fo.rhat_raw, "rhat_raw(stage1)");

    // ===================== 4) Enhanced forecast (predict_day) =====================
    std::cout << "[2] predict_day -> rhat_enh...\n";

    // ✅ FIX: Predictor ctor / API 已改成綁定 dl
    prediction::Predictor pred(dl, tr, pxcfg, cal_cfg, enh_cfg);
    prediction::PredictionOutput pred_out = pred.predict_day(t, prediction::TargetType::OC);
    print_vector_summary(pred_out.rhat_enh, "rhat_enh");

    // ===================== 5) SOCP sweep -> candidates =====================
    std::cout << "[3] SOCP sweep...\n";

    socp::SocpSweep sweep;
    sweep.k_keep = 200;

    socp::SocpProblem pb;
    pb.rhat_enh = pred_out.rhat_enh;
    pb.L        = risk.L;
    pb.w0       = Eigen::VectorXd::Zero(N);
    pb.W_turn   = Eigen::VectorXd::Ones(N);

    pb.P0 = 10'000.0;

    pb.ADV      = dl.feat_Liquidity().row(t).transpose();
    pb.sigma_GK = dl.feat_GKVol().row(t).transpose();
    pb.Imb      = dl.feat_Imbalance().row(t).transpose();
    pb.BF       = dl.feat_BrokerStrength().row(t).transpose();
    pb.BIAS     = dl.feat_BIAS().row(t).transpose();
    pb.TS       = dl.feat_TurnoverShare().row(t).transpose();

    pb.group     = Eigen::VectorXi::Constant(N, -1);
    pb.m         = bs.m;
    pb.turn_norm = socp::TurnoverNorm::L1;

    socp::SolverConfig scfg;
    scfg.debug_on = true;
    scfg.debug_relax = false;
    scfg.scs.verbose = true;

    socp::SocpCandidateGenerator gen(scfg);
    std::vector<socp::PortfolioCandidate> cands = gen.generate(pb, sweep);

    if (cands.empty()) {
        std::cerr << "[FATAL] SOCP candidate generation failed.\n";
        return 1;
    }
    std::cout << "  candidates = " << cands.size() << "\n";

    // ===================== 6) pick best SOCP & report =====================
    const int best_idx = pick_best_candidate_net(cands);
    const auto& chosen = cands[best_idx];
    const double best_net = chosen.ret_part - chosen.lin_cost - chosen.imp_cost;

    std::cout << "[Report] chosen SOCP #" << best_idx
              << " net=" << best_net
              << " risk=" << chosen.risk * 100 << "%"
              << " turnover=" << chosen.turnover * 100 << "%"
              << " (sigma*=" << chosen.sigma_star
              << ", tau=" << chosen.tau_max << ")\n";

    Eigen::VectorXd A_socp = pb.P0 * chosen.w.cwiseAbs();
    Eigen::VectorXi side_socp = Eigen::VectorXi::Zero(N);
    for (int i = 0; i < N; ++i) {
        side_socp(i) = (chosen.w(i) > 1e-12) ? +1
                     : (chosen.w(i) < -1e-12) ? -1 : 0;
    }

    // ✅ 立刻輸出報表
    {
        process::ProcessExporter pe(dl);

        process::DumpExcelConfig out;
        out.xlsx_path   = "./advance_parameter_csv/advance_parameter.xlsx";
        out.out_dir_csv = "./advance_parameter_csv";
        out.prefer_xlsx = true;

        std::optional<Eigen::MatrixXd> next_day_returns = std::nullopt;

        pe.dump_to_excel(out, t,
                         pxcfg, rcfg, cost_cfg, bs_cfg, fcfg,
                         A_socp, side_socp, next_day_returns);

        pe.dump_trade_plan_excel("./advance_parameter_csv/trade_plan.xlsx",
                                 t, A_socp, side_socp, pb.P0);

        std::cout << "[Report] Wrote ./advance_parameter_csv/advance_parameter.xlsx\n";
        std::cout << "[Report] Wrote ./advance_parameter_csv/trade_plan.xlsx\n";
    }

    std::cout << "\n--- End of Daily Workflow ---\n";
    return 0;
}
