// main_refac.cpp
/*
g++ -std=c++17 -O2 main_refac.cpp -o run \
  -DUSE_XLSX \
  -I/usr/include/eigen3 \
  -I$HOME/scs/include \
  -L$HOME/scs/build/lib \
  -lcurl -lm -lpthread -fopenmp \
  -lscsdir -lscsindir \
  -lxlsxwriter -lz


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

// ✅ prescan 需要
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

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
    const std::string base_path = ".";
    std::filesystem::create_directories("./OUT");
    std::filesystem::create_directories("./advance_parameter_csv");

    // ===================== 0) ETL =====================
    namespace fs = std::filesystem;

    auto file_ok = [&](const fs::path& p)->bool{
        std::error_code ec;
        if (!fs::exists(p, ec)) return false;
        if (!fs::is_regular_file(p, ec)) return false;
        auto sz = fs::file_size(p, ec);
        return (!ec && sz > 0);
    };

    auto pick_existing = [&](const std::vector<std::string>& cands,
                             const std::string& fallback)->std::string {
        for (const auto& c : cands) {
            if (file_ok(fs::path(c))) return c;
        }
        return fallback;
    };

    const std::string meta_path    = "meta.csv";
    const std::string brokers_path = "brokers.csv";

    // ✅ 自動吃 DAILY_60D.CSV / daily_60d.csv
    std::string daily = pick_existing(
        { "daily_60d.csv", "DAILY_60D.CSV", "daily_60d.CSV", "DAILY_60D.csv" },
        "daily_60d.csv"
    );

    std::cout << "[cwd] " << fs::current_path() << "\n";
    std::cout << "[ETL] daily file chosen: " << daily << "\n";

    curl_global_init(CURL_GLOBAL_DEFAULT);

    // 1) 抓 TWSE daily quotes -> daily（若已存在則跳過）
    if (file_ok(daily)) {
        std::cout << "[ETL] Found existing " << daily << " -> skip TWSE quotes fetch.\n";
    } else {
        twse::DumpOptions opt;
        opt.days = 60;
        opt.throttle_ms = 120;
        opt.months_back = 4;

        const std::string out_daily = "daily_60d.csv";
        twse::TwseQuoteDumper dumper(out_daily);
        if (!dumper.dump_recent_days(opt)) {
            std::cerr << "failed to fetch TWSE quotes -> " << out_daily << "\n";
            curl_global_cleanup();
            return 1;
        }
        daily = out_daily;
        std::cout << "Done. Wrote " << daily << "\n";
    }

    // 2) meta.csv（若已存在則跳過）
    if (file_ok(meta_path)) {
        std::cout << "[ETL] Found existing " << meta_path << " -> skip meta build.\n";
    } else {
        meta::TwseMetaBuilder b;
        meta::MetaOptions opt;
        if (!b.build_meta_csv_from_daily(daily, meta_path, opt)) {
            std::cerr << "failed to build meta.csv from " << daily << "\n";
            curl_global_cleanup();
            return 1;
        }
        std::cout << "Done. Wrote " << meta_path << "\n";
    }

    // 3) brokers.csv（若已存在則跳過）
    if (file_ok(brokers_path)) {
        std::cout << "[ETL] Found existing " << brokers_path << " -> skip brokers fetch.\n";
    } else {
        BrokerFetcher fetcher(daily);
        if (!fetcher.run()) {
            std::cerr << "failed to fetch brokers from T86 using " << daily << "\n";
            curl_global_cleanup();
            return 1;
        }
        std::cout << "Done. Wrote " << brokers_path << "\n";
    }

    curl_global_cleanup();

    // ===== optional prescan (只印資訊，不影響流程) =====
    auto prescan_daily = [&](const std::string& path, int max_lines = 2'000'000) {
        std::ifstream fin(path);
        if (!fin) {
            std::cerr << "[FATAL] cannot open " << path << "\n";
            std::exit(1);
        }
        std::string line;
        std::getline(fin, line); // skip header

        std::unordered_set<std::string> sym;
        std::unordered_set<std::string> date;
        sym.reserve(3000);
        date.reserve(500);

        int lines = 0;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            if (++lines > max_lines) break;

            std::stringstream ss(line);
            std::string c0, c1;
            if (!std::getline(ss, c0, ',')) continue; // date
            if (!std::getline(ss, c1, ',')) continue; // symbol

            date.insert(c0);
            sym.insert(c1);
        }

            std::cout << "[PreScan] daily unique dates=" << date.size()
                      << ", unique symbols=" << sym.size()
                      << ", scanned lines=" << lines << "\n";

            return std::pair<int,int>((int)date.size(), (int)sym.size());
        };

        auto [T_est, N_est] = prescan_daily(daily);
        (void)T_est; (void)N_est;
        const int MAX_SYMBOLS = 1070;   // 你可以調 400 / 500 / 800

    if (N_est > MAX_SYMBOLS) {
        std::cerr << "[WARN] Universe too large: N=" << N_est
                  << " > MAX_SYMBOLS=" << MAX_SYMBOLS
                  << " -> will cap in DataLoader by max_symbols.\n";
    }



    // ===================== 1) Load & build features =====================
    dlx::DataLoader dl(base_path);
    dl.config().winsorize_p = 0.02;
    dl.config().max_symbols = 600;
    if (!dl.loadDailyPanelAndBuildFeatures()) {
        std::cerr << "[FATAL] DataLoader build features failed.\n";
        return 1;
    }

    // ✅ sanity checks: 防止 segfault
    auto check_mat = [&](const Eigen::MatrixXd& M, const char* name) {
        if (M.rows() == 0 || M.cols() == 0) {
            std::cerr << "[FATAL] " << name << " is empty: "
                      << M.rows() << "x" << M.cols() << "\n";
            std::exit(1);
        }
    };
    auto check_same_shape = [&](const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                const char* a, const char* b) {
        if (A.rows() != B.rows() || A.cols() != B.cols()) {
            std::cerr << "[FATAL] shape mismatch: " << a << "="
                      << A.rows() << "x" << A.cols()
                      << " vs " << b << "="
                      << B.rows() << "x" << B.cols() << "\n";
            std::exit(1);
        }
    };

    check_mat(dl.C(), "C");
    check_mat(dl.O(), "O");
    check_mat(dl.feat_Mom10(), "feat_Mom10");
    check_mat(dl.feat_Liquidity(), "feat_Liquidity");
    check_mat(dl.feat_GKVol(), "feat_GKVol");
    check_mat(dl.feat_Imbalance(), "feat_Imbalance");
    check_mat(dl.feat_BrokerStrength(), "feat_BrokerStrength");
    check_mat(dl.feat_BIAS(), "feat_BIAS");
    check_mat(dl.feat_TurnoverShare(), "feat_TurnoverShare");

    check_same_shape(dl.feat_Mom10(), dl.C(), "feat_Mom10", "C");
    check_same_shape(dl.feat_Liquidity(), dl.C(), "feat_Liquidity", "C");
    check_same_shape(dl.feat_GKVol(), dl.C(), "feat_GKVol", "C");
    check_same_shape(dl.feat_Imbalance(), dl.C(), "feat_Imbalance", "C");
    check_same_shape(dl.feat_BrokerStrength(), dl.C(), "feat_BrokerStrength", "C");
    check_same_shape(dl.feat_BIAS(), dl.C(), "feat_BIAS", "C");
    check_same_shape(dl.feat_TurnoverShare(), dl.C(), "feat_TurnoverShare", "C");

    const int T = (int)dl.C().rows();
    const int N = (int)dl.C().cols();

    if ((int)dl.symbols().size() != N) {
        std::cerr << "[FATAL] symbols.size() != N: symbols="
                  << dl.symbols().size() << " vs N=" << N << "\n";
        return 1;
    }
    if (T < 3) {
        std::cerr << "[FATAL] Not enough dates: T=" << T << "\n";
        return 1;
    }

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

    prediction::TrainConfig tr;
    tr.half_life = 60.0;
    tr.model = prediction::ModelType::Ridge;

    prediction::CalibrationConfig cal_cfg;
    prediction::EnhancementConfig enh_cfg;

    // ===================== 3) Risk & Black Swan & Stage1 forecast =====================
    std::cout << "[1] Build risk & black-swan...\n";
    process::ProcessEngineForecast eng(dl);

    auto risk = eng.build_liquidity_scaled_risk(t, rcfg);
    std::cout << "  Sigma_tilde: " << risk.Sigma_tilde.rows() << "x" << risk.Sigma_tilde.cols() << "\n";

    auto bs = eng.black_swan_scale(t, bs_cfg, /*external_event=*/false);
    std::cout << "  Black-swan m = " << bs.m << "\n";

    auto fo = eng.stage1_forecast(t, pxcfg, fcfg, std::nullopt);
    print_vector_summary(fo.rhat_raw, "rhat_raw(stage1)");

    // ===================== 4) Enhanced forecast (predict_day) =====================
    std::cout << "[2] predict_day -> rhat_enh...\n";
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
    pb.P0       = 10'000.0;

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
        // ✅ DebugReporter（一定要宣告）
    socp::DebugReporter dbg(scfg.debug_on);

    // 在 solve 前先檢查輸入（rhat/L/w0）
    dbg.check_inputs(pb.rhat_enh, pb.L, pb.w0);

    // ===================== 5) SOCP sweep -> candidates =====================
    std::cout << "[3] SOCP sweep...\n";

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

    // --- 把 w 轉成報表要用的 A / side ---
    Eigen::VectorXd A_socp = pb.P0 * chosen.w.cwiseAbs();
    Eigen::VectorXi side_socp = Eigen::VectorXi::Zero(N);
    for (int i = 0; i < N; ++i) {
        side_socp(i) = (chosen.w(i) > 1e-12) ? +1
                     : (chosen.w(i) < -1e-12) ? -1 : 0;
    }

    // ✅ quick_feas_sanity：先用「簡化版」讓它能跑
    // wmax：先用你 pb 的上限（如果你有 pb.wmax_base 就用它；沒有就先給 5%）
    Eigen::VectorXd wmax = Eigen::VectorXd::Constant(N, 0.05);

    // group_cap：你目前沒做 group，就給空 map
    std::map<int,double> group_cap;

    // tau_max：用 chosen 的
    double tau_max = chosen.tau_max;

    // W_turn：你有 pb.W_turn
    const Eigen::VectorXd& W_turn = pb.W_turn;

    // budget：如果你 long-only budget = 1.0（或 pb.budget_long_only）
    double budget = 1.0;

    dbg.quick_feas_sanity(wmax, group_cap, pb.group, tau_max, W_turn, budget);

    // ✅ 寫檔前防呆：避免 NaN/Inf 把 xlsx 寫壞
    auto has_bad = [&](const Eigen::VectorXd& v){
        for (int i=0;i<v.size();++i) if (!std::isfinite(v(i))) return true;
        return false;
    };
    if (has_bad(A_socp) || !std::isfinite(best_net)) {
        std::cerr << "[WARN] bad numbers detected (NaN/Inf). Skip XLSX, write CSV only.\n";
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

        // CSV 一定寫（最穩）
        pe.dump_trade_plan_csv("./advance_parameter_csv/trade_plan.csv",
                               t, A_socp, side_socp, pb.P0);

        // XLSX 只有在資料乾淨時才寫
        if (!has_bad(A_socp) && std::isfinite(best_net)) {
            pe.dump_trade_plan_excel("./advance_parameter_csv/trade_plan.xlsx",
                                     t, A_socp, side_socp, pb.P0);
            std::cout << "[Report] Wrote ./advance_parameter_csv/trade_plan.xlsx\n";
        } else {
            std::cout << "[Report] Skip trade_plan.xlsx due to NaN/Inf. Use CSV.\n";
        }

        std::cout << "[Report] Wrote ./advance_parameter_csv/advance_parameter.xlsx\n";
        std::cout << "[Report] Wrote ./advance_parameter_csv/trade_plan.csv\n";
    }


    std::cout << "\n--- End of Daily Workflow ---\n";
    return 0;
}
