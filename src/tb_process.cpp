// tb_process.cpp — 單獨測試 process 模組
#include "data_loader.h"
#include "process.h"
#include <iostream>
#include <iomanip>

int main() {
    // 路徑請指到有 daily_60d.csv 的資料夾
    DataLoader loader("/mnt/e/backup_portfolio_dsa/portfolio_dsa/cpp_core/src");


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
    pxcfg.clip = 3.0;
    pxcfg.fill_missing_with_zero = true;

    process::ForecastConfig fcfg;
    fcfg.use_linear = true;     // 用線性 WLS
    fcfg.lookback   = 60;
    fcfg.half_life  = 20.0;     // WLS 半衰期（你的 stage1 會用）

    auto fo = process::stage1_forecast(loader, t, pxcfg, fcfg, std::nullopt);
    std::cout << "[TB] rhat_raw size = " << fo.rhat_raw.size() << "\n";
    if (fo.rhat_raw.size()>0) {
        double mean=0; int m=0;
        for (int i=0;i<fo.rhat_raw.size(); ++i) if (std::isfinite(fo.rhat_raw(i))) { mean+=fo.rhat_raw(i); ++m; }
        std::cout << "[TB] rhat_raw mean ≈ " << (m>0? mean/m : 0.0) << "\n";
        std::cout << "[TB] rhat_raw head: ";
        for (int i=0;i<std::min<int>(5, fo.rhat_raw.size()); ++i) std::cout << fo.rhat_raw(i) << " ";
        std::cout << "\n";
    }

    std::cout << "[TB] process module OK.\n";
    
    return 0;
}

