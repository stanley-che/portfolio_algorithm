// tests/test_process_math_40cases.cpp
/*
g++ -std=c++17 \
  -Isrc/revise \
  -Ithird_party/third_party/googletest/googletest \
  -Ithird_party/third_party/googletest/googletest/include \
  -Ithird_party/scs/include \
  -I/usr/include/eigen3 \
  test/test_socp_40cases.cpp \
  third_party/third_party/googletest/googletest/src/gtest-all.cc \
  third_party/third_party/googletest/googletest/src/gtest_main.cc \
  -Lthird_party/scs/out -lscsdir \
  -lpthread -lm \
  -o test_all
./test_all
*/
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include "process_math.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <optional>

#include "data_loader.hpp"
#include "process_export.hpp"
using namespace process::detail;
namespace fs = std::filesystem;

// ============================================================
// safe_div (6)
// ============================================================

TEST(ProcessMath_safe_div, NormalDivision) {
    EXPECT_DOUBLE_EQ(safe_div(10, 2), 5);
}

TEST(ProcessMath_safe_div, DivideByZero) {
    EXPECT_DOUBLE_EQ(safe_div(10, 0), 0);
}

TEST(ProcessMath_safe_div, NearZeroDenominator) {
    EXPECT_DOUBLE_EQ(safe_div(10, 1e-13), 0);
}

TEST(ProcessMath_safe_div, ZeroNumerator) {
    EXPECT_DOUBLE_EQ(safe_div(0, 10), 0);
}

TEST(ProcessMath_safe_div, NegativeDivision) {
    EXPECT_DOUBLE_EQ(safe_div(-10, 2), -5);
}

TEST(ProcessMath_safe_div, BothNegative) {
    EXPECT_DOUBLE_EQ(safe_div(-10, -2), 5);
}

// ============================================================
// alpha_from_half_life (6)
// ============================================================

TEST(ProcessMath_alpha, HalfLifeOne) {
    EXPECT_NEAR(alpha_from_half_life(1), 0.5, 1e-12);
}

TEST(ProcessMath_alpha, HalfLifeLarge) {
    EXPECT_NEAR(alpha_from_half_life(1e6), 0.0, 1e-6);
}

TEST(ProcessMath_alpha, HalfLifeZeroSafe) {
    EXPECT_TRUE(std::isfinite(alpha_from_half_life(0)));
}

TEST(ProcessMath_alpha, HalfLifeNegativeSafe) {
    EXPECT_TRUE(std::isfinite(alpha_from_half_life(-10)));
}

TEST(ProcessMath_alpha, AlphaInRange) {
    double a = alpha_from_half_life(10);
    EXPECT_GE(a, 0.0);
    EXPECT_LE(a, 1.0);
}

TEST(ProcessMath_alpha, Monotonic) {
    EXPECT_GT(alpha_from_half_life(5), alpha_from_half_life(50));
}

// ============================================================
// cs_zscore (8)
// ============================================================

TEST(ProcessMath_cs_zscore, ZeroVarianceGivesZero) {
    Eigen::VectorXd v = Eigen::VectorXd::Ones(5);
    auto z = cs_zscore(v, 3.0);
    EXPECT_TRUE(z.isZero());
}

TEST(ProcessMath_cs_zscore, MeanZero) {
    Eigen::VectorXd v(3); v << 1,2,3;
    auto z = cs_zscore(v, 3.0);
    EXPECT_NEAR(z.mean(), 0.0, 1e-12);
}

TEST(ProcessMath_cs_zscore, FiniteOutput) {
    Eigen::VectorXd v(3); v << 1, NAN, 3;
    auto z = cs_zscore(v, 3.0);
    EXPECT_TRUE(z.allFinite());
}

TEST(ProcessMath_cs_zscore, ClipUpperBound) {
    Eigen::VectorXd v(3); v << 0,0,100;
    auto z = cs_zscore(v, 1.0);
    EXPECT_LE(z.maxCoeff(), 1.0);
}

TEST(ProcessMath_cs_zscore, ClipLowerBound) {
    Eigen::VectorXd v(3); v << -100,0,0;
    auto z = cs_zscore(v, 1.0);
    EXPECT_GE(z.minCoeff(), -1.0);
}

TEST(ProcessMath_cs_zscore, PreserveOrder) {
    Eigen::VectorXd v(3); v << 1,2,3;
    auto z = cs_zscore(v, 3.0);
    EXPECT_LT(z(0), z(1));
    EXPECT_LT(z(1), z(2));
}

TEST(ProcessMath_cs_zscore, SmallVectorReturnsZero) {
    Eigen::VectorXd v(2); v << 1,2;
    auto z = cs_zscore(v, 3.0);
    EXPECT_TRUE(z.isZero());
}

TEST(ProcessMath_cs_zscore, AllNaNReturnsZero) {
    Eigen::VectorXd v(3); v << NAN,NAN,NAN;
    auto z = cs_zscore(v, 3.0);
    EXPECT_TRUE(z.isZero());
}

// ============================================================
// cs_quantile_map (6)
// ============================================================

TEST(ProcessMath_quantile, RangeWithinClip) {
    Eigen::VectorXd v(5); v << 1,2,3,4,5;
    auto q = cs_quantile_map(v, 2.0);
    EXPECT_GE(q.minCoeff(), -2.0);
    EXPECT_LE(q.maxCoeff(),  2.0);
}

TEST(ProcessMath_quantile, MonotonicOrder) {
    Eigen::VectorXd v(3); v << 10,20,30;
    auto q = cs_quantile_map(v, 3.0);
    EXPECT_LT(q(0), q(1));
    EXPECT_LT(q(1), q(2));
}

TEST(ProcessMath_quantile, HandlesNaN) {
    Eigen::VectorXd v(3); v << 1,NAN,3;
    auto q = cs_quantile_map(v, 3.0);
    EXPECT_TRUE(q.allFinite());
}

TEST(ProcessMath_quantile, SymmetricOutput) {
    Eigen::VectorXd v(2); v << 1,2;
    auto q = cs_quantile_map(v, 1.0);
    EXPECT_NEAR(q.sum(), 0.0, 1e-12);
}

TEST(ProcessMath_quantile, SingleElement) {
    Eigen::VectorXd v(1); v << 5;
    auto q = cs_quantile_map(v, 3.0);
    EXPECT_DOUBLE_EQ(q(0), -3.0);
}

TEST(ProcessMath_quantile, Deterministic) {
    Eigen::VectorXd v(4); v << 1,2,3,4;
    EXPECT_TRUE(cs_quantile_map(v,2.0).isApprox(cs_quantile_map(v,2.0)));
}

// ============================================================
// rolling + helpers (8)
// ============================================================

TEST(ProcessMath_row_at, InRange) {
    Eigen::MatrixXd M(2,2); M << 1,2,3,4;
    auto r = row_at(M,1);
    EXPECT_DOUBLE_EQ(r(0),3);
}

TEST(ProcessMath_row_at, OutOfRangeReturnsZero) {
    Eigen::MatrixXd M(2,2); M << 1,2,3,4;
    EXPECT_TRUE(row_at(M,5).isZero());
}

TEST(ProcessMath_rolling_mean, SimpleMean) {
    Eigen::MatrixXd M(3,1); M << 1,2,3;
    EXPECT_DOUBLE_EQ(rolling_mean_last(M,2,2)(0), 2.5);
}

TEST(ProcessMath_rolling_std, ZeroStd) {
    Eigen::MatrixXd M(3,1); M << 2,2,2;
    EXPECT_DOUBLE_EQ(rolling_std_last(M,2,3)(0), 0.0);
}

TEST(ProcessMath_median, OddCount) {
    Eigen::VectorXd v(3); v << 1,3,2;
    EXPECT_DOUBLE_EQ(median_vec(v), 2);
}

TEST(ProcessMath_median, EvenCount) {
    Eigen::VectorXd v(4); v << 1,2,3,4;
    EXPECT_DOUBLE_EQ(median_vec(v), 2.5);
}

TEST(ProcessMath_median, IgnoreNaN) {
    Eigen::VectorXd v(3); v << 1,NAN,3;
    EXPECT_DOUBLE_EQ(median_vec(v), 2);
}

TEST(ProcessMath_fin, ReplaceNaN) {
    EXPECT_DOUBLE_EQ(fin(NAN), 0.0);
}
static void write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p.string());
    ASSERT_TRUE(f.is_open()) << "cannot open " << p.string();
    f << s;
}

static fs::path make_fixture_dir() {
    // unique folder under temp
    auto base = fs::temp_directory_path() / ("dlx_fixture_" + std::to_string(::getpid()) + "_" + std::to_string(std::rand()));
    fs::create_directories(base);
    return base;
}

// Minimal daily_60d.csv with required columns: date,symbol,o,h,l,c,v,val
// plus optional inside/outside and adj_close.
static void write_daily_60d(const fs::path& dir) {
    std::string csv;
    csv += "date,symbol,o,h,l,c,v,val,inside_vol,outside_vol,adj_close\n";
    // 3 dates x 2 symbols
    // date format is normalized by DataLoader; we keep YYYY-MM-DD
    struct Row { const char* d; const char* s; double o,h,l,c; long long v; long long val; long long inV; long long outV; double adj; };
    std::vector<Row> rows = {
        {"2025-12-09","2330",100,110, 95,105,1000,105000,400,600,105},
        {"2025-12-09","2317", 50, 55, 48, 52,2000,104000,900,1100,52},
        {"2025-12-10","2330",106,112,100,110,1200,132000,500,700,110},
        {"2025-12-10","2317", 52, 56, 50, 54,2100,113400,1000,1100,54},
        {"2025-12-11","2330",109,115,108,114, 900,102600,350,550,114},
        {"2025-12-11","2317", 54, 57, 53, 56,2200,123200,1050,1150,56},
    };
    for (auto& r: rows) {
        csv += std::string(r.d)+","+r.s+","
            +std::to_string(r.o)+","+std::to_string(r.h)+","+std::to_string(r.l)+","+std::to_string(r.c)+","
            +std::to_string(r.v)+","+std::to_string(r.val)+","
            +std::to_string(r.inV)+","+std::to_string(r.outV)+","
            +std::to_string(r.adj)
            +"\n";
    }
    write_text(dir/"daily_60d.csv", csv);
}

// Optional brokers.csv (if missing, DataLoader should still run).
static void write_brokers(const fs::path& dir) {
    std::string csv;
    csv += "date,symbol,net_buy\n";
    csv += "2025-12-09,2330,100\n";
    csv += "2025-12-09,2317,-200\n";
    csv += "2025-12-10,2330,50\n";
    csv += "2025-12-10,2317,0\n";
    csv += "2025-12-11,2330,-30\n";
    csv += "2025-12-11,2317,20\n";
    write_text(dir/"brokers.csv", csv);
}

struct ProcessFixture : public ::testing::Test {
    fs::path dir;
    dlx::DataLoader* dl = nullptr;

    void SetUp() override {
        dir = make_fixture_dir();
        write_daily_60d(dir);
        // For some tests we overwrite/delete brokers.csv; default: present
        write_brokers(dir);

        dl = new dlx::DataLoader(dir.string());
        ASSERT_TRUE(dl->loadDailyPanelAndBuildFeatures()) << "fixture load failed";
    }

    void TearDown() override {
        delete dl;
        dl = nullptr;
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

// ------------------------------------------------------------
// 1) DataLoader invariants (8)
// ------------------------------------------------------------

TEST_F(ProcessFixture, DL_LoadsDatesAndSymbols) {
    EXPECT_EQ(dl->dates().size(), 3u);
    EXPECT_EQ(dl->symbols().size(), 2u);
}

TEST_F(ProcessFixture, DL_OHLCVShapesMatch) {
    EXPECT_EQ(dl->C().rows(), (int)dl->dates().size());
    EXPECT_EQ(dl->C().cols(), (int)dl->symbols().size());
    EXPECT_EQ(dl->O().rows(), dl->C().rows());
    EXPECT_EQ(dl->V().cols(), dl->C().cols());
}

TEST_F(ProcessFixture, DL_PricesAreFinite) {
    EXPECT_TRUE(dl->C().allFinite());
    EXPECT_TRUE(dl->O().allFinite());
    EXPECT_TRUE(dl->H().allFinite());
    EXPECT_TRUE(dl->L().allFinite());
}

TEST_F(ProcessFixture, DL_FeatureMatricesHaveSamePanelSize) {
    const int T = dl->C().rows(), N = dl->C().cols();
    EXPECT_EQ(dl->feat_Gap().rows(), T);
    EXPECT_EQ(dl->feat_Gap().cols(), N);
    EXPECT_EQ(dl->feat_GKVol().rows(), T);
    EXPECT_EQ(dl->feat_Imbalance().cols(), N);
}

TEST_F(ProcessFixture, DL_BrokersOptionalMissingStillLoads) {
    // remove brokers.csv then reload
    std::error_code ec;
    fs::remove(dir/"brokers.csv", ec);
    dlx::DataLoader dl2(dir.string());
    EXPECT_TRUE(dl2.loadDailyPanelAndBuildFeatures());
}

TEST_F(ProcessFixture, DL_AdjustOHLCByAdjCloseNoCrash) {
    // should be callable even if already adjusted / or no-op
    EXPECT_NO_THROW(dl->adjustOHLCbyAdjCloseIfAny());
}

TEST_F(ProcessFixture, DL_FeaturesFiniteAfterBuild) {
    EXPECT_TRUE(dl->feat_Gap().allFinite());
    EXPECT_TRUE(dl->feat_Mom5().allFinite());
    EXPECT_TRUE(dl->feat_Mom10().allFinite());
    EXPECT_TRUE(dl->feat_Mom20().allFinite());
    EXPECT_TRUE(dl->feat_GKVol().allFinite());
}

TEST_F(ProcessFixture, DL_TurnoverShareFinite) {
    EXPECT_TRUE(dl->feat_TurnoverShare().allFinite());
}

// ------------------------------------------------------------
// 2) ProcessEngine (16)
// ------------------------------------------------------------

TEST_F(ProcessFixture, Engine_Constructs) {
    process::ProcessEngine eng(*dl);
    EXPECT_EQ(&eng.dl(), dl);
}

TEST_F(ProcessFixture, Engine_XsectionStandardize_ZScoreClipFinite) {
    process::ProcessEngine eng(*dl);
    process::ProcessingConfig cfg;
    cfg.mode = process::XSectionStandardize::ZScoreClip;
    cfg.clip = 3.0;
    auto X = dl->feat_Gap();
    auto Z = eng.xsection_standardize(X, cfg);
    EXPECT_EQ(Z.rows(), X.rows());
    EXPECT_EQ(Z.cols(), X.cols());
    EXPECT_TRUE(Z.allFinite());
}

TEST_F(ProcessFixture, Engine_XsectionStandardize_QuantileFinite) {
    process::ProcessEngine eng(*dl);
    process::ProcessingConfig cfg;
    cfg.mode = process::XSectionStandardize::QuantileMap;
    cfg.clip = 2.0;
    auto X = dl->feat_Mom5();
    auto Q = eng.xsection_standardize(X, cfg);
    EXPECT_TRUE(Q.allFinite());
    EXPECT_LE(Q.maxCoeff(), cfg.clip + 1e-9);
    EXPECT_GE(Q.minCoeff(), -cfg.clip - 1e-9);
}

TEST_F(ProcessFixture, Engine_RiskOutputShapes) {
    process::ProcessEngine eng(*dl);
    process::RiskConfig rc;
    auto out = eng.build_liquidity_scaled_risk(/*t=*/2, rc);
    const int N = (int)dl->symbols().size();
    EXPECT_EQ(out.Sigma_tilde.rows(), N);
    EXPECT_EQ(out.Sigma_tilde.cols(), N);
    EXPECT_EQ(out.L.rows(), N);
    EXPECT_EQ(out.L.cols(), N);
    EXPECT_EQ(out.sigma_i.size(), N);
    EXPECT_TRUE(out.Sigma_tilde.allFinite());
    EXPECT_TRUE(out.sigma_i.allFinite());
}

TEST_F(ProcessFixture, Engine_RiskSigmaNonNegative) {
    process::ProcessEngine eng(*dl);
    process::RiskConfig rc;
    auto out = eng.build_liquidity_scaled_risk(2, rc);
    EXPECT_GE(out.sigma_i.minCoeff(), 0.0);
}

TEST_F(ProcessFixture, Engine_RiskEpsStabilizes) {
    process::ProcessEngine eng(*dl);
    process::RiskConfig rc;
    rc.eps = 1e-3;
    auto out = eng.build_liquidity_scaled_risk(2, rc);
    EXPECT_TRUE(out.L.allFinite());
}

TEST_F(ProcessFixture, Engine_CostsShapesFinite) {
    process::ProcessEngine eng(*dl);
    process::CostConfig cc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, -1;
    auto cst = eng.estimate_costs(/*t=*/2, A, side, cc);
    EXPECT_EQ(cst.fee_buy.size(), 2);
    EXPECT_EQ(cst.impact.size(), 2);
    EXPECT_TRUE(cst.impact.allFinite());
}

TEST_F(ProcessFixture, Engine_CostsZeroAZeroCost) {
    process::ProcessEngine eng(*dl);
    process::CostConfig cc;
    Eigen::VectorXd A = Eigen::VectorXd::Zero(2);
    Eigen::VectorXi side(2); side << +1, -1;
    auto cst = eng.estimate_costs(2, A, side, cc);
    EXPECT_NEAR(cst.impact.norm(), 0.0, 1e-12);
    EXPECT_NEAR(cst.fee_buy.norm(), 0.0, 1e-12);
}

TEST_F(ProcessFixture, Engine_BlackSwanDefaultInRange) {
    process::ProcessEngine eng(*dl);
    process::BlackSwanConfig bs;
    auto out = eng.black_swan_scale(/*t=*/2, bs, /*external_event=*/false);
    EXPECT_GE(out.m, 0.7);
    EXPECT_LE(out.m, 1.3);
}

TEST_F(ProcessFixture, Engine_BlackSwanExternalEventNotLargerThan1_3) {
    process::ProcessEngine eng(*dl);
    process::BlackSwanConfig bs;
    auto out = eng.black_swan_scale(2, bs, true);
    EXPECT_LE(out.m, 1.3);
}

TEST_F(ProcessFixture, Engine_BlackSwanDeterministic) {
    process::ProcessEngine eng(*dl);
    process::BlackSwanConfig bs;
    auto a = eng.black_swan_scale(2, bs, false).m;
    auto b = eng.black_swan_scale(2, bs, false).m;
    EXPECT_DOUBLE_EQ(a,b);
}

TEST_F(ProcessFixture, Engine_StandardizeHandlesNaN) {
    process::ProcessEngine eng(*dl);
    process::ProcessingConfig cfg;
    cfg.mode = process::XSectionStandardize::ZScoreClip;
    cfg.clip = 3.0;
    Eigen::MatrixXd X = dl->feat_Gap();
    X(1,0) = std::numeric_limits<double>::quiet_NaN();
    auto Z = eng.xsection_standardize(X, cfg);
    EXPECT_TRUE(Z.allFinite());
}

TEST_F(ProcessFixture, Engine_RiskDifferentSigmaSourceStillFinite) {
    process::ProcessEngine eng(*dl);
    process::RiskConfig rc;
    rc.sigma_source = process::SigmaSource::Std30;
    rc.std_window = 2; // small window ok for tiny fixture
    auto out = eng.build_liquidity_scaled_risk(2, rc);
    EXPECT_TRUE(out.Sigma_tilde.allFinite());
}

TEST_F(ProcessFixture, Engine_RiskC_LiqAffectsDliq) {
    process::ProcessEngine eng(*dl);
    process::RiskConfig rc0; rc0.c_liq = 0.0;
    process::RiskConfig rc1; rc1.c_liq = 10.0;
    auto a = eng.build_liquidity_scaled_risk(2, rc0).Dliq_i;
    auto b = eng.build_liquidity_scaled_risk(2, rc1).Dliq_i;
    EXPECT_TRUE(a.allFinite() && b.allFinite());
    EXPECT_GE(b.minCoeff(), a.minCoeff());
}

TEST_F(ProcessFixture, Engine_CostSideSwitchAffectsBuySell) {
    process::ProcessEngine eng(*dl);
    process::CostConfig cc;
    Eigen::VectorXd A(2); A << 1e6, 1e6;
    Eigen::VectorXi buy(2); buy << +1, +1;
    Eigen::VectorXi sell(2); sell << -1, -1;
    auto cb = eng.estimate_costs(2, A, buy, cc);
    auto cs = eng.estimate_costs(2, A, sell, cc);
    EXPECT_GT(cs.tax_sell.sum(), cb.tax_sell.sum()); // tax on sell
}

TEST_F(ProcessFixture, Engine_CostImpactNonNegative) {
    process::ProcessEngine eng(*dl);
    process::CostConfig cc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, -1;
    auto cst = eng.estimate_costs(2, A, side, cc);
    EXPECT_GE(cst.impact.minCoeff(), 0.0);
}

// ------------------------------------------------------------
// 3) ProcessEngineForecast (8)
// ------------------------------------------------------------

TEST_F(ProcessFixture, Forecast_Stage1Shapes) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc;
    auto fo = eng.stage1_forecast(/*t=*/2, px, fc, std::nullopt);
    EXPECT_EQ(fo.rhat_raw.size(), (int)dl->symbols().size());
    EXPECT_TRUE(fo.rhat_raw.allFinite());
}

TEST_F(ProcessFixture, Forecast_Stage1Deterministic) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc;
    auto a = eng.stage1_forecast(2, px, fc).rhat_raw;
    auto b = eng.stage1_forecast(2, px, fc).rhat_raw;
    EXPECT_TRUE(a.isApprox(b, 1e-12));
}

TEST_F(ProcessFixture, Forecast_Stage1WithNextDayReturnsNoCrash) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc;
    Eigen::MatrixXd next(3,2); next.setZero();
    auto fo = eng.stage1_forecast(2, px, fc, next);
    EXPECT_TRUE(fo.rhat_raw.allFinite());
}

TEST_F(ProcessFixture, Forecast_Stage1OutOfRangeReturnsZero) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc;
    auto fo = eng.stage1_forecast(/*t=*/999, px, fc);
    EXPECT_NEAR(fo.rhat_raw.norm(), 0.0, 1e-12);
}

TEST_F(ProcessFixture, Forecast_Stage1DifferentStandardizeMode) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    px.mode = process::XSectionStandardize::QuantileMap;
    px.clip = 2.0;
    process::ForecastConfig fc;
    auto fo = eng.stage1_forecast(2, px, fc);
    EXPECT_TRUE(fo.rhat_raw.allFinite());
}

TEST_F(ProcessFixture, Forecast_Stage1HalfLifeEffectFinite) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc1; fc1.half_life = 5;
    process::ForecastConfig fc2; fc2.half_life = 50;
    auto a = eng.stage1_forecast(2, px, fc1).rhat_raw;
    auto b = eng.stage1_forecast(2, px, fc2).rhat_raw;
    EXPECT_TRUE(a.allFinite() && b.allFinite());
}

TEST_F(ProcessFixture, Forecast_Stage1ThetaBiasFinite) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc; fc.theta_bias = 0.5;
    auto fo = eng.stage1_forecast(2, px, fc);
    EXPECT_TRUE(fo.rhat_raw.allFinite());
}

TEST_F(ProcessFixture, Forecast_OutputVectorsSameSize) {
    process::ProcessEngineForecast eng(*dl);
    process::ProcessingConfig px;
    process::ForecastConfig fc;
    auto fo = eng.stage1_forecast(2, px, fc);
    EXPECT_EQ(fo.rhat_cal.size(), fo.rhat_raw.size());
    EXPECT_EQ(fo.rhat_enh.size(), fo.rhat_raw.size());
}

// ------------------------------------------------------------
// 4) ProcessExporter (8)
// ------------------------------------------------------------

static bool file_nonempty(const fs::path& p) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    return fs::file_size(p, ec) > 10;
}

TEST_F(ProcessFixture, Exporter_CSVFallbackCreatesFiles) {
    process::ProcessExporter ex(*dl);

    process::DumpExcelConfig out;
    out.prefer_xlsx = false;
    out.out_dir_csv = dir.string();

    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;

    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, -1;

    ex.dump_to_excel(out, /*t=*/2, px, risk, cost, bs, fc, A, side);

    EXPECT_TRUE(file_nonempty(dir/"risk_Sigma_tilde.csv"));
    EXPECT_TRUE(file_nonempty(dir/"risk_sigma_i.csv"));
    EXPECT_TRUE(file_nonempty(dir/"cost_impact.csv"));
    EXPECT_TRUE(file_nonempty(dir/"stage1_rhat_raw.csv"));
    EXPECT_TRUE(file_nonempty(dir/"inputs_A.csv"));
    EXPECT_TRUE(file_nonempty(dir/"inputs_side.csv"));
}

TEST_F(ProcessFixture, Exporter_DifferentTNoCrash) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 0.0, 0.0;
    Eigen::VectorXi side(2); side << +1, +1;
    EXPECT_NO_THROW(ex.dump_to_excel(out, /*t=*/1, px, risk, cost, bs, fc, A, side));
}

TEST_F(ProcessFixture, Exporter_WithNextDayReturnsNoCrash) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 1e6, 1e6;
    Eigen::VectorXi side(2); side << +1, -1;
    Eigen::MatrixXd next(3,2); next.setZero();
    EXPECT_NO_THROW(ex.dump_to_excel(out, 2, px, risk, cost, bs, fc, A, side, next));
}

TEST_F(ProcessFixture, Exporter_OverwritesExistingCSVs) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, -1;
    ex.dump_to_excel(out, 2, px, risk, cost, bs, fc, A, side);
    ex.dump_to_excel(out, 2, px, risk, cost, bs, fc, A, side);
    EXPECT_TRUE(file_nonempty(dir/"stage1_rhat_raw.csv"));
}

TEST_F(ProcessFixture, Exporter_ProducesAll11CSVOutputs) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, -1;
    ex.dump_to_excel(out, 2, px, risk, cost, bs, fc, A, side);

    std::vector<std::string> names = {
        "risk_Sigma_tilde.csv","risk_Cholesky_L.csv","risk_sigma_i.csv","risk_Dliq_i.csv",
        "cost_fee_buy.csv","cost_fee_sell.csv","cost_tax_sell.csv","cost_impact.csv",
        "stage1_rhat_raw.csv","inputs_A.csv","inputs_side.csv"
    };
    for (auto& n: names) EXPECT_TRUE(file_nonempty(dir/n)) << n;
}

TEST_F(ProcessFixture, Exporter_SideAllBuyStillWrites) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, +1;
    ex.dump_to_excel(out, 2, px, risk, cost, bs, fc, A, side);
    EXPECT_TRUE(file_nonempty(dir/"cost_fee_buy.csv"));
}

TEST_F(ProcessFixture, Exporter_SideAllSellStillWritesTax) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << -1, -1;
    ex.dump_to_excel(out, 2, px, risk, cost, bs, fc, A, side);
    EXPECT_TRUE(file_nonempty(dir/"cost_tax_sell.csv"));
}

TEST_F(ProcessFixture, Exporter_TOutOfRangeStillNoCrashAndWrites) {
    process::ProcessExporter ex(*dl);
    process::DumpExcelConfig out; out.prefer_xlsx=false; out.out_dir_csv=dir.string();
    process::ProcessingConfig px;
    process::RiskConfig risk;
    process::CostConfig cost;
    process::BlackSwanConfig bs;
    process::ForecastConfig fc;
    Eigen::VectorXd A(2); A << 1e6, 2e6;
    Eigen::VectorXi side(2); side << +1, -1;
    EXPECT_NO_THROW(ex.dump_to_excel(out, 999, px, risk, cost, bs, fc, A, side));
    // Some outputs may be zeros but files should exist.
    EXPECT_TRUE(fs::exists(dir/"inputs_A.csv"));
}

// Total tests = 8 + 16 + 8 + 8 = 40
