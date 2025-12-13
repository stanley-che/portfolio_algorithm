// tests/test_data_loader_40cases.cpp
// ------------------------------------------------------------
// DataLoader unit tests (40 test items)
// - Designed for your dlx::DataLoader (data_loader_all.hpp)
// - Focus: deterministic unit-level behaviors + small toy ETL run
//
/*

the link is at cpp_core

g++ -std=c++17 -Isrc/revise -Ithird_party/third_party/googletest \
-Ithird_party/third_party/googletest/googletest \
-Ithird_party/third_party/googletest/googletest/include \
-I/usr/include/eigen3 \
test/test_twse_quotes.cpp test/test_twse_meta.cpp test/test_broker_unit.cpp test/test_data_loader_40cases.cpp \
third_party/third_party/googletest/googletest/src/gtest-all.cc \
third_party/third_party/googletest/googletest/src/gtest_main.cc \
-o test_all -lpthread -lcurl

  */
// ------------------------------------------------------------

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>

// If you want to test private helpers (winsorize, impute, etc.)
// enable this hack ONLY in tests.
#ifndef DLX_TEST_PRIVATE_ACCESS
#define DLX_TEST_PRIVATE_ACCESS 1
#endif

#if DLX_TEST_PRIVATE_ACCESS
  #define private public
  #define protected public
#endif

#include "data_loader_all.hpp"

#if DLX_TEST_PRIVATE_ACCESS
  #undef private
  #undef protected
#endif

namespace fs = std::filesystem;
using dlx::DataLoader;

static void write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p);
    ASSERT_TRUE(f.is_open()) << "Cannot write file: " << p.string();
    f << s;
}

static bool is_finite(double x){
    return std::isfinite(x);
}

// ---------------------------
// Fixture: create a tiny dataset
// ---------------------------
struct DataLoaderToy : public ::testing::Test {
    fs::path dir;

    void SetUp() override {
        dir = fs::path("testdata_dl40");
        fs::remove_all(dir);
        fs::create_directories(dir);

        // daily_60d.csv: minimal OHLCV/VAL, 3 days, 2 symbols
        write_text(dir / "daily_60d.csv",
            "date,symbol,O,H,L,C,V,VAL\n"
            "2024-01-01,2330,10,12,9,11,100,1100\n"
            "2024-01-01,2303,20,21,19,20,50,1000\n"
            "2024-01-02,2330,11,13,10,12,120,1440\n"
            "2024-01-02,2303,21,22,20,21,60,1260\n"
            "2024-01-03,2330,12,14,11,13,80,1040\n"
            "2024-01-03,2303,22,23,21,22,70,1540\n"
        );

        // brokers.csv: optional; keep simple netbuy column that your loader may read
        write_text(dir / "brokers.csv",
            "date,symbol,broker_netbuy_topk\n"
            "2024-01-01,2330,100\n"
            "2024-01-01,2303,50\n"
            "2024-01-02,2330,80\n"
            "2024-01-02,2303,40\n"
            "2024-01-03,2330,60\n"
            "2024-01-03,2303,30\n"
        );

        // meta.csv
        write_text(dir / "meta.csv",
            "symbol,lot,odd_lot_unit,industry,group,tradable_flag\n"
            "2330,1000,1,semicon,listed,1\n"
            "2303,1000,1,semicon,listed,1\n"
        );
    }

    void TearDown() override {
        // keep for debugging if needed; uncomment to auto clean
        // fs::remove_all(dir);
    }
};

// ============================================================
// A. Utils (12)
// ============================================================

TEST(DataLoader_Utils, StripBom_RemovesUtf8Bom) {
    std::string s = "\xEF\xBB\xBF" "date,code\n";
    DataLoader::strip_bom(s);
    EXPECT_EQ(s, "date,code\n");
}

TEST(DataLoader_Utils, StripBom_NoBomNoChange) {
    std::string s = "date,code\n";
    DataLoader::strip_bom(s);
    EXPECT_EQ(s, "date,code\n");
}

TEST(DataLoader_Utils, Trim_RemovesSpacesTabs) {
    EXPECT_EQ(DataLoader::trim("  abc\t"), "abc");
    EXPECT_EQ(DataLoader::trim("\t  abc  \t"), "abc");
}

TEST(DataLoader_Utils, Lower_ToLowerAscii) {
    EXPECT_EQ(DataLoader::lower("AbC_XyZ"), "abc_xyz");
}

TEST(DataLoader_Utils, SplitCsvLine_Simple) {
    auto v = DataLoader::splitCsvLine("a,b,c");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b");
    EXPECT_EQ(v[2], "c");
}

TEST(DataLoader_Utils, SplitCsvLine_QuotedComma) {
    auto v = DataLoader::splitCsvLine("a,\"b,c\",d");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[1], "b,c");
}

TEST(DataLoader_Utils, NormalizeDate_YYYYMMDD) {
    EXPECT_EQ(DataLoader::normalize_date("20240103"), "2024-01-03");
}

TEST(DataLoader_Utils, NormalizeDate_YYYY_MM_DD_Passthrough) {
    EXPECT_EQ(DataLoader::normalize_date("2024-01-03"), "2024-01-03");
}

TEST(DataLoader_Utils, NormalizeDate_SlashToDash) {
    EXPECT_EQ(DataLoader::normalize_date("2024/01/03"), "2024-01-03");
}

TEST(DataLoader_Utils, NormalizeDate_ROCToAD) {
    // 112/01/03 -> 2023-01-03
    EXPECT_EQ(DataLoader::normalize_date("112/01/03"), "2023-01-03");
}

TEST(DataLoader_Utils, ParseNumClean_RemovesCommas) {
    EXPECT_NEAR(DataLoader::parse_num_clean("1,234.5"), 1234.5, 1e-12);
}

TEST(DataLoader_Utils, ParseLlClean_RemovesCommas) {
    EXPECT_EQ(DataLoader::parse_ll_clean("9,876"), 9876LL);
}

// ============================================================
// B. IO / Shapes / Indexing (10)
// Many of these are validated through a tiny ETL run.
// ============================================================

TEST(DataLoader_IO, ReadDaily_NoFile_ReturnFalse) {
    dlx::DataLoader dl("no_such_dir");
    // If your load routine returns false when daily file missing
    EXPECT_FALSE(dl.loadDailyPanelAndBuildFeatures());
}

TEST_F(DataLoaderToy, LoadAndBuildFeatures_ReturnTrue) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
}

TEST_F(DataLoaderToy, DatesAndSymbols_NotEmpty) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    EXPECT_GE(dl.dates().size(), 3u);
    EXPECT_EQ(dl.symbols().size(), 2u);
}

TEST_F(DataLoaderToy, MatricesShape_TxN) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const int T = (int)dl.dates().size();
    const int N = (int)dl.symbols().size();

    EXPECT_EQ(dl.C().rows(), T);
    EXPECT_EQ(dl.C().cols(), N);
    EXPECT_EQ(dl.O().rows(), T);
    EXPECT_EQ(dl.VAL().cols(), N);
}

TEST_F(DataLoaderToy, FeatureShapes_MatchPriceShape) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const int T = dl.C().rows();
    const int N = dl.C().cols();

    EXPECT_EQ(dl.feat_Liquidity().rows(), T);
    EXPECT_EQ(dl.feat_Liquidity().cols(), N);
    EXPECT_EQ(dl.feat_Imbalance().rows(), T);
    EXPECT_EQ(dl.feat_Imbalance().cols(), N);
    EXPECT_EQ(dl.feat_BIAS().rows(), T);
    EXPECT_EQ(dl.feat_BIAS().cols(), N);
}

TEST_F(DataLoaderToy, SymbolsMatchColumns) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    EXPECT_EQ((int)dl.symbols().size(), dl.C().cols());
}

TEST_F(DataLoaderToy, Liquidity_IsLogValPlus1_FirstRowCheck) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());

    // locate col for 2330
    int col2330 = (dl.symbols()[0] == "2330") ? 0 : 1;
    double val = dl.VAL()(0, col2330);
    double expected = std::log(std::max(0.0, val) + 1.0);
    EXPECT_NEAR(dl.feat_Liquidity()(0, col2330), expected, 1e-9);
}

TEST_F(DataLoaderToy, TurnoverShare_RowSumsToOne_WhenValPositive) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& ts = dl.feat_TurnoverShare();
    for (int r = 0; r < ts.rows(); ++r) {
        double s = ts.row(r).sum();
        EXPECT_NEAR(s, 1.0, 1e-8);
    }
}

TEST_F(DataLoaderToy, Imbalance_InRangeMinus1To1) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& imb = dl.feat_Imbalance();
    for (int r = 0; r < imb.rows(); ++r) {
        for (int c = 0; c < imb.cols(); ++c) {
            EXPECT_GE(imb(r,c), -1.0000001);
            EXPECT_LE(imb(r,c), +1.0000001);
        }
    }
}

TEST_F(DataLoaderToy, BrokerStrength_Finite) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& bf = dl.feat_BrokerStrength();
    for (int r = 0; r < bf.rows(); ++r) {
        for (int c = 0; c < bf.cols(); ++c) {
            EXPECT_TRUE(is_finite(bf(r,c)));
        }
    }
}

// ============================================================
// C. Cleaning (8)
// These rely on private helpers; enabled via DLX_TEST_PRIVATE_ACCESS.
// If you don't want to access private helpers, mark these as DISABLED_.
// ============================================================

TEST_F(DataLoaderToy, Cleaning_ImputeForwardFill_FillsGaps) {
    DataLoader dl(dir.string());
    dl.config().forward_fill = true; // (If your config uses bool, change True->true)
    dl.config().back_fill = false;

    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();

    // Make a gap: set one close to NaN, then impute
    dl.C_(1,0) = std::numeric_limits<double>::quiet_NaN();
    dl.C_(0,0) = 11.0;
    dl.imputeMissing();

    EXPECT_NEAR(dl.C_(1,0), 11.0, 1e-12);
}

TEST_F(DataLoaderToy, Cleaning_ImputeBackFill_FillsLeadingGaps) {
    DataLoader dl(dir.string());
    dl.config().forward_fill = false;
    dl.config().back_fill = true;

    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();

    dl.C_(0,0) = std::numeric_limits<double>::quiet_NaN();
    dl.C_(1,0) = 12.0;

    dl.imputeMissing();
    EXPECT_NEAR(dl.C_(0,0), 12.0, 1e-12);
}

TEST_F(DataLoaderToy, Cleaning_ForwardFillDisabled_DoesNotFill) {
    DataLoader dl(dir.string());
    dl.config().forward_fill = false;
    dl.config().back_fill = false;

    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();

    dl.C_(1,0) = std::numeric_limits<double>::quiet_NaN();
    dl.C_(0,0) = 11.0;

    dl.imputeMissing();
    EXPECT_TRUE(std::isnan(dl.C_(1,0)));
}

TEST_F(DataLoaderToy, Cleaning_AutoNonTradableByVolVal_MarksFlag) {
    DataLoader dl(dir.string());
    dl.config().auto_non_tradable = true;

    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();

    // Force V=0, VAL=0 for a symbol => should become non-tradable
    dl.V_(0,0) = 0.0;
    dl.VAL_(0,0) = 0.0;

    dl.readMetaCsv((dir/"meta.csv").string());
    dl.markNonTradableByVolVal();

    // tradable_flag should be 0 for that stock
    EXPECT_EQ(dl.stock_info_[0].tradable_flag, 0);
}

TEST_F(DataLoaderToy, Cleaning_AutoNonTradableDisabled_NoChange) {
    DataLoader dl(dir.string());
    dl.config().auto_non_tradable = false;

    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    dl.readMetaCsv((dir/"meta.csv").string());

    int before = dl.stock_info_[0].tradable_flag;
    dl.markNonTradableByVolVal();
    EXPECT_EQ(dl.stock_info_[0].tradable_flag, before);
}

TEST(DataLoader_Cleaning, WinsorizeVec_ShortVector_NoChange) {
    Eigen::VectorXd v(5);
    v << 1,2,3,4,100;
    Eigen::VectorXd before = v;

    DataLoader::winsorize_vec(v, 0.01); // static private -> 你已 hack 成 public

    EXPECT_EQ(v.size(), before.size());
    for (int i=0;i<v.size();++i) EXPECT_DOUBLE_EQ(v(i), before(i));
}

TEST(DataLoader_Cleaning, WinsorizeVec_ClampsOutliers) {
    Eigen::VectorXd v(100);
    for (int i=0;i<100;++i) v(i) = (double)i;
    v(0) = -1000;
    v(99)= 1000;

    DataLoader::winsorize_vec(v, 0.01);

    // extremes should be clamped inward
    EXPECT_GE(v(0), v(1));
    EXPECT_LE(v(99), v(98));
}


TEST_F(DataLoaderToy, Cleaning_WinsorizeAll_DoesNotChangeShape) {
    DataLoader dl(dir.string());
    dl.config().winsorize_p = 0.01;

    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    EXPECT_EQ(dl.feat_Mom10().rows(), dl.C().rows());
    EXPECT_EQ(dl.feat_Mom10().cols(), dl.C().cols());
}

// ============================================================
// D. Features (10)
// Many are validated by behavior checks on toy data.
// ============================================================

TEST_F(DataLoaderToy, Feature_Gap_FirstRowIsZeroOrFinite) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    // gap(0,*) typically 0
    EXPECT_TRUE(is_finite(dl.feat_Gap()(0,0)));
    EXPECT_TRUE(is_finite(dl.feat_Gap()(0,1)));
}

TEST_F(DataLoaderToy, Feature_Gap_UsesPrevClose) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());

    // For 2330: day2 gap = (O2 - C1)/C1 = (11 - 11)/11 = 0
    int col2330 = (dl.symbols()[0] == "2330") ? 0 : 1;
    double gap = dl.feat_Gap()(1, col2330);
    EXPECT_NEAR(gap, 0.0, 1e-12);
}

TEST_F(DataLoaderToy, Feature_Mom10_NotNaN) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    // With only 3 days, mom10 likely 0; ensure finite
    for (int r=0;r<dl.feat_Mom10().rows();++r)
      for (int c=0;c<dl.feat_Mom10().cols();++c)
        EXPECT_TRUE(is_finite(dl.feat_Mom10()(r,c)));
}

TEST_F(DataLoaderToy, Feature_GKVol_NonNegative) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& gk = dl.feat_GKVol();
    for(int r=0;r<gk.rows();++r)
      for(int c=0;c<gk.cols();++c)
        EXPECT_GE(gk(r,c), -1e-12);
}

TEST_F(DataLoaderToy, Feature_TurnoverShare_In01) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& ts = dl.feat_TurnoverShare();
    for(int r=0;r<ts.rows();++r)
      for(int c=0;c<ts.cols();++c){
        EXPECT_GE(ts(r,c), -1e-12);
        EXPECT_LE(ts(r,c),  1.0+1e-12);
      }
}

TEST_F(DataLoaderToy, Feature_Imbalance_Clamp) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& imb = dl.feat_Imbalance();
    EXPECT_LE(imb.maxCoeff(),  1.0000001);
    EXPECT_GE(imb.minCoeff(), -1.0000001);
}

TEST_F(DataLoaderToy, Feature_BIAS_Finite) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& b = dl.feat_BIAS();
    for (int r=0;r<b.rows();++r)
      for (int c=0;c<b.cols();++c)
        EXPECT_TRUE(is_finite(b(r,c)));
}

TEST_F(DataLoaderToy, Feature_BIAS_ZeroWhenMAZero) {
    // Construct minimal by forcing C to zeros then recompute bias if method is public.
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    dl.C_.setZero();
    dl.buildBIASandBrokerStrength(); // if private accessible
    EXPECT_NEAR(dl.feat_bias_(0,0), 0.0, 1e-12);
}

TEST_F(DataLoaderToy, Feature_BrokerStrength_FiniteAfterCompute) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& bf = dl.feat_BrokerStrength();
    EXPECT_TRUE(std::isfinite(bf.sum()));
}

TEST_F(DataLoaderToy, Feature_Liquidity_NonNegative) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    const auto& liq = dl.feat_Liquidity();
    for(int r=0;r<liq.rows();++r)
      for(int c=0;c<liq.cols();++c)
        EXPECT_GE(liq(r,c), -1e-12);
}

// ============================================================
// E. Config / Limits (additional checks to reach 40 total)
// ============================================================

TEST_F(DataLoaderToy, Config_MaxSymbols_LimitsRegistration) {
    DataLoader dl(dir.string());
    dl.config().max_symbols = 1;

    // create 2-symbol daily; loader should keep at most 1 symbol
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    EXPECT_LE(dl.symbols().size(), 1u);
}

TEST_F(DataLoaderToy, Config_WinsorizeP_ValidRange) {
    DataLoader dl(dir.string());
    dl.config().winsorize_p = 0.02;
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    // Just ensure it runs and produces finite features.
    EXPECT_TRUE(std::isfinite(dl.feat_Liquidity().sum()));
}

TEST(DataLoader_Utils, SafeDiv_ZeroDenominator) {
    EXPECT_DOUBLE_EQ(DataLoader::safe_div(1.0, 0.0), 0.0);
}

TEST(DataLoader_Utils, EmaAlphaFromHalfLife_Bounds) {
    double a1 = DataLoader::ema_alpha_from_half_life(10.0);
    EXPECT_GT(a1, 0.0);
    EXPECT_LE(a1, 1.0);

    double a2 = DataLoader::ema_alpha_from_half_life(0.0);
    EXPECT_DOUBLE_EQ(a2, 1.0);
}

TEST_F(DataLoaderToy, AdjustOHLCbyAdjClose_NoAdj_NoCrash) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    // should be no-op if no adj close present
    dl.adjustOHLCbyAdjCloseIfAny();
    EXPECT_TRUE(std::isfinite(dl.C().sum()));
}

TEST_F(DataLoaderToy, MetaRead_PopulatesStockInfo) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    ASSERT_TRUE(dl.readMetaCsv((dir/"meta.csv").string()));
    EXPECT_EQ(dl.stock_info_.size(), dl.symbols_.size());
}

TEST_F(DataLoaderToy, MetaMissingFile_ReturnFalse) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    EXPECT_FALSE(dl.readMetaCsv((dir/"meta_missing.csv").string()));
}

TEST_F(DataLoaderToy, BrokersMissingFile_DoesNotCrash) {
    // If your loader supports missing brokers file gracefully
    fs::remove(dir/"brokers.csv");
    DataLoader dl(dir.string());
    // load may still succeed (depends on your implementation). We accept either true or false,
    // but it must not crash and must leave broker_strength finite if it succeeds.
    bool ok = dl.loadDailyPanelAndBuildFeatures();
    if (ok) {
        EXPECT_TRUE(std::isfinite(dl.feat_BrokerStrength().sum()));
    } else {
        SUCCEED();
    }
}

TEST_F(DataLoaderToy, DailyPanel_ReadsAllRows) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.loadDailyPanelAndBuildFeatures());
    // ensure at least the last day close is correct for some symbol
    int col2330 = (dl.symbols()[0] == "2330") ? 0 : 1;
    EXPECT_NEAR(dl.C()(2, col2330), 13.0, 1e-12);
}

TEST_F(DataLoaderToy, TurnoverShare_HandlesZeroSumVal) {
    // Make VAL zero for one day so sumVAL=0; TurnoverShare should be 0s (or finite)
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    dl.VAL_.row(0).setZero();
    dl.buildLiquidityTurnoverShare();
    EXPECT_TRUE(std::isfinite(dl.feat_turnover_share_.row(0).sum()));
}

TEST_F(DataLoaderToy, Imbalance_FallbackUsesHLCRange) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    // Ensure H>L so range>0, and C at mid => imbalance near 0
    dl.H_(0,0) = 12;
    dl.L_(0,0) = 10;
    dl.C_(0,0) = 11;
    dl.buildImbalance();
    EXPECT_NEAR(dl.feat_imb_(0,0), 0.0, 1e-6);

}

TEST_F(DataLoaderToy, Imbalance_ZeroRangeGivesZero) {
    DataLoader dl(dir.string());
    ASSERT_TRUE(dl.readDailyPricesCsv((dir/"daily_60d.csv").string()));
    dl.ensureShapes();
    dl.H_(0,0) = 10;
    dl.L_(0,0) = 10; // range=0
    dl.buildImbalance();
    EXPECT_NEAR(dl.feat_Imbalance()(0,0), 0.0, 1e-6);

}

// -----------------------
// End: 40 tests
// -----------------------
