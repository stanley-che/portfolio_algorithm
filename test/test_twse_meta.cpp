/*
g++ -std=c++17 \
  -Isrc/revise \
  -Ithird_party/third_party/googletest \
  -Ithird_party/third_party/googletest/googletest \
  -Ithird_party/third_party/googletest/googletest/include \
  test/test_twse_quotes.cpp \
  test/test_twse_meta.cpp \
  test/test_broker_unit.cpp \
  third_party/third_party/googletest/googletest/src/gtest-all.cc \
  third_party/third_party/googletest/googletest/src/gtest_main.cc \
  -lpthread -lcurl \    
  -o test_all
*/
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#define private public
#include "twse_meta.hpp"
#undef private

namespace fs = std::filesystem;
using meta::TwseMetaBuilder;
using meta::MetaOptions;
using meta::MetaRow;

static void write_file(const fs::path& p, const std::string& s){
    std::ofstream f(p);
    ASSERT_TRUE(f.is_open()) << "cannot write " << p;
    f << s;
}

TEST(TwseMeta_Utils, SplitCsvLine_Quotes) {
    auto v = TwseMetaBuilder::split_csv_line("a,\"b,c\",d");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], "a");
    EXPECT_EQ(v[1], "b,c");
    EXPECT_EQ(v[2], "d");
}

TEST(TwseMeta_ReadSymbols, AcceptCodeOrSymbolOrChinese) {
    fs::path dir = "test_meta_case";
    fs::create_directories(dir);

    // 1) code 欄位
    write_file(dir/"daily_code.csv",
        "date,code,close\n"
        "2024-01-01,2330,100\n"
        "2024-01-02,2303,50\n"
        "2024-01-03,2330,101\n"
    );

    // 2) symbol 欄位
    write_file(dir/"daily_symbol.csv",
        "date,symbol,close\n"
        "2024-01-01,0050,100\n"
        "2024-01-02,2330,50\n"
    );

    // 3) 中文「代號」
    write_file(dir/"daily_cn.csv",
        "日期,公司代號,收盤\n"
        "2024-01-01,2881,50\n"
    );

    TwseMetaBuilder b;
    MetaOptions opt;

    std::vector<std::string> syms;

    ASSERT_TRUE(b.read_symbols_from_daily_csv((dir/"daily_code.csv").string(), syms, opt));
    EXPECT_EQ(syms.size(), 2u); // 去重後 2330, 2303

    ASSERT_TRUE(b.read_symbols_from_daily_csv((dir/"daily_symbol.csv").string(), syms, opt));
    EXPECT_EQ(syms.size(), 2u);

    ASSERT_TRUE(b.read_symbols_from_daily_csv((dir/"daily_cn.csv").string(), syms, opt));
    EXPECT_EQ(syms.size(), 1u);
    EXPECT_EQ(syms[0], "2881");
}

TEST(TwseMeta_ETF, LooksLikeEtfRules) {
    EXPECT_TRUE(TwseMetaBuilder::looks_like_etf("0050", "ETF", ""));
    EXPECT_TRUE(TwseMetaBuilder::looks_like_etf("00878", "", "ETF"));
    EXPECT_TRUE(TwseMetaBuilder::looks_like_etf("00919", "", "受益證券"));
    EXPECT_TRUE(TwseMetaBuilder::looks_like_etf("0050", "etf", ""));
    EXPECT_TRUE(TwseMetaBuilder::looks_like_etf("006208", "上市", "etf"));

    // code 00xx 規則
    EXPECT_TRUE(TwseMetaBuilder::looks_like_etf("00123", "", ""));

    // 非 ETF
    EXPECT_FALSE(TwseMetaBuilder::looks_like_etf("2330", "上市", "股票"));
}

TEST(TwseMeta_Compose, MissingMetaAndLotInference) {
    TwseMetaBuilder b;
    MetaOptions opt;
    opt.infer_etf_lot = true;

    std::vector<std::string> symbols = {"0050", "2330", "XXXX"};

    // 手工建立 idx（避免打網路 fetch）
    std::unordered_map<std::string, TwseMetaBuilder::TwseMetaRec> idx;
    idx["0050"] = {"ETF產業", "ETF", "受益證券"};
    idx["2330"] = {"半導體", "上市", "股票"};
    // "XXXX" 不放 -> missing

    std::vector<MetaRow> rows;
    int miss=0, etf=0;

    ASSERT_TRUE(b.compose_rows(symbols, idx, rows, miss, etf, opt));
    ASSERT_EQ(rows.size(), 3u);

    // 0050 -> ETF -> lot=100
    EXPECT_EQ(rows[0].symbol, "0050");
    EXPECT_EQ(rows[0].lot, 100);

    // 2330 -> 非ETF -> lot=1000
    EXPECT_EQ(rows[1].symbol, "2330");
    EXPECT_EQ(rows[1].lot, 1000);

    // XXXX -> missing meta
    EXPECT_EQ(rows[2].symbol, "XXXX");
    EXPECT_EQ(rows[2].industry, "-1");
    EXPECT_EQ(rows[2].group, "-1");

    EXPECT_EQ(miss, 1);
    EXPECT_EQ(etf, 1);
}

TEST(TwseMeta_Write, WriteMetaCsv) {
    fs::path dir = "test_meta_out";
    fs::create_directories(dir);

    std::vector<MetaRow> rows = {
        {"2330", 1000, 1, "半導體", "上市", 1},
        {"0050",  100, 1, "ETF",    "ETF",  1},
    };

    TwseMetaBuilder b;
    fs::path out = dir/"meta.csv";

    ASSERT_TRUE(b.write_meta_csv(out.string(), rows));

    std::ifstream f(out);
    ASSERT_TRUE(f.is_open());

    std::string header;
    std::getline(f, header);
    EXPECT_EQ(header, "symbol,lot,odd_lot_unit,industry,group,tradable_flag");

    int line_cnt = 0;
    for(std::string line; std::getline(f, line); ){
        if(!line.empty()) ++line_cnt;
    }
    EXPECT_EQ(line_cnt, 2);
}
