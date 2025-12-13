/*
GTEST=third_party/third_party/googletest

g++ -std=c++17 \
  -Isrc/revise \
  -Ithird_party/third_party/googletest \
  -Ithird_party/third_party/googletest/googletest \
  -Ithird_party/third_party/googletest/googletest/include \
  test/test_twse_quotes.cpp \
  third_party/third_party/googletest/googletest/src/gtest-all.cc \
  third_party/third_party/googletest/googletest/src/gtest_main.cc \
  -lpthread -lcurl \
  -o test_twse



*/
#include <gtest/gtest.h>
#include <cmath>

// 只在 test 內打開 private（測 private static helper）
#define private public
#include "twse_quotes.hpp"
#undef private

using namespace twse;

static void expect_near(double a, double b, double eps=1e-12) {
    EXPECT_TRUE(std::fabs(a - b) <= eps) << "a=" << a << " b=" << b;
}

/* =========================
 * 日期相關
 * ========================= */

TEST(TwseQuotes_Date, RocToAdDate_Normal) {
    EXPECT_EQ(TwseQuoteDumper::roc_to_ad_date("112/01/03"), "2023-01-03");
}

TEST(TwseQuotes_Date, RocToAdDate_EarlyROC) {
    EXPECT_EQ(TwseQuoteDumper::roc_to_ad_date("1/2/3"), "1912-02-03");
}

TEST(TwseQuotes_Date, RocToAdDate_Invalid) {
    EXPECT_EQ(TwseQuoteDumper::roc_to_ad_date("BAD"), "BAD");
    EXPECT_EQ(TwseQuoteDumper::roc_to_ad_date(""), "");
    EXPECT_EQ(TwseQuoteDumper::roc_to_ad_date("112-01-03"), "112-01-03"); // 不符合格式就原樣回傳(依你實作)
}

TEST(TwseQuotes_Date, AddMonths_ClampLeapYear) {
    YMD out = TwseQuoteDumper::add_months({2024, 3, 31}, -1);
    EXPECT_EQ(out.y, 2024);
    EXPECT_EQ(out.m, 2);
    EXPECT_EQ(out.d, 29);
}

TEST(TwseQuotes_Date, AddMonths_ClampNormalYear) {
    YMD out = TwseQuoteDumper::add_months({2023, 3, 31}, -1);
    EXPECT_EQ(out.y, 2023);
    EXPECT_EQ(out.m, 2);
    EXPECT_EQ(out.d, 28);
}

TEST(TwseQuotes_Date, Yyyymm01) {
    EXPECT_EQ(TwseQuoteDumper::yyyymm01({2024, 1, 15}), "20240101");
    EXPECT_EQ(TwseQuoteDumper::yyyymm01({2024, 11, 2}), "20241101");
}

/* =========================
 * CSV / 字串處理
 * ========================= */

TEST(TwseQuotes_String, CsvEscape) {
    EXPECT_EQ(TwseQuoteDumper::csv_escape("ABC"), "ABC");
    EXPECT_EQ(TwseQuoteDumper::csv_escape("A,B"), "\"A,B\"");
    EXPECT_EQ(TwseQuoteDumper::csv_escape("A\"B"), "\"A\"\"B\"");
    EXPECT_EQ(TwseQuoteDumper::csv_escape("A\nB"), "\"A\nB\"");
    EXPECT_EQ(TwseQuoteDumper::csv_escape(""), "");
}

TEST(TwseQuotes_String, RemoveCommas) {
    EXPECT_EQ(TwseQuoteDumper::remove_commas("1,234,567"), "1234567");
    EXPECT_EQ(TwseQuoteDumper::remove_commas("123"), "123");
    EXPECT_EQ(TwseQuoteDumper::remove_commas(""), "");
}

/* =========================
 * 數值轉換
 * ========================= */

TEST(TwseQuotes_Number, ToDouble) {
    expect_near(TwseQuoteDumper::to_double("1,234.5"), 1234.5);
    expect_near(TwseQuoteDumper::to_double("  3.14 "), 3.14);
    expect_near(TwseQuoteDumper::to_double("N/A"), 0.0);
    expect_near(TwseQuoteDumper::to_double(""), 0.0);
}

TEST(TwseQuotes_Number, ToLongLong) {
    EXPECT_EQ(TwseQuoteDumper::to_ll("9,876"), 9876LL);
    EXPECT_EQ(TwseQuoteDumper::to_ll(" 42 "), 42LL);
    EXPECT_EQ(TwseQuoteDumper::to_ll("N/A"), 0LL);
    EXPECT_EQ(TwseQuoteDumper::to_ll(""), 0LL);
}

/* =========================
 * 漲跌欄位解析（重要）
 * ========================= */

TEST(TwseQuotes_Change, ParseChangeField) {
    expect_near(TwseQuoteDumper::parse_change_field("+1.25"), 1.25);
    expect_near(TwseQuoteDumper::parse_change_field(" -0.5 "), -0.5);
    expect_near(TwseQuoteDumper::parse_change_field("▲1.2"), 1.2);
    expect_near(TwseQuoteDumper::parse_change_field("▼0.8"), -0.8); // 建議：下跌應該是負
    expect_near(TwseQuoteDumper::parse_change_field("N/A"), 0.0);
    expect_near(TwseQuoteDumper::parse_change_field(""), 0.0);

    // 常見奇怪字元（有些資料會有全形空白）
    expect_near(TwseQuoteDumper::parse_change_field("▲ 0.0"), 0.0);
    expect_near(TwseQuoteDumper::parse_change_field("▼ 0.0"), -0.0);
}
