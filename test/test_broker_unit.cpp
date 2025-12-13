//test_broker_unit.cpp
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
#include <fstream>
#include <filesystem>

#define private public
#include "broker.hpp"
#undef private

namespace fs = std::filesystem;

/* ===============================
 *  日期轉換：toYYYYMMDD
 * =============================== */

TEST(Broker_Util, ToYYYYMMDD_Normal) {
    EXPECT_EQ(BrokerFetcher::toYYYYMMDD("2024-01-03"), "20240103");
}

TEST(Broker_Util, ToYYYYMMDD_ROC) {
    EXPECT_EQ(BrokerFetcher::toYYYYMMDD("112/01/03"), "20230103");
}

TEST(Broker_Util, ToYYYYMMDD_ROC_Short) {
    EXPECT_EQ(BrokerFetcher::toYYYYMMDD("1/2/3"), "19120203");
}

TEST(Broker_Util, ToYYYYMMDD_Already) {
    EXPECT_EQ(BrokerFetcher::toYYYYMMDD("20240103"), "20240103");
}

TEST(Broker_Util, ToYYYYMMDD_Invalid) {
    EXPECT_EQ(BrokerFetcher::toYYYYMMDD("BAD"), "BAD");
}

/* ===============================
 *  BOM 處理
 * =============================== */

TEST(Broker_Util, StripBom) {
    std::string s = "\xEF\xBB\xBFdate,code\n";
    BrokerFetcher::stripBom(s);
    EXPECT_EQ(s, "date,code\n");
}
