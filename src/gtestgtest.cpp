#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include "data_loader.h"

// 讓我們能執行你的 main()
int run_pipeline();

TEST(GlobalPipelineTest, SmokeRun) {
    // 1. 執行整個 pipeline（ETL → SOCP → DSA）
    int rc = run_pipeline();
    EXPECT_EQ(rc, 0) << "Pipeline main() did not return 0";

    // 2. 確認 SOCP/DSA pipeline 至少產生 CSV/Excel
    std::ifstream f1("./advance_parameter_csv/trade_plan.csv");
    EXPECT_TRUE(f1.good()) << "trade_plan.csv not generated";

    // 3. 可以檢查檔案大小是否 > 0，表示內容不是空的
    f1.seekg(0, std::ios::end);
    EXPECT_GT(f1.tellg(), 10) << "trade_plan.csv is empty or corrupted";

    // 4. 確認至少一列 order（簡易檢查）
    std::ifstream f2("./advance_parameter_csv/symbols_output.csv");
    EXPECT_TRUE(f2.good()) << "symbols_output.csv missing";

    
}
