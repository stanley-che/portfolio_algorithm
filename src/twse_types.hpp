//twse_types.hpp
#pragma once
#include <string>
#include <vector>

namespace twse {

struct CodeItem {
    std::string code;
    std::string name;
};

struct YMD {
    int y=1970, m=1, d=1;
};

struct DumpOptions {
    int days = 60;             // 每檔最多輸出幾天
    int throttle_ms = 120;     // 每次 API 呼叫間隔
    int months_back = 4;       // 往前抓幾個月（建議 4 足夠 cover 60 交易日）
    bool listed_only_digits = true; // 只收 4-digit(或純數字) code
};

} // namespace twse
