// twse_quotes.hpp
#pragma once
#include <string>
#include <vector>
#include "twse_types.hpp"

namespace twse {
class TwseQuoteDumper {
public:
    explicit TwseQuoteDumper(std::string out_csv = "daily_60d.csv");
    bool dump_recent_days(const DumpOptions& opt = DumpOptions{});
    bool dump_recent_days_for_codes(const std::vector<std::string>& codes,
                                    const DumpOptions& opt = DumpOptions{});
private:
    std::string out_csv_;
    static size_t write_cb(void* c, size_t s, size_t n, void* outp);
    static bool http_get(const std::string& url, std::string& body);

    static std::vector<CodeItem> fetch_code_list(const DumpOptions& opt);

    static YMD today_local();
    static YMD add_months(YMD a, int delta);
    static std::string yyyymm01(YMD a);
    static std::string roc_to_ad_date(const std::string& roc);

    static std::string csv_escape(const std::string& s);
    static std::string remove_commas(const std::string& s);
    static double to_double(const std::string& s);
    static long long to_ll(const std::string& s);
    static double parse_change_field(std::string s);

    bool dump_impl(const std::vector<CodeItem>& items, const DumpOptions& opt);
};
} // namespace twse

#include "twse_quotes_dump.hpp" // << 只 include 這個即可（它會 include core）
