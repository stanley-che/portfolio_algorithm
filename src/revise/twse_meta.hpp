//twse_meta.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace meta {

struct MetaRow {
    std::string symbol;
    int lot = 1000;
    int odd_lot_unit = 1;
    std::string industry = "-1";
    std::string group    = "-1";
    int tradable_flag = 1;
};

struct MetaOptions {
    std::string twse_meta_url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L";

    // 代號欄位候選：code/symbol/含「代號」
    bool daily_header_relaxed = true;

    // ETF lot 規則：ETF 100，其餘 1000
    bool infer_etf_lot = true;

    // group 用哪個欄位（你原本用 市場別/上市別）
    bool group_use_market_field = true;

    // curl timeout
    long connect_timeout_sec = 15;
    long timeout_sec = 30;
};

class TwseMetaBuilder {
public:
    TwseMetaBuilder() = default;

    // 讀 daily_csv -> 抓 TWSE meta -> 寫 meta.csv
    bool build_meta_csv_from_daily(const std::string& daily_csv,
                                   const std::string& out_meta_csv = "meta.csv",
                                   const MetaOptions& opt = MetaOptions{});

    // 也提供：直接給 symbols 產 meta rows（不寫檔）
    bool build_meta_rows_for_symbols(const std::vector<std::string>& symbols,
                                     std::vector<MetaRow>& out_rows,
                                     const MetaOptions& opt = MetaOptions{});

private:
    struct TwseMetaRec {
        std::string industry;
        std::string market;
        std::string secType;
    };

    // pipeline steps
    bool read_symbols_from_daily_csv(const std::string& daily_csv,
                                    std::vector<std::string>& out_symbols,
                                    const MetaOptions& opt);

    bool fetch_twse_meta_index(std::unordered_map<std::string, TwseMetaRec>& out_index,
                              const MetaOptions& opt);

    bool compose_rows(const std::vector<std::string>& symbols,
                      const std::unordered_map<std::string, TwseMetaRec>& idx,
                      std::vector<MetaRow>& out_rows,
                      int& out_missing_meta,
                      int& out_etf_count,
                      const MetaOptions& opt);

    bool write_meta_csv(const std::string& out_csv,
                        const std::vector<MetaRow>& rows);

    // helpers
    static bool looks_like_etf(const std::string& code,
                              const std::string& market,
                              const std::string& secType);

    static std::string trim(const std::string& s);
    static std::string lower(std::string s);
    static void strip_bom(std::string& s);
    static std::vector<std::string> split_csv_line(const std::string& line);

    // http/json helpers（在 twse_meta_http.hpp 實作）
    static bool http_get_json(const std::string& url, std::string& body,
                              long connect_timeout_sec, long timeout_sec);
};

} // namespace twse

// header-only: include impl
#include "twse_meta_http.hpp"
#include "twse_meta_impl.hpp"
