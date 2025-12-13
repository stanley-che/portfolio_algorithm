//twse_quotes_dump.hpp
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "twse_quotes.hpp"
#include "twse_quotes_core.hpp"

namespace twse {

using json = nlohmann::json;

// ---------------- public ----------------
inline TwseQuoteDumper::TwseQuoteDumper(std::string out_csv)
    : out_csv_(std::move(out_csv)) {}

inline bool TwseQuoteDumper::dump_recent_days(const DumpOptions& opt) {
    auto items = fetch_code_list(opt);
    if (items.empty()) {
        std::cerr << "[twse] code list empty.\n";
        return false;
    }
    return dump_impl(items, opt);
}

inline bool TwseQuoteDumper::dump_recent_days_for_codes(const std::vector<std::string>& codes,
                                                        const DumpOptions& opt) {
    if (codes.empty()) return false;

    auto all = fetch_code_list(opt);
    std::unordered_map<std::string, std::string> code2name;
    code2name.reserve(all.size());
    for (auto& x : all) code2name[x.code] = x.name;

    std::vector<CodeItem> items;
    items.reserve(codes.size());
    for (auto& c : codes) {
        auto it = code2name.find(c);
        items.push_back(CodeItem{c, (it==code2name.end()? "" : it->second)});
    }
    return dump_impl(items, opt);
}

// ---------------- API helpers ----------------
inline std::vector<CodeItem> TwseQuoteDumper::fetch_code_list(const DumpOptions& opt) {
    std::string body;
    if (!http_get("https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL", body)) {
        std::cerr << "[twse] BWIBBU_ALL fetch failed\n";
        return {};
    }

    json j = json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_array()) {
        std::cerr << "[twse] BWIBBU_ALL parse failed\n";
        return {};
    }

    std::vector<CodeItem> out;
    out.reserve(j.size());

    for (auto& r : j) {
        if (!r.contains("Code") || !r["Code"].is_string()) continue;
        std::string c = r["Code"].get<std::string>();

        if (opt.listed_only_digits) {
            bool digits = !c.empty() && std::all_of(c.begin(), c.end(),
                [](unsigned char x){ return std::isdigit(x); });
            if (!digits) continue;
        }

        std::string name = (r.contains("Name") && r["Name"].is_string())
            ? r["Name"].get<std::string>() : "";
        out.push_back({c, name});
    }
    return out;
}

// ---------------- dump pipeline ----------------
inline bool TwseQuoteDumper::dump_impl(const std::vector<CodeItem>& items, const DumpOptions& opt) {
    YMD td = today_local();
    std::vector<YMD> months;
    months.reserve(std::max(1, opt.months_back));
    for(int k=0;k<std::max(1, opt.months_back); ++k){
        YMD m = add_months({td.y, td.m, 1}, -k);
        months.push_back({m.y, m.m, 1});
    }

    std::ofstream fout(out_csv_, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "[twse] cannot open output: " << out_csv_ << "\n";
        return false;
    }
    fout << "Code,Name,Date,Open,High,Low,Close,Change,Volume,Turnover,Trades\n";

    std::string resp;
    for (size_t idx=0; idx<items.size(); ++idx) {
        const auto& ci = items[idx];
        int written = 0;
        std::unordered_set<std::string> seen_dates;
        seen_dates.reserve((size_t)opt.days * 2);

        for (auto& m : months) {
            std::ostringstream url;
            url << "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json"
                << "&date=" << yyyymm01(m)
                << "&stockNo=" << ci.code;

            if (!http_get(url.str(), resp)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(opt.throttle_ms));
                continue;
            }

            json j = json::parse(resp, nullptr, false);
            if (j.is_discarded() || !j.contains("data")) {
                std::this_thread::sleep_for(std::chrono::milliseconds(opt.throttle_ms));
                continue;
            }

            for (auto& row : j["data"]) {
                if (!row.is_array() || row.size() < 9) continue;

                std::string d_ad  = roc_to_ad_date(row[0].get<std::string>());
                if (seen_dates.count(d_ad)) continue;
                seen_dates.insert(d_ad);

                double open  = to_double(row[3].get<std::string>());
                double high  = to_double(row[4].get<std::string>());
                double low   = to_double(row[5].get<std::string>());
                double close = to_double(row[6].get<std::string>());
                double change= parse_change_field(row[7].get<std::string>());

                long long vol      = to_ll(row[1].get<std::string>());
                long long turnover = to_ll(row[2].get<std::string>());
                long long trades   = to_ll(row[8].get<std::string>());

                fout << ci.code << "," << csv_escape(ci.name) << ","
                     << d_ad << ","
                     << open << "," << high << "," << low << "," << close << ","
                     << change << "," << vol << "," << turnover << "," << trades << "\n";

                if (++written >= opt.days) break;
            }

            if (written >= opt.days) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.throttle_ms));
        }

        if ((idx+1) % 100 == 0) std::cerr << "[twse] done " << (idx+1) << " codes...\n";
    }

    std::cerr << "[twse] OK -> " << out_csv_ << "\n";
    return true;
}

} // namespace twse
