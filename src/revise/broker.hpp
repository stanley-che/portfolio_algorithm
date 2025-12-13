#pragma once

// Header-only library 版本
// 使用方式：
//   #include "broker.hpp"
//   #include <curl/curl.h>
//
//   int main() {
//       curl_global_init(CURL_GLOBAL_DEFAULT);      // 使用前做一次
//       BrokerFetcher fetcher("daily_60d.csv");
//       bool ok = fetcher.run();                   // 會輸出 brokers.csv
//       curl_global_cleanup();                     // 程式結束前
//   }
//
// 需要: -lcurl 與 nlohmann/json
// g++ -O2 -std=c++17 main.cpp -lcurl -o main

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class BrokerFetcher {
public:
    /// 建構時指定 daily 檔名（預設 daily_60d.csv）
    explicit BrokerFetcher(const std::string& dailyCsv = "daily_60d.csv")
        : dailyCsv_(dailyCsv) {}

    /// 執行整個流程：讀 daily_60d.csv → 呼叫 T86 API → 寫 brokers.csv
    /// 成功回傳 true，失敗回傳 false
    ///
    /// 注意：
    ///   呼叫前請先在程式某處呼叫一次
    ///     curl_global_init(CURL_GLOBAL_DEFAULT);
    ///   結束時再呼叫
    ///     curl_global_cleanup();
    bool run() {
        return fetchFromT86();
    }

private:
    std::string dailyCsv_;

    // ================== 靜態 helper functions ==================

    // libcurl 寫入 callback
    static size_t writeCallback(void* c, size_t s, size_t n, void* outp) {
        reinterpret_cast<std::string*>(outp)->append(
            static_cast<char*>(c),
            s * n
        );
        return s * n;
    }

    // 簡單 GET 封裝
    static bool httpGet(const std::string& url, std::string& body) {
        body.clear();
        CURL* h = curl_easy_init();
        if (!h) return false;

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "User-Agent: Mozilla/5.0");
        headers = curl_slist_append(headers, "Accept: application/json,text/plain,*/*");
        headers = curl_slist_append(headers, "Referer: https://www.twse.com.tw/");

        curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &BrokerFetcher::writeCallback);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 15L);
        curl_easy_setopt(h, CURLOPT_TIMEOUT, 30L);

        auto rc = curl_easy_perform(h);
        curl_slist_free_all(headers);
        curl_easy_cleanup(h);
        return rc == CURLE_OK;
    }

    // 去掉 UTF-8 BOM（目前沒用到，但留著當工具）
    static void stripBom(std::string& s) {
        if (s.size() >= 3 &&
            static_cast<unsigned char>(s[0]) == 0xEF &&
            static_cast<unsigned char>(s[1]) == 0xBB &&
            static_cast<unsigned char>(s[2]) == 0xBF) {
            s.erase(0, 3);
        }
    }

    // 轉換 yyyy-mm-dd / ROC yyyy/mm/dd → YYYYMMDD
    static std::string toYYYYMMDD(const std::string& d) {
        // yyyy-mm-dd
        if (d.size() == 10 && d[4] == '-' && d[7] == '-') {
            return d.substr(0, 4) + d.substr(5, 2) + d.substr(8, 2);
        }

        // ROC yyyy/mm/dd or yy/mm/dd
        int y = 0, m = 0, dd = 0;
        if (std::sscanf(d.c_str(), "%d/%d/%d", &y, &m, &dd) == 3) {
            if (y < 200) y += 1911;  // 只有民國年才 +1911
            char buf[16];
            std::sprintf(buf, "%04d%02d%02d", y, m, dd);
            return buf;
        }

        // already YYYYMMDD
        if (d.size() == 8 &&
            std::all_of(d.begin(), d.end(),
                        [](unsigned char ch) { return std::isdigit(ch); })) {
            return d;
        }

        // fallback: 原樣丟回去
        return d;
    }

    // 真正執行 T86 的下載與 brokers.csv 輸出
    bool fetchFromT86() {
        std::ifstream f(dailyCsv_);
        if (!f.is_open()) {
            std::cerr << "open fail: " << dailyCsv_ << "\n";
            return false;
        }

        std::string header;
        if (!std::getline(f, header)) {
            std::cerr << "empty daily\n";
            return false;
        }

        auto splitCsv = [](const std::string& line) {
            std::vector<std::string> out;
            std::string cur;
            bool q = false;
            for (char ch : line) {
                if (ch == '"') {
                    q = !q;
                } else if (ch == ',' && !q) {
                    out.push_back(cur);
                    cur.clear();
                } else {
                    cur.push_back(ch);
                }
            }
            out.push_back(cur);
            return out;
        };

        auto trim = [](const std::string& s) {
            size_t a = 0, b = s.size();
            while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
            while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
            return s.substr(a, b - a);
        };

        auto lower = [trim](std::string s) {
            for (auto& c : s)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };

        auto h = splitCsv(header);
        int iCode = -1, iDate = -1;
        for (int i = 0; i < static_cast<int>(h.size()); ++i) {
            std::string k = lower(trim(h[i]));
            if (k == "code" || k == "symbol") iCode = i;
            if (k == "date") iDate = i;
        }
        if (iCode < 0 || iDate < 0) {
            std::cerr << "need Code/Date columns\n";
            return false;
        }

        std::set<std::string> syms, dates;

        for (std::string line; std::getline(f, line);) {
            if (line.empty()) continue;
            auto c = splitCsv(line);
            if (static_cast<int>(c.size()) <= std::max(iCode, iDate)) continue;
            std::string sym = trim(c[iCode]);
            std::string d   = trim(c[iDate]);
            if (!sym.empty()) syms.insert(sym);
            if (!d.empty())   dates.insert(d);
        }
        f.close();

        std::vector<std::string> dlist(dates.begin(), dates.end());
        std::sort(dlist.begin(), dlist.end());
        const int MAX_D = 60;
        if (static_cast<int>(dlist.size()) > MAX_D) {
            dlist.erase(dlist.begin(), dlist.end() - MAX_D);
        }

        std::ofstream out("brokers.csv");
        out << "date,symbol,broker_netbuy_topk\n";

        for (const auto& d : dlist) {
            std::string ymd = toYYYYMMDD(d);
            std::string url =
                "https://www.twse.com.tw/fund/T86?response=json&selectType=ALL&date=" + ymd;

            std::string body;
            if (!httpGet(url, body)) {
                std::cerr << "[T86] http fail " << ymd << "\n";
                continue;
            }

            auto j = nlohmann::json::parse(body, nullptr, false);
            if (j.is_discarded()) {
                std::cerr << "[T86] parse fail " << ymd << "\n";
                continue;
            }
            if (!j.contains("stat") || j["stat"] != "OK") {
                // 不是交易日或查無資料
                std::cerr << "[T86] skip " << ymd << " stat="
                          << (j.contains("stat") ? j["stat"].get<std::string>() : "")
                          << "\n";
                continue;
            }
            if (!j.contains("data")) continue;

            auto to_ll_clean = [](const std::string& s) -> long long {
                std::string t;
                for (char ch : s) {
                    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-')
                        t.push_back(ch);
                }
                if (t.empty() || t == "-") return 0LL;
                try {
                    return std::stoll(t);
                } catch (...) {
                    return 0LL;
                }
            };

            // code → 合計差
            std::unordered_map<std::string, long long> diff;

            for (auto& row : j["data"]) {
                if (!row.is_array() || row.size() < 3) continue;
                std::string code = row[0].get<std::string>();
                if (!syms.count(code)) continue;  // 只留 universe 裡的

                long long val = 0;
                // 從最後一欄往前找第一個可 parse 的數字（合計差）
                for (int k = static_cast<int>(row.size()) - 1; k >= 2; --k) {
                    if (row[k].is_string()) {
                        val = to_ll_clean(row[k].get<std::string>());
                        break;
                    }
                }
                diff[code] = val;
            }

            for (const auto& s : syms) {
                auto it = diff.find(s);
                if (it != diff.end()) {
                    out << d << "," << s << "," << it->second << "\n";
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }

        std::cerr << "brokers.csv ready\n";
        return true;
    }
};
