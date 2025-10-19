// g++ -O2 -std=c++17 get_last_60d.cpp -lcurl -o get_last_60d
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <thread>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <iomanip>

using json = nlohmann::json;
using namespace std::chrono_literals;

// ---------------- curl helpers ----------------
static size_t write_cb(void* c, size_t s, size_t n, void* outp) {
    static_cast<std::string*>(outp)->append((char*)c, s*n); return s*n;
}
static bool http_get(const std::string& url, std::string& body) {
    body.clear();
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    CURLcode rc = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return rc == CURLE_OK;
}

// -------------- date helpers ------------------
struct YMD { int y,m,d; };
static YMD today_local() {
    std::time_t t = std::time(nullptr);
    std::tm tm{}; localtime_r(&t,&tm);
    return {tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday};
}
static YMD add_months(YMD a, int delta) {
    int y = a.y, m = a.m + delta, d = a.d;
    while (m<=0) { m+=12; y--; }
    while (m>12){ m-=12; y++; }
    int mdays[]={0,31,28,31,30,31,30,31,31,30,31,30,31};
    bool leap = (y%4==0 && y%100!=0) || (y%400==0);
    if (m==2 && leap) mdays[2]=29;
    if (d>mdays[m]) d=mdays[m];
    return {y,m,d};
}
static std::string yyyymmdd(YMD a){ std::ostringstream o; o<<a.y<<std::setw(2)<<std::setfill('0')<<a.m<<"01"; return o.str(); }

// -------------- utils -------------------------
static std::string csv_escape(const std::string& s){
    bool need=false; std::string out; out.reserve(s.size()+8);
    for(char c:s){ if(c=='"'||c==','||c=='\n'||c=='\r') need=true; out += (c=='"')? "\"\"" : std::string(1,c); }
    return need?("\""+out+"\""):out;
}
static std::string remove_commas(const std::string& s){
    std::string t; t.reserve(s.size());
    for(char c:s) if(c!=',') t.push_back(c);
    return t;
}
static double to_double(const std::string& s){
    try { return std::stod(remove_commas(s)); } catch(...) { return 0.0; }
}
static long long to_ll(const std::string& s){
    try { return std::stoll(remove_commas(s)); } catch(...) { return 0LL; }
}
// 114/08/15 → 2025-08-15 （ROC→AD）
static std::string roc_to_ad_date(const std::string& roc){
    // "114/08/15"
    int y=0,m=0,d=0;
    if(sscanf(roc.c_str(), "%d/%d/%d", &y,&m,&d)==3){
        y += 1911;
        std::ostringstream o;
        o<<y<<"-"<<std::setw(2)<<std::setfill('0')<<m<<"-"<<std::setw(2)<<d;
        return o.str();
    }
    return roc;
}

// ---------------- main ------------------------
int main(int argc, char** argv){
    int days = 60;    // 取最近 60 天
    int throttle_ms = 120; // 每次 API 呼叫之間稍微休息，避免過度頻繁
    if (argc>=2) days = std::max(1, std::atoi(argv[1]));

    // 1) 代碼清單（上市）
    std::cout << "Fetching code list (BWIBBU_ALL)...\n";
    std::string body;
    if(!http_get("https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL", body)){
        std::cerr<<"BWIBBU_ALL fetch failed\n"; return 1;
    }
    json jcodes = json::parse(body);
    struct CodeItem { std::string code, name; };
    std::vector<CodeItem> codes; codes.reserve(jcodes.size());
    for(auto& r: jcodes){
        if(r.contains("Code") && r["Code"].is_string()){
            std::string c = r["Code"].get<std::string>();
            bool digits = !c.empty() && std::all_of(c.begin(), c.end(), [](unsigned char x){return std::isdigit(x);});
            if(digits){
                std::string n = r.contains("Name") ? r["Name"].get<std::string>() : "";
                codes.push_back({c,n});
            }
        }
    }
    std::cout<<"Codes: "<<codes.size()<<"\n";

    // 2) 需要涵蓋的月份（多取 3~4 個月以涵蓋 60 天）
    YMD td = today_local();
    std::vector<YMD> months;
    // 至少取最近 4 個月（足夠覆蓋 60 個交易日）
    for(int k=0;k<4;++k){
        YMD m = add_months({td.y,td.m,1}, -k);
        months.push_back({m.y,m.m,1});
    }

    // 3) 準備輸出
    std::ofstream fout("daily_60d.csv", std::ios::binary);
    const unsigned char bom[3] = {0xEF,0xBB,0xBF};
    fout.write((const char*)bom,3);
    fout << "Code,Name,Date,Open,High,Low,Close,Change,Volume,Turnover,Trades\n";

    // 4) 每檔逐月抓，保留最近 60 天
    for(size_t idx=0; idx<codes.size(); ++idx){
        const auto& ci = codes[idx];
        int written = 0;
        std::unordered_set<std::string> seen_dates; // 避免跨月重複

        for(auto& m : months){
            std::ostringstream url;
            url << "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json"
                << "&date=" << yyyymmdd(m)      // YYYYMM01
                << "&stockNo=" << ci.code;

            std::string resp;
            if(!http_get(url.str(), resp)){ std::this_thread::sleep_for(std::chrono::milliseconds(throttle_ms)); continue; }
            json j = json::parse(resp, nullptr, false);
            if(j.is_discarded() || !j.contains("data")){ std::this_thread::sleep_for(std::chrono::milliseconds(throttle_ms)); continue; }

            // j["data"] 是每日一列的陣列（中文欄位）
            // 典型欄位：日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
            for (auto& row : j["data"]) {
                if (!row.is_array() || row.size() < 9) continue;
                std::string d_roc = row[0].get<std::string>();
                std::string d_ad  = roc_to_ad_date(d_roc);

                if (seen_dates.count(d_ad)) continue;
                seen_dates.insert(d_ad);

                double open  = to_double(row[3].get<std::string>());
                double high  = to_double(row[4].get<std::string>());
                double low   = to_double(row[5].get<std::string>());
                double close = to_double(row[6].get<std::string>());
                std::string chg_str = row[7].get<std::string>();
                // 漲跌價差可能含▲/▼或空白，去掉非數字符號
                for(char& c: chg_str) if(c==' '||c=='\t'||c==',') c=' ';
                // 只取數字與 - . 號
                std::string chg_clean;
                for(char c: chg_str) if (std::isdigit((unsigned char)c)||c=='-'||c=='.') chg_clean.push_back(c);
                double change = to_double(chg_clean);

                long long vol      = to_ll(row[1].get<std::string>());
                long long turnover = to_ll(row[2].get<std::string>());
                long long trades   = to_ll(row[8].get<std::string>());

                fout << ci.code << "," << csv_escape(ci.name) << ","
                     << d_ad << ","
                     << open << "," << high << "," << low << "," << close << ","
                     << change << "," << vol << "," << turnover << "," << trades << "\n";

                if(++written >= days) break;
            }
            if(written >= days) break;

            std::this_thread::sleep_for(std::chrono::milliseconds(throttle_ms));
        }

        // 小提示：大量代碼會跑比較久，可以只給你關注的清單來加速
        if ((idx+1)%100==0) std::cerr << "done " << (idx+1) << " codes...\n";
    }

    fout.close();
    std::cout << "OK -> daily_60d.csv\n";
    return 0;
}

