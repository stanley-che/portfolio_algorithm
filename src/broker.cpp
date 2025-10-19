// g++ -O2 -std=c++17 src/broker.cpp -lcurl -o broker
// 需要: -lcurl 與 nlohmann/json
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
#include<set>
using namespace std;
using json = nlohmann::json;

static size_t write_cb(void* c, size_t s, size_t n, void* outp){
    ((string*)outp)->append((char*)c, s*n); return s*n;
}
static bool http_get(const string& url, string& body){
    body.clear();
    CURL* h = curl_easy_init();
    if(!h) return false;
    struct curl_slist* headers=nullptr;
    headers = curl_slist_append(headers, "User-Agent: Mozilla/5.0");
    headers = curl_slist_append(headers, "Accept: application/json,text/plain,*/*");
    headers = curl_slist_append(headers, "Referer: https://www.twse.com.tw/");
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 30L);
    auto rc = curl_easy_perform(h);
    curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    return rc==CURLE_OK;
}

// 轉 yyyy-mm-dd / ROC yyyy/mm/dd → YYYYMMDD
static string toYYYYMMDD(const string& d){
    if (d.size()==10 && d[4]=='-' && d[7]=='-')
        return d.substr(0,4)+d.substr(5,2)+d.substr(8,2);
    int y=0,m=0,dd=0;
    if (sscanf(d.c_str(), "%d/%d/%d",&y,&m,&dd)==3){
        if (y < 200) y += 1911;  // 只有民國年才 +1911
        char buf[16]; sprintf(buf,"%04d%02d%02d",y,m,dd);
        return buf;
    }
    if (d.size()==8 && all_of(d.begin(), d.end(), ::isdigit)) return d;
    return d;
}

// 用 T86 取三大法人合計買賣超
static bool fetch_brokers_from_T86(const string& daily_csv){
    // 讀 daily_60d.csv → 取 symbol + date（只保留最後 60 天）
    ifstream f(daily_csv);
    if(!f.is_open()){ cerr<<"open fail: "<<daily_csv<<"\n"; return false; }
    string header; if(!getline(f,header)){ cerr<<"empty daily\n"; return false; }
    auto splitCsv = [](const string& line){
        vector<string> out; string cur; bool q=false;
        for(char ch: line){
            if(ch=='"') q=!q;
            else if(ch==',' && !q){ out.push_back(cur); cur.clear(); }
            else cur.push_back(ch);
        }
        out.push_back(cur); return out;
    };
    auto trim = [](const string& s){
        size_t a=0,b=s.size(); while(a<b && isspace((unsigned char)s[a])) ++a;
        while(b>a && isspace((unsigned char)s[b-1])) --b; return s.substr(a,b-a);
    };
    auto lower = [](string s){ for(auto& c:s) c=(char)tolower((unsigned char)c); return s; };

    auto h = splitCsv(header);
    int iCode=-1, iDate=-1;
    for(int i=0;i<(int)h.size();++i){
        string k = lower(trim(h[i]));
        if(k=="code"||k=="symbol") iCode=i;
        if(k=="date") iDate=i;
    }
    if(iCode<0||iDate<0){ cerr<<"need Code/Date columns\n"; return false; }

    set<string> syms, dates;
    for(string line; getline(f,line); ){
        if(line.empty()) continue;
        auto c = splitCsv(line);
        if((int)c.size()<=max(iCode,iDate)) continue;
        string sym = trim(c[iCode]), d = trim(c[iDate]);
        if(!sym.empty()) syms.insert(sym);
        if(!d.empty())   dates.insert(d);
    }
    f.close();

    vector<string> dlist(dates.begin(), dates.end());
    sort(dlist.begin(), dlist.end());
    const int MAX_D = 60;
    if ((int)dlist.size() > MAX_D) dlist.erase(dlist.begin(), dlist.end()-MAX_D);

    ofstream out("brokers.csv");
    out << "date,symbol,broker_netbuy_topk\n";

    for (const auto& d : dlist){
        string ymd = toYYYYMMDD(d);
        string url = "https://www.twse.com.tw/fund/T86?response=json&selectType=ALL&date="+ymd;
        string body;
        if(!http_get(url, body)){ cerr<<"[T86] http fail "<<ymd<<"\n"; continue; }
        auto j = json::parse(body, nullptr, false);
        if(j.is_discarded()){ cerr<<"[T86] parse fail "<<ymd<<"\n"; continue; }
        if(!j.contains("stat") || j["stat"]!="OK"){  // 不是交易日或查無資料
            cerr<<"[T86] skip "<<ymd<<" stat="<<(j.contains("stat")?j["stat"].get<string>():"")<<"\n";
            continue;
        }
        if(!j.contains("data")) continue;

        // data: [ 證券代號, 證券名稱, 外資買, 外資賣, 外資差, 投信買, 投信賣, 投信差, 自營買, 自營賣, 自營差, 合計買, 合計賣, 合計差 ]
        // 我們取「合計差」欄（index 14 或 13，依版本；保險起見從右往左找第一個可解析成數字的欄位）
        auto to_ll_clean = [](const string& s)->long long{
            string t; for(char ch: s) if(isdigit((unsigned char)ch)||ch=='-') t.push_back(ch);
            if(t.empty() || t=="-" ) return 0LL;
            try{ return stoll(t); }catch(...){ return 0LL; }
        };

        unordered_map<string,long long> diff;
        for (auto& row : j["data"]){
            if(!row.is_array() || row.size()<3) continue;
            string code = row[0].get<string>();
            if (!syms.count(code)) continue;       // 只留 universe 裡的
            long long val = 0;
            for (int k=(int)row.size()-1; k>=2; --k){
                if(row[k].is_string()){ val = to_ll_clean(row[k].get<string>()); break; }
            }
            diff[code] = val;
        }

        for (const auto& s : syms){
            auto it = diff.find(s);
            if (it!=diff.end())
                out << d << "," << s << "," << it->second << "\n";
        }
        this_thread::sleep_for(chrono::milliseconds(120));
    }
    cerr<<"brokers.csv ready\n";
    return true;
}

// ---- 放在檔案最後 ----
int main(int argc, char** argv){
    std::string daily = "daily_60d.csv";      // 預設檔名
    if (argc >= 2) daily = argv[1];           // 也可以從參數指定

    curl_global_init(CURL_GLOBAL_DEFAULT);

    bool ok = fetch_brokers_from_T86(daily);  // 會輸出 brokers.csv 到目前目錄
    if (!ok) {
        std::cerr << "failed to fetch brokers from T86 using " << daily << "\n";
        curl_global_cleanup();
        return 1;
    }

    curl_global_cleanup();
    std::cout << "Done. Wrote brokers.csv\n";
    return 0;
}


