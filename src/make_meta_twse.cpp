// g++ -O2 -std=c++17 src/make_meta_twse.cpp -lcurl -o make_meta_twse
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <bits/stdc++.h>
using namespace std;
using json = nlohmann::json;

// ---------- curl ----------
static size_t write_cb(void* c, size_t s, size_t n, void* outp){
    ((string*)outp)->append((char*)c, s*n); return s*n;
}
static bool http_get(const string& url, string& body){
    body.clear();
    CURL* h = curl_easy_init();
    if(!h) return false;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "User-Agent: Mozilla/5.0");
    headers = curl_slist_append(headers, "Accept: application/json,text/plain,*/*");
    headers = curl_slist_append(headers, "Referer: https://www.twse.com.tw/");
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, ""); // enable gzip/deflate
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 30L);
    auto rc = curl_easy_perform(h);
    curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    return rc == CURLE_OK;
}

// ---------- utils ----------
static string trim(const string& s){
    size_t a=0,b=s.size(); while(a<b && isspace((unsigned char)s[a])) ++a;
    while(b>a && isspace((unsigned char)s[b-1])) --b; return s.substr(a,b-a);
}
static vector<string> splitCsv(const string& line){
    vector<string> out; string cur; bool q=false;
    for(char ch: line){
        if(ch=='"') q=!q;
        else if(ch==',' && !q){ out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    out.push_back(cur); return out;
}
static string lower(string s){ for(auto& ch:s) ch=(char)tolower((unsigned char)ch); return s; }

// ---------- heuristics ----------
static bool looks_like_etf(const string& code, const string& market, const string& secType){
    // 台灣多數 ETF 代號以 "00" 開頭，如 0050、0056、006208...
    if (code.size()>=2 && code[0]=='0' && code[1]=='0') return true;
    auto L = [](string x){ for(char& c:x) c=(char)tolower((unsigned char)c); return x; };
    string m = L(market), t=L(secType);
    if (m.find("etf")!=string::npos) return true;
    if (t.find("etf")!=string::npos) return true;
    if (t.find("受益")!=string::npos) return true; // 受益證券
    return false;
}

// ---------- main ----------
int main(int argc, char** argv){
    if (argc<2){
        cerr << "Usage: " << argv[0] << " daily_60d.csv\n";
        return 1;
    }
    string daily_csv = argv[1];

    // 1) 讀 daily_60d.csv 取用到的代號集合
    ifstream f(daily_csv);
    if(!f.is_open()){ cerr<<"open fail: "<<daily_csv<<"\n"; return 1; }
    string header; if(!getline(f, header)){ cerr<<"empty: "<<daily_csv<<"\n"; return 1; }
    auto h = splitCsv(header);
    int iCode=-1;
    for(int i=0;i<(int)h.size();++i){
        string k = lower(trim(h[i]));
        if(k=="code" || k=="symbol"){ iCode=i; break; }
    }
    if(iCode<0){ cerr<<"daily_60d.csv needs Code/symbol column\n"; return 1; }
    set<string> used_syms;
    for(string line; getline(f,line); ){
        if(line.empty()) continue;
        auto c = splitCsv(line);
        if(iCode<(int)c.size()){
            string sym = trim(c[iCode]);
            if(!sym.empty()) used_syms.insert(sym);
        }
    }
    f.close();
    cerr<<"symbols in use: "<<used_syms.size()<<"\n";

    // 2) 抓 TWSE 基本資料
    string body;
    const string TWSE_META = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L";
    if(!http_get(TWSE_META, body)){
        cerr<<"fetch TWSE meta failed\n"; return 1;
    }
    json j = json::parse(body, nullptr, false);
    if(j.is_discarded()){ cerr<<"parse TWSE meta failed\n"; return 1; }

    // 3) 建索引：代號 -> (industry, market/board, secType)
    struct Meta { string industry; string market; string secType; };
    unordered_map<string, Meta> mp;
    mp.reserve(j.size()*2);

    auto getS = [](const json& r, const string& key)->string{
        if(r.contains(key) && r[key].is_string()) return r[key].get<string>();
        return "";
    };

    for (auto& r : j){
        // 常見欄位：公司代號、公司名稱、產業別、上市別/市場別、有價證券別...
        string code = getS(r, "公司代號");
        if(code.empty()) code = getS(r, "有價證券代號");
        if(code.empty()) continue;

        Meta m;
        m.industry = getS(r, "產業別");
        m.market   = getS(r, "市場別");          // 有些資料集用「上市別」或「市場別」
        if(m.market.empty()) m.market = getS(r, "上市別");
        m.secType  = getS(r, "有價證券別");

        mp[code] = m;
    }
    cerr<<"twse meta rows: "<<mp.size()<<"\n";

    // 4) 生成 meta.csv（只輸出 used_syms）
    ofstream out("src/meta.csv");
    out << "symbol,lot,odd_lot_unit,industry,group,tradable_flag\n";

    int wrote=0, miss=0, etfcnt=0;
    for (const auto& code : used_syms){
        auto it = mp.find(code);
        string industry="-1", group="-1", secType="";
        if(it!=mp.end()){
            industry = it->second.industry.empty()? "-1" : it->second.industry;
            group    = it->second.market.empty()?   "-1" : it->second.market;  // 暫用「市場別」當 group
            secType  = it->second.secType;
        } else {
            ++miss;
        }

        bool isETF = looks_like_etf(code, group, secType);
        int lot = isETF ? 100 : 1000;
        if(isETF) ++etfcnt;

        out << code << "," << lot << "," << 1 << ","
            << industry << "," << group << "," << 1 << "\n";
        ++wrote;
    }
    out.close();

    cout << "meta.csv written. rows="<<wrote
         << " (ETF lot=100 count="<<etfcnt<<", missing_meta="<<miss<<")\n";
    return 0;
}

