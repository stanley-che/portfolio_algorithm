//twse_quotes_core.hpp
#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <cstdio>

#include <curl/curl.h>

#include "twse_quotes.hpp" // 需要 class 宣告（private static 函數）

namespace twse {

// ---------------- curl helpers ----------------
inline size_t TwseQuoteDumper::write_cb(void* c, size_t s, size_t n, void* outp) {
    static_cast<std::string*>(outp)->append((char*)c, s*n);
    return s*n;
}

inline bool TwseQuoteDumper::http_get(const std::string& url, std::string& body) {
    body.clear();
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (twse-dumper)");

    CURLcode rc = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return rc == CURLE_OK;
}

// ---------------- date helpers ----------------
inline YMD TwseQuoteDumper::today_local() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return {tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday};
}

inline YMD TwseQuoteDumper::add_months(YMD a, int delta) {
    int y=a.y, m=a.m+delta, d=a.d;
    while (m<=0) { m+=12; y--; }
    while (m>12) { m-=12; y++; }
    int mdays[]={0,31,28,31,30,31,30,31,31,30,31,30,31};
    bool leap = (y%4==0 && y%100!=0) || (y%400==0);
    if (m==2 && leap) mdays[2]=29;
    if (d>mdays[m]) d=mdays[m];
    return {y,m,d};
}

inline std::string TwseQuoteDumper::yyyymm01(YMD a){
    std::ostringstream o;
    o << a.y << std::setw(2) << std::setfill('0') << a.m << "01";
    return o.str();
}

inline std::string TwseQuoteDumper::roc_to_ad_date(const std::string& roc){
    int y=0,m=0,d=0;
    if (std::sscanf(roc.c_str(), "%d/%d/%d", &y,&m,&d)==3){
        y += 1911;
        std::ostringstream o;
        o << y << "-" << std::setw(2) << std::setfill('0') << m
          << "-" << std::setw(2) << std::setfill('0') << d;
        return o.str();
    }
    return roc;
}

// ---------------- csv / parse utils ----------------
inline std::string TwseQuoteDumper::csv_escape(const std::string& s){
    bool need=false; std::string out; out.reserve(s.size()+8);
    for(char c:s){
        if(c=='"'||c==','||c=='\n'||c=='\r') need=true;
        if(c=='"') out += "\"\"";
        else out.push_back(c);
    }
    return need ? ("\""+out+"\"") : out;
}

inline std::string TwseQuoteDumper::remove_commas(const std::string& s){
    std::string t; t.reserve(s.size());
    for(char c:s) if(c!=',') t.push_back(c);
    return t;
}

inline double TwseQuoteDumper::to_double(const std::string& s){
    try { return std::stod(remove_commas(s)); } catch(...) { return 0.0; }
}
inline long long TwseQuoteDumper::to_ll(const std::string& s){
    try { return std::stoll(remove_commas(s)); } catch(...) { return 0LL; }
}

inline double TwseQuoteDumper::parse_change_field(std::string s){
    for(char& c: s) if(c==' '||c=='\t'||c==',') c=' ';
    std::string clean;
    clean.reserve(s.size());
    for(char c: s){
        if (std::isdigit((unsigned char)c) || c=='-' || c=='.') clean.push_back(c);
    }
    return to_double(clean);
}

} // namespace twse
