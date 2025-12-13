//data_loader_utils.hpp
#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>

#include "data_loader.hpp"

namespace dlx {

// ---------------- ctor（也可放在 io.hpp，但放這也 OK） ----------------
inline DataLoader::DataLoader(std::string basePath) : base_path_(std::move(basePath)) {}

// ---------------- string utils ----------------
inline void DataLoader::strip_bom(std::string& s){
    if (s.size() >= 3 &&
        (unsigned char)s[0]==0xEF &&
        (unsigned char)s[1]==0xBB &&
        (unsigned char)s[2]==0xBF) {
        s.erase(0,3);
    }
}
inline std::string DataLoader::lower(std::string s){
    for (auto &ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}
inline std::string DataLoader::trim(const std::string& s){
    size_t a=0,b=s.size();
    while (a<b && std::isspace((unsigned char)s[a])) ++a;
    while (b>a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a,b-a);
}
inline std::vector<std::string> DataLoader::splitCsvLine(const std::string& line) {
    std::vector<std::string> out; out.reserve(32);
    std::string cur; cur.reserve(64);
    bool inQuotes = false;
    for (char ch : line){
        if (ch=='"') inQuotes = !inQuotes;
        else if (ch==',' && !inQuotes) { out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    out.push_back(cur);
    return out;
}

// 允許 yyyy-mm-dd / yyyy/mm/dd / yyyymmdd / ROC YYY/MM/DD
inline std::string DataLoader::normalize_date(const std::string& raw){
    std::string s = trim(raw);
    if (s.empty()) return s;

    if (s.size()==8 && std::all_of(s.begin(), s.end(), ::isdigit)){
        return s.substr(0,4) + "-" + s.substr(4,2) + "-" + s.substr(6,2);
    }
    if (s.size()==10 && s[4]=='-' && s[7]=='-') return s;

    int y=0,m=0,d=0;
    if (std::sscanf(s.c_str(), "%d/%d/%d", &y, &m, &d)==3){
        if (y < 1900) y += 1911;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d);
        return std::string(buf);
    }
    return s;
}

inline bool DataLoader::is_digit_sign_dot(char c){
    return (c>='0'&&c<='9') || c=='-' || c=='.';
}

inline double DataLoader::parse_num_clean(const std::string& raw){
    std::string s; s.reserve(raw.size());
    for (char ch : raw) if (is_digit_sign_dot(ch)) s.push_back(ch);
    if (s.empty() || s=="-" || s=="--") return std::numeric_limits<double>::quiet_NaN();
    try { return std::stod(s); } catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

inline long long DataLoader::parse_ll_clean(const std::string& raw){
    std::string s; s.reserve(raw.size());
    for (char ch : raw) if ((ch>='0'&&ch<='9') || ch=='-') s.push_back(ch);
    if (s.empty() || s=="-") return 0;
    try { return std::stoll(s); } catch (...) { return 0; }
}

} 
