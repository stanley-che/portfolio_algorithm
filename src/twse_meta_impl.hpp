//twse_meta_impl.hpp
#pragma once
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cctype>

namespace meta {

using json = nlohmann::json;

// ---------- small helpers ----------
inline std::string TwseMetaBuilder::trim(const std::string& s){
    size_t a=0,b=s.size();
    while(a<b && std::isspace((unsigned char)s[a])) ++a;
    while(b>a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a,b-a);
}
inline std::string TwseMetaBuilder::lower(std::string s){
    for(auto& ch:s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}
inline void TwseMetaBuilder::strip_bom(std::string& s){
    if (s.size() >= 3 &&
        (unsigned char)s[0]==0xEF && (unsigned char)s[1]==0xBB && (unsigned char)s[2]==0xBF)
        s.erase(0,3);
}
inline std::vector<std::string> TwseMetaBuilder::split_csv_line(const std::string& line){
    std::vector<std::string> out;
    std::string cur;
    bool q=false;
    for(char ch: line){
        if(ch=='"') q=!q;
        else if(ch==',' && !q){ out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    out.push_back(cur);
    return out;
}

inline bool TwseMetaBuilder::looks_like_etf(const std::string& code,
                                           const std::string& market,
                                           const std::string& secType){
    if (code.size()>=2 && code[0]=='0' && code[1]=='0') return true;

    auto L = [](std::string x){
        for(char& c:x) c=(char)std::tolower((unsigned char)c);
        return x;
    };
    std::string m = L(market), t = L(secType);

    if (m.find("etf")!=std::string::npos) return true;
    if (t.find("etf")!=std::string::npos) return true;
    // 中文關鍵字（受益證券）
    if (secType.find("受益")!=std::string::npos) return true;

    return false;
}

// ---------- public ----------
inline bool TwseMetaBuilder::build_meta_csv_from_daily(const std::string& daily_csv,
                                                      const std::string& out_meta_csv,
                                                      const MetaOptions& opt) {
    std::vector<std::string> symbols;
    if (!read_symbols_from_daily_csv(daily_csv, symbols, opt)) return false;

    std::vector<MetaRow> rows;
    if (!build_meta_rows_for_symbols(symbols, rows, opt)) return false;

    if (!write_meta_csv(out_meta_csv, rows)) return false;

    std::cerr << "[twse-meta] wrote " << out_meta_csv
              << " rows=" << rows.size() << "\n";
    return true;
}

inline bool TwseMetaBuilder::build_meta_rows_for_symbols(const std::vector<std::string>& symbols,
                                                        std::vector<MetaRow>& out_rows,
                                                        const MetaOptions& opt) {
    if (symbols.empty()) return false;

    std::unordered_map<std::string, TwseMetaRec> idx;
    if (!fetch_twse_meta_index(idx, opt)) return false;

    int miss=0, etf=0;
    out_rows.clear();
    out_rows.reserve(symbols.size());

    if (!compose_rows(symbols, idx, out_rows, miss, etf, opt)) return false;

    std::cerr << "[twse-meta] symbols=" << symbols.size()
              << ", idx=" << idx.size()
              << ", etf=" << etf
              << ", missing_meta=" << miss << "\n";
    return true;
}

// ---------- pipeline: read symbols ----------
inline bool TwseMetaBuilder::read_symbols_from_daily_csv(const std::string& daily_csv,
                                                        std::vector<std::string>& out_symbols,
                                                        const MetaOptions& opt) {
    std::ifstream f(daily_csv);
    if(!f.is_open()){
        std::cerr << "[twse-meta] open fail: " << daily_csv << "\n";
        return false;
    }

    std::string header;
    if(!std::getline(f, header)){
        std::cerr << "[twse-meta] empty: " << daily_csv << "\n";
        return false;
    }
    strip_bom(header);
    auto h = split_csv_line(header);

    int iCode = -1;
    for (int i=0;i<(int)h.size();++i){
        std::string k = lower(trim(h[i]));
        if (k=="code" || k=="symbol" || k.find("代號")!=std::string::npos){
            iCode = i; break;
        }
    }
    if (iCode < 0){
        std::cerr << "[twse-meta] daily csv needs Code/symbol column\n";
        return false;
    }

    std::set<std::string> used;
    for(std::string line; std::getline(f,line); ){
        if(line.empty()) continue;
        auto c = split_csv_line(line);
        if(iCode < (int)c.size()){
            std::string sym = trim(c[iCode]);
            if(!sym.empty()) used.insert(sym);
        }
    }

    out_symbols.assign(used.begin(), used.end());
    std::cerr << "[twse-meta] symbols in use: " << out_symbols.size() << "\n";
    return !out_symbols.empty();
}

// ---------- pipeline: fetch meta index ----------
inline bool TwseMetaBuilder::fetch_twse_meta_index(std::unordered_map<std::string, TwseMetaRec>& out_index,
                                                  const MetaOptions& opt) {
    std::string body;
    if(!http_get_json(opt.twse_meta_url, body, opt.connect_timeout_sec, opt.timeout_sec)){
        std::cerr << "[twse-meta] fetch TWSE meta failed\n";
        return false;
    }

    json j = json::parse(body, nullptr, false);
    if(j.is_discarded() || !j.is_array()){
        std::cerr << "[twse-meta] parse TWSE meta failed\n";
        return false;
    }

    out_index.clear();
    out_index.reserve(j.size()*2);

    auto getS = [](const json& r, const char* key)->std::string{
        auto it = r.find(key);
        if (it != r.end() && it->is_string()) return it->get<std::string>();
        return "";
    };

    for (auto& r : j){
        std::string code = getS(r, "公司代號");
        if(code.empty()) code = getS(r, "有價證券代號");
        if(code.empty()) continue;

        TwseMetaRec m;
        m.industry = getS(r, "產業別");
        m.market   = getS(r, "市場別");
        if(m.market.empty()) m.market = getS(r, "上市別");
        m.secType  = getS(r, "有價證券別");

        out_index[code] = std::move(m);
    }

    std::cerr << "[twse-meta] twse meta rows: " << out_index.size() << "\n";
    return !out_index.empty();
}

// ---------- pipeline: compose rows ----------
inline bool TwseMetaBuilder::compose_rows(const std::vector<std::string>& symbols,
                                         const std::unordered_map<std::string, TwseMetaRec>& idx,
                                         std::vector<MetaRow>& out_rows,
                                         int& out_missing_meta,
                                         int& out_etf_count,
                                         const MetaOptions& opt) {
    out_missing_meta = 0;
    out_etf_count = 0;

    for (const auto& code : symbols){
        MetaRow row;
        row.symbol = code;

        std::string industry = "-1";
        std::string group    = "-1";
        std::string secType  = "";

        auto it = idx.find(code);
        if (it != idx.end()){
            industry = it->second.industry.empty() ? "-1" : it->second.industry;
            group    = it->second.market.empty()   ? "-1" : it->second.market;
            secType  = it->second.secType;
        } else {
            ++out_missing_meta;
        }

        row.industry = industry;
        row.group    = group;

        bool is_etf = false;
        if (opt.infer_etf_lot) {
            is_etf = looks_like_etf(code, group, secType);
        }
        row.lot = is_etf ? 100 : 1000;
        if (is_etf) ++out_etf_count;

        row.odd_lot_unit = 1;
        row.tradable_flag = 1;

        out_rows.push_back(std::move(row));
    }
    return true;
}

// ---------- pipeline: write csv ----------
inline bool TwseMetaBuilder::write_meta_csv(const std::string& out_csv,
                                           const std::vector<MetaRow>& rows) {
    std::ofstream out(out_csv);
    if(!out.is_open()){
        std::cerr << "[twse-meta] cannot open out: " << out_csv << "\n";
        return false;
    }

    out << "symbol,lot,odd_lot_unit,industry,group,tradable_flag\n";
    for (const auto& r : rows){
        out << r.symbol << ","
            << r.lot << ","
            << r.odd_lot_unit << ","
            << r.industry << ","
            << r.group << ","
            << r.tradable_flag << "\n";
    }
    return true;
}

} // namespace twse
