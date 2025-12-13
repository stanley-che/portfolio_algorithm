// data_loader_io.hpp
#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <initializer_list>

#include "data_loader.hpp"
#include "data_loader_utils.hpp"

#ifndef DLX_TOOLS
#define DLX_TOOLS 0
#endif

#if DLX_TOOLS
  #include "data_loader_dlx_tools.hpp"
#endif

namespace dlx {

inline void DataLoader::ensureShapes() {
    const int T = (int)dates_.size();
    const int N = (int)symbols_.size();
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    auto Znan = [&](Eigen::MatrixXd& m){ m = Eigen::MatrixXd::Constant(T, N, NaN); };

    Znan(O_); Znan(H_); Znan(L_); Znan(C_);
    Znan(V_); Znan(VAL_); Znan(inside_vol_); Znan(outside_vol_);

    Znan(feat_gap_); Znan(feat_mom5_); Znan(feat_mom10_); Znan(feat_mom20_); Znan(feat_gkvol_);
    Znan(feat_liquidity_); Znan(feat_turnover_share_);
    Znan(feat_imb_); Znan(feat_bias_);

    feat_broker_strength_ = Eigen::MatrixXd::Zero(T, N);
}

inline bool DataLoader::readDailyPricesCsv(const std::string& filename) {
    std::ifstream f(base_path_ + "/" + filename);
    if (!f.is_open()) { std::cerr << "Cannot open " << filename << "\n"; return false; }

    std::string line;
    if (!std::getline(f, line)) { std::cerr << "daily_prices is empty.\n"; return false; }
    strip_bom(line);

    auto headers = splitCsvLine(line);
    for (auto& h : headers) h = lower(trim(h));

    auto findIdx = [&](std::initializer_list<const char*> names)->int{
        for (auto n: names) {
            std::string key = lower(std::string(n));
            for (int i=0;i<(int)headers.size();++i)
                if (headers[i] == key) return i;
        }
        return -1;
    };

    const int iDate = findIdx({"date"});
    const int iSym  = findIdx({"symbol","code","ticker"});
    const int iO    = findIdx({"o","open"});
    const int iH    = findIdx({"h","high"});
    const int iL    = findIdx({"l","low"});
    const int iC    = findIdx({"c","close","adj close","adjclose","adjustedclose"});
    const int iV    = findIdx({"v","volume"});
    const int iVAL  = findIdx({"val","turnover","value","turnovervalue"});
    const int iInV  = findIdx({"inside_vol","insidevol","inside_volume"});
    const int iOutV = findIdx({"outside_vol","outsidevol","outside_volume"});
    const int iAdjC = findIdx({"adj_close","adj close","adjclose","adjustedclose"});

    if (iDate<0 || iSym<0 || iO<0 || iH<0 || iL<0 || iC<0 || iV<0 || iVAL<0) {
        std::cerr << "daily_prices header missing required columns.\n";
        return false;
    }

    std::set<std::string> dateSet, symSet;
    std::streampos afterHeader = f.tellg();

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = splitCsvLine(line);
        if ((int)cells.size() <= std::max(iDate,iSym)) continue;

        std::string date = normalize_date(cells[iDate]);
        std::string sym  = trim(cells[iSym]);
        if (!date.empty() && !sym.empty()) { dateSet.insert(date); symSet.insert(sym); }
    }
    if (dateSet.empty() || symSet.empty()) { std::cerr << "daily_prices has no rows.\n"; return false; }

    dates_.assign(dateSet.begin(), dateSet.end());
    symbols_.assign(symSet.begin(), symSet.end());
    date2row_.clear(); sym2col_.clear();
    for (int i=0;i<(int)dates_.size();++i)    date2row_[dates_[i]]=i;
    for (int j=0;j<(int)symbols_.size();++j)  sym2col_[symbols_[j]]=j;

    ensureShapes();

    f.clear(); f.seekg(afterHeader);

    int numRows = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = splitCsvLine(line);

        std::string date = normalize_date( (iDate<(int)cells.size()) ? cells[iDate] : "" );
        std::string sym  = trim        ( (iSym <(int)cells.size()) ? cells[iSym ] : "" );
        if (date.empty() || sym.empty()) continue;

        auto itR = date2row_.find(date);
        auto itC = sym2col_.find(sym);
        if (itR==date2row_.end() || itC==sym2col_.end()) continue;

        int r = itR->second, c = itC->second;

        double o   = (iO   <(int)cells.size()) ? parse_num_clean(cells[iO])   : std::numeric_limits<double>::quiet_NaN();
        double h   = (iH   <(int)cells.size()) ? parse_num_clean(cells[iH])   : std::numeric_limits<double>::quiet_NaN();
        double l   = (iL   <(int)cells.size()) ? parse_num_clean(cells[iL])   : std::numeric_limits<double>::quiet_NaN();
        double cp  = (iC   <(int)cells.size()) ? parse_num_clean(cells[iC])   : std::numeric_limits<double>::quiet_NaN();
        double v   = (iV   <(int)cells.size()) ? parse_num_clean(cells[iV])   : std::numeric_limits<double>::quiet_NaN();
        double val = (iVAL <(int)cells.size()) ? parse_num_clean(cells[iVAL]) : std::numeric_limits<double>::quiet_NaN();
        double inV = (iInV <(int)cells.size()) ? parse_num_clean(cells[iInV]) : std::numeric_limits<double>::quiet_NaN();
        double outV= (iOutV<(int)cells.size()) ? parse_num_clean(cells[iOutV]): std::numeric_limits<double>::quiet_NaN();
        double adjc= (iAdjC<(int)cells.size()) ? parse_num_clean(cells[iAdjC]): std::numeric_limits<double>::quiet_NaN();

        if (cfg_.adjust_ohlc_by_adjclose && std::isfinite(adjc) && std::isfinite(cp) && cp!=0.0) {
            double ratio = adjc / cp;
            if (std::isfinite(o)) o *= ratio;
            if (std::isfinite(h)) h *= ratio;
            if (std::isfinite(l)) l *= ratio;
            cp = adjc;
        }

        O_(r,c)=o; H_(r,c)=h; L_(r,c)=l; C_(r,c)=cp;
        V_(r,c)=v; VAL_(r,c)=val;
        inside_vol_(r,c)=inV; outside_vol_(r,c)=outV;

        ++numRows;
    }

    if (numRows==0) { std::cerr << "daily_prices has no data rows after header.\n"; return false; }
    return true;
}

inline bool DataLoader::readBrokersCsv(const std::string& filename) {
#if DLX_TOOLS
    TimerGuard tg("[readBrokersCsv]");
#endif
    std::ifstream f(base_path_ + "/" + filename);
    if (!f.is_open()) return false;

    Eigen::MatrixXd netbuy = Eigen::MatrixXd::Zero((int)dates_.size(), (int)symbols_.size());

    std::string line;
    if (!std::getline(f, line)) return true;
    strip_bom(line);

    auto headers = splitCsvLine(line);
    for (auto& h : headers) h = lower(trim(h));

    auto findIdx = [&](std::initializer_list<const char*> names)->int{
        for (auto n: names) {
            std::string key = lower(std::string(n));
            for (int i=0;i<(int)headers.size();++i)
                if (headers[i]==key) return i;
        }
        return -1;
    };

    const int iDate = findIdx({"date"});
    const int iSym  = findIdx({"symbol","code","ticker"});
    const int iNB   = findIdx({"broker_netbuy_topk","netbuytopk","netbuy"});
    const int iBuyK = findIdx({"broker_buy_topk","buytopk"});
    const int iSellK= findIdx({"broker_sell_topk","selltopk"});

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = splitCsvLine(line);

        std::string date = normalize_date( (iDate<(int)cells.size()) ? cells[iDate] : "" );
        std::string sym  = trim        ( (iSym <(int)cells.size()) ? cells[iSym ] : "" );
        if (date.empty() || sym.empty()) continue;

        auto itR = date2row_.find(date), itC = sym2col_.find(sym);
        if (itR==date2row_.end() || itC==sym2col_.end()) continue;

        auto parseD = [&](int idx)->double{
            if (idx<0 || idx>=(int)cells.size()) return 0.0;
            return (double)parse_ll_clean(cells[idx]);
        };

        double nb = (iNB>=0) ? parseD(iNB) : (parseD(iBuyK) - parseD(iSellK));
        netbuy(itR->second, itC->second) = nb;
    }

    feat_broker_strength_ = Eigen::MatrixXd::Zero(netbuy.rows(), netbuy.cols());
    double alpha = 2.0 / (std::max(1, cfg_.broker_strength_ema) + 1.0);

    for (int c=0;c<netbuy.cols();++c) {
        double s=0.0;
        for (int r=0;r<netbuy.rows();++r) {
            double val_denom = (std::isfinite(VAL_(r,c)) ? VAL_(r,c) : 0.0) + 1.0;
            double x = netbuy(r,c) / val_denom;
            s = alpha * x + (1.0 - alpha) * s;
            feat_broker_strength_(r,c) = s;
        }
    }

#if DLX_TOOLS
    if (feat_broker_strength_.cols()>0)
        dlx_log_quantiles(feat_broker_strength_.col(0), "broker_strength");
#endif
    return true;
}

inline bool DataLoader::readMetaCsv(const std::string& filename) {
#if DLX_TOOLS
    TimerGuard tg("[readMetaCsv]");
#endif
    std::ifstream f(base_path_ + "/" + filename);
    if (!f.is_open()) return false;

    std::string line;
    if (!std::getline(f, line)) return true;
    strip_bom(line);

    auto headers = splitCsvLine(line);
    for (auto& h : headers) h = lower(trim(h));

    auto findIdx = [&](std::initializer_list<const char*> names)->int{
        for (auto n: names) {
            std::string key = lower(std::string(n));
            for (int i=0;i<(int)headers.size();++i)
                if (headers[i]==key) return i;
        }
        return -1;
    };

    const int iSym = findIdx({"symbol","code","ticker"});
    const int iLot = findIdx({"lot","boardlot","lot_size"});
    const int iOdd = findIdx({"odd_lot_unit","oddlotunit","oddlot"});
    const int iInd = findIdx({"industry"});
    const int iGrp = findIdx({"group"});
    const int iTrad= findIdx({"tradable_flag","tradable"});

    std::unordered_map<std::string, StockInfo> mp;

    auto parseI = [](const std::vector<std::string>& v, int idx, int defVal)->int{
        if (idx<0 || idx>=(int)v.size()) return defVal;
        const std::string& x = v[idx];
        if (x.empty()) return defVal;
        try { return std::stoi(x); } catch(...) { return defVal; }
    };

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = splitCsvLine(line);

        StockInfo s;
        s.symbol = (iSym<(int)cells.size()) ? trim(cells[iSym]) : "";
        if (s.symbol.empty()) continue;

        s.lot           = parseI(cells, iLot, 1000);
        s.odd_lot_unit  = parseI(cells, iOdd, 1);
        s.industry      = parseI(cells, iInd, -1);
        s.group         = parseI(cells, iGrp, -1);
        s.tradable_flag = parseI(cells, iTrad, 1);

        mp[s.symbol]=s;
    }

    stock_info_.assign(symbols_.size(), StockInfo{});
    for (int j=0;j<(int)symbols_.size();++j) {
        auto it = mp.find(symbols_[j]);
        stock_info_[j] = (it==mp.end()? StockInfo{symbols_[j]} : it->second);
    }

#if DLX_TOOLS
    Logger::info("[readMetaCsv] loaded meta for symbols=", (int)stock_info_.size());
#endif
    return true;
}

} // namespace dlx
