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
    std::ifstream fin(base_path_ + "/" + filename);
    if (!fin) return false;

    std::string line;
    if (!std::getline(fin, line)) return false; // header

    // 先清空（避免重複 load 時累積）
    dates_.clear(); symbols_.clear();
    date2row_.clear(); sym2col_.clear();
    stock_info_.clear();

    // 先用 vector 暫存資料列（如果你本來就是兩階段 allocate，照你的做法）
    // 這裡示範「邊讀邊收 date/symbol，資料寫入你原本的暫存結構」
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        auto cols = splitCsvLine(line);
        if (cols.size() < 2) continue;

        std::string d   = normalize_date(trim(cols[0]));
        std::string sym = trim(cols[1]);

        // date register
        int r;
        auto itD = date2row_.find(d);
        if (itD == date2row_.end()) {
            r = (int)dates_.size();
            dates_.push_back(d);
            date2row_[d] = r;
        } else {
            r = itD->second;
        }

        // ✅ symbol register / filter
        if (!accept_or_register_symbol_(sym)) {
            continue; // 超過 max_symbols 的 symbol 整列跳過
        }
        int c = sym2col_[sym];

        // ====== 下面照你 csv 欄位寫入 O/H/L/C/V/VAL... ======
        // 例：假設 cols[2]=open cols[3]=high cols[4]=low cols[5]=close cols[6]=vol cols[7]=val
        // 你要用你自己的 index
        // double o = parse_num_clean(cols[2]); ...
        // 暫存 or 直接寫 matrix（看你原本架構）
    }

    // ✅ 最後照你原本流程 allocate matrices： rows=dates_.size(), cols=symbols_.size()
    ensureShapes();

    // ✅ 如果你原本是二階段（先收集再寫入矩陣），這裡把資料寫進 O_/H_/... 略
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
