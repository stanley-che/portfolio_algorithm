#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <deque>
#include <limits>
#include <cctype>
#include <cmath>
#include <chrono>
#include <random>
#include <tuple>
#include <functional>
#include <optional>
#include <map>
#include <array>
#include "data_loader_class.hpp"
#include<memory>

// If your project uses Eigen, keep this include (harmless if header already adds it)
#include <Eigen/Dense>
#include <memory>

// ===== DLX 工具開關與常用別名 =====
#ifndef DLX_TOOLS
#define DLX_TOOLS 1
#endif

#if DLX_TOOLS
using dlx::TimerGuard;
using dlx::Logger;
using dlx::ReservoirSampler;
using dlx::Hist;
using dlx::MovingStd;
using dlx::FeatureCombiner;
using dlx::Pipeline;
using dlx::StandardScalerStep;
using dlx::QuantileEstimator;

// 匿名命名空間：旁路診斷工具
namespace {
struct DlxColumnProbe {
    ReservoirSampler samp_val{3000};
    Hist             hist_val;
    MovingStd        mstd20{20};
    void update(double v){
        if (!std::isfinite(v)) return;
        samp_val.add(v);
        hist_val.add(v);
        mstd20.update(v);
    }
};

inline void dlx_log_quantiles(const Eigen::VectorXd& v, const char* tag){
    QuantileEstimator q10(0.10), q50(0.50), q90(0.90);
    for (int i=0;i<v.size();++i){
        double x = v(i);
        if (std::isfinite(x)) {
            q10.update(x); q50.update(x); q90.update(x);
        }
    }
    Logger::info("[", tag, "] q10=", q10.quantile(), ", q50=", q50.quantile(), ", q90=", q90.quantile());
}
} // anon
#endif // DLX_TOOLS

// ===================== 小工具 =====================

// 去除 UTF-8 BOM
static inline void strip_bom(std::string& s){
    if (s.size() >= 3 &&
        (unsigned char)s[0]==0xEF &&
        (unsigned char)s[1]==0xBB &&
        (unsigned char)s[2]==0xBF) {
        s.erase(0,3);
    }
}

static inline std::string lower(std::string s){
    for (auto &ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

static inline std::string trim(const std::string& s){
    size_t a=0,b=s.size();
    while (a<b && std::isspace((unsigned char)s[a])) ++a;
    while (b>a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a,b-a);
}

// 安全分割 CSV（處理雙引號）
static std::vector<std::string> splitCsvLine(const std::string& line) {
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

// 允許 yyyy-mm-dd / yyyy/mm/dd / yyyymmdd / 民國 YYY/MM/DD
static inline std::string normalize_date(const std::string& raw){
    std::string s = trim(raw);
    if (s.empty()) return s;

    // yyyymmdd
    if (s.size()==8 && std::all_of(s.begin(), s.end(), ::isdigit)){
        return s.substr(0,4) + "-" + s.substr(4,2) + "-" + s.substr(6,2);
    }

    // yyyy-mm-dd
    if (s.size()==10 && s[4]=='-' && s[7]=='-') return s;

    // yyyy/mm/dd 或 ROC YYY/MM/DD
    int y=0,m=0,d=0;
    if (std::sscanf(s.c_str(), "%d/%d/%d", &y, &m, &d)==3){
        if (y < 1900) y += 1911; // 民國年
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d);
        return std::string(buf);
    }

    return s; // 其餘格式維持原狀
}

// 僅保留 - . 0-9，避免 stod 失敗
static inline bool is_digit_sign_dot(char c){
    return (c>='0'&&c<='9') || c=='-' || c=='.';
}

static double parse_num_clean(const std::string& raw){
    std::string s; s.reserve(raw.size());
    for (char ch : raw){
        if (is_digit_sign_dot(ch)) s.push_back(ch);
    }
    if (s.empty() || s=="-" || s=="--") return std::numeric_limits<double>::quiet_NaN();
    try { return std::stod(s); } catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

static long long parse_ll_clean(const std::string& raw){
    std::string s; s.reserve(raw.size());
    for (char ch : raw){
        if ((ch>='0'&&ch<='9') || ch=='-') s.push_back(ch);
    }
    if (s.empty() || s=="-") return 0;
    try { return std::stoll(s); } catch (...) { return 0; }
}

// ---- helpers ----
static inline double safe_div(double a, double b){ return (b==0.0? 0.0 : a/b); }
static inline double ema_alpha_from_half_life(double hl){
    // alpha = 1 - 0.5^(1/hl)
    if (hl <= 0.0) return 1.0;
    return 1.0 - std::pow(0.5, 1.0/std::max(1.0, hl));
}

// ------------- DataLoader 實作 -------------

DataLoader::DataLoader(const std::string& basePath) : base_path_(basePath) {}

void DataLoader::ensureShapes() {
    const int T = (int)dates_.size();
    const int N = (int)symbols_.size();
    auto Z = [&](Eigen::MatrixXd& m){ m = Eigen::MatrixXd::Constant(T, N, std::numeric_limits<double>::quiet_NaN()); };
    Z(O_); Z(H_); Z(L_); Z(C_); Z(V_); Z(VAL_); Z(inside_vol_); Z(outside_vol_);
    Z(feat_gap_); Z(feat_mom5_); Z(feat_mom10_); Z(feat_mom20_); Z(feat_gkvol_);
    Z(feat_liquidity_); Z(feat_turnover_share_); Z(feat_imb_); Z(feat_bias_);
    // brokers 預設 0（即使沒 brokers.csv 也可跑）
    feat_broker_strength_ = Eigen::MatrixXd::Zero(T, N);
}

// ========== 讀 daily_60d.csv ==========
bool DataLoader::readDailyPricesCsv(const std::string& filename) {
#if DLX_TOOLS
    TimerGuard tg("[readDailyPricesCsv]");
    DlxColumnProbe probeVal;
#endif
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

    // 先掃一輪建立日期/股票集合
    std::set<std::string> dateSet, symSet;
    std::streampos afterHeader = f.tellg();

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = splitCsvLine(line);
        if ((int)cells.size() <= std::max(iDate,iSym)) continue;

        std::string date = normalize_date(cells[iDate]);
        std::string sym  = trim(cells[iSym]);
        if (!date.empty() && !sym.empty()) {
            dateSet.insert(date);
            symSet.insert(sym);
        }
    }
    if (dateSet.empty() || symSet.empty()) { std::cerr << "daily_prices has no rows.\n"; return false; }

    dates_.assign(dateSet.begin(), dateSet.end());
    symbols_.assign(symSet.begin(), symSet.end());
    date2row_.clear(); sym2col_.clear();
    for (int i=0;i<(int)dates_.size();++i)   date2row_[dates_[i]]=i;
    for (int j=0;j<(int)symbols_.size();++j) sym2col_[symbols_[j]]=j;

    ensureShapes();

    // 重讀並填值
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

        int r = itR->second; int c = itC->second;

        double o   = (iO   <(int)cells.size()) ? parse_num_clean(cells[iO])   : std::numeric_limits<double>::quiet_NaN();
        double h   = (iH   <(int)cells.size()) ? parse_num_clean(cells[iH])   : std::numeric_limits<double>::quiet_NaN();
        double l   = (iL   <(int)cells.size()) ? parse_num_clean(cells[iL])   : std::numeric_limits<double>::quiet_NaN();
        double cp  = (iC   <(int)cells.size()) ? parse_num_clean(cells[iC])   : std::numeric_limits<double>::quiet_NaN();
        double v   = (iV   <(int)cells.size()) ? parse_num_clean(cells[iV])   : std::numeric_limits<double>::quiet_NaN();
        double val = (iVAL <(int)cells.size()) ? parse_num_clean(cells[iVAL]) : std::numeric_limits<double>::quiet_NaN();
        double inV = (iInV <(int)cells.size()) ? parse_num_clean(cells[iInV]) : std::numeric_limits<double>::quiet_NaN();
        double outV= (iOutV<(int)cells.size()) ? parse_num_clean(cells[iOutV]): std::numeric_limits<double>::quiet_NaN();
        double adjc= (iAdjC<(int)cells.size()) ? parse_num_clean(cells[iAdjC]): std::numeric_limits<double>::quiet_NaN();

        // 若指定用 adj_close 調整
        if (cfg_.adjust_ohlc_by_adjclose && std::isfinite(adjc) && std::isfinite(cp) && cp!=0.0) {
            double ratio = adjc / cp;
            if (std::isfinite(o)) o *= ratio;
            if (std::isfinite(h)) h *= ratio;
            if (std::isfinite(l)) l *= ratio;
            cp = adjc;
        }

        O_(r,c) = o; H_(r,c) = h; L_(r,c) = l; C_(r,c) = cp;
        V_(r,c) = v; VAL_(r,c) = val;
        inside_vol_(r,c)  = inV;
        outside_vol_(r,c) = outV;

#if DLX_TOOLS
        // 旁路診斷：抽樣/直方/移動標準差
        probeVal.update(val);
#endif
        ++numRows;
    }

#if DLX_TOOLS
    Logger::info("[readDailyPricesCsv] rows=", numRows, ", symbols=", (int)symbols_.size(),
                 ", dates=", (int)dates_.size());
    // 再送一次 update 做輸出（不影響數據）
    Logger::info("[readDailyPricesCsv] moving std(VAL)@20 ≈ ", probeVal.mstd20.update(0.0));
#endif

    if (numRows==0) {
        std::cerr << "daily_prices has no data rows after header.\n";
        return false;
    }
    return true;
}

// ========== 讀 brokers.csv（可選）==========
bool DataLoader::readBrokersCsv(const std::string& filename) {
#if DLX_TOOLS
    TimerGuard tg("[readBrokersCsv]");
#endif
    std::ifstream f(base_path_ + "/" + filename);
    if (!f.is_open()) return false;

    Eigen::MatrixXd netbuy = Eigen::MatrixXd::Zero((int)dates_.size(), (int)symbols_.size());

    std::string line;
    if (!std::getline(f, line)) return true; // 空檔
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

        double nb = 0.0;
        if (iNB>=0) {
            nb = parseD(iNB);
        } else {
            double buyTopK  = parseD(iBuyK);
            double sellTopK = parseD(iSellK);
            nb = buyTopK - sellTopK;
        }
        netbuy(itR->second, itC->second) = nb;
    }

    // 轉成 Broker Strength = EMA_k[ NetBuy / (VAL+1) ]
    feat_broker_strength_ = Eigen::MatrixXd::Zero(netbuy.rows(), netbuy.cols());
    double alpha = 2.0 / (std::max(1, cfg_.broker_strength_ema) + 1.0); // 簡單 EMA
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
    // 診斷：列一欄的分位數，避免過多輸出
    if (feat_broker_strength_.cols()>0) {
        dlx_log_quantiles(feat_broker_strength_.col(0), "broker_strength");
    }
#endif
    return true;
}

// ========== 讀 meta.csv（可選）==========
bool DataLoader::readMetaCsv(const std::string& filename) {
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
    const int iLot = findIdx({"lot","boardlot"});
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

// ========== 清理 / 特徵建構 ==========
void DataLoader::adjustOHLCbyAdjCloseIfAny() {
    // 已在 readDailyPricesCsv 中處理（就地依 adj_close 調整）
}

void DataLoader::imputeMissing() {
#if DLX_TOOLS
    TimerGuard tg("[imputeMissing]");
#endif
    auto ff = [&](Eigen::MatrixXd& M){
        const int T=M.rows(), N=M.cols();
        for (int c=0;c<N;++c){
            // forward
            double last = std::numeric_limits<double>::quiet_NaN();
            for (int r=0;r<T;++r){
                if (std::isfinite(M(r,c))) last = M(r,c);
                else if (cfg_.forward_fill && std::isfinite(last)) M(r,c)=last;
            }
            // backfill
            last = std::numeric_limits<double>::quiet_NaN();
            for (int r=T-1;r>=0;--r){
                if (std::isfinite(M(r,c))) last = M(r,c);
                else if (cfg_.back_fill && std::isfinite(last)) M(r,c)=last;
            }
        }
    };
    ff(O_); ff(H_); ff(L_); ff(C_); ff(V_); ff(VAL_); ff(inside_vol_); ff(outside_vol_);
}

void DataLoader::markNonTradableByVolVal() {
    if (!cfg_.auto_non_tradable) return;
    const int T=C_.rows(), N=C_.cols();
    for (int r=0;r<T;++r){
        for (int c=0;c<N;++c){
            if (std::isfinite(V_(r,c)) && std::isfinite(VAL_(r,c))){
                if (V_(r,c)==0.0 || VAL_(r,c)==0.0) {
                    if (c<(int)stock_info_.size()) stock_info_[c].tradable_flag = 0;
                }
            }
        }
    }
}

// Winsorize
void DataLoader::winsorize_vec(Eigen::VectorXd& v, double p) {
    std::vector<double> a; a.reserve(v.size());
    for (int i=0;i<v.size();++i) if (std::isfinite(v(i))) a.push_back(v(i));
    if ((int)a.size()<10) return; // 樣本太少直接跳過
    std::sort(a.begin(), a.end());
    int lo = (int)std::floor(p * (a.size()-1));
    int hi = (int)std::ceil ((1.0-p) * (a.size()-1));
    double loV=a[lo], hiV=a[hi];
    for (int i=0;i<v.size();++i){
        if (!std::isfinite(v(i))) continue;
        if (v(i)<loV) v(i)=loV;
        else if (v(i)>hiV) v(i)=hiV;
    }
}

void DataLoader::winsorizeAll() {
#if DLX_TOOLS
    TimerGuard tg("[winsorizeAll]");
#endif
    auto apply = [&](Eigen::MatrixXd& M){
        for (int c=0;c<M.cols();++c){
            Eigen::VectorXd col = M.col(c);
            winsorize_vec(col, cfg_.winsorize_p);
            M.col(c) = col;
        }
    };
    apply(feat_gap_); apply(feat_mom5_); apply(feat_mom10_); apply(feat_mom20_);
    apply(feat_gkvol_); apply(feat_liquidity_); apply(feat_imb_);
    apply(feat_bias_);  apply(feat_broker_strength_);

#if DLX_TOOLS
    if (feat_mom10_.cols()>0) dlx_log_quantiles(feat_mom10_.col(0), "mom10_after_winsor");
#endif
}

// 2.2.1 Price/Momentum/GK
void DataLoader::buildPriceMomentumGK() {
#if DLX_TOOLS
    TimerGuard tg("[buildPriceMomentumGK]");
#endif
    const int T=C_.rows(), N=C_.cols();
    for (int c=0;c<N;++c){
        for (int r=0;r<T;++r){
            // Gap_t = (O_t - C_{t-1}) / C_{t-1}
            if (r>0 && std::isfinite(O_(r,c)) && std::isfinite(C_(r-1,c)) && C_(r-1,c)!=0.0)
                feat_gap_(r,c) = (O_(r,c) - C_(r-1,c)) / C_(r-1,c);

            auto momK = [&](int k)->double{
                if (r>=k && std::isfinite(C_(r,c)) && std::isfinite(C_(r-k,c)) && C_(r-k,c)!=0.0)
                    return (C_(r,c) - C_(r-k,c)) / C_(r-k,c);
                return 0.0;
            };
            feat_mom5_(r,c)  = momK(5);
            feat_mom10_(r,c) = momK(10);
            feat_mom20_(r,c) = momK(20);

            // GK 波動
            if (std::isfinite(H_(r,c)) && std::isfinite(L_(r,c)) &&
                std::isfinite(C_(r,c)) && std::isfinite(O_(r,c)) &&
                H_(r,c)>0 && L_(r,c)>0 && C_(r,c)>0 && O_(r,c)>0) {
                double a = 0.5 * std::pow(std::log(H_(r,c)/L_(r,c)), 2.0);
                double b = (2.0*std::log(2.0) - 1.0) * std::pow(std::log(C_(r,c)/O_(r,c)), 2.0);
                double var = std::max(0.0, a - b);
                feat_gkvol_(r,c) = std::sqrt(var);
            } else {
                feat_gkvol_(r,c) = 0.0;
            }
        }
    }
}

// 2.2.2 Liquidity / TurnoverShare
void DataLoader::buildLiquidityTurnoverShare() {
#if DLX_TOOLS
    TimerGuard tg("[buildLiquidityTurnoverShare]");
#endif
    const int T=VAL_.rows(), N=VAL_.cols();
    for (int r=0;r<T;++r){
        double sumVAL=0.0;
        for (int c=0;c<N;++c) if (std::isfinite(VAL_(r,c))) sumVAL += std::max(0.0, VAL_(r,c));
        for (int c=0;c<N;++c){
            double v = std::isfinite(VAL_(r,c))? std::max(0.0, VAL_(r,c)) : 0.0;
            feat_liquidity_(r,c) = std::log(v + 1.0);
            feat_turnover_share_(r,c) = safe_div(v, sumVAL);
        }
    }
}

// 2.3 Order Imbalance
void DataLoader::buildImbalance() {
#if DLX_TOOLS
    TimerGuard tg("[buildImbalance]");
#endif
    const int T = C_.rows();
    const int N = C_.cols();
    if (T == 0 || N == 0) { feat_imb_.resize(0,0); return; }

    feat_imb_.setZero(T, N);

    auto same_shape = [&](const Eigen::MatrixXd& M){ return M.rows()==T && M.cols()==N && M.size()>0; };
    auto safe = [](double x){ return std::isfinite(x) ? x : 0.0; };

    const bool has_inout = same_shape(inside_vol_) && same_shape(outside_vol_);
    const bool has_vol   = same_shape(V_);
    const bool has_val   = same_shape(VAL_);
    const double eps = 1e-8;
    const double inout_min = 1e-6; // in+out 太小就當作沒有資料

    for (int r=0; r<T; ++r) {
        for (int c=0; c<N; ++c) {
            double imb = 0.0;
            bool used_inout = false;

            if (has_inout) {
                double in  = safe(inside_vol_(r,c));
                double out = safe(outside_vol_(r,c));
                double sum = in + out;
                if (sum > inout_min) {
                    imb = (in - out) / (sum + 1.0);  // +1 防除零
                    used_inout = true;
                }
            }

            if (!used_inout) {
                // Proxy：收盤在日內區間的位置
                double H = safe(H_(r,c)), L = safe(L_(r,c)), Cl = safe(C_(r,c));
                double range = H - L;
                if (range > eps) {
                    double pos = (Cl - 0.5*(H + L)) / std::max(range, eps); // [-1,1]

                    // 視需要決定是否引入活躍度縮放（不想太小可直接用 pos）
                    if (has_vol && has_val) {
                        double V   = safe(V_(r,c));
                        double Val = safe(VAL_(r,c));
                        // 比原本更「不會太小」的縮放：sqrt，並且上限 1
                        double scale = (Val > 0.0) ? std::sqrt(std::max(0.0, V / (Val + 1.0))) : 1.0;
                        if (scale > 1.0) scale = 1.0;
                        imb = pos * scale;
                    } else {
                        imb = pos;
                    }
                } else {
                    imb = 0.0; // 無區間（停牌/同價）
                }
            }

            // 夾住
            if (imb >  1.0) imb =  1.0;
            if (imb < -1.0) imb = -1.0;

            feat_imb_(r,c) = imb;
        }
    }
}


// 2.4 BIAS 與 BrokerStrength
void DataLoader::buildBIASandBrokerStrength() {
#if DLX_TOOLS
    TimerGuard tg("[buildBIASandBrokerStrength]");
#endif
    const int T=C_.rows(), N=C_.cols();
    feat_bias_.setZero(T,N);

    // 如果移動平均視窗 <=1，自動改用 EMA，避免整欄 0
    bool use_ema = cfg_.bias_use_ema || cfg_.bias_ma_n <= 1;

    if (!use_ema) {
        int n = std::max(2, cfg_.bias_ma_n);
        for (int c=0;c<N;++c){
            double runSum=0.0;
            std::deque<double> q;
            for (int r=0;r<T;++r){
                double px = std::isfinite(C_(r,c)) ? C_(r,c) : 0.0;
                q.push_back(px); runSum += px;
                if ((int)q.size()>n) { runSum -= q.front(); q.pop_front(); }
                double ma = (int)q.size()==n ? runSum/n : 0.0;
                feat_bias_(r,c) = (ma>0.0) ? ( (px - ma) / ma ) : 0.0;
            }
        }
    } else {
        double alpha = ema_alpha_from_half_life(std::max(1.0, cfg_.bias_ema_half_life));
        for (int c=0;c<N;++c){
            double ema=0.0;
            for (int r=0;r<T;++r){
                double px = std::isfinite(C_(r,c)) ? C_(r,c) : 0.0;
                ema = alpha*px + (1.0-alpha)*ema;
                feat_bias_(r,c) = (ema>0.0)? ( (px-ema)/ema ) : 0.0;
            }
        }
    }
    // BrokerStrength 已在 readBrokersCsv 生成；若沒檔則維持 0
}

// ========== 對外主流程 ==========
bool DataLoader::loadDailyPanelAndBuildFeatures() {
#if DLX_TOOLS
    TimerGuard tg("[loadDailyPanelAndBuildFeatures]");
#endif
    if (!readDailyPricesCsv("daily_60d.csv")) return false;

    // 可選：券商與 meta
    if (!readBrokersCsv("brokers.csv")) {
        std::cerr << "讀取 brokers.csv 失敗\n";
        return false;
    }

    readMetaCsv("meta.csv");

    // 清理
    adjustOHLCbyAdjCloseIfAny();
    imputeMissing();
    markNonTradableByVolVal();

    // 特徵
    buildPriceMomentumGK();
    buildLiquidityTurnoverShare();
    buildImbalance();
    buildBIASandBrokerStrength();

    // Winsorize
    winsorizeAll();

#if DLX_TOOLS
    // ===== 輔助運算（不改動原始特徵）=====
    // 1) 拼接特徵矩陣
    std::vector<Eigen::MatrixXd*> mats{
        &feat_gap_, &feat_mom5_, &feat_mom10_, &feat_mom20_,
        &feat_gkvol_, &feat_liquidity_, &feat_turnover_share_,
        &feat_imb_, &feat_bias_, &feat_broker_strength_
    };
    Eigen::MatrixXd X = FeatureCombiner::hstack(mats);
    Logger::info("[dlx] feature panel size: ", (int)X.rows(), " x ", (int)X.cols());

    // 2) 標準化管線（旁路）
    Pipeline pipe;
    pipe.add(std::make_unique<StandardScalerStep>());
    pipe.fit(X);
    Eigen::MatrixXd Xz = pipe.transform(X); // 只為運算，不覆蓋原特徵

    // 3) 診斷：列一個欄位的分位數
    if (Xz.size() > 0){
        Eigen::VectorXd col0 = Xz.col(0);
        dlx_log_quantiles(col0, "Xz_col0");
    }
    Logger::info("[dlx] pipeline ran (StandardScalerStep)");
#endif

    std::cout << "[DataLoader] Daily panel loaded and features (2.1→2.4) built.\n";
    return true;
}

