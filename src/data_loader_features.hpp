#pragma once
#include <deque>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <memory>   // DLX_TOOLS pipeline 會用到 make_unique

#include "data_loader.hpp"
#include "data_loader_io.hpp"     // 如果 features 會用到 readXXX 或 ensureShapes
#include "data_loader_utils.hpp"  // 如果 features 會用到 parse/trim 等

// 如果你真的要開 DLX_TOOLS=1，請把 dlx 的那些工具宣告搬到這個檔案：
//   data_loader_dlx_tools.hpp (內含 using dlx::TimerGuard... + dlx_log_quantiles + FeatureCombiner/Pipeline 等)
// 然後在此 include。
#if DLX_TOOLS
  #include "data_loader_dlx_tools.hpp"
#endif

namespace dlx {

// ---------------- cleaning ----------------
inline void DataLoader::imputeMissing() {
#if DLX_TOOLS
    TimerGuard tg("[imputeMissing]");
#endif
    auto ff = [&](Eigen::MatrixXd& M){
        const int T=M.rows(), N=M.cols();
        for (int c=0;c<N;++c){
            double last = std::numeric_limits<double>::quiet_NaN();
            for (int r=0;r<T;++r){
                if (std::isfinite(M(r,c))) last = M(r,c);
                else if (cfg_.forward_fill && std::isfinite(last)) M(r,c)=last;
            }
            last = std::numeric_limits<double>::quiet_NaN();
            for (int r=T-1;r>=0;--r){
                if (std::isfinite(M(r,c))) last = M(r,c);
                else if (cfg_.back_fill && std::isfinite(last)) M(r,c)=last;
            }
        }
    };
    ff(O_); ff(H_); ff(L_); ff(C_);
    ff(V_); ff(VAL_); ff(inside_vol_); ff(outside_vol_);
}

inline void DataLoader::markNonTradableByVolVal() {
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

// 你目前 readDailyPricesCsv 已做 adj_close 調整，所以這裡給 no-op，避免缺符號
inline void DataLoader::adjustOHLCbyAdjCloseIfAny() { ... }


// ---------------- winsorize ----------------
inline void DataLoader::winsorize_vec(Eigen::VectorXd& v, double p) {
    std::vector<double> a; a.reserve(v.size());
    for (int i=0;i<v.size();++i) if (std::isfinite(v(i))) a.push_back(v(i));
    if ((int)a.size()<10) return;

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

inline void DataLoader::winsorizeAll() {
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
    apply(feat_gkvol_); apply(feat_liquidity_); apply(feat_turnover_share_);
    apply(feat_imb_); apply(feat_bias_); apply(feat_broker_strength_);

#if DLX_TOOLS
    if (feat_mom10_.cols()>0) dlx_log_quantiles(feat_mom10_.col(0), "mom10_after_winsor");
#endif
}

// ---------------- features ----------------
inline void DataLoader::buildPriceMomentumGK() {
#if DLX_TOOLS
    TimerGuard tg("[buildPriceMomentumGK]");
#endif
    const int T=C_.rows(), N=C_.cols();
    for (int c=0;c<N;++c){
        for (int r=0;r<T;++r){
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

inline void DataLoader::buildLiquidityTurnoverShare() {
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

inline void DataLoader::buildImbalance() {
#if DLX_TOOLS
    TimerGuard tg("[buildImbalance]");
#endif
    const int T=C_.rows();
    const int N=C_.cols();
    if (T==0 || N==0) { feat_imb_.resize(0,0); return; }

    feat_imb_.setZero(T, N);

    auto same_shape = [&](const Eigen::MatrixXd& M){ return M.rows()==T && M.cols()==N && M.size()>0; };
    auto safe = [](double x){ return std::isfinite(x) ? x : 0.0; };

    const bool has_inout = same_shape(inside_vol_) && same_shape(outside_vol_);
    const bool has_vol   = same_shape(V_);
    const bool has_val   = same_shape(VAL_);
    const double eps = 1e-8;
    const double inout_min = 1e-6;

    for (int r=0; r<T; ++r) {
        for (int c=0; c<N; ++c) {
            double imb = 0.0;
            bool used_inout = false;

            if (has_inout) {
                double in  = safe(inside_vol_(r,c));
                double out = safe(outside_vol_(r,c));
                double sum = in + out;
                if (sum > inout_min) {
                    imb = (in - out) / (sum + 1.0);
                    used_inout = true;
                }
            }

            if (!used_inout) {
                double Hh = safe(H_(r,c)), Ll = safe(L_(r,c)), Cl = safe(C_(r,c));
                double range = Hh - Ll;
                if (range > eps) {
                    double pos = (Cl - 0.5*(Hh + Ll)) / std::max(range, eps);
                    if (has_vol && has_val) {
                        double Vv   = safe(V_(r,c));
                        double Valv = safe(VAL_(r,c));
                        double scale = (Valv > 0.0) ? std::sqrt(std::max(0.0, Vv / (Valv + 1.0))) : 1.0;
                        if (scale > 1.0) scale = 1.0;
                        imb = pos * scale;
                    } else {
                        imb = pos;
                    }
                } else {
                    imb = 0.0;
                }
            }

            if (imb >  1.0) imb =  1.0;
            if (imb < -1.0) imb = -1.0;
            feat_imb_(r,c) = imb;
        }
    }
}

inline void DataLoader::buildBIASandBrokerStrength() {
#if DLX_TOOLS
    TimerGuard tg("[buildBIASandBrokerStrength]");
#endif
    const int T=C_.rows(), N=C_.cols();
    feat_bias_.setZero(T,N);

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
                double ma = ((int)q.size()==n) ? runSum/n : 0.0;
                feat_bias_(r,c) = (ma>0.0) ? ((px - ma) / ma) : 0.0;
            }
        }
    } else {
        double alpha = ema_alpha_from_half_life(std::max(1.0, cfg_.bias_ema_half_life));
        for (int c=0;c<N;++c){
            double ema=0.0;
            for (int r=0;r<T;++r){
                double px = std::isfinite(C_(r,c)) ? C_(r,c) : 0.0;
                ema = alpha*px + (1.0-alpha)*ema;
                feat_bias_(r,c) = (ema>0.0)? ((px-ema)/ema) : 0.0;
            }
        }
    }
    // feat_broker_strength_ already built in readBrokersCsv (or stays 0 if no file)
}

// ---------------- pipeline ----------------
inline bool DataLoader::loadDailyPanelAndBuildFeatures() {
#if DLX_TOOLS
    TimerGuard tg("[loadDailyPanelAndBuildFeatures]");
#endif
    if (!readDailyPricesCsv("daily_60d.csv")) return false;

    if (!readBrokersCsv("brokers.csv")) {
        std::cerr << "讀取 brokers.csv 失敗\n";
        return false;
    }

    (void)readMetaCsv("meta.csv");

    adjustOHLCbyAdjCloseIfAny();
    imputeMissing();
    markNonTradableByVolVal();

    buildPriceMomentumGK();
    buildLiquidityTurnoverShare();
    buildImbalance();
    buildBIASandBrokerStrength();

    winsorizeAll();

#if DLX_TOOLS
    std::vector<Eigen::MatrixXd*> mats{
        &feat_gap_, &feat_mom5_, &feat_mom10_, &feat_mom20_,
        &feat_gkvol_, &feat_liquidity_, &feat_turnover_share_,
        &feat_imb_, &feat_bias_, &feat_broker_strength_
    };
    Eigen::MatrixXd X = FeatureCombiner::hstack(mats);

    Pipeline pipe;
    pipe.add(std::make_unique<StandardScalerStep>());
    pipe.fit(X);
    Eigen::MatrixXd Xz = pipe.transform(X);
    if (Xz.size() > 0) dlx_log_quantiles(Xz.col(0), "Xz_col0");
#endif

    std::cout << "[DataLoader] Daily panel loaded and features built.\n";
    return true;
}

} // namespace dlx
