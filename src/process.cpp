//process.cpp
#include "process.h"
#include <algorithm>
#include <deque>
#include <numeric>
#include <cmath>
#include <tuple>
#include <iostream>
// ---------- helpers ----------
static inline double safe_div(double a, double b){ return (std::abs(b)<1e-12)?0.0:(a/b); }
static inline double alpha_from_half_life(double hl){ return 1.0 - std::exp(std::log(0.5)/std::max(1.0,hl)); }
static Eigen::VectorXd cs_zscore(const Eigen::VectorXd& v, double clip){
    Eigen::VectorXd out=v;
    std::vector<double> a; a.reserve(v.size());
    for(int i=0;i<v.size();++i) if(std::isfinite(v(i))) a.push_back(v(i));
    if(a.size()<3){ out.setZero(); return out; }
    double mean = std::accumulate(a.begin(),a.end(),0.0)/a.size();
    double var=0; for(double x:a){ var+=(x-mean)*(x-mean); } var/=std::max(1.0,(double)a.size()-1.0);
    double sd = std::sqrt(std::max(1e-12,var));
    for(int i=0;i<v.size();++i){
        double z = std::isfinite(v(i)) ? (v(i)-mean)/sd : 0.0;
        if (z> clip) z= clip;
        if (z<-clip) z=-clip;
        out(i)=z;
    }
    return out;
}
static Eigen::VectorXd cs_quantile_map(const Eigen::VectorXd& v, double clip, double p=0.01){
    // 依分位數線性映射到 [-clip, clip]
    std::vector<std::pair<double,int>> a;
    a.reserve(v.size());
    for(int i=0;i<v.size();++i) a.push_back({std::isfinite(v(i))?v(i):0.0, i});
    std::sort(a.begin(), a.end(), [](auto&x,auto&y){return x.first<y.first;});
    Eigen::VectorXd out(v.size());
    for(int r=0;r<(int)a.size();++r){
        double q = safe_div(r, std::max(1,(int)a.size()-1));
        double x = -clip + (2*clip)*q;
        out(a[r].second) = x;
    }
    return out;
}

// ================= 2.6 Processing =================
Eigen::MatrixXd process::xsection_standardize(const Eigen::MatrixXd& X, const ProcessingConfig& cfg){
    Eigen::MatrixXd Y(X.rows(), X.cols());
    for(int r=0;r<X.rows();++r){
        Eigen::VectorXd row = X.row(r).transpose();
        if (cfg.mode==XSectionStandardize::ZScoreClip) Y.row(r) = cs_zscore(row, cfg.clip).transpose();
        else Y.row(r) = cs_quantile_map(row, cfg.clip, cfg.winsor_p).transpose();
        if (cfg.fill_missing_with_zero){
            for(int c=0;c<Y.cols();++c) if(!std::isfinite(Y(r,c))) Y(r,c)=0.0;
        }
    }
    return Y;
}

// ===== 取第 t 天的向量 =====
static Eigen::VectorXd row_at(const Eigen::MatrixXd& M, int t){
    if (t < 0 || t >= M.rows()) {
        std::cerr << "[row_at] out of range: t=" << t 
                  << " rows=" << M.rows() << std::endl;
        return Eigen::VectorXd::Zero(M.cols());
    }
    return M.row(t).transpose().eval();
}

static Eigen::VectorXd rolling_mean_last(const Eigen::MatrixXd& M, int t, int win){
    int T = M.rows(); win = std::min(win, t+1);
    Eigen::VectorXd m = Eigen::VectorXd::Zero(M.cols());
    for(int c=0;c<M.cols();++c){
        double s=0; int n=0;
        for(int r=t-win+1; r<=t; ++r){ double x=M(r,c); if(std::isfinite(x)){ s+=x; ++n; } }
        m(c) = (n>0)? s/n : 0.0;
    }
    return m;
}
static Eigen::VectorXd rolling_std_last(const Eigen::MatrixXd& M, int t, int win){
    int T=M.rows(); win=std::min(win, t+1);
    Eigen::VectorXd m = rolling_mean_last(M,t,win);
    Eigen::VectorXd s = Eigen::VectorXd::Zero(M.cols());
    for(int c=0;c<M.cols();++c){
        double var=0; int n=0;
        for(int r=t-win+1;r<=t;++r){
            double x=M(r,c); if(!std::isfinite(x)) continue;
            var += (x-m(c))*(x-m(c)); ++n;
        }
        s(c) = (n>1)? std::sqrt(var/(n-1)) : 0.0;
    }
    return s;
}

// ================= 2.3 Liquidity-Scaled Risk =================
process::RiskOutput process::build_liquidity_scaled_risk(
    const DataLoader& dl, int t, const RiskConfig& cfg)
{
    const int N = (int)dl.symbols().size();
    RiskOutput out;
    out.Sigma_tilde = Eigen::MatrixXd::Zero(N,N);
    out.L           = Eigen::MatrixXd::Zero(N,N);
    out.sigma_i     = Eigen::VectorXd::Zero(N);
    out.Dliq_i      = Eigen::VectorXd::Ones(N);

    // 基礎相關矩陣 Σ̂_corr：取報酬 R = ln(C_t/C_{t-1})
    Eigen::MatrixXd R(dl.C().rows(), dl.C().cols()); R.setZero();
    for(int r=1;r<dl.C().rows();++r)
        for(int c=0;c<dl.C().cols();++c){
            double c0=dl.C()(r-1,c), c1=dl.C()(r,c);
            R(r,c) = (std::isfinite(c0)&&std::isfinite(c1)&&c0>0&&c1>0)? std::log(c1/c0) : 0.0;
        }
    // EWMA 協方差 -> 轉相關
    double lam = std::exp(std::log(0.5)/cfg.corr_half_life);
    Eigen::VectorXd w(R.rows()); w(R.rows()-1)=1.0;
    for(int i=R.rows()-2;i>=0;--i) w(i)=w(i+1)*lam;
    double wsum = w.sum();

    Eigen::RowVectorXd mu = Eigen::RowVectorXd::Zero(N);
    for(int r=0;r<R.rows();++r) mu += w(r)*R.row(r);
    mu/=wsum;

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N,N);
    for(int r=0;r<R.rows();++r){
        Eigen::RowVectorXd x = R.row(r)-mu;
        S += w(r) * (x.transpose()*x);
    }
    S/=wsum;

    Eigen::VectorXd sd = S.diagonal().array().sqrt();
    Eigen::MatrixXd Corr = Eigen::MatrixXd::Zero(N,N);
    for(int i=0;i<N;++i) for(int j=0;j<N;++j)
        Corr(i,j) = (sd(i)>0 && sd(j)>0)? S(i,j)/(sd(i)*sd(j)) : (i==j?1.0:0.0);

	// σ_i：GK 或 30D std
	if (cfg.sigma_source == process::SigmaSource::GK) {
	    out.sigma_i = row_at(dl.feat_GKVol(), t);
	} else {
	    out.sigma_i = rolling_std_last(R, t, cfg.std_window);
	}



    // D_liq,i：以 TurnoverShare/VAL/ADV 調整
    Eigen::VectorXd ts = row_at(dl.feat_TurnoverShare(), t);
    double ts_bar = 0.0; // 橫斷面均值
    for(int i=0;i<N;++i) ts_bar += ts(i); ts_bar = ts_bar / std::max(1,N);

    Eigen::VectorXd VALt  = row_at(dl.VAL(), t);
    Eigen::VectorXd ADV   = rolling_mean_last(dl.VAL(), t, 30); // 30日均值
    for(int i=0;i<N;++i){
        double liq_term = 1.0 + cfg.c_liq * safe_div(1.0, std::sqrt( safe_div(VALt(i), std::max(1e-12,ADV(i))) ));
        double conc     = safe_div(ts_bar, ts(i)+cfg.eps); // 市場換手集中度（越小越鬆）
        out.Dliq_i(i)   = liq_term * conc;
        if (!std::isfinite(out.Dliq_i(i))) out.Dliq_i(i)=1.0;
    }

    // Σ̃ = D Σ̂_corr D ，其中 D=diag(Dliq_i * σ_i)
    Eigen::VectorXd diagD = out.Dliq_i.array() * out.sigma_i.array();
    Eigen::MatrixXd D = diagD.asDiagonal();
    out.Sigma_tilde = D * Corr * D;

    // Cholesky
    Eigen::LLT<Eigen::MatrixXd> llt(out.Sigma_tilde);
    if (llt.info()==Eigen::Success) out.L = llt.matrixL();
    else {
        // 若失敗，加一點 jitter
        out.Sigma_tilde += 1e-8 * Eigen::MatrixXd::Identity(N,N);
        out.L = out.Sigma_tilde.llt().matrixL();
    }
    return out;
}

// ================= 2.4 Cost Model =================
process::CostBreakdown process::estimate_costs(
    const DataLoader& dl, int t, const Eigen::VectorXd& A,
    const Eigen::VectorXi& side, const CostConfig& cfg)
{
    const int N = (int)A.size();
    CostBreakdown cb;
    cb.fee_buy  = Eigen::VectorXd::Zero(N);
    cb.fee_sell = Eigen::VectorXd::Zero(N);
    cb.tax_sell = Eigen::VectorXd::Zero(N);
    cb.impact   = Eigen::VectorXd::Zero(N);

    Eigen::VectorXd ADV = rolling_mean_last(dl.VAL(), t, cfg.adv_window);
    Eigen::VectorXd sigma = row_at(dl.feat_GKVol(), t); // 以 GK 當日波動做 proxy

    Eigen::VectorXd imb = row_at(dl.feat_Imbalance(), t);
    for(int i=0;i<N;++i){
        double Ai = std::max(0.0, A(i));
        if (side(i) > 0){ // buy
            cb.fee_buy(i) = std::max(cfg.min_fee, cfg.fee_rate * Ai);
        } else if (side(i) < 0){ // sell
            cb.fee_sell(i) = std::max(cfg.min_fee, cfg.fee_rate * Ai);
            cb.tax_sell(i) = cfg.tax_rate * Ai;
        }
        double liq = std::sqrt( safe_div(Ai, std::max(1e-12, ADV(i))) );
        double impact_rate = cfg.gamma_init * sigma(i) * (1.0 + cfg.beta_imb * std::abs(imb(i))) * liq;
        cb.impact(i) = impact_rate * Ai;
    }
    return cb;
}

// ================= 2.5 m-scaling =================
process::BlackSwanOutput process::black_swan_scale(
    const DataLoader& dl, int t, const BlackSwanConfig& cfg, bool external_event){
    BlackSwanOutput out; out.m = cfg.m_default;
    // 市場統計用橫斷面中位數
    auto median = [](const Eigen::VectorXd& v){
        std::vector<double> a; a.reserve(v.size());
        for(int i=0;i<v.size();++i) if(std::isfinite(v(i))) a.push_back(v(i));
        if (a.empty()) return 0.0;
        std::sort(a.begin(), a.end());
        size_t k=a.size()/2; return (a.size()%2)? a[k] : 0.5*(a[k-1]+a[k]);
    };
    double m_gap = median(row_at(dl.feat_Gap(), t));
    double gk_now = median(row_at(dl.feat_GKVol(), t));
    // 30d 中位 GK
    Eigen::MatrixXd GK = dl.feat_GKVol();
    int win=30; win = std::min(win, t+1);
    std::vector<double> acc;
    for(int r=t-win+1;r<=t;++r) acc.push_back(median(row_at(GK,r)));
    double gk_30 = std::accumulate(acc.begin(),acc.end(),0.0)/std::max(1,(int)acc.size());
    double ratio = (gk_30>0.0)? gk_now/gk_30 : 1.0;
    double imb_med = median(row_at(dl.feat_Imbalance(), t));

    if (m_gap <= cfg.gap_threshold) out.m = std::min(out.m, cfg.m_gap_bad);
    if (ratio  >= cfg.vol_ratio_th) out.m = std::min(out.m, cfg.m_vol_high);
    if (imb_med<= cfg.imb_median_th)out.m = std::min(out.m, cfg.m_imb_bad);
    if (external_event)             out.m = std::min(out.m, cfg.m_event);
    out.m = std::clamp(out.m, 0.5, 1.5);
    return out;
}

// ================= 2.6 Stage-1 Forecast =================

// 計算第 t-1→t 的真實報酬（對齊 t）
static Eigen::VectorXd realized_ret_t(const DataLoader& dl, int t){
    Eigen::VectorXd r = Eigen::VectorXd::Zero((int)dl.symbols().size());
    if (t==0) return r;
    for(int c=0;c<dl.C().cols();++c){
        double c0=dl.C()(t-1,c), c1=dl.C()(t,c);
        r(c) = (std::isfinite(c0)&&std::isfinite(c1)&&c0>0)? (c1/c0 - 1.0) : 0.0;
    }
    return r;
}

// 帶半衰期權重的最小平方法 β = (X'WX)^{-1} X'Wy
static Eigen::VectorXd wls_beta(const std::vector<Eigen::VectorXd>& Xhist,
                                const std::vector<double>& yhist,
                                double half_life)
{
    if (Xhist.empty()) return Eigen::VectorXd();
    int p = (int)Xhist[0].size();
    Eigen::MatrixXd XtWX = Eigen::MatrixXd::Zero(p,p);
    Eigen::VectorXd XtWy = Eigen::VectorXd::Zero(p);
    double alpha = alpha_from_half_life(half_life);
    double w=1.0;
    for (int k=(int)Xhist.size()-1; k>=0; --k){
        const auto& x = Xhist[k];
        double y = yhist[k];
        XtWX += w * (x * x.transpose());
        XtWy += w * (x * y);
        w *= (1.0 - alpha);
    }
    // 小小 ridge，避免奇異
    double lambda = 1e-6;
    Eigen::VectorXd beta = (XtWX + lambda * Eigen::MatrixXd::Identity(p,p)).ldlt().solve(XtWy);
    return beta;
}

// 加入這個版本（保留你原本的 row_at 也行）
static Eigen::VectorXd row_at_n(const Eigen::MatrixXd& M, int t, int N_expect){
    if (t < 0 || t >= M.rows() || M.cols() != N_expect) {
        return Eigen::VectorXd::Zero(N_expect);
    }
    return M.row(t).transpose().eval();
}

process::ForecastOutput process::stage1_forecast(
    const DataLoader& dl, int t,
    const ProcessingConfig& px_cfg,
    const ForecastConfig& fc,
    const std::optional<Eigen::MatrixXd>& next_day_returns)
{
    ForecastOutput out;
    const int N = (int)dl.symbols().size();
    out.rhat_raw = Eigen::VectorXd::Zero(N);

    // 邊界：至少要能取到 y(t) = ret(t-1->t)
    const int T = dl.C().rows();
    if (N == 0 || T < 2 || t <= 0 || t >= T) {
        return out;
    }
	//std::cout << "邊界"<< std::endl;
    // --- 取第 t 天的特徵，並做橫斷面標準化 ---
    auto stdize = [&](const Eigen::VectorXd& v){
	    Eigen::VectorXd out = Eigen::VectorXd::Zero(N);
	    if (v.size() != N) return out;            // 關鍵保護
	    Eigen::MatrixXd tmp(1, N);
	    tmp.row(0) = v.transpose();
	    auto z = process::xsection_standardize(tmp, px_cfg);
	    out = z.row(0).transpose();
	    for (int i=0;i<N;++i) if (!std::isfinite(out(i))) out(i)=0.0;
	    return out;
	};

    auto get_row = [&](const Eigen::MatrixXd& M, int day) -> Eigen::VectorXd {
    	return row_at_n(M, day, N);   // N 維保證
	};
	//std::cout << "get_row "<< std::endl;
    Eigen::VectorXd gap  = stdize(get_row(dl.feat_Gap(),            t));
	Eigen::VectorXd m5   = stdize(get_row(dl.feat_Mom5(),           t));
	Eigen::VectorXd m10  = stdize(get_row(dl.feat_Mom10(),          t));
	Eigen::VectorXd m20  = stdize(get_row(dl.feat_Mom20(),          t));  // ← 修正
	Eigen::VectorXd bias = stdize(get_row(dl.feat_BIAS(),           t));
	Eigen::VectorXd bf   = stdize(get_row(dl.feat_BrokerStrength(), t));
	Eigen::VectorXd lv   = stdize(get_row(dl.feat_Liquidity(),      t));
	Eigen::VectorXd ts   = stdize(get_row(dl.feat_TurnoverShare(),  t));
	Eigen::VectorXd im   = stdize(get_row(dl.feat_Imbalance(),      t));
	Eigen::VectorXd gk   = stdize(get_row(dl.feat_GKVol(),          t));

    // 任一向量尺寸不對就直接回傳 0
    auto okN = [&](const Eigen::VectorXd& v){ return v.size()==N; };
    if (!(okN(gap)&&okN(m5)&&okN(m10)&&okN(m20)&&okN(bias)&&okN(bf)&&okN(lv)&&okN(ts)&&okN(im)&&okN(gk))) {
        return out;
    }

    // 設計矩陣（N×P）
    constexpr int P = 10;
    Eigen::MatrixXd X(N, P);
    X.col(0)=gap;  X.col(1)=m5;  X.col(2)=m10; X.col(3)=m20;
    X.col(4)=bias; X.col(5)=bf;  X.col(6)=lv;  X.col(7)=ts;
    X.col(8)=im;   X.col(9)=gk;
	//std::cout << "0"<< std::endl;
    if (!fc.use_linear) {
        return out; // 原始 rhat_raw 維持 0 向量
    }
	//std::cout << "10"<< std::endl;
    // -------- 收集歷史樣本：tau ∈ [start..t-1]，y 對齊 (tau+1) --------
    const int start = std::max(1, t - fc.lookback);
    const int hist_len = std::max(0, (t - 1) - start + 1);
    if (hist_len <= 0) return out;

    std::vector<Eigen::VectorXd> xhist;
    std::vector<double>          yhist;
    xhist.reserve(static_cast<size_t>(hist_len) * static_cast<size_t>(N));
    yhist.reserve(static_cast<size_t>(hist_len) * static_cast<size_t>(N));
	//std::cout << "1"<< std::endl;
    auto build_row_std = [&](int day) {
        // 全部以 by value 取出，避免懸掛引用
        Eigen::VectorXd g   = stdize(get_row(dl.feat_Gap(),            day));
        Eigen::VectorXd m5_ = stdize(get_row(dl.feat_Mom5(),           day));
        Eigen::VectorXd m10_= stdize(get_row(dl.feat_Mom10(),          day));
        Eigen::VectorXd m20_= stdize(get_row(dl.feat_Mom20(),          day));
        Eigen::VectorXd b   = stdize(get_row(dl.feat_BIAS(),           day));
        Eigen::VectorXd bf_ = stdize(get_row(dl.feat_BrokerStrength(), day));
        Eigen::VectorXd lv_ = stdize(get_row(dl.feat_Liquidity(),      day));
        Eigen::VectorXd ts_ = stdize(get_row(dl.feat_TurnoverShare(),  day));
        Eigen::VectorXd im_ = stdize(get_row(dl.feat_Imbalance(),      day));
        Eigen::VectorXd gk_ = stdize(get_row(dl.feat_GKVol(),          day));
        return std::tuple{g,m5_,m10_,m20_,b,bf_,lv_,ts_,im_,gk_};
    };

    const int Ty = next_day_returns ? next_day_returns->rows() : T;
    const int Ny = next_day_returns ? next_day_returns->cols() : N;
	//std::cout << "2"<< std::endl;
    for (int tau = start; tau <= t - 1; ++tau) {
        // y(tau) 需用到 (tau+1)
		//std::cout << "3"<< std::endl;
        if (tau + 1 >= T) break;
        if (next_day_returns && (tau >= Ty || Ny != N)) break;

        auto [g_,m5_,m10_,m20_,b_,bf_,lv_,ts_,im_,gk_] = build_row_std(tau);
        if (!(okN(g_)&&okN(m5_)&&okN(m10_)&&okN(m20_)&&okN(b_)&&okN(bf_)&&okN(lv_)&&okN(ts_)&&okN(im_)&&okN(gk_))) {
            continue;
        }
		//std::cout << "4"<< std::endl;
        Eigen::VectorXd y_vec = next_day_returns
            ? next_day_returns->row(tau).transpose()
            : realized_ret_t(dl, tau + 1);

        if (y_vec.size() != N) continue;

        for (int c = 0; c < N; ++c) {
			//std::cout << "5"<< std::endl;
            Eigen::VectorXd x(P);
            x << g_(c), m5_(c), m10_(c), m20_(c), b_(c), bf_(c), lv_(c), ts_(c), im_(c), gk_(c);
            xhist.push_back(std::move(x));
            yhist.push_back(std::isfinite(y_vec(c)) ? y_vec(c) : 0.0);
        }
    }
	//std::cout << "6"<< std::endl;
    if (xhist.empty()) {
        return out; // 樣本不足，回 0
    }

    // -------- 帶半衰期權重的 ridge WLS --------
    auto wls_beta = [&](const std::vector<Eigen::VectorXd>& Xs,
                        const std::vector<double>& ys,
                        double half_life)->Eigen::VectorXd {
        const int p = (int)Xs[0].size();
        Eigen::MatrixXd XtWX = Eigen::MatrixXd::Zero(p,p);
        Eigen::VectorXd XtWy = Eigen::VectorXd::Zero(p);
        const double alpha = alpha_from_half_life(half_life);
        double w = 1.0;
        for (int k = (int)Xs.size()-1; k >= 0; --k) {
            const auto& x = Xs[k];
            const double y = ys[k];
            XtWX.noalias() += w * (x * x.transpose());
            XtWy.noalias() += w * (x * y);
            w *= (1.0 - alpha);
        }
        const double lambda = 1e-6;
        return (XtWX + lambda * Eigen::MatrixXd::Identity(p,p)).ldlt().solve(XtWy);
    };

    Eigen::VectorXd beta = wls_beta(xhist, yhist, fc.half_life);
    if (beta.size() == P) {
        out.rhat_raw = (X * beta).eval();
    }
    return out;
}



