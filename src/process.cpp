// process.cpp (relaxed for convergence/stability)
#include "process.h"
#include <algorithm>
#include <deque>
#include <numeric>
#include <cmath>
#include <tuple>
#include <iostream>
#include <fstream>

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
        s(c) = (n>1)? std::sqrt(std::max(1e-12, var/(n-1))) : 0.0;
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
    double wsum = std::max(1e-12, w.sum());

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
    for (int i=0;i<N;++i) sd(i) = std::max(sd(i), 1e-6); // floor 防爆

    Eigen::MatrixXd Corr = Eigen::MatrixXd::Zero(N,N);
    for(int i=0;i<N;++i) for(int j=0;j<N;++j)
        Corr(i,j) = (i==j) ? 1.0 : (S(i,j)/(sd(i)*sd(j)));

    // shrinkage：避免病態（收斂更穩）
    const double eta = 0.10; // 10% towards I
    Corr = (1.0-eta)*Corr + eta*Eigen::MatrixXd::Identity(N,N);

    // σ_i：GK 或 30D std
    if (cfg.sigma_source == process::SigmaSource::GK) {
        out.sigma_i = row_at(dl.feat_GKVol(), t);
    } else {
        out.sigma_i = rolling_std_last(R, t, cfg.std_window);
    }
    for (int i=0;i<N;++i) out.sigma_i(i) = std::clamp(out.sigma_i(i), 1e-6, 10.0);

    // D_liq,i：以 TurnoverShare/VAL/ADV 調整
    Eigen::VectorXd ts = row_at(dl.feat_TurnoverShare(), t);
    double ts_bar = 0.0; for(int i=0;i<N;++i) ts_bar += (std::isfinite(ts(i))?ts(i):0.0); ts_bar = ts_bar / std::max(1,N);

    Eigen::VectorXd VALt  = row_at(dl.VAL(), t);
    Eigen::VectorXd ADV   = rolling_mean_last(dl.VAL(), t, 30); // 30日均值
    for(int i=0;i<N;++i){
        double adv_i = std::max(ADV(i), 1e-8);
        double val_i = std::max(VALt(i), 1e-8);
        double liq = std::sqrt(safe_div(std::max(0.0,val_i), adv_i));
        // 放寬：限制在合理區間，避免極端值
        double liq_term = 1.0 + cfg.c_liq * safe_div(1.0, std::max(1e-6, liq));
        liq_term = std::clamp(liq_term, 0.5, 3.0);
        double conc = safe_div(ts_bar, std::max(1e-6, ts(i)+cfg.eps));
        conc = std::clamp(conc, 0.5, 2.0);
        out.Dliq_i(i) = std::clamp(liq_term * conc, 0.5, 5.0);
        if (!std::isfinite(out.Dliq_i(i))) out.Dliq_i(i)=1.0;
    }

    // Σ̃ = D Σ̂_corr D ，其中 D=diag(Dliq_i * σ_i)
    Eigen::VectorXd diagD = out.Dliq_i.array() * out.sigma_i.array();
    Eigen::MatrixXd D = diagD.asDiagonal();
    out.Sigma_tilde = D * Corr * D;

    // 更強的 diagonal loading（放寬/穩定）
    const double load = 1e-4;
    for(int i=0;i<N;++i) out.Sigma_tilde(i,i) += load;

    // Cholesky 或最近 PSD 修正
    Eigen::LLT<Eigen::MatrixXd> llt(out.Sigma_tilde);
    if (llt.info()==Eigen::Success) out.L = llt.matrixL();
    else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(out.Sigma_tilde);
        Eigen::VectorXd ev = es.eigenvalues();
        Eigen::MatrixXd U  = es.eigenvectors();
        for (int i=0;i<ev.size();++i) ev(i) = std::max(ev(i), 1e-10);
        out.Sigma_tilde = U * ev.asDiagonal() * U.transpose();
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
    Eigen::VectorXd sigma = row_at(dl.feat_GKVol(), t); // GK 當日波動
    Eigen::VectorXd imb = row_at(dl.feat_Imbalance(), t);

    for(int i=0;i<N;++i){
        double Ai = std::max(0.0, A(i));
        if (side(i) > 0){ // buy
            cb.fee_buy(i) = std::max(cfg.min_fee, cfg.fee_rate * Ai);
        } else if (side(i) < 0){ // sell
            cb.fee_sell(i) = std::max(cfg.min_fee, cfg.fee_rate * Ai);
            cb.tax_sell(i) = cfg.tax_rate * Ai;
        }
        double adv_i = std::max(ADV(i), 1e-8);
        double liq = std::sqrt( safe_div(Ai, adv_i) );
        double sig = std::clamp(sigma(i), 0.0, 5.0);
        double imb_fac = 1.0 + cfg.beta_imb * std::min(1.0, std::abs(imb(i)));
        // 放寬：降低初始衝擊係數，並加上上限
        double impact_rate = (0.5 * cfg.gamma_init) * sig * imb_fac * std::clamp(liq, 0.0, 10.0);
        cb.impact(i) = std::min(impact_rate * Ai, 1e6); // 上限避免爆炸
    }
    return cb;
}

// ================= 2.5 m-scaling =================
process::BlackSwanOutput process::black_swan_scale(
    const DataLoader& dl, int t, const BlackSwanConfig& cfg, bool external_event){
    BlackSwanOutput out; out.m = cfg.m_default;
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

    auto soften = [](double cur, double target){
        // 朝 target 靠近一半，避免瞬間重手
        return 0.5*cur + 0.5*target;
    };

    if (m_gap <= cfg.gap_threshold) out.m = std::min(out.m, soften(out.m, cfg.m_gap_bad));
    if (ratio  >= cfg.vol_ratio_th) out.m = std::min(out.m, soften(out.m, cfg.m_vol_high));
    if (imb_med<= cfg.imb_median_th)out.m = std::min(out.m, soften(out.m, cfg.m_imb_bad));
    if (external_event)             out.m = std::min(out.m, soften(out.m, cfg.m_event));
    out.m = std::clamp(out.m, 0.7, 1.3); // 放寬：縮小 m 變動幅度
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
        XtWX.noalias() += w * (x * x.transpose());
        XtWy.noalias() += w * (x * y);
        w *= (1.0 - alpha);
    }
    // 放寬/穩定：較大的 ridge，避免奇異與過擬合
    double lambda = 1e-3;
    Eigen::VectorXd beta = (XtWX + lambda * Eigen::MatrixXd::Identity(p,p)).ldlt().solve(XtWy);
    return beta;
}

// 尺寸保護版 row 取值
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

    auto stdize = [&](const Eigen::VectorXd& v){
        Eigen::VectorXd zN = Eigen::VectorXd::Zero(N);
        if (v.size() != N) return zN;
        Eigen::MatrixXd tmp(1, N);
        tmp.row(0) = v.transpose();
        auto z = process::xsection_standardize(tmp, px_cfg);
        zN = z.row(0).transpose();
        for (int i=0;i<N;++i) if (!std::isfinite(zN(i))) zN(i)=0.0;
        return zN;
    };

    auto get_row = [&](const Eigen::MatrixXd& M, int day) -> Eigen::VectorXd {
        return row_at_n(M, day, N);
    };

    Eigen::VectorXd gap  = stdize(get_row(dl.feat_Gap(),            t));
    Eigen::VectorXd m5   = stdize(get_row(dl.feat_Mom5(),           t));
    Eigen::VectorXd m10  = stdize(get_row(dl.feat_Mom10(),          t));
    Eigen::VectorXd m20  = stdize(get_row(dl.feat_Mom20(),          t));
    Eigen::VectorXd bias = stdize(get_row(dl.feat_BIAS(),           t));
    Eigen::VectorXd bf   = stdize(get_row(dl.feat_BrokerStrength(), t));
    Eigen::VectorXd lv   = stdize(get_row(dl.feat_Liquidity(),      t));
    Eigen::VectorXd ts   = stdize(get_row(dl.feat_TurnoverShare(),  t));
    Eigen::VectorXd im   = stdize(get_row(dl.feat_Imbalance(),      t));
    Eigen::VectorXd gk   = stdize(get_row(dl.feat_GKVol(),          t));

    auto okN = [&](const Eigen::VectorXd& v){ return v.size()==N; };
    if (!(okN(gap)&&okN(m5)&&okN(m10)&&okN(m20)&&okN(bias)&&okN(bf)&&okN(lv)&&okN(ts)&&okN(im)&&okN(gk))) {
        return out;
    }

    constexpr int P = 10;
    Eigen::MatrixXd X(N, P);
    X.col(0)=gap;  X.col(1)=m5;  X.col(2)=m10; X.col(3)=m20;
    X.col(4)=bias; X.col(5)=bf;  X.col(6)=lv;  X.col(7)=ts;
    X.col(8)=im;   X.col(9)=gk;

    if (!fc.use_linear) {
        return out;
    }

    const int start = std::max(1, t - fc.lookback);
    const int hist_len = std::max(0, (t - 1) - start + 1);
    if (hist_len <= 0) return out;

    std::vector<Eigen::VectorXd> xhist;
    std::vector<double>          yhist;
    xhist.reserve(static_cast<size_t>(hist_len) * static_cast<size_t>(N));
    yhist.reserve(static_cast<size_t>(hist_len) * static_cast<size_t>(N));

    auto build_row_std = [&](int day) {
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

    for (int tau = start; tau <= t - 1; ++tau) {
        if (tau + 1 >= T) break;
        if (next_day_returns && (tau >= Ty || Ny != N)) break;

        auto [g_,m5_,m10_,m20_,b_,bf_,lv_,ts_,im_,gk_] = build_row_std(tau);
        if (!(okN(g_)&&okN(m5_)&&okN(m10_)&&okN(m20_)&&okN(b_)&&okN(bf_)&&okN(lv_)&&okN(ts_)&&okN(im_)&&okN(gk_))) {
            continue;
        }

        Eigen::VectorXd y_vec = next_day_returns
            ? next_day_returns->row(tau).transpose()
            : realized_ret_t(dl, tau + 1);

        if (y_vec.size() != N) continue;

        for (int c = 0; c < N; ++c) {
            Eigen::VectorXd x(P);
            x << g_(c), m5_(c), m10_(c), m20_(c), b_(c), bf_(c), lv_(c), ts_(c), im_(c), gk_(c);
            xhist.push_back(std::move(x));
            yhist.push_back(std::isfinite(y_vec(c)) ? y_vec(c) : 0.0);
        }
    }

    if (xhist.empty()) return out;

    auto wls_beta_local = [&](const std::vector<Eigen::VectorXd>& Xs,
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
        const double lambda = 1e-3; // 放寬/穩定
        return (XtWX + lambda * Eigen::MatrixXd::Identity(p,p)).ldlt().solve(XtWy);
    };

    Eigen::VectorXd beta = wls_beta_local(xhist, yhist, fc.half_life);
    if (beta.size() == P) out.rhat_raw = (X * beta).eval();
    return out;
}

// ======================= Excel / CSV Export Impl =======================

#ifdef USE_XLSX
extern "C" {
#include <xlsxwriter.h>
}
#endif

namespace {
// ---- 小工具：避免 NaN / inf 寫入 ----
inline double fin(double x) { return std::isfinite(x) ? x : 0.0; }

#ifdef USE_XLSX
// ---- xlsx 寫向量 ----
static void xlsx_write_vector(lxw_worksheet* ws, int row0, int col0,
                              const Eigen::VectorXd& v,
                              const char* title,
                              const char* colName = "value") {
    int r = row0;
    if (title && *title) worksheet_write_string(ws, r++, col0, title, nullptr);
    worksheet_write_string(ws, r, col0, "index", nullptr);
    worksheet_write_string(ws, r, col0 + 1, colName, nullptr);
    ++r;
    for (int i = 0; i < v.size(); ++i) {
        worksheet_write_number(ws, r + i, col0, i, nullptr);
        worksheet_write_number(ws, r + i, col0 + 1, fin(v(i)), nullptr);
    }
}

static void xlsx_write_int_vector(lxw_worksheet* ws, int row0, int col0,
                                  const Eigen::VectorXi& v,
                                  const char* title,
                                  const char* colName = "value") {
    int r = row0;
    if (title && *title) worksheet_write_string(ws, r++, col0, title, nullptr);
    worksheet_write_string(ws, r, col0, "index", nullptr);
    worksheet_write_string(ws, r, col0 + 1, colName, nullptr);
    ++r;
    for (int i = 0; i < v.size(); ++i) {
        worksheet_write_number(ws, r + i, col0, i, nullptr);
        worksheet_write_number(ws, r + i, col0 + 1, v(i), nullptr);
    }
}

// ---- xlsx 寫矩陣 ----
static void xlsx_write_matrix(lxw_worksheet* ws, int row0, int col0,
                              const Eigen::MatrixXd& M,
                              const char* title) {
    int r = row0;
    int m = (int)M.rows(), n = (int)M.cols();
    if (title && *title) worksheet_write_string(ws, r++, col0, title, nullptr);

    worksheet_write_string(ws, r, col0, "row", nullptr);
    for (int j = 0; j < n; ++j) {
        std::string h = "C" + std::to_string(j);
        worksheet_write_string(ws, r, col0 + 1 + j, h.c_str(), nullptr);
    }
    ++r;

    for (int i = 0; i < m; ++i) {
        std::string rn = "R" + std::to_string(i);
        worksheet_write_string(ws, r + i, col0, rn.c_str(), nullptr);
        for (int j = 0; j < n; ++j) {
            worksheet_write_number(ws, r + i, col0 + 1 + j, fin(M(i, j)), nullptr);
        }
    }
}
#endif // USE_XLSX

// ---- CSV 版：向量 / 矩陣 ----
static void csv_write_vector(const std::string& path, const Eigen::VectorXd& v,
                             const char* colName = "value") {
    std::ofstream f(path);
    f << "index," << colName << "\n";
    for (int i = 0; i < v.size(); ++i) f << i << "," << fin(v(i)) << "\n";
}

static void csv_write_int_vector(const std::string& path, const Eigen::VectorXi& v,
                                 const char* colName = "value") {
    std::ofstream f(path);
    f << "index," << colName << "\n";
    for (int i = 0; i < v.size(); ++i) f << i << "," << v(i) << "\n";
}

static void csv_write_matrix(const std::string& path, const Eigen::MatrixXd& M) {
    std::ofstream f(path);
    int m = (int)M.rows(), n = (int)M.cols();
    f << "row";
    for (int j = 0; j < n; ++j) f << ",C" << j;
    f << "\n";
    for (int i = 0; i < m; ++i) {
        f << "R" << i;
        for (int j = 0; j < n; ++j) f << "," << fin(M(i, j));
        f << "\n";
    }
}
} // anonymous namespace

// ======================= dump_to_excel (xlsx) =======================
void process::dump_to_excel(const std::string& xlsx_path,
                            const DataLoader& dl, int t,
                            const ProcessingConfig& px_cfg,
                            const RiskConfig& risk_cfg,
                            const CostConfig& cost_cfg,
                            const BlackSwanConfig& bs_cfg,
                            const ForecastConfig& fc_cfg,
                            const Eigen::VectorXd& A,
                            const Eigen::VectorXi& side,
                            const std::optional<Eigen::MatrixXd>& next_day_returns)
{
#ifndef USE_XLSX
    std::cerr << "[dump_to_excel] USE_XLSX 未啟用或未連結 libxlsxwriter，請改用 dump_to_excel_csv_fallback().\n";
    return;
#else
    // 1) 計算全部元件
    RiskOutput      rsk = build_liquidity_scaled_risk(dl, t, risk_cfg);
    CostBreakdown   cst = estimate_costs(dl, t, A, side, cost_cfg);
    BlackSwanOutput bs  = black_swan_scale(dl, t, bs_cfg, /*external_event=*/false);
    ForecastOutput  fo  = stage1_forecast(dl, t, px_cfg, fc_cfg, next_day_returns);

    // 2) 建立 xlsx
    lxw_workbook* wb = workbook_new(xlsx_path.c_str());
    if (!wb) {
        std::cerr << "[dump_to_excel] 無法建立 " << xlsx_path << "\n";
        return;
    }

    // ---- Sheet: risk ----
    {
        lxw_worksheet* ws = workbook_add_worksheet(wb, "risk");
        if (ws) {
            int base = 0;
            xlsx_write_matrix(ws, base, 0, rsk.Sigma_tilde, "Sigma_tilde"); base += (int)rsk.Sigma_tilde.rows() + 3;
            xlsx_write_matrix(ws, base, 0, rsk.L,           "Cholesky_L");  base += (int)rsk.L.rows()           + 3;
            xlsx_write_vector(ws, base, 0, rsk.sigma_i,     "sigma_i");     base += (int)rsk.sigma_i.size()     + 3;
            xlsx_write_vector(ws, base, 0, rsk.Dliq_i,      "Dliq_i");
        }
    }

    // ---- Sheet: corr_hint ----
    {
        lxw_worksheet* ws = workbook_add_worksheet(wb, "corr_hint");
        if (ws) {
            Eigen::VectorXd D = (rsk.Dliq_i.array() * rsk.sigma_i.array()).matrix();
            Eigen::VectorXd invD = D.unaryExpr([](double x){ return (std::abs(x) < 1e-12) ? 0.0 : 1.0 / x; });
            Eigen::MatrixXd Corr = invD.asDiagonal() * rsk.Sigma_tilde * invD.asDiagonal();
            xlsx_write_matrix(ws, 0, 0, Corr, "Corr (approx)");
        }
    }

    // ---- Sheet: cost ----
    {
        lxw_worksheet* ws = workbook_add_worksheet(wb, "cost");
        if (ws) {
            int base = 0;
            xlsx_write_vector(ws, base, 0, cst.fee_buy,  "fee_buy");  base += (int)cst.fee_buy.size()  + 3;
            xlsx_write_vector(ws, base, 0, cst.fee_sell, "fee_sell"); base += (int)cst.fee_sell.size() + 3;
            xlsx_write_vector(ws, base, 0, cst.tax_sell, "tax_sell"); base += (int)cst.tax_sell.size() + 3;
            xlsx_write_vector(ws, base, 0, cst.impact,   "impact");
        }
    }

    // ---- Sheet: black_swan ----
    {
        lxw_worksheet* ws = workbook_add_worksheet(wb, "black_swan");
        if (ws) {
            worksheet_write_string(ws, 0, 0, "m", nullptr);
            worksheet_write_number(ws, 0, 1, bs.m, nullptr);
        }
    }

    // ---- Sheet: stage1 ----
    {
        lxw_worksheet* ws = workbook_add_worksheet(wb, "stage1");
        if (ws) xlsx_write_vector(ws, 0, 0, fo.rhat_raw, "rhat_raw");
    }

    // ---- Sheet: inputs ----
    {
        lxw_worksheet* ws = workbook_add_worksheet(wb, "inputs");
        if (ws) {
            int base = 0;
            xlsx_write_vector(ws, base, 0, A, "A (notional)");           base += (int)A.size() + 3;
            xlsx_write_int_vector(ws, base, 0, side, "side (+1/-1/0)");   base += (int)side.size() + 3;

            auto write_row = [&](const char* title, const Eigen::MatrixXd& M){
                if (t >= 0 && t < M.rows()) {
                    Eigen::VectorXd v = M.row(t).transpose();
                    xlsx_write_vector(ws, base, 0, v, title);
                    base += (int)v.size() + 3;
                }
            };
            write_row("VAL(t)",             dl.VAL());
            write_row("GKVol(t)",           dl.feat_GKVol());
            write_row("TurnoverShare(t)",   dl.feat_TurnoverShare());
            write_row("Imbalance(t)",       dl.feat_Imbalance());
        }
    }

    workbook_close(wb);
#endif // USE_XLSX
}

// ======================= dump_to_excel_csv_fallback =======================
void process::dump_to_excel_csv_fallback(const std::string& out_dir,
                                         const DataLoader& dl, int t,
                                         const ProcessingConfig& px_cfg,
                                         const RiskConfig& risk_cfg,
                                         const CostConfig& cost_cfg,
                                         const BlackSwanConfig& bs_cfg,
                                         const ForecastConfig& fc_cfg,
                                         const Eigen::VectorXd& A,
                                         const Eigen::VectorXi& side,
                                         const std::optional<Eigen::MatrixXd>& next_day_returns)
{
    RiskOutput      rsk = build_liquidity_scaled_risk(dl, t, risk_cfg);
    CostBreakdown   cst = estimate_costs(dl, t, A, side, cost_cfg);
    BlackSwanOutput bs  = black_swan_scale(dl, t, bs_cfg, /*external_event=*/false);
    ForecastOutput  fo  = stage1_forecast(dl, t, px_cfg, fc_cfg, next_day_returns);

    // 風險
    {
        auto csv_write_matrix = [](const std::string& path, const Eigen::MatrixXd& M){
            std::ofstream f(path);
            int m = (int)M.rows(), n = (int)M.cols();
            f << "row"; for (int j=0;j<n;++j) f << ",C" << j; f << "\n";
            for (int i=0;i<m;++i){ f << "R" << i; for(int j=0;j<n;++j) f << "," << (std::isfinite(M(i,j))?M(i,j):0.0); f << "\n"; }
        };
        auto csv_write_vector_loc = [](const std::string& path, const Eigen::VectorXd& v, const char* col){
            std::ofstream f(path); f << "index," << col << "\n";
            for(int i=0;i<v.size();++i) f << i << "," << (std::isfinite(v(i))?v(i):0.0) << "\n";
        };
        csv_write_matrix(out_dir + "/risk_Sigma_tilde.csv", rsk.Sigma_tilde);
        csv_write_matrix(out_dir + "/risk_Cholesky_L.csv",  rsk.L);
        csv_write_vector_loc(out_dir + "/risk_sigma_i.csv",     rsk.sigma_i, "sigma_i");
        csv_write_vector_loc(out_dir + "/risk_Dliq_i.csv",      rsk.Dliq_i,  "Dliq_i");
    }

    // 成本
    auto csv_write_vector2 = [](const std::string& path, const Eigen::VectorXd& v, const char* col){
        std::ofstream f(path); f << "index," << col << "\n";
        for(int i=0;i<v.size();++i) f << i << "," << (std::isfinite(v(i))?v(i):0.0) << "\n";
    };
    csv_write_vector2(out_dir + "/cost_fee_buy.csv",   cst.fee_buy,  "fee_buy");
    csv_write_vector2(out_dir + "/cost_fee_sell.csv",  cst.fee_sell, "fee_sell");
    csv_write_vector2(out_dir + "/cost_tax_sell.csv",  cst.tax_sell, "tax_sell");
    csv_write_vector2(out_dir + "/cost_impact.csv",    cst.impact,   "impact");

    // 黑天鵝 m
    { std::ofstream f(out_dir + "/black_swan.csv"); f << "m\n" << bs.m << "\n"; }

    // Stage-1
    csv_write_vector2(out_dir + "/stage1_rhat_raw.csv", fo.rhat_raw, "rhat_raw");

    // Inputs
    csv_write_vector2(out_dir + "/inputs_A.csv",    A,    "A");
    {
        std::ofstream f(out_dir + "/inputs_side.csv"); f << "index,side\n";
        for(int i=0;i<side.size();++i) f << i << "," << side(i) << "\n";
    }

    // t 當日特徵快照
    auto dump_row = [&](const std::string& path, const Eigen::MatrixXd& M, const char* col){
        if (t >= 0 && t < M.rows()){
            std::ofstream f(path); f << "index," << col << "\n";
            Eigen::VectorXd v = M.row(t).transpose();
            for(int i=0;i<v.size();++i) f << i << "," << (std::isfinite(v(i))?v(i):0.0) << "\n";
        }
    };
    dump_row(out_dir + "/VAL_t.csv",           dl.VAL(),               "VAL");
    dump_row(out_dir + "/GKVol_t.csv",         dl.feat_GKVol(),        "GKVol");
    dump_row(out_dir + "/TurnoverShare_t.csv", dl.feat_TurnoverShare(),"TurnoverShare");
    dump_row(out_dir + "/Imbalance_t.csv",     dl.feat_Imbalance(),    "Imbalance");
}

// === 實作（放在 process.cpp 末尾；和既有 xlsx/CSV 工具共用） ===
#ifdef USE_XLSX
extern "C" { #include <xlsxwriter.h> }
#endif

void process::dump_trade_plan_excel(const std::string& out_path,
                                    const DataLoader& dl, int t,
                                    const Eigen::VectorXd& A_socp,
                                    const Eigen::VectorXi& side_socp,
                                    double P0)
{
    const int N = (int)A_socp.size();
    // 取當日收盤/價格；若超界回傳 0
    auto price_row = [&](int day)->Eigen::VectorXd{
        if (day < 0 || day >= dl.C().rows()) return Eigen::VectorXd::Zero(N);
        Eigen::VectorXd v = dl.C().row(day).transpose();
        for (int i=0;i<N;++i) if (!std::isfinite(v(i)) || v(i) <= 0) v(i)=0.0;
        return v;
    };
    Eigen::VectorXd price = price_row(t);

#ifdef USE_XLSX
    lxw_workbook* wb = workbook_new(out_path.c_str());
    if (!wb) { std::cerr << "[trade_plan] cannot create " << out_path << "\n"; return; }
    lxw_worksheet* ws = workbook_add_worksheet(wb, "trade_plan");

    // 標題列
    int r = 0;
    worksheet_write_string(ws, r, 0, "index",      nullptr);
    worksheet_write_string(ws, r, 1, "w_est",      nullptr);
    worksheet_write_string(ws, r, 2, "abs_w",      nullptr);
    worksheet_write_string(ws, r, 3, "A_notional", nullptr);
    worksheet_write_string(ws, r, 4, "side",       nullptr);
    worksheet_write_string(ws, r, 5, "price_t",    nullptr);
    worksheet_write_string(ws, r, 6, "shares_abs", nullptr);
    worksheet_write_string(ws, r, 7, "shares",     nullptr);
    ++r;

    for (int i=0;i<N;++i){
        double Ai   = std::max(0.0, A_socp(i));
        double pr   = price(i);
        double abs_w= (P0>0.0)? std::min(1.0, Ai / std::max(1e-12, P0)) : 0.0;
        double w    = abs_w * (double)side_socp(i);
        double sh_abs = (pr>0.0)? Ai / pr : 0.0;
        double sh     = sh_abs * (double)side_socp(i);

        worksheet_write_number(ws, r, 0, i,        nullptr);
        worksheet_write_number(ws, r, 1, w,        nullptr);
        worksheet_write_number(ws, r, 2, abs_w,    nullptr);
        worksheet_write_number(ws, r, 3, Ai,       nullptr);
        worksheet_write_number(ws, r, 4, side_socp(i), nullptr);
        worksheet_write_number(ws, r, 5, pr,       nullptr);
        worksheet_write_number(ws, r, 6, sh_abs,   nullptr);
        worksheet_write_number(ws, r, 7, sh,       nullptr);
        ++r;
    }

    // 寫上 P0 與日期
    worksheet_write_string(ws, r+1, 0, "P0", nullptr);
    worksheet_write_number(ws, r+1, 1, P0,  nullptr);
    worksheet_write_string(ws, r+2, 0, "t",  nullptr);
    worksheet_write_number(ws, r+2, 1, t,   nullptr);

    workbook_close(wb);
#else
    // CSV 後備
    std::ofstream f(out_path);
    f << "index,w_est,abs_w,A_notional,side,price_t,shares_abs,shares\n";
    for (int i=0;i<N;++i){
        double Ai   = std::max(0.0, A_socp(i));
        double pr   = price(i);
        double abs_w= (P0>0.0)? std::min(1.0, Ai / std::max(1e-12, P0)) : 0.0;
        double w    = abs_w * (double)side_socp(i);
        double sh_abs = (pr>0.0)? Ai / pr : 0.0;
        double sh     = sh_abs * (double)side_socp(i);
        f << i << "," << w << "," << abs_w << "," << Ai << "," << side_socp(i)
          << "," << pr << "," << sh_abs << "," << sh << "\n";
    }
    f << "P0," << P0 << "\n";
    f << "t,"  << t  << "\n";
#endif
}


