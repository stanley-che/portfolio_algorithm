//process_risk.hpp
#pragma once
#include "data_loader_all.hpp"   // 確保看到 dlx::DataLoader 完整宣告
#include "process_types.hpp"
#include "process_math.hpp"
#include "process_risk.hpp"
#include "process_forecast.hpp"
#include <iostream>

namespace process {

class ProcessEngine {
public:
    explicit ProcessEngine(const dlx::DataLoader& dl) : dl_(dl) {}
    const dlx::DataLoader& dl() const { return this->dl_; } 
    Eigen::MatrixXd xsection_standardize(const Eigen::MatrixXd& X, const ProcessingConfig& cfg) const;

    RiskOutput build_liquidity_scaled_risk(int t, const RiskConfig& cfg) const;
    CostBreakdown estimate_costs(int t, const Eigen::VectorXd& A,
                                 const Eigen::VectorXi& side,
                                 const CostConfig& cfg) const;
    BlackSwanOutput black_swan_scale(int t, const BlackSwanConfig& cfg,
                                     bool external_event=false) const;

protected:
        const dlx::DataLoader& dl_;
        Eigen::VectorXd realized_ret_t(int t) const;
    
private:
    

    // ---- internal helpers (private) ----
    Eigen::MatrixXd build_log_return_matrix() const;
    Eigen::VectorXd stdize_row(const Eigen::VectorXd& v, const ProcessingConfig& cfg) const;
};

// -------- impl (inline) --------
inline Eigen::MatrixXd ProcessEngine::xsection_standardize(const Eigen::MatrixXd& X,
                                                          const ProcessingConfig& cfg) const {
    Eigen::MatrixXd Y(X.rows(), X.cols());
    for(int r=0;r<X.rows();++r){
        Eigen::VectorXd row = X.row(r).transpose();
        if (cfg.mode==XSectionStandardize::ZScoreClip)
            Y.row(r) = detail::cs_zscore(row, cfg.clip).transpose();
        else
            Y.row(r) = detail::cs_quantile_map(row, cfg.clip).transpose();

        if (cfg.fill_missing_with_zero){
            for(int c=0;c<Y.cols();++c) if(!std::isfinite(Y(r,c))) Y(r,c)=0.0;
        }
    }
    return Y;
}

inline Eigen::MatrixXd ProcessEngine::build_log_return_matrix() const {
    Eigen::MatrixXd R(dl_.C().rows(), dl_.C().cols());
    R.setZero();
    for(int r=1;r<dl_.C().rows();++r)
        for(int c=0;c<dl_.C().cols();++c){
            double c0=dl_.C()(r-1,c), c1=dl_.C()(r,c);
            R(r,c) = (std::isfinite(c0)&&std::isfinite(c1)&&c0>0&&c1>0)? std::log(c1/c0) : 0.0;
        }
    return R;
}

inline Eigen::VectorXd ProcessEngine::realized_ret_t(int t) const {
    const int N = (int)dl_.symbols().size();
    Eigen::VectorXd r = Eigen::VectorXd::Zero(N);
    if (t<=0) return r;
    for(int c=0;c<dl_.C().cols();++c){
        double c0=dl_.C()(t-1,c), c1=dl_.C()(t,c);
        r(c) = (std::isfinite(c0)&&std::isfinite(c1)&&c0>0)? (c1/c0 - 1.0) : 0.0;
    }
    return r;
}

// ===== Risk =====
inline RiskOutput ProcessEngine::build_liquidity_scaled_risk(int t, const RiskConfig& cfg) const {
    const int N = (int)dl_.symbols().size();
    RiskOutput out;
    out.Sigma_tilde = Eigen::MatrixXd::Zero(N,N);
    out.L           = Eigen::MatrixXd::Zero(N,N);
    out.sigma_i     = Eigen::VectorXd::Zero(N);
    out.Dliq_i      = Eigen::VectorXd::Ones(N);

    const int T = dl_.C().rows();
    if (N==0 || T<2 || t<=0 || t>=T) return out;

    Eigen::MatrixXd R = build_log_return_matrix();

    // EWMA cov
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
    for(int i=0;i<N;++i) sd(i) = std::max(sd(i), 1e-6);

    Eigen::MatrixXd Corr = Eigen::MatrixXd::Zero(N,N);
    for(int i=0;i<N;++i) for(int j=0;j<N;++j)
        Corr(i,j) = (i==j) ? 1.0 : (S(i,j)/(sd(i)*sd(j)));

    // shrinkage
    Corr = 0.90*Corr + 0.10*Eigen::MatrixXd::Identity(N,N);

    // sigma_i
    if (cfg.sigma_source == SigmaSource::GK) out.sigma_i = detail::row_at(dl_.feat_GKVol(), t);
    else out.sigma_i = detail::rolling_std_last(R, t, cfg.std_window);
    for(int i=0;i<N;++i) out.sigma_i(i)=std::clamp(out.sigma_i(i), 1e-6, 10.0);

    // Dliq
    Eigen::VectorXd ts   = detail::row_at(dl_.feat_TurnoverShare(), t);
    Eigen::VectorXd VALt = detail::row_at(dl_.VAL(), t);
    Eigen::VectorXd ADV  = detail::rolling_mean_last(dl_.VAL(), t, 30);

    double ts_bar=0.0; for(int i=0;i<N;++i) ts_bar += (std::isfinite(ts(i))?ts(i):0.0);
    ts_bar /= std::max(1,N);

    for(int i=0;i<N;++i){
        double adv_i = std::max(ADV(i), 1e-8);
        double val_i = std::max(VALt(i), 1e-8);
        double liq = std::sqrt(detail::safe_div(std::max(0.0,val_i), adv_i));

        double liq_term = 1.0 + cfg.c_liq * detail::safe_div(1.0, std::max(1e-6, liq));
        liq_term = std::clamp(liq_term, 0.5, 3.0);

        double conc = detail::safe_div(ts_bar, std::max(1e-6, ts(i)+cfg.eps));
        conc = std::clamp(conc, 0.5, 2.0);

        out.Dliq_i(i) = std::clamp(liq_term * conc, 0.5, 5.0);
        if(!std::isfinite(out.Dliq_i(i))) out.Dliq_i(i)=1.0;
    }

    Eigen::VectorXd diagD = out.Dliq_i.array() * out.sigma_i.array();
    Eigen::MatrixXd D = diagD.asDiagonal();
    out.Sigma_tilde = D * Corr * D;

    // diagonal loading
    for(int i=0;i<N;++i) out.Sigma_tilde(i,i) += 1e-4;

    // PSD fix
    Eigen::LLT<Eigen::MatrixXd> llt(out.Sigma_tilde);
    if (llt.info()==Eigen::Success) out.L = llt.matrixL();
    else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(out.Sigma_tilde);
        Eigen::VectorXd ev=es.eigenvalues();
        Eigen::MatrixXd U=es.eigenvectors();
        for(int i=0;i<ev.size();++i) ev(i)=std::max(ev(i), 1e-10);
        out.Sigma_tilde = U * ev.asDiagonal() * U.transpose();
        out.L = out.Sigma_tilde.llt().matrixL();
    }
    return out;
}

// ===== Cost =====
inline CostBreakdown ProcessEngine::estimate_costs(int t, const Eigen::VectorXd& A,
                                                   const Eigen::VectorXi& side,
                                                   const CostConfig& cfg) const {
    const int N = (int)A.size();
    CostBreakdown cb;
    cb.fee_buy  = Eigen::VectorXd::Zero(N);
    cb.fee_sell = Eigen::VectorXd::Zero(N);
    cb.tax_sell = Eigen::VectorXd::Zero(N);
    cb.impact   = Eigen::VectorXd::Zero(N);

    Eigen::VectorXd ADV   = detail::rolling_mean_last(dl_.VAL(), t, cfg.adv_window);
    Eigen::VectorXd sigma = detail::row_at(dl_.feat_GKVol(), t);
    Eigen::VectorXd imb   = detail::row_at(dl_.feat_Imbalance(), t);

    for(int i=0;i<N;++i){
        double Ai = std::max(0.0, A(i));
        if (side(i) > 0) cb.fee_buy(i)  = std::max(cfg.min_fee, cfg.fee_rate * Ai);
        if (side(i) < 0) {
            cb.fee_sell(i) = std::max(cfg.min_fee, cfg.fee_rate * Ai);
            cb.tax_sell(i) = cfg.tax_rate * Ai;
        }
        double adv_i = std::max(ADV(i), 1e-8);
        double liq = std::sqrt(detail::safe_div(Ai, adv_i));
        double sig = std::clamp(sigma(i), 0.0, 5.0);
        double imb_fac = 1.0 + cfg.beta_imb * std::min(1.0, std::abs(imb(i)));
        double impact_rate = (0.5 * cfg.gamma_init) * sig * imb_fac * std::clamp(liq, 0.0, 10.0);
        cb.impact(i) = std::min(impact_rate * Ai, 1e6);
    }
    return cb;
}

// ===== Black Swan =====
inline BlackSwanOutput ProcessEngine::black_swan_scale(int t, const BlackSwanConfig& cfg,
                                                      bool external_event) const {
    BlackSwanOutput out; out.m = cfg.m_default;

    double m_gap  = detail::median_vec(detail::row_at(dl_.feat_Gap(), t));
    double gk_now = detail::median_vec(detail::row_at(dl_.feat_GKVol(), t));

    int win=std::min(30, t+1);
    double gk_30=0.0;
    for(int r=t-win+1;r<=t;++r) gk_30 += detail::median_vec(detail::row_at(dl_.feat_GKVol(), r));
    gk_30 /= std::max(1, win);

    double ratio = (gk_30>0.0)? gk_now/gk_30 : 1.0;
    double imb_med = detail::median_vec(detail::row_at(dl_.feat_Imbalance(), t));

    auto soften = [](double cur,double target){ return 0.5*cur + 0.5*target; };

    if (m_gap <= cfg.gap_threshold) out.m = std::min(out.m, soften(out.m, cfg.m_gap_bad));
    if (ratio  >= cfg.vol_ratio_th) out.m = std::min(out.m, soften(out.m, cfg.m_vol_high));
    if (imb_med<= cfg.imb_median_th)out.m = std::min(out.m, soften(out.m, cfg.m_imb_bad));
    if (external_event)            out.m = std::min(out.m, soften(out.m, cfg.m_event));

    out.m = std::clamp(out.m, 0.7, 1.3);
    return out;
}

} // namespace process
