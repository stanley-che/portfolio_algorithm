//process_forecast.hpp
#pragma once
#include "process_risk.hpp"   // 直接重用 ProcessEngine class / helpers
#include <tuple>

namespace process {

class ProcessEngineForecast : public ProcessEngine {
public:
    using ProcessEngine::ProcessEngine; // 繼承 ctor

    ForecastOutput stage1_forecast(int t,
                                   const ProcessingConfig& px_cfg,
                                   const ForecastConfig& fc,
                                   const std::optional<Eigen::MatrixXd>& next_day_returns = std::nullopt) const;

private:
    Eigen::VectorXd wls_beta_(const std::vector<Eigen::VectorXd>& Xhist,
                             const std::vector<double>& yhist,
                             double half_life) const;
};

inline Eigen::VectorXd ProcessEngineForecast::wls_beta_(
    const std::vector<Eigen::VectorXd>& Xhist,
    const std::vector<double>& yhist,
    double half_life) const
{
    if (Xhist.empty()) return Eigen::VectorXd();
    const int p = (int)Xhist[0].size();
    Eigen::MatrixXd XtWX = Eigen::MatrixXd::Zero(p,p);
    Eigen::VectorXd XtWy = Eigen::VectorXd::Zero(p);

    const double alpha = detail::alpha_from_half_life(half_life);
    double w=1.0;
    for(int k=(int)Xhist.size()-1;k>=0;--k){
        const auto& x=Xhist[k]; const double y=yhist[k];
        XtWX.noalias() += w*(x*x.transpose());
        XtWy.noalias() += w*(x*y);
        w *= (1.0 - alpha);
    }
    const double lambda = 1e-3;
    return (XtWX + lambda*Eigen::MatrixXd::Identity(p,p)).ldlt().solve(XtWy);
}

inline ForecastOutput ProcessEngineForecast::stage1_forecast(
    int t, const ProcessingConfig& px_cfg, const ForecastConfig& fc,
    const std::optional<Eigen::MatrixXd>& next_day_returns) const
{
    ForecastOutput out;
    const int N = (int)this->dl_.symbols().size();
    out.rhat_raw = Eigen::VectorXd::Zero(N);
    out.rhat_cal = Eigen::VectorXd::Zero(N);
    out.rhat_enh = Eigen::VectorXd::Zero(N);

    const int T = this->dl_.C().rows();
    if (N==0 || T<2 || t<=0 || t>=T || !fc.use_linear) return out;

    // 取特徵 row(t) 並做橫斷面標準化（用 xsection_standardize）
    auto stdize = [&](const Eigen::VectorXd& v){
        Eigen::MatrixXd tmp(1, N);
        tmp.row(0) = (v.size()==N? v.transpose() : Eigen::RowVectorXd::Zero(N));
        Eigen::VectorXd z = this->xsection_standardize(tmp, px_cfg).row(0).transpose();
        for(int i=0;i<N;++i) if(!std::isfinite(z(i))) z(i)=0.0;
        return z;
    };

    Eigen::VectorXd gap  = stdize(detail::row_at(this->dl_.feat_Gap(),            t));
    Eigen::VectorXd m5   = stdize(detail::row_at(this->dl_.feat_Mom5(),           t));
    Eigen::VectorXd m10  = stdize(detail::row_at(this->dl_.feat_Mom10(),          t));
    Eigen::VectorXd m20  = stdize(detail::row_at(this->dl_.feat_Mom20(),          t));
    Eigen::VectorXd bias = stdize(detail::row_at(this->dl_.feat_BIAS(),           t));
    Eigen::VectorXd bf   = stdize(detail::row_at(this->dl_.feat_BrokerStrength(), t));
    Eigen::VectorXd lv   = stdize(detail::row_at(this->dl_.feat_Liquidity(),      t));
    Eigen::VectorXd ts   = stdize(detail::row_at(this->dl_.feat_TurnoverShare(),  t));
    Eigen::VectorXd im   = stdize(detail::row_at(this->dl_.feat_Imbalance(),      t));
    Eigen::VectorXd gk   = stdize(detail::row_at(this->dl_.feat_GKVol(),          t));

    constexpr int P=10;
    Eigen::MatrixXd X(N,P);
    X.col(0)=gap; X.col(1)=m5; X.col(2)=m10; X.col(3)=m20;
    X.col(4)=bias;X.col(5)=bf; X.col(6)=lv;  X.col(7)=ts;
    X.col(8)=im;  X.col(9)=gk;

    // ===== build history for WLS =====
    const int start = std::max(1, t - fc.lookback);
    std::vector<Eigen::VectorXd> xhist;
    std::vector<double> yhist;
    xhist.reserve((size_t)(t-start)* (size_t)N);
    yhist.reserve((size_t)(t-start)* (size_t)N);

    const int Ty = next_day_returns ? next_day_returns->rows() : T;
    const int Ny = next_day_returns ? next_day_returns->cols() : N;

    for(int tau=start; tau<=t-1; ++tau){
        if (next_day_returns && (tau>=Ty || Ny!=N)) break;

        Eigen::VectorXd g   = stdize(detail::row_at(this->dl_.feat_Gap(),            tau));
        Eigen::VectorXd m5_ = stdize(detail::row_at(this->dl_.feat_Mom5(),           tau));
        Eigen::VectorXd m10_= stdize(detail::row_at(this->dl_.feat_Mom10(),          tau));
        Eigen::VectorXd m20_= stdize(detail::row_at(this->dl_.feat_Mom20(),          tau));
        Eigen::VectorXd b   = stdize(detail::row_at(this->dl_.feat_BIAS(),           tau));
        Eigen::VectorXd bf_ = stdize(detail::row_at(this->dl_.feat_BrokerStrength(), tau));
        Eigen::VectorXd lv_ = stdize(detail::row_at(this->dl_.feat_Liquidity(),      tau));
        Eigen::VectorXd ts_ = stdize(detail::row_at(this->dl_.feat_TurnoverShare(),  tau));
        Eigen::VectorXd im_ = stdize(detail::row_at(this->dl_.feat_Imbalance(),      tau));
        Eigen::VectorXd gk_ = stdize(detail::row_at(this->dl_.feat_GKVol(),          tau));

        Eigen::VectorXd y = next_day_returns
            ? next_day_returns->row(tau).transpose()
            : this->realized_ret_t(tau+1);

        for(int c=0;c<N;++c){
            Eigen::VectorXd x(P);
            x << g(c), m5_(c), m10_(c), m20_(c), b(c), bf_(c), lv_(c), ts_(c), im_(c), gk_(c);
            xhist.push_back(std::move(x));
            yhist.push_back(std::isfinite(y(c))?y(c):0.0);
        }
    }

    if(xhist.empty()) return out;
    Eigen::VectorXd beta = wls_beta_(xhist, yhist, fc.half_life);
    if(beta.size()==P) out.rhat_raw = (X*beta).eval();

    // 這裡你原本還有 rhat_cal / rhat_enh（你可以接著把規則塞回來）
    out.rhat_cal = out.rhat_raw;
    out.rhat_enh = out.rhat_raw;
    return out;
}

} // namespace process
