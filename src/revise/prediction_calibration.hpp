// prediction_calibration.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include "prediction_types.hpp"
#include "data_loader.hpp"
//#include "process_all.hpp"
#include "process_risk.hpp"
namespace prediction {

class Calibrator {
public:
    Calibrator(const CalibrationConfig& cal,
               const EnhancementConfig& enh,
               const process::ProcessingConfig& px,
               const process::ProcessEngine& eng)
        : cal_(cal), enh_(enh), px_(px), eng_(eng) {}

    Eigen::VectorXd calibrate_ecdf_probit(const Eigen::VectorXd& raw) const {
        const int N = (int)raw.size();
        std::vector<std::pair<double,int>> a(N);
        for (int i=0;i<N;++i) a[i] = { std::isfinite(raw(i)) ? raw(i) : 0.0, i };
        std::sort(a.begin(), a.end(), [](auto&x,auto&y){ return x.first < y.first; });

        Eigen::VectorXd z = Eigen::VectorXd::Zero(N);
        for (int r=0;r<N;++r){
            double u  = (r+1.0) / (N+1.0);
            double zz = inv_norm_cdf(u);
            zz = std::clamp(zz, -cal_.z_clip, cal_.z_clip);
            z(a[r].second) = zz;
        }

        constexpr double z75 = 0.67448975; // N(0,1) 75th quantile
        double acoef = (cal_.q75 - cal_.q50) / z75;
        double bcoef = cal_.q50;

        return (acoef * z.array() + bcoef).matrix().eval();
    }

    Eigen::VectorXd enhance(const dlx::DataLoader& dl, int t, const Eigen::VectorXd& base) const {
        Eigen::VectorXd bf   = stdize_row(dl.feat_BrokerStrength().row(t).transpose());
        Eigen::VectorXd imb  = stdize_row(dl.feat_Imbalance().row(t).transpose());
        Eigen::VectorXd ts   = stdize_row(dl.feat_TurnoverShare().row(t).transpose());
        Eigen::VectorXd bias = stdize_row(dl.feat_BIAS().row(t).transpose());

        Eigen::VectorXd penalty = (bias.array().abs() - enh_.theta_bias).cwiseMax(0.0);

        Eigen::VectorXd res =
            base
          + enh_.theta_b   * bf
          + enh_.theta_imb * imb
          + enh_.theta_ts  * ts
          - enh_.kappa_bias* penalty;

        return res.eval();
    }

private:
    CalibrationConfig cal_;
    EnhancementConfig enh_;
    process::ProcessingConfig px_;
    const process::ProcessEngine& eng_;
    Eigen::VectorXd stdize_row(const Eigen::VectorXd& v) const {
        Eigen::MatrixXd tmp(1, v.size());
        tmp.row(0) = v.transpose();
        Eigen::MatrixXd Y = eng_.xsection_standardize(tmp, px_);
        return Y.row(0).transpose().eval();
    }

    // Acklam inverse normal CDF (probit)
    static double inv_norm_cdf(double p){
        static const double a1=-3.969683028665376e+01,a2= 2.209460984245205e+02,
                            a3=-2.759285104469687e+02,a4= 1.383577518672690e+02,
                            a5=-3.066479806614716e+01,a6= 2.506628277459239e+00;
        static const double b1=-5.447609879822406e+01,b2= 1.615858368580409e+02,
                            b3=-1.556989798598866e+02,b4= 6.680131188771972e+01,
                            b5=-1.328068155288572e+01;
        static const double c1=-7.784894002430293e-03,c2=-3.223964580411365e-01,
                            c3=-2.400758277161838e+00,c4=-2.549732539343734e+00,
                            c5= 4.374664141464968e+00,c6= 2.938163982698783e+00;
        static const double d1= 7.784695709041462e-03,d2= 3.224671290700398e-01,
                            d3= 2.445134137142996e+00,d4= 3.754408661907416e+00;

        const double plow=0.02425, phigh=1-plow;
        if (p<=0.0) return -1e9;
        if (p>=1.0) return  1e9;

        double q,r;
        if (p<plow){
            q=std::sqrt(-2*std::log(p));
            return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/
                   ((((d1*q+d2)*q+d3)*q+d4)*q+1);
        } else if (p>phigh){
            q=std::sqrt(-2*std::log(1-p));
            return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/
                     ((((d1*q+d2)*q+d3)*q+d4)*q+1);
        } else {
            q=p-0.5; r=q*q;
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/
                   (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
        }
    }
};

} // namespace prediction
