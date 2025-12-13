// prediction.hpp
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "data_loader.h"
#include "process.h"
#include "prediction_types.hpp"
#include "prediction_features.hpp"
#include "prediction_model.hpp"
#include "prediction_calibration.hpp"

namespace prediction {

class Predictor {
public:
    Predictor(TrainConfig tr,
              process::ProcessingConfig px,
              CalibrationConfig cal,
              EnhancementConfig enh)
        : tr_(std::move(tr))
        , px_(std::move(px))
        , cal_(std::move(cal))
        , enh_(std::move(enh))
        , feat_(px_)
        , trainer_(tr_)
        , calib_(cal_, enh_, px_)
    {}

    PredictionOutput predict_day(const DataLoader& dl, int t, TargetType target) const {
        PredictionOutput out;
        const int N = (int)dl.symbols().size();
        const int P = FeatureBuilder::P;

        out.rhat_raw = Eigen::VectorXd::Zero(N);
        out.rhat_cal = Eigen::VectorXd::Zero(N);
        out.rhat_enh = Eigen::VectorXd::Zero(N);
        out.beta     = Eigen::VectorXd::Zero(P);

        if (N == 0) return out;
        if (t <= 0 || t >= dl.C().rows()) return out;

        // ---- (1) rolling train on [t-W, t-1] with time weights ----
        int start = std::max(1, t - tr_.window);
        double lam_time = trainer_.lambda_time();

        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(P, P);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(P);

        double w = 1.0; // s=t-1 weight largest
        for (int s=t-1; s>=start; --s) {
            Eigen::MatrixXd Xs = feat_.build_X_at(dl, s);         // (N×P)
            Eigen::VectorXd ys = target_return_between(dl, s, target); // (N)
            accumulate_gram_and_xTy(Xs, ys, w, G, g);
            w *= lam_time;
        }

        out.beta = trainer_.fit_beta(G, g);

        // ---- (2) predict at day t ----
        Eigen::MatrixXd Xt = feat_.build_X_at(dl, t);
        out.rhat_raw = (Xt * out.beta).eval();

        // ---- (3) calibration ----
        out.rhat_cal = calib_.calibrate_ecdf_probit(out.rhat_raw);

        // ---- (4) enhancement ----
        out.rhat_enh = calib_.enhance(dl, t, out.rhat_cal);

        return out;
    }

private:
    TrainConfig tr_;
    process::ProcessingConfig px_;
    CalibrationConfig cal_;
    EnhancementConfig enh_;

    FeatureBuilder feat_;
    LinearModelTrainer trainer_;
    Calibrator calib_;

    // target return aligned with day s features
    static Eigen::VectorXd target_return_between(const DataLoader& dl, int s, TargetType target) {
        const int N = (int)dl.symbols().size();
        Eigen::VectorXd r = Eigen::VectorXd::Zero(N);
        if (s+1 >= dl.C().rows()) return r;

        for (int i=0;i<N;++i){
            double O_s   = dl.O()(s, i);
            double O_sp1 = dl.O()(s+1, i);
            double C_s   = dl.C()(s, i);
            double C_sp1 = dl.C()(s+1, i);

            double val = 0.0;
            if (target==TargetType::OO) {
                val = (std::isfinite(O_s) && std::isfinite(O_sp1) && O_s>0) ? (O_sp1/O_s - 1.0) : 0.0;
            } else if (target==TargetType::OC) {
                val = (std::isfinite(O_sp1) && std::isfinite(C_sp1) && O_sp1>0) ? (C_sp1/O_sp1 - 1.0) : 0.0;
            } else { // CC
                val = (std::isfinite(C_s) && std::isfinite(C_sp1) && C_s>0) ? (C_sp1/C_s - 1.0) : 0.0;
            }
            r(i) = val;
        }
        return r;
    }

    // G += w X^T X, g += w X^T y
    static void accumulate_gram_and_xTy(
        const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double w,
        Eigen::MatrixXd& G, Eigen::VectorXd& g)
    {
        G.noalias() += w * (X.transpose() * X);
        g.noalias() += w * (X.transpose() * y);
    }
};

} // namespace prediction
