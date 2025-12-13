//prediction_features.hpp
// prediction_features.hpp
#pragma once
#include <Eigen/Dense>
#include "data_loader.h"
#include "process.h" // 只用 xsection_standardize
#include "prediction_types.hpp"

namespace prediction {

class FeatureBuilder {
public:
    static constexpr int P = 10;

    explicit FeatureBuilder(const process::ProcessingConfig& px)
        : px_(px) {}

    // 外部只需要拿 (N×P) 的 Xt
    Eigen::MatrixXd build_X_at(const DataLoader& dl, int t) const {
        const int N = (int)dl.symbols().size();
        Eigen::MatrixXd X(N, P);

        Eigen::VectorXd gap   = stdize_row(dl.feat_Gap().row(t).transpose());
        Eigen::VectorXd m5    = stdize_row(dl.feat_Mom5().row(t).transpose());
        Eigen::VectorXd m10   = stdize_row(dl.feat_Mom10().row(t).transpose());
        Eigen::VectorXd m20   = stdize_row(dl.feat_Mom20().row(t).transpose());
        Eigen::VectorXd bias  = stdize_row(dl.feat_BIAS().row(t).transpose());
        Eigen::VectorXd bf    = stdize_row(dl.feat_BrokerStrength().row(t).transpose());
        Eigen::VectorXd logV  = stdize_row(dl.feat_Liquidity().row(t).transpose());
        Eigen::VectorXd ts    = stdize_row(dl.feat_TurnoverShare().row(t).transpose());
        Eigen::VectorXd imb   = stdize_row(dl.feat_Imbalance().row(t).transpose());
        Eigen::VectorXd gk    = stdize_row(dl.feat_GKVol().row(t).transpose());

        X.col(0)=gap;  X.col(1)=m5;  X.col(2)=m10; X.col(3)=m20;
        X.col(4)=bias; X.col(5)=bf;  X.col(6)=logV;X.col(7)=ts;
        X.col(8)=imb;  X.col(9)=gk;

        return X;
    }

private:
    process::ProcessingConfig px_;

    // 逐日橫斷面標準化：封裝 process::xsection_standardize
    Eigen::VectorXd stdize_row(const Eigen::VectorXd& v) const {
        Eigen::MatrixXd tmp(1, v.size());
        tmp.row(0) = v.transpose();
        Eigen::MatrixXd Y = process::xsection_standardize(tmp, px_);
        return Y.row(0).transpose().eval();
    }
};

} // namespace prediction
