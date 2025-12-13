// prediction_model.hpp
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "prediction_types.hpp"

namespace prediction {

class LinearModelTrainer {
public:
    explicit LinearModelTrainer(const TrainConfig& tr) : tr_(tr) {}

    // 由 Gram G=X'WX, g=X'Wy 解 beta
    Eigen::VectorXd fit_beta(const Eigen::MatrixXd& G, const Eigen::VectorXd& g) const {
        const int p = (int)g.size();
        if (p == 0) return Eigen::VectorXd();

        if (tr_.model == ModelType::Ridge || tr_.model == ModelType::GBDT) {
            Eigen::MatrixXd A = G;
            for (int j=0;j<p;++j) A(j,j) += tr_.lambda;
            return A.ldlt().solve(g);
        }

        // Lasso
        return lasso_coordinate_descent(G, g, tr_.lambda, tr_.lasso_max_iter, tr_.lasso_tol);
    }

    // half-life -> daily weight decay
    double lambda_time() const {
        return std::exp(std::log(0.5) / std::max(1e-12, tr_.half_life));
    }

private:
    TrainConfig tr_;

    static double soft_threshold(double x, double k) {
        double s = std::abs(x) - k;
        return (s > 0.0) ? std::copysign(s, x) : 0.0;
    }

    static Eigen::VectorXd lasso_coordinate_descent(
        const Eigen::MatrixXd& G, const Eigen::VectorXd& g,
        double lambda, int max_iter, double tol)
    {
        const int p = (int)g.size();
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);

        for (int it=0; it<max_iter; ++it){
            double maxdiff = 0.0;
            for (int j=0;j<p;++j){
                double rho = g(j) - (G.row(j).dot(beta) - G(j,j)*beta(j));
                double new_b = soft_threshold(rho, lambda) / (G(j,j) + 1e-12);
                maxdiff = std::max(maxdiff, std::abs(new_b - beta(j)));
                beta(j) = new_b;
            }
            if (maxdiff < tol) break;
        }
        return beta;
    }
};

} // namespace prediction
