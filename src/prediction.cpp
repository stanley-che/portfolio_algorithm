#include "prediction.h"
#include <algorithm>
#include <deque>
#include <cmath>

// --------- 小工具 ----------
static inline double safe_div(double a,double b){ return (std::abs(b)<1e-12)?0.0:a/b; }
static inline double alpha_from_half_life(double hl){ return 1.0 - std::exp(std::log(0.5)/std::max(1.0,hl)); }

// Acklam 近似的標準常態逆CDF（probit）
static double inv_norm_cdf(double p){
    // p in (0,1)
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
    double q,r;
    if (p<=0.0) return -1e9;
    if (p>=1.0) return  1e9;
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

// 逐日橫斷面標準化（包裝 process::xsection_standardize）
static Eigen::VectorXd stdize_row(const Eigen::VectorXd& v,
                                  const process::ProcessingConfig& px){
    Eigen::MatrixXd tmp(1, v.size());
    tmp.row(0) = v.transpose();
    Eigen::MatrixXd Y = process::xsection_standardize(tmp, px); // 用具名型別，別用 auto
    return Y.row(0).transpose().eval();  // <-- 這行是關鍵
}

// ========== 取特徵：與 §2.6 一致 ==========
Eigen::MatrixXd build_feature_matrix_at(
    const DataLoader& dl, int t, const process::ProcessingConfig& px)
{
    const int N = (int)dl.symbols().size();
    const int P = 10; // [Gap, Mom5, Mom10, Mom20, BIAS10, BF(EMA3), logVAL, TS, Imb, GK]
    Eigen::MatrixXd X(N,P);

    Eigen::VectorXd gap   = stdize_row(dl.feat_Gap().row(t).transpose(), px);
    Eigen::VectorXd m5    = stdize_row(dl.feat_Mom5().row(t).transpose(), px);
    Eigen::VectorXd m10   = stdize_row(dl.feat_Mom10().row(t).transpose(), px);
    Eigen::VectorXd m20   = stdize_row(dl.feat_Mom20().row(t).transpose(), px);
    Eigen::VectorXd bias  = stdize_row(dl.feat_BIAS().row(t).transpose(), px);
    Eigen::VectorXd bf    = stdize_row(dl.feat_BrokerStrength().row(t).transpose(), px);
    Eigen::VectorXd logVAL= stdize_row(dl.feat_Liquidity().row(t).transpose(), px);
    Eigen::VectorXd ts    = stdize_row(dl.feat_TurnoverShare().row(t).transpose(), px);
    Eigen::VectorXd imb   = stdize_row(dl.feat_Imbalance().row(t).transpose(), px);
    Eigen::VectorXd gk    = stdize_row(dl.feat_GKVol().row(t).transpose(), px);

    X.col(0)=gap; X.col(1)=m5; X.col(2)=m10; X.col(3)=m20;
    X.col(4)=bias;X.col(5)=bf; X.col(6)=logVAL; X.col(7)=ts; X.col(8)=imb; X.col(9)=gk;
    return X;
}

// ========== 3.1 目標報酬 ==========
Eigen::VectorXd target_return_between(const DataLoader& dl, int s, TargetType target){
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

// ========== 3.2 累積 X'WX / X'Wy（日權重同一天所有股票共享） ==========
static void accumulate_gram_and_xTy(
    const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double w,
    Eigen::MatrixXd& G, Eigen::VectorXd& g)
{
    // G += w * X^T X,  g += w * X^T y
    G.noalias() += w * (X.transpose() * X);
    g.noalias() += w * (X.transpose() * y);
}

// Lasso：座標下降在 Gram 形式下
static Eigen::VectorXd lasso_coordinate_descent(
    const Eigen::MatrixXd& G, const Eigen::VectorXd& g,
    double lambda, int max_iter, double tol)
{
    const int p = (int)g.size();
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    auto soft = [](double x, double k){ double s = std::abs(x) - k; return (s>0)? std::copysign(s,x) : 0.0; };

    for (int it=0; it<max_iter; ++it){
        double maxdiff = 0.0;
        for (int j=0;j<p;++j){
            // ρ_j = g_j - Σ_{k≠j} G_{jk} β_k
            double rho = g(j) - (G.row(j).dot(beta) - G(j,j)*beta(j));
            double new_b = soft(rho, lambda) / (G(j,j) + 1e-12);
            maxdiff = std::max(maxdiff, std::abs(new_b - beta(j)));
            beta(j) = new_b;
        }
        if (maxdiff < tol) break;
    }
    return beta;
}

// ========== 3.3 ECDF→Probit 校準 ==========
static Eigen::VectorXd quantile_calibrate(const Eigen::VectorXd& raw,
                                          const CalibrationConfig& c){
    const int N = (int)raw.size();
    std::vector<std::pair<double,int>> a(N);
    for (int i=0;i<N;++i) a[i] = { std::isfinite(raw(i)) ? raw(i) : 0.0, i };
    std::sort(a.begin(), a.end(), [](auto&x,auto&y){ return x.first < y.first; });

    Eigen::VectorXd z = Eigen::VectorXd::Zero(N);
    for (int r=0;r<N;++r){
        double u  = (r+1.0) / (N+1.0);
        double zz = inv_norm_cdf(u);
        zz = std::clamp(zz, -c.z_clip, c.z_clip);
        z(a[r].second) = zz;
    }

    constexpr double z75 = 0.67448975;
    double acoef = (c.q75 - c.q50) / z75;
    double bcoef = c.q50;

    // 這裡若直接 return acoef * z.array() + bcoef; 會延遲求值且引用 z（已被銷毀）
    Eigen::VectorXd out = (acoef * z.array() + bcoef).matrix();
    return out;  // 或寫成：return (acoef * z.array() + bcoef).matrix().eval();
}

// ========== 3.x 增強： +θ_b BF + θ_imb Imb + θ_ts TS − κ_bias max(0,|BIAS|−θ_bias) ==========
static Eigen::VectorXd enhance_with_flows_bias(const DataLoader& dl, int t,
    const process::ProcessingConfig& px, const EnhancementConfig& e,
    const Eigen::VectorXd& base){
    auto stdize = [&](const Eigen::VectorXd& vv){ return stdize_row(vv, px); };

    Eigen::VectorXd bf   = stdize(dl.feat_BrokerStrength().row(t).transpose());
    Eigen::VectorXd imb  = stdize(dl.feat_Imbalance().row(t).transpose());
    Eigen::VectorXd ts   = stdize(dl.feat_TurnoverShare().row(t).transpose());
    Eigen::VectorXd bias = stdize(dl.feat_BIAS().row(t).transpose());

    Eigen::VectorXd penalty = (bias.array().abs() - e.theta_bias).cwiseMax(0.0);

    Eigen::VectorXd res =
        base
      + e.theta_b   * bf
      + e.theta_imb * imb
      + e.theta_ts  * ts
      - e.kappa_bias* penalty;

    return res;  // 或：return res.eval();
}

// ========== 主函式：predict_day ==========
PredictionOutput predict_day(
    const DataLoader& dl, int t, TargetType target,
    const TrainConfig& tr,
    const process::ProcessingConfig& px,
    const CalibrationConfig& cal,
    const EnhancementConfig& enh)
{
    PredictionOutput out;
    const int N = (int)dl.symbols().size();
    const int P = 10;

    // （1）在 [t-W, t-1] 走步訓練
    int start = std::max(1, t - tr.window);
    double lam = std::exp(std::log(0.5)/tr.half_life); // ω_s = lam^{(t-1-s)}
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(P,P); // X' W X
    Eigen::VectorXd g = Eigen::VectorXd::Zero(P);   // X' W y

    double ww = 1.0; // t-1 權重將最大
    // 自近至遠迭代（更穩定）：s = t-1, t-2, ...
    for (int s=t-1; s>=start; --s){
        Eigen::MatrixXd Xs = build_feature_matrix_at(dl, s, px);       // (N×P)
        Eigen::VectorXd ys = target_return_between(dl, s, target);     // (N)

        // 當日共用一個時間權重
        accumulate_gram_and_xTy(Xs, ys, ww, G, g);
        ww *= lam;
    }

    // 解 β
    if (tr.model == ModelType::Ridge) {
        Eigen::MatrixXd A = G;
        for (int j=0;j<P;++j) A(j,j) += tr.lambda;
        out.beta = A.ldlt().solve(g);
    } else if (tr.model == ModelType::Lasso) {
        // 先把 ridge 當 warm start
        Eigen::MatrixXd A = G;
        for (int j=0;j<P;++j) A(j,j) += 1e-8;
        Eigen::VectorXd beta0 = A.ldlt().solve(g);
        out.beta = lasso_coordinate_descent(G, g, tr.lambda, tr.lasso_max_iter, tr.lasso_tol);
        // 若想用 warm start，可將 beta0 做為初值加到 CD 中（此處採簡潔版）
        (void)beta0;
    } else { // GBDT 介面：目前以 linear 當 placeholder
        Eigen::MatrixXd A = G;
        for (int j=0;j<P;++j) A(j,j) += tr.lambda;
        out.beta = A.ldlt().solve(g);
        // TODO: 若要接 xgboost/lightgbm，這裡回傳外部預測
    }

    // （2）在 t 當天做預測：rhat_raw = X_t β
    Eigen::MatrixXd Xt = build_feature_matrix_at(dl, t, px);
    out.rhat_raw = Xt * out.beta;

    // （3）3.3 量化校準（ECDF→Probit 對齊 q50/q75）
    out.rhat_cal = quantile_calibrate(out.rhat_raw, cal);

    // （4）增強（BF/BIAS/TS）
    out.rhat_enh = enhance_with_flows_bias(dl, t, px, enh, out.rhat_cal);

    return out;
}

