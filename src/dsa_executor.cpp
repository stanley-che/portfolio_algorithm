#include "dsa_executor.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

using Eigen::VectorXd; using Eigen::VectorXi; using Eigen::MatrixXd;
using std::vector; using std::pair;

namespace dsa {

static inline double sqr(double x){ return x*x; }
static inline double clamp(double x,double lo,double hi){ return std::max(lo, std::min(hi,x)); }
static inline double safe_div(double a,double b){ return (std::abs(b)<1e-12)?0.0:(a/b); }

// =============== ctor & init ===============
Executor::Executor(const Config& cfg)
: C_(cfg), rng_(cfg.seed) {
    Sigma_ = C_.Sigma_tilde;
    N_ = (int)Sigma_.rows();
    if (Sigma_.cols()!=N_) throw std::runtime_error("Sigma_tilde must be square N×N");
    if (C_.price.size()!=N_ || C_.lot.size()!=N_ || C_.n0.size()!=N_ || C_.r_enh.size()!=N_)
        throw std::runtime_error("Dimension mismatch in inputs");

    unit_w_lot_   = C_.price.array() * C_.lot.cast<double>().array();
    unit_w_lot_  /= C_.P0;
    unit_w_share_ = C_.price.array() / C_.P0;

    // baseline holding weights
    w_hold_ = (C_.n0.cast<double>().array() * C_.lot.cast<double>().array() * C_.price.array()) / C_.P0;

    // impact coef K_i = γ σ (1+β|Imb|) * P0^{3/2} / sqrt(ADV)
    K_imp_ = VectorXd::Zero(N_);
    for (int i=0;i<N_;++i) {
        double scale = std::sqrt( (C_.P0*C_.P0*C_.P0) / std::max(1e-12, C_.ADV.size()==N_? C_.ADV(i):1.0) );
        double imb   = C_.Imb.size()==N_? std::abs(C_.Imb(i)):0.0;
        double gk    = C_.sigma_GK.size()==N_? C_.sigma_GK(i):1.0;
        K_imp_(i) = C_.gamma_init * gk * (1.0 + C_.beta_imb_impact*imb) * scale;
    }

    build_cholesky();
}

void Executor::build_cholesky(){
    if (C_.L.size()==0){
        Eigen::LLT<MatrixXd> llt(Sigma_ + 1e-12*MatrixXd::Identity(N_,N_));
        if (llt.info()!=Eigen::Success) throw std::runtime_error("Cholesky failed");
        L_ = llt.matrixL();
    } else {
        L_ = C_.L;
    }
}

// round-to-lot helper
static VectorXi round_to_lot_from_w(const VectorXd& w, const VectorXi& lot,
                                    const VectorXd& price, double P0){
    const int N = (int)w.size();
    VectorXi n(N);
    for (int i=0;i<N;++i){
        double shares = w(i) * P0 / std::max(1e-12, price(i));
        n(i) = (int)std::llround(shares / std::max(1, lot(i)));
    }
    return n.cwiseMax(0);
}

void Executor::initialize(const std::optional<VectorXd>& w_seed){
    // start from seed weights if provided, else from current holding n0
    if (w_seed.has_value()){
        n_ = round_to_lot_from_w(*w_seed, C_.lot, C_.price, C_.P0);
    } else {
        n_ = C_.n0;
    }
    w_ = (n_.cast<double>().array() * C_.lot.cast<double>().array() * C_.price.array()) / C_.P0;

    // hard cap to wmax / group caps / non-negativity
    if (C_.wmax.size()==N_) {
        for (int i=0;i<N_;++i) w_(i) = std::min(w_(i), std::max(0.0, C_.wmax(i)));
        n_ = round_to_lot_from_w(w_, C_.lot, C_.price, C_.P0);
        w_ = (n_.cast<double>().array() * C_.lot.cast<double>().array() * C_.price.array()) / C_.P0;
    }

    // caches
    q_       = Sigma_ * w_;
    s_sum_   = w_.sum();
    lambdaB_ = C_.lambdaB_start;

    dW_      = w_ - w_hold_;
    buy_pos_ = dW_.cwiseMax(0.0);
    sell_pos_= (-dW_).cwiseMax(0.0);
    abs_pos_ = dW_.cwiseAbs();

    // precompute costs
    lin_cost_ = C_.P0 * (C_.fee_buy * buy_pos_.sum() + C_.fee_sell * sell_pos_.sum());
    imp_cost_ = (K_imp_.array() * abs_pos_.array().pow(1.5)).sum();

    refresh_sampling_weights();
}

void Executor::quick_scale_to_risk(){
    if (!C_.pre_scale_to_risk) return;
    double risk = current_risk();
    const double cap = C_.m * C_.sigma_star;
    if (risk <= cap) return;
    // scale w -> ρ w
    double rho = clamp(C_.risk_quick_scale * cap / std::max(1e-12, risk), 0.0, 1.0);
    VectorXd w_scaled = rho * w_;
    n_ = round_to_lot_from_w(w_scaled, C_.lot, C_.price, C_.P0);
    w_ = (n_.cast<double>().array() * C_.lot.cast<double>().array() * C_.price.array()) / C_.P0;

    q_       = Sigma_ * w_;
    s_sum_   = w_.sum();

    dW_      = w_ - w_hold_;
    buy_pos_ = dW_.cwiseMax(0.0);
    sell_pos_= (-dW_).cwiseMax(0.0);
    abs_pos_ = dW_.cwiseAbs();
    lin_cost_= C_.P0 * (C_.fee_buy * buy_pos_.sum() + C_.fee_sell * sell_pos_.sum());
    imp_cost_= (K_imp_.array() * abs_pos_.array().pow(1.5)).sum();
}

// =============== priorities & moves ===============
void Executor::refresh_sampling_weights(){
    buy_score_  = VectorXd::Ones(N_);
    sell_score_ = VectorXd::Ones(N_);

    auto pos = [](double x){ return x>0?x:0.0; };
    for (int i=0;i<N_;++i){
        double ADV = C_.ADV.size()==N_? C_.ADV(i) : 1.0;
        double Imb = C_.Imb.size()==N_? C_.Imb(i) : 0.0;
        double BF  = C_.BF.size()==N_ ? C_.BF(i)  : 0.0;
        double BIAS= C_.BIAS.size()==N_? std::abs(C_.BIAS(i)) : 0.0;
        double TS  = C_.TurnoverShare.size()==N_? C_.TurnoverShare(i) : 0.0;

        // SellScore ∝ |BIAS| + η1[-BF]_+ + η2[-Imb]_+ + η3/√ADV + η4/(TS+ε)
        double sell = BIAS + 0.7*pos(-BF) + 0.5*pos(-Imb) + 0.2/sqrt(std::max(1e-8,ADV))
                      + 0.2 / (TS + 1e-6);
        // BuyScore ∝ [BF]_+ + ζ1(−|BIAS|)_+ + ζ2[Imb]_+ + ζ3√ADV + ζ4*TS
        double buy  = pos(BF) + 0.4*pos(0.10 - BIAS) + 0.5*pos(Imb)
                      + 0.2*sqrt(std::max(1e-8,ADV)) + 0.2*TS;
        buy_score_(i)  = std::max(1e-12, buy);
        sell_score_(i) = std::max(1e-12, sell);
    }
    // build pools (indices repeated proportionally → roulette wheel)
    buy_pool_.clear();  sell_pool_.clear();
    std::discrete_distribution<int> db(buy_score_.data(), buy_score_.data()+N_);
    std::discrete_distribution<int> ds(sell_score_.data(), sell_score_.data()+N_);
    for (int k=0;k<5*N_;++k){ buy_pool_.push_back(db(rng_)); sell_pool_.push_back(ds(rng_)); }

    // swap candidates (pre-build ~N pairs)
    swap_pairs_.clear();
    std::uniform_int_distribution<int> uni(0, N_-1);
    for (int k=0;k<N_;++k) swap_pairs_.push_back({sell_pool_[k% sell_pool_.size()], buy_pool_[k% buy_pool_.size()]});
}

void Executor::propose_buy_lot(int& i, VectorXd& dw){
    std::uniform_int_distribution<int> pick(0, (int)buy_pool_.size()-1);
    i = buy_pool_[pick(rng_)];
    dw.setZero(N_);
    dw(i) = unit_w_lot_(i); // +1 lot
}
void Executor::propose_sell_lot(int& i, VectorXd& dw){
    std::uniform_int_distribution<int> pick(0, (int)sell_pool_.size()-1);
    i = sell_pool_[pick(rng_)];
    dw.setZero(N_);
    dw(i) = -unit_w_lot_(i); // -1 lot
}
void Executor::propose_swap_pair(int& i, int& j, VectorXd& dw){
    std::uniform_int_distribution<int> pick(0, (int)swap_pairs_.size()-1);
    auto pr = swap_pairs_[pick(rng_)];
    i = pr.first; j = pr.second;
    dw.setZero(N_);
    dw(i) = -unit_w_lot_(i);
    dw(j) = +unit_w_lot_(j);
}
bool Executor::propose_odd_lot(int& i, VectorXd& dw){
    if (!C_.enable_odd_lot_phase) return false;
    std::uniform_int_distribution<int> uni(0, N_-1);
    i = uni(rng_);
    dw.setZero(N_);
    // ±1 share
    std::bernoulli_distribution coin(0.5);
    dw(i) = (coin(rng_) ? +1.0 : -1.0) * unit_w_share_(i);
    return true;
}

// =============== hard checks ===============
bool Executor::hard_violations_after_delta(const std::vector<int>& idxs,
                                           const VectorXd& dw) const
{
    // non-neg & wmax
    for (int k=0;k<(int)idxs.size();++k){
        int i = idxs[k];
        double wi_new = w_(i) + dw(i);
        if (wi_new < -1e-12) return true;
        if (C_.wmax.size()==N_ && wi_new > C_.wmax(i) + 1e-12) return true;
        if (C_.reduce_only && dw(i) > 1e-14) return true; // disable buys
    }
    // group caps
    if (!C_.group_cap.empty() && C_.group_id.size()==N_){
        std::map<int,double> gsum;
        for (int i=0;i<N_;++i){
            gsum[C_.group_id(i)] += w_(i);
        }
        for (int k=0;k<(int)idxs.size();++k){
            int i = idxs[k];
            gsum[C_.group_id(i)] += dw(i);
        }
        for (auto& kv: C_.group_cap){
            if (gsum[kv.first] > kv.second + 1e-12) return true;
        }
    }
    // risk cap using cached q: risk^2_new = (w+dw)^T Σ (w+dw) = w^TΣw + 2 dw^T q + dw^T Σ dw
    double risk2 = (L_.transpose()*w_).squaredNorm();
    VectorXd sdelta = Sigma_ * dw;
    risk2 += 2.0 * dw.dot(q_) + dw.dot(sdelta);
    double risk_new = std::sqrt(std::max(0.0, risk2));
    if (risk_new > C_.m * C_.sigma_star + 1e-10) return true;

    return false;
}

// =============== ΔC & accept ===============
double Executor::delta_objective(const std::vector<int>& idxs,
                                 const VectorXd& dw) const
{
    // Δforecast
    double d_forecast = C_.r_enh.dot(dw);

    // Δrisk in objective (soft quadratic penalty)
    double d_risk = 0.0;
    if (C_.lambda_risk > 0.0){
        VectorXd Sd = Sigma_ * dw;
        d_risk = -0.5 * C_.lambda_risk * (2.0 * dw.dot(q_) + dw.dot(Sd));
    }

    // Δbudget penalty
    double ds = 0.0; for (int k=0;k<(int)idxs.size();++k) ds += dw(idxs[k]);
    double d_budget = -C_.lambdaB_start * ( ( (s_sum_ + ds)*(s_sum_ + ds) - s_sum_*s_sum_ ) );

    // Δ costs vs baseline (only changed names)
    double d_lin = 0.0, d_imp = 0.0;
    for (int k=0;k<(int)idxs.size();++k){
        int i = idxs[k];
        double old = dW_(i);
        double neu = old + dw(i);

        double old_buy  = std::max(old, 0.0);
        double old_sell = std::max(-old, 0.0);
        double new_buy  = std::max(neu, 0.0);
        double new_sell = std::max(-neu, 0.0);

        d_lin += C_.P0 * ( C_.fee_buy*(new_buy-old_buy) + C_.fee_sell*(new_sell-old_sell) );
        d_imp += K_imp_(i) * ( std::pow(std::abs(neu),1.5) - std::pow(std::abs(old),1.5) );
    }

    return d_forecast + d_risk - d_lin - d_imp + d_budget;
}

void Executor::accept_move(const std::vector<int>& idxs,
                           const VectorXd& dw, double /*dC*/)
{
    // update state
    for (int k=0;k<(int)idxs.size();++k){
        int i = idxs[k];
        // update lots (round exactly by what we applied)
        // For lot moves: dw(i) equals ±unit_w_lot -> exact int update; for odd-lot we don't touch lots.
        if (std::abs(dw(i) - unit_w_lot_(i)) < 1e-12) n_(i) += 1;
        else if (std::abs(dw(i) + unit_w_lot_(i)) < 1e-12) n_(i) -= 1;
    }
    w_ += dw;
    q_ += Sigma_ * dw;
    s_sum_ += dw.sum();

    // update cost caches vs baseline
    for (int k=0;k<(int)idxs.size();++k){
        int i = idxs[k];
        double old = dW_(i);
        double neu = old + dw(i);

        double old_buy  = std::max(old, 0.0);
        double old_sell = std::max(-old, 0.0);
        double new_buy  = std::max(neu, 0.0);
        double new_sell = std::max(-neu, 0.0);

        lin_cost_ += C_.P0 * ( C_.fee_buy*(new_buy-old_buy) + C_.fee_sell*(new_sell-old_sell) );
        imp_cost_ += K_imp_(i) * ( std::pow(std::abs(neu),1.5) - std::pow(std::abs(old),1.5) );

        dW_(i)      = neu;
        buy_pos_(i) = new_buy;
        sell_pos_(i)= new_sell;
        abs_pos_(i) = std::abs(neu);
    }
}

// =============== totals ===============
double Executor::compute_objective_full() const {
    double ret = C_.r_enh.dot(w_);
    double risk_pen = (C_.lambda_risk>0.0)? -0.5*C_.lambda_risk*(w_.dot(Sigma_*w_)) : 0.0;
    double budg = -lambdaB_ * sqr(w_.sum()-1.0);
    double costs = lin_cost_ + imp_cost_;
    return ret + risk_pen - costs + budg;
}
double Executor::current_risk() const {
    return (L_.transpose()*w_).norm();
}

// =============== main run() ===============
Result Executor::run(const std::optional<VectorXd>& w_seed){
    initialize(w_seed);
    quick_scale_to_risk();

    Result best;
    best.n_best = n_;
    best.w_best = w_;
    best.C_best = compute_objective_full();
    best.risk_best = current_risk();
    best.budget_violation = std::abs(w_.sum()-1.0);

    // ---- warmup for T0: sample 200 random proposals to estimate median |ΔC|
    vector<double> warm;
    warm.reserve(200);
    VectorXd dw = VectorXd::Zero(N_);
    std::uniform_real_distribution<double> u01(0.0,1.0);
    for (int k=0;k<200;++k){
        int i=-1,j=-1;
        dw.setZero();
        double r = u01(rng_);
        if (r < 0.4)      propose_buy_lot(i, dw);
        else if (r < 0.8) propose_sell_lot(i, dw);
        else              propose_swap_pair(i,j,dw);
        std::vector<int> idxs; if (i>=0) idxs.push_back(i); if (j>=0) idxs.push_back(j);
        if (hard_violations_after_delta(idxs, dw)) continue;
        warm.push_back(std::abs(delta_objective(idxs, dw)));
    }
    std::nth_element(warm.begin(), warm.begin()+warm.size()/2, warm.end());
    double T = C_.initT_eta * (warm.empty()? 1e-3 : warm[warm.size()/2]);

    // ---- annealing loop ----
    int no_gain_T = 0;
    for (int tstep=0; tstep<C_.max_T_steps && T > C_.T_floor; ++tstep){
        int accepted_this_T = 0;
        for (int m=0; m<C_.moves_per_T; ++m){
            best.iters++;
            int i=-1,j=-1;
            dw.setZero();
            double r = u01(rng_);
            if (C_.enable_odd_lot_phase && T < C_.odd_lot_start_T && u01(rng_) < 0.2){
                if (!propose_odd_lot(i, dw)) continue;
            } else if (r < 0.4) {
                propose_buy_lot(i, dw);
            } else if (r < 0.8) {
                propose_sell_lot(i, dw);
            } else {
                propose_swap_pair(i, j, dw);
            }
            std::vector<int> idxs; if (i>=0) idxs.push_back(i); if (j>=0) idxs.push_back(j);

            if (hard_violations_after_delta(idxs, dw)) continue;

            double dC = delta_objective(idxs, dw);
            double acc_p = std::min(1.0, std::exp(dC / std::max(1e-12, T)));
            if (u01(rng_) < acc_p){
                accept_move(idxs, dw, dC);
                accepted_this_T++; best.accepts++;

                double Cnow = compute_objective_full();
                if (Cnow > best.C_best + 1e-12){
                    best.C_best = Cnow;
                    best.n_best = n_;
                    best.w_best = w_;
                    best.risk_best = current_risk();
                    best.budget_violation = std::abs(w_.sum()-1.0);
                }
            }
        }
        // cooling
        T *= C_.cool;
        // ramp λ_B
        lambdaB_ = std::min(C_.lambdaB_max, std::max(lambdaB_, C_.lambdaB_start * std::pow(1.2, tstep)));

        // stagnation / reheat
        if (accepted_this_T==0) no_gain_T++; else no_gain_T=0;
        if (no_gain_T >= C_.stagnate_reheat_T){
            T = std::max(T, 0.5 * C_.initT_eta); // small reheat
            no_gain_T = 0;
        }
    }
    return best;
}

} // namespace dsa

