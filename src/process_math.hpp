//process_math.hpp
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace process::detail {

inline double safe_div(double a,double b){ return (std::abs(b)<1e-12)?0.0:(a/b); }
inline double alpha_from_half_life(double hl){
    return 1.0 - std::exp(std::log(0.5)/std::max(1.0,hl));
}

inline Eigen::VectorXd cs_zscore(const Eigen::VectorXd& v, double clip){
    Eigen::VectorXd out=v;
    std::vector<double> a; a.reserve(v.size());
    for(int i=0;i<v.size();++i) if(std::isfinite(v(i))) a.push_back(v(i));
    if(a.size()<3){ out.setZero(); return out; }
    double mean = std::accumulate(a.begin(),a.end(),0.0)/a.size();
    double var=0; for(double x:a) var+=(x-mean)*(x-mean);
    var/=std::max(1.0,(double)a.size()-1.0);
    double sd = std::sqrt(std::max(1e-12,var));
    for(int i=0;i<v.size();++i){
        double z = std::isfinite(v(i)) ? (v(i)-mean)/sd : 0.0;
        z = std::clamp(z, -clip, clip);
        out(i)=z;
    }
    return out;
}

inline Eigen::VectorXd cs_quantile_map(const Eigen::VectorXd& v, double clip){
    std::vector<std::pair<double,int>> a;
    a.reserve(v.size());
    for(int i=0;i<v.size();++i) a.push_back({std::isfinite(v(i))?v(i):0.0, i});
    std::sort(a.begin(), a.end(), [](auto&x,auto&y){return x.first<y.first;});
    Eigen::VectorXd out(v.size());
    for(int r=0;r<(int)a.size();++r){
        double q = safe_div(r, std::max(1,(int)a.size()-1));
        out(a[r].second) = -clip + (2*clip)*q;
    }
    return out;
}

inline Eigen::VectorXd row_at(const Eigen::MatrixXd& M, int t){
    if (t<0 || t>=M.rows()) return Eigen::VectorXd::Zero(M.cols());
    return M.row(t).transpose().eval();
}

inline Eigen::VectorXd rolling_mean_last(const Eigen::MatrixXd& M, int t, int win){
    win = std::min(win, t+1);
    Eigen::VectorXd m = Eigen::VectorXd::Zero(M.cols());
    for(int c=0;c<M.cols();++c){
        double s=0; int n=0;
        for(int r=t-win+1; r<=t; ++r){
            double x=M(r,c);
            if(std::isfinite(x)){ s+=x; ++n; }
        }
        m(c) = (n>0)? s/n : 0.0;
    }
    return m;
}

inline Eigen::VectorXd rolling_std_last(const Eigen::MatrixXd& M, int t, int win){
    win = std::min(win, t+1);
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

inline double median_vec(const Eigen::VectorXd& v){
    std::vector<double> a; a.reserve(v.size());
    for(int i=0;i<v.size();++i) if(std::isfinite(v(i))) a.push_back(v(i));
    if(a.empty()) return 0.0;
    std::sort(a.begin(),a.end());
    size_t k=a.size()/2;
    return (a.size()%2)? a[k] : 0.5*(a[k-1]+a[k]);
}

inline double fin(double x){ return std::isfinite(x)?x:0.0; }

} // namespace process::detail
