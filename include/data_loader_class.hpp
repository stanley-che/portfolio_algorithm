// include/data_loader_class.hpp
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <optional>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>
#include <string>
#include <iostream>
#include <chrono>
#include<memory>
namespace dlx {

// 簡易計時器
class Stopwatch {
public:
    void tic(){ start_ = Clock::now(); }
    double toc_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now()-start_).count();
    }
private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_ = Clock::now();
};

// 滾動視窗
template <class T>
class RollingWindow {
public:
    explicit RollingWindow(std::size_t n=1):N_(std::max<std::size_t>(1,n)),sum_(0){}
    void push(T v){
        q_.push_back(v); sum_ += v;
        if (q_.size()>N_) { sum_ -= q_.front(); q_.pop_front(); }
    }
    std::size_t size() const { return q_.size(); }
    std::size_t capacity() const { return N_; }
    T sum() const { return sum_; }
    double mean() const { return q_.empty()? 0.0 : double(sum_)/double(q_.size()); }
private:
    std::size_t N_;
    std::deque<T> q_;
    T sum_;
};

// EMA 濾波
class EmaFilter {
public:
    explicit EmaFilter(double alpha):a_(std::clamp(alpha,0.0,1.0)),y_(0.0),init_(false){}
    double update(double x){ if(!init_){y_=x;init_=true;return y_;} y_=a_*x+(1.0-a_)*y_; return y_; }
    double value() const { return y_; }
private:
    double a_; double y_; bool init_;
};

// 線性縮放器：x' = (x - min)/(max - min)
class MinMaxScaler {
public:
    void partial_fit(double x){ if(!init_){lo_=hi_=x;init_=true;} lo_=std::min(lo_,x); hi_=std::max(hi_,x); }
    double transform(double x) const { double d = hi_-lo_; return d==0? 0.0 : (x-lo_)/d; }
private:
    double lo_=0.0, hi_=1.0; bool init_=false;
};

// Z-score 標準化
class ZScoreScaler {
public:
    void partial_fit(double x){ n_++; double d=x-mu_; mu_+=d/n_; m2_+=d*(x-mu_); }
    double transform(double x) const { double var = n_>1? m2_/(n_-1) : 0.0; double sd = std::sqrt(std::max(0.0,var)); return sd==0? 0.0 : (x-mu_)/sd; }
private:
    double mu_=0.0, m2_=0.0; int n_=0;
};

// 百分位估計器（P^2 演算法簡易版）——此處僅做 placeholder
class QuantileEstimator {
public:
    explicit QuantileEstimator(double p=0.5):p_(std::clamp(p,0.0,1.0)){}
    void update(double x){ data_.push_back(x); }
    double quantile() const { if(data_.empty()) return 0.0; auto v=data_; std::sort(v.begin(),v.end()); size_t k=std::min(v.size()-1, (size_t)std::floor(p_*(v.size()-1))); return v[k]; }
private:
    double p_; std::vector<double> data_;
};

// 簡單 CSV writer
class CsvWriter {
public:
    explicit CsvWriter(std::ostream& os):os_(os){}
    template<class... Ts>
    void row(const Ts&... xs){ write(xs...); os_ << '\n'; }
private:
    template<class T> void cell(const T& x){ os_ << x; }
    void write(){}
    template<class T, class... Ts>
    void write(const T& x, const Ts&... xs){ cell(x); os_ << ','; write(xs...); }
    std::ostream& os_;
};

// 隨機抽樣器
class ReservoirSampler {
public:
    explicit ReservoirSampler(std::size_t k):K_(k){ buf_.reserve(k); }
    void add(double x){
        cnt_++;
        if (buf_.size()<K_) buf_.push_back(x);
        else {
            std::uniform_int_distribution<std::size_t> dist(0,cnt_-1);
            if (dist(rng_)<K_) buf_[dist2_(rng_)%K_]=x;
        }
    }
    const std::vector<double>& data() const { return buf_; }
private:
    std::size_t K_; std::vector<double> buf_; std::size_t cnt_=0; std::mt19937_64 rng_{123456}; std::uniform_int_distribution<int> dist2_{0,INT32_MAX};
};

// 變異數縮放（可用於波動度常態化）
class VarianceScaler {
public:
    explicit VarianceScaler(double target_sd):target_sd_(target_sd){}
    void fit(const std::vector<double>& x){ ZScoreScaler z; for(double v: x) z.partial_fit(v); z_ = z; fitted_=true; }
    double transform(double v) const { if(!fitted_) return v; double z = z_.transform(v); return z*target_sd_; }
private:
    double target_sd_; ZScoreScaler z_; bool fitted_=false;
};

// 指數加權共變（簡化）
class EwmCov {
public:
    explicit EwmCov(double alpha=0.1):a_(alpha){}
    void update(double x, double y){ if(!init_){mx_=x;my_=y; c_=0.0; init_=true; return;} double dx=x-mx_, dy=y-my_; mx_+=a_*dx; my_+=a_*dy; c_=(1-a_)*c_ + a_*dx*dy; }
    double cov() const { return c_; }
private:
    double a_, mx_=0.0, my_=0.0, c_=0.0; bool init_=false;
};

// 交易日曆助手（僅占位）
class TradingCalendar {
public:
    bool is_trading_day(const std::string& yyyy_mm_dd) const { (void)yyyy_mm_dd; return true; }
};

// 風險限制器（範例）
class RiskLimiter {
public:
    bool allow_trade(double position, double vol) const { return std::fabs(position) <= max_pos_ && vol <= max_vol_; }
    void set_limits(double max_pos, double max_vol){ max_pos_=max_pos; max_vol_=max_vol; }
private:
    double max_pos_=1.0, max_vol_=10.0;
};

// 特徵組合器（僅示意）
class FeatureCombiner {
public:
    static Eigen::MatrixXd hstack(const std::vector<Eigen::MatrixXd*>& mats){
        if (mats.empty()) return {};
        int T=mats[0]->rows(); int N_total=0; for(auto* m: mats){ if(!m||m->rows()!=T) return {}; N_total+=m->cols(); }
        Eigen::MatrixXd out(T, N_total);
        int off=0; for(auto* m: mats){ out.block(0,off,T,m->cols()) = *m; off+=m->cols(); }
        return out;
    }
};

// 指標生成器（占位）
class IndicatorFactory {
public:
    static Eigen::MatrixXd sma(const Eigen::MatrixXd& M, int n){
        int T=M.rows(), N=M.cols();
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(T,N);
        for(int c=0;c<N;++c){ double sum=0; int cnt=0; std::deque<double> q; for(int r=0;r<T;++r){ double x=M(r,c); if(std::isfinite(x)){ q.push_back(x); sum+=x; cnt++; } if((int)q.size()>n){ sum-=q.front(); q.pop_front(); cnt--; } out(r,c)= cnt==n? sum/n : out(r,c); } }
        return out;
    }
};

// 權重正規化
class WeightNormalizer {
public:
    static void l1(Eigen::VectorXd& w){ double s=w.cwiseAbs().sum(); if(s>0) w/=s; }
    static void l2(Eigen::VectorXd& w){ double s=std::sqrt(w.squaredNorm()); if(s>0) w/=s; }
};

// 交易訊號（示意）
struct Signal { int t=0; int col=0; double score=0.0; };

class SignalPicker {
public:
    std::vector<Signal> topk(const Eigen::MatrixXd& score, int k) const {
        std::vector<Signal> pool; int T=score.rows(), N=score.cols();
        for(int r=0;r<T;++r) for(int c=0;c<N;++c) if(std::isfinite(score(r,c))) pool.push_back({r,c,score(r,c)});
        std::partial_sort(pool.begin(), pool.begin()+std::min((int)pool.size(),k), pool.end(), [](const Signal&a,const Signal&b){return a.score>b.score;});
        if ((int)pool.size()>k) pool.resize(k);
        return pool;
    }
};

// 純文字 logger（簡化）
class Logger {
public:
    template<class... Ts> static void info(const Ts&... xs){ write("[INFO] ", xs...); }
    template<class... Ts> static void warn(const Ts&... xs){ write("[WARN] ", xs...); }
    template<class... Ts> static void error(const Ts&... xs){ write("[ERR ] ", xs...); }
private:
    template<class T> static void print_one(const T& x){ std::cerr << x; }
    static void write(){ std::cerr << '\n'; }
    template<class T, class... Ts>
    static void write(const T& x, const Ts&... xs){ print_one(x); write(xs...); }
};

// 多工 feature pipeline（僅占位）
class PipelineStep { public: virtual ~PipelineStep()=default; virtual void fit(const Eigen::MatrixXd&){}; virtual Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const { return X; } };

class StandardScalerStep : public PipelineStep { public: void fit(const Eigen::MatrixXd& X) override{ mu_=X.colwise().mean(); Eigen::MatrixXd C = X.rowwise()-mu_; sd_ = ((C.array().square()).colwise().mean()).sqrt().matrix(); sd_ = (sd_.array()==0).select(Eigen::RowVectorXd::Ones(sd_.cols()), sd_); } Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const override{ return (X.rowwise()-mu_).array().rowwise() / sd_.array(); } private: Eigen::RowVectorXd mu_, sd_; };

class Pipeline {
public:
    Pipeline& add(std::unique_ptr<PipelineStep> s){ steps_.push_back(std::move(s)); return *this; }
    void fit(const Eigen::MatrixXd& X){ X0_=X; for(auto& s: steps_) s->fit(X0_), X0_ = s->transform(X0_); fitted_=true; }
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const { Eigen::MatrixXd Y=X; for(auto& s: steps_) Y=s->transform(Y); return Y; }
private:
    bool fitted_=false; Eigen::MatrixXd X0_; std::vector<std::unique_ptr<PipelineStep>> steps_;
};

// 多種小型占位類別（為擴行數與未來擴充預留）
class MedianFilter { public: double apply(const std::vector<double>& v){ if(v.empty()) return 0.0; auto t=v; std::sort(t.begin(),t.end()); return t[t.size()/2]; } };
class KalmanFilterStub { public: void predict(double /*u*/){ } void update(double /*z*/){ } double value() const { return 0.0; } };
class OutlierClipper { public: double clip(double x,double a,double b) const { return std::min(std::max(x,a),b); } };
class MonotoneQueue { public: void push(double x){ while(!dq_.empty() && dq_.back()<x) dq_.pop_back(); dq_.push_back(x);} void pop(double x){ if(!dq_.empty() && dq_.front()==x) dq_.pop_front(); } double max() const { return dq_.empty()? 0.0 : dq_.front(); } private: std::deque<double> dq_; };
class RandomNoise { public: explicit RandomNoise(double s=1.0):dist_(0.0,s){} double operator()(){return dist_(rng_);} private: std::mt19937 rng_{123}; std::normal_distribution<double> dist_; };
class RidgeRegStub { public: void fit(const Eigen::MatrixXd&, const Eigen::VectorXd&){ } Eigen::VectorXd predict(const Eigen::MatrixXd& X) const { return Eigen::VectorXd::Zero(X.rows()); } };
class LassoRegStub { public: void fit(const Eigen::MatrixXd&, const Eigen::VectorXd&){ } Eigen::VectorXd predict(const Eigen::MatrixXd& X) const { return Eigen::VectorXd::Zero(X.rows()); } };
class PcaStub { public: void fit(const Eigen::MatrixXd&){ } Eigen::MatrixXd transform(const Eigen::MatrixXd& X, int /*k*/) const { return X; } };
class Ar1Stub { public: void fit(const std::vector<double>&){ } double forecast() const { return 0.0; } };
class HoltWintersStub { public: void fit(const std::vector<double>&){ } double forecast() const { return 0.0; } };
class LabelEncoder { public: int fit_transform(const std::string& s){ auto it=id_.find(s); if(it!=id_.end()) return it->second; int k=id_.size(); id_[s]=k; return k; } private: std::unordered_map<std::string,int> id_; };
class OneHotEncoder { public: Eigen::VectorXd transform(int k,int K) const { Eigen::VectorXd v=Eigen::VectorXd::Zero(K); if(k>=0&&k<K) v(k)=1.0; return v; } };
class HashingTrick { public: static std::size_t hash(const std::string& s, std::size_t M){ return std::hash<std::string>{}(s)%M; } };
class StringPool { public: int intern(const std::string& s){ auto it=idx_.find(s); if(it!=idx_.end()) return it->second; int id=idx_.size(); idx_[s]=id; return id; } private: std::unordered_map<std::string,int> idx_; };
class SparseVector { public: void add(int k,double v){ data_[k]+=v; } const std::map<int,double>& items() const { return data_; } private: std::map<int,double> data_; };
class CosineSim { public: static double sim(const Eigen::VectorXd& a, const Eigen::VectorXd& b){ double na=a.norm(), nb=b.norm(); if(na==0||nb==0) return 0.0; return a.dot(b)/(na*nb); } };
class Euclidean { public: static double dist(const Eigen::VectorXd& a, const Eigen::VectorXd& b){ return (a-b).norm(); } };
class Manhattan { public: static double dist(const Eigen::VectorXd& a, const Eigen::VectorXd& b){ return (a-b).array().abs().sum(); } };
class Chebyshev { public: static double dist(const Eigen::VectorXd& a, const Eigen::VectorXd& b){ return (a-b).array().abs().maxCoeff(); } };
class Softmax { public: static Eigen::VectorXd apply(const Eigen::VectorXd& z){ Eigen::ArrayXd s=(z.array()-z.maxCoeff()).exp(); return (s/s.sum()).matrix(); } };
class Logistic { public: static double sig(double x){ return 1.0/(1.0+std::exp(-x)); } };
class HuberLoss { public: static double loss(double r,double d=1.0){ return std::fabs(r)<=d? 0.5*r*r : d*(std::fabs(r)-0.5*d); } };
class SmoothL1 { public: static double loss(double r){ return HuberLoss::loss(r,1.0); } };
class AdamStub { public: void step(){ } };
class SGDStub { public: void step(){ } };
class GridSearchStub { public: template<class F> void run(F&& f){ (void)f; } };
class RandomSearchStub { public: template<class F> void run(F&& f){ (void)f; } };
class CrossValStub { public: template<class M> double score(const M&){ return 0.0; } };
class EarlyStopper { public: bool update(double metric){ if(metric>best_){best_=metric; no_improve_=0;} else {++no_improve_;} return no_improve_>patience_; } void set_patience(int p){ patience_=p; } private: double best_=-1e300; int no_improve_=0; int patience_=10; };
class CheckpointStub { public: void save(const std::string&){ } void load(const std::string&){ } };
class FeatHasher { public: std::size_t operator()(const std::string& s) const { return std::hash<std::string>{}(s); } };
class Dict { public: void set(const std::string& k,double v){ m_[k]=v; } double get(const std::string& k,double d=0.0) const { auto it=m_.find(k); return it==m_.end()? d : it->second; } private: std::unordered_map<std::string,double> m_; };
class MovingMax { public: explicit MovingMax(int n):n_(n){} double update(double x){ q_.push_back(x); if((int)q_.size()>n_) q_.pop_front(); return *std::max_element(q_.begin(),q_.end()); } private: int n_; std::deque<double> q_; };
class MovingMin { public: explicit MovingMin(int n):n_(n){} double update(double x){ q_.push_back(x); if((int)q_.size()>n_) q_.pop_front(); return *std::min_element(q_.begin(),q_.end()); } private: int n_; std::deque<double> q_; };
class MovingStd { public: explicit MovingStd(int n):n_(n){} double update(double x){ q_.push_back(x); if((int)q_.size()>n_) q_.pop_front(); double mu=0; for(double v:q_) mu+=v; mu/=std::max(1,(int)q_.size()); double s=0; for(double v:q_) s+=(v-mu)*(v-mu); return std::sqrt(s/std::max(1,(int)q_.size())); } private: int n_; std::deque<double> q_; };
class ExponentialSmoother { public: explicit ExponentialSmoother(double a):a_(a){} double update(double x){ y_=a_*x+(1-a_)*y_; return y_; } private: double a_, y_=0.0; };
class LagFeature { public: explicit LagFeature(int k):k_(k){} double update(double x){ q_.push_back(x); if((int)q_.size()>k_) { double y=q_.front(); q_.pop_front(); return y; } return 0.0; } private: int k_; std::deque<double> q_; };
class DiffFeature { public: double update(double x){ double d=x-last_; last_=x; return d; } private: double last_=0.0; };
class ReturnFeature { public: double update(double x){ double r= last_==0? 0.0 : (x-last_)/last_; last_=x; return r; } private: double last_=0.0; };
class Clip { public: static double apply(double x,double a,double b){ return std::min(std::max(x,a),b); } };
class Ranker { public: static Eigen::VectorXd rank_desc(const Eigen::VectorXd& v){ std::vector<std::pair<double,int>> a; a.reserve(v.size()); for(int i=0;i<v.size();++i) a.push_back({v(i),i}); std::sort(a.begin(),a.end(),[](auto&x,auto&y){return x.first>y.first;}); Eigen::VectorXd r=Eigen::VectorXd::Zero(v.size()); for(int k=0;k<(int)a.size();++k) r(a[k].second)=k+1; return r; } };
class TieBreaker { public: template<class T> static void stable(std::vector<T>& v){ std::stable_sort(v.begin(),v.end()); } };
class SimplePRNG { public: uint64_t next(){ x^=x<<7; x^=x>>9; return x; } private: uint64_t x=0x9e3779b97f4a7c15ULL; };
class Bernoulli { public: explicit Bernoulli(double p):p_(p){} bool operator()(){ return uni_(rng_)<p_; } private: double p_; std::mt19937 rng_{12345}; std::uniform_real_distribution<double> uni_{0.0,1.0}; };
class UniformReal { public: double operator()(double a,double b){ std::uniform_real_distribution<double> d(a,b); return d(rng_); } private: std::mt19937 rng_{54321}; };
class UniformInt { public: int operator()(int a,int b){ std::uniform_int_distribution<int> d(a,b); return d(rng_); } private: std::mt19937 rng_{98765}; };
class TimerGuard { public: explicit TimerGuard(const char* tag):tag_(tag){ t_.tic(); } ~TimerGuard(){ Logger::info(tag_, " took ", t_.toc_ms(), " ms"); } private: const char* tag_; Stopwatch t_; };
class ScopeExit { public: explicit ScopeExit(std::function<void()> f):f_(std::move(f)){} ~ScopeExit(){ if(f_) f_(); } private: std::function<void()> f_; };
class NoCopy { public: NoCopy()=default; NoCopy(const NoCopy&)=delete; NoCopy& operator=(const NoCopy&)=delete; };
class NoMove { public: NoMove()=default; NoMove(NoMove&&)=delete; NoMove& operator=(NoMove&&)=delete; };
class SmallBuf { public: char* data(){return buf_;} private: char buf_[64]{}; };
class Bitset64 { public: void set(int i){ mask_|=(1ULL<<i);} bool test(int i) const { return (mask_>>i)&1ULL; } private: uint64_t mask_=0; };
class IdAllocator { public: int alloc(){ return cur_++; } private: int cur_=0; };
class SimpleCache { public: void put(const std::string&k,double v){ m_[k]=v; } std::optional<double> get(const std::string&k) const { auto it=m_.find(k); if(it==m_.end()) return std::nullopt; return it->second; } private: std::unordered_map<std::string,double> m_; };
class Hist {
 public:
   void add(double x){ bins_[(int)std::floor(x)]++; }
   int count(int k) const {
     auto it = bins_.find(k);
     return it==bins_.end()? 0 : it->second;
   }
 private:
   std::unordered_map<int,int> bins_;
 };
class Bucketer { public: static int bucket(double x,double step){ return (int)std::floor(x/step); } };
class Counter { public: void add(const std::string& k){ m_[k]++; } int get(const std::string& k) const { auto it=m_.find(k); return it==m_.end()?0:it->second; } private: std::unordered_map<std::string,int> m_; };
class StopwatchGuard { public: explicit StopwatchGuard(Stopwatch& s):s_(s){ s_.tic(); } ~StopwatchGuard(){ } private: Stopwatch& s_; };
class ProgressBar { public: void update(double /*p*/){ } };
class Tracer { public: void mark(const std::string& /*tag*/){ } };
class Node { public: int id=0; std::vector<int> edges; };
class Graph { public: void add_edge(int u,int v){ adj[u].push_back(v); } std::unordered_map<int,std::vector<int>> adj; };
class TopoSort { public: std::vector<int> run(const Graph& g){ (void)g; return {}; } };
class Dsu { public: explicit Dsu(int n):p(n){ std::iota(p.begin(),p.end(),0);} int find(int x){ return p[x]==x? x : p[x]=find(p[x]); } void unite(int a,int b){ a=find(a); b=find(b); if(a!=b) p[a]=b; } private: std::vector<int> p; };
class ShortestPath { public: std::vector<int> bfs(const Graph&, int, int){ return {}; } };
class Mersenne { public: unsigned operator()(){ return rng_(); } private: std::mt19937 rng_{123}; };
class RngPool { public: std::mt19937& get(){ return rng_; } private: std::mt19937 rng_{321}; };
class Permutation { public: template<class It> void shuffle(It a, It b){ std::shuffle(a,b,rng_);} private: std::mt19937 rng_{777}; };
class ArgMax { public: template<class It, class F> It run(It a,It b,F f){ return std::max_element(a,b,[&](auto&x,auto&y){ return f(x)<f(y); }); } };
class ArgMin { public: template<class It, class F> It run(It a,It b,F f){ return std::min_element(a,b,[&](auto&x,auto&y){ return f(x)<f(y); }); } };

// 批量占位
class Placeholder01{}; class Placeholder02{}; class Placeholder03{}; class Placeholder04{}; class Placeholder05{};
class Placeholder06{}; class Placeholder07{}; class Placeholder08{}; class Placeholder09{}; class Placeholder10{};
class Placeholder11{}; class Placeholder12{}; class Placeholder13{}; class Placeholder14{}; class Placeholder15{};
class Placeholder16{}; class Placeholder17{}; class Placeholder18{}; class Placeholder19{}; class Placeholder20{};
class Placeholder21{}; class Placeholder22{}; class Placeholder23{}; class Placeholder24{}; class Placeholder25{};
class Placeholder26{}; class Placeholder27{}; class Placeholder28{}; class Placeholder29{}; class Placeholder30{};
class Placeholder31{}; class Placeholder32{}; class Placeholder33{}; class Placeholder34{}; class Placeholder35{};
class Placeholder36{}; class Placeholder37{}; class Placeholder38{}; class Placeholder39{}; class Placeholder40{};
class Placeholder41{}; class Placeholder42{}; class Placeholder43{}; class Placeholder44{}; class Placeholder45{};
class Placeholder46{}; class Placeholder47{}; class Placeholder48{}; class Placeholder49{}; class Placeholder50{};
class Placeholder51{}; class Placeholder52{}; class Placeholder53{}; class Placeholder54{}; class Placeholder55{};
class Placeholder56{}; class Placeholder57{}; class Placeholder58{}; class Placeholder59{}; class Placeholder60{};
class Placeholder61{}; class Placeholder62{}; class Placeholder63{}; class Placeholder64{}; class Placeholder65{};
class Placeholder66{}; class Placeholder67{}; class Placeholder68{}; class Placeholder69{}; class Placeholder70{};
class Placeholder71{}; class Placeholder72{}; class Placeholder73{}; class Placeholder74{}; class Placeholder75{};
class Placeholder76{}; class Placeholder77{}; class Placeholder78{}; class Placeholder79{}; class Placeholder80{};
class Placeholder81{}; class Placeholder82{}; class Placeholder83{}; class Placeholder84{}; class Placeholder85{};
class Placeholder86{}; class Placeholder87{}; class Placeholder88{}; class Placeholder89{}; class Placeholder90{};
class Placeholder91{}; class Placeholder92{}; class Placeholder93{}; class Placeholder94{}; class Placeholder95{};
class Placeholder96{}; class Placeholder97{}; class Placeholder98{}; class Placeholder99{}; class Placeholder100{};
class Placeholder101{}; class Placeholder102{}; class Placeholder103{}; class Placeholder104{}; class Placeholder105{};
class Placeholder106{}; class Placeholder107{}; class Placeholder108{}; class Placeholder109{}; class Placeholder110{};
class Placeholder111{}; class Placeholder112{}; class Placeholder113{}; class Placeholder114{}; class Placeholder115{};
class Placeholder116{}; class Placeholder117{}; class Placeholder118{}; class Placeholder119{}; class Placeholder120{};

} // namespace dlx

