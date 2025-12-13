// data_loader.hpp
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace dlx {

// ----------------------- 基本結構 -----------------------
struct StockInfo {
    std::string symbol;
    int lot = 1;
    int odd_lot_unit = 1;
    int industry = -1;
    int group = -1;
    int tradable_flag = 1; // 1=可交易, 0=不可交易
};

// 清理與特徵設定
struct CleanFeatureConfig {
    // 清理
    bool adjust_ohlc_by_adjclose = true;
    bool forward_fill = true;
    bool back_fill = true;
    bool auto_non_tradable = true;
    double winsorize_p = 0.01;

    // BIAS 與 Broker Strength 平滑
    int bias_ma_n = 10;
    bool bias_use_ema = false;
    double bias_ema_half_life = 10.0;

    int broker_strength_ema = 3;
};

class DataLoader {
public:
    explicit DataLoader(std::string basePath);
    bool loadDailyPanelAndBuildFeatures();
    void adjustOHLCbyAdjCloseIfAny();
    // getters...
    CleanFeatureConfig& config() { return cfg_; }

private:
    std::string base_path_;
    CleanFeatureConfig cfg_;

    // maps
    std::vector<std::string> dates_, symbols_;
    std::unordered_map<std::string,int> date2row_, sym2col_;

    // raw fields + features + stock_info
    Eigen::MatrixXd O_, H_, L_, C_, V_, VAL_, inside_vol_, outside_vol_;
    std::vector<StockInfo> stock_info_;

    Eigen::MatrixXd feat_gap_, feat_mom5_, feat_mom10_, feat_mom20_, feat_gkvol_;
    Eigen::MatrixXd feat_liquidity_, feat_turnover_share_;
    Eigen::MatrixXd feat_imb_, feat_bias_, feat_broker_strength_;

    // IO
    void ensureShapes();
    bool readDailyPricesCsv(const std::string& filename);
    bool readBrokersCsv(const std::string& filename);
    bool readMetaCsv(const std::string& filename);

    // cleaning/features
    void imputeMissing();
    void markNonTradableByVolVal();
    static void winsorize_vec(Eigen::VectorXd& v, double p);
    void winsorizeAll();
 
    void buildPriceMomentumGK();
    void buildLiquidityTurnoverShare();
    void buildImbalance();
    void buildBIASandBrokerStrength();

    // util
    static void strip_bom(std::string& s);
    static std::string lower(std::string s);
    static std::string trim(const std::string& s);
    static std::vector<std::string> splitCsvLine(const std::string& line);
    static std::string normalize_date(const std::string& raw);
    static bool is_digit_sign_dot(char c);
    static double parse_num_clean(const std::string& raw);
    static long long parse_ll_clean(const std::string& raw);

    static inline double safe_div(double a, double b){ return (b==0.0? 0.0 : a/b); }
    static inline double ema_alpha_from_half_life(double hl){
        if (hl <= 0.0) return 1.0;
        return 1.0 - std::pow(0.5, 1.0/std::max(1.0, hl));
    }
};

} // namespace dlx
