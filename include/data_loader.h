#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <Eigen/Dense>

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
    bool adjust_ohlc_by_adjclose = true; // 若有 adj_close 欄，O/H/L/C 依 C->adj_close 比例調整
    bool forward_fill = true;            // 缺值 forward-fill
    bool back_fill = true;               // 首段缺值 back-fill
    bool auto_non_tradable = true;       // 若 V==0 或 VAL==0，自動標記 tradable_flag=0
    double winsorize_p = 0.01;           // 1%/99% winsorize（用於報酬與特徵的極端值處理）

    // BIAS 與 Broker Strength 平滑
    int bias_ma_n = 10;                  // BIAS 的 MA 視窗（或用 EMA 半衰期）
    bool bias_use_ema = false;
    double bias_ema_half_life = 10.0;    // 若使用 EMA：半衰期=10

    int broker_strength_ema = 3;         // 3~5 天 EMA 平滑
};

// ----------------------- DataLoader -----------------------
class DataLoader {
public:
    explicit DataLoader(const std::string& basePath);

    // 讀檔 + 清理 + 特徵（2.1→2.4）
    // 回傳 true 代表成功建好所有矩陣
    bool loadDailyPanelAndBuildFeatures();

    // ===== Getters：原始欄位 (T x N) =====
    const std::vector<std::string>& dates()   const { return dates_; }
    const std::vector<std::string>& symbols() const { return symbols_; }

    const Eigen::MatrixXd& O()   const { return O_; }
    const Eigen::MatrixXd& H()   const { return H_; }
    const Eigen::MatrixXd& L()   const { return L_; }
    const Eigen::MatrixXd& C()   const { return C_; }
    const Eigen::MatrixXd& V()   const { return V_; }    // shares
    const Eigen::MatrixXd& VAL() const { return VAL_; }  // traded value (TWD)
    const Eigen::MatrixXd& insideVol()  const { return inside_vol_; }
    const Eigen::MatrixXd& outsideVol() const { return outside_vol_; }
    const std::vector<StockInfo>& stock_info() const { return stock_info_; }

    // ===== Getters：特徵 (T x N) =====
    // 2.2.1 Price/Momentum
    const Eigen::MatrixXd& feat_Gap()   const { return feat_gap_; }     // (O_t - C_{t-1})/C_{t-1}
    const Eigen::MatrixXd& feat_Mom5()  const { return feat_mom5_; }    // (C_t - C_{t-5})/C_{t-5}
    const Eigen::MatrixXd& feat_Mom10() const { return feat_mom10_; }
    const Eigen::MatrixXd& feat_Mom20() const { return feat_mom20_; }
    const Eigen::MatrixXd& feat_GKVol() const { return feat_gkvol_; }   // Garman–Klass σ_t

    // 2.2.2 Volume/Liquidity
    const Eigen::MatrixXd& feat_Liquidity()    const { return feat_liquidity_; }     // log(VAL+1)
    const Eigen::MatrixXd& feat_TurnoverShare()const { return feat_turnover_share_; }// VAL_i / Σ VAL

    // 2.3 Order Imbalance
    const Eigen::MatrixXd& feat_Imbalance()    const { return feat_imb_; }

    // 2.4 BIAS & Broker Strength
    const Eigen::MatrixXd& feat_BIAS()         const { return feat_bias_; }
    const Eigen::MatrixXd& feat_BrokerStrength() const { return feat_broker_strength_; }

    // 設定（可選）
    CleanFeatureConfig& config() { return cfg_; }
    #ifdef DATA_LOADER_TESTING 
        public:
            // 測試專用：把 private 的 readDailyPricesCsv 暴露給單元測試使用
    		bool TEST_readDailyPricesCsv(const std::string& filename) { 
    		    return readDailyPricesCsv(filename); 
    		}
	#endif

private:
    std::string base_path_;
    CleanFeatureConfig cfg_;

    // 映射
    std::vector<std::string> dates_;   // size=T
    std::vector<std::string> symbols_; // size=N
    std::unordered_map<std::string,int> date2row_;
    std::unordered_map<std::string,int> sym2col_;

    // 原始欄位 (T x N)
    Eigen::MatrixXd O_, H_, L_, C_, V_, VAL_, inside_vol_, outside_vol_;
    std::vector<StockInfo> stock_info_;

    // 特徵 (T x N)
    Eigen::MatrixXd feat_gap_, feat_mom5_, feat_mom10_, feat_mom20_, feat_gkvol_;
    Eigen::MatrixXd feat_liquidity_, feat_turnover_share_;
    Eigen::MatrixXd feat_imb_, feat_bias_, feat_broker_strength_;

    // 讀檔
    bool readDailyPricesCsv(const std::string& filename);   // 必含：date,symbol,O,H,L,C,V,VAL，可選：inside_vol,outside_vol,adj_close
    bool readBrokersCsv(const std::string& filename);       // 可含：date,symbol,broker_netbuy_topk 或 broker_buy_topk,broker_sell_topk
    bool readMetaCsv(const std::string& filename);          // 可含：symbol,lot_size,odd_lot_unit,industry,group,tradable_flag

    // 清理
    void adjustOHLCbyAdjCloseIfAny();      // 2.1 Price Adjustment
    void imputeMissing();                  // 2.1 Missing-Value Imputation
    void markNonTradableByVolVal();        // 2.1 Non-tradable Flagging
    void winsorizeAll();                   // 2.1 Winsorization of Extremes (對報酬與特徵)

    // 特徵
    void buildPriceMomentumGK();           // 2.2.1
    void buildLiquidityTurnoverShare();    // 2.2.2
    void buildImbalance();                 // 2.3
    void buildBIASandBrokerStrength();     // 2.4

    // 小工具
    void ensureShapes();
    static double safe_div(double num, double den) { return (std::abs(den) < 1e-12) ? 0.0 : (num/den); }
    static double ema_alpha_from_half_life(double hl) { return 1.0 - std::exp(std::log(0.5)/hl); }
    static void  winsorize_vec(Eigen::VectorXd& v, double p);
};

