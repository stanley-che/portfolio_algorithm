//data_capture.h 
#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>

// ===== 行情聚合資料（含粗估內外盤） =====
struct LiveQuoteAgg {
    double bestBid = 0.0;
    double bestAsk = 0.0;
    long long totalVolume = 0;   // 成交總量
    long long outsideVolume = 0; // 以賣一(或以上)成交 => 主動買
    long long insideVolume  = 0; // 以買一(或以下)成交 => 主動賣
    double lastPrice = 0.0;
    std::string lastTime;
};

/**
 * @brief 兆豐 Speedy/Starwave 的簡易封裝
 *
 * 需求：
 * - 同資料夾放好：spdOrderAPI.h/.cpp, spdQuoteAPI.h/.cpp
 * - 執行檔同層需有：megaSpeedyAPI_64.dll（或 x86 版）、common/speedyAPI_config.json、common/Temp/
 *
 * 用法概要：
 *   DataCapture dc;
 *   dc.LogonStarwave(ip, port, id, pw, true);
 *   dc.SetupOrderCert(pfx, pfxId, pfxPwd);
 *   dc.SetAccount("TWSE", brokerId, account);
 *   dc.ConnectSpeedy(orderIp, orderPort, 10);
 *   dc.LogonSpeedy(orderHostId, orderHostPw, account);
 *   dc.Subscribe("TWSE", "2330");
 *   auto q = dc.GetLiveQuote("2330");
 *   auto dk = dc.GetDKChartData("2330");
 */
class DataCapture {
public:
    DataCapture();
    ~DataCapture();

    // 不允許 copy；允許 move
    DataCapture(const DataCapture&) = delete;
    DataCapture& operator=(const DataCapture&) = delete;
    DataCapture(DataCapture&&) noexcept;
    DataCapture& operator=(DataCapture&&) noexcept;

    // ===== 行情：登入 / 訂閱 / 取消 =====
    bool LogonStarwave(const std::string& ip, int port,
                       const std::string& id, const std::string& pw,
                       bool downloadContracts = true);
    bool Subscribe(const std::string& exchange, const std::string& symbol);
    void UnsubscribeAll();

    // 取得行情快照（thread-safe，回傳副本）
    LiveQuoteAgg GetLiveQuote(const std::string& symbol);

    // ===== 交易：憑證 / 帳號 / 連線 / 登入 / 斷線 =====
    bool SetupOrderCert(const std::string& pfxPath,
                        const std::string& pfxId,
                        const std::string& pfxPwd);
    bool SetAccount(const std::string& exchange,  // 大寫：TWSE/OTC/TAIFEX
                    const std::string& brokerId,
                    const std::string& account);
    void ConnectSpeedy(const std::string& ip, int port, int timeoutSec);
    bool LogonSpeedy(const std::string& hostId,
                     const std::string& hostPw,
                     const std::string& account);
    void DisconnectSpeedy();

    // ===== 盤後 K 線：JSON 字串直接回傳 =====
    std::string GetDKChartData(const std::string& symbol); // 日K
    std::string GetWKChartData(const std::string& symbol); // 週K
    std::string GetMKChartData(const std::string& symbol); // 月K

private:
    struct Impl;     // Pimpl：把第三方相依藏在 .cpp
    Impl* impl_ = nullptr;
};

