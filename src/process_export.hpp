//process_export.hpp
#pragma once
#include "process_forecast.hpp"
#include <fstream>
#include <iostream>

#ifdef USE_XLSX
extern "C" {
#include <xlsxwriter.h>
}
#endif

namespace process {

class ProcessExporter {
public:
    explicit ProcessExporter(const DataLoader& dl) : eng_(dl) {}

    void dump_to_excel(const DumpExcelConfig& out,
                       int t,
                       const ProcessingConfig& px_cfg,
                       const RiskConfig& risk_cfg,
                       const CostConfig& cost_cfg,
                       const BlackSwanConfig& bs_cfg,
                       const ForecastConfig& fc_cfg,
                       const Eigen::VectorXd& A,
                       const Eigen::VectorXi& side,
                       const std::optional<Eigen::MatrixXd>& next_day_returns = std::nullopt) const;

    void dump_trade_plan_excel(const std::string& out_path,
                               int t,
                               const Eigen::VectorXd& A_socp,
                               const Eigen::VectorXi& side_socp,
                               double P0) const;

private:
    ProcessEngineForecast eng_;

    // ---- csv helpers ----
    static void csv_write_vector_(const std::string& path, const Eigen::VectorXd& v, const char* col);
    static void csv_write_int_vector_(const std::string& path, const Eigen::VectorXi& v, const char* col);
    static void csv_write_matrix_(const std::string& path, const Eigen::MatrixXd& M);

#ifdef USE_XLSX
    // ---- xlsx helpers ----
    static void xlsx_write_vector_(lxw_worksheet* ws, int row0, int col0,
                                   const Eigen::VectorXd& v, const char* title, const char* colName="value");
    static void xlsx_write_int_vector_(lxw_worksheet* ws, int row0, int col0,
                                       const Eigen::VectorXi& v, const char* title, const char* colName="value");
    static void xlsx_write_matrix_(lxw_worksheet* ws, int row0, int col0,
                                   const Eigen::MatrixXd& M, const char* title);
#endif
};

inline void ProcessExporter::csv_write_vector_(const std::string& path, const Eigen::VectorXd& v, const char* col){
    std::ofstream f(path);
    f << "index," << col << "\n";
    for(int i=0;i<v.size();++i) f << i << "," << detail::fin(v(i)) << "\n";
}
inline void ProcessExporter::csv_write_int_vector_(const std::string& path, const Eigen::VectorXi& v, const char* col){
    std::ofstream f(path);
    f << "index," << col << "\n";
    for(int i=0;i<v.size();++i) f << i << "," << v(i) << "\n";
}
inline void ProcessExporter::csv_write_matrix_(const std::string& path, const Eigen::MatrixXd& M){
    std::ofstream f(path);
    int m=M.rows(), n=M.cols();
    f << "row"; for(int j=0;j<n;++j) f << ",C" << j; f << "\n";
    for(int i=0;i<m;++i){
        f << "R" << i;
        for(int j=0;j<n;++j) f << "," << detail::fin(M(i,j));
        f << "\n";
    }
}

#ifdef USE_XLSX
inline void ProcessExporter::xlsx_write_vector_(lxw_worksheet* ws,int row0,int col0,
                                               const Eigen::VectorXd& v,const char* title,const char* colName){
    int r=row0;
    if(title && *title) worksheet_write_string(ws,r++,col0,title,nullptr);
    worksheet_write_string(ws,r,col0,"index",nullptr);
    worksheet_write_string(ws,r,col0+1,colName,nullptr);
    ++r;
    for(int i=0;i<v.size();++i){
        worksheet_write_number(ws,r+i,col0,i,nullptr);
        worksheet_write_number(ws,r+i,col0+1,detail::fin(v(i)),nullptr);
    }
}
inline void ProcessExporter::xlsx_write_int_vector_(lxw_worksheet* ws,int row0,int col0,
                                                   const Eigen::VectorXi& v,const char* title,const char* colName){
    int r=row0;
    if(title && *title) worksheet_write_string(ws,r++,col0,title,nullptr);
    worksheet_write_string(ws,r,col0,"index",nullptr);
    worksheet_write_string(ws,r,col0+1,colName,nullptr);
    ++r;
    for(int i=0;i<v.size();++i){
        worksheet_write_number(ws,r+i,col0,i,nullptr);
        worksheet_write_number(ws,r+i,col0+1,v(i),nullptr);
    }
}
inline void ProcessExporter::xlsx_write_matrix_(lxw_worksheet* ws,int row0,int col0,
                                               const Eigen::MatrixXd& M,const char* title){
    int r=row0; int m=M.rows(), n=M.cols();
    if(title && *title) worksheet_write_string(ws,r++,col0,title,nullptr);
    worksheet_write_string(ws,r,col0,"row",nullptr);
    for(int j=0;j<n;++j){
        std::string h="C"+std::to_string(j);
        worksheet_write_string(ws,r,col0+1+j,h.c_str(),nullptr);
    }
    ++r;
    for(int i=0;i<m;++i){
        std::string rn="R"+std::to_string(i);
        worksheet_write_string(ws,r+i,col0,rn.c_str(),nullptr);
        for(int j=0;j<n;++j)
            worksheet_write_number(ws,r+i,col0+1+j,detail::fin(M(i,j)),nullptr);
    }
}
#endif

inline void ProcessExporter::dump_to_excel(const DumpExcelConfig& out,
                                          int t,
                                          const ProcessingConfig& px_cfg,
                                          const RiskConfig& risk_cfg,
                                          const CostConfig& cost_cfg,
                                          const BlackSwanConfig& bs_cfg,
                                          const ForecastConfig& fc_cfg,
                                          const Eigen::VectorXd& A,
                                          const Eigen::VectorXi& side,
                                          const std::optional<Eigen::MatrixXd>& next_day_returns) const
{
    auto rsk = eng_.build_liquidity_scaled_risk(t, risk_cfg);
    auto cst = eng_.estimate_costs(t, A, side, cost_cfg);
    auto bs  = eng_.black_swan_scale(t, bs_cfg, false);
    auto fo  = eng_.stage1_forecast(t, px_cfg, fc_cfg, next_day_returns);

#ifdef USE_XLSX
    if(out.prefer_xlsx){
        lxw_workbook* wb = workbook_new(out.xlsx_path.c_str());
        if(!wb){ std::cerr<<"[dump_to_excel] cannot create "<<out.xlsx_path<<"\n"; return; }

        // risk
        if(auto* ws=workbook_add_worksheet(wb,"risk")){
            int base=0;
            xlsx_write_matrix_(ws, base,0, rsk.Sigma_tilde,"Sigma_tilde"); base+=rsk.Sigma_tilde.rows()+3;
            xlsx_write_matrix_(ws, base,0, rsk.L,          "Cholesky_L");  base+=rsk.L.rows()+3;
            xlsx_write_vector_(ws, base,0, rsk.sigma_i,    "sigma_i");     base+=rsk.sigma_i.size()+3;
            xlsx_write_vector_(ws, base,0, rsk.Dliq_i,     "Dliq_i");
        }
        // cost
        if(auto* ws=workbook_add_worksheet(wb,"cost")){
            int base=0;
            xlsx_write_vector_(ws, base,0, cst.fee_buy, "fee_buy"); base+=cst.fee_buy.size()+3;
            xlsx_write_vector_(ws, base,0, cst.fee_sell,"fee_sell");base+=cst.fee_sell.size()+3;
            xlsx_write_vector_(ws, base,0, cst.tax_sell,"tax_sell");base+=cst.tax_sell.size()+3;
            xlsx_write_vector_(ws, base,0, cst.impact,  "impact");
        }
        // black_swan
        if(auto* ws=workbook_add_worksheet(wb,"black_swan")){
            worksheet_write_string(ws,0,0,"m",nullptr);
            worksheet_write_number(ws,0,1,bs.m,nullptr);
        }
        // stage1
        if(auto* ws=workbook_add_worksheet(wb,"stage1")){
            xlsx_write_vector_(ws,0,0,fo.rhat_raw,"rhat_raw");
        }

        workbook_close(wb);
        return;
    }
#endif

    // CSV fallback
    csv_write_matrix_(out.out_dir_csv + "/risk_Sigma_tilde.csv", rsk.Sigma_tilde);
    csv_write_matrix_(out.out_dir_csv + "/risk_Cholesky_L.csv",  rsk.L);
    csv_write_vector_(out.out_dir_csv + "/risk_sigma_i.csv",     rsk.sigma_i, "sigma_i");
    csv_write_vector_(out.out_dir_csv + "/risk_Dliq_i.csv",      rsk.Dliq_i,  "Dliq_i");

    csv_write_vector_(out.out_dir_csv + "/cost_fee_buy.csv",  cst.fee_buy,  "fee_buy");
    csv_write_vector_(out.out_dir_csv + "/cost_fee_sell.csv", cst.fee_sell, "fee_sell");
    csv_write_vector_(out.out_dir_csv + "/cost_tax_sell.csv", cst.tax_sell, "tax_sell");
    csv_write_vector_(out.out_dir_csv + "/cost_impact.csv",   cst.impact,   "impact");

    { std::ofstream f(out.out_dir_csv + "/black_swan.csv"); f<<"m\n"<<bs.m<<"\n"; }
    csv_write_vector_(out.out_dir_csv + "/stage1_rhat_raw.csv", fo.rhat_raw, "rhat_raw");
    csv_write_vector_(out.out_dir_csv + "/inputs_A.csv", A, "A");
    csv_write_int_vector_(out.out_dir_csv + "/inputs_side.csv", side, "side");
}

inline void ProcessExporter::dump_trade_plan_excel(const std::string& out_path,
                                                   int t,
                                                   const Eigen::VectorXd& A_socp,
                                                   const Eigen::VectorXi& side_socp,
                                                   double P0) const
{
    const int N = (int)A_socp.size();
    auto price_row = [&](int day)->Eigen::VectorXd{
        if(day<0 || day>=eng_.dl_.C().rows()) return Eigen::VectorXd::Zero(N);
        Eigen::VectorXd v = eng_.dl_.C().row(day).transpose();
        for(int i=0;i<N;++i) if(!std::isfinite(v(i))||v(i)<=0) v(i)=0.0;
        return v;
    };
    Eigen::VectorXd price = price_row(t);

#ifdef USE_XLSX
    lxw_workbook* wb = workbook_new(out_path.c_str());
    if(!wb){ std::cerr<<"[trade_plan] cannot create "<<out_path<<"\n"; return; }
    lxw_worksheet* ws = workbook_add_worksheet(wb,"trade_plan");
    int r=0;
    worksheet_write_string(ws,r,0,"index",nullptr);
    worksheet_write_string(ws,r,1,"w_est",nullptr);
    worksheet_write_string(ws,r,2,"abs_w",nullptr);
    worksheet_write_string(ws,r,3,"A_notional",nullptr);
    worksheet_write_string(ws,r,4,"side",nullptr);
    worksheet_write_string(ws,r,5,"price_t",nullptr);
    worksheet_write_string(ws,r,6,"shares_abs",nullptr);
    worksheet_write_string(ws,r,7,"shares",nullptr);
    ++r;

    for(int i=0;i<N;++i){
        double Ai = std::max(0.0, A_socp(i));
        double pr = price(i);
        double abs_w = (P0>0.0)? std::min(1.0, Ai/std::max(1e-12,P0)) : 0.0;
        double w = abs_w * (double)side_socp(i);
        double sh_abs = (pr>0.0)? Ai/pr : 0.0;
        double sh = sh_abs * (double)side_socp(i);

        worksheet_write_number(ws,r,0,i,nullptr);
        worksheet_write_number(ws,r,1,w,nullptr);
        worksheet_write_number(ws,r,2,abs_w,nullptr);
        worksheet_write_number(ws,r,3,Ai,nullptr);
        worksheet_write_number(ws,r,4,side_socp(i),nullptr);
        worksheet_write_number(ws,r,5,pr,nullptr);
        worksheet_write_number(ws,r,6,sh_abs,nullptr);
        worksheet_write_number(ws,r,7,sh,nullptr);
        ++r;
    }
    worksheet_write_string(ws,r+1,0,"P0",nullptr);
    worksheet_write_number(ws,r+1,1,P0,nullptr);
    worksheet_write_string(ws,r+2,0,"t",nullptr);
    worksheet_write_number(ws,r+2,1,t,nullptr);
    workbook_close(wb);
#else
    std::ofstream f(out_path);
    f<<"index,w_est,abs_w,A_notional,side,price_t,shares_abs,shares\n";
    for(int i=0;i<N;++i){
        double Ai = std::max(0.0, A_socp(i));
        double pr = price(i);
        double abs_w = (P0>0.0)? std::min(1.0, Ai/std::max(1e-12,P0)) : 0.0;
        double w = abs_w * (double)side_socp(i);
        double sh_abs = (pr>0.0)? Ai/pr : 0.0;
        double sh = sh_abs * (double)side_socp(i);
        f<<i<<","<<w<<","<<abs_w<<","<<Ai<<","<<side_socp(i)<<","<<pr<<","<<sh_abs<<","<<sh<<"\n";
    }
    f<<"P0,"<<P0<<"\n";
    f<<"t,"<<t<<"\n";
#endif
}

} // namespace process
