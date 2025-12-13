// excel_dump.h
#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

// 啟用 .xlsx 寫出（需安裝 libxlsxwriter，見下）
#define USE_XLSX 1

#ifdef USE_XLSX
extern "C" {
#include <xlsxwriter.h>
}
#endif

namespace xls {

struct SheetSpec {
  std::string name;
};

inline std::vector<std::string> defaultColNames(int n) {
  std::vector<std::string> h(n);
  for (int i = 0; i < n; ++i) h[i] = "C" + std::to_string(i);
  return h;
}

inline std::vector<std::string> defaultRowNames(int n) {
  std::vector<std::string> h(n);
  for (int i = 0; i < n; ++i) h[i] = "R" + std::to_string(i);
  return h;
}

#ifdef USE_XLSX
inline void writeHeaderRow(lxw_worksheet* ws, int row, int col0,
                           const std::vector<std::string>& headers) {
  for (int j = 0; j < (int)headers.size(); ++j)
    worksheet_write_string(ws, row, col0 + j, headers[j].c_str(), nullptr);
}

inline void writeVector(lxw_worksheet* ws, int row0, int col0,
                        const Eigen::VectorXd& v,
                        const std::string& title = "",
                        const std::string& colName = "value") {
  int r = row0;
  if (!title.empty()) {
    worksheet_write_string(ws, r++, col0, title.c_str(), nullptr);
  }
  worksheet_write_string(ws, r, col0, "index", nullptr);
  worksheet_write_string(ws, r, col0 + 1, colName.c_str(), nullptr);
  ++r;
  for (int i = 0; i < v.size(); ++i) {
    worksheet_write_number(ws, r + i, col0, i, nullptr);
    worksheet_write_number(ws, r + i, col0 + 1,
                           std::isfinite(v(i)) ? v(i) : 0.0, nullptr);
  }
}

inline void writeIntVector(lxw_worksheet* ws, int row0, int col0,
                           const Eigen::VectorXi& v,
                           const std::string& title = "",
                           const std::string& colName = "value") {
  int r = row0;
  if (!title.empty()) {
    worksheet_write_string(ws, r++, col0, title.c_str(), nullptr);
  }
  worksheet_write_string(ws, r, col0, "index", nullptr);
  worksheet_write_string(ws, r, col0 + 1, colName.c_str(), nullptr);
  ++r;
  for (int i = 0; i < v.size(); ++i) {
    worksheet_write_number(ws, r + i, col0, i, nullptr);
    worksheet_write_number(ws, r + i, col0 + 1, v(i), nullptr);
  }
}

inline void writeMatrix(lxw_worksheet* ws, int row0, int col0,
                        const Eigen::MatrixXd& M,
                        const std::vector<std::string>& colNames = {},
                        const std::vector<std::string>& rowNames = {},
                        const std::string& title = "") {
  int r = row0;
  int m = (int)M.rows(), n = (int)M.cols();
  if (!title.empty())
    worksheet_write_string(ws, r++, col0, title.c_str(), nullptr);

  // header
  worksheet_write_string(ws, r, col0, "", nullptr);
  auto cols = colNames.empty() ? defaultColNames(n) : colNames;
  writeHeaderRow(ws, r, col0 + 1, cols);
  ++r;

  // rows
  auto rows = rowNames.empty() ? defaultRowNames(m) : rowNames;
  for (int i = 0; i < m; ++i) {
    worksheet_write_string(ws, r + i, col0, rows[i].c_str(), nullptr);
    for (int j = 0; j < n; ++j) {
      double x = M(i, j);
      if (!std::isfinite(x)) x = 0.0;
      worksheet_write_number(ws, r + i, col0 + 1 + j, x, nullptr);
    }
  }
}
#endif // USE_XLSX

} // namespace xls

