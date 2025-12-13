# ======== 基本設定 ========
CXX = g++
CXXFLAGS = -O2 -std=c++17 -pthread -fopenmp -DSOCP_LOG_ON=1 -DSOCP_USE_OMP=1
INCLUDES = -Isrc -Iinclude -Iinclude/full_include -Iinclude/full_include/eigen3 -Isrc/eigen3
LIBS = -lopenblas -lscsdir -lm -lcurl -fopenmp

DATA_DIR ?= ./src
PRICES_CSV ?= $(DATA_DIR)/daily_60d.csv
RAW_CSV := ./daily_60d.csv   # twse 程式固定輸出在目前目錄

# ======== 路徑與目標 ========
TOOLS_DIR := build/tools
TOOLS := $(TOOLS_DIR)/make_meta_twse $(TOOLS_DIR)/broker $(TOOLS_DIR)/twse

SRC = \
  src/policy_solver.cpp \
  src/socp_generator.cpp \
  src/data_loader.cpp \
  src/dsa_executor.cpp \
  src/main.cpp \
  src/prediction.cpp \
  src/process.cpp
TARGET = build/portfolio_dsa

.PHONY: all clean run pipeline prices

# ======== 預設：全建 ========
all: $(TOOLS) $(TARGET)
	@echo "✅ All builds complete!"

$(TOOLS_DIR):
	@mkdir -p $(TOOLS_DIR)

# ======== 工具：編譯到 build/tools/ ========
$(TOOLS_DIR)/twse: src/dump_twse_quotes.cpp | $(TOOLS_DIR)
	$(CXX) -O2 -std=c++17 $< -lcurl -o $@
	@echo "✅ Built: $@"

$(TOOLS_DIR)/make_meta_twse: src/make_meta_twse.cpp | $(TOOLS_DIR)
	$(CXX) -O2 -std=c++17 $< -lcurl -o $@
	@echo "✅ Built: $@"

$(TOOLS_DIR)/broker: src/broker.cpp | $(TOOLS_DIR)
	$(CXX) -O2 -std=c++17 $< -lcurl -o $@
	@echo "✅ Built: $@"

# ======== 主程式 ========
$(TARGET): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(LIBS) -o $(TARGET)
	@echo "✅ Build finished: $(TARGET)"

# ======== 產生/同步 CSV（搬到 DATA_DIR 並去 BOM） ========
prices: $(TOOLS_DIR)/twse
	@[ -d "$(DATA_DIR)" ] || mkdir -p "$(DATA_DIR)"
	@echo "▶ run twse ..."
	@$(TOOLS_DIR)/twse || echo "(twse optional: skip if not required)"
	@# 若 twse 產生了 RAW_CSV，就 strip BOM（若有）後複製到 PRICES_CSV
	@RAW="$(RAW_CSV)"; OUT="$(PRICES_CSV)"; \
	if [ -f "$$RAW" ]; then \
	  echo "▶ normalize csv (strip BOM if present) → $$OUT"; \
	  if [ "$$(head -c 3 "$$RAW" | od -An -tx1 | tr -d ' \n')" = "efbbbf" ]; then \
	    tail -c +4 "$$RAW" > "$$OUT"; \
	  else \
	    cp -f "$$RAW" "$$OUT"; \
	  fi; \
	fi
	@[ -f "$(PRICES_CSV)" ] || (echo "❌ missing $(PRICES_CSV)"; exit 1)
	@echo "✅ prices ready: $(PRICES_CSV)"


# ======== 一鍵流程：先備好 CSV，再跑工具與主程式 ========
pipeline: all prices
	@echo "▶ run make_meta_twse ..."
	@$(TOOLS_DIR)/make_meta_twse "$(PRICES_CSV)" || (echo "make_meta_twse failed"; exit 1)
	@echo "▶ run broker ..."
	@$(TOOLS_DIR)/broker "$(PRICES_CSV)" || (echo "broker failed"; exit 1)
	@echo "▶ run portfolio_dsa ..."
	@DATA_DIR="$(DATA_DIR)" ./$(TARGET) --prices "$(PRICES_CSV)"

run: pipeline

# ======== 清理 ========
clean:
	rm -rf build
	@echo "🧹 Cleaned up build files."
# ======== 快速手動同步 CSV ========
copycsv:
	cp -f daily_60d.csv src/daily_60d.csv

linkcsv:
	ln -sf ../daily_60d.csv src/daily_60d.csv

runlocal:
	make run DATA_DIR=. PRICES_CSV=./daily_60d.csv

