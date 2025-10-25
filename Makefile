# ======== 基本設定 ========
CXX = g++
CXXFLAGS = -O2 -std=c++17 -pthread -fopenmp -DSOCP_LOG_ON=1 -DSOCP_USE_OMP=1
INCLUDES = -Isrc -Iinclude -Iinclude/full_include -Iinclude/full_include/eigen3 -Isrc/eigen3
LIBS = -lopenblas -lscsdir -lm -lcurl -fopenmp


SRC = \
    src/policy_solver.cpp \
    src/socp_generator.cpp \
    src/data_loader.cpp \
    src/dsa_executor.cpp \
    src/main.cpp \
    src/prediction.cpp \
    src/process.cpp

TARGET = build/portfolio_dsa

# ======== 編譯規則 ========
$(TARGET): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(LIBS) -o $(TARGET)
	@echo "✅ Build finished: $(TARGET)"

# ======== 清理 ========
clean:
	rm -rf build/*.o $(TARGET)
	@echo "🧹 Cleaned up build files."

# ======== 執行 ========
run: $(TARGET)
	./$(TARGET)

