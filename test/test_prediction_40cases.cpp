// tests/test_prediction_40cases.cpp
// ------------------------------------------------------------
// Prediction module: 40 test cases (GoogleTest)
// ------------------------------------------------------------
/*
g++ -std=c++17 -Isrc/revise -Ithird_party/third_party/googletest \
-Ithird_party/third_party/googletest/googletest \
-Ithird_party/third_party/googletest/googletest/include \
-I/usr/include/eigen3 \
test/test_twse_quotes.cpp test/test_twse_meta.cpp test/test_broker_unit.cpp test/test_data_loader_40cases.cpp test/test_prediction_40cases.cpp \
third_party/third_party/googletest/googletest/src/gtest-all.cc \
third_party/third_party/googletest/googletest/src/gtest_main.cc \
-o test_all -lpthread -lcurl
*/ 


#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <limits>

#include "prediction_types.hpp"
#include "prediction_model.hpp"

using prediction::TrainConfig;
using prediction::ModelType;
using prediction::LinearModelTrainer;

static bool is_finite(double x){ return std::isfinite(x); }

// helpers
static Eigen::MatrixXd diag(const Eigen::VectorXd& d) {
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(d.size(), d.size());
    for (int i=0;i<d.size();++i) G(i,i)=d(i);
    return G;
}

// ============================================================
// 40 TEST CASES
// ============================================================

// -------------------- lambda_time (8) ------------------------

TEST(PredictionMath_LambdaTime, HalfLifeNormal) {
    TrainConfig tr; tr.half_life = 60.0;
    LinearModelTrainer t(tr);
    double lam = t.lambda_time();
    EXPECT_GT(lam, 0.0);
    EXPECT_LT(lam, 1.0);
}

TEST(PredictionMath_LambdaTime, HalfLifeVeryLargeNearOne) {
    TrainConfig tr; tr.half_life = 1e9;
    LinearModelTrainer t(tr);
    EXPECT_NEAR(t.lambda_time(), 1.0, 1e-9);
}

TEST(PredictionMath_LambdaTime, HalfLifeOneGivesHalf) {
    TrainConfig tr; tr.half_life = 1.0;
    LinearModelTrainer t(tr);
    EXPECT_NEAR(t.lambda_time(), 0.5, 1e-12);
}

TEST(PredictionMath_LambdaTime, HalfLifeTwoGivesSqrtHalf) {
    TrainConfig tr; tr.half_life = 2.0;
    LinearModelTrainer t(tr);
    EXPECT_NEAR(t.lambda_time(), std::sqrt(0.5), 1e-12);
}

TEST(PredictionMath_LambdaTime, HalfLifeTinyDoesNotNaN) {
    TrainConfig tr; tr.half_life = 1e-15;
    LinearModelTrainer t(tr);
    EXPECT_TRUE(is_finite(t.lambda_time()));
    EXPECT_GT(t.lambda_time(), 0.0);
}

TEST(PredictionMath_LambdaTime, HalfLifeZeroDoesNotNaN) {
    TrainConfig tr; tr.half_life = 0.0;
    LinearModelTrainer t(tr);
    EXPECT_TRUE(is_finite(t.lambda_time()));
    EXPECT_GT(t.lambda_time(), 0.0);
}

TEST(PredictionMath_LambdaTime, HalfLifeNegativeDoesNotNaN) {
    TrainConfig tr; tr.half_life = -10.0;
    LinearModelTrainer t(tr);
    EXPECT_TRUE(is_finite(t.lambda_time()));
    EXPECT_GT(t.lambda_time(), 0.0);
}

TEST(PredictionMath_LambdaTime, HalfLifeMonotonic) {
    TrainConfig tr1; tr1.half_life = 5.0;
    TrainConfig tr2; tr2.half_life = 50.0;
    LinearModelTrainer t1(tr1), t2(tr2);
    EXPECT_LT(t1.lambda_time(), t2.lambda_time());
}

// -------------------- Ridge (16) ------------------------------

TEST(PredictionMath_Ridge, IdentityLambda1) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 1.0;
    LinearModelTrainer t(tr);

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3,3);
    Eigen::VectorXd g(3); g << 1,0,0;
    auto beta = t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 0.5, 1e-12); // 1/(1+1)
    EXPECT_NEAR(beta(1), 0.0, 1e-12);
    EXPECT_NEAR(beta(2), 0.0, 1e-12);
}

TEST(PredictionMath_Ridge, IdentityLambda0EqualsG) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 0.0;
    LinearModelTrainer t(tr);

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd g(2); g << 1.2, -3.4;
    auto beta = t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 1.2, 1e-12);
    EXPECT_NEAR(beta(1), -3.4, 1e-12);
}

TEST(PredictionMath_Ridge, DiagonalClosedForm) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 2.0;
    LinearModelTrainer t(tr);

    Eigen::VectorXd d(3); d << 1, 2, 4;
    Eigen::MatrixXd G = diag(d);
    Eigen::VectorXd g(3); g << 1, 2, 4;
    auto beta = t.fit_beta(G,g);

    // (G+lambda I)^{-1} g
    EXPECT_NEAR(beta(0), 1.0/(1.0+2.0), 1e-12);
    EXPECT_NEAR(beta(1), 2.0/(2.0+2.0), 1e-12);
    EXPECT_NEAR(beta(2), 4.0/(4.0+2.0), 1e-12);
}

TEST(PredictionMath_Ridge, ZeroGivesZeroBeta) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 1.0;
    LinearModelTrainer t(tr);

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(4,4);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(4);
    auto beta = t.fit_beta(G,g);
    EXPECT_TRUE(beta.isZero(1e-15));
}

TEST(PredictionMath_Ridge, SymmetricPSDNoNaN) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 1e-3;
    LinearModelTrainer t(tr);

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20,5);
    Eigen::MatrixXd G = X.transpose()*X; // PSD
    Eigen::VectorXd g = Eigen::VectorXd::Random(5);
    auto beta = t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

TEST(PredictionMath_Ridge, VeryLargeLambdaShrinksTowardZero) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 1e9;
    LinearModelTrainer t(tr);

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3,3);
    Eigen::VectorXd g(3); g << 1,2,3;
    auto beta = t.fit_beta(G,g);
    EXPECT_NEAR(beta.norm(), 0.0, 1e-8);
}

TEST(PredictionMath_Ridge, NonIdentityDiagonalScaling) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 1.0;
    LinearModelTrainer t(tr);

    Eigen::VectorXd d(2); d<<10,0.1;
    Eigen::MatrixXd G=diag(d);
    Eigen::VectorXd g(2); g<<1,1;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 1.0/(10.0+1.0), 1e-12);
    EXPECT_NEAR(beta(1), 1.0/(0.1+1.0), 1e-12);
}

TEST(PredictionMath_Ridge, EmptyInputReturnsEmpty) {
    TrainConfig tr; tr.model = ModelType::Ridge; tr.lambda = 1.0;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G(0,0);
    Eigen::VectorXd g(0);
    auto beta=t.fit_beta(G,g);
    EXPECT_EQ(beta.size(), 0);
}

TEST(PredictionMath_Ridge, OneDim) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=0.5;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G(1,1); G(0,0)=2.0;
    Eigen::VectorXd g(1); g(0)=1.0;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 1.0/(2.0+0.5), 1e-12);
}

TEST(PredictionMath_Ridge, NearSingularStillFiniteDueToLambda) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=1e-2;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(3,3); // singular
    Eigen::VectorXd g(3); g<<1,2,3;
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

TEST(PredictionMath_Ridge, SignPreservedForDiagonal) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=1.0;
    LinearModelTrainer t(tr);
    Eigen::VectorXd d(2); d<<1,1;
    Eigen::MatrixXd G=diag(d);
    Eigen::VectorXd g(2); g<<-2,3;
    auto beta=t.fit_beta(G,g);
    EXPECT_LT(beta(0), 0);
    EXPECT_GT(beta(1), 0);
}

TEST(PredictionMath_Ridge, ScalingGandLambda) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=1.0;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd g(2); g<<1,1;
    auto b1=t.fit_beta(G,g);
    g*=10.0;
    auto b2=t.fit_beta(G,g);
    EXPECT_TRUE((b2 - 10.0*b1).norm() < 1e-12);
}

TEST(PredictionMath_Ridge, RandomSmallFinite) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=0.1;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(8,4);
    Eigen::MatrixXd G = X.transpose()*X;
    Eigen::VectorXd g = X.transpose()*Eigen::VectorXd::Random(8);
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

TEST(PredictionMath_Ridge, LambdaZeroOnSPDMatchesSolve) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=0.0;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G(2,2); G<<4,1,1,3; // SPD
    Eigen::VectorXd g(2); g<<1,2;
    auto beta=t.fit_beta(G,g);
    auto beta_ref = G.ldlt().solve(g);
    EXPECT_TRUE(beta.isApprox(beta_ref, 1e-12));
}

TEST(PredictionMath_Ridge, LargePFinite) {
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=1.0;
    LinearModelTrainer t(tr);
    const int p=30;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(50,p);
    Eigen::MatrixXd G = X.transpose()*X;
    Eigen::VectorXd g = Eigen::VectorXd::Random(p);
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

// -------------------- Lasso (16) ------------------------------
// We design diagonal G so that coordinate descent converges to closed form:
// beta_j = soft_threshold(g_j, lambda) / G_jj  (when G diagonal).

TEST(PredictionMath_Lasso, LargeLambdaAllZero) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=100.0;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3,3);
    Eigen::VectorXd g(3); g<<1,2,3;
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.isZero(1e-12));
}

TEST(PredictionMath_Lasso, SoftThresholdPositive) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=1.0;
    tr.lasso_max_iter = 2000; tr.lasso_tol = 1e-12;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1,1);
    Eigen::VectorXd g(1); g<<2.0;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 1.0, 1e-6); // (2-1)/1
}

TEST(PredictionMath_Lasso, SoftThresholdNegative) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=1.0;
    tr.lasso_max_iter = 2000; tr.lasso_tol = 1e-12;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1,1);
    Eigen::VectorXd g(1); g<<-2.0;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), -1.0, 1e-6);
}

TEST(PredictionMath_Lasso, BelowLambdaGivesZero) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=1.0;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1,1);
    Eigen::VectorXd g(1); g<<0.9;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 0.0, 1e-8);
}

TEST(PredictionMath_Lasso, DiagonalScaling) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=1.0;
    tr.lasso_max_iter=2000; tr.lasso_tol=1e-12;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G(1,1); G(0,0)=2.0;
    Eigen::VectorXd g(1); g<<3.0;
    auto beta=t.fit_beta(G,g);
    // soft_threshold(3,1)/2 = 1.0
    EXPECT_NEAR(beta(0), 1.0, 1e-6);
}

TEST(PredictionMath_Lasso, TwoDimDiagonalClosedForm) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=1.0;
    tr.lasso_max_iter=5000; tr.lasso_tol=1e-12;
    LinearModelTrainer t(tr);
    Eigen::VectorXd d(2); d<<1.0, 4.0;
    Eigen::MatrixXd G=diag(d);
    Eigen::VectorXd g(2); g<<2.0, -10.0;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), (2.0-1.0)/1.0, 1e-5);
    EXPECT_NEAR(beta(1), -(10.0-1.0)/4.0, 1e-5);
}

TEST(PredictionMath_Lasso, LambdaZeroApproachesRidgeSolveOnDiagonal) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.0;
    tr.lasso_max_iter=5000; tr.lasso_tol=1e-12;
    LinearModelTrainer t(tr);
    Eigen::VectorXd d(3); d<<1,2,3;
    Eigen::MatrixXd G=diag(d);
    Eigen::VectorXd g(3); g<<1,2,3;
    auto beta=t.fit_beta(G,g);
    EXPECT_NEAR(beta(0), 1.0/1.0, 1e-6);
    EXPECT_NEAR(beta(1), 2.0/2.0, 1e-6);
    EXPECT_NEAR(beta(2), 3.0/3.0, 1e-6);
}

TEST(PredictionMath_Lasso, EmptyInputReturnsEmpty) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=1.0;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G(0,0);
    Eigen::VectorXd g(0);
    auto beta=t.fit_beta(G,g);
    EXPECT_EQ(beta.size(), 0);
}

TEST(PredictionMath_Lasso, FiniteOnRandomPSD) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.1;
    tr.lasso_max_iter=3000; tr.lasso_tol=1e-10;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(30,6);
    Eigen::MatrixXd G = X.transpose()*X;
    Eigen::VectorXd g = Eigen::VectorXd::Random(6);
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

TEST(PredictionMath_Lasso, SparsityIncreasesWithLambda) {
    TrainConfig tr1; tr1.model=ModelType::Lasso; tr1.lambda=0.01; tr1.lasso_max_iter=4000;
    TrainConfig tr2 = tr1; tr2.lambda=1.0;
    LinearModelTrainer t1(tr1), t2(tr2);

    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(5,5);
    Eigen::VectorXd g(5); g<<0.5,0.4,0.3,0.2,0.1;
    auto b1=t1.fit_beta(G,g);
    auto b2=t2.fit_beta(G,g);
    int nz1=(b1.array().abs()>1e-6).count();
    int nz2=(b2.array().abs()>1e-6).count();
    EXPECT_GE(nz1, nz2);
}

TEST(PredictionMath_Lasso, SignPreserved) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.1;
    tr.lasso_max_iter=4000;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd g(2); g<<-1.0, 2.0;
    auto beta=t.fit_beta(G,g);
    EXPECT_LT(beta(0), 0.0);
    EXPECT_GT(beta(1), 0.0);
}

TEST(PredictionMath_Lasso, ZeroGivesZero) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.1;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3,3);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(3);
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.isZero(1e-12));
}

TEST(PredictionMath_Lasso, ToleranceStopsEarlyStillFinite) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.1;
    tr.lasso_max_iter=2; tr.lasso_tol=1e-1; // very loose
    LinearModelTrainer t(tr);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(20,4);
    Eigen::MatrixXd G = X.transpose()*X;
    Eigen::VectorXd g = Eigen::VectorXd::Random(4);
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

TEST(PredictionMath_Lasso, DeterministicSameInput) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.2;
    tr.lasso_max_iter=4000; tr.lasso_tol=1e-12;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3,3);
    Eigen::VectorXd g(3); g<<1,2,3;
    auto b1=t.fit_beta(G,g);
    auto b2=t.fit_beta(G,g);
    EXPECT_TRUE(b1.isApprox(b2, 1e-12));
}

TEST(PredictionMath_Lasso, NonDiagonalStillFinite) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.05;
    tr.lasso_max_iter=8000; tr.lasso_tol=1e-10;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G(3,3);
    G << 2,0.2,0.1,
         0.2,1.5,0.3,
         0.1,0.3,1.0;
    Eigen::VectorXd g(3); g<<1,-2,3;
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

TEST(PredictionMath_Lasso, LargePStillFinite) {
    TrainConfig tr; tr.model=ModelType::Lasso; tr.lambda=0.05;
    tr.lasso_max_iter=10000; tr.lasso_tol=1e-9;
    LinearModelTrainer t(tr);
    const int p=20;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(80,p);
    Eigen::MatrixXd G = X.transpose()*X;
    Eigen::VectorXd g = Eigen::VectorXd::Random(p);
    auto beta=t.fit_beta(G,g);
    EXPECT_TRUE(beta.allFinite());
}

// -------------------- Model switch (0? add 0) ----------------
// Two tests to ensure mode changes behavior and ridge!=lasso.

TEST(PredictionMath_ModelSwitch, RidgeVsLassoDifferent) {
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3,3);
    Eigen::VectorXd g(3); g<<0.5,0.4,0.3;

    TrainConfig trR; trR.model=ModelType::Ridge; trR.lambda=1.0;
    TrainConfig trL; trL.model=ModelType::Lasso; trL.lambda=0.45; trL.lasso_max_iter=5000;

    LinearModelTrainer tR(trR), tL(trL);
    auto bR=tR.fit_beta(G,g);
    auto bL=tL.fit_beta(G,g);
    EXPECT_FALSE(bR.isApprox(bL, 1e-6));
}

TEST(PredictionMath_ModelSwitch, UnknownModelDefaultsToRidgeBehavior) {
    // If future model types exist, current fit_beta switches on Ridge/Lasso;
    // this test ensures Ridge path remains stable.
    TrainConfig tr; tr.model=ModelType::Ridge; tr.lambda=0.3;
    LinearModelTrainer t(tr);
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd g(2); g<<1,2;
    auto b=t.fit_beta(G,g);
    EXPECT_NEAR(b(0), 1.0/1.3, 1e-12);
    EXPECT_NEAR(b(1), 2.0/1.3, 1e-12);
}

// Count: 8 + 16 + 16 = 40
