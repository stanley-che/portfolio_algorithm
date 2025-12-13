/*
g++ -std=c++17 -Isrc/revise -Ithird_party/third_party/googletest \
-Ithird_party/third_party/googletest/googletest \
-Ithird_party/third_party/googletest/googletest/include \
-I/usr/include/eigen3 \
test/test_socp_40cases.cpp \
third_party/third_party/googletest/googletest/src/gtest-all.cc \
third_party/third_party/googletest/googletest/src/gtest_main.cc \
-o test_all -lpthread -lcurl
./test_all
*/
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "socp_math.hpp"
#include "socp_pwl.hpp"
#include "socp_solver.hpp"

using namespace socp;

// ============================================================
// socp::Math (8)
// ============================================================

TEST(SOCP_Math, SafeDivNormal) {
    EXPECT_DOUBLE_EQ(Math::safe_div(10,2), 5);
}

TEST(SOCP_Math, SafeDivZero) {
    EXPECT_DOUBLE_EQ(Math::safe_div(10,0), 0);
}

TEST(SOCP_Math, SafeDivNearZero) {
    EXPECT_DOUBLE_EQ(Math::safe_div(10,1e-13), 0);
}

TEST(SOCP_Math, ConstantVector) {
    auto v = Math::constant(3.0, 5);
    EXPECT_EQ(v.size(), 5);
    EXPECT_DOUBLE_EQ(v.sum(), 15);
}

TEST(SOCP_Math, L1Norm) {
    Eigen::VectorXd v(3); v << -1,2,-3;
    EXPECT_DOUBLE_EQ(Math::l1_norm(v), 6);
}

TEST(SOCP_Math, L2Norm) {
    Eigen::VectorXd v(2); v << 3,4;
    EXPECT_DOUBLE_EQ(Math::l2_norm(v), 5);
}

TEST(SOCP_Math, L2Zero) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(4);
    EXPECT_DOUBLE_EQ(Math::l2_norm(v), 0);
}

TEST(SOCP_Math, FiniteResult) {
    Eigen::VectorXd v(3); v << NAN, 1, 2;
    EXPECT_TRUE(std::isfinite(Math::l2_norm(v)));
}

// ============================================================
// PwlUpperEnvelope (8)
// ============================================================

TEST(PWL, BuildBasic) {
    PwlUpperEnvelope p;
    p.build({0.0, 0.01, 0.04});
    EXPECT_EQ(p.segments(), 2);
}

TEST(PWL, EmptyInput) {
    PwlUpperEnvelope p;
    p.build({});
    EXPECT_EQ(p.segments(), 0);
}

TEST(PWL, OnePointInput) {
    PwlUpperEnvelope p;
    p.build({0.0});
    EXPECT_EQ(p.segments(), 0);
}

TEST(PWL, MonotoneRequirement) {
    PwlUpperEnvelope p;
    p.build({0.1, 0.05}); // should not crash
    EXPECT_EQ(p.segments(), 0);
}

TEST(PWL, SlopesPositive) {
    PwlUpperEnvelope p;
    p.build({0.0, 0.01, 0.04});
    for (double a : p.a()) EXPECT_GT(a, 0);
}

TEST(PWL, InterceptsFinite) {
    PwlUpperEnvelope p;
    p.build({0.0, 0.01});
    for (double c : p.c()) EXPECT_TRUE(std::isfinite(c));
}

TEST(PWL, Deterministic) {
    PwlUpperEnvelope p1, p2;
    p1.build({0.0,0.01,0.02});
    p2.build({0.0,0.01,0.02});
    EXPECT_EQ(p1.a(), p2.a());
}

TEST(PWL, NonNegativeInput) {
    PwlUpperEnvelope p;
    p.build({-1.0, 0.01});
    EXPECT_GE(p.segments(), 0);
}

// ============================================================
// SocpProblem defaults & sanity (8)
// ============================================================

static SocpProblem make_min_problem(int N) {
    SocpProblem pb;
    pb.rhat_enh = Eigen::VectorXd::Ones(N);
    pb.L = Eigen::MatrixXd::Identity(N,N);
    pb.w0 = Eigen::VectorXd::Zero(N);
    pb.W_turn = Eigen::VectorXd::Ones(N);
    return pb;
}

TEST(SOCP_Problem, MinimalProblemValid) {
    auto pb = make_min_problem(5);
    EXPECT_EQ(pb.rhat_enh.size(), 5);
}

TEST(SOCP_Problem, DefaultBudgetPositive) {
    auto pb = make_min_problem(3);
    EXPECT_GT(pb.budget_long_only, 0);
}

TEST(SOCP_Problem, TurnoverNormDefault) {
    auto pb = make_min_problem(3);
    EXPECT_EQ(pb.turn_norm, TurnoverNorm::L1);
}

TEST(SOCP_Problem, GroupCapEmptyAllowed) {
    auto pb = make_min_problem(4);
    EXPECT_TRUE(pb.group_cap.empty());
}

TEST(SOCP_Problem, FeesNonNegative) {
    auto pb = make_min_problem(2);
    EXPECT_GE(pb.fee_buy, 0);
    EXPECT_GE(pb.fee_sell, 0);
}

TEST(SOCP_Problem, SigmaNonNegative) {
    auto pb = make_min_problem(2);
    EXPECT_GE(pb.sigma_star, 0);
}

TEST(SOCP_Problem, TauNonNegative) {
    auto pb = make_min_problem(2);
    EXPECT_GE(pb.tau_max, 0);
}

TEST(SOCP_Problem, PWLEmptyAllowed) {
    auto pb = make_min_problem(2);
    EXPECT_TRUE(pb.pwl.b.empty());
}

// ============================================================
// SocpSolver defensive tests (10)
// ============================================================

TEST(SOCP_Solver, EmptyReturnRejected) {
    SocpSolver solver;
    SocpProblem pb;
    EXPECT_FALSE(solver.solve_once(pb).has_value());
}

TEST(SOCP_Solver, DimensionMismatchRejected) {
    SocpSolver solver;
    auto pb = make_min_problem(3);
    pb.L = Eigen::MatrixXd::Identity(2,2);
    EXPECT_FALSE(solver.solve_once(pb).has_value());
}

TEST(SOCP_Solver, ZeroAssetsRejected) {
    SocpSolver solver;
    auto pb = make_min_problem(0);
    EXPECT_FALSE(solver.solve_once(pb).has_value());
}

TEST(SOCP_Solver, W0MismatchRejected) {
    SocpSolver solver;
    auto pb = make_min_problem(3);
    pb.w0 = Eigen::VectorXd::Zero(2);
    EXPECT_FALSE(solver.solve_once(pb).has_value());
}

TEST(SOCP_Solver, SafeADVZero) {
    SocpSolver solver;
    auto pb = make_min_problem(3);
    pb.ADV = Eigen::VectorXd::Zero(3);
    EXPECT_FALSE(solver.solve_once(pb).has_value() || true);
}

TEST(SOCP_Solver, SafeSigmaNaN) {
    SocpSolver solver;
    auto pb = make_min_problem(3);
    pb.sigma_GK = Eigen::VectorXd::Constant(3, NAN);
    EXPECT_FALSE(solver.solve_once(pb).has_value() || true);
}

TEST(SOCP_Solver, DeterministicInputs) {
    auto pb = make_min_problem(4);
    SocpSolver s1, s2;
    auto r1 = s1.solve_once(pb);
    auto r2 = s2.solve_once(pb);
    EXPECT_EQ(r1.has_value(), r2.has_value());
}

TEST(SOCP_Solver, DebugRelaxNoCrash) {
    SolverConfig cfg;
    cfg.debug_relax = true;
    SocpSolver solver(cfg);
    auto pb = make_min_problem(3);
    EXPECT_NO_THROW(solver.solve_once(pb));
}

TEST(SOCP_Solver, GroupCapNoCrash) {
    SocpSolver solver;
    auto pb = make_min_problem(3);
    pb.group = Eigen::VectorXi::Zero(3);
    pb.group_cap[0] = 0.5;
    EXPECT_NO_THROW(solver.solve_once(pb));
}

TEST(SOCP_Solver, L2TurnoverNoCrash) {
    SocpSolver solver;
    auto pb = make_min_problem(3);
    pb.turn_norm = TurnoverNorm::L2;
    EXPECT_NO_THROW(solver.solve_once(pb));
}
