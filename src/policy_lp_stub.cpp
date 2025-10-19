// src/policy_lp_stub.cpp
#include "policy_solver.h"

PolicyLPSolver::PolicyLPSolver(const PolicySamples&,
                               const Transitions&,
                               const PolicyLPConfig&) {}

PolicyLPSolver::~PolicyLPSolver() = default;

PolicySolution PolicyLPSolver::solve() {
    PolicySolution s{};
    s.solved = false;
    s.objective = 0.0;
    return s;
}

void PolicyLPSolver::write_lp_file(const std::string&) const {
    // no-op
}

