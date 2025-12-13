//socp_config.hpp
#pragma once
#include <string>

namespace socp {

struct ScsSettingsConfig {
    double eps_abs = 1e-12;
    double eps_rel = 1e-12;
    int    max_iters = 80000;
    double time_limit_secs = 50.0;
    bool   verbose = true;
};

struct SolverConfig {
    ScsSettingsConfig scs;
    bool debug_on = true;
    bool debug_relax = false; // feasibility probe
    int  omp_threads = 0;     // 0=use env/default
};

} // namespace socp
