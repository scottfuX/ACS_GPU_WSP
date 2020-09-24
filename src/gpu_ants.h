#ifndef GPU_ANTS
#define GPU_ANTS

#include "stopcondition.h"
#include "common.h"
#include "ants.h"


struct TSPData {
    const std::vector<std::vector<double>> &dist_matrix_;
    const std::vector<std::vector<double>> &heuristic_matrix_;
    const std::vector<std::vector<uint32_t>> &nn_lists_;
    const uint32_t dimension_;
};


/**
 * Runs the unmodified ACS with a full pheromone matrix.
 */
std::map<std::string, pj>
gpu_run_acs(TSPData &problem,
                 std::mt19937 &rng,
                 ACSParams &params,
                 int pher_mem_update_freq,
                 uint32_t threads_per_block,
                 StopCondition *stop_cond);


/**
 * This is the fastest version of the ACS in which ants' solutions are
 * constructed during a single kernel execution.
 */
std::map<std::string, pj>
gpu_run_acs_alt(TSPData &tsp_data,
             std::mt19937 &rng,
             ACSParams &params,
             int pher_mem_update_freq,
             uint32_t threads_per_block,
             StopCondition *stop_cond);


/**
 * Run the ACS with the selective pheromone memory (SPM)
 */
std::map<std::string, pj>
gpu_run_acs_spm(TSPData &problem,
                 std::mt19937 &rng,
                 ACSParams &params,
                 int pher_mem_update_freq,
                 uint32_t threads_per_block,
                 StopCondition *stop_cond);

/*
 * Run ACS with full pheromone matrix.
 */
std::map<std::string, pj>
gpu_run_acs_atomic(TSPData &problem,
                    std::mt19937 &rng,
                    ACSParams &params,
                    int pher_mem_update_freq,
                    uint32_t threads_per_block,
                    StopCondition *stop_cond);

/*
 * Modified by BUPT, 
 * faster than "gpu_run_acs_alt" version in most cases
 */
std::map<std::string, pj>
gpu_run_acs_wsp(TSPData &problem,
                    std::mt19937 &rng,
                    ACSParams &params,
                    int pher_mem_update_freq,
                    uint32_t threads_per_block,
                    StopCondition *stop_cond);

#endif
