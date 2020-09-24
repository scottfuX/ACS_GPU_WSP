#include "gpu_phmem.h"


/*
 * Update pheromone values atomically using atomicCAS
 * instruction -- it is slow but we are sure that the update is
 * not lost in case of concurrent update.
 */
__device__ void update_pheromone_atomic(float * __restrict__ pheromone_matrix,
                                 uint32_t dimension,
                                 uint32_t u, uint32_t v,
                                 const float evap_ratio,
                                 const float pheromone_delta,
                                 int *mutexes) {
    const auto evap = 1 - evap_ratio;
    const auto delta = evap_ratio * pheromone_delta;

    float *v_ptr = pheromone_matrix + u * dimension + v;
    auto old = *v_ptr;
    auto val = old * evap + delta;
    while (atomicCAS(v_ptr, old, val) != old) {
        old = *v_ptr;
        val = old * evap + delta;
    }

    float *u_ptr = pheromone_matrix + v * dimension + u;
    old = *u_ptr;
    val = old * evap + delta;
    while (atomicCAS(u_ptr, old, val) != old) {
        old = *u_ptr;
        val = old * evap + delta;
    }
}

