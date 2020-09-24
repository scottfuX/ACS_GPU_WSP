#ifndef TSP_LS_GPU_H
#define TSP_LS_GPU_H

#include <cstdint>

//////////////////////////Modify by BUPT//////////////////////////
//#define NN_SIZE  512 // main.cc
#include "bupt_global_config.h"
/////////////////////////////Modify end////////////////////////////


__global__ 
void opt2_cs(const float * __restrict__ dist_matrix,
                uint32_t dimension,
                uint32_t *routes,
                float *routes_len,
                uint32_t route_size,
                const uint32_t * __restrict__ nn_lists,
                int *route_node_indices,
                uint32_t num_cluster,
                uint32_t* cluster_nn_start,
                uint32_t* cluster_nn_len,
                uint32_t* nearest_indices );
/**
 * GPU version of various local search algorithms for the TSP problem
 */

/**
  GPU version of the 2-opt heuristic with additional improvements in the form of
  dont_look_ bits and changes restricted to nearest neighbours of each node.
  Based on the ACOTSP source code by T. Stuzle
**/
__global__ 
void opt2(const float * __restrict__ dist_matrix,
                     uint32_t dimension,
                     uint32_t *routes,
                     float *routes_len,
                     uint32_t route_size,
                     const uint32_t * __restrict__ nn_lists,
                     int *cust_pos_);


__global__ 
void opt3(const float * __restrict__ dist_matrix,
          uint32_t dimension,
          uint32_t *routes,
          float *routes_len,
          uint32_t route_size,
          const uint32_t * __restrict__ nn_lists,
          int *route_node_indices);

#endif
