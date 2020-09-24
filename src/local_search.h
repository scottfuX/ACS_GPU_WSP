#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#include <vector>
#include <cstdint>
#include <functional>

void opt2(std::vector<uint32_t> &route,
          const std::vector<std::vector<uint32_t>> &nn_lists,
          std::function<double (int, int)> get_distance,
          std::function<double (const std::vector<uint32_t>&)> calc_route_len
          );

/*
 * An implementation of 3-opt heuristic for the TSP problem with "don't look
 * bits"
 */
void opt3(std::vector<uint32_t> &route,
          const std::vector<std::vector<uint32_t>> &nn_lists,
          std::function<double (int, int)> get_distance,
          std::function<double (const std::vector<uint32_t>&)> calc_route_len
         );

#endif
