#include "ants.h"

#include <iostream>

using namespace std;

bool is_valid_route(const std::vector<uint32_t> &route, uint32_t nodes_count) {
    std::unordered_set<uint32_t> visited{ route.begin(), route.end() };
    if (visited.size() != nodes_count) {
        std::cerr << "Some nodes are duplicated, count: " << route.size() - visited.size() << std::endl;
    }
    const uint32_t max_el = *max_element(route.begin(), route.end());
    if (max_el >= nodes_count) {
        std::cerr << "Invalid node: " << max_el << std::endl;
    }
    return visited.size() == nodes_count
        && *max_element(route.begin(), route.end()) < nodes_count;
}
