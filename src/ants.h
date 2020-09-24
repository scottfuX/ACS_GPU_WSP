#ifndef ANTS_H
#define ANTS_H

#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}


template<typename T, typename R>
static void flatten(const std::vector<std::vector<T>> &matrix, std::vector<R> &out) {
    out.clear();
    for (auto &row : matrix) {
        for (auto el : row) {
            out.push_back((R)el);
        }
    }
}


template<typename T>
std::ostream& operator << (std::ostream &out, const std::vector<T> &vec) {
    for (auto el : vec) {
        out << el << " ";
    }
    return out;
}


extern bool is_valid_route(const std::vector<uint32_t> &route, uint32_t nodes_count);


struct Ant {
    uint32_t nodes_count_ = 0;
    double value_ = 0.0;
    uint32_t position_ = 0;
    std::vector<uint32_t> visited_;
    uint32_t visited_count_ = 0;
    std::vector<uint8_t> is_visited_;
};


struct ACSParams {
    uint32_t ants_count_ = 10;
    double beta_ = 3.0;
    double q0_ = 0.9;
    double rho_ = 0.2; 
    double phi_ = 0.01; // local pheromone update
    double initial_pheromone_ = 0.0;
    bool use_local_search_ = false;

	//modify by BUPT
    uint32_t sol_rec_freq_ = 0;

    uint32_t phmem_print_ = 0;
    uint32_t phmem_print_freq_ = 0;

    uint32_t gs_cand_list_size_ = 32;
    uint32_t gs_pub_cand_size_ = 32;

    //modify end

	//System Params add by huangzb.
	std::string outdir;
	std::string test_name;
};


#endif
