#ifndef GPU_ACS_H
#define GPU_ACS_H

#include "cuda_utils.h"

struct GPU_ACSParams : public Managed {
    uint32_t ants_count_ = 10;
    float beta_ = 3.0;
    float q0_ = 0.9;
    float rho_ = 0.2; 
    float phi_ = 0.01; // local pheromone update
    float initial_pheromone_ = 0.0;

    //modify by BUPT
    uint32_t sol_rec_freq_ = 0; 

    uint32_t phmem_print_ = 0;
    uint32_t phmem_print_freq_ = 0;

    uint32_t gs_cand_list_size_ = 32;
    uint32_t gs_pub_cand_size_ = 32;

    //modify end

    GPU_ACSParams &operator=(const ACSParams &other) {
        ants_count_ = other.ants_count_;
        beta_ = (float)other.beta_;
        q0_ = (float)other.q0_;
        rho_ = (float)other.rho_;
        phi_ = (float)other.phi_;
        initial_pheromone_ = (float)other.initial_pheromone_;
        
        //modify by BUPT
        sol_rec_freq_ = (uint32_t)other.sol_rec_freq_;

        phmem_print_ = (uint32_t)other.phmem_print_;
        phmem_print_freq_ = (uint32_t)other.phmem_print_freq_;

        gs_cand_list_size_ = (uint32_t)other.gs_cand_list_size_;
        gs_pub_cand_size_ = (uint32_t)other.gs_pub_cand_size_;

        //modify end
        return *this;
    }
};


class ACSGPU {
};

#endif
