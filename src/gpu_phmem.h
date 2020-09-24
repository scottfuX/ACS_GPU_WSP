#ifndef GPU_PHMEM_H
#define GPU_PHMEM_H

#include "cuda_utils.h"
#include <cassert>
#include <iostream>


/*
 * Basic pheromone update. Updates the values both for (u, v) and (v, u).
 */
__device__ __forceinline__ void update_pheromone_symmetric(float * pheromone_matrix,
                                 uint32_t dimension,
                                 uint32_t u, uint32_t v,
                                 const float evap_ratio,
                                 const float delta) {
    auto *v_ptr = pheromone_matrix + u * dimension + v;
    auto val = *v_ptr * (1 - evap_ratio) + evap_ratio * delta;
    *v_ptr = val;
    pheromone_matrix[v * dimension + u] = val;
}


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
                                 int *mutexes);



template<typename Derived>
struct GenericMatrixPhmem {
    const size_t dimension_;
    float *cells_ = nullptr;

    GenericMatrixPhmem(size_t size) :
        dimension_(size)
    {}


    __device__ __host__ size_t size() const { return dimension_ * dimension_; }


    __device__ __forceinline__ float &get(int index) {
        assert(index >= 0 && index < (int)(dimension_ * dimension_));
        return cells_[index];
    }

    __device__ __forceinline__ float* get_all(int city){
        assert(city >= 0 && city < (int)(dimension_));
        return &cells_[city * dimension_];
    }


    __device__ __forceinline__ float &operator[](int index) {
        assert(index >= 0 && index < (int)(dimension_ * dimension_));
        return cells_[index];
    }


    __device__ __forceinline__ void update(uint32_t u, uint32_t v, float evap_ratio, float pheromone_delta) {
        derived().update_trail(u, v, evap_ratio, pheromone_delta);
    }


    __device__ __forceinline__ void update_lockfree(uint32_t u, uint32_t v, float evap_ratio, float pheromone_delta) {
        derived().update_trail_lockfree(u, v, evap_ratio, pheromone_delta);
    }


    __device__ __forceinline__ Derived& derived() {
        return *static_cast<Derived*>(this);
    }


    __host__ void copyTo(std::vector<float> &vec) {
        auto size = dimension_ * dimension_;
        vec.resize(size);
        gpuErrchk( cudaMemcpy(&vec.front(), cells_,
                              size * sizeof(float),
                              cudaMemcpyDeviceToHost) );
    }

    __host__ void copyTo(std::vector<float> &vec,bool &is_end,uint32_t start_index,uint32_t once_trans_size) {
        auto maxsize = dimension_ * dimension_;
        uint32_t size = once_trans_size;
        is_end = false;
        if(start_index + once_trans_size >= maxsize ){
            is_end = true;
            size = maxsize - start_index ;
        }
        vec.resize(size);
        gpuErrchk( cudaMemcpy(&vec.front(), cells_ + start_index,
                              size * sizeof(float),
                              cudaMemcpyDeviceToHost) );     
    }
};


struct MatrixPhmemDevice : public GenericMatrixPhmem<MatrixPhmemDevice> {

    MatrixPhmemDevice(size_t size) 
        : GenericMatrixPhmem<MatrixPhmemDevice>(size),
          is_memory_owner_(true)
    {
        gpuErrchk( cudaMalloc((void **)&cells_,
                              sizeof(float) * dimension_ * dimension_) );
    }

    MatrixPhmemDevice(MatrixPhmemDevice &rhs) :
        GenericMatrixPhmem<MatrixPhmemDevice>( rhs.dimension_ ),
        is_memory_owner_( false )
    {
        cells_ = rhs.cells_;
    }

    virtual ~MatrixPhmemDevice() {
        release();
    }


    __host__ void release() {
        if (is_memory_owner_) {
            gpuErrchk( cudaFree(cells_) );
            cells_ = nullptr;
        }
    }

    __device__ __forceinline__ void update_trail(uint32_t u, uint32_t v, float evap_ratio, float pheromone_delta) {
        update_pheromone_symmetric(cells_, dimension_, u, v, evap_ratio, pheromone_delta);
    }

    __device__ __forceinline__ void update_trail_lockfree(uint32_t u, uint32_t v, float evap_ratio, float pheromone_delta) {
        update_pheromone_symmetric(cells_, dimension_, u, v, evap_ratio, pheromone_delta);
    }


    bool is_memory_owner_ = false;
};


/**
 * This is a slower version of the MatrixPhmemDevice which uses atomic CAS to
 * ensure that concurrent changes to the pheromone values are not lost.
 */
struct MatrixPhmemAtomic : public GenericMatrixPhmem<MatrixPhmemAtomic> {

    MatrixPhmemAtomic(size_t size)
        : GenericMatrixPhmem<MatrixPhmemAtomic>(size),
          is_memory_owner_(true)
    {
        std::cout << "Allocating memory" << std::endl;
        gpuErrchk( cudaMalloc((void **)&cells_,
                              sizeof(float) * dimension_ * dimension_) );
        gpuErrchk( cudaMalloc((void **)&mutexes_,
                              sizeof(int) * (dimension_ + 1)) );
        gpuErrchk( cudaMemset((void *)mutexes_, 0,
                              sizeof(int) * (dimension_ + 1)) );
        gpuErrchk( cudaDeviceSynchronize() );
    }


    MatrixPhmemAtomic(MatrixPhmemAtomic &rhs) :
        GenericMatrixPhmem<MatrixPhmemAtomic>( rhs.dimension_ ),
        is_memory_owner_( false )
    {
        cells_ = rhs.cells_;
        mutexes_ = rhs.mutexes_;
    }

    virtual ~MatrixPhmemAtomic() {
        release();
    }

    __device__ void update_trail(uint32_t u, uint32_t v, float evap_ratio, float pheromone_delta) {
        auto a = ( u < v ) ? u : v;
        auto b = ( u < v ) ? v : u;

        assert( mutexes_ != nullptr );

        lock( &mutexes_[a] );
        lock( &mutexes_[b] );

        const auto evap = 1 - evap_ratio;
        const auto delta = evap_ratio * pheromone_delta;

        float *v_ptr = cells_ + u * dimension_ + v;
        auto old = *v_ptr;
        auto val = old * evap + delta;
        atomicCAS(v_ptr, old, val);

        float *u_ptr = cells_ + v * dimension_ + u;
        old = *u_ptr;
        val = old * evap + delta;
        atomicCAS(u_ptr, old, val);

        assert( cells_[u * dimension_ + v] == cells_[v * dimension_ + u] );

        unlock( &mutexes_[a] );
        unlock( &mutexes_[b] );
    }


    __device__ void update_trail_lockfree(uint32_t u, uint32_t v, float evap_ratio, float pheromone_delta) {
        ::update_pheromone_atomic( cells_, dimension_, u, v, evap_ratio,
                                   pheromone_delta, mutexes_ );
    }

    __host__ void release() {
        if (is_memory_owner_) {
            gpuErrchk( cudaFree(cells_) );
            cells_ = nullptr;
            gpuErrchk( cudaFree(mutexes_) );
            mutexes_ = nullptr;
        }
    }


    __device__ void lock(int *mutex) {
        while( atomicCAS(mutex, 0, 1) != 0 )
            ;
    }

    __device__ void unlock(int *mutex) {
        atomicExch(mutex, 0); 
    }

    bool is_memory_owner_ = false;
    int* mutexes_ = nullptr;
};


/**
 * This is the GPU version of the Selective Pheromone Memory (SPM) which can be
 * used instead of the full pheromone matrix as impl. in MatrixPhmemDevice
 */
struct SelectivePhmem {

    typedef float T;
    typedef T value_type;
    typedef int index_type;

    enum { capacity_ = 8 };
    enum { InvalidIndex = 0xFFFFu };


    struct Trails {
        value_type pheromone_[capacity_ + 1];
        index_type indices_[capacity_ + 1];
        int head_;

        void fill_indices(index_type value) {
            for (int i = 0; i < capacity_; ++i) { indices_[i] = value; }
        }

        __device__ void push(index_type index, value_type pheromone) {
            /* Assumming capacity_ is a power of 2 */
            auto old_head = atomicAdd(&head_, 1);
            auto new_head = (old_head + 1) & (capacity_ - 1);
            indices_[new_head] = index;
            pheromone_[new_head] = pheromone;
        }
    };

    int size_;
    T default_pheromone_;
    Trails *trails_;
    bool is_symmetric_;
    int mutex_;
    //int mutexes_[10000] = {};


    SelectivePhmem(int msize = 0, int bucket_size = 8, bool is_symmetric=true) :
        size_(msize), 
        is_symmetric_(is_symmetric),
        trails_(nullptr),
        mutex_(0)
    {
        std::cout << "Constructing SelectivePhmem" << std::endl;
        //gpuErrchk( cudaMallocManaged(&trails_, sizeof(Trails) * size_) );
        //gpuErrchk( cudaMemset((void *)trails_, 0, sizeof(Trails) * size_) );
        /*trails_ = new Trails[size_];*/
        /*for (int i = 0; i < msize; ++i) {*/
            /*trails_[i].fill_indices(InvalidIndex);*/
        /*}*/
        std::cout << "Done." << std::endl;
    }


    void init() {
        trails_ = new Trails[size_];
        for (int i = 0; i < size_; ++i) {
            trails_[i].fill_indices(InvalidIndex);
        }
        std::cout << "Done." << std::endl;
    }


    ~SelectivePhmem() {
        //gpuErrchk( cudaFree(trails_) );
        //delete [] trails_;
    }

    void set_default_pheromone(T value) {
        default_pheromone_ = value;
    }

    __device__ T get(uint32_t u, uint32_t v) {
        assert(u < size_ && v < size_);

        index_type *node_ids = trails_[u].indices_;
        for (int i = 0; i < capacity_; ++i) {
            if (node_ids[i] == v) {
                return trails_[u].pheromone_[i];
            }
        }
        return default_pheromone_;
    }


    // Use whole warp to get a single value
    // Assuming that number of trails is <= warp size
    __device__ T warp_get(int u, int v) {
        assert(u < size_ && v < size_);

        const auto loc_id = threadIdx.x;
        index_type my_id = loc_id < capacity_ ? trails_[u].indices_[loc_id] : InvalidIndex;
        const auto mask = __ballot(my_id == v);
        if (mask != 0) {
            const auto idx = __ffs(mask) - 1;
            assert(idx < capacity_);
            return trails_[u].pheromone_[idx];
        }
        return default_pheromone_;
    }


    __device__ void set_helper(uint32_t u, uint32_t v, T value) { 
        assert(u < size_ && v < size_);
        
        index_type *node_ids = trails_[u].indices_;
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (node_ids[i] == v) {
                trails_[u].pheromone_[i] = value;
                return ;
            }
        }
        // Else replace eldest element with new entry
        trails_[u].push(v, value);
    }


    __device__ void set(uint32_t i, uint32_t j, T value) { 
        set_helper(i, j, value);
        if (is_symmetric_) {
            set_helper(j, i, value);
        }
    }

    __device__ void warp_set_helper(int u, int v, T value) { 
        assert(u < size_ && v < size_);
        
        const auto loc_id = threadIdx.x;
        auto *node_ids = trails_[u].indices_;
        auto my_id = loc_id < capacity_ ? node_ids[loc_id] : InvalidIndex;

        const auto mask = __ballot(my_id == v);
        if (loc_id == 0) {
            if (mask != 0) {
                auto idx = __ffs(mask) - 1;
                trails_[u].pheromone_[idx] = value;
            } else {
                // Else replace eldest element with new entry
                trails_[u].push(v, value);
            }
        }
    }


    __device__ void warp_update_helper(int u, int v, T evap_ratio, T delta) { 
        assert(u < size_ && v < size_);
        
        const auto loc_id = threadIdx.x;
        auto &trail = trails_[u];
        auto *node_ids = trail.indices_;
        auto my_id = loc_id < capacity_ ? node_ids[loc_id] : InvalidIndex;

        /* Find which thread has 'v' entry */
        const auto mask = __ballot(my_id == v);
        if (loc_id == 0) {
            if (mask != 0) { /* There is one */
                auto idx = __ffs(mask) - 1;
                auto &pher = trail.pheromone_[idx];
                pher = pher * (1 - evap_ratio) + evap_ratio * delta;
            } else { /* No v was found */
                // Else replace eldest element with new entry
                trail.push(v, (1 - evap_ratio) * default_pheromone_ + evap_ratio * delta);
            }
        }
    }


    __device__ void unsafe_warp_update(int u, int v, T evap_ratio, T value) { 
        warp_update_helper(u, v, evap_ratio, value);
        warp_update_helper(v, u, evap_ratio, value);
    }


    __device__ T default_pheromone() const { 
        return default_pheromone_; 
    }


    __device__ __forceinline__ index_type *get_indices(uint32_t node) {
        assert(node < size_);
        return trails_[node].indices_;
    }


    __device__ __forceinline__ value_type *get_pheromone(uint32_t node) {
        assert(node < size_);
        return trails_[node].pheromone_;
    }


    __device__ uint32_t capacity() const { return (uint32_t)capacity_; }


    __device__ void lock() {
        while( atomicCAS(&mutex_, 0, 1) != 0 )
            ;
    }

    __device__ void unlock() {
        atomicExch(&mutex_, 0); 
    }
};

#endif
