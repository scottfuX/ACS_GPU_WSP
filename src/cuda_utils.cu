#include "cuda_utils.h"

/*
__device__ inline float atomicCAS(float* address, float compare, float val) {
    auto r = atomicCAS((int *)address, __float_as_int(compare), __float_as_int(val));
    return __int_as_float(r);
}*/

void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ __forceinline__ void swape(uint32_t* a,uint32_t* b){
    uint32_t t = (*a);
    (*a) = (*b);
    (*b) = t;
}

__device__ void cudaOddEvenSortAccShared(float* data,int len,int* in){
    int tid = threadIdx.x;
    for(int i=tid; i < len; i+=blockDim.x) {
        in[i] = i;
    }
 
    for(int i=1;i<=len;i++){
         if(i%2 == 0){ //even
            int idx = tid * 2 + 1;
            for(int j=idx ;j < len;j+=blockDim.x * 2){
                if(j+1 < len){
                    int a= in[j];
                    int b=in[j+1];
                    if(data[a] < data[b] ){
                        int t = in[j+1];
                        in[j+1] = in[j]; 
                        in[j] = t;
                    }
                }    
            }
            __syncthreads();
        }
        else{//odd
            int idx = tid * 2;
            for(int j=idx ;j < len;j += blockDim.x * 2){
                if(j+1 < len){
                    int a= in[j];
                    int b=in[j+1];
                    if(data[a] < data[b] ){
                        int t = in[j+1];
                        in[j+1] = in[j]; 
                        in[j] = t;
                    }
                }    
            }
            __syncthreads();
        }
    }
}



