# ACS_GPU_WSP

This is a fast GPU version of ACS in the way of producer and consumer by using special warp to solve the Traveling Salesman Problem（TSP）.
It is developed by research group of Compute Intelligence and Visualization,BUPT.

Thanks for the code support of Skinderowicz, Rafał.
This code is modified based on [GPUbasedACS](https://github.com/RSkinderowicz/GPUBasedACS) of Skinderowicz.



# Compilation Environment 

This code is tested in the following environment:

    Intel i7-7700K 4.20GHz CPU
    RTX2080ti GPU
    Ubuntu 16.04.6 LTS 
    CUDA 10.0
    GCC v5.4



# Building

To compile:

    make

If there are no errors in the compilation, the "gpuants" executable will be generated.


Note that if it doesn't work, you may need to adjust makefil's GPU architecture parameters, for example:

    -gencode arch=compute_50,code=sm_70


# Operating parameters

- --alg: is the name of the algorithm to run.
- --iter: is the number of the ACS iterations to execute,
- --test: is the path to the TSP data instance,
- --outdir: is the path to a directory in which a file with results should be created. Results are saved in JSON format (*.js)

- --gs_cand_size: set the number of dynamic candidate sets (GS_List) for each city, default: 32

Valid values for the --alg argument:
- acs_gpu_wsp: a fast GPU version of ACS in the way of producer and consumer by using special wap


# Running

If everything goes well, you can do it with the following example:

    ./gpuants --test tsp/usa13509.tsp --outdir results --alg acs_gpu_wsp --iter 100 


You can get most of the parameters with the following command:

    ./gpuants --help

