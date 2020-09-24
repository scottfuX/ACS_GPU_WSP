#ifndef _BUPT_GLOBAL_CONFIG_H_
#define _BUPT_GLOBAL_CONFIG_H_

#include <unistd.h>


//==========================================GLOBAL CONFIG===============================================//
#define NN_SIZE                             32              //The default is 32.
#define MY_WARP_SIZE                        32
#define MAX_VISITED_SIZE                    1536            //Each bit represents whether a point is accessed or not. 
#define LOCAL_SEARCH_N                      32768           //在tsp_ls_gpu 里面local search共享内存数量 要不然大tsp会出错
#define UINT32_MAX_VAL                      4294967295      //4294967295
#define FLOAT_MAX_VAL                       1000000000      //FLT_MAX

//=============================================ALT CONFIG================================================//
//在使用 USE_HEUR_FUNC 的情况下 heuristic_matrix 是不需要的,此处仍需修改
#define USE_HEUR_FUNC                       0          //1: use get heuristic function\ 0:use heuristic matrix

//===========================================PRINT============================================//
#define PRINT_GS_COUNT                      1
#define PRINT_WORK_LENGTH                   0
#define PRINT_GS_CAND_LIST                  0

#define CLUSTERING_ITER_NUM                 100

#endif //_BUPT_GLOBAL_CONFIG_H_