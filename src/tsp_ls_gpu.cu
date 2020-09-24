
#include <cassert>
#include "tsp_ls_gpu.h"
#include "cuda_utils.h"

/* Local search */

/* Auxilary kernel used in 2-opt LS impl. */
__device__ void reverse_subroute_inner(uint32_t *route, int *pos, int beg, int end) {
    // Reverse subroute fragment from beg to end
    int i = pos[beg];
    int j = pos[end];
    assert(i <= j);

    int tid = threadIdx.x;
    const int threads = blockDim.x;
    
    while (i + tid < j - tid) {
        auto k = i + tid;
        auto l = j - tid;

        auto c1 = route[k];
        auto c2 = route[l];

        route[k] = c2;
        route[l] = c1;
        pos[c1] = l;
        pos[c2] = k;

        i += threads;
        j -= threads;
    }
    __syncthreads();
}


/* Auxilary kernel used in 2-opt LS impl. */
__device__ void reverse_subroute_outer(uint32_t *route, const int route_size, int *pos, int beg, int end) {
    const int N = route_size;
    const int i = pos[beg];
    const int j = pos[end];
    assert( i >= j );
    //assert( (i - j) >= (N - 1 - i + j) );

    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    uint32_t c1, c2;
    int ii = 0;
    int jj = j - i + N;
    while (ii + tid <= jj/2) {
        int k = (i + (ii + tid)) % N;
        int l = (j - (ii + tid)) >= 0 ? (j - (ii + tid)) : (j - (ii + tid) + N);

        c1 = route[k];
        c2 = route[l];

        route[k] = c2;
        route[l] = c1;
        pos[c1] = l;
        pos[c2] = k;
        ii += threads;
    }
    __syncthreads();
}


__device__ void reverse_subroute(uint32_t *route, const int route_size,
                                 int *pos, int beg, int end) {
    int i = pos[beg];
    int j = pos[end];
    if (i <= j) {
        reverse_subroute_inner(route, pos, beg, end);
    } else {
        reverse_subroute_outer(route, route_size, pos, beg, end);
    }
}


__device__ void reverse_subroute_flip(uint32_t *route,
                                     const int route_size,
                                     int *pos,
                                     int beg, int end) {
    const int N = route_size;
    int i = pos[beg];
    int j = pos[end];
    if (i <= j) {
        if ((j - i) <= N/2 + 1) {
            reverse_subroute_inner(route, pos, beg, end);
        } else {
            int end_next = route[(j + 1) % N];
            int beg_prev = route[ i > 0 ? i-1 : N-1 ];
            reverse_subroute(route, route_size, pos, end_next, beg_prev);
        }
    } else {
        if ((i - j) <= N/2 + 1) {
            reverse_subroute_outer(route, route_size, pos, beg, end);
        } else {
            int end_next = route[(j + 1) % N];
            int beg_prev = route[ i > 0 ? i-1 : N-1 ];
            reverse_subroute(route, route_size, pos, end_next, beg_prev);
        }
    }
}


template<typename T>
__device__ void swap(T &a, T &b) {
    auto temp = a;
    a = b;
    b = temp;
}


__device__ float calc_route_len(uint32_t *route, 
                     uint32_t route_size,
                     const float * __restrict__ dist_matrix
                     ) {
    #define get_distance1( a, b ) dist_matrix[ (a) * route_size + (b) ]

    float len = get_distance1( route[ route_size - 1 ], route[ 0 ] );

    for (uint32_t i = 1; i < route_size; ++i) {
        len += get_distance1( route[i-1], route[i] );
    }
    return len;
}



__device__ bool bit_array_get(volatile const uint32_t *array, int index) {
    return array[ index >> 5 ] & (1 << (index & 31));
}


__device__ void bit_array_set(volatile uint32_t *array, int index) {
    array[ index >> 5 ] |= 1 << (index & 31);
}

__device__ void bit_array_clear(volatile uint32_t *array, int index) {
    array[ index >> 5 ] &= ~(1 << (index & 31));
}

/*
  GPU version of the 2-opt heuristic with additional improvements in the form of
  dont_look_ bits and changes restricted to nearest neighbours of each node.
*/
__global__ void opt2(const float * __restrict__ dist_matrix,
                     uint32_t dimension,
                     uint32_t *routes,
                     float *routes_len,
                     uint32_t route_size,
                     const uint32_t * __restrict__ nn_lists,
                     int *route_node_indices
                     ) {
    #define get_distance( a, b ) dist_matrix[ (a) * dimension + (b) ]

    assert( blockDim.x == 32 ); // assuming single warp

    int improvements_count = 0;
    bool improvement = true;
    const int N = route_size;
    const uint32_t route_id = blockIdx.x;
    uint32_t *route = routes + route_id * dimension;

    //const int N_MAX = 8192 / 32;
    const int N_MAX = LOCAL_SEARCH_N / 32;
    volatile __shared__ uint32_t dont_look[N_MAX];

    int *cust_pos_ = route_node_indices + route_id * dimension;

    for (uint32_t i = 0; i < route_size; ++i) {
        cust_pos_[ route[i] ] = i;
        bit_array_clear(dont_look, i);
    }
#ifndef NDEBUG
    float start_len = calc_route_len(route, route_size, dist_matrix);
#endif

    int cust_i = 0;
    int offset = (int)(route[0]);

    auto const tid = threadIdx.x;
    __shared__ uint32_t nn_cache[ NN_SIZE ];
    __shared__ float c1_c2_dist_cache[ NN_SIZE ];

    while (improvement) {
        improvement = false;

        int c1, c2;
        for (cust_i = 0; cust_i < N; ++cust_i) {
            // A bit of randomness
            //c1 = (cust_i + blockIdx.x + offset) % N;
            c1 = cust_i;

            if (bit_array_get(dont_look, c1)) {
                continue;
            }
            int pos_c1 = cust_pos_[c1];
            int c1_next = route[(pos_c1+1) % N];
            int c2_next;
            int c1_prev, c2_prev;
            float radius = get_distance(c1, c1_next);
            bool improve_node = false;
            int h1, h2, h3, h4;

            // look through c1's nearest neighbours list
            //auto const nn_list = nn_lists + c1 * NN_SIZE;
            auto const nn_list = nn_lists + c1 * NN_SIZE;

            for (int j = tid; j < NN_SIZE; j += blockDim.x) {
                c2 = nn_list[j];
                nn_cache[j] = c2;
                c1_c2_dist_cache[j] = get_distance(c1, c2);
            }

            // Look in parallel for possible improvement
            for (int j = tid; j < NN_SIZE; j += blockDim.x) {
                c2 = nn_cache[j];
                float c1_c2_dist = c1_c2_dist_cache[j];

                if (c1_c2_dist < radius) { // improvement possible
                    c2_next = route[ (cust_pos_[c2]+1) % N ];
                    improve_node = (c1_c2_dist + get_distance(c1_next, c2_next)) 
                                 < (radius     + get_distance(c2, c2_next));
                } 
            }
            //__syncthreads();

            if (__any(improve_node)) {
                // Find idx of the first thread which found improvement
                auto const idx = 31 - __clz( __ballot(improve_node) );
                c2 = warp_bcast(c2, idx); // Get customer id
                c2_next = route[ (cust_pos_[c2]+1) % N ];
                h1 = c1;
                h2 = c1_next;
                h3 = c2;
                h4 = c2_next;
                improve_node = true;
            }
            //__syncthreads();
            if (improve_node == false) {
                auto c1_pos = cust_pos_[c1];
                c1_prev = (c1_pos > 0) ? route[ c1_pos - 1 ] : route[N-1];
                radius = get_distance(c1_prev, c1);

                for (int j = tid; j < NN_SIZE; j += blockDim.x) {
                    c2 = nn_cache[j];
                    float c1_c2_dist = c1_c2_dist_cache[j];
                    //选择 其中一个替换 c1
                    if (c1_c2_dist < radius) { // improvement possible
                        auto c2_pos = cust_pos_[c2];
                        c2_prev = (c2_pos > 0) ? route[ c2_pos - 1 ] 
                                               : route[ N - 1 ];

                        if (c2_prev != c1 && c1_prev != c2) {
                            improve_node = (c1_c2_dist + get_distance(c1_prev, c2_prev))
                                         < (radius     + get_distance(c2_prev, c2));
                        }
                    } 
                }
                //__syncthreads();
                if (__any(improve_node)) {
                    auto const idx = 31 - __clz( __ballot(improve_node) );
                    c2 = warp_bcast(c2, idx);
                    c2_prev = (cust_pos_[c2] > 0) ? route[ cust_pos_[c2]-1 ] : route[N-1];
                    h1 = c1_prev;
                    h2 = c1;
                    h3 = c2_prev;
                    h4 = c2;
                    improve_node = true;
                }
                //__syncthreads();
            }

            if (improve_node) {
                improvement = true;
                ++improvements_count;

                if (cust_pos_[h3] < cust_pos_[h1]) {
                    swap(h1, h3);
                    swap(h2, h4);
                }
                assert( cust_pos_[h2] < cust_pos_[h3] );

                if (cust_pos_[h3] - cust_pos_[h2] < N/2 + 1) {
                    // reverse inner part from cust_pos[h2] to cust_pos[h3]
                    reverse_subroute_inner(route, cust_pos_, h2, h3);
                } else {
                    // reverse outer part from cust_pos[h4] to cust_pos[h1]
                    if ( cust_pos_[h4] > cust_pos_[h1] ) {
                        reverse_subroute_outer(route, route_size, cust_pos_, h4, h1);
                    } else {
                        reverse_subroute_inner(route, cust_pos_, h4, h1);
                    }
                }
                bit_array_clear(dont_look, h1);
                bit_array_clear(dont_look, h2);
                bit_array_clear(dont_look, h3);
                bit_array_clear(dont_look, h4);
            } else { // no improvement
                bit_array_set(dont_look, c1);
            }
        }
    }
    float final_len = calc_route_len(route, route_size, dist_matrix);
    routes_len[ blockIdx.x ] = final_len;
#ifndef NDEBUG
    assert( final_len <= start_len );
#endif

}


__global__ 
void opt3(const float * __restrict__ dist_matrix,
          uint32_t dimension,
          uint32_t *routes,
          float *routes_len,
          uint32_t route_size,
          const uint32_t * __restrict__ nn_lists,
          int *route_node_indices
          ) {
    //int improvements_count = 0;
    bool improvement = true;
    const int N = route_size;

    const uint32_t route_id = blockIdx.x;
    uint32_t *route = routes + route_id * dimension;

    //const int N_MAX = 8192 / 32;
    const int N_MAX = LOCAL_SEARCH_N / 32;
    volatile __shared__ uint32_t dont_look[N_MAX];
    
    int *cust_pos_ = route_node_indices + route_id * dimension;

    for (uint32_t i = 0; i < route_size; ++i) {
        cust_pos_[ route[i] ] = i;
        bit_array_clear(dont_look, i);
    }

    #ifndef NDEBUG
    float start_len = calc_route_len(route, route_size, dist_matrix);
    #endif

    int cust_i = 0;

    /*
    We restric our attention to 6-tuples of cities (a, b, c, d, e, f) 

    Each 3-opt move results in removal of the edges (a,b) (c,d) (e,f)

    There are two possibilities for new edges to connect the route together again
    
    A) edges (a,d) (c,e) (f,b) are created
    B) edges (a,d) (e,b) (c,f) are created

    Ad. A)

    This move needs reversal of subroutes (f..a) and (d..e)

    Ad. B)

    This move needs reversal of three subroutes: (f..a) (b..c) (d..e)

    There is also 2-opt move possible

    edges (a,b) (c,d) are replaced with (a,c) (b,d)

    This move needs reversal of subroute (b..c) or (d..a)

    We need to check only tuples in which d(a,b) > d(a,d) and 

    d(a,b) + d(c,d) > d(a,d) + d(c,e)

    or 

    d(a,b) + d(c,d) > d(a,d) + d(e,b)
    */
    int total_improvements = 0;
    while (improvement) {
        improvement = false;

        int a, b, c, d, e, f;

        for (cust_i = 0; cust_i < N; ++cust_i) {
            a = cust_i; // random_order_[cust_i];
            if (bit_array_get(dont_look, a)) {
                continue;
            }
            int pos_a = cust_pos_[a];
            b = route[(pos_a+1) % N];
            float radius = get_distance(a, b);
            float dist_a_b = radius;
            bool improve_node = false;

            // look for c in a's nearest neighbours list
            //auto const nn_list = nn_lists + a * NN_SIZE;
            auto const nn_list = nn_lists + a * NN_SIZE;

            // search for 2-opt move
            for (int j = 0; (improve_node == false) && (j < NN_SIZE); ++j) {
                c = nn_list[j];
                float dist_a_c = get_distance(a, c);
                if (dist_a_c < dist_a_b) { // 2-Opt possible
                    int pos_c = cust_pos_[c];
                    d = route[(pos_c + 1) % N];
                    float gain = (dist_a_b + get_distance(c,d)) -
                                  (dist_a_c + get_distance(b, d));
                    if (gain > 0) {
                        reverse_subroute_flip(route, route_size, cust_pos_, b, c);
                        bit_array_clear(dont_look, a);
                        bit_array_clear(dont_look, b);
                        bit_array_clear(dont_look, c);
                        bit_array_clear(dont_look, d);

                        improve_node = true;
                    }
                } else {
                    // the rest of neighbours are further than c
                    break ;
                }
            }
            int prev_a = route[ pos_a > 0 ? pos_a-1 : N-1 ];
            float dist_prev_a_a = get_distance(prev_a, a);
            for (int j = 0; (improve_node == false) && (j < NN_SIZE); ++j) {
                c = nn_list[j];
                float dist_a_c = get_distance(a, c);

                if (dist_a_c < dist_prev_a_a) { // 2-Opt possible
                    int pos_c = cust_pos_[c];
                    int prev_c = route[ pos_c > 0 ? pos_c - 1 : N-1 ]; // d is now a predecessor of c

                    if (prev_c == a || prev_a == c)
                        continue ;

                    float gain = (dist_prev_a_a + get_distance(prev_c, c)) -
                                  (dist_a_c + get_distance(prev_a, prev_c));

                    if (gain > 0) {
                        reverse_subroute_flip(route, route_size, cust_pos_, c, prev_a);
                        bit_array_clear(dont_look, prev_a);
                        bit_array_clear(dont_look, a);
                        bit_array_clear(dont_look, prev_c);
                        bit_array_clear(dont_look, c);
                        improve_node = true;
                    }
                } else {
                    // the rest of neighbours are further than c
                    break ;
                }
            }

            // search for 3-opt move
            for (int j = 0; (improve_node == false) && (j < NN_SIZE); ++j) {
                c = nn_list[j];
                int pos_c = cust_pos_[c];
                d = route[(pos_c + 1) % N];

                if (d == a) {
                    continue ;
                }
                
                float dist_a_d = get_distance(a, d);

                if (dist_a_d < dist_a_b) { // improvement possible -> now find e

                    //auto const nn_list_c = nn_lists + c * NN_SIZE;
                    auto const nn_list_c = nn_lists + c * NN_SIZE;
                    
                    // look for e in c's neighbours list
                    for (int k = 0; (improve_node == false) && (k < NN_SIZE); ++k) {
                        e = nn_list_c[k];
                        int pos_e = cust_pos_[e];
                        f = route[ (cust_pos_[e] + 1) % N ];
                        // node e has to lay between nodes c and a, i.e. a..c..e
                        if ( (f == a) ||
                             !( (pos_a < pos_c && (pos_c < pos_e || pos_e < pos_a)) ||
                                (pos_a > pos_c && (pos_c < pos_e && pos_e < pos_a)) ) )
                            continue;

                        // now check two possibilities
                        float dist_c_d = get_distance(c, d);
                        float dist_e_f = get_distance(e, f);
                        
                        // A) edges (a,d) (c,e) (f,b) are created
                        float gain = dist_a_b + dist_c_d + dist_e_f -
                            (dist_a_d + get_distance(c, e) + get_distance(f, b));
                        if (gain > 0) {
                            // This move needs reversal of subroutes (f..a) and (d..e)
                            reverse_subroute(route, route_size, cust_pos_, d, e);
                            reverse_subroute(route, route_size, cust_pos_, f, a);

                            bit_array_clear(dont_look, a);
                            bit_array_clear(dont_look, b);
                            bit_array_clear(dont_look, c);
                            bit_array_clear(dont_look, d);
                            bit_array_clear(dont_look, e);
                            bit_array_clear(dont_look, f);

                            improve_node = true;
                            ++total_improvements;
                        }
                        // B) edges (a,d) (e,b) (c,f) are created
                        if (!improve_node) {
                            gain = dist_a_b + dist_c_d + dist_e_f -
                                (dist_a_d + get_distance(e, b) + get_distance(c, f));
                        }
                        if (!improve_node && gain > 0) {
                            // This move needs reversal of three subroutes: (f..a) (b..c) (d..e)
                            reverse_subroute(route, route_size, cust_pos_, f, a);
                            reverse_subroute(route, route_size, cust_pos_, b, c);
                            reverse_subroute(route, route_size, cust_pos_, d, e);

                            bit_array_clear(dont_look, a);
                            bit_array_clear(dont_look, b);
                            bit_array_clear(dont_look, c);
                            bit_array_clear(dont_look, d);
                            bit_array_clear(dont_look, e);
                            bit_array_clear(dont_look, f);

                            improve_node = true;
                            ++total_improvements;
                        }
                        /**
                         * Stop searching if edge (c,e) gets too long
                         */
                        if (!improve_node && (dist_a_b + dist_c_d) < get_distance(c, e))
                            break ;
                    }
                } else {
                    // each next c will be farther then current so there is no
                    // sense in further search
                    break;
                }
            }
            if (improve_node) {
                improvement = true;
            } else {
                int prev_a = pos_a > 0 ? route[pos_a-1] : route[(pos_a+1) % N];
                if (get_distance(prev_a, a) < dist_a_b) {
                    bit_array_set(dont_look, a);
                }
            }
        }
    }
    const float final_len = calc_route_len(route, route_size, dist_matrix);
    routes_len[ blockIdx.x ] = final_len;

    #ifndef NDEBUG
    assert(final_len <= start_len);
    if (total_improvements > 0) {
        printf( "Total improvements: %d"
                "\n\tstart_len: %.0f"
                "\n\tfinal_len: %.0f" 
                "\n\tgain: %.1f\n",
                total_improvements,
                start_len,
                final_len,
                (start_len - final_len) );
    }
    #endif
}
