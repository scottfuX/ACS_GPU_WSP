#include "local_search.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <iostream>

using namespace std;

extern std::mt19937 g_rng;

void gen_random_permutation(int n, std::vector<int> &perm_vec) {
    perm_vec.clear();
    for (int i = 0; i < n; ++i) {
        perm_vec.push_back(i);
    }
    std::uniform_int_distribution<> random(0, std::numeric_limits<int>::max());
    // random shuffle elements
    // randomly chose element from 0..i-1 and swap it with the i-th element
    // which won't further change
    for (int i = n-1; i >= 1; --i) {
        int j = random(g_rng) % i;
        swap(perm_vec[i], perm_vec[j]);
    }
}


static void reverse_subroute_inner(std::vector<uint32_t> &route, std::vector<int> &pos, int beg, int end) {
    int i = pos[beg];
    int j = pos[end];
    assert(i <= j);
    uint32_t c1, c2;
    while (i < j) {
        c1 = route[i];
        c2 = route[j];
        route[i] = c2;
        route[j] = c1;
        pos[c1] = j;
        pos[c2] = i;
        ++i;
        --j;
    }
}


static void reverse_subroute_outer(std::vector<uint32_t> &route, std::vector<int> &pos, int beg, int end) {
    const int N = (int)route.size();
    int i = pos[beg];
    int j = pos[end];
    //assert(i >= j);
    uint32_t c1, c2;

    int mid = (N - i) + j + 1;
    mid /= 2;

    for (int k = 0; k < mid; ++k) {
        c1 = route[i];
        c2 = route[j];
        route[i] = c2;
        route[j] = c1;
        pos[c1] = j;
        pos[c2] = i;
        i = (i + 1) % N;
        j = (j > 0 ? j-1 : N - 1);
    }
}

/*
a b c d e f g
....|.....|..
a g c d e f b


a b c d e f g
......|.....|
c b a d e f g
*/

static void reverse_subroute_outer2(std::vector<uint32_t> &route,
        std::vector<int> &pos, int beg, int end) {
    const int N = (int)route.size();
    const int i = pos[beg];
    const int j = pos[end];

    /*if (i > j) {
        assert( N/2 + 1 >= (N - i + j) );
    } else {
        assert( N/2 + 1 >= (j - i) );
    }*/

    uint32_t c1, c2;
    auto ii = 0;
    auto const jj = j - i + N;
    while (ii <= jj/2) {
        auto k = (i + ii) % N;
        auto l = (j - ii) >= 0 ? (j - ii) : (j - ii + N);
        c1 = route[k];
        c2 = route[l];
        route[k] = c2;
        route[l] = c1;
        pos[c1] = l;
        pos[c2] = k;
        ++ii;
    }
}


static void reverse_subroute(std::vector<uint32_t> &route,
                             std::vector<int> &pos,
                             int beg, int end) {
    int i = pos[beg];
    int j = pos[end];
    if (i <= j) {
        reverse_subroute_inner(route, pos, beg, end);
    } else {
        reverse_subroute_outer(route, pos, beg, end);
    }
}


static void reverse_subroute_flip(std::vector<uint32_t> &route,
                                  std::vector<int> &pos,
                                  int beg, int end) {
    const int N = (int)route.size();
    int i = pos[beg];
    int j = pos[end];
    if (i <= j) {
        if ((j - i) <= N/2 + 1) {
            reverse_subroute_inner(route, pos, beg, end);
        } else {
            int end_next = route[(j + 1) % N];
            int beg_prev = route[ i > 0 ? i-1 : N-1 ];
            reverse_subroute(route, pos, end_next, beg_prev);
        }
    } else {
        if ((i - j) <= N/2 + 1) {
            reverse_subroute_outer(route, pos, beg, end);
        } else {
            int end_next = route[(j + 1) % N];
            int beg_prev = route[ i > 0 ? i-1 : N-1 ];
            reverse_subroute(route, pos, end_next, beg_prev);
        }
    }
    /*
    const int N = (int)route.size();
    int i = pos[beg];
    int j = pos[end];
    auto const half = N / 2 + N % 2;

    if (i <= j) {
        if ((j - i) <= half) {
            reverse_subroute_inner(route, pos, beg, end);
        } else {
            j = (j + 1) % N;
            i = (i > 0) ? i-1 : N-1;
            reverse_subroute_outer2(route, pos, route[j], route[i]);
        }
    } else { // i > j
        if ((i - j) >= half) {
            reverse_subroute_outer2(route, pos, route[i], route[j]);
        } else {
            j = (j + 1) % N;
            i = (i > 0) ? i-1 : N-1;
            reverse_subroute_inner(route, pos, route[j], route[i]);
        }
    }*/
}

void opt2(std::vector<uint32_t> &route,
          const std::vector<std::vector<uint32_t>> &nn_lists,
          std::function<double (int, int)> get_distance,
          std::function<double (const std::vector<uint32_t>&)> calc_route_len
          ) {
    int improvements_count = 0;
    bool improvement = true;
    const int N = route.size();

    static std::vector<uint8_t> dont_look_;
    static std::vector<int> cust_pos_;
    static std::vector<int> random_order_;

    dont_look_.resize((size_t)(N+1));
    cust_pos_.resize((size_t)(N+1));
    for (size_t i = 0; i < route.size(); ++i) {
        cust_pos_[ route[i] ] = i;
        dont_look_[i] = 0;
    }
    random_order_.reserve((size_t)(N+1));
    for (int i = 0; i < N; ++i) {
        random_order_[i] = i;
    }
    gen_random_permutation(N, random_order_);

#ifndef NDEBUG
    const double start_len = calc_route_len(route);
#endif
    int cust_i = 0;

    while (improvement) {
        improvement = false;

        int c1, c2;

        for (cust_i = 0; cust_i < N; ++cust_i) {
            c1 = random_order_[cust_i];
            if (dont_look_[c1] == 1) {
                continue;
            }
            int pos_c1 = cust_pos_[c1];
            int c1_next = route[(pos_c1+1) % N];
            int c2_next;
            int c1_prev, c2_prev;
            double radius = get_distance(c1, c1_next);
            bool improve_node = false;
            int h1, h2, h3, h4;

            // look through c1's nearest neighbours list
            auto const &nn_list = nn_lists.at(c1);
            const int NNCount = (int)nn_list.size();
            for (int j = 0; (improve_node == false) && (j < NNCount); ++j) {
                c2 = nn_list[j];

                double c1_c2_dist = get_distance(c1, c2);

                if (c1_c2_dist < radius) { // improvement possible
                    c2_next = route[ (cust_pos_[c2]+1) % N ];
                    double gain = (c1_c2_dist + get_distance(c1_next, c2_next))
                                - (radius + get_distance(c2, c2_next));
                    if (gain < 0) {
                        h1 = c1;
                        h2 = c1_next;
                        h3 = c2;
                        h4 = c2_next;
                        improve_node = true;
                    }
                } else {
                    // each next c2 will be farther then current c2 so there is no
                    // sense in checking c1_c2_dist < radius condition
                    break;
                }
            }
            if (improve_node == false) {
                c1_prev = (cust_pos_[c1] > 0) ? route[ cust_pos_[c1]-1 ] : route[N-1];
                radius = get_distance(c1_prev, c1);

                for (int j = 0; (improve_node == false) && (j < NNCount); ++j) {
                    c2 = nn_list[j];
                    double c1_c2_dist = get_distance(c1, c2);

                    if (c1_c2_dist < radius) { // improvement possible
                        c2_prev = (cust_pos_[c2] > 0) ? route[ cust_pos_[c2]-1 ] : route[N-1];

                        if (c2_prev == c1 || c1_prev == c2) {
                            continue;
                        }
                        double gain = (c1_c2_dist + get_distance(c1_prev, c2_prev))
                                      - (radius + get_distance(c2_prev, c2));
                        if (gain < 0) {
                            h1 = c1_prev;
                            h2 = c1;
                            h3 = c2_prev;
                            h4 = c2;
                            improve_node = true;
                        }
                    } else {
                        // each next c2 will be farther then current c2 so there is no
                        // sense in checking c1_c2_dist < radius condition
                        break;
                    }
                }
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
                        reverse_subroute_outer2(route, cust_pos_, h4, h1);
                    } else {
                        reverse_subroute_inner(route, cust_pos_, h4, h1);
                    }
                }
                dont_look_[h1] = dont_look_[h2] = dont_look_[h3] = dont_look_[h4] = 0;
            } else { // no improvement
                dont_look_[c1] = 1;
            }
        }
    }

#ifndef NDEBUG
    double final_len = calc_route_len(route);
#endif
    assert(final_len <= start_len);
    /*if (improvements_count > 0) {
        cout << "Total improvements: " << improvements_count
             << "\n\tstart_len: " << start_len
             << "\n\tfinal_len: " << final_len
             << "\n\tgain: "
             << (start_len - final_len) << endl;
    }*/
}


void opt3(std::vector<uint32_t> &route,
          const std::vector<std::vector<uint32_t>> &nn_lists,
          std::function<double (int, int)> get_distance,
          std::function<double (const std::vector<uint32_t>&)> calc_route_len
          ) {
    //int improvements_count = 0;
    bool improvement = true;
    const int N = route.size();
    
    static std::vector<uint8_t> dont_look_;
    static std::vector<int> cust_pos_;
    static std::vector<int> random_order_;

    dont_look_.resize((size_t)(N+1));
    cust_pos_.resize((size_t)(N+1));
    for (size_t i = 0; i < route.size(); ++i) {
        cust_pos_[ route[i] ] = i;
        dont_look_[ route[i] ] = 0;
    }
    random_order_.reserve((size_t)(N+1));
    gen_random_permutation(N, random_order_);

    #ifndef NDEBUG
    double start_len = calc_route_len(route);
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
            //a = random_order_[cust_i];
            a = cust_i;
            if (dont_look_[a] == 1) {
                continue;
            }
            int pos_a = cust_pos_[a];
            b = route[(pos_a+1) % N];
            double radius = get_distance(a, b);
            double dist_a_b = radius;
            bool improve_node = false;

            // look for c in a's nearest neighbours list
            auto const &nn_list = nn_lists.at(a);
            const int NNCount = (int)nn_list.size();

            // search for 2-opt move
            for (int j = 0; (improve_node == false) && (j < NNCount); ++j) {
                c = nn_list[j];
                double dist_a_c = get_distance(a, c);
                if (dist_a_c < dist_a_b) { // 2-Opt possible
                    int pos_c = cust_pos_[c];
                    d = route[(pos_c + 1) % N];
                    double gain = (dist_a_b + get_distance(c,d)) -
                                  (dist_a_c + get_distance(b, d));
                    if (gain > 0) {
                        reverse_subroute_flip(route, cust_pos_, b, c);
                        dont_look_[a] = dont_look_[b] = dont_look_[c] = dont_look_[d] = 0;

                        improve_node = true;
                    }
                } else {
                    // the rest of neighbours are further than c
                    break ;
                }
            }
            int prev_a = route[ pos_a > 0 ? pos_a-1 : N-1 ];
            double dist_prev_a_a = get_distance(prev_a, a);
            for (int j = 0; (improve_node == false) && (j < NNCount); ++j) {
                c = nn_list[j];
                double dist_a_c = get_distance(a, c);

                if (dist_a_c < dist_prev_a_a) { // 2-Opt possible
                    int pos_c = cust_pos_[c];
                    int prev_c = route[ pos_c > 0 ? pos_c - 1 : N-1 ]; // d is now a predecessor of c

                    if (prev_c == a || prev_a == c)
                        continue ;

                    double gain = (dist_prev_a_a + get_distance(prev_c, c)) -
                                  (dist_a_c + get_distance(prev_a, prev_c));

                    if (gain > 0) {
                        reverse_subroute_flip(route, cust_pos_, c, prev_a);
                        dont_look_[prev_a] = dont_look_[a] = dont_look_[prev_c] = dont_look_[c] = 0;
                        improve_node = true;
                    }
                } else {
                    // the rest of neighbours are further than c
                    break ;
                }
            }

            // search for 3-opt move
            for (int j = 0; (improve_node == false) && (j < NNCount); ++j) {
                c = nn_list[j];
                int pos_c = cust_pos_[c];
                d = route[(pos_c + 1) % N];

                if (d == a) {
                    continue ;
                }
                
                double dist_a_d = get_distance(a, d);

                if (dist_a_d < dist_a_b) { // improvement possible -> now find e

                    auto const &nn_list_c = nn_lists.at(c);
                    // look for e in c's neighbours list
                    for (int k = 0; (improve_node == false) && (k < NNCount); ++k) {
                        e = nn_list_c[k];
                        int pos_e = cust_pos_[e];
                        f = route[ (cust_pos_[e] + 1) % N ];
                        // node e has to lay between nodes c and a, i.e. a..c..e
                        if ( (f == a) ||
                             !( (pos_a < pos_c && (pos_c < pos_e || pos_e < pos_a)) ||
                                (pos_a > pos_c && (pos_c < pos_e && pos_e < pos_a)) ) )
                            continue;

                        // now check two possibilities
                        double dist_c_d = get_distance(c, d);
                        double dist_e_f = get_distance(e, f);
                        
                        // A) edges (a,d) (c,e) (f,b) are created
                        double gain = dist_a_b + dist_c_d + dist_e_f -
                            (dist_a_d + get_distance(c, e) + get_distance(f, b));
                        if (gain > 0) {
                            // This move needs reversal of subroutes (f..a) and (d..e)
                            reverse_subroute(route, cust_pos_, d, e);
                            reverse_subroute(route, cust_pos_, f, a);
                            dont_look_[a] = dont_look_[b] = dont_look_[c] = 0;
                            dont_look_[d] = dont_look_[e] = dont_look_[f] = 0;
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
                            reverse_subroute(route, cust_pos_, f, a);
                            reverse_subroute(route, cust_pos_, b, c);
                            reverse_subroute(route, cust_pos_, d, e);
                            dont_look_[a] = dont_look_[b] = dont_look_[c] = 0;
                            dont_look_[d] = dont_look_[e] = dont_look_[f] = 0;
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
                    dont_look_[a] = 1;
                }
            }
        }
    }
    //cout << "total_improvements = " << total_improvements << endl;

    #ifndef NDEBUG
    double final_len = calc_route_len(route);
    assert(final_len <= start_len);
    if (total_improvements > 0) {
        std::cout << "Total improvements: " << total_improvements
             << "\n\tstart_len: " << start_len
             << "\n\tfinal_len: " << final_len
             << "\n\tgain: "
             << (start_len - final_len) << std::endl;
    }
    #endif
}
