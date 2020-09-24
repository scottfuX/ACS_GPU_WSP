#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <map>
#include <chrono>
#include <cassert>
#include <memory>
#include <iomanip>
#include <sstream>
#include <unistd.h>

#include "ants.h"
#include "gpu_ants.h"
#include "common.h"
#include "stopcondition.h"
#include "local_search.h"

#include"cusparse.h"
#include"cpu_phmem.h"

#include "bupt_global_config.h"

using namespace std;

// Seed with a real random value, if available
std::random_device rd;
std::mt19937 g_rng(rd());


struct TSPProblemData {
    enum EdgeWeightType { NONE, EUC_2D };

    std::vector<std::pair<double, double>> coords_;
    std::map<std::string, std::string> desc_;
    uint32_t dimension_ = 0;
};


static TSPProblemData read_problem_data_tsp(const std::string &path) {
    TSPProblemData pd;
    ifstream file(path);

    if (file.is_open()) {
        string line;
        bool parse_points = false;
        while (getline(file, line)) {
            transform(line.begin(), line.end(), line.begin(), ::tolower);

            if (line.find("eof") != string::npos) {
                parse_points = false;
            } else if (parse_points) {
                istringstream in(line);
                int _;
                double x, y;
                in >> _ >> x >> y;
                pd.coords_.push_back( {x, y} ); 
            } else if (line.find("node_coord_section") != string::npos) {
                parse_points = true;
            } else if (line.find(":") != string::npos) {
                istringstream in(line);                
                string key, value, token;
                in >> key;
                while (in >> token) {
                    if (token != ":") {
                        value += token;
                        value += " ";
                    }
                }
                trim(key);
                trim(value);
                if (key.back() == ':') {
                    key = key.substr(0, key.size() - 1);
                }
                cout << "key: [" << key << "] value: [" << value << "]" << endl;
                pd.desc_[key] = value;
            }
        }

        file.close(); 
    }
    cout << "Coords size: " << pd.coords_.size() << endl;
    pd.dimension_ = pd.coords_.size();
    return pd;
}


static double calc_euc2d_distance(double x1, double y1, double x2, double y2) {
    return (int) ( sqrt( ((x2-x1) * (x2-x1)) + (y2-y1)*(y2-y1) ) + 0.5 );
}


/*
 * Compute ceiling distance between two nodes rounded to next integer for TSPLIB instances.
 */
static double calc_ceil2d_distance(double x1, double y1, double x2, double y2) {
    double xd = x1 - x2;
    double yd = y1 - y2;
    double r = sqrt(xd * xd + yd * yd);
    return (long int)(ceil(r));
}

// static double calc_euc2d_distance(double x1, double y1, double x2, double y2) {
//     return (int) ( ( ((x2-x1) * (x2-x1)) + (y2-y1)*(y2-y1) ) + 0.5 );
// }


// /*
//  * Compute ceiling distance between two nodes rounded to next integer for TSPLIB instances.
//  */
// static double calc_ceil2d_distance(double x1, double y1, double x2, double y2) {
//     double xd = x1 - x2;
//     double yd = y1 - y2;
//     double r = (xd * xd + yd * yd);
//     return (long int)(ceil(r));
// }



template<typename DistanceFunction_t>
static void calc_dist_matrix(DistanceFunction_t calc_distance,
                             const std::vector<std::pair<double, double>> coords,
                             std::vector<std::vector<double>> &matrix
                             ) {
    const auto size = coords.size();
    matrix.resize(size);
    for (auto &row : matrix) {
        row.resize(size, 0.0);
    }
    for (uint32_t i = 0; i < size; ++i) {
        for (uint32_t j = 0; j < size; ++j) {
            if (i < j) {
                auto p1 = coords[i];
                auto x1 = p1.first, y1 = p1.second;
                auto p2 = coords[j];
                auto x2 = p2.first, y2 = p2.second;
                auto dist = calc_distance(x1, y1, x2, y2); 
                matrix[i][j] = dist;
                matrix[j][i] = dist; // Symmetric TSP
            }
        }
    }
}


static void calc_dist_matrix(TSPProblemData &problem, std::vector<std::vector<double>> &matrix) {
    if (problem.desc_["edge_weight_type"] == "euc_2d") {
        calc_dist_matrix(calc_euc2d_distance, problem.coords_, matrix);
    } else if (problem.desc_["edge_weight_type"] == "ceil_2d") {
        calc_dist_matrix(calc_ceil2d_distance, problem.coords_, matrix);
    } else {
        throw runtime_error("Unknown edge weight type");
    }
}


static double eval_route(const vector<vector<double>> &dist_matrix,
                  const vector<uint32_t> &route) {
    double r = 0.0;
    if (!route.empty()) {
        uint32_t u = route.back();
        for (uint32_t v : route) {
            r += dist_matrix[u][v];
            u = v;
        }
    }
    return r;
}


static void ant_move_to(Ant &ant, uint32_t node) {
    assert( node < ant.nodes_count_ );
    assert( find(ant.visited_.begin(), ant.visited_.end(), node) == ant.visited_.end() );

    ant.position_ = node;
    ant.visited_.push_back(node);
    ++ant.visited_count_;
    ant.is_visited_[node] = true;
}


static void ant_init(Ant &ant, uint32_t nodes_count, uint32_t start_node) {
    assert( start_node < nodes_count );

    ant.nodes_count_ = nodes_count;
    ant.is_visited_.clear();
    ant.is_visited_.resize(nodes_count, false);
    ant.visited_.clear();
    ant.visited_.reserve(nodes_count);
    ant.visited_count_ = 0;
    ant_move_to(ant, start_node);
}


static
ostream &operator << (ostream &out, const Ant &ant) {
    out << "Ant[ value: " << ant.value_ << "\tpos: " << ant.position_
         << "\tvisited: " << ant.visited_.size()
         << " ]";
    return out;
}


template<typename T>
static
std::vector<T> range(T beg, T end) {
    std::vector<T> r;
    for ( ; beg < end; ++beg) { r.push_back(beg); }
    return r;
}

/**
 * Calculate nearest neighbours list for each node
 */
static vector<vector<uint32_t>> 
init_nn_lists(const vector<vector<double>> &dist_matrix, uint32_t nn_size = 32) {
    const uint32_t size = dist_matrix.size();
     
    vector<vector<uint32_t>> nn_lists;
    for (uint32_t i = 0; i < size; ++i) {
        vector<uint32_t> all_nodes = range(0u, size);
        all_nodes.erase( all_nodes.begin() + i ); // We don't want node to be a neighbour of self
        const auto &dist = dist_matrix[i];

        sort(all_nodes.begin(), all_nodes.end(),
            [&](uint32_t u, uint32_t v) -> bool { return dist[u] < dist[v]; });
        
        // assert( dist[all_nodes[0]] <= dist[all_nodes[1]] );
        all_nodes.resize(nn_size); // drop size - nn_size farthest nodes

        nn_lists.push_back(all_nodes);
    }
    return nn_lists;
}


static
vector<uint32_t> build_route_greedy(const vector<vector<double>> &dist_matrix,
        uint32_t start_node = 0) {
    const uint32_t size = dist_matrix.size();

    vector<uint32_t> route;
    route.push_back(start_node);
    auto nodes = range(0u, size);
    nodes.erase( find(nodes.begin(), nodes.end(), start_node) );
    auto curr = route.back();
    while (!nodes.empty()) {
        auto &dist = dist_matrix.at(curr);
        auto closest = *min_element(nodes.begin(), nodes.end(),
                [&] (uint32_t i, uint32_t j) { return dist.at(i) < dist.at(j); });
        route.push_back(closest);
        nodes.erase( find(nodes.begin(), nodes.end(), closest) );
        curr = closest;
    }
    assert(route.size() == size);
    return route;
}


template<typename T>
class MatrixPheromoneMemory {
public:
    typedef T value_type;
public:

    MatrixPheromoneMemory(uint32_t msize, bool is_symmetric = true) : 
        size_(msize),
        is_symmetric_(is_symmetric)
    {
        matrix_.resize(size_);
        for (auto &row : matrix_) {
            row.resize(size_);
        }
    }

    void set_all(const T value) {
        for (auto &row : matrix_) {
            row.assign(size_, value);
        }
    }

    void set(uint32_t i, uint32_t j, const T value) {
        assert(i < matrix_.size() && j < matrix_.at(i).size());
        matrix_[i][j] = value;
        if (is_symmetric_) {
            matrix_[j][i] = value;
        }
    }


    T get(uint32_t i, uint32_t j) const {
        assert(i < matrix_.size() && j < matrix_.at(i).size());
        return matrix_[i][j];
    }

    uint32_t size() const { return size_; }


    std::vector<T> get_all() {
        std::vector<T> out;
        for (auto &row : matrix_) {
            for (auto v : row) {
                out.push_back(v);
            }
        }
        return out;
    }

public:
    uint32_t size_; 
    bool is_symmetric_;
    std::vector<std::vector<T>> matrix_;
};


template<typename T>
struct Bucket {
    typedef T value_type;

    Bucket() :
        size_(0) {
    }

    void reserve(uint32_t capacity) {
        elements_.resize(capacity);
    }

    T &operator[](uint32_t index) {
        assert(index < size_);
        return elements_[index];
    }

    uint32_t size() const {
        return size_;
    }

    void clear() {
        size_ = 0;
    }

    void pop_back() {
        assert(size_ > 0);
        --size_;
    }

    void insert(uint32_t index, T value) {
        assert(size_ < elements_.size());
        for (uint32_t i = size_; i > index; --i) {
            elements_[i] = elements_[i-1];
        }
        elements_[index] = value;
        ++size_;
    }


    uint32_t size_;
    std::vector<T> elements_; 
};


class SelectivePheromoneMemory {
public:
    typedef double value_type;

    typedef std::pair<uint32_t, double> Entry;
    typedef Bucket<Entry> Trails;
public:
    SelectivePheromoneMemory(uint32_t msize, uint32_t bucket_size, bool is_symmetric=true) :
        size_(msize), 
        bucket_size_(bucket_size),
        is_symmetric_(is_symmetric) 
    {
        buckets_ = new Trails[ size_ ];
        for (auto i = 0u; i < size_; ++i) {
            auto &b = buckets_[i];
            b.reserve((uint32_t)bucket_size);
        }
    }

    ~SelectivePheromoneMemory() {
        delete [] buckets_;
    }

    void set_all(double value) {
        default_pheromone_ = value;
        for (uint32_t i = 0; i < size_; ++i) {
            buckets_[i].clear();
        }
        hits_ = misses_ = 0;
    }

    double get(uint32_t i, uint32_t j) {
        assert(i < size_ && j < size_);
        auto &bucket = buckets_[i];
        for (uint32_t k = 0; k < bucket.size(); ++k) {
            auto &e = bucket[k];
            if (e.first == j) {
                return e.second;
            }
        }
        return default_pheromone_;
    }


    void set_helper(uint32_t i, uint32_t j, double value) { 
        assert(i < size_ && j < size_);
        auto &trails = buckets_[i];
        Entry *entry = nullptr;
        for (uint32_t k = 0; k < trails.size(); ++k) {
            auto &e = trails[k];
            if (e.first == j) {
                entry = &e;
                break ;
            }
        }
        if (entry != nullptr) {
            ++hits_;
            entry->second = value;
        } else if (trails.size() >= bucket_size_) {
            ++misses_;
            trails.pop_back();
            trails.insert(  0, { j, value } );
        } else { // There is a place for the new element
            ++misses_;
            trails.insert( 0, { j, value } );
        }
    }


    void set(uint32_t i, uint32_t j, double value) { 
        set_helper(i, j, value);
        if (is_symmetric_) {
            set_helper(j, i, value);
        }
    }


    Trails &get_trails(uint32_t node) {
        return buckets_[node];
    }

    double default_pheromone() const { 
        return default_pheromone_; 
    }

    uint64_t get_mem_hits() const { return hits_; }

    uint64_t get_mem_misses() const { return misses_; }

private:
    uint32_t size_;
    uint32_t bucket_size_;
    bool is_symmetric_;
    Trails *buckets_;
    double default_pheromone_;
    uint64_t hits_;
    uint64_t misses_;
};


static
double calc_initial_pheromone(const vector<vector<double>> &dist_matrix) {
    const uint32_t sample_size = 3;
    double mean_len = 0.0;
    for (uint32_t i = 0; i < sample_size; ++i) {
        auto route = build_route_greedy(dist_matrix, i);
        const double value = eval_route(dist_matrix, route);
        mean_len += value;
    }
    mean_len /= sample_size;
    const double initial_level = 1.0 / (dist_matrix.size() * mean_len);
    return initial_level;
}


static
void init_heuristic_matrix(const std::vector<std::vector<double>> &dist_matrix,
                           double beta,
                           std::vector<std::vector<double>> &heuristic_matrix) {
    const uint32_t size = dist_matrix.size();
    heuristic_matrix.resize(size);

    for (uint32_t i = 0; i < size; ++i) {
        auto &row = heuristic_matrix[i];
        row.resize(size);
        for (uint32_t j = 0; j < size; ++j) {
            if (i != j) {
                row[j] = 1.0 / pow(dist_matrix[i][j], beta);
            }
        }
    }
}


static
void init_total_matrix(const std::vector<std::vector<double>> &dist_matrix,
                       double beta,
                       double initial_pheromone,
                       std::vector<std::vector<double>> &total_matrix) {

    const uint32_t size = dist_matrix.size();
    total_matrix.resize(size);
    for (uint32_t i = 0; i < size; ++i) { total_matrix[i].resize(size); }

    for (uint32_t i = 0; i < size; ++i) {
        for (uint32_t j = 0; j < i; ++j) { // Assuming symmetric problem
            const double h = pow(1.0 / dist_matrix[i][j], beta);
            total_matrix[j][i] = total_matrix[i][j] = initial_pheromone * h;
        }
    }
}


static uint32_t select_among_others_counter = 0;
static int q_below_count = 0;
static int q_above_count = 0;


template<typename PheromoneMemory_t>
static
uint32_t acs_calc_move(Ant &ant, 
                       const vector<vector<double>> & heuristic_matrix,
                       const vector<vector<double>> &total_matrix,
                       const vector<vector<uint32_t>> &nn_lists,
                       double q0,
                       mt19937 &rng,
                       PheromoneMemory_t * phmem) {

    uint32_t next_node = ant.nodes_count_ + 1;
    const uint32_t curr = ant.visited_.back();
    const auto &total = total_matrix[curr];
    static uint32_t cand_set[32];
    uint32_t cand_count = 0;
    for (auto node : nn_lists[curr]) {
        if (!ant.is_visited_[node]) {
            cand_set[cand_count++] = node;
        }
    }

    if (cand_count > 0) {
        uniform_real_distribution<> random_double(0.0, 1.0);
        double q = random_double(rng);

        if (q < 0.5) {
            q_below_count++;
        } else {
            q_above_count++;
        }

        if (q < q0) { // Greedy move
            double max_product = 0;
            for (uint32_t i = 0; i < cand_count; ++i) {
                const double product = total[ cand_set[i] ];
                if (product > max_product) {
                    max_product = product;
                    next_node = cand_set[i];
                }
            }
        } else { // Select using proportional rule
            double sum = 0.0;
            for (uint32_t i = 0; i < cand_count; ++i) {
                auto node = cand_set[i];
                sum += total[node];
            }
            next_node = cand_set[0];
            const double r = random_double(rng) * sum;
            double v = 0.0;
            for (uint32_t i = 0; i < cand_count; ++i) {
                auto node = cand_set[i];
                v += total[node];
                if (r <= v) {
                    next_node = node;
                    break ;
                }
            }
        }
    } else { // Select best next node among all the unvisited nodes
        double max_product = 0.0;
        
        for (uint32_t node = 0; node < ant.nodes_count_; ++node) {
            if (!ant.is_visited_[node]) {
                const double product = total[node];
                if (product > max_product) {
                    max_product = product;
                    next_node = node;
                }
            }
        }
        ++select_among_others_counter;
    }
    assert(next_node != ant.nodes_count_ + 1);
    return next_node;
}


/*
 * Calculates the next node the given ant should move to from its current
 * position. This is a version optimized for SelectivePheromoneMemory
 */
uint32_t acs_calc_move(Ant &ant, 
                       const vector<vector<double>> &heuristic_matrix,
                       const vector<vector<double>> & /*total_matrix*/,
                       const vector<vector<uint32_t>> &nn_lists,
                       double q0,
                       mt19937 &rng,
                       SelectivePheromoneMemory *pheromone_memory
                       ) {
    auto &memory = *pheromone_memory;
    uint32_t next_node = ant.nodes_count_ + 1;
    const uint32_t curr = ant.visited_.back();
    static uint32_t cand_set[64];
    assert( heuristic_matrix.size() > curr );
    auto &heuristic = heuristic_matrix.at(curr);
    uint32_t cand_count = 0;
    auto &nn = nn_lists[curr];
    for (auto node : nn) {
        if (!ant.is_visited_[node]) {
            cand_set[cand_count++] = node;
            break ;
        }
    }

    if (cand_count) {
        uniform_real_distribution<> random_double(0.0, 1.0);
        double q = random_double(rng);

        if (q < q0) { // Greedy move
            double max_product = 0;

            auto &trails = memory.get_trails(curr);

            for (uint32_t i = 0; i < trails.size(); ++i) {
                auto &e = trails[i];
                auto node = e.first;
                if ( !ant.is_visited_[node] ) {
                    const double product = e.second * heuristic[node];
                    if (product > max_product) {
                        max_product = product;
                        next_node = node;
                    }
                }
            }
            auto node = cand_set[0];
            const double product = memory.default_pheromone() * heuristic[node];

            if (product > max_product) {
                next_node = node;
            }
        } else { // Select using proportional rule
            cand_count = 0;
            double sum = 0.0;
            for (auto node : nn) {
                if (!ant.is_visited_[node]) {
                    cand_set[cand_count++] = node;
                    const double product = memory.default_pheromone() * heuristic[node];
                    sum += product;
                }
            }
            const uint32_t nn_end = cand_count;

            auto &trails = memory.get_trails(curr);
            for (uint32_t i = 0; i < trails.size(); ++i) {
                auto &e = trails[i];
                auto node = e.first;
                if ( !ant.is_visited_[node] ) {
                    cand_set[cand_count++] = node;
                    const double product = e.second * heuristic[node];
                    sum += product;
                }
            }

            next_node = cand_set[0];
            const double r = random_double(rng) * sum;
            double v = 0.0;

            for (uint32_t i = 0; i < cand_count; ++i) {
                auto node = cand_set[i];
                if (i < nn_end) {
                    v += memory.default_pheromone() * heuristic[node];
                } else {
                    v += memory.get(curr, node) * heuristic[node];
                }
                if (r <= v) {
                    next_node = node;
                    break ;
                }
            }
        }
    } else { // Select best next node among all the unvisited nodes
        double max_product = 0.0;
        auto &trails = memory.get_trails(curr);
        for (uint32_t i = 0; i < trails.size(); ++i) {
            auto &e = trails[i];
            auto node = e.first;
            if ( !ant.is_visited_[node] ) {
                const double product = e.second * heuristic[node];
                if (product > max_product) {
                    max_product = product;
                    next_node = node;
                }
            }
        }

        if (next_node == ant.nodes_count_+1) {
        for (uint32_t node = 0; node < ant.nodes_count_; ++node) {
            if (!ant.is_visited_[node]) {
                const double product = memory.default_pheromone() * heuristic[node];
                if (product > max_product) {
                    max_product = product;
                    next_node = node;
                }
            }
        }
        }
    }
    assert(next_node != ant.nodes_count_ + 1);
    return next_node;
}


template<class PheromoneMemory>
void pheromone_update(PheromoneMemory &pheromone_memory,
                      const vector<vector<double>> &dist_matrix,
                      double beta,
                      vector<vector<double>> &total_matrix,
                      uint32_t u,
                      uint32_t v,
                      double evap_ratio,
                      double delta) {
    typedef typename PheromoneMemory::value_type T;
    const T val = pheromone_memory.get(u, v);
    const T updated = (T) ((1 - evap_ratio) * val + evap_ratio * delta);

    const double h = __builtin_powi(1.0 / dist_matrix[u][v], (int)beta);
    //const double h = powf(1.0 / dist_matrix[u][v], beta);
    pheromone_memory.set(u, v, updated);
    total_matrix[u][v] = total_matrix[v][u] = updated * h;
}


template<class PheromoneMemory>
void local_pheromone_update(PheromoneMemory &pheromone_memory,
                            const vector<vector<double>> &dist_matrix,
                            double beta,
                            vector<vector<double>> &total_matrix,
                            const Ant &ant,
                            uint32_t node_index,
                            double phi,
                            double initial_pheromone) {
    assert( ant.visited_.size() >= 2 );
    assert( node_index < ant.visited_.size() );

    uint32_t u = ant.visited_[node_index];
    uint32_t v = ant.visited_[(node_index + 1) % ant.visited_.size()];
    pheromone_update(pheromone_memory, dist_matrix, beta, total_matrix, u, v, phi, initial_pheromone);
}


template<class PheromoneMemory>
static
void global_pheromone_update(PheromoneMemory &pheromone_memory,
                            const vector<vector<double>> &dist_matrix,
                            double beta,
                            vector<vector<double>> &total_matrix,
                            const Ant &ant,
                            double rho) {
    assert( is_valid_route(ant.visited_, ant.nodes_count_) );
    assert( ant.value_ > 1.0 );

    const double delta = 1.0 / ant.value_;
    uint32_t prev = ant.visited_.back();
    for (auto node : ant.visited_) {
        pheromone_update(pheromone_memory, dist_matrix, beta, total_matrix, prev, node, rho, delta);
        prev = node;
    }
}


template<class PheromoneMemory>
static
void pheromone_update(PheromoneMemory &pheromone_memory,
                      const vector<vector<double>> &dist_matrix,
                      double beta,
                      uint32_t u,
                      uint32_t v,
                      double evap_ratio,
                      double delta) {
    typedef typename PheromoneMemory::value_type T;
    const T val = pheromone_memory.get(u, v);
    const T updated = (T) ((1 - evap_ratio) * val + evap_ratio * delta);
    pheromone_memory.set(u, v, updated);
}


template<class PheromoneMemory>
void local_pheromone_update(PheromoneMemory &pheromone_memory,
                            const vector<vector<double>> &dist_matrix,
                            double beta,
                            const Ant &ant,
                            uint32_t node_index,
                            double phi,
                            double initial_pheromone) {
    assert( ant.visited_.size() >= 2 );
    assert( node_index < ant.visited_.size() );

    uint32_t u = ant.visited_[node_index];
    uint32_t v = ant.visited_[(node_index + 1) % ant.visited_.size()];
    pheromone_update(pheromone_memory, dist_matrix, beta, u, v, phi, initial_pheromone);
}


template<class PheromoneMemory>
static
void global_pheromone_update(PheromoneMemory &pheromone_memory,
                            const vector<vector<double>> &dist_matrix,
                            double beta,
                            const Ant &ant,
                            double rho) {
    assert( is_valid_route(ant.visited_, ant.nodes_count_) );
    assert( ant.value_ > 1.0 );

    const double delta = 1.0 / ant.value_;
    uint32_t prev = ant.visited_.back();
    for (auto node : ant.visited_) {
        pheromone_update(pheromone_memory, dist_matrix, beta, prev, node, rho, delta);
        prev = node;
    }
}


/*
 * ACS with the Selective Pheromone Memory version
 */
static std::map<std::string, pj> 
run_acs_spm(const vector<vector<double>> &dist_matrix,
             const vector<vector<double>> &heuristic_matrix,
             const vector<vector<uint32_t>> &nn_lists,
             mt19937 &rng,
             ACSParams &params,
             StopCondition *stop_cond,
             uint32_t bucket_size) {

    assert(params.initial_pheromone_ > 0.0);
    const uint32_t dimension = dist_matrix.size();

    const bool is_symmetric = true;
    SelectivePheromoneMemory pheromone_memory(dimension + 1, bucket_size, is_symmetric);
    pheromone_memory.set_all(params.initial_pheromone_);

    double total_sol_calc_time = 0;
 
    Ant global_best;
    global_best.value_ = numeric_limits<double>::max();

    // Calc time for all iterations
    auto start = std::chrono::steady_clock::now();
    uint32_t best_found_iteration = 0; 

    for (stop_cond->init(); !stop_cond->is_reached(); stop_cond->next_iteration()) {
        vector<Ant> ants{ params.ants_count_ };
        std::uniform_int_distribution<> random_node(0, (int)dimension-1);
        for (auto &ant : ants) {
            ant_init(ant, dimension, (uint32_t)random_node(rng));
        }
        /**
         * Calc time needed to create single solution by each ant
         * Count also local pheromone update because it is calculated
         * similarly in the GPU version of the algorithm.
         */
        auto sol_calc_start = std::chrono::steady_clock::now();

        for (uint32_t j = 0; j+1 < dimension; ++j) {
            for (auto &ant : ants) {
                auto node = acs_calc_move(ant, heuristic_matrix, heuristic_matrix, nn_lists, params.q0_, rng, &pheromone_memory);
                ant_move_to(ant, node);

                local_pheromone_update(pheromone_memory, dist_matrix,
                                       params.beta_,
                                       ant, j, params.phi_, params.initial_pheromone_);
            }
        }
        auto sol_calc_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(sol_calc_end - sol_calc_start);
        total_sol_calc_time += elapsed.count();

        for (auto &ant : ants) {
            assert(is_valid_route(ant.visited_, dimension));

            ant.value_ = eval_route(dist_matrix, ant.visited_);
        }

        uint32_t best_index = 0;
        for (uint32_t j = 1; j < ants.size(); ++j) {
            if (ants[j].value_ < ants[best_index].value_) {
                best_index = j;
            }
        }
        auto &local_best = ants[best_index];

        if (global_best.value_ > local_best.value_) {
            global_best = local_best;
            best_found_iteration = stop_cond->get_iteration();
        }
        global_pheromone_update(pheromone_memory, dist_matrix, params.beta_, global_best, params.rho_);
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto total_calc_time = elapsed.count() / 1.0e6;

    cout << "Global best ant: " << global_best << endl;

    std::map<std::string, pj> out;

    out["iterations_made"] = pj( (int64_t)stop_cond->get_iteration() );
    out["best_value"] = pj( global_best.value_ );
    out["best_solution"] = pj( sequence_to_string(global_best.visited_.begin(),
                                                  global_best.visited_.end()) );
    out["best_found_iteration"] = pj( (int64_t)best_found_iteration );
    out["sol_calc_time_msec"] = pj( total_sol_calc_time / (1000.0 * stop_cond->get_iteration()) );
    out["total_sol_calc_time"] = pj( total_calc_time );
    out["spm_mem_hits"] = pj( (int64_t)pheromone_memory.get_mem_hits() );
    out["spm_mem_misses"] = pj( (int64_t)pheromone_memory.get_mem_misses() );

    return out;
}



/*
 * Standard ACS algorithm implementation.
 */
static std::map<std::string, pj> 
run_acs(const vector<vector<double>> &dist_matrix,
        const vector<vector<double>> &heuristic_matrix,
        vector<vector<double>> &total_matrix,
        const vector<vector<uint32_t>> &nn_lists,
        mt19937 &rng,
        ACSParams &params,
        StopCondition *stop_cond,
        std::function<void (std::vector<uint32_t>&)> local_search
        ) {
    assert(params.initial_pheromone_ > 0.0);
    const uint32_t dimension = total_matrix.size();

    const bool is_symmetric = true;
    MatrixPheromoneMemory<double> pheromone_memory(dimension, is_symmetric);
    pheromone_memory.set_all(params.initial_pheromone_);

	//record("pheromone_memory.initial=",pj( sequence_to_string(pheromone_memory.get_all().begin(),pheromone_memory.get_all().end())));
	//record("pheromone_memory.initial=","test");
    std::uniform_int_distribution<> random_node(0, (int)dimension-1);

    IntervalTimer sol_calc_timer;
    IntervalTimer local_search_timer;
    IntervalTimer total_calc_timer;
 
    Ant global_best;
    global_best.value_ = numeric_limits<double>::max();

    // Calc time for all iterations
    total_calc_timer.start_interval();

    uint32_t best_found_iteration = 0; 
    //---------------modify by BUPT 
    std::string iter_solution;

    for ( stop_cond->init(); 
          !stop_cond->is_reached();
          stop_cond->next_iteration() ) {

        sol_calc_timer.start_interval();

        vector<Ant> ants{ params.ants_count_ };
        uint32_t i = 0u;
        for (auto &ant : ants) {
            ant_init(ant, dimension, (uint32_t)random_node(rng));
            ++i;
        }
        /* 
         * Calc time needed to create single solution by each ant * Count also
         * local pheromone update because it is calculated  similarly in the GPU
         * version of the algorithm.
         */

        for (uint32_t j = 0; j+1 < dimension; ++j) {
            for (auto &ant : ants) {
                auto node = acs_calc_move(ant, heuristic_matrix, total_matrix, nn_lists, params.q0_, rng, &pheromone_memory);
                ant_move_to(ant, node);
                local_pheromone_update(pheromone_memory, dist_matrix,
                                       params.beta_, total_matrix,
                                       ant, j, params.phi_, params.initial_pheromone_);
            }
        }
        for (auto &ant : ants) {
            local_pheromone_update(pheromone_memory, dist_matrix,
                                   params.beta_, total_matrix,
                                   ant, 
                                   dimension - 1, // last node + first node
                                   params.phi_, params.initial_pheromone_);
        }

        sol_calc_timer.stop_interval();

        if (local_search) {
            local_search_timer.start_interval();
            for (auto &ant : ants) {
                local_search(ant.visited_);

                if (stop_cond->is_reached()) { // LS is costly, we may not have time for each ant
                    break ;
                }
            }
            local_search_timer.stop_interval();
        }

        for (auto &ant : ants) {
            assert(is_valid_route(ant.visited_, dimension));

            ant.value_ = eval_route(dist_matrix, ant.visited_);
        }

        uint32_t best_index = 0;
        for (uint32_t j = 1; j < ants.size(); ++j) {
            if (ants[j].value_ < ants[best_index].value_) {
                best_index = j;
            }
        }
        auto &local_best = ants[best_index];

        if (global_best.value_ > local_best.value_) {
            global_best = local_best;
            best_found_iteration = stop_cond->get_iteration();
        }

        //Modify by BUPT record best solution
        if(params.sol_rec_freq_ != 0 && stop_cond->get_iteration() % params.sol_rec_freq_ == 0){ 
            iter_solution = iter_solution + "[" + to_string(stop_cond->get_iteration()) + \
                                    ": " + to_string((uint32_t)global_best.value_) + "] ";
        }

        global_pheromone_update(pheromone_memory, dist_matrix, params.beta_,
                                total_matrix, global_best, params.rho_);
        //Modify by BUPT-----------------------------PRINT_PHMEM-----------------------------
        if(params.phmem_print_){
            std::vector<double> vec;
            bool is_end = false;
            uint32_t start_id = 0;
            uint32_t once_trans_size = dimension;
            uint32_t maxsize = pheromone_memory.size();
            std::string str_title = params.outdir+"/" 
                + "[phmem_acs]"+ params.test_name + "_"
                + get_current_date_time("%G-%m-%d_%H_%M_%S")
                + "-" + to_string(getpid()) + ".txt";

            if(stop_cond->get_iteration() == 0){
                bupt_chk_and_del(str_title); //if have file delete it
                printf("Start copying phmem to \"%s\" file\n",str_title.c_str());
            }
            if(stop_cond->get_iteration() == stop_cond->get_max_iterations() - 1){
                is_end = false;
                bupt_outfile_app_newline(str_title,"FINAL phmem_data:");
                for (auto &row : pheromone_memory.matrix_) {
                    bupt_outfile_app(str_title,row);
                    printf(". ");
                }
                printf("\nCopy phmem to \"%s\" file completion\n",str_title.c_str());
            }else{
                is_end = false;
                if(params.phmem_print_freq_ != 0 && stop_cond->get_iteration() % params.phmem_print_freq_ == 0){
                    std::string str = "N0." + std::to_string(stop_cond->get_iteration()) + " phmem_data:";
                    bupt_outfile_app_newline(str_title,str);
                    while(!is_end){
                        uint32_t trans_size = once_trans_size;
                        if(start_id + 1 >= maxsize){
                            is_end = true;
                        }
                        vec.resize(trans_size);
                        std::copy(&vec[0],&vec[trans_size-1],&pheromone_memory.matrix_[start_id][0]);
                        bupt_outfile_app(str_title,vec);
                        start_id ++;
                        printf(". ");
                    }
                    printf("\n");
                }
            }
        }
        //Modify end-----------------------------PRINT_PHMEM-----------------------------
    }
    total_calc_timer.stop_interval();

    cout << "Global best ant: " << global_best << endl;

    std::map<std::string, pj> out;

    out["iterations_made"] = pj( (int64_t)stop_cond->get_iteration() );
    out["best_value"] = pj( global_best.value_ );
    out["best_solution"] = pj( sequence_to_string(global_best.visited_.begin(),
                                                  global_best.visited_.end()) );
    out["best_found_iteration"] = pj( (int64_t)best_found_iteration );
    out["sol_calc_time_msec"] = pj( sol_calc_timer.get_total_time_ms() / stop_cond->get_iteration() );
    out["total_sol_calc_time"] = pj( total_calc_timer.get_total_time() );
    out["total_local_search_time"] = pj( local_search_timer.get_total_time() );

    if(params.sol_rec_freq_ != 0){
        out["iter_solution"] = pj(iter_solution);
    }
    //std::cout << "Final solution: " <<
        //sequence_to_string( global_best.visited_.begin(), global_best.visited_.end() )
        //<< std::endl;
    auto trails = pheromone_memory.get_all();
    auto sum = 0.0;
    int j = 0;
    for (auto &v : trails) {
        sum += v;
        ++j;
    }
    std::cout << "Trails sum: " << sum << std::endl;
    std::cout << "trails.size(): " << trails.size() << std::endl;
    std::cout << "* j = " << j << std::endl;


    return out;
}


template<typename T>
T calc_mean(const std::vector<T> &vec) {
    if (vec.empty()) {
        return (T)0;
    }
    T sum = 0;
    for (auto v : vec) {
        sum += v;
    }
    return sum / vec.size();
}


template<typename Fun>
void run_many(Fun f, uint32_t repetitions, std::map< std::string, std::vector<pj> > &all_results) {
    for (uint32_t i = 0; i < repetitions; ++i) {
        cout << "Starting run " << i << endl;
        auto run_start = std::chrono::steady_clock::now();

        auto res = f();

        auto run_end = std::chrono::steady_clock::now();
        auto run_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
        res["run_elapsed"] = pj(run_elapsed.count() / 1.0e6);
        cout << "Run finished in " << (run_elapsed.count() / 1.0e6) << " seconds" << endl;

        for (auto it : res) {
            auto key = it.first;
            auto val = it.second;
            all_results[key].push_back(val);
        }
    }
}


void print_usage() {
    cout << "Program options:\n"
         << setw(20) << left << " --test file "
         << "path to the test file\n";

    cout << setw(20) << left << " --alg name "
         << "Name of the algorithm to run\n";

    cout << setw(20) << left << " --ants count "
         << "Number of ants\n";

    cout << setw(20) << left << " --iter count "
         << "Number of iterations per run\n";

    cout << setw(20) << left << " --time_limit seconds "
         << "Time limit per single run in seconds\n";

    cout << setw(20) << left << " --eval_per_node count "
         << "Alternative to --iter & --time_limit. Total num. "
            "of costructed sol. equals dimension * eval_per_node / ants \n";

    cout << setw(20) << left << " --runs count "
         << "Number of independent algorithm runs\n";

    cout << setw(20) << left << " --q0 value "
         << "Value of the q0 ACS parameter\n";

    cout << setw(20) << left << " --outdir path "
         << "Path to the results folder\n";

    cout << setw(20) << left << " --ls [0|1] "
         << "Set to 1 if local search should be used\n";

    cout << setw(20) << left << " --gs_cand_size count "
         << "Number of dynamic candidate sets for each city\n";

    cout << setw(20) << left << " --threads_per_block count "
         << "Number of threads per block  (the acs_gpu_wsp version has an additional 32 threads)\n";

    cout.flush();
}


string extract_test_name(string path) {
    auto slash_pos = path.rfind("/");
    auto file_name = (slash_pos == string::npos) ? path : path.substr(slash_pos + 1);
    auto dot_pos = file_name.find(".");
    auto test_name = (dot_pos == string::npos) ? file_name : file_name.substr(0, dot_pos);
    return test_name;
}


int main(int argc, char *argv[]) {
    auto init_start = std::chrono::steady_clock::now();

    CmdOptions options(argv, argv + argc);

    if (options.has_option("--help")) {
        print_usage();
        return EXIT_SUCCESS;
    }

    string outdir = options.get_option_or_die("--outdir");

    const string problem_file{ options.get_option_or_die("--test") };
	//获取TSP问题的信息
    auto problem = read_problem_data_tsp(problem_file);
    const uint32_t dimension = problem.coords_.size();

    vector<vector<double>> dist_matrix = { {} };
    calc_dist_matrix(problem, dist_matrix);

    ACSParams params;
    params.initial_pheromone_ = calc_initial_pheromone(dist_matrix);
	params.outdir = outdir;
	params.test_name = extract_test_name(problem_file);

    auto default_q0 = (dimension - 20.0) / dimension;
    params.q0_ =  options.get_option("--q0", default_q0);
    params.ants_count_ = (uint32_t)options.get_option("--ants", (int)50);

    string alg(options.get_option_or_die("--alg"));
    record("algorithm", alg);

    //--------------------modify by BUPT--------------------------

    //solutions record
    params.sol_rec_freq_ = options.get_option("--sol_rec_freq",1);
    params.phmem_print_ = options.get_option("--ph_print",0);
    params.phmem_print_freq_ = options.get_option("--ph_print_freq",0);
  
    //global search length options
    params.gs_cand_list_size_ = options.get_option("--gs_cand_size",32);
    params.gs_pub_cand_size_ = options.get_option("--gs_pub_cand_size",32);

    record("sol_rec_freq",(int64_t)params.sol_rec_freq_);
    record("phmem_print",(int64_t)params.phmem_print_);
    record("phmem_print_freq",(int64_t)params.phmem_print_freq_);
    record("gs_cand_list_size",(int64_t)params.gs_cand_list_size_);
    record("gs_pub_cand_size",(int64_t)params.gs_pub_cand_size_);

    //----------------------modify end---------------------------
    record("start_date_time", get_current_date_time());
    record("problem_file", problem_file);
    record("problem_size", (int64_t)dimension);
    record("initial_pheromone", params.initial_pheromone_);
    record("q0", params.q0_);
    record("beta", params.beta_);
    record("rho", params.rho_);
    record("phi", params.phi_);
    record("ants_count", (int64_t)params.ants_count_);

    uint32_t bucket_size = 8;
    bucket_size =  (uint32_t)options.get_option("--bucket_size", (int)bucket_size);
    record("bucket_size", (int64_t)bucket_size);

    vector<vector<double>> heuristic_matrix;
    init_heuristic_matrix(dist_matrix, params.beta_, heuristic_matrix);

    vector<vector<double>> total_matrix;

    //Modify by BUPT  初始化候选集 这里是nn_list.
    vector<vector<uint32_t>> nn_lists;
    { 
        auto start = std::chrono::steady_clock::now();
        nn_lists = init_nn_lists(dist_matrix,NN_SIZE);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "init exact nn time: "<< elapsed.count() / 1.0e6 << std::endl;
        record("init_exact_nn_time", elapsed.count() / 1.0e6);
    }
    //Modify end  

    auto start = std::chrono::steady_clock::now();

    shared_ptr<StopCondition> stop_cond;
    double time_limit = options.get_option("--time_limit", 0.0);
    record("time_limit", time_limit);
    int iterations = options.get_option("--iter", 0);
    record("iterations", (int64_t)iterations);
    int eval_per_node = options.get_option("--eval_per_node", 0);
    record("eval_per_node", (int64_t)eval_per_node);

    if (time_limit > 0.0) {
        stop_cond.reset(new TimeoutStopCondition(time_limit));
    } else if (iterations > 0) {
        stop_cond.reset(new FixedIterationsStopCondition((uint32_t)iterations));
    } else if (eval_per_node > 0) {
        iterations = (dimension * (size_t)eval_per_node) / params.ants_count_;
        stop_cond.reset(new FixedIterationsStopCondition((uint32_t)iterations));
    } else {
        cerr << "Number of iterations or time limit is required" << endl;
        return EXIT_FAILURE;
    }

    auto init_end = std::chrono::steady_clock::now();
    auto init_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    //初始化时间
    record("initialization_time", init_elapsed.count() / 1.0e6);
    std::cout << "initialization_time: " << init_elapsed.count() / 1.0e6 << std::endl;

    int pher_mem_update_freq = options.get_option("--pher_mem_update_freq", 1);
    int threads_per_block = options.get_option("--threads_per_block", 32);
    record("pher_mem_update_freq", (int64_t)pher_mem_update_freq);
    record("threads_per_block", (int64_t)threads_per_block);

    auto get_distance = [&] (int a, int b) {
        return dist_matrix[a][b];
    };

    auto calc_route_len = [&] (const std::vector<uint32_t> &route) {
        return eval_route(dist_matrix, route);
    };

    int ls_option = options.get_option("--ls", 0);
    bool use_local_search = (ls_option == 1);
    params.use_local_search_ = use_local_search;
    record("local_search", (int64_t)ls_option);

    auto local_search = [&] (std::vector<uint32_t> &route) -> void {
        opt2(route, nn_lists, get_distance, calc_route_len);
    };

    uint32_t runs = (uint32_t)options.get_option("--runs", 1);
    std::map< std::string, std::vector<pj> > all_results;

    if (alg == "acs") {
        auto f = [&]() {
            init_total_matrix(dist_matrix, params.beta_, params.initial_pheromone_, total_matrix);
            if (use_local_search) {
                return run_acs(dist_matrix, heuristic_matrix, total_matrix,
                               nn_lists, g_rng,
                               params, stop_cond.get(), local_search);
            } else {
                return run_acs(dist_matrix, heuristic_matrix, total_matrix,
                               nn_lists, g_rng,
                               params, stop_cond.get(), nullptr);
            }
        };
        run_many(f, runs, all_results);
    } else if (alg == "acs_spm") {
        auto f = [&]() {
            return run_acs_spm(dist_matrix, heuristic_matrix, nn_lists,
                               g_rng, params, stop_cond.get(), bucket_size);
        };
        run_many(f, runs, all_results);
    } else if (alg == "acs_spm_gpu") {
        threads_per_block = 32;
        TSPData problem { dist_matrix, heuristic_matrix, nn_lists, (uint32_t)nn_lists.size() };

        auto f = [&]() {
            return gpu_run_acs_spm( problem,
                                    g_rng, params, pher_mem_update_freq, threads_per_block,
                                    stop_cond.get() );
        };
        run_many(f, runs, all_results);
    } else if (alg == "acs_gpu") {
        TSPData problem { dist_matrix, heuristic_matrix, nn_lists, (uint32_t)nn_lists.size() };

        auto f = [&]() {
            return gpu_run_acs( problem,
                                g_rng, params, pher_mem_update_freq, threads_per_block,
                                stop_cond.get() );
        };
        run_many(f, runs, all_results);
    } else if (alg == "acs_gpu_atomic") {
        TSPData problem { dist_matrix, heuristic_matrix, nn_lists, (uint32_t)nn_lists.size() };

        auto f = [&]() {
            return gpu_run_acs_atomic( problem, g_rng, params, pher_mem_update_freq,
                                       threads_per_block, stop_cond.get() );
        };
        run_many(f, runs, all_results);
    } else if (alg == "acs_gpu_alt") {
        TSPData problem { dist_matrix, heuristic_matrix, nn_lists, (uint32_t)nn_lists.size() };

        auto f = [&]() {
            return gpu_run_acs_alt( problem, g_rng, params, pher_mem_update_freq,
                                    threads_per_block, stop_cond.get() );
        };
        run_many(f, runs, all_results);
    } 
    //////////////////////////////////////////////////////////////////////////////////////
    //Modify by BUPT
    else if(alg == "acs_gpu_wsp"){
        TSPData problem { dist_matrix, heuristic_matrix, nn_lists, (uint32_t)nn_lists.size() };

        auto f = [&]() {
            return gpu_run_acs_wsp( problem, g_rng, params, pher_mem_update_freq,
                                    threads_per_block, stop_cond.get() );
        };
        run_many(f, runs, all_results);
    //Modify end
    //////////////////////////////////////////////////////////////////////////////////////
    } else {
        cerr << "Unknown algorithm [" << alg << "]." << endl;
        return EXIT_FAILURE;
    }

    // Record all algorithm's results
    for (auto it : all_results) {
        auto key = it.first;
        auto val = it.second;
        record(key, val);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Calc. time: " << elapsed.count() / 1.0e6 << " sec" << std::endl;
    record("calc_time", elapsed.count() / 1.0e6);
    // Record command line arguments
    string cmd_args;
    for (int i = 0; i < argc; ++i) {
        cmd_args += argv[i];
        cmd_args += " ";
    }
    record("cmd_args", cmd_args);
    record("end_date_time", get_current_date_time());
    record("experiment", options.get_option("--experiment", "-"));
    record("comment", options.get_option("--comment", "-"));
    record("select_among_others_counter", (int64_t)select_among_others_counter);
    cout << "select_among_others_counter: " << select_among_others_counter << endl;


    auto test_name = extract_test_name(problem_file);
    record("test_name", test_name);

    // Assuming linux OS
    string outfile_path = outdir + "/"
        + "[" + alg + "]" + test_name + "_" 
        + get_current_date_time("%G-%m-%d_%H_%M_%S")
        + "-" + to_string(getpid())
        + ".js";

    cout << outfile_path << endl;

    ofstream outfile(outfile_path);
    if (outfile.is_open()) {
        outfile << global_record_to_string();
        outfile.close();
    }
    return EXIT_SUCCESS;
}
