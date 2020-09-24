#ifndef STOPCONDITION
#define STOPCONDITION

#include <cstdint>
#include <chrono>

class StopCondition {
public:
    /**
     * This should be called before the first use of the other methods.
     */
    virtual void init() = 0;

    /**
     * This should be called after each iteration.
     */
    virtual void next_iteration() = 0;

    /**
     * This method returns true when the stopping criterion has been reached,
     * i.e. the maximum number of iteration has been performed or the time limit
     * has been exceeded
     */
    virtual bool is_reached() = 0;

    /**
     * Returns current iteration number
     */
    virtual uint32_t get_iteration() const = 0;

    /**
     * Returns max iteration number
     */
    virtual uint32_t get_max_iterations() const {return 0;};

};



class FixedIterationsStopCondition : public StopCondition {
public:

    FixedIterationsStopCondition(const uint32_t max_iterations) :
        iteration_(0),
        max_iterations_(max_iterations) {
    }

    virtual void init() {
        iteration_ = 0;
    }

    virtual void next_iteration() {
        if (iteration_ < max_iterations_) {
            ++iteration_;
        }
    }

    virtual bool is_reached() {
        return iteration_ == max_iterations_;
    }

    virtual uint32_t get_iteration() const { return iteration_; }
    
    virtual uint32_t get_max_iterations() const {return max_iterations_;};

private:
    uint32_t iteration_;
    uint32_t max_iterations_; 
};



class TimeoutStopCondition : public StopCondition {
public:

    TimeoutStopCondition(double max_seconds);

    virtual void init();

    virtual void next_iteration();

    virtual bool is_reached();

    virtual uint32_t get_iteration() const { return iteration_; }
private:
    double max_seconds_;
    std::chrono::steady_clock::time_point start_time_;  
    uint32_t iteration_;
};

#endif
