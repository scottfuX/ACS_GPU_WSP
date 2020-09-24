#include <algorithm>
#include "stopcondition.h"


TimeoutStopCondition::TimeoutStopCondition(double max_seconds) :
    max_seconds_(std::max(0.0, max_seconds)),
    iteration_(0) {
}


void TimeoutStopCondition::init() {
    start_time_ = std::chrono::steady_clock::now();
    iteration_ = 0;
}


void TimeoutStopCondition::next_iteration() {
    ++iteration_;
}


bool TimeoutStopCondition::is_reached() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
    return (elapsed.count() / 1.0e6) >= max_seconds_;
}

