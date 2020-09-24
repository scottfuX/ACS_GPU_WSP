#ifndef COMMON_H
#define COMMON_H

/* Use nice picjson library with int64_t support */
#define PICOJSON_USE_INT64
#include "picojson.h"

/* For convenience define alias for picojson::value */
typedef picojson::value pj;

#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <chrono>

extern std::map<std::string, pj> global_record;

void record(std::string key, int64_t value);

void record(std::string key, double value);

void record(std::string key, const std::string value);

void record(std::string key, const std::vector<int> &values);

void record(std::string key, const std::vector<float> &values);

void record(std::string key, const std::vector<double> &values);

void record(std::string key, const std::vector<pj> &values);

void record(std::string key, const std::map<std::string, pj> &dict);

std::string global_record_to_string();

/* 
 * Returns string with current date/time. 
 * Default format is YYYY-MM-DD.HH:mm:ss
 */
const std::string get_current_date_time(std::string format="%Y-%m-%d.%X");


template<typename iter_t>
static std::string
sequence_to_string(iter_t iter, iter_t end, std::string sep = " ") {
    std::ostringstream out;
    bool first = true;
    for ( ; iter != end; ++iter) {
        if (!first) {
            out << sep;
        } else {
            first = false;
        }
        out << *iter;
    }
    return out.str();
}


/* 
 * Simple struct to parse command line options
 */
struct CmdOptions {
    char **begin_;
    char **end_;

    CmdOptions(char **begin, char **end) :
        begin_(begin), end_(end) {}


    const char* get_option(const std::string & option, const std::string &default_value="") const {
        char ** itr = std::find(begin_, end_, option);
        if (itr != end_ && ++itr != end_) {
            return *itr;
        }
        return default_value.size() > 0 ? default_value.c_str() : nullptr;
    }


    double get_option(const std::string & option, double default_value) const {
        return has_option(option) ? std::stof(get_option(option)) : default_value;
    }


    double get_option(const std::string & option, int default_value) const {
        return has_option(option) ? std::stoi(get_option(option)) : default_value;
    }


    const char* get_option_or_die(const std::string & option) const {
        if (!has_option(option)) {
            std::cerr << "\n" << option << " option is required" << std::endl;
            exit(EXIT_FAILURE);
        }
        return get_option(option);
    }


    bool has_option(const std::string& option) const {
        return std::find(begin_, end_, option) != end_;
    }
};


/*
 * A helper struct to simplify the timing of specified program fragments
 * execution.
 */
struct IntervalTimer {

    void start_interval() {
        interval_start_ = std::chrono::steady_clock::now();
    }

    void stop_interval() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - interval_start_);
        total_ += (double)elapsed.count();
    }

    void reset() { total_ = 0; }

    /*
     * Returns total time in seconds
     */
    double get_total_time() const { return total_ / 1.0e6; }

    double get_total_time_ms() const { return total_ / 1.0e3; }

    std::chrono::time_point<std::chrono::steady_clock> interval_start_;
    double total_ = 0.0;
};


int bupt_outfile_app_newline(std::string filename,std::string str);
int bupt_outfile_app_newline(std::string filename,const char* str);
int bupt_outfile_app_line(std::string filename,std::string str);


void bupt_chk_and_del(std::string str_title);

int bupt_outfile_app(std::string filename,std::vector<float> &vec);
int bupt_outfile_app(std::string filename,std::vector<double> &vec);
int bupt_outfile_out(std::string filename,std::vector<float> &vec);
int bupt_outfile_out(std::string filename,std::vector<double> &vec);

#endif
