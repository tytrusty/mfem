#pragma once

#include <EigenTypes.h>
#include <chrono>
#include <unordered_map>

namespace mfem {

  class Timer {

    using Time = std::chrono::high_resolution_clock;
    using T = std::tuple<std::chrono::time_point<Time>, double, int>;

  public:
    // Start timer for a key (creates entry if does not exist)
    void start(const std::string& key);

    // Records elapsed time for given key
    void stop(const std::string& key);

    double total(const std::string& key)  const;

    double average(const std::string& key)  const;

    void print() const;

    void reset() {
      times_.clear();
    }

  private:
    // For each key, store the clock, total time, and # of measurements
    std::map<std::string, T> times_;	
  };


  struct OptimizerData {
    

    OptimizerData(std::string output_filename)
        : output_filename_(output_filename) {
    }

    OptimizerData() : output_filename_("../data/output/results.mat") {
    }

    virtual void clear();
    virtual void write() const;
    virtual void print_data(bool print_timing = true) const;
    void add(const std::string& key, double value);

    std::string output_filename_;
    std::map<std::string, std::vector<double>> map_;	
    size_t min_length_ = 11;
    Timer timer;

  }; 

}