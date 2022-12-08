#include "optimizer_data.h"
#include <igl/writeDMAT.h>
#include <iostream>
#include <iomanip>      // std::setw

using namespace mfem;
using namespace Eigen;

void OptimizerData::clear() {
  timer.reset();
  map_.clear();
}

void OptimizerData::write() const {
}

void OptimizerData::add(const std::string& key, double value) {
  auto it = map_.find(key);
  if (it == map_.end()) {
    std::vector<double> new_values(1, value);
    map_[key] = new_values;
  } else {
    std::vector<double>& values = it->second;
    values.push_back(value);
  }
}

void OptimizerData::print_data(bool print_timing) const {

  // Copy cout state so we can restore it later
  std::ios cout_state(nullptr);
  cout_state.copyfmt(std::cout);

  // Header Top
  std::cout << "┌─";
  for (auto it = map_.begin(); it != map_.end(); ) {
    int len = std::max(min_length_, it->first.length());
    for (int i = 0; i < len; ++i) {
      std::cout << "─";
    }
    if (++it == map_.end())
      std::cout << "─┐\n";
    else
      std::cout << "─┬─";
  }

  // Labels
  std::cout << "│ ";
  for (auto it = map_.begin(); it != map_.end(); ) {
    std::cout << it->first;
    int padding = std::max(min_length_, it->first.length())
        - it->first.length();
    if (padding > 0) {
      for (int i = 0; i < padding; ++i) {
        std::cout << " ";
      }
    }

    if (++it == map_.end())
      std::cout << " │\n";
    else
      std::cout << " │ ";
  }

  // Header Bottom
  std::cout << "├─";
  for (auto it = map_.begin(); it != map_.end(); ) {
    int len = std::max(min_length_, it->first.length());
    for (int i = 0; i < len; ++i) {
      std::cout << "─";
    }
    if (++it == map_.end())
      std::cout << "─┤\n";
    else
      std::cout << "─┼─";
  }


  // Data
  size_t max_size = 0;
  for (auto it = map_.begin(); it != map_.end(); ++it) {
    max_size = std::max(max_size, it->second.size());
  }

  for (size_t i = 0; i < max_size; ++i) {
    std::cout << "│ ";

    for (auto it = map_.begin(); it != map_.end(); ++it) {

      if (it->first == " Iteration") {
        std::cout << std::defaultfloat;
      } else {
        std::cout << std::scientific;
      }
      int len = std::max(min_length_, it->first.length());
      std::cout << std::setprecision(5);

      std::cout << std::setw(len) << it->second[i];
      std::cout << " │ ";
    }
    std::cout << std::endl;
  }

  // Footer
  std::cout << "└─";
  for (auto it = map_.begin(); it != map_.end(); ) {
    int len = std::max(min_length_, it->first.length());
    for (int i = 0; i < len; ++i) {
      std::cout << "─";
    }
    if (++it == map_.end())
      std::cout << "─┘\n";
    else
      std::cout << "─┴─";
  }

  if (print_timing) {
    timer.print();
  }
  
  // Restore cout format
  std::cout.copyfmt(cout_state);
}

void Timer::start(const std::string& key) {

  auto it = times_.find(key);

  if (it == times_.end()) {
    times_[key] = std::make_tuple(Time::now(), 0.0, 0);
  } else {
    T& tup = times_[key];
    std::get<0>(tup) = Time::now();
  }
}

void Timer::stop(const std::string& key) {
  auto end = Time::now();
  auto it = times_.find(key);

  if (it == times_.end()) {
    std::cerr << "Invalid timer key: " << key << std::endl;
  } else {
    T& tup = it->second;
    std::chrono::duration<double, std::milli> fp_ms = end - std::get<0>(tup);
    std::get<1>(tup) += fp_ms.count(); // add to total time
    std::get<2>(tup) += 1;             // increment measurement count
  }
}

double Timer::total(const std::string& key) const{
  auto it = times_.find(key);
  if (it != times_.end()) {
    auto p = it->second;
    return std::get<1>(p);
  }
  return 0;
}

double Timer::average(const std::string& key) const {
  auto it = times_.find(key);
  if (it != times_.end()) {
    auto p = it->second;
    return std::get<1>(p) / std::get<2>(p);
  }
  return 0;
}

void Timer::print() const {
  std::cout << "Timing (in ms): " << std::endl;
  auto it = times_.begin();
  while(it != times_.end()) {
    std::string key = it->first;
    const T& tup = it->second;
    double t = std::get<1>(tup);
    int n = std::get<2>(tup);

    std::cout << "  [" << std::setw(10) << key << "] "
        << std::fixed << " Avg: " << std::setw(10) << t/((double) n)
        << "   Total: " << std::setw(10) << t << std::endl;
    ++it;
  }
}