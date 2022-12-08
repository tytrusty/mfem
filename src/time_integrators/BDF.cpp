#include "BDF.h"

using namespace mfem;
using namespace Eigen;

template <int I>
Eigen::VectorXd BDF<I>::x_tilde() const {
  return weighted_sum(x_prevs_) + dt() * weighted_sum(v_prevs_);
}

template <int I>
double BDF<I>::dt() const {
  return beta() * h_;
}

template <int I>
void BDF<I>::update(const VectorXd& x) {
  VectorXd wx = weighted_sum(x_prevs_);

  x_prevs_.push_front(x);
  v_prevs_.push_front((x - wx) / dt());

  if (x_prevs_.size() > I) {
    x_prevs_.pop_back();
    v_prevs_.pop_back();
  }
}

template <int I>
VectorXd BDF<I>::weighted_sum(const std::deque<VectorXd>& x) const {
  const std::array<double,I>& a = alphas();

  assert(x.size() > 0 && x.size() <= I);

  VectorXd wx = VectorXd::Zero(x.front().size());

  for (size_t i = 0; i < x.size(); ++i) {
    wx += a[i] * x[i];
  }
  return wx;
}


// Specializations for each BDF integrator
template <>
constexpr std::array<double,1> BDF<1>::alphas() const {
  return {1.0};
}

template <>
constexpr std::array<double,2> BDF<2>::alphas() const {
  return {4.0 / 3.0, -1.0 / 3.0};
}

template <>
constexpr std::array<double,3> BDF<3>::alphas() const {
  return {18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0};
}

template <>
constexpr std::array<double,4> BDF<4>::alphas() const {
  return {48.0 / 25.0, -36.0 / 25.0, 16.0 / 25.0, -3.0 / 25.0};
}

template <>
constexpr std::array<double,5> BDF<5>::alphas() const {
  return {300.0 / 137.0, -300.0 / 137.0, 200.0 / 137.0, -75.0 / 137.0,
      12.0 / 137.0};
}

template <>
constexpr std::array<double,6> BDF<6>::alphas() const {
  return {360.0 / 147.0, -450.0 / 147.0, 400.0 / 147.0, -225.0 / 147.0,
    72.0 / 147.0, -10.0 / 147.0};
}

template <int I>
constexpr double BDF<I>::beta() const {
  switch(I) {
    case 1:
      return 1.0;
    case 2:
      return 2.0 / 3.0;
    case 3:
      return 6.0 / 11.0;
    case 4:
      return 12.0 / 25.0;
    case 5:
      return 60.0 / 137.0;
    case 6:
      return 60.0 / 147.0;
  }
}

template class mfem::BDF<1>;
template class mfem::BDF<2>;
template class mfem::BDF<3>;
template class mfem::BDF<4>;
template class mfem::BDF<5>;
template class mfem::BDF<6>;
