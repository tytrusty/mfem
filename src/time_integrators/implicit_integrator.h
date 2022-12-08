#pragma once

#include "EigenTypes.h"
#include <deque>

namespace mfem {

  class ImplicitIntegrator {
  public:

    ImplicitIntegrator(Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        : h_(h) {}
        
    virtual ~ImplicitIntegrator() = default;
    virtual Eigen::VectorXd x_tilde() const = 0;
    virtual double dt() const = 0;
    virtual void update(const Eigen::VectorXd& x) = 0;
    virtual void reset() {
      x_prevs_.clear();
      v_prevs_.clear();
    }
    virtual const Eigen::VectorXd& x_prev() const {
      return x_prevs_.front();
    }
    virtual const Eigen::VectorXd& v_prev() const {
      return v_prevs_.front();
    }
  protected:

    double h_;
    std::deque<Eigen::VectorXd> x_prevs_;
    std::deque<Eigen::VectorXd> v_prevs_;
  };

}
