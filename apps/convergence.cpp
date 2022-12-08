
// libigl
#include <igl/boundary_facets.h>
#include <igl/IO>

#include "args/args.hxx"
#include "json/json.hpp"

#include "factories/optimizer_factory.h"

using namespace Eigen;
using namespace mfem;

const int DIM = 3;

std::shared_ptr<Optimizer<DIM>> optimizer;
std::shared_ptr<Optimizer<DIM>> newton_optimizer;
std::shared_ptr<Mesh> mesh;

double min_decrement;
std::vector<double> curr_decrements;

static double newton_decrement(const SimState<DIM>& state) {
  // Get full configuration vector
  VectorXd x = state.x_->value();
  state.x_->unproject(x);

  SparseMatrix<double, RowMajor> lhs;
  VectorXd rhs;

  const SimState<DIM>& newton_state = newton_optimizer->state();

  newton_state.x_->value() = state.x_->value();
  // Add LHS and RHS from each variable
  lhs = newton_state.x_->lhs();
  rhs = newton_state.x_->rhs();

  for (auto& var : newton_state.vars_) {
    var->update(x, state.x_->integrator()->dt());
    lhs += var->lhs();
    rhs += var->rhs();
  }

  //newton_state.solver_->compute(lhs);
  //VectorXd dx = newton_state.solver_->solve(rhs);

  // Use infinity norm of deltas as termination criteria
  // double decrement = dx.lpNorm<Infinity>();
  double decrement = rhs.norm();
  return decrement;
}

void callback(const SimState<DIM>& state) {
  double decrement = newton_decrement(state);
  // if (decrement < min_decrement)
    min_decrement = decrement;
    curr_decrements.push_back(decrement);
  std::cout << "decrement: " << decrement << std::endl;
}

int main(int argc, char **argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Mixed FEM");
  args::Positional<std::string> inFile(parser, "json", "input scene json file");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  std::string filename = args::get(inFile);

  // Read and parse file
  std::ifstream input(filename);
  if (!input.good()) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return -1;
  }
  nlohmann::json args = nlohmann::json::parse(input);
  OptimizerFactory<DIM> factory;

  int M = args["max_newton_iterations"];
  int N = 6*2;
  // VectorXd exps(N);// = VectorXd::LinSpaced(N, 8, 14);
  VectorXd exps = VectorXd::LinSpaced(N, 8, 14);
  // exps << 8, 10, 11, 12, 13, 14;
  std::cout << " exps : " << exps << std::endl;
  VectorXd min_decrements(N);
  VectorXd ym = exps;
  MatrixXd decrements;
  std::cout << "M: " << M << std::endl;
  decrements.resize(N, M);

  for (int i = 0; i < N; ++i) {
    // Read and parse file
    std::ifstream input(filename);
    nlohmann::json args = nlohmann::json::parse(input);

    curr_decrements.clear();
    min_decrement = std::numeric_limits<double>::max();
    args["material_models"][1]["youngs_modulus"] = std::pow(10,exps(i));
    std::cout << std::pow(10,exps(i)) << std::endl;
    ym(i) = std::pow(10,exps(i));
    SimState<3> state;
    state.load(args);
    optimizer = factory.create(state.config_->optimizer, state);
    optimizer->reset();
    optimizer->callback = callback;
    
    // Switch to newton FEM
    // Note just supports stretch right now
    args.erase("mixed_variables");
    args["variables"] = {"stretch"};

    SimState<DIM> newton_state;
    newton_state.load(args);
    newton_optimizer = factory.create(newton_state.config_->optimizer, newton_state);
    newton_optimizer->reset();

    double decrement = newton_decrement(optimizer->state());
    std::cout << "decrement: " << decrement << std::endl;
    optimizer->step();
    min_decrements(i) = min_decrement;
    decrements.row(i) = Map<RowVectorXd>(curr_decrements.data(), M);
  }
  std::cout << "min_decrements: " << min_decrements << std::endl;
  igl::writeDMAT("../output/convergence.dmat", decrements);
  igl::writeDMAT("../output/finalconvergence_converge.dmat", min_decrements);
  igl::writeDMAT("../output/ym_values.dmat", ym);

  return 0;
}

