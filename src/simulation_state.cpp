#include "simulation_state.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include "config.h"

#include "mesh/mesh.h"
#include "mesh/tri2d_mesh.h"
#include "mesh/meshes.h"
#include "utils/linear_blend_skinning.h"

// Factories
#include "factories/linear_solver_factory.h"
#include "factories/variable_factory.h"
#include "factories/optimizer_factory.h"
#include "factories/integrator_factory.h"
#include "factories/material_model_factory.h"
#include "factories/boundary_condition_factory.h"

// libigl
#include <igl/IO>
#include <igl/remove_unreferenced.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>

using json = nlohmann::json;
using namespace mfem;
using namespace Eigen;
namespace fs = std::filesystem;

namespace {

  template<typename T>
  void read_and_assign(const nlohmann::json& args, const std::string& key,
      T& value) {
    if (const auto& it = args.find(key); it != args.end()) {
      value = it->get<T>();
    }
  }
  
}

// Read 2D Mesh
// TODO separate 3D or maybe even templated loading
template <int DIM>
void SimState<DIM>::load_mesh(const std::string& path, MatrixXd& V, MatrixXi& T) {
  if constexpr (DIM == 2) {
    MatrixXd NV;
    MatrixXi NT;
    igl::read_triangle_mesh(path,V,T);
    VectorXi VI,VJ;
    igl::remove_unreferenced(V,T,NV,NT,VI,VJ);
    V = NV;
    T = NT;

    // Truncate z data
    MatrixXd tmp;
    tmp.resize(V.rows(),2);
    tmp.col(0) = V.col(0);
    tmp.col(1) = V.col(1);
    V = tmp;
  } else {
    MatrixXd F;
    igl::readMESH(path, V, T, F);
    double fac = V.maxCoeff();
    V.array() /= fac;
    std::cout << "normalizing vertices: SimState<DIM>::load_mesh" << std::endl;
  }
}

template <int DIM>
bool SimState<DIM>::load(const std::string& json_file) {
  // Confirm file is .json
  if (std::filesystem::path(json_file).extension() != ".json") {
    std::cerr << "File: " << json_file << " needs to be json" << std::endl;
    return false;
  }

  // Read and parse file
  json args;
  std::ifstream input(json_file);
  if (input.good()) {
    args = json::parse(input);
  } else {
    std::cerr << "Unable to open file: " << json_file << std::endl;
    return false;
  }
  return load(args);
}

template <int DIM>
bool SimState<DIM>::load(const nlohmann::json& args) {

  config_ = std::make_shared<SimConfig>();
  load_params(args);

  OptimizerFactory<DIM> optimizer_factory;
  if (const auto& it = args.find("optimizer"); it != args.end()) {
    std::string name = it->get<std::string>();
    config_->optimizer = optimizer_factory.type_by_name(name);
  }

  IntegratorFactory integrator_factory;
  if (const auto& it = args.find("time_integrator"); it != args.end()) {
    std::string name = it->get<std::string>();
    config_->ti_type = integrator_factory.type_by_name(name);
  }

  MaterialModelFactory material_factory;
  std::vector<std::shared_ptr<MaterialModel>> materials;
  std::vector<std::shared_ptr<MaterialConfig>> mat_configs;
  if (const auto& obj_it = args.find("material_models"); obj_it != args.end())
  {
    for (const auto& obj : *obj_it) {
      std::shared_ptr<MaterialConfig> cfg = std::make_shared<MaterialConfig>();

      read_and_assign(obj, "youngs_modulus", cfg->ym);
      read_and_assign(obj, "poissons_ratio", cfg->pr);
      read_and_assign(obj, "density", cfg->density);

      if (const auto& it = obj.find("energy"); it != obj.end()) {
        std::string name = it->get<std::string>();
        cfg->material_model = material_factory.type_by_name(name);
      }
      Enu_to_lame(cfg->ym, cfg->pr, cfg->la, cfg->mu);
      mat_configs.push_back(cfg);
      materials.push_back(material_factory.create(cfg->material_model, cfg));
    }
  } else {
    mat_configs = {std::make_shared<MaterialConfig>()};
    materials = {material_factory.create(
        mat_configs[0]->material_model, mat_configs[0])};
  }
  material_models_ = materials;

  // TODO mesh factory <DIM>
  std::vector<std::shared_ptr<Mesh>> meshes;
  if (const auto& obj_it = args.find("objects"); obj_it != args.end()) {
    for (const auto& obj : *obj_it) {

      std::string path;
      std::vector<double> offset = {0.0, 0.0, 0.0};
      std::vector<double> transformation;
      std::vector<double> initial_velocity;
      uint idx = 0;
      VectorXi material_ids;
      bool has_material_ids = false;

      // Get File path
      if (const auto& it = obj.find("path"); it != obj.end()) {
        path = it->get<std::string>();
      } else {
        std::cerr << "Object missing path!" << std::endl;
        return false;
      }

      if (const auto& it = obj.find("offset"); it != obj.end()) {
        offset = it->get<std::vector<double>>();
        assert(offset.size() == 3);
      }

      if (const auto& it = obj.find("transformation"); it != obj.end()) {
        transformation = it->get<std::vector<double>>();
        assert(transformation.size() == DIM*DIM);
      }

      if (const auto& it = obj.find("initial_velocity"); it != obj.end()) {
        initial_velocity = it->get<std::vector<double>>();
        assert(initial_velocity.size() == 3);
      }

      
      if (const auto& it = obj.find("material_ids"); it != obj.end()) {
        std::string mat_path = it->get<std::string>();
        has_material_ids = true;
        igl::readDMAT(mat_path, material_ids);
      } else {
        // If no per-tetrahedron material ids, look for material index for entire
        // mesh
        if (const auto& it = obj.find("material_index"); it != obj.end()) {
          idx = it->get<uint>();
          assert(idx < mat_configs.size());
        }
      }

      MatrixXd V, V0;
      MatrixXi T;
      load_mesh(path, V, T);
      V0 = V;

      // Apply translation offset
      for (int i = 0; i < V.cols(); ++i) {
        V.col(i).array() += offset[i];
      }
      
      // Apply transformation matrix
      if (transformation.size() > 0) {
        Matrix<double,1,DIM> centroid = V.colwise().sum() / V.rows();
        Matrix<double,DIM,DIM> T = Map<Matrix<double,DIM,DIM>>(
            transformation.data());
        V = ((V.rowwise()-centroid)*T.transpose()).rowwise() + centroid;
      }

      if constexpr (DIM == 2) {
        if (has_material_ids) {
          std::cout << "Missing heterogeneous 2Dtrimesh support!" << std::endl;
          meshes.push_back(std::make_shared<Tri2DMesh>(V, T, materials[idx]));
        } else {
          meshes.push_back(std::make_shared<Tri2DMesh>(V, T, materials[idx]));
        }
      } else {
        if (has_material_ids) {
          meshes.push_back(std::make_shared<TetrahedralMesh>(V, T, material_ids,
              materials));
        } else {
          meshes.push_back(std::make_shared<TetrahedralMesh>(V, T,
              materials[idx]));
        }
      }

      if (initial_velocity.size() > 0) {
        RowVector<double,DIM> v;
        for (int i = 0; i < DIM; ++i) {
          v(i) = initial_velocity[i];
        }
        meshes.back()->initial_velocity_ = v.replicate(V.rows(),1);
      }

      // Skinning mesh
      if (const auto& it = obj.find("skinning_mesh"); it != obj.end()) {
        std::string skinning_path = it->get<std::string>();
        SkinningData sd;
        sd.empty_ = false;
        igl::readOBJ(skinning_path,
            sd.V_, sd.TC_, sd.N_, sd.F_, sd.FTC_, sd.FN_);
        double fac = sd.V_.maxCoeff();
        sd.V_.array() /= fac;

        linear_blend_skinning(V0, T, sd.V_, sd.W_); 
        meshes.back()->skinning_data_ = sd;
      }

      // After loading and creating mesh, check if any boundary conditions
      // are specified.
      if (const auto& it = obj.find("boundary_condition"); it != obj.end()) {
        BoundaryConditionFactory factory;
        BoundaryConditionConfig cfg;

        std::string name;
        read_and_assign(*it, "type", name);
        cfg.type = factory.type_by_name(name);

        read_and_assign(*it, "axis", cfg.axis);
        read_and_assign(*it, "ratio", cfg.ratio);
        read_and_assign(*it, "velocity", cfg.velocity);
        meshes.back()->bc_config_ = cfg;
      }
    }
  }
  mesh_ = std::make_shared<Meshes>(meshes);
  x_ = std::make_unique<Displacement<DIM>>(mesh_, config_);

  if (const auto& it = args.find("initial_state"); it != args.end()) {
    
    // Check for initial vertex positions
    if (const auto& path_it = it->find("x_path"); path_it != it->end()) {
      std::string x_path = path_it->get<std::string>();
      if (fs::path(x_path).extension() == ".dmat") {
        MatrixXd x;
        igl::readDMAT(x_path, x);
        mesh_->Vinit_ = x;
        mesh_->V_ = x;

        size_t start_V = 0;
        for (size_t i = 0; i < meshes.size(); ++i) {
          size_t sz_V = meshes[i]->V_.rows();
          meshes[i]->V_ = x.block(start_V, 0, sz_V, x.cols());
          meshes[i]->Vinit_ = meshes[i]->V_;
          start_V += sz_V;
        }

        assert(mesh_->V_.rows() == x.rows() && mesh_->V_.cols() == x.cols());
      } else {
        std::cerr << "initial_state x_path must be a dmat file" << std::endl;
      }
    }

    // Check for initial velocities
    if (const auto& path_it = it->find("v_path"); path_it != it->end()) {
      std::string v_path = path_it->get<std::string>();
      if (fs::path(v_path).extension() == ".dmat") {
        MatrixXd v;
        igl::readDMAT(v_path, v);
        mesh_->initial_velocity_ = v;
        
        size_t start_V = 0;
        for (size_t i = 0; i < meshes.size(); ++i) {
          size_t sz_V = meshes[i]->V_.rows();
          meshes[i]->initial_velocity_ = v.block(start_V, 0, sz_V, v.cols());
          start_V += sz_V;
        }

        assert(mesh_->V_.rows() == v.rows() && mesh_->V_.cols() == v.cols());
      } else {
        std::cerr << "initial_state v_path must be a dmat file" << std::endl;
      }
    }
  }


  MixedVariableFactory<DIM> mixed_variable_factory;
  std::set<VariableType> mixed_variables;
  if (const auto& it = args.find("mixed_variables"); it != args.end()) {
    for(const auto& name : it->get<std::vector<std::string>>()) {
      mixed_variables.insert(mixed_variable_factory.type_by_name(name));
    }
  }
  config_->mixed_variables = mixed_variables;
  for (VariableType type : config_->mixed_variables) {
    mixed_vars_.push_back(mixed_variable_factory.create(type, mesh_, config_));
  }

  VariableFactory<DIM> variable_factory;
  std::set<VariableType> variables;
  if (const auto& it = args.find("variables"); it != args.end()) {
    for(const auto& name : it->get<std::vector<std::string>>()) {
      variables.insert(variable_factory.type_by_name(name));
    }
  }
  config_->variables = variables;
  for (VariableType type : config_->variables) {
    vars_.push_back(variable_factory.create(type, mesh_, config_));
  }

  LinearSolverFactory<DIM> solver_factory;
  if (const auto& it = args.find("linear_solver"); it != args.end()) {
    std::string name = it->get<std::string>();
    config_->solver_type = solver_factory.type_by_name(name);
  }
  return true;
}

template <int DIM>
void SimState<DIM>::load_params(const nlohmann::json& args) {


  if (const auto& it = args.find("body_force"); it != args.end()) {
    std::vector<float> ext = it->get<std::vector<float>>();
    assert(ext.size() == 3);
    config_->ext[0] = ext[0];
    config_->ext[1] = ext[1];
    config_->ext[2] = ext[2];
  }

  read_and_assign(args, "dt", config_->h);
  read_and_assign(args, "print_timing", config_->show_timing);
  read_and_assign(args, "print_stats", config_->show_data);
  read_and_assign(args, "enable_ccd", config_->enable_ccd);
  read_and_assign(args, "dhat", config_->dhat);
  read_and_assign(args, "kappa", config_->kappa);
  read_and_assign(args, "max_newton_iterations", config_->outer_steps);
  read_and_assign(args, "max_linesearch_iterations", config_->ls_iters);
  read_and_assign(args, "max_iterative_solver_iters",
      config_->max_iterative_solver_iters);
  read_and_assign(args, "iterative_solver_tolerance", config_->itr_tol);
}

template class mfem::SimState<3>; // 3D
template class mfem::SimState<2>; // 2D
