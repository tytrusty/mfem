#pragma once

#include "finitediff.hpp"
#include "igl/readMESH.h"
//#include "simulator.h"
#include "optimizers/mixed_alm_optimizer.h"
#include "energies/stable_neohookean.h"
#include "mesh/tet_mesh.h"

namespace Test {

  using namespace Eigen;
  using namespace mfem;
  using namespace fd;

  template <class T = MixedALMOptimizer>
  struct App {

    App() {

      std::string filename = "../models/two_tets.mesh";
      //std::string filename = "../models/tet.mesh";
      MatrixXd meshV;
      MatrixXi meshF;
      MatrixXi meshT;

      // Read the mesh
      igl::readMESH(filename, meshV, meshT, meshF);
      double fac = meshV.maxCoeff();
      meshV.array() /= fac;


      // Initialize simulator
      config = std::make_shared<SimConfig>();
      config->show_data = false;
      config->show_timing = false;
      material_config = std::make_shared<MaterialConfig>();
      material = std::make_shared<StableNeohookean>(material_config);
      obj = std::make_shared<TetrahedralMesh>(meshV,
          meshT, material, material_config);
      sim = std::make_shared<T>(obj,config);
      sim->reset();

      // Perform simulation step
      config->inner_steps=1;
      sim->step();
    }

    std::shared_ptr<SimConfig> config;
    std::shared_ptr<MaterialConfig> material_config;
    std::shared_ptr<MaterialModel> material;
    std::shared_ptr<Mesh> obj;
    std::shared_ptr<T> sim;
  };
}