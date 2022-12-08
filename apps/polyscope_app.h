#pragma once

#include <igl/IO>

// Polyscope
#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/volume_mesh.h"
#include "polyscope/surface_mesh.h"
#include "args/args.hxx"

#include "mesh/mesh.h"
#include "optimizers/optimizer.h"
#include "energies/material_model.h"

#include "factories/linear_solver_factory.h"
#include "factories/variable_factory.h"
#include "factories/optimizer_factory.h"
#include "factories/integrator_factory.h"
#include "factories/material_model_factory.h"

namespace mfem {

  // Create a combobox for all types in a given factory
  template <typename Factory, typename TypeEnum>
  bool FactoryCombo(const char* id, TypeEnum& type) {
    static Factory factory;
    const std::vector<std::string>& names = factory.names();
    std::string name = factory.name_by_type(type);
    bool ret = false;;

    if (ImGui::BeginCombo(id, name.c_str())) {
      for (size_t i = 0; i < names.size(); ++i) {
        TypeEnum type_i = factory.type_by_name(names[i]);
        const bool is_selected = (type_i == type);
        if (ImGui::Selectable(names[i].c_str(), is_selected)) {
          type = type_i;
          //optimizer->reset();
          ret = true;
        }

        // Set the initial focus when opening the combo
        // (scrolling + keyboard navigation focus)
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    return ret;
  }

  // Build a checkbox for each entry in a factory
  template <typename Factory, typename TypeEnum>
  bool FactoryCheckbox(const char* id, std::set<TypeEnum>& types) {
    static Factory factory;
    const std::vector<std::string>& names = factory.names();

    bool ret = false;
    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    if (ImGui::TreeNode(id)) {
      for (size_t i = 0; i < names.size(); ++i) {
        TypeEnum type_i = factory.type_by_name(names[i]);
        bool is_selected = (types.find(type_i) != types.end());
        if (ImGui::Checkbox(names[i].c_str(), &is_selected)) {
          std::cout << "Changed: " << is_selected << std::endl;
          ret = true;
          if (is_selected) {
            types.insert(type_i);
          } else {
            types.erase(type_i);
          }
        }
      }
      ImGui::TreePop();
    }
    return ret;
  }

  template <int DIM>
  struct PolyscopeApp {
    
    // Helper to display a little (?) mark which shows a tooltip when hovered.
    static void HelpMarker(const char* desc) {
      ImGui::TextDisabled("(?)");
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
      }
    }

    virtual void simulation_step() {

      optimizer->step();
      meshV = mesh->vertices();

      if (DIM == 3) {
        srf->updateVertexPositions(meshV);
      } else {
        srf->updateVertexPositions2D(meshV);
      }
    }

    virtual void reset() {
      optimizer->reset();

      if (initMeshV.size() != 0) {
        // optimizer->update_vertices(initMeshV);
        srf->updateVertexPositions(initMeshV);
      }

      if (x0.size() != 0) {
        if constexpr (DIM == 3) {
          srf->updateVertexPositions(mesh->V_);
        } else {
          srf->updateVertexPositions2D(mesh->V_);
        }          
      } else {
        if constexpr (DIM == 3) {
          srf->updateVertexPositions(mesh->Vinit_);
        } else {
          srf->updateVertexPositions2D(mesh->Vinit_);
        }
      }
    }

    virtual void write_obj(int step) {
      char buffer [50];
      int n = sprintf(buffer, "../output/obj/tet_%04d.obj", step); 
      buffer[n] = 0;
      if (skinV.rows() > 0)
        igl::writeOBJ(std::string(buffer),skinV,skinF);
      else {
        if (meshV.cols() == 3) {
          igl::writeOBJ(std::string(buffer),meshV,meshF);
        } else {
          // If in 2D, pad the matrix
          Eigen::MatrixXd tmp(meshV.rows(), 3);
          tmp.setZero();
          tmp.col(0) = meshV.col(0);
          tmp.col(1) = meshV.col(1);
          igl::writeOBJ(std::string(buffer),tmp,meshF);
        }
      }
    }

    virtual void callback() {

      static bool export_obj = false;
      static bool export_mesh = false;
      static bool export_sim_substeps = false;
      static bool simulating = false;
      static int step = 0;
      static int export_step = 0;
      static int max_steps = 300;

      ImGui::PushItemWidth(100);


      ImGui::Checkbox("export obj",&export_obj);
      ImGui::SameLine();
      ImGui::Checkbox("export substeps",&config->save_substeps);
      ImGui::SameLine();
      ImGui::Checkbox("export mesh",&export_mesh);

      for (size_t i = 0; i < callback_funcs.size(); ++i) {
        callback_funcs[i]();
      }

      //if (ImGui::TreeNode("Material Params")) {

      //  if (FactoryCombo<MaterialModelFactory, MaterialModelType>(
      //      "Material Model", material_config->material_model)) {
      //    material = material_factory.create(material_config->material_model,
      //        material_config);
      //    mesh->material_ = material;
      //  }

      //  double lo=0.1,hi=0.5;
      //  if (ImGui::InputScalar("Young's Modulus", ImGuiDataType_Double,
      //      &material_config->ym, NULL, NULL, "%.3e")) {
      //    
      //    Enu_to_lame(material_config->ym, material_config->pr,
      //        material_config->la, material_config->mu);
      //    config->kappa = material_config->mu;

      //    if (config->optimizer == OPTIMIZER_SQP_BENDING) {
      //      double E = material_config->ym;
      //      double nu = material_config->pr;
      //      double thickness = material_config->thickness;
      //      config->kappa = E / (24 * (1.0 - nu * nu))
      //          * thickness * thickness * thickness;
      //    }
      //  }
      //  // ImGui::SameLine(); 
      //  // HelpMarker("Young's Modulus");
      //  if (ImGui::SliderScalar("Poisson's Ratio", ImGuiDataType_Double,
      //      &material_config->pr, &lo, &hi)) {
      //    
      //    Enu_to_lame(material_config->ym, material_config->pr,
      //        material_config->la, material_config->mu);

      //    if (config->optimizer == OPTIMIZER_SQP_BENDING) {
      //      double E = material_config->ym;
      //      double nu = material_config->pr;
      //      double thickness = material_config->thickness;
      //      config->kappa = E / (24 * (1.0 - nu * nu))
      //          * thickness * thickness * thickness;
      //    }
      //  }

      //  if (config->optimizer == OPTIMIZER_SQP_BENDING) {
      //    if (ImGui::InputDouble("Thickness", &material_config->thickness)) {
      //      double E = material_config->ym;
      //      double nu = material_config->pr;
      //      double thickness = material_config->thickness;
      //      config->kappa = E / (24 * (1.0 - nu * nu))
      //          * thickness * thickness * thickness;
      //    }

      //    ImGui::InputDouble("Density", &material_config->density);
      //  }
      //  ImGui::TreePop();
      //}

      //ImGui::SetNextItemOpen(true, ImGuiCond_Once);
      if (ImGui::TreeNode("Variables")) {

        if (FactoryCheckbox<MixedVariableFactory<DIM>, VariableType>(
            "Mixed Variables", config->mixed_variables)) {

          auto& vars = optimizer->state().mixed_vars_;
          vars.clear();
          for (VariableType type : config->mixed_variables) {
            vars.push_back(mixed_variable_factory.create(type, mesh, config));
            vars.back()->reset();
          }
        }

        if (FactoryCheckbox<VariableFactory<DIM>, VariableType>(
            "Nodal Variables", config->variables)) {
          auto& vars = optimizer->state().vars_;
          vars.clear();
          for (VariableType type : config->variables) {
            vars.push_back(variable_factory.create(type, mesh, config));
            vars.back()->reset();
          }
        }
        ImGui::TreePop();
      }

      //ImGui::SetNextItemOpen(true, ImGuiCond_Once);
      if (ImGui::TreeNode("Sim Params")) {

        ImGui::InputDouble("Timestep", &config->h, 0,0,"%.5f");

        // Collision parameters
        if (ImGui::TreeNode("Contact")) {
          ImGui::InputDouble("dhat", &config->dhat, 0,0,"%.5g");
          ImGui::InputDouble("mu", &config->mu, 0,0,"%.5g");
          ImGui::InputDouble("espv", &config->espv, 0,0,"%.5g");
          ImGui::Checkbox("enable_ccd", &config->enable_ccd);
        }

        // Optimizer parameters
        if (FactoryCombo<OptimizerFactory<DIM>, OptimizerType>(
            "Optimizer", config->optimizer)) {
          SimState<DIM>& state = optimizer->state();
          optimizer = optimizer_factory.create(config->optimizer, state);
          optimizer->reset();
        }

        ImGui::InputInt("Max Newton Iters", &config->outer_steps);
        ImGui::InputInt("Max LS Iters", &config->ls_iters);
        ImGui::InputDouble("Newton Tol", &config->newton_tol,0,0,"%.5g");

        // Iterative solver params
        if (config->solver_type == LinearSolverType::SOLVER_AFFINE_PCG
            || config->solver_type == LinearSolverType::SOLVER_EIGEN_CG_DIAG
            || config->solver_type == LinearSolverType::SOLVER_MINRES_ID
            || config->solver_type == LinearSolverType::SOLVER_MINRES_ADMM
            || config->solver_type == LinearSolverType::SOLVER_EIGEN_CG_IC
            || config->solver_type == LinearSolverType::SOLVER_EIGEN_GS
            || config->solver_type == LinearSolverType::SOLVER_SUBSPACE
            || config->solver_type == LinearSolverType::SOLVER_ADMM
            || config->solver_type == LinearSolverType::SOLVER_AMGCL) {
          ImGui::InputInt("Max CG Iters", &config->max_iterative_solver_iters);
          ImGui::InputDouble("CG Tol", &config->itr_tol,0,0,"%.5g");
        }

        if (ImGui::InputFloat3("Body Force", config->ext)) {
        }

        ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
        if (config->optimizer == OPTIMIZER_ALM
            || config->optimizer == OPTIMIZER_ADMM) {
          ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
          ImGui::SameLine(); 
          ImGui::InputDouble("max kappa", &config->max_kappa, 0,0,"%.5g");
          ImGui::InputDouble("constraint tol",&config->constraint_tol, 0,0,"%.5g");
          ImGui::InputDouble("lamda update tol",&config->update_zone_tol,0,0,"%.5g");
        }
        if (config->optimizer == OPTIMIZER_SQP_BENDING) {
          ImGui::InputDouble("kappa", &config->kappa,0,0,"%.5g");
        }

        if (FactoryCombo<LinearSolverFactory<DIM>, LinearSolverType>(
            "Linear Solver", config->solver_type)) {
          optimizer->reset();
        }

        if (FactoryCombo<IntegratorFactory, TimeIntegratorType>(
            "Integrator", config->ti_type)) {
          optimizer->reset();
        }

        ImGui::TreePop();
      }

      ImGui::Checkbox("Output optimizer data",&config->show_data);
      ImGui::SameLine();
      ImGui::Checkbox("Output timing info",&config->show_timing);

      ImGui::Checkbox("simulate",&simulating);
      ImGui::SameLine();
      if(ImGui::Button("step") || simulating) {
        std::cout << "Timestep: " << step << std::endl;
        simulation_step();
        ++step;

        if (export_obj) {
          write_obj(export_step++);
        }

        if (export_mesh) {
          char buffer [50];
          int n = sprintf(buffer, "../output/mesh/tet_%04d.mesh", step); 
          buffer[n] = 0;
          igl::writeMESH(std::string(buffer),meshV, meshT, meshF);
          std::ofstream outfile;
          outfile.open(std::string(buffer), std::ios_base::app); 
          outfile << "End"; 
        }

        if (config->save_substeps) {
          char buffer[50];
          int n;

          const SimState<DIM>& state = optimizer->state();
          Eigen::VectorXd x0 = state.x_->integrator()->x_prev();
          Eigen::VectorXd v0 = state.x_->integrator()->v_prev();
          Eigen::MatrixXd X = Eigen::Map<Eigen::MatrixXd>(x0.data(), DIM,
              state.mesh_->V_.rows());
          X.transposeInPlace();
          Eigen::MatrixXd V = Eigen::Map<Eigen::MatrixXd>(v0.data(), DIM,
              state.mesh_->V_.rows());
          V.transposeInPlace();

          n = sprintf(buffer, "../output/sim_x0_%04d.dmat", step); 
          buffer[n] = 0;
          igl::writeDMAT(std::string(buffer), X);

          n = sprintf(buffer, "../output/sim_v_%04d.dmat", step); 
          buffer[n] = 0;
          igl::writeDMAT(std::string(buffer), V);
        }

      }
      ImGui::SameLine();
      if(ImGui::Button("reset")) {
        reset();
        export_step = 0;
        step = 0;
      }
      if (step >= max_steps) {
        simulating = false;
      }
      ImGui::InputInt("Max Steps", &max_steps);
      ImGui::PopItemWidth();
    }

    polyscope::SurfaceMesh* srf = nullptr;
    polyscope::SurfaceMesh* srf_skin = nullptr;

    std::vector<std::function<void()>> callback_funcs;

    // The mesh, Eigen representation
    Eigen::MatrixXd meshV, meshV0, skinV, initMeshV;
    Eigen::MatrixXi meshF, skinF;
    Eigen::MatrixXi meshT; // tetrahedra
    Eigen::SparseMatrixd lbs; // linear blend skinning matrix
    Eigen::VectorXd x0, v;

    MaterialModelFactory material_factory;
    VariableFactory<DIM> variable_factory;
    MixedVariableFactory<DIM> mixed_variable_factory;
    OptimizerFactory<DIM> optimizer_factory;
    LinearSolverFactory<DIM> solver_factory;

    std::shared_ptr<MaterialConfig> material_config;
    std::shared_ptr<MaterialModel> material;
    std::shared_ptr<Optimizer<DIM>> optimizer;
    std::shared_ptr<SimConfig> config;
    std::shared_ptr<Mesh> mesh;

    std::vector<std::string> bc_list;
  };
}
