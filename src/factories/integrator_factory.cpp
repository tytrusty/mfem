#include "integrator_factory.h"
#include "time_integrators/BDF.h"

using namespace mfem;
using namespace Eigen;

IntegratorFactory::IntegratorFactory() {

    // BDF1
    register_type(TimeIntegratorType::TI_BDF1, "BDF1",
        [](Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        ->std::unique_ptr<ImplicitIntegrator>
        {return std::make_unique<BDF<1>>(x0,v0,h);});

    // BDF2
    register_type(TimeIntegratorType::TI_BDF2, "BDF2",
        [](Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        ->std::unique_ptr<ImplicitIntegrator>
        {return std::make_unique<BDF<2>>(x0,v0,h);});

    // BDF3
    register_type(TimeIntegratorType::TI_BDF3, "BDF3",
        [](Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        ->std::unique_ptr<ImplicitIntegrator>
        {return std::make_unique<BDF<3>>(x0,v0,h);});

    // BDF4
    register_type(TimeIntegratorType::TI_BDF4, "BDF4",
        [](Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        ->std::unique_ptr<ImplicitIntegrator>
        {return std::make_unique<BDF<4>>(x0,v0,h);});

    // BDF5
    register_type(TimeIntegratorType::TI_BDF5, "BDF5",
        [](Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        ->std::unique_ptr<ImplicitIntegrator>
        {return std::make_unique<BDF<5>>(x0,v0,h);});

    // BDF6
    register_type(TimeIntegratorType::TI_BDF6, "BDF6",
        [](Eigen::VectorXd x0, Eigen::VectorXd v0, double h)
        ->std::unique_ptr<ImplicitIntegrator>
        {return std::make_unique<BDF<6>>(x0,v0,h);});
}
