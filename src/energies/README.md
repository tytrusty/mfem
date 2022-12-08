## Material models

Includes implementation of several different materials.

### Adding a material model
- Create .h/.cpp file, extending the material_model pure virtual class, implementing the necessary energy, gradient, hessian methods
- Add enum entry into "MaterialModelType" in config.h
- Implement name() static function, and modify constructor in material_model_factory.cpp, adding an entry for the new material model