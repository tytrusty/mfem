## Optimizers

Includes implementation of several mixed & standard optimizers

### Adding an optimizer
- Create .h/.cpp file, extending the optimizer or mixed_optimizer pure virtual classes.
- Add enum entry into "OptimizerType" in config.h
- Implement name() static function, and modify constructor in optimizer_factory.cpp, adding an entry for the new optimizer