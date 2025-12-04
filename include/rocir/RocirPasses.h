#ifndef ROCIR_PASSES_H
#define ROCIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rocir {

// Lowering passes
std::unique_ptr<Pass> createRocirToStandardPass();
std::unique_ptr<Pass> createRocirCoordLoweringPass();
std::unique_ptr<Pass> createRocirToRocmPass();

// Registration helpers (for Python bindings)
inline void registerRocirToStandardPassManual() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createRocirToStandardPass();
  });
}

inline void registerRocirCoordLoweringPassManual() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createRocirCoordLoweringPass();
  });
}

} // namespace rocir
} // namespace mlir

#endif // ROCIR_PASSES_H
