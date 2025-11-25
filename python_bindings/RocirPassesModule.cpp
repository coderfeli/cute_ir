//===- RocirPassesModule.cpp - Rocir Passes Python Module -----------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "rocir/RocirPasses.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
// Wrapper to convert std::unique_ptr<Pass> to MlirPass for Python
MlirPass createRocirCoordLoweringPassWrapper() {
  return wrap(mlir::rocir::createRocirCoordLoweringPass().release());
}

MlirPass createRocirToStandardPassWrapper() {
  return wrap(mlir::rocir::createRocirToStandardPass().release());
}
} // namespace

PYBIND11_MODULE(_rocirPassesExt, m) {
  m.doc() = "Rocir transformation passes extension module";

  // Expose pass creation functions directly to Python
  m.def("create_rocir_coord_lowering_pass", 
        &createRocirCoordLoweringPassWrapper,
        "Create an instance of the Rocir coordinate lowering pass");
  
  m.def("create_rocir_to_standard_pass",
        &createRocirToStandardPassWrapper,
        "Create an instance of the Rocir to standard lowering pass");
  
  // Also register for text-based pipeline parsing (fallback)
  ::mlir::rocir::registerRocirToStandardPassManual();
  ::mlir::rocir::registerRocirCoordLoweringPassManual();
}
