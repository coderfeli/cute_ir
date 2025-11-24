#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "cute/CuteDialect.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_cuteDialect, m) {
  m.doc() = "CuTe dialect Python bindings";
  
  auto cute_m = m.def_submodule("cute");
  
  cute_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__cute__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(),
      py::arg("load") = true);
}
