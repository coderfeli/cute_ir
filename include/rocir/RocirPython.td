//===- RocirPython.td - Python Bindings Definition ---------*- tablegen -*-===//
//
// Combined definition file for generating Python bindings
// Includes both core Rocir ops and RocirRocm ops
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_PYTHON
#define ROCIR_PYTHON

include "rocir/RocirOps.td"
include "rocir/RocirRocmOps.td"

#endif // ROCIR_PYTHON
