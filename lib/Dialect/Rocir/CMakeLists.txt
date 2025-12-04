add_mlir_dialect_library(RocirDialect
  RocirDialect.cpp
  RocirOps.cpp
  
  DEPENDS
  RocirDialectIncGen
  RocirOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
