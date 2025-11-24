# Generate Python bindings for CuTe dialect

add_custom_target(CutePythonModules)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/cute/CuteOps.td)

# Add include path for TableGen
set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE})
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include)
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include/cute)

# Generate Python bindings
mlir_tablegen(_cute_ops_gen.py -gen-python-op-bindings -bind-dialect=cute)
add_public_tablegen_target(CutePythonOpsIncGen)
add_dependencies(CutePythonModules CutePythonOpsIncGen)

# Generate Python enum bindings if needed
mlir_tablegen(_cute_enum_gen.py -gen-python-enum-bindings)
add_public_tablegen_target(CutePythonEnumIncGen)
add_dependencies(CutePythonModules CutePythonEnumIncGen)
