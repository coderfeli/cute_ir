cmake_minimum_required(VERSION 3.18)
project(CuteIR VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_RUNTIME "Build C++ runtime library" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" OFF)

# Find MLIR (optional)
find_package(MLIR CONFIG)
if(MLIR_FOUND)
    message(STATUS "Found MLIR: ${MLIR_DIR}")
    include_directories(${MLIR_INCLUDE_DIRS})
    add_definitions(${MLIR_DEFINITIONS})
    
    # TableGen targets
    set(LLVM_TARGET_DEFINITIONS include/cute/CuteDialect.td)
    tablegen(MLIR CuteDialect.h.inc -gen-dialect-decls)
    tablegen(MLIR CuteDialect.cpp.inc -gen-dialect-defs)
    add_public_tablegen_target(CuteDialectIncGen)
    
    set(LLVM_TARGET_DEFINITIONS include/cute/CuteOps.td)
    tablegen(MLIR CuteOps.h.inc -gen-op-decls)
    tablegen(MLIR CuteOps.cpp.inc -gen-op-defs)
    add_public_tablegen_target(CuteOpsIncGen)
    
    set(LLVM_TARGET_DEFINITIONS include/cute/CuteNvgpuDialect.td)
    tablegen(MLIR CuteNvgpuDialect.h.inc -gen-dialect-decls)
    tablegen(MLIR CuteNvgpuDialect.cpp.inc -gen-dialect-defs)
    add_public_tablegen_target(CuteNvgpuDialectIncGen)
    
    set(LLVM_TARGET_DEFINITIONS include/cute/CuteNvgpuOps.td)
    tablegen(MLIR CuteNvgpuOps.h.inc -gen-op-decls)
    tablegen(MLIR CuteNvgpuOps.cpp.inc -gen-op-defs)
    add_public_tablegen_target(CuteNvgpuOpsIncGen)
    
    set(LLVM_TARGET_DEFINITIONS include/cute/CutePasses.td)
    tablegen(MLIR CutePasses.h.inc -gen-pass-decls -name Cute)
    add_public_tablegen_target(CutePassesIncGen)
    
    message(STATUS "TableGen targets configured")
else()
    message(WARNING "MLIR not found. TableGen targets will not be built.")
    message(STATUS "Set MLIR_DIR to enable MLIR features.")
endif()

# Build runtime library
if(BUILD_RUNTIME)
    add_subdirectory(runtime)
endif()

# Installation
install(DIRECTORY include/cute
    DESTINATION include
    FILES_MATCHING PATTERN "*.td"
)

install(DIRECTORY docs
    DESTINATION share/cute_ir
)

# Summary
message(STATUS "")
message(STATUS "CuTe IR Configuration Summary")
message(STATUS "==============================")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build Runtime:     ${BUILD_RUNTIME}")
message(STATUS "  Build Python:      ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build Tests:       ${BUILD_TESTS}")
message(STATUS "  MLIR Found:        ${MLIR_FOUND}")
message(STATUS "")
