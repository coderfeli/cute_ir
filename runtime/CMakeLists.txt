cmake_minimum_required(VERSION 3.18)
project(CuteRuntime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# CUDA architectures (can be overridden)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===#
# CuTe Runtime Library (C++)
#===----------------------------------------------------------------------===#

add_library(cute_runtime SHARED
    src/cute_runtime.cpp
)

target_link_libraries(cute_runtime
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(cute_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

#===----------------------------------------------------------------------===#
# Python Bindings Module
#===----------------------------------------------------------------------===#

pybind11_add_module(_cute_bindings
    ../python/cute_runtime/bindings.cpp
)

target_link_libraries(_cute_bindings PRIVATE
    cute_runtime
    CUDA::cudart
    CUDA::cuda_driver
)

target_include_directories(_cute_bindings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Install targets
install(TARGETS cute_runtime _cute_bindings
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/cute_runtime.h
    DESTINATION include
)
