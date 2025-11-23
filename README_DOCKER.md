# Building and Testing CuTe IR in Docker (CUDA-only)

This project is configured for **CUDA-only** mode by default. ROCm/HIP support is available but disabled.

## Prerequisites

- Docker container `felixatt` must be running
- Container must have NVIDIA GPU access
- Project directory mounted at `/mnt/raid0/felix/cute_ir_tablegen`

## Quick Start

### 1. Build the Project

```bash
./build_in_docker.sh
```

This will:
- Clean any previous build
- Configure CMake with CUDA support only (ROCm disabled)
- Build the project inside the `felixatt` container

### 2. Test the Build

```bash
./test_in_docker.sh
```

This will:
- Show CUDA device information
- List build artifacts
- Run MLIR tests if available
- Show project structure

### 3. Interactive Shell

```bash
./docker_shell.sh
```

Opens an interactive bash shell inside the container at the project directory.

## Build Configuration

### CUDA-Only Mode (Default)

```bash
cmake .. -DBUILD_RUNTIME=ON -DENABLE_ROCM=OFF
```

### Enable ROCm (Optional)

```bash
cmake .. -DBUILD_RUNTIME=ON -DENABLE_ROCM=ON -DUSE_ROCM=ON
```

## Manual Build Inside Container

```bash
# Enter container
docker exec -it felixatt bash

# Navigate to project
cd /mnt/raid0/felix/cute_ir_tablegen

# Configure and build
mkdir -p build && cd build
cmake .. -DBUILD_RUNTIME=ON -DENABLE_ROCM=OFF
make -j$(nproc)
```

## Verify CUDA Support

Inside the container:

```bash
nvidia-smi              # Check GPU
nvcc --version          # Check CUDA compiler
```

## Project Structure

```
.
├── build_in_docker.sh    # Build script
├── test_in_docker.sh     # Test script  
├── docker_shell.sh       # Interactive shell
├── CMakeLists.txt        # ENABLE_ROCM=OFF by default
├── include/cute/         # Dialect definitions
│   ├── CuteDialect.td
│   ├── CuteOps.td
│   ├── CuteNvgpuDialect.td  # NVIDIA GPU support
│   ├── CuteNvgpuOps.td
│   ├── CuteRocmDialect.td   # AMD GPU (optional)
│   └── CuteRocmOps.td
├── lib/Transforms/       # Transformation passes
├── runtime/              # CUDA runtime (HIP optional)
└── tests/                # Test files
```

## Notes

- **Default**: CUDA-only, no HIP/ROCm dependencies
- **ROCm files** are included but not built by default
- To enable ROCm: set `-DENABLE_ROCM=ON -DUSE_ROCM=ON`
- The `felixatt` container should have CUDA toolkit installed
