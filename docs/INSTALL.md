# CuTe Runtime Installation Guide

## Quick Start

```bash
# Prerequisites
export CUDA_HOME=/usr/local/cuda
pip install numpy pybind11

# Install
cd cute_ir_tablegen/
pip install .

# Test
python -c "import cute_runtime; print(cute_runtime.get_device_info())"
```

## Prerequisites

### Required

1. **CUDA Toolkit 11.0+**
   ```bash
   # Verify installation
   nvcc --version
   which nvcc
   
   # Set environment
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

2. **CMake 3.18+**
   ```bash
   cmake --version
   ```

3. **Python 3.8+**
   ```bash
   python3 --version
   pip install numpy pybind11
   ```

4. **C++ Compiler with C++17 support**
   ```bash
   # GCC 7+ or Clang 5+
   g++ --version
   ```

### Optional (for full MLIR compilation)

5. **MLIR/LLVM Installation**
   ```bash
   # Build LLVM/MLIR from source
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   mkdir build && cd build
   
   cmake ../llvm \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_PROJECTS="mlir" \
     -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
   
   make -j$(nproc)
   make install
   
   # Set environment
   export MLIR_INSTALL_DIR=$HOME/llvm-install
   export PATH=$MLIR_INSTALL_DIR/bin:$PATH
   ```

## Installation Methods

### Method 1: Python Package Install (Recommended)

```bash
cd cute_ir_tablegen/

# Option A: Development install (editable)
pip install -e .

# Option B: Production install
pip install .

# Option C: Specify CUDA architecture
CUDA_ARCH="80;90" pip install .

# Option D: With MLIR support
MLIR_INSTALL_DIR=/path/to/llvm-install pip install .
```

### Method 2: Build C++ Library Manually

```bash
cd cute_ir_tablegen/runtime/
mkdir build && cd build

# Configure
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES="80;90" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/cute-install

# Build
make -j8

# Install
make install

# Libraries will be in:
# - $HOME/cute-install/lib/libcute_runtime.so
# - $HOME/cute-install/lib/_cute_bindings.so
# - $HOME/cute-install/include/cute_runtime.h
```

### Method 3: Docker Container

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy pybind11

# Copy source
COPY cute_ir_tablegen /workspace/cute_ir_tablegen
WORKDIR /workspace/cute_ir_tablegen

# Install
RUN pip3 install .

# Test
RUN python3 -c "import cute_runtime; print(cute_runtime.get_device_info())"
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_HOME` | CUDA Toolkit location | `/usr/local/cuda` |
| `MLIR_INSTALL_DIR` | MLIR installation path | `/home/user/llvm-install` |
| `CUDA_ARCH` | Target architectures | `"80;90;100"` |
| `CMAKE_PREFIX_PATH` | CMake search paths | `/opt/cuda:/opt/mlir` |

## Verification

### Test 1: Import Check

```bash
python3 << EOF
import cute_runtime as cute
print("✓ CuTe Runtime imported successfully")
print(f"  Version: {cute.__version__}")
EOF
```

### Test 2: Device Query

```bash
python3 << EOF
import cute_runtime as cute

info = cute.get_device_info()
print(f"✓ Device: {info['name']}")
print(f"  Compute Capability: {info['compute_capability']}")
print(f"  SMs: {info['multiprocessor_count']}")
print(f"  Total Memory: {info['total_memory'] / 1e9:.2f} GB")
EOF
```

### Test 3: GEMM Example

```bash
cd python/examples/
python3 test_gemm.py
```

## Troubleshooting

### Issue 1: CUDA Not Found

```
Error: CUDA not found at /usr/local/cuda
```

**Solution:**
```bash
# Find CUDA installation
which nvcc
ls /usr/local/cuda*

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.0  # Adjust version
pip install . --force-reinstall
```

### Issue 2: CMake Version Too Old

```
CMake Error: CMake 3.18 or higher is required
```

**Solution:**
```bash
# Ubuntu/Debian
pip install cmake --upgrade

# Or download latest CMake
wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh
bash cmake-3.27.0-linux-x86_64.sh --prefix=$HOME/cmake --skip-license
export PATH=$HOME/cmake/bin:$PATH
```

### Issue 3: pybind11 Not Found

```
Could not find pybind11
```

**Solution:**
```bash
pip install pybind11[global]
# or
conda install -c conda-forge pybind11
```

### Issue 4: CUDA Architecture Mismatch

```
Error: Unsupported GPU architecture
```

**Solution:**
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Install with matching architecture
# For RTX 3090 (Ampere, SM80):
CUDA_ARCH="80" pip install .

# For H100 (Hopper, SM90):
CUDA_ARCH="90" pip install .

# For multiple GPUs:
CUDA_ARCH="80;90" pip install .
```

### Issue 5: Import Error After Installation

```python
ImportError: _cute_bindings.so: cannot open shared object file
```

**Solution:**
```bash
# Check library path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Or add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue 6: MLIR Tools Not Found

```
Error: mlir-opt not found
```

**Solution:**
```bash
# Option 1: Build MLIR (see Prerequisites section)

# Option 2: Use pre-built MLIR
# Download from: https://github.com/llvm/llvm-project/releases
# Extract and set:
export MLIR_INSTALL_DIR=/path/to/llvm-install
export PATH=$MLIR_INSTALL_DIR/bin:$PATH

# Option 3: Skip MLIR (runtime-only mode)
# The runtime will work for loading pre-compiled kernels
# but MLIR compilation features will be unavailable
```

## Platform-Specific Notes

### Ubuntu/Debian

```bash
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    cmake \
    python3-dev python3-pip \
    libpython3-dev

pip3 install numpy pybind11
cd cute_ir_tablegen && pip3 install .
```

### CentOS/RHEL

```bash
sudo yum install -y \
    cuda-toolkit-12-0 \
    cmake3 \
    python3-devel python3-pip

pip3 install numpy pybind11
cd cute_ir_tablegen && pip3 install .
```

### macOS (CPU-only, for development)

```bash
# Note: CUDA not available on macOS
# Can build for CPU testing only

brew install cmake python@3.11
pip3 install numpy pybind11

# CPU-only build (for testing)
cd cute_ir_tablegen/runtime
mkdir build && cd build
cmake .. -DBUILD_CUDA=OFF
make -j8
```

### Windows (WSL2)

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2:
# 1. Install CUDA for WSL2:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-0

# 2. Install CuTe Runtime:
cd cute_ir_tablegen
pip install .
```

## Uninstallation

```bash
# Python package
pip uninstall cute-runtime

# Manual C++ library
rm -rf $HOME/cute-install
```

## Next Steps

After installation:

1. **Run Examples**
   ```bash
   cd python/examples/
   python test_gemm.py
   ```

2. **Read Documentation**
   - [API_INTEGRATION.md](API_INTEGRATION.md) - API usage guide
   - [SUMMARY.md](SUMMARY.md) - Project overview
   - [PassPipeline.md](PassPipeline.md) - Compilation pipeline

3. **Explore Source**
   - `runtime/include/cute_runtime.h` - C++ API
   - `python/cute_runtime/__init__.py` - Python API
   - `examples/cute_gemm_example.mlir` - MLIR example

4. **Join Development**
   - Implement remaining passes (see `lib/Transforms/`)
   - Add new kernel examples
   - Contribute to documentation

---

For issues, please check:
- CUDA installation: `nvcc --version`
- Python environment: `python --version`, `pip list | grep numpy`
- CMake: `cmake --version`
- GPU: `nvidia-smi`
