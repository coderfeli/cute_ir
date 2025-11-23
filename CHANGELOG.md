# Changelog

All notable changes to the CuTe IR project will be documented in this file.

## [0.1.0] - 2025-11-23

### Added

**MLIR Infrastructure**
- Complete TableGen definitions for `cute_ir` dialect (8 types, 85+ operations)
- Complete TableGen definitions for `cute_nvgpu_ir` dialect (5 types, 30+ operations)
- 18 pass definitions in `CutePasses.td`
- 2 pass implementations: `cute-to-standard`, `cute-nvgpu-to-nvgpu`

**Runtime Library**
- C++ runtime with `KernelExecutor`, `GemmExecutor`, `CuteCompiler`
- TMA descriptor management for Hopper (SM90+)
- Device memory management with RAII
- MLIR compilation pipeline (MLIR → PTX → CUBIN)

**Python Bindings**
- High-level Python API with NumPy integration
- pybind11 bindings for C++ runtime
- `cute.Gemm()` - GEMM executor interface
- `cute.Kernel()` - Generic kernel launcher
- `cute.Compiler()` - MLIR compiler interface

**Build System**
- CMake configuration with MLIR integration
- Python setup.py with custom CMake build
- Multi-architecture CUDA support (SM80, SM90, SM100)

**Documentation**
- README with quick start guide
- Complete installation guide (INSTALL.md)
- API integration documentation
- Pass pipeline documentation
- Project summary with architecture diagrams
- Contributing guide

**Examples**
- Hopper GEMM example (216 lines MLIR)
- Python test script
- MLIR test cases

### Project Statistics
- 30+ files
- ~8,000 lines of code
- Support for 3 GPU architectures
- 115+ MLIR operations defined

## [Unreleased]

### Planned

**Pass Implementations** (Priority 1)
- `cute-canonicalize` - Canonicalization patterns
- `cute-layout-fusion` - Layout fusion optimization
- `cute-vectorization` - Vectorization pass
- `cute-memory-coalescing` - Memory coalescing
- `cute-swizzle-optimization` - Swizzle optimization
- `cute-partition-optimization` - Partition optimization
- 10+ additional passes

**Features** (Priority 2)
- INT8/BF16 data type support
- Multi-GPU execution support
- Kernel auto-tuning framework
- Profiling and benchmarking utilities
- Kernel compilation cache

**Testing** (Priority 3)
- Comprehensive MLIR FileCheck tests
- Python unit tests
- Integration tests
- Benchmark suite

**Documentation** (Priority 4)
- API reference documentation
- Tutorial series
- Architecture deep-dive guides
- Performance optimization guide

---

Format: [version] - YYYY-MM-DD
Types: Added, Changed, Deprecated, Removed, Fixed, Security
