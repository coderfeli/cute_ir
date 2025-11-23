# AMD ROCm Support for GFX942

This document describes the AMD ROCm/HIP support added to CuTe IR for AMD GFX942 (MI300 series) GPUs.

## Overview

CuTe IR now supports AMD GPUs through the `cute_rocm` dialect, targeting the GFX942 architecture (MI300A/MI300X). This enables layout-aware tensor programming on AMD hardware with MFMA (Matrix Fused Multiply-Add) instructions.

## Architecture

### Dialect Structure

- **CuteRocmDialect** (`include/cute/CuteRocmDialect.td`): Core dialect definition with AMD-specific types
- **CuteRocmOps** (`include/cute/CuteRocmOps.td`): AMD GPU operations including MFMA, LDS, and synchronization
- **CuteToRocm** (`lib/Transforms/CuteToRocm.cpp`): Lowering pass from cute dialect to cute_rocm

### Key Features

1. **MFMA Support**: Matrix core operations for GFX942
2. **LDS Management**: Local Data Share (shared memory) with bank conflict avoidance
3. **Wavefront Operations**: 64-lane wavefront support
4. **Memory Operations**: Optimized global↔LDS↔register transfers

## GFX942 Specifications

- **Architecture**: CDNA 3 (MI300 series)
- **Wavefront Size**: 64 lanes
- **LDS Size**: 64KB per compute unit
- **Memory Coalescing**: 128-byte aligned for optimal performance
- **MFMA Instructions**: F16, BF16, F64, I8 support

## Supported MFMA Instructions

| Instruction | Shape (M×N×K) | Input Type | Output Type |
|-------------|---------------|------------|-------------|
| `mfma_f32_32x32x8_f16` | 32×32×8 | FP16 | FP32 |
| `mfma_f32_16x16x16_f16` | 16×16×16 | FP16 | FP32 |
| `mfma_f32_32x32x16_bf16` | 32×32×16 | BF16 | FP32 |
| `mfma_f64_16x16x4_f64` | 16×16×4 | FP64 | FP64 |
| `mfma_i32_32x32x16_i8` | 32×32×16 | INT8 | INT32 |

## Building with ROCm Support

### Prerequisites

```bash
# Install ROCm (version 5.7 or later recommended)
# For Ubuntu/Debian:
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*_all.deb
sudo dpkg -i amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocminfo
hipcc --version
```

### CMake Configuration

```bash
# Configure with ROCm support enabled
cmake -B build \
  -DENABLE_ROCM=ON \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DHIP_ARCHITECTURES=gfx942

# Build
cmake --build build -j$(nproc)
```

### Build Options

- `ENABLE_ROCM=ON`: Enable AMD ROCm support (default: ON)
- `USE_ROCM=ON`: Enable HIP runtime (default: ON)
- `HIP_ARCHITECTURES`: Target architecture (default: gfx942)

## Layout Programming for GFX942

### MFMA-Compatible Layouts

```mlir
// 32×32×8 FP16 MFMA layout
func.func @mfma_layout() -> !cute.layout<3> {
  %shape = cute.make_shape 32, 32, 8 : !cute.shape<3>
  %stride = cute.make_stride 1, 32, 1024 : !cute.stride<3>
  %layout = cute.make_layout %shape, %stride : !cute.layout<3>
  return %layout : !cute.layout<3>
}
```

### LDS-Optimized Layouts

```mlir
// Padded layout to avoid bank conflicts
// GFX942 has 32 LDS banks, 4-byte width
func.func @lds_layout() -> !cute.layout<2> {
  %shape = cute.make_shape 64, 64 : !cute.shape<2>
  // Padding: 68 = 64 + 4 (avoid bank conflicts)
  %stride = cute.make_stride 1, 68 : !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : !cute.layout<2>
  return %layout : !cute.layout<2>
}
```

### Wavefront-Partitioned Layouts

```mlir
// Layout for 64-lane wavefront
func.func @wavefront_layout() -> !cute.layout<2> {
  %shape = cute.make_shape 64, 4 : !cute.shape<2>
  %stride = cute.make_stride 1, 64 : !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : !cute.layout<2>
  return %layout : !cute.layout<2>
}
```

## Using ROCm Operations

### MFMA Operation

```mlir
%d = cute_rocm.mfma %a, %b, %c {
  shape = [32, 32, 8],
  arch = "gfx942"
} : (!cute.tensor<f16, ...>, !cute.tensor<f16, ...>, 
     !cute.tensor<f32, ...>) -> !cute.tensor<f32, ...>
```

### LDS Allocation

```mlir
%lds = cute_rocm.lds_alloc {
  element_type = f16,
  layout = !cute.layout<...>,
  size = 32768
} : !cute_rocm.lds_buffer<f16, ..., 32768>
```

### Synchronization

```mlir
// Workgroup barrier (s_barrier)
cute_rocm.barrier

// Wavefront barrier (s_waitcnt)
cute_rocm.wavefront_barrier
```

## Testing

Run the AMD-specific layout tests:

```bash
# Using mlir-opt
mlir-opt tests/test_layout_amd.mlir -pass-pipeline='builtin.module(cute-to-rocm)'

# Check layout patterns
mlir-opt tests/test_layout_amd.mlir --cute-canonicalize
```

## Runtime Usage

### C++ Runtime

```cpp
#include "cute_runtime.h"

// Initialize HIP device
cute::runtime::hip::HipDevice::instance().initialize();

// Allocate device memory
void* d_ptr = cute::runtime::hip::allocateDeviceMemory(size);

// Validate LDS allocation
cute::runtime::hip::LdsAllocator::validateAllocation(32768);

// Execute kernel
cute::runtime::hip::HipKernelExecutor executor;
executor.loadCodeObject("kernel.hsaco");
executor.launch(grid, block, shared_mem, args...);
```

### Python Bindings

```python
import cute_runtime

# Device information
device = cute_runtime.get_device_info()
print(f"Device: {device['name']}")
print(f"Architecture: {device['arch']}")

# Memory operations
d_buffer = cute_runtime.allocate_device(size)
cute_runtime.copy_to_device(d_buffer, h_buffer)
```

## Performance Considerations

### LDS Bank Conflicts

- GFX942 has 32 LDS banks with 4-byte width
- Add padding to avoid conflicts: `stride = base_size + padding`
- Typical padding: 4-8 bytes per row

### Memory Coalescing

- Optimal coalescing: 128-byte aligned (32 × FP32)
- Use contiguous layouts for global memory access
- Wavefront (64 threads) should access consecutive memory

### MFMA Optimization

- Prefer 32×32×8 for FP16 operations
- Use 16×16×16 for smaller tiles
- BF16 support via 32×32×16 instruction
- Consider register pressure when tiling

## Lowering Pipeline

```
cute dialect
    ↓ (cute-to-rocm pass)
cute_rocm dialect
    ↓ (future work)
rocdl dialect
    ↓
LLVM IR (AMDGPU)
    ↓
GCN assembly
```

## Examples

See `tests/test_layout_amd.mlir` for comprehensive examples including:
- MFMA layouts (32×32×8, 16×16×16, etc.)
- LDS bank conflict avoidance
- Wavefront partitioning
- GEMM layout composition
- BF16 and FP64 layouts

## Troubleshooting

### HIP Not Found

```bash
# Set ROCm path
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### LDS Allocation Errors

```
Error: LDS allocation exceeds 64KB limit
```
Solution: Reduce LDS usage or use hierarchical tiling

### MFMA Instruction Not Found

Verify GFX942 support:
```bash
rocminfo | grep gfx942
```

## References

- [AMD CDNA 3 Architecture](https://www.amd.com/en/products/accelerators/instinct/mi300)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [MFMA Instructions Reference](https://rocm.docs.amd.com/projects/MFMA-doc/en/latest/)

## License

Same as the main CuTe IR project.
