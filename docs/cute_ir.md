# CuTe IR Technical Summary

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Python CuTe DSL                         │
│  make_tensor(), make_layout(), make_tiled_mma()         │
└─────────────────┬───────────────────────────────────────┘
                  │ Python API calls
                  ▼
┌─────────────────────────────────────────────────────────┐
│              cute_ir (Layout Algebra)                    │
│  • Types: Shape, Stride, Layout, Tile, Coord           │
│  • Ops: composition, product, partition, slice         │
│  • Hardware-agnostic abstraction                        │
└─────────────────┬───────────────────────────────────────┘
                  │ Accepts LayoutType as partition descriptors
                  ▼
┌─────────────────────────────────────────────────────────┐
│          cute_nvgpu_ir (GPU Hardware-Aware)             │
│  • Atom Types: MmaAtom, CopyAtom, TmaLoad/Store        │
│  • Tiled Types: TiledMma, TiledCopy                     │
│  • Ops: warpgroup_mma, tma_load_execute, ldmatrix      │
│  • Wraps MLIR nvgpu with CuTe Layout interfaces         │
└─────────────────┬───────────────────────────────────────┘
                  │ Lowering Pass (cute → gpu/nvgpu)
                  ▼
┌─────────────────────────────────────────────────────────┐
│           MLIR Standard Dialects                         │
│  gpu.launch → nvgpu.mma_sync → nvvm.mma.sync            │
└─────────────────┬───────────────────────────────────────┘
                  │ NVVM → LLVM → PTX
                  ▼
                 SASS
```

## Dialect Relationship

### cute_ir ⊆ Hardware-Agnostic
- **Purpose**: Layout algebra for multi-dimensional data mapping
- **Key Concept**: `crd2idx(coord, layout) → linear_offset`
- **Core Types**:
  - `LayoutType = (Shape, Stride)` - Maps coordinates to memory offsets
  - `TileType = ((Layout, Layout, ...), ...)` - Hierarchical tiling
  - `TensorType = (ptr, Layout)` - Typed memory view

### cute_nvgpu_ir ⊇ GPU Hardware-Specific
- **Purpose**: Encode GPU instruction metadata + CuTe Layout partitioning
- **Key Innovation**: Atom types accept `cute_ir::LayoutType` as parameters
- **Example**:
  ```mlir
  %layout_tv = cute.make_layout ...  // From cute_ir
  %tiler = cute.make_tile ...        // From cute_ir
  %atom = cute_nvgpu.make_mma_atom !f16, !f16, !f32, [64,64,16], "SM90"
  %tiled_mma = cute_nvgpu.make_tiled_mma %atom, %layout_tv, %tiler
  ```

### cute_nvgpu_ir → nvgpu Lowering
- `cute_nvgpu.warpgroup_mma` → `nvgpu.warpgroup.mma.sync`
- `cute_nvgpu.tma_load_execute` → `nvgpu.tma.async.load`
- `cute_nvgpu.ldmatrix` → `nvgpu.ldmatrix`

**Critical Difference**: cute_nvgpu ops carry CuTe Layout context, while nvgpu ops use raw indices/strides.

## Type System

| Dialect | Type | Parameters | Example |
|---------|------|------------|---------|
| cute_ir | IntType | value | `!cute.int<64>` |
| cute_ir | LayoutType | shape, stride | `!cute.layout<(64,128), (1,64)>` |
| cute_ir | TileType | layouts | `!cute.tile<((8,8), (16,16))>` |
| cute_nvgpu | MmaAtomType | elem_types, shape, arch | `!cute_nvgpu.mma_atom<!f16,!f16,!f32,[64,64,16],SM90>` |
| cute_nvgpu | TiledMmaType | atom, layout_v, tiler | `!cute_nvgpu.tiled_mma<...>` |
| cute_nvgpu | TmaLoadType | elem_type, tile, swizzle | `!cute_nvgpu.tma_load<!f16,[128,128],"128B">` |

## Operation Inventory

### cute_ir Operations (85+)

| Category | Operations | Count |
|----------|-----------|-------|
| Construction | make_shape, make_stride, make_layout, make_coord, make_tile | 5 |
| Query | get_shape, get_stride, rank, size, depth, shape_at, stride_at | 7 |
| Mapping | crd2idx, idx2crd | 2 |
| Transform | composition, complement, left/right_inverse, coalesce, filter | 6 |
| Product | logical_product, blocked_product, raked_product, tiled_product | 4 |
| Partition | local_partition, local_tile, slice, zipped_divide, dice | 5 |
| Tensor | make_tensor, load, store, copy, clear, fill, axpby, gemv, gemm | 10 |
| MMA | make_tiled_mma, mma_make_fragment_[a/b/c], tiled_mma | 5 |
| Copy | make_tiled_copy, partition_s/d, copy_atom_call, tiled_copy | 5 |

### cute_nvgpu_ir Operations (30+)

| Category | Operations | Architectures |
|----------|-----------|---------------|
| Atom Construction | make_mma_atom, make_copy_atom | All |
| TMA Descriptors | make_tma_load, make_tma_store, make_exec_tma_* | SM90+ |
| Runtime Values | atom_get_value, atom_set_value | SM90+ |
| Warp MMA | warp_mma_f16bf16, warp_mma_tf32, warp_mma_sparse | SM80+ |
| Warpgroup MMA | warpgroup_mma, warpgroup_mma_sparse | SM90+ |
| TCGEN05 MMA | tcgen05_mma, tcgen05_block_scaled_mma | SM100+ |
| Copy | copy_universal, ldmatrix, cpasync_g2s | SM75+ |
| TMA Execution | tma_load_execute, tma_store_execute | SM90+ |
| Barriers | mbarrier_init, mbarrier_arrive, mbarrier_wait | SM90+ |

## GPU Architecture Support

| GPU | Compute Capability | Key Features | Operations |
|-----|-------------------|--------------|------------|
| Ampere | SM80 | Warp MMA, cp.async | warp_mma_*, cpasync_g2s |
| Hopper | SM90 | Warpgroup MMA, TMA, mbarrier | warpgroup_mma, tma_*, mbarrier_* |
| Blackwell | SM100 | TCGEN05, FP4/FP6, Block Scaling | tcgen05_mma, tcgen05_block_scaled_mma |

## Example: Hopper GEMM Lowering

### High-Level Python
```python
tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16BF16F32,
    Layout((4,1), (1,0)),  # 4 warps in M dimension
    Tile(64, 64, 16)
)
```

### cute_ir (Hardware-Agnostic)
```mlir
%shape_tv = cute.make_shape %c4, %c1 : !cute.shape<(4,1)>
%stride_tv = cute.make_stride %c1, %c0 : !cute.stride<(1,0)>
%layout_tv = cute.make_layout %shape_tv, %stride_tv : !cute.layout<(4,1),(1,0)>
%tiler = cute.make_tile %c64, %c64, %c16 : !cute.tile<(64,64,16)>
```

### cute_nvgpu_ir (GPU-Aware)
```mlir
%atom = cute_nvgpu.make_mma_atom !f16, !f16, !f32, [64,64,16], "SM90"
  : !cute_nvgpu.mma_atom<!f16,!f16,!f32,[64,64,16],SM90>
%tiled_mma = cute_nvgpu.make_tiled_mma %atom, %layout_tv, %tiler
  : !cute_nvgpu.tiled_mma<...>
```

### MLIR nvgpu (Lowered)
```mlir
nvgpu.warpgroup.mma.sync %fragA, %fragB, %fragC
  {m=64, n=64, k=16} : (vector<8xf16>, vector<8xf16>, vector<4xf32>)
```

### PTX (Final)
```ptx
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {...};
```

## Memory Hierarchy Mapping

| CuTe Concept | Hardware | Address Space | MLIR Type |
|--------------|----------|---------------|-----------|
| Global Tensor | HBM/GDDR | global | `memref<?xf16, #gpu.address_space<global>>` |
| Shared Tensor | SMEM | shared | `memref<128x128xf16, #gpu.address_space<workgroup>>` |
| Register Fragment | RF | register | `vector<8xf16>` |
| TMA Descriptor | L2/SMEM | constant | `!llvm.ptr` |

## Design Patterns

### Pattern 1: Layout Composition
```mlir
// Thread-local partition
%thr_layout = cute.local_partition %tensor, %tiler, %thread_idx
// Then index
%offset = cute.crd2idx %coord, cute.get_layout %thr_layout
```

### Pattern 2: TMA Pipeline (Hopper)
```mlir
// Setup (host-side)
%tma = cute_nvgpu.make_tma_load !f16, [128,128], "128B"
cute_nvgpu.atom_set_value %tma, "tma_descriptor", %desc_ptr

// Execution (device-side)
%barrier = nvgpu.mbarrier.create : !nvgpu.mbarrier
%exec_tma = cute_nvgpu.make_exec_tma_load %tma, %barrier
cute_nvgpu.mbarrier_arrive %barrier
cute_nvgpu.tma_load_execute %exec_tma, %src, %dst, %x, %y
cute_nvgpu.mbarrier_wait %barrier, %phase
```

### Pattern 3: Warpgroup MMA (Hopper)
```mlir
%atom = cute_nvgpu.make_mma_atom !f16, !f16, !f32, [64,128,16], "SM90"
%tiled_mma = cute_nvgpu.make_tiled_mma %atom, %layout_tv, %tiler

%fragA = cute.mma_make_fragment_a %tiled_mma, %tensorA
%fragB = cute.mma_make_fragment_b %tiled_mma, %tensorB
%fragC = cute.mma_make_fragment_c %tiled_mma, %tensorC

%result = cute_nvgpu.warpgroup_mma %fragA, %fragB, %fragC, %atom
```

## File Organization

```
cute_ir_tablegen/
├── CuteDialect.td           (181 lines) - Core types
├── CuteOps.td              (442 lines) - Layout algebra ops
├── CuteNvgpuDialect.td     (260 lines) - GPU types
├── CuteNvgpuOps.td         (337 lines) - GPU hardware ops
├── CMakeLists.txt          - MLIR TableGen build
├── README.md               - Usage documentation
├── SUMMARY.md              - This file
└── examples/
    └── cute_gemm_example.mlir - MLIR code sample
```

## References

- **CUTLASS CuTe**: https://github.com/NVIDIA/cutlass/tree/main/include/cute
- **MLIR NVGPU Dialect**: https://mlir.llvm.org/docs/Dialects/NVGPU/
- **Hopper TMA**: CUDA Programming Guide Appendix J
- **TableGen Syntax**: https://mlir.llvm.org/docs/DefiningDialects/
