# CuTe IR TableGen Definitions

基于 CUTLASS CuTe DSL Python API 逆向分析重构的 MLIR TableGen 定义。

## 文件说明

### 已创建
- ✅ `CuteDialect.td` - CuTe 核心 dialect (类型系统)

### 待完整创建
由于文件较大，其他文件内容请参考原始回复或手动创建：

- `CuteOps.td` - 85+ Layout 代数操作定义
- `CuteNvgpuDialect.td` - GPU 硬件感知 dialect
- `CuteNvgpuOps.td` - GPU 硬件操作定义
- `CMakeLists.txt` - 构建配置
- `SUMMARY.md` - 技术总结

## 核心类型 (CuteDialect.td)

```
cute_ir:
- IntType - 编译期整数
- ShapeType - 多维形状
- StrideType - 多维步幅
- LayoutType - Shape + Stride 映射
- TileType - 层次化分块
- CoordType - 多维坐标
- TensorType - 数据视图
```

## 核心操作分类

### 构造 (5 ops)
- make_shape, make_stride, make_layout, make_coord, make_tile

### 查询 (5 ops)
- get_shape, get_stride, rank, size, depth

### 映射 (2 ops)
- crd2idx: coordinate → linear index
- idx2crd: linear index → coordinate

### 变换 (6 ops)
- composition, complement, left_inverse, right_inverse
- coalesce, filter

### 积 (4 ops)
- logical_product, blocked_product, raked_product, tiled_product

### 分区 (5 ops)
- local_partition, local_tile, slice, zipped_divide

### 张量 (10+ ops)
- make_tensor, copy, tensor_load, tensor_store

### MMA (5 ops)
- make_tiled_mma, mma_make_fragment, tiled_mma_partition

### Copy (5 ops)
- make_tiled_copy, copy_atom_call, tiled_copy_partition_S/D

## GPU 扩展类型 (cute_nvgpu_ir)

```
- TiledMmaType - 封装 Atom + LayoutTV + Tiler
- TiledCopyType - 封装 Atom + LayoutTV + Tiler  
- MmaAtomType - MMA 指令元数据
- CopyAtomType - Copy 指令元数据
- TmaLoadType/TmaStoreType - TMA 操作
```

## 架构支持

- Ampere (SM80+): warp_mma_f16bf16, warp_mma_sparse
- Hopper (SM90+): warpgroup_mma, TMA operations
- Blackwell (SM100+): tcgen05_mma, tcgen05_block_scaled_mma

## 编译流程

```
Python CuTe DSL
    ↓
cute_ir (Layout 代数)
    ↓
cute_nvgpu_ir (GPU 硬件感知)
    ↓ [Lowering Pass]
gpu/nvgpu dialect (MLIR 标准)
    ↓
nvvm dialect
    ↓
LLVM IR → PTX → SASS
```

## 使用方法

### 生成 C++ 代码
```bash
mlir-tblgen --gen-dialect-decls CuteDialect.td -I /path/to/mlir/include
mlir-tblgen --gen-op-decls CuteOps.td -I /path/to/mlir/include
```

### MLIR 代码示例
```mlir
// 创建 Layout
%shape = cute.make_shape %c64, %c128 : !cute.shape<(64, 128)>
%layout = cute.make_layout %shape, %stride : !cute.layout<...>

// 创建 Tensor
%tensor = cute.make_tensor %ptr, %layout : !cute.tensor<f16, ...>

// 坐标映射
%idx = cute.crd2idx %coord, %layout : index

// TiledMma
%mma_atom = cute_nvgpu.warpgroup_mma !f16, !f16, !f32, [64, 64, 16]
%tiled_mma = cute.make_tiled_mma %mma_atom, %atom_layout
```

## 参考资源

- [CUTLASS CuTe Docs](https://github.com/NVIDIA/cutlass)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [TableGen Reference](https://mlir.llvm.org/docs/DefiningDialects/)

## License

遵循 CUTLASS 项目许可证 (NVIDIA Proprietary License)。
