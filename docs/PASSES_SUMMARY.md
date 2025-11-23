# CuTe IR Lowering Passes - 技术总结

## 文件清单

### TableGen 定义 (5 个 .td 文件)
| 文件 | 行数 | 内容 |
|------|------|------|
| CuteDialect.td | 181 | cute_ir 类型系统定义 |
| CuteOps.td | 442 | 85+ Layout 代数操作 |
| CuteNvgpuDialect.td | 260 | GPU 硬件感知类型 |
| CuteNvgpuOps.td | 393 | 30+ GPU 硬件操作 |
| **CutePasses.td** | **493** | **18 个 Pass 定义** |

### Pass 实现 (2 个 .cpp 文件)
| 文件 | 行数 | 功能 |
|------|------|------|
| CuteToStandard.cpp | 239 | cute_ir → arith/scf/memref |
| CuteNvgpuToNvgpu.cpp | 370 | cute_nvgpu_ir → nvgpu |

### 文档 (4 个 .md 文件)
- README.md (128 行)
- SUMMARY.md (218 行)
- PassPipeline.md (47 行)
- PASSES_SUMMARY.md (本文件)

## Pass 分类体系

### 1. Lowering Passes (降级转换) - 3 个

#### CuteToStandardPass
```
cute_ir → Standard MLIR Dialects (arith, scf, memref)
```
**核心转换**:
- `cute.crd2idx` → `arith.muli + arith.addi` 链
- `cute.copy` → `scf.for + memref.load/store`
- `cute.local_partition` → Thread-indexed access

**示例**:
```mlir
// Before:
%idx = cute.crd2idx %coord, %layout

// After:
%tmp0 = arith.muli %coord_0, %stride_0
%tmp1 = arith.muli %coord_1, %stride_1
%idx = arith.addi %tmp0, %tmp1
```

#### CuteNvgpuToNvgpuPass
```
cute_nvgpu_ir → MLIR nvgpu dialect
```
**支持操作**:
- Warpgroup MMA (SM90+): `cute_nvgpu.warpgroup_mma` → `nvgpu.warpgroup.mma.sync`
- Warp MMA (SM80+): `cute_nvgpu.warp_mma_*` → `nvgpu.mma.sync`
- TMA (SM90+): `cute_nvgpu.tma_load_execute` → `nvgpu.tma.async.load`
- Copy: `ldmatrix`, `cp.async`

**参数**:
- `--target-arch`: sm_80 / sm_90 / sm_100
- `--enable-tma`: 启用 TMA 操作 (SM90+)

#### CuteNvgpuTMAMaterializationPass
```
TMA descriptor lifecycle management
```
**处理流程**:
1. Host: `cuTensorMapEncodeTiled()` 创建 descriptor
2. Device: `atom_set_value` 设置 descriptor pointer
3. Kernel: `tma_load_execute` 使用 descriptor

### 2. Optimization Passes (优化) - 6 个

#### CuteLayoutCanonicalizePass
**作用**: 代数化简、常量折叠
- 消除恒等变换: `compose(L, Identity) → L`
- 合并 coalesce: `coalesce(coalesce(L)) → coalesce(L)`
- 静态计算 shape/stride

#### CuteLayoutFusionPass
**作用**: 融合相邻 Layout 操作
- 减少运行时计算
- 改善常量传播
- 简化代码生成

#### CuteVectorizationPass
**作用**: 基于 Layout 自动向量化
- 检测 contiguous access (stride == 1)
- 生成 `vector<4xf16>`, `vector<8xf16>` 等
- 尊重 alignment 约束

#### CuteMemoryCoalescingPass
**作用**: 优化全局内存访问模式
- 检测非合并访问
- 建议 `blocked_product` / `raked_product`
- 确保相邻线程访问相邻内存

#### CuteSMEMSwizzlingPass
**作用**: Shared memory swizzling (避免 bank conflict)
- 支持 32B/64B/128B swizzle (Hopper)
- XOR-based 置换
- 兼容 TMA 对齐要求

#### CuteTensorPartitionPass
**作用**: 将分区操作具体化为显式索引
- `local_partition` → Thread-local 计算
- `local_tile` → Tile 提取 + 边界检查
- `slice` → 维度削减

### 3. Pipeline Passes (流水线) - 2 个

#### CuteAsyncPipelinePass
**功能**: 插入异步拷贝流水线 (重叠计算和数据传输)

**Ampere (cp.async, 最多 8 stages)**:
```mlir
// Prologue: Fill pipeline
scf.for %stage = 0 to %depth {
  nvgpu.device_async_copy ...
  nvgpu.device_async_create_group
}

// Steady state
scf.for %k {
  nvgpu.device_async_wait {numGroups = depth-1}
  nvgpu.mma.sync ...
  nvgpu.device_async_copy ...  // Prefetch next
}

// Epilogue: Drain
scf.for %stage { ... }
```

**Hopper (TMA + mbarrier)**:
- Producer thread 发出 TMA
- mbarrier 协调阶段
- Warpgroup MMA 异步消费

**参数**:
- `--pipeline-depth`: 2-8 (默认 2)
- `--warp-specialization`: Hopper producer-consumer

#### CuteWarpSpecializationPass
**功能**: Warp 角色分工 (SM90+)

**Producer Warps**:
- 发出 TMA 操作
- 管理 mbarrier phases
- 最小寄存器使用

**Consumer Warps**:
- 执行 warpgroup MMA
- 等待 mbarrier
- 完整寄存器压力

**参数**:
- `--num-producer-warps`: 通常为 1 (TMA issuer)

### 4. Analysis Passes (分析) - 2 个

#### CuteLayoutAnalysisPass
**输出指标**:
- Coalescing score (合并分数)
- Bank conflict probability (bank 冲突概率)
- Vectorization width (向量化宽度)
- Alignment guarantees (对齐保证)

**参数**:
- `--print-analysis`: 打印分析结果

#### CuteAtomValidationPass
**验证内容**:
- Shape 兼容性 (M, N, K 维度)
- Element type 支持 (f16/bf16/tf32/fp8/int8)
- Architecture 可用性 (SM 版本)
- TMA descriptor 有效性 (对齐、swizzle)

**错误报告**:
- 不支持的 shape
- 不匹配的 fragment 类型
- 非法 TMA swizzle 模式

### 5. Utility Passes (辅助) - 3 个

#### CuteNvgpuMmaLoweringPass
**作用**: TiledMma 分解为架构特定 MMA 指令
- Hopper: warpgroup layout → warpgroup.mma.sync
- Ampere: warp layout → mma.sync (m16n8k16)

#### CuteNvgpuCopyLoweringPass
**作用**: TiledCopy 选择最优拷贝指令
- Global→Shared (aligned, SM80+): `cp.async`
- Global→Shared (SM90+, large): `TMA`
- Shared→Register (matrix): `ldmatrix`
- Default: 向量化 load/store

#### CuteFullLoweringPipeline (Pipeline)
**完整编译流程** (6 个阶段):
1. Optimization (cute_ir level)
2. GPU preparation
3. Async pipeline insertion
4. Lowering (cute_nvgpu → nvgpu)
5. Further lowering (nvgpu → nvvm → LLVM)
6. Final optimizations

**参数**:
- `--target-arch`: sm_80 / sm_90 / sm_100
- `--O`: 优化级别 0-3

## 编译流程示例

### 基础流程 (Ampere SM80)
```bash
mlir-opt input.mlir \
  --cute-layout-canonicalize \
  --cute-tensor-partition \
  --cute-nvgpu-to-nvgpu="target-arch=sm_80 enable-tma=false" \
  --cute-async-pipeline="pipeline-depth=4" \
  --convert-nvgpu-to-nvvm \
  --gpu-to-llvm \
  -o output.mlir
```

### 高级流程 (Hopper SM90)
```bash
mlir-opt input.mlir \
  --cute-layout-canonicalize \
  --cute-layout-fusion \
  --cute-memory-coalescing \
  --cute-smem-swizzling \
  --cute-warp-specialization="num-producer-warps=1" \
  --cute-nvgpu-to-nvgpu="target-arch=sm_90 enable-tma=true" \
  --cute-nvgpu-tma-materialize \
  --cute-async-pipeline="pipeline-depth=2 warp-specialization=true" \
  --convert-nvgpu-to-nvvm \
  -o hopper_output.mlir
```

### 完整 Pipeline (一键编译)
```bash
mlir-opt input.mlir \
  --cute-full-lowering="target-arch=sm_90 O=2" \
  -o optimized_output.mlir
```

## Pass 依赖关系

```
CuteLayoutCanonicalizePass
  ↓
CuteLayoutFusionPass
  ↓
CuteLayoutAnalysisPass
  ↓
CuteTensorPartitionPass
  ↓
CuteMemoryCoalescingPass + CuteSMEMSwizzlingPass
  ↓
CuteNvgpuToNvgpuPass
  ↓
CuteNvgpuMmaLoweringPass + CuteNvgpuCopyLoweringPass
  ↓
CuteAsyncPipelinePass + CuteWarpSpecializationPass
  ↓
CuteNvgpuTMAMaterializationPass (if TMA enabled)
  ↓
Standard MLIR lowering (nvgpu→nvvm→LLVM)
```

## 性能优化建议

### Layout Level
1. ✅ 使用 `cute-layout-canonicalize` 尽早简化
2. ✅ 启用 `cute-layout-fusion` 减少运行时计算
3. ✅ 运行 `cute-layout-analysis` 检查 coalescing score

### GPU Level
1. ✅ **必须启用** `cute-smem-swizzling` (Hopper)
2. ✅ 使用 `cute-memory-coalescing` 验证访问模式
3. ✅ 启用 `cute-vectorization` (stride == 1 时)

### Pipeline Level
1. ✅ Ampere: `pipeline-depth=2~4` (cp.async)
2. ✅ Hopper: `pipeline-depth=2` + `warp-specialization=true` (TMA)
3. ✅ 监控寄存器使用 (避免溢出)

### MMA Level
1. ✅ 优先 Warpgroup MMA (Hopper, 4x 吞吐)
2. ✅ 选择合适的 fragment size (寄存器 vs 复用)
3. ✅ FP32 accumulator (精度) vs TF32 (速度)

## 总结

### 统计数据
- **Pass 总数**: 18 个
- **Lowering Passes**: 3 个
- **Optimization Passes**: 6 个
- **Pipeline Passes**: 2 个
- **Analysis Passes**: 2 个
- **Utility Passes**: 3 个
- **Pipeline**: 1 个
- **Pipeline**: 1 个

### 核心创新
1. **Layout-aware lowering**: 基于 Layout 信息优化内存访问
2. **Architecture-specific**: SM80/SM90/SM100 差异化处理
3. **Automatic pipelining**: 自动插入异步拷贝流水线
4. **Warp specialization**: Hopper producer-consumer 模式

### 适用场景
- ✅ GEMM / Convolution 等密集计算
- ✅ 需要高吞吐的矩阵运算
- ✅ Hopper GPU (TMA + Warpgroup MMA)
- ✅ 自动化 GPU kernel 生成
