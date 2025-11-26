#!/usr/bin/env python3
"""
Simple MFMA GEMM 64x64x64 - Single wavefront implementation
Uses rocdl.mfma_f32_16x16x16f16 for FP16->FP32 matrix multiplication
"""

import sys
sys.path.insert(0, '/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu, arith
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import numpy as np
from mlir import ir
from mlir.dialects import memref, scf, vector, rocdl
from mlir.dialects.arith import ConstantOp
import mlir.extras.types as T
from hip import hip
import ctypes

def compile_to_hsaco(mlir_module):
    lowered = run_pipeline(mlir_module, Pipeline().canonicalize().cse()
        .rocdl_attach_target(chip="gfx942")
        .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP"))
        .gpu_to_llvm().lower_to_llvm().gpu_module_to_binary(format="bin"))
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

gpu_arch = get_hip_arch()

@gpu.module("mfma_gemm", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">'])
def gpu_mod():
    pass

ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
ip.__enter__()

@gpu.func(emit=True)
def gemm_kernel(A: T.memref(64, 64, T.f16()), B: T.memref(64, 64, T.f16()), C: T.memref(64, 64, T.f32())):
    """
    Single wavefront GEMM: 64 threads compute 16x16 output tile
    Each thread computes 4 output elements (one column of 16x16/4)
    """
    lane_id = gpu.thread_id('x')
    
    # Decompose thread ID: row_group (0-3), col (0-15)
    row_group = (lane_id // arith.index(16))._value  
    col = (lane_id % arith.index(16))._value
    
    # Loop constants
    c_0 = arith.index(0)._value
    c_64 = arith.index(64)._value
    c_16 = arith.index(16)._value
    
    # Initialize accumulator
    f32_zero = arith.f32(0.0)
    acc = vector.splat(T.vector(4, T.f32()), f32_zero.value)
    
    # K-loop: 4 iterations (64/16)
    for_op = scf.ForOp(c_0.value, c_64.value, c_16.value, [acc])
    
    with ir.InsertionPoint(for_op.body):
        k_start = for_op.induction_variable
        acc_iter = for_op.inner_iter_args[0]
        
        # Build A and B vectors (4 elements each)
        f16_zero_val = ConstantOp(T.f16(), ir.FloatAttr.get(T.f16(), 0.0)).result
        a_vec = vector.splat(T.vector(4, T.f16()), f16_zero_val)
        b_vec = vector.splat(T.vector(4, T.f16()), f16_zero_val)
        
        # Load A[row_group*4+i, k_start+col] for i=0..3
        for i in range(4):
            row_idx = (row_group * arith.index(4) + arith.index(i))._value
            col_idx = (k_start + col)._value
            val = memref.load(A, [row_idx.value if hasattr(row_idx, "value") else row_idx, 
                                 col_idx.value if hasattr(col_idx, "value") else col_idx])
            a_vec = vector.InsertOp(val, a_vec, [], [i]).result
        
        # Load B[k_start+col, row_group*4+i] for i=0..3
        for i in range(4):
            row_idx = (k_start + col)._value
            col_idx = (row_group * arith.index(4) + arith.index(i))._value
            val = memref.load(B, [row_idx.value if hasattr(row_idx, "value") else row_idx,
                                 col_idx.value if hasattr(col_idx, "value") else col_idx])
            b_vec = vector.InsertOp(val, b_vec, [], [i]).result
        
        # MFMA: C[16x16] += A[16x16] @ B[16x16]
        i32_zero = arith.i32(0)
        new_acc = rocdl.mfma_f32_16x16x16f16(T.vector(4, T.f32()), 
                                             [a_vec, b_vec, acc_iter, 
                                              i32_zero.value, i32_zero.value, i32_zero.value])
        scf.yield_([new_acc])
    
    # Store results: C[row_group*4+i, col] for i=0..3
    final_acc = for_op.results[0]
    for i in range(4):
        row_idx = (row_group * arith.index(4) + arith.index(i))._value
        val = vector.extractelement(final_acc, position=arith.constant(T.i32(), i))
        memref.store(val, C, [row_idx.value if hasattr(row_idx, "value") else row_idx, 
                              col.value if hasattr(col, "value") else col])

ip.__exit__(None, None, None)

# ============================================================================
# Test execution
# ============================================================================

print("="*80)
print("MFMA GEMM 64x64x64 - Single Wavefront Test")
print("="*80)

hsaco = compile_to_hsaco(ctx.module)
print("Compiled:", len(hsaco), "bytes")

# Prepare test data
np.random.seed(42)
a = np.random.randn(64, 64).astype(np.float16)
b = np.random.randn(64, 64).astype(np.float16)
c = np.zeros((64, 64), dtype=np.float32)
expect = a.astype(np.float32) @ b.astype(np.float32)

# Allocate GPU memory
d_a = hip_check(hip.hipMalloc(64*64*2))
d_b = hip_check(hip.hipMalloc(64*64*2))
d_c = hip_check(hip.hipMalloc(64*64*4))
hip_check(hip.hipMemcpy(d_a, a.ctypes.data, 64*64*2, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(d_b, b.ctypes.data, 64*64*2, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(d_c, c.ctypes.data, 64*64*4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# Load and launch kernel
mod = hip_check(hip.hipModuleLoadData(hsaco))
kern = hip_check(hip.hipModuleGetFunction(mod, b"gemm_kernel"))
args_list = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
args = (ctypes.c_void_p * 3)(*[ctypes.addressof(p) for p in args_list])

hip_check(hip.hipModuleLaunchKernel(kern, 1, 1, 1, 64, 1, 1, 0, 0, args, None))
hip_check(hip.hipDeviceSynchronize())
hip_check(hip.hipMemcpy(c.ctypes.data, d_c, 64*64*4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# Verify results (16x16 tile only)
res = c[:16, :16]
exp = expect[:16, :16]
err = np.max(np.abs(res - exp)) / (np.max(np.abs(exp)) + 1e-8)

print("Error:", err)
print("Sample result:")
print(res[:3, :3])
print("Expected:")
print(exp[:3, :3])

# Cleanup
hip_check(hip.hipFree(d_a))
hip_check(hip.hipFree(d_b))
hip_check(hip.hipFree(d_c))
hip_check(hip.hipModuleUnload(mod))

print("="*80)
if err < 1e-2:
    print("✅ PASSED!")
else:
    print("❌ FAILED! Error:", err)
print("="*80)
