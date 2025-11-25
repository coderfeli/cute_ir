#!/usr/bin/env python3
"""GPU kernel tests using rocdsl Python API with REAL GPU EXECUTION"""

import sys
sys.path.insert(0, '/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import arith, memref, scf
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes


def compile_to_hsaco(mlir_module):
    lowered = run_pipeline(
        mlir_module,
        Pipeline()
        .canonicalize()
        .cse()
        .rocdl_attach_target(chip="gfx942")
        .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP"))
        .gpu_to_llvm()
        .lower_to_llvm()
        .gpu_module_to_binary(format="bin")
    )
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)


def test_vector_add():
    """Vector addition with GPU execution and validation"""
    print("\n" + "="*80)
    print("Test 1: Vector Addition (C = A + B) on GPU")
    print("="*80)
    
    SIZE = 2048
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("vec_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def vecAdd(A: T.memref(SIZE, T.f32()), B: T.memref(SIZE, T.f32()), C: T.memref(SIZE, T.f32())):
        tid = arith.addi(arith.muli(gpu.block_id("x"), gpu.block_dim("x")), gpu.thread_id("x"))
        size_c = arith.constant(T.index(), SIZE)
        valid = arith.cmpi(arith.CmpIPredicate.slt, tid, size_c)
        with ir.InsertionPoint(scf.IfOp(valid).then_block):
            a = memref.load(A, [tid])
            b = memref.load(B, [tid])
            c = arith.addf(a, b)
            memref.store(c, C, [tid])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    print("✓ MLIR module created")
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    np.random.seed(42)
    a_host = np.random.randn(SIZE).astype(np.float32)
    b_host = np.random.randn(SIZE).astype(np.float32)
    c_host = np.zeros(SIZE, dtype=np.float32)
    expected = a_host + b_host
    
    d_a = hip_check(hip.hipMalloc(SIZE * 4))
    d_b = hip_check(hip.hipMalloc(SIZE * 4))
    d_c = hip_check(hip.hipMalloc(SIZE * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, SIZE * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, SIZE * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"vecAdd"))
    
    threads_per_block = 256
    num_blocks = (SIZE + threads_per_block - 1) // threads_per_block
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    print(f"✓ Launching: {num_blocks} blocks × {threads_per_block} threads")
    hip_check(hip.hipModuleLaunchKernel(kernel_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, SIZE * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(c_host - expected))
    
    print(f"✓ Max error: {error:.2e}")
    print(f"  Results[:5]: {c_host[:5]}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if error < 1e-5:
        print("✅ TEST PASSED!")
        return True
    else:
        print(f"❌ TEST FAILED: error = {error}")
        return False


def test_matrix_transpose():
    """Matrix transpose with GPU execution and validation"""
    print("\n" + "="*80)
    print("Test 2: Matrix Transpose (B = A^T) on GPU")
    print("="*80)
    
    M, N = 32, 64
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("transpose_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def matrixTranspose(A: T.memref(M, N, T.f32()), B: T.memref(N, M, T.f32())):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        row = arith.addi(arith.muli(by, arith.constant(T.index(), 16)), ty)
        col = arith.addi(arith.muli(bx, arith.constant(T.index(), 16)), tx)
        
        m_c = arith.constant(T.index(), M)
        n_c = arith.constant(T.index(), N)
        
        row_valid = arith.cmpi(arith.CmpIPredicate.slt, row, m_c)
        col_valid = arith.cmpi(arith.CmpIPredicate.slt, col, n_c)
        valid = arith.andi(row_valid, col_valid)
        
        with ir.InsertionPoint(scf.IfOp(valid).then_block):
            val = memref.load(A, [row, col])
            memref.store(val, B, [col, row])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    print("✓ MLIR module created")
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    np.random.seed(123)
    a_host = np.random.randn(M, N).astype(np.float32)
    b_host = np.zeros((N, M), dtype=np.float32)
    expected = a_host.T
    
    d_a = hip_check(hip.hipMalloc(M * N * 4))
    d_b = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matrixTranspose"))
    
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    print(f"✓ Launching: ({grid_x}, {grid_y}) blocks × ({block_size}, {block_size}) threads")
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(b_host - expected))
    
    print(f"✓ Max error: {error:.2e}")
    print(f"  B[0,:5]: {b_host[0,:5]}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if error < 1e-5:
        print("✅ TEST PASSED!")
        return True
    else:
        print(f"❌ TEST FAILED: error = {error}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ROCm GPU Execution Tests with rocdsl Python API")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    result1 = test_vector_add()
    result2 = test_matrix_transpose()
    
    print("\n" + "="*80)
    if result1 and result2:
        print("🎉 ALL GPU TESTS PASSED!")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED")
        sys.exit(1)
