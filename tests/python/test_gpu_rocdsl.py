#!/usr/bin/env python3
"""GPU kernel tests demonstrating integration with Rocir Layout concepts"""

import sys
sys.path.insert(0, '/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu, rocir
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


def demonstrate_rocir_layouts():
    """
    Demonstrate Rocir layout algebra concepts at host level.
    Layouts describe how multi-dimensional tensors map to linear memory.
    """
    print("\n" + "="*80)
    print("Rocir Layout Algebra Demo (Host-Side)")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    with ctx.context:
        # Example 1: 1D contiguous vector layout
        size_1d = arith.constant(T.index(), 1024)
        stride_1d = arith.constant(T.index(), 1)
        
        shape_1d = rocir.make_shape(size_1d)
        stride_vec = rocir.make_stride(stride_1d)
        layout_1d = rocir.make_layout(shape_1d, stride_vec)
        
        print("✓ 1D Vector Layout:")
        print("  - Shape: (1024,)")
        print("  - Stride: (1,) [contiguous]")
        print("  - Usage: Maps vector index i to memory offset i*1")
        
        # Example 2: 2D row-major matrix layout
        m = arith.constant(T.index(), 32)
        n = arith.constant(T.index(), 64)
        one = arith.constant(T.index(), 1)
        
        shape_2d = rocir.make_shape(m, n)
        stride_row_major = rocir.make_stride(n, one)  # (64, 1)
        layout_row_major = rocir.make_layout(shape_2d, stride_row_major)
        
        print("\n✓ 2D Row-Major Matrix Layout (32×64):")
        print("  - Shape: (32, 64)")
        print("  - Stride: (64, 1)")
        print("  - Usage: A[i,j] maps to offset i*64 + j*1")
        
        # Example 3: 2D column-major matrix layout
        stride_col_major = rocir.make_stride(one, m)  # (1, 32)
        layout_col_major = rocir.make_layout(shape_2d, stride_col_major)
        
        print("\n✓ 2D Column-Major Matrix Layout (32×64):")
        print("  - Shape: (32, 64)")
        print("  - Stride: (1, 32)")
        print("  - Usage: A[i,j] maps to offset i*1 + j*32")
        
        # Layout composition: Tiling
        tile_m = arith.constant(T.index(), 16)
        tile_n = arith.constant(T.index(), 16)
        tile_shape = rocir.make_shape(tile_m, tile_n)
        
        print("\n✓ Layout Composition (Tiling):")
        print("  - Original: (32, 64) with stride (64, 1)")
        print("  - Tile: (16, 16)")
        print("  - Composition creates hierarchical layout for blocked access")
        print("  - Available: logical_product, tiled_product, blocked_product")
        
        print("\n✓ Available Rocir Layout Operations:")
        print("  - Construction: make_shape, make_stride, make_layout")
        print("  - Query: size, cosize, rank, get_shape, get_stride")
        print("  - Composition: composition, logical_product, zipped_product,")
        print("                 tiled_product, flat_product, raked_product, blocked_product")
        print("  - Partitioning: logical_divide, zipped_divide, tiled_divide, flat_divide,")
        print("                  local_partition, local_tile")


def test_vector_add():
    """Vector addition with layout annotations"""
    print("\n" + "="*80)
    print("Test 1: Vector Addition (C = A + B)")
    print("Layout: 1D contiguous, shape=(2048,), stride=(1,)")
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
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, SIZE * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(c_host - expected))
    print(f"✓ Max error: {error:.2e}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5


def test_matrix_transpose():
    """Matrix transpose demonstrating layout transformations"""
    print("\n" + "="*80)
    print("Test 2: Matrix Transpose (B = A^T)")
    print("Layout A: shape=(32,64), stride=(64,1) [row-major]")
    print("Layout B: shape=(64,32), stride=(32,1) [row-major, transposed]")
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
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(b_host - expected))
    print(f"✓ Max error: {error:.2e}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5


def test_matmul():
    """Matrix multiply"""
    print("\n" + "="*80)
    print("Test 3: Matrix Multiply (C = A * B)")
    print("Layout A (32×64): shape=(32,64), stride=(64,1) [row-major]")
    print("Layout B (64×32): shape=(64,32), stride=(32,1) [row-major]")
    print("Layout C (32×32): shape=(32,32), stride=(32,1) [row-major]")
    print("="*80)
    
    M, N, K = 32, 32, 64
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("matmul_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def matmul(A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        row = arith.addi(arith.muli(by, arith.constant(T.index(), 16)), ty)
        col = arith.addi(arith.muli(bx, arith.constant(T.index(), 16)), tx)
        
        m_c = arith.constant(T.index(), M)
        n_c = arith.constant(T.index(), N)
        k_c = arith.constant(T.index(), K)
        one = arith.constant(T.index(), 1)
        
        row_valid = arith.cmpi(arith.CmpIPredicate.slt, row, m_c)
        col_valid = arith.cmpi(arith.CmpIPredicate.slt, col, n_c)
        valid = arith.andi(row_valid, col_valid)
        
        with ir.InsertionPoint(scf.IfOp(valid).then_block):
            sum_val = arith.constant(T.f32(), 0.0)
            k_idx = arith.constant(T.index(), 0)
            
            for_op = scf.ForOp(k_idx, k_c, one, [sum_val])
            with ir.InsertionPoint(for_op.body):
                k = for_op.induction_variable
                acc = for_op.inner_iter_args[0]
                
                a_val = memref.load(A, [row, k])
                b_val = memref.load(B, [k, col])
                prod = arith.mulf(a_val, b_val)
                new_acc = arith.addf(acc, prod)
                
                scf.yield_([new_acc])
            
            result = for_op.results[0]
            memref.store(result, C, [row, col])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    np.random.seed(456)
    a_host = np.random.randn(M, K).astype(np.float32)
    b_host = np.random.randn(K, N).astype(np.float32)
    c_host = np.zeros((M, N), dtype=np.float32)
    expected = a_host @ b_host
    
    d_a = hip_check(hip.hipMalloc(M * K * 4))
    d_b = hip_check(hip.hipMalloc(K * N * 4))
    d_c = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * K * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, K * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matmul"))
    
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)
    
    print(f"✓ Max absolute error: {error:.2e}")
    print(f"✓ Max relative error: {rel_error:.2e}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return rel_error < 1e-3


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ROCm GPU Tests - Integration with Rocir Layout Algebra")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    # Demonstrate Rocir layout concepts
    demonstrate_rocir_layouts()
    
    # Run GPU tests
    result1 = test_vector_add()
    result2 = test_matrix_transpose()
    result3 = test_matmul()
    
    print("\n" + "="*80)
    if result1 and result2 and result3:
        print("🎉 ALL GPU TESTS PASSED!")
        print("\nNote: Rocir layout algebra demonstrated at host level.")
        print("Future work: Add lowering passes to use layouts in GPU kernels.")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED")
        sys.exit(1)
