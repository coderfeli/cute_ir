#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using @gpu.func decorator pattern."""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.runtime.fp8_util import to_byte
from utils import compile_to_hsaco
import numpy as np
from mlir import ir
from mlir.dialects import vector, memref, builtin
from rocdsl.dialects.ext import arith, scf, gpu
from mlir.dialects import arith as _arith_mlir
import mlir.dialects.rocdl as rocdl
import mlir.extras.types as T
from hip import hip
import ctypes

def unwrap(v):
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    return v


def test_mfma_fp8_rocir():
    print("="*80)
    print("MFMA FP8 GEMM Test (@gpu.func Decorator) - 1024x1024x1280")
    print("="*80)
    
    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")

    # Constants
    M, N, K = 1024, 1024, 1280
    
    ctx = RAIIMLIRContextModule()
    
    f8 = ir.Float8E4M3FNType.get()
    f32 = ir.F32Type.get()
    
    size_c = M * N
    size_a = M * K
    size_b = N * K  # Transposed B (NxK)
    
    # LDS Globals (Tile size 32x128)
    lds_mem_type = ir.MemRefType.get([4096], f8, memory_space=ir.Attribute.parse("3"))
    
    # Create LDS globals before the module
    with ir.InsertionPoint(ctx.module.body):
        lds_a_global = memref.GlobalOp(
            sym_name="lds_a",
            type_=lds_mem_type,
            initial_value=ir.UnitAttr.get(),
            sym_visibility=ir.StringAttr.get("private")
        )
        lds_b_global = memref.GlobalOp(
            sym_name="lds_b",
            type_=lds_mem_type,
            initial_value=ir.UnitAttr.get(),
            sym_visibility=ir.StringAttr.get("private")
        )
    
    @gpu.module("mfma_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'])
    def gpu_mod():
        
        @gpu.func(emit=True)
        def kernel(
            arg_c: T.memref(size_c, T.f32()),
            arg_a: T.memref(size_a, f8),
            arg_b: T.memref(size_b, f8)
        ):
            c0_i32 = arith.ConstantOp(ir.IntegerType.get_signless(32), 0).result
            c128 = arith.ConstantOp(ir.IndexType.get(), 128).result
            c32 = arith.ConstantOp(ir.IndexType.get(), 32).result
            c16 = arith.ConstantOp(ir.IndexType.get(), 16).result
            c8 = arith.ConstantOp(ir.IndexType.get(), 8).result
            c4 = arith.ConstantOp(ir.IndexType.get(), 4).result
            c2 = arith.ConstantOp(ir.IndexType.get(), 2).result
            c64 = arith.ConstantOp(ir.IndexType.get(), 64).result
            c1024 = arith.ConstantOp(ir.IndexType.get(), 1024).result
            
            identity_map = ir.AffineMap.get_identity(1)
            
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            lds_a = memref.GetGlobalOp(lds_mem_type, ir.FlatSymbolRefAttr.get("lds_a")).result
            lds_b = memref.GetGlobalOp(lds_mem_type, ir.FlatSymbolRefAttr.get("lds_b")).result
            
            # Accumulator Init
            vec4_f32 = ir.VectorType.get([4], f32)
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            # Global Load Indices
            tx_16 = arith.MulIOp(tx._value, c16).result
            
            row_a_local = arith.DivUIOp(tx_16, c128).result
            col_a_local = arith.RemUIOp(tx_16, c128).result
            
            bx_32 = arith.MulIOp(bx._value, c32).result
            row_a_global = arith.AddIOp(bx_32, row_a_local).result
            
            # For B (Transposed)
            row_b_local = arith.DivUIOp(tx_16, c128).result
            col_b_local = arith.RemUIOp(tx_16, c128).result
            
            by_32 = arith.MulIOp(by._value, c32).result
            row_b_global = arith.AddIOp(by_32, row_b_local).result
            
            # LDS Write Index
            lds_write_idx = tx_16
            
            vec16_f8 = ir.VectorType.get([16], f8)
            pad_f8 = _arith_mlir.ConstantOp(f8, ir.FloatAttr.get(f8, 0.0)).result
            
            # Pre-calculate LDS read indices
            wave_id = arith.DivUIOp(tx._value, c64).result
            lane_id = arith.RemUIOp(tx._value, c64).result
            
            wave_row = arith.DivUIOp(wave_id, c2).result
            wave_col = arith.RemUIOp(wave_id, c2).result
            
            lane_mod_16 = arith.RemUIOp(lane_id, c16).result
            lane_div_16 = arith.DivUIOp(lane_id, c16).result
            
            row_a_lds_base = arith.MulIOp(wave_row, c16).result
            row_a_lds = arith.AddIOp(row_a_lds_base, lane_mod_16).result
            
            col_offset_base = arith.MulIOp(lane_div_16, c8).result
            
            row_b_lds_base = arith.MulIOp(wave_col, c16).result
            row_b_lds = arith.AddIOp(row_b_lds_base, lane_mod_16).result
            
            # Main Loop K
            current_acc = acc_init
            for k in range(0, 1280, 128):
                k_const = arith.ConstantOp(ir.IndexType.get(), k).result
                
                # Load A
                col_a_global_k = arith.AddIOp(k_const, col_a_local).result
                row_a_g_1024 = arith.MulIOp(row_a_global, c1024).result
                idx_a = arith.AddIOp(row_a_g_1024, col_a_global_k).result
                
                vec_a = vector.TransferReadOp(vec16_f8, arg_a, [idx_a], identity_map, pad_f8, [True]).result
                vector.StoreOp(vec_a, lds_a, [lds_write_idx])
                
                # Load B (Transposed)
                col_b_global_k = arith.AddIOp(k_const, col_b_local).result
                row_b_g_1024 = arith.MulIOp(row_b_global, c1024).result
                idx_b = arith.AddIOp(row_b_g_1024, col_b_global_k).result
                
                vec_b = vector.TransferReadOp(vec16_f8, arg_b, [idx_b], identity_map, pad_f8, [True]).result
                vector.StoreOp(vec_b, lds_b, [lds_write_idx])
                
                gpu.barrier()
                
                # Inner Loop
                acc = current_acc
                for ki in range(0, 128, 32):
                    ki_const = arith.ConstantOp(ir.IndexType.get(), ki).result
                    col_lds = arith.AddIOp(ki_const, col_offset_base).result
                    
                    # A LDS Index
                    row_a_lds_128 = arith.MulIOp(row_a_lds, c128).result
                    idx_a_mfma = arith.AddIOp(row_a_lds_128, col_lds).result
                    
                    # B LDS Index
                    row_b_lds_128 = arith.MulIOp(row_b_lds, c128).result
                    idx_b_mfma = arith.AddIOp(row_b_lds_128, col_lds).result
                    
                    vec8_f8 = ir.VectorType.get([8], f8)
                    vec8_i8 = ir.VectorType.get([8], ir.IntegerType.get_signless(8))
                    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                    
                    vec_a_load = vector.LoadOp(vec8_f8, lds_a, [idx_a_mfma]).result
                    vec_b_load = vector.LoadOp(vec8_f8, lds_b, [idx_b_mfma]).result
                    
                    a_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_a_load).result
                    b_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_b_load).result
                    
                    a_vec64 = vector.BitCastOp(vec1_i64, a_bytes).result
                    b_vec64 = vector.BitCastOp(vec1_i64, b_bytes).result
                    
                    a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                    b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                    
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        vec4_f32, [a_pack, b_pack, acc, c0_i32, c0_i32, c0_i32]
                    ).result
                    
                gpu.barrier()
                current_acc = acc

            final_acc = current_acc
            
            # Store Result
            lane_div_16 = arith.DivUIOp(lane_id, c16).result
            lane_rem_16 = arith.RemUIOp(lane_id, c16).result
            
            row_wave_base = arith.MulIOp(wave_row, c16).result
            col_wave_base = arith.MulIOp(wave_col, c16).result
            
            bx_32 = arith.MulIOp(bx._value, c32).result
            by_32 = arith.MulIOp(by._value, c32).result
            
            row_base_g = arith.AddIOp(bx_32, row_wave_base).result
            col_base_g = arith.AddIOp(by_32, col_wave_base).result
            
            for i in range(4):
                val = vector.ExtractOp(final_acc, [], [i]).result
                
                c_i = arith.ConstantOp(ir.IndexType.get(), i).result
                row_offset_base = arith.MulIOp(lane_div_16, c4).result
                row_offset = arith.AddIOp(row_offset_base, c_i).result
                
                col_offset = lane_rem_16
                
                row_g = arith.AddIOp(row_base_g, row_offset).result
                col_g = arith.AddIOp(col_base_g, col_offset).result
                
                row_g_1024 = arith.MulIOp(row_g, c1024).result
                idx = arith.AddIOp(row_g_1024, col_g).result
                
                memref.StoreOp(val, arg_c, [idx])
    
    print("✓ MLIR module constructed via @gpu.func decorator")
    
    # Set kernel attributes
    gpu_func_op = None
    for op in ctx.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            for inner_op in op.body.blocks[0].operations:
                if hasattr(inner_op, 'OPERATION_NAME') and inner_op.OPERATION_NAME == "gpu.func":
                    gpu_func_op = inner_op
                    break
    
    if gpu_func_op:
        gpu_func_op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get("256,256")
        gpu_func_op.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([256, 1, 1])
        gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    print("Compiling...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    print("Executing kernel...")
    
    # Random inputs
    a_host = np.random.randint(-16, 16, size=(M, K)).astype(np.float32)
    b_host = np.random.randint(-16, 16, size=(K, N)).astype(np.float32)
    
    # Transpose B for the kernel (NxK)
    b_host_T = np.ascontiguousarray(b_host.T)
    
    a_bytes = np.array([to_byte(x) for x in a_host.flatten()], dtype=np.uint8)
    b_bytes = np.array([to_byte(x) for x in b_host_T.flatten()], dtype=np.uint8)
    
    c_host = np.zeros(size_c, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(size_a))
    d_b = hip_check(hip.hipMalloc(size_b))
    d_c = hip_check(hip.hipMalloc(size_c * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_bytes.ctypes.data, size_a, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_bytes.ctypes.data, size_b, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c)), ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args_array = (ctypes.c_void_p * 3)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Grid: 32x32 blocks. Block: 256 threads.
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 32, 32, 1, 256, 1, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, size_c * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Verification
    print("Computing expected result with np.matmul...")
    expected_matrix = np.matmul(a_host, b_host)
    expected = expected_matrix.flatten()
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    print("="*80)
    print(f"Max Absolute Difference: {np.max(np.abs(c_host - expected))}")
    
    if np.allclose(c_host, expected, atol=1.0):
        print(f"✓ Kernel executed correctly (Matches np.matmul)")
        return True
    else:
        print(f"✗ Unexpected result")
        print(f"  Min: {np.min(c_host)}")
        print(f"  Max: {np.max(c_host)}")
        failures = np.where(np.abs(c_host - expected) > 1.0)[0]
        if len(failures) > 0:
            print(f"  First failure at index {failures[0]}: Expected {expected[failures[0]]}, Got {c_host[failures[0]]}")
            print(f"  Total failures: {len(failures)}")
        raise ValueError("Kernel result does not match expected values")
    

if __name__ == "__main__":
    test_mfma_fp8_rocir()
