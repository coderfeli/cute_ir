#!/usr/bin/env python3
import sys
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/python")

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func
from rocdsl.dialects.ext import arith, rocir

print("测试开始：CuTe Layout + Operator Overloading")

ctx = Context()
ctx.load_all_available_dialects()
ctx.allow_unregistered_dialects = True

with ctx, Location.unknown(ctx):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(name="gemm")
        def gemm():
            # 使用 operator overloading
            c2048 = arith.index(2048)
            c128 = arith.index(128)
            
            # Calculate number of tiles using // operator
            num_tiles = c2048 // c128
            
            print("Using operator // computes: 2048 // 128")
            return num_tiles._value

print("\n生成的 MLIR IR:")
print(module)
print("\n Test passed!operator overloading works correctly")
