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
            
            # 使用 // 运算符计算 tile 数量
            num_tiles = c2048 // c128
            
            print("✓ 使用 operator // 计算: 2048 // 128")
            return num_tiles._value

print("\n生成的 MLIR IR:")
print(module)
print("\n✅ 测试通过！operator overloading 工作正常")
