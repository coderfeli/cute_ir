#!/usr/bin/env python3
import sys
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func
from rocdsl.dialects.ext import arith, scf

print('==> 演示 mlir-python-extras 风格的操作符重载')
print()

ctx = Context()
ctx.load_all_available_dialects()

with ctx, Location.unknown(ctx):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(name='demo')
        def demo_func():
            # 使用 operator overloading 的方式
            a = arith.index(10)
            b = arith.index(20)
            
            # 使用运算符而不是显式的 Op
            c = a + b              # 相当于 AddIOp
            d = c * arith.index(2) # 相当于 MulIOp
            e = d - arith.index(5) # 相当于 SubIOp
            f = e // arith.index(3) # 相当于 DivSIOp
            
            print('✓ 使用 + - * // 运算符')
            
            # 比较运算
            cmp = a < b  # 相当于 CmpIOp
            print('✓ 使用 < > == != 比较运算符')
            
            return f._value

print('生成的 MLIR IR:')
print(module)
print()
print('✅ 成功演示 operator overloading!')
