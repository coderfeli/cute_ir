# RocDSL Python Testing Guide

## Running Tests in Docker

All tests must run inside the `felixatt` Docker container with proper PYTHONPATH:

```bash
docker exec -it felixatt bash -c "export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core && cd /mnt/raid0/felix/cute_ir_tablegen && python3 -m pytest python/tests/test_passes.py -v"
```

## Test Suites

### 1. Pass Management Tests ✅
**File**: `python/tests/test_passes.py`  
**Status**: 16/16 passing

Tests the complete pass management API:
- Pipeline construction and fluent API
- Pass options and parameters
- Pipeline composition (+=, +)
- Nested pipelines (Func, Gpu)
- Pipeline recipes
- All CuTe passes
- Complex multi-stage pipelines

```bash
# Run all pass tests
docker exec -it felixatt bash -c "export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core && cd /mnt/raid0/felix/cute_ir_tablegen && python3 -m pytest python/tests/test_passes.py -v"
```

### 2. Operator Overloading Tests ⚠️
**File**: `python/tests/test_arith_operators.py`  
**Status**: 2/5 passing (needs minor fixes)

Tests Pythonic operator syntax:
- Arithmetic operators: +, -, *, //, %
- Comparison operators: <, >, <=, >=, ==, !=
- Mixed operations with Python literals
- Chained operations

```bash
# Run operator tests
docker exec -it felixatt bash -c "export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core && cd /mnt/raid0/felix/cute_ir_tablegen && python3 -m pytest python/tests/test_arith_operators.py -v"
```

### 3. CuTe Dialect Tests ❌
**Files**: `test_basic_ops.py`, `test_product_divide.py`, etc.  
**Status**: Require compiled CuTe dialect

These tests need the CuTe dialect to be compiled and available:

```bash
# Build CuTe dialect first
cd /mnt/raid0/felix/cute_ir_tablegen/build
make -j8
```

## Quick Import Test

```bash
docker exec -it felixatt bash -c "export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core && cd /mnt/raid0/felix/cute_ir_tablegen && python3 -c 'import sys; sys.path.insert(0, \"python\"); from rocdsl import Pipeline, run_pipeline, cute, arith, scf; print(\"✅ All imports successful\")'"
```

## Environment Setup

### Required Environment Variables

```bash
export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core
```

This points to the MLIR Python bindings built from llvm-project.

### Python Path Setup

When using the package:
```python
import sys
sys.path.insert(0, '/mnt/raid0/felix/cute_ir_tablegen/python')
from rocdsl import Pipeline, arith, scf
```

## Test Results Summary

| Test Suite | Status | Passing | Total | Notes |
|------------|--------|---------|-------|-------|
| Pass Management | ✅ | 16 | 16 | All working |
| Operator Overload | ⚠️ | 2 | 5 | Minor predicate fixes needed |
| CuTe Basic Ops | ❌ | 0 | 7 | Needs compiled dialect |
| Pythonic Examples | ⚠️ | 0 | 4 | Needs dialect + fixes |

## Known Issues

1. **Operator tests**: Need to fix comparison predicate names (uppercase vs lowercase)
2. **CuTe tests**: Require `mlir.dialects.cute` to be built and importable
3. **ArithValue unwrapping**: Tests need to call `.value` when returning to functions

## Development Workflow

```bash
# 1. Enter docker container
docker exec -it felixatt bash

# 2. Set environment
export PYTHONPATH=/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core
cd /mnt/raid0/felix/cute_ir_tablegen

# 3. Run tests
python3 -m pytest python/tests/test_passes.py -v

# 4. Test imports
python3 -c 'import sys; sys.path.insert(0, "python"); from rocdsl import Pipeline; print(Pipeline().cute_to_standard())'
```
