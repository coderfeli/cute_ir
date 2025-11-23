# Contributing to CuTe IR

Thank you for your interest in contributing to CuTe IR!

## 🎯 Areas for Contribution

### High Priority

1. **Pass Implementations** (16 remaining)
   - Located in `lib/Transforms/`
   - See `include/cute/CutePasses.td` for pass definitions
   - Currently implemented: `cute-to-standard`, `cute-nvgpu-to-nvgpu`
   - Needed: canonicalize, fusion, vectorization, etc.

2. **Operation Lowering Patterns**
   - Add more conversion patterns in existing passes
   - Improve coverage of CuTe operations

3. **Tests**
   - MLIR FileCheck tests in `tests/`
   - Python unit tests for runtime
   - End-to-end integration tests

### Medium Priority

4. **Examples**
   - More MLIR kernel examples
   - Benchmark implementations (GEMM, Conv, etc.)
   - Tutorial notebooks

5. **Documentation**
   - API reference
   - Tutorial guides
   - Architecture deep-dives

6. **Build System**
   - CI/CD pipeline
   - Cross-platform support
   - Package distribution

### Future Enhancements

7. **New Features**
   - INT8/BF16 support
   - Multi-GPU kernels
   - Auto-tuning framework
   - Kernel cache

## 📋 Development Workflow

### Setup Development Environment

```bash
# Clone and setup
cd cute_ir_tablegen/

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black mypy pylint
```

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   # Run Python tests
   pytest tests/

   # Format code
   black python/

   # Type check
   mypy python/cute_runtime/
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

## 📝 Coding Standards

### C++ Code

- Follow LLVM coding standards
- Use C++17 features
- Add Doxygen comments for public APIs
- Format with clang-format

### Python Code

- Follow PEP 8
- Use type hints
- Format with black
- Add docstrings (Google style)

### MLIR/TableGen

- Follow MLIR conventions
- Add operation documentation
- Include examples in comments

## 🧪 Testing

### MLIR Tests

```mlir
// RUN: mlir-opt %s -pass-pipeline='...' | FileCheck %s

// Your test case
// CHECK: expected output
```

### Python Tests

```python
import pytest
import cute_runtime as cute

def test_gemm():
    gemm = cute.Gemm(64, 64, 64)
    # ... test code ...
    assert result.shape == (64, 64)
```

## 📬 Submitting Changes

1. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Describe changes clearly
   - Reference related issues
   - Include test results

3. **Code Review**
   - Address reviewer comments
   - Update as needed

## 🐛 Reporting Bugs

When reporting bugs, include:
- OS and CUDA version
- Python version
- Complete error message
- Minimal reproduction code
- Expected vs actual behavior

## 💡 Feature Requests

For feature requests, describe:
- Use case
- Proposed API/interface
- Implementation ideas (optional)
- Benefits to the project

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Questions?

- Open an issue for discussion
- Check existing documentation
- Review similar implementations

Thank you for contributing! 🎉
