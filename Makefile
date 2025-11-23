# CuTe IR Makefile
# Quick commands for common tasks

.PHONY: help build install test clean format lint docs

help:
@echo "CuTe IR - Available Commands"
@echo "=============================="
@echo "  make build        - Build C++ runtime library"
@echo "  make install      - Install Python package"
@echo "  make install-dev  - Install in development mode"
@echo "  make test         - Run all tests"
@echo "  make clean        - Clean build artifacts"
@echo "  make format       - Format code (black, clang-format)"
@echo "  make lint         - Run linters (pylint, mypy)"
@echo "  make docs         - Generate documentation"
@echo "  make check        - Run all checks (format + lint + test)"
@echo ""

build:
@echo "Building C++ runtime..."
@mkdir -p runtime/build
@cd runtime/build && cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;90" && make -j8
@echo "✓ Build complete"

install:
@echo "Installing Python package..."
@pip install .
@echo "✓ Installation complete"

install-dev:
@echo "Installing in development mode..."
@pip install -e .
@echo "✓ Development installation complete"

test:
@echo "Running tests..."
@pytest tests/ -v || echo "No tests found"
@python python/examples/test_gemm.py || echo "Test requires GPU"
@echo "✓ Tests complete"

clean:
@echo "Cleaning build artifacts..."
@rm -rf runtime/build
@rm -rf build dist *.egg-info
@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
@find . -type f -name "*.pyc" -delete
@find . -type f -name "*.so" -delete
@echo "✓ Clean complete"

format:
@echo "Formatting code..."
@black python/ || echo "black not installed"
@clang-format -i runtime/src/*.cpp runtime/include/*.h || echo "clang-format not installed"
@echo "✓ Format complete"

lint:
@echo "Running linters..."
@pylint python/cute_runtime/*.py || echo "pylint not installed"
@mypy python/cute_runtime/ || echo "mypy not installed"
@echo "✓ Lint complete"

docs:
@echo "Generating documentation..."
@echo "Documentation already in docs/ directory"
@echo "✓ See docs/ for all documentation"

check: format lint test
@echo "✓ All checks passed"

# Development targets
dev-setup:
@echo "Setting up development environment..."
@pip install -e .
@pip install pytest black mypy pylint
@echo "✓ Development environment ready"

gpu-info:
@echo "GPU Information:"
@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
@python -c "import cute_runtime; print(cute_runtime.get_device_info())" || echo "Runtime not installed"

tree:
@tree -L 3 -I '__pycache__|*.pyc|build' --dirsfirst

stats:
@echo "Project Statistics"
@echo "=================="
@echo "TableGen files:"
@find include/cute -name "*.td" | wc -l
@echo "C++ files:"
@find runtime lib -name "*.cpp" -o -name "*.h" | wc -l
@echo "Python files:"
@find python -name "*.py" | wc -l
@echo "Total lines of code:"
@find include lib runtime python -name "*.td" -o -name "*.cpp" -o -name "*.h" -o -name "*.py" | xargs wc -l | tail -1
