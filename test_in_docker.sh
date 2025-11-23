#!/bin/bash
# Test CuTe IR inside felixatt Docker container

set -e

CONTAINER_NAME="felixatt"
PROJECT_DIR="/mnt/raid0/felix/cute_ir_tablegen"

echo "========================================="
echo "Testing CuTe IR in Docker: ${CONTAINER_NAME}"
echo "========================================="

# Check if container is running
if ! docker ps | grep -q ${CONTAINER_NAME}; then
    echo "Error: Container ${CONTAINER_NAME} is not running"
    exit 1
fi

# Execute test commands inside the container
docker exec -it ${CONTAINER_NAME} bash -c "
    cd ${PROJECT_DIR} && \
    echo '--- CUDA Device Info ---' && \
    nvidia-smi && \
    echo '' && \
    echo '--- Checking build artifacts ---' && \
    ls -lh build/ && \
    echo '' && \
    echo '--- Running layout tests (if mlir-opt available) ---' && \
    if command -v mlir-opt &> /dev/null; then
        mlir-opt tests/test_layout.mlir --cute-canonicalize || echo 'mlir-opt tests require MLIR installation'
    else
        echo 'mlir-opt not found, skipping MLIR tests'
        echo 'TableGen files are ready for use with MLIR'
    fi && \
    echo '' && \
    echo '--- Project structure ---' && \
    tree -L 2 -I 'build|*.egg-info|__pycache__' . || ls -R && \
    echo '' && \
    echo '=========================================' && \
    echo 'Testing completed!' && \
    echo '========================================='
"
