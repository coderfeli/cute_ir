#!/bin/bash
# Enter interactive shell in felixatt Docker container

CONTAINER_NAME="felixatt"
PROJECT_DIR="/mnt/raid0/felix/cute_ir_tablegen"

echo "Entering ${CONTAINER_NAME} container..."
echo "Project directory: ${PROJECT_DIR}"

docker exec -it ${CONTAINER_NAME} bash -c "cd ${PROJECT_DIR} && exec bash"
