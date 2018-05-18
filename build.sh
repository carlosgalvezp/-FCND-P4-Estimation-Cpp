#!/bin/bash

REPO_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="${REPO_ROOT}/build"

# Build code
mkdir -p "${BUILD_DIR}"

docker run --rm -it --volume="${REPO_ROOT}":"${REPO_ROOT}" --workdir="${BUILD_DIR}" \
           --user=$(id -u):$(id -g) carlosgalvezp/fcnd_p3_cpp cmake ..
docker run --rm -it --volume="${REPO_ROOT}":"${REPO_ROOT}" --workdir="${BUILD_DIR}" \
           --user=$(id -u):$(id -g) carlosgalvezp/fcnd_p3_cpp make -j8
