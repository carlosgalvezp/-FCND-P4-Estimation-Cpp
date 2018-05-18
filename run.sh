#!/bin/bash

REPO_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="${REPO_ROOT}/build"

# Build code
"${REPO_ROOT}/build.sh"

# Run simulator
# https://github.com/NVIDIA/nvidia-docker/issues/136#issuecomment-232755805
xhost +local:root
nvidia-docker run --rm \
                  --interactive \
                  --tty \
                  --env="DISPLAY" \
                  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                  --volume="${REPO_ROOT}":"${REPO_ROOT}" \
                  --workdir="${BUILD_DIR}" \
                  --user=$(id -u):$(id -g) \
                  carlosgalvezp/fcnd_p3_cpp \
                  ./CPPSim
xhost -local:root
