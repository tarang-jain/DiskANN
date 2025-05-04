#!/bin/bash

# NOTE: This file is temporary for the proof-of-concept branch and will be removed before this PR is merged

BUILD_TYPE=Release
BUILD_DIR=build/

EXTRA_CMAKE_ARGS=""
set -e

if [ "$1" == "clean" ]; then
  rm -rf build
  rm -rf .cache
  exit 0
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
 -DCMAKE_C_COMPILER_LAUNCHER=ccache \
 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
 ${EXTRA_CMAKE_ARGS} \
 ../

make -j30