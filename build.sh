#!/bin/bash

echo "Starting build process for the HLSCore"
echo "Note: Building MLIR is very resource intensive. Please ensure that you have enough RAM/swap space (at least 64GB) and disk-space (~100GB), or it might successfully link. If you have less than 128GB of RAM/swap, ensure that MLIR links with only 1 core by entering ./external_libs/llvm/build and running 'ninja -j1' (only required during linking)"

cd ./external_libs/circt/llvm/
mkdir build
cd ./build

cmake -G Ninja ../llvm   -DLLVM_ENABLE_PROJECTS="mlir"   -DLLVM_TARGETS_TO_BUILD="host"   -DLLVM_ENABLE_ASSERTIONS=ON   -DCMAKE_BUILD_TYPE=DEBUG   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   -DLLVM_ENABLE_RTTI=ON
ninja

echo "Successfully build MLIR"
echo "Building CIRCT"

cd ../../
mkdir build
cd ./build
cmake -G Ninja ..     -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir     -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm     -DLLVM_ENABLE_ASSERTIONS=ON     -DCMAKE_BUILD_TYPE=DEBUG     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON     -DBUILD_SHARED_LIBS=OFF     -DLLVM_ENABLE_RTTI=ON
ninja

echo "Successfully built CIRCT"
echo "Building HLSCore"

cd ../../../build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../external_libs/circt/llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/../external_libs/circt/llvm/build/lib/cmake/llvm -DCIRCT_DIR=$PWD/../external_libs/circt/build/lib/cmake/circt

ninja
 cd ../

echo "Successfully built HLSCore"
echo "Run: 'cd ./build/' and 'sudo ninja install' to install system wide. "
