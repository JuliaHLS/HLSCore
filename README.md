# HLSCore


# Build Instructions
Choose one of the two methods to install CIRCT. Note: static linking is recommended for portability, but dynamic linking is recommended for development.

If you plan to use the dynamically linked tool with Hardware.jl, you need to run `sudo ninja install` on both the llvm and circt directory.

## Building the dependencies

### Statically Linked
#### Step 1: Build LLVM and MLIR
```
cd ./external_libs/circt/llvm/
mkdir build
cd ./build
cmake -G Ninja ../llvm   -DLLVM_ENABLE_PROJECTS="mlir"   -DLLVM_TARGETS_TO_BUILD="host"   -DLLVM_ENABLE_ASSERTIONS=ON   -DCMAKE_BUILD_TYPE=DEBUG   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   -DLLVM_ENABLE_RTTI=ON
ninja
```

#### Step 2: Build CIRCT

```
cd ./external_libs/circt/
mkdir build
cd ./build
cmake -G Ninja ..     -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir     -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm     -DLLVM_ENABLE_ASSERTIONS=ON     -DCMAKE_BUILD_TYPE=DEBUG     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLVM_ENABLE_RTTI=ON
ninja
```



### Dynamically Linked
#### Step 1: Build LLVM and MLIR
```
cd ./external_libs/circt/llvm/
mkdir build
cd ./build
cmake -G Ninja ../llvm   -DLLVM_ENABLE_PROJECTS="mlir"   -DLLVM_TARGETS_TO_BUILD="host"   -DLLVM_ENABLE_ASSERTIONS=ON   -DCMAKE_BUILD_TYPE=DEBUG   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_ENABLE_RTTI=ON
ninja
```

#### Step 2: Build CIRCT

```
cd ./external_libs/circt/
mkdir build
cd ./build
cmake -G Ninja ..     -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir     -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm     -DLLVM_ENABLE_ASSERTIONS=ON     -DCMAKE_BUILD_TYPE=DEBUG     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON     -DBUILD_SHARED_LIBS=ON     -DLLVM_ENABLE_RTTI=ON
ninja
```



## Build HLSCore
First, ensure you are in the root directory of the project.

```
mkdir build
cd ./build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../external_libs/circt/llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/../external_libs/circt/llvm/build/lib/cmake/llvm -DCIRCT_DIR=$PWD/../external_libs/circt/build/lib/cmake/circt
ninja
```

If you would like to install HLSCore, run `sudo ninja install`
