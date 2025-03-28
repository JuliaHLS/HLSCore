cmake_minimum_required(VERSION 3.13)
project(HLSCore LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR")

# Find CIRCT
find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Found CIRCT")

# Set LLVM link components
set(LLVM_LINK_COMPONENTS Support)

# Include directories
include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CIRCT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define the executable
add_executable(HLSCore 
    src/Options.cpp
    src/HLSDynamic.cpp
    src/HLS.cpp
    src/HLSCore.cpp
    src/WriteSingleFileOutput.cpp
    src/OutputMemrefPassByRef.cpp
    src/CirctFriendlyLoops.cpp
)

add_library(HLSCore_C_API SHARED
    src/Options.cpp
    src/HLSDynamic.cpp
    src/HLS.cpp
    src/HLSCore.cpp
    src/WriteSingleFileOutput.cpp
    src/HLS_C_API.cpp
    src/OutputMemrefPassByRef.cpp
    src/CirctFriendlyLoops.cpp
)

# Update compile flags for LLVM
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

set(CIRCT_LIBS 
        ${llvm_libs}
        CIRCTDC
        CIRCTDCTransforms
        CIRCTDCToHW
        CIRCTESI
        CIRCTExportChiselInterface
        CIRCTExportVerilog
        CIRCTHandshake
        CIRCTHandshakeToDC
        CIRCTHandshakeToHW
        CIRCTHandshakeTransforms
        CIRCTHW
        CIRCTHWTransforms
        CIRCTSeq
        CIRCTSeqToSV
        CIRCTSeqTransforms
        CIRCTCFToHandshake
        CIRCTSV
        CIRCTSVTransforms
        CIRCTSCFToCalyx
        CIRCTCalyx
        CIRCTCalyxNative
        CIRCTCalyxTransforms
        CIRCTCalyxToHW
        CIRCTCalyxToFSM
        CIRCTFSM
        CIRCTFSMTransforms
        CIRCTFSMToSV
        CIRCTTransforms
        MLIRIR
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIROptLib
        MLIRParser
        MLIRControlFlowDialect
        MLIRControlFlowTransforms
        MLIRTosaDialect
        MLIRBufferizationDialect
        MLIRTensorDialect
        MLIRSupport
        MLIRTransforms
        MLIRSCFToControlFlow
        MLIRTosaToLinalg
        MLIRLinalgDialect
        MLIRLinalgTransforms
)

# Link MLIR and CIRCT dependencies
target_link_libraries(HLSCore
    PRIVATE
    ${CIRCT_LIBS}
    LLVMSupport
)

target_link_libraries(HLSCore_C_API
    PRIVATE
    ${CIRCT_LIBS}
    LLVMSupport
)

# Include MLIR and LLVM headers
target_include_directories(HLSCore PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CIRCT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(HLSCore_C_API PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CIRCT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

install (
    FILES include/HLS_C_API.hpp
    DESTINATION include
)

install (TARGETS HLSCore_C_API HLSCore
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
