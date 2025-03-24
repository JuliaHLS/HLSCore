//===- hlstool.cpp - The hlstool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'hlstool', which composes together a variety of
// CIRCT libraries that can be used to realise HLS (High Level Synthesis)
// flows.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "IRLevel.hpp"
#include "HLS.hpp"
#include "HLSDynamic.hpp"
#include "Options.hpp"

using namespace HLSCore;

/// Main driver for hlstool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeHlstool'.  This is set
/// up so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
    // Input MLIR string
    std::string inputMLIR = R"mlir(func.func @add(%arg0: tensor<5xi64>, %arg1: tensor<5xi64>) -> tensor<5xi64> {
          %0 = "tosa.add"(%arg0, %arg1) : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
          return %0 : tensor<5xi64>
        })mlir";
    /* std::string inputMLIR = R"mlir( */
    /* #map = affine_map<(d0) -> (d0)> */
    /* module { */
    /*   func.func @add(%arg0: memref<5xi64>, %arg1: memref<5xi64>) -> memref<5xi64> { */
    /*     %alloc = memref.alloc() : memref<5xi64> */
    /*     linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<5xi64>, memref<5xi64>) outs(%alloc : memref<5xi64>) { */
    /*     ^bb0(%in: i64, %in_0: i64, %out: i64): */
    /*       %0 = arith.addi %in, %in_0 : i64 */
    /*       linalg.yield %0 : i64 */
    /*     } */
    /*     return %alloc : memref<5xi64> */
    /*   } */
    /* })mlir"; */

    /* std::string inputMLIR = R"mlir( */
    /* module { */
    /*   func.func @add(%arg0: memref<5xi64>, %arg1: memref<5xi64>, %out: memref<5xi64>) -> i1 { */
    /*     %c0 = arith.constant 0 : index */
    /*     %c5 = arith.constant 5 : index */
    /*     %c1 = arith.constant 1 : index */
    /*     cf.br ^bb1(%c0 : index) */
    /*   ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2 */
    /*     %1 = arith.cmpi slt, %0, %c5 : index */
    /*     cf.cond_br %1, ^bb2, ^bb3 */
    /*   ^bb2:  // pred: ^bb1 */
    /*     %2 = memref.load %arg0[%0] : memref<5xi64> */
    /*     %3 = memref.load %arg1[%0] : memref<5xi64> */
    /*     %4 = arith.addi %2, %3 : i64 */
    /*     memref.store %4, %out[%0] : memref<5xi64> */
    /*     %5 = arith.addi %0, %c1 : index */
    /*     cf.br ^bb1(%5 : index) */
    /*   ^bb3:  // pred: ^bb1 */
    /*     %true = arith.constant true */
    /*     return %true : i1 */
    /*   } */
    /* })mlir"; */
    /* std::string inputMLIR = R"mlir(func.func @add(%arg0: i64, %arg1: i64) -> i64 { */
    /*     %0 = arith.addi %arg0, %arg1 : i64 */
    /*     return %0 : i64 */
    /* })mlir"; */
 

    HLSTool hls;
    std::unique_ptr<Options> opt = std::make_unique<Options>(inputMLIR, "-");
    hls.setOptions(std::move(opt));
    auto result = hls.synthesise();
}
