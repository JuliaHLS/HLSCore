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
    std::string inputMLIR = R"mlir(func.func @t2(%arg0: i64, %arg1: i64) -> i64 {
      %0 = arith.cmpi slt, %arg0, %arg1 : i64
      cf.cond_br %0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %1 = arith.addi %arg0, %arg1 : i64
      cf.br ^bb4(%1 : i64)
    ^bb2:  // pred: ^bb0
      %2 = arith.cmpi slt, %arg1, %arg0 : i64
      %c0_i64 = arith.constant 0 : i64
      cf.cond_br %2, ^bb3, ^bb4(%c0_i64 : i64)
    ^bb3:  // pred: ^bb2
      %3 = arith.addi %arg0, %arg1 : i64
      cf.br ^bb4(%3 : i64)
    ^bb4(%4: i64):  // 3 preds: ^bb1, ^bb2, ^bb3
      return %4 : i64
    })mlir";
    
    HLSTool hls;
    std::unique_ptr<Options> opt = std::make_unique<Options>(inputMLIR, "-");
    hls.setOptions(std::move(opt));
    auto result = hls.synthesise();
}
