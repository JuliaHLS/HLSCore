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
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"

#include "IRLevel.hpp"
#include "HLS.hpp"
#include "HLSDynamic.hpp"
#include "Options.hpp"

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
