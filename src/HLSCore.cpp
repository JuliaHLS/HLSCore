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
#include "HLSDynamic.hpp"


using namespace llvm;
using namespace mlir;
using namespace circt;



// --------------------------------------------------------------------------
// Tool driver code
// --------------------------------------------------------------------------
/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Parse the input.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::sys::TimePoint<> parseStartTime;

  auto parserTimer = ts.nest("MLIR Parser");
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module)
    return failure();

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

    if (failed(doHLSFlowDynamic(pm, module.get(), outputFile)))
      return failure();

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether
/// the user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeHlstool(MLIRContext &context) {

  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  if (outputFormat != OutputSplitVerilog) {
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!*outputFile) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  }

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
}

/// Main driver for hlstool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeHlstool'.  This is set
/// up so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);


  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "CIRCT HLS tool\n");

  DialectRegistry registry;
  // Register MLIR dialects.
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Register MLIR passes.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  // Register CIRCT dialects.
  registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  sv::SVDialect, handshake::HandshakeDialect, esi::ESIDialect,
                  calyx::CalyxDialect>();

  // Do the guts of the hlstool process.
  MLIRContext context(registry);
  auto result = executeHlstool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
