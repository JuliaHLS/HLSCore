#pragma once

#include "CompileRepeat.hpp"

#include "Options.hpp"
#include <memory>


#include <iostream>
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

#include "IRLevel.hpp"
#include "LoweringPipelines.hpp"

using namespace llvm;
using namespace mlir;

namespace HLSCore {

class HLSTool {
public:
    HLSTool();

    void setOptions(std::unique_ptr<Options>&& _opt);
    bool synthesise();

protected:
    std::unique_ptr<Options> opt;
    DialectRegistry registry;

    LogicalResult processBuffer(MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile);

    LogicalResult processInputSplit(MLIRContext &context, TimingScope &ts, std::unique_ptr<llvm::MemoryBuffer> buffer, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile);

    LogicalResult processInput(MLIRContext &context, TimingScope &ts, std::unique_ptr<llvm::MemoryBuffer> input, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile);

    // this is where you initialise the HLS flow.
    [[nodiscard]] virtual LogicalResult runHLSFlow(PassManager &pm, ModuleOp module, const std::string &outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) = 0;


    [[nodiscard]] bool targetAbstractionLayer(IRLevel currentLevel);

    [[nodiscard]] LogicalResult writeSingleFileOutput(const mlir::ModuleOp& module, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>>& outputFile);
    
    std::unique_ptr<Pass> createSimpleCanonicalizerPass();
};


}
