#include "HLS.hpp"
#include <llvm/ADT/StringRef.h>
#include <string>
#include "logging.hpp"

namespace HLSCore {

/// Create a simple canonicalizer pass.
std::unique_ptr<Pass> HLSTool::createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  return mlir::createCanonicalizerPass(config);
}


/// Process a single buffer of the input.
LogicalResult HLSTool::processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr, const std::string& outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

    // Parse the input
    mlir::OwningOpRef<mlir::ModuleOp> module;
    llvm::sys::TimePoint<> parseStartTime;

    auto parserTimer = ts.nest("MLIR Parser");
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);

    if (!module)
    return failure();

    // apply pass manager command line options.
    PassManager pm(&context);
    pm.enableVerifier(opt->verifyPasses);
    pm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pm)))
    return failure();

    if (failed(runHLSFlow(pm, module.get(), outputFilename, outputFile)))
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
LogicalResult HLSTool::processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer, const std::string& outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

    HLSCore::logging::runtime_log<std::string>("Processing InputSplit");

    if (!buffer) HLSCore::logging::runtime_log<std::string>("Error: Buffer empty");

    llvm::SourceMgr sourceMgr;
    HLSCore::logging::runtime_log<std::string>("Adding new source buffer");
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    HLSCore::logging::runtime_log<std::string>("Successfully added new source buffer");

    if (!opt->verifyDiagnostics) {
        SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
        return processBuffer(context, ts, sourceMgr, outputFilename, outputFile);
    }

    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    context.printOpOnDiagnostic(false);

    HLSCore::logging::runtime_log<std::string>("Processing buffer");
    (void)processBuffer(context, ts, sourceMgr, outputFilename, outputFile);
    HLSCore::logging::runtime_log<std::string>("Processed buffer");
    return sourceMgrHandler.verify();
}

/// process the input provided by the user, splitting it up if the
/// corresponding option was specified.
LogicalResult HLSTool::processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input, const std::string& outputFilename,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

    if (!opt->splitInputFile) {
        HLSCore::logging::runtime_log("Processing InputSplit");
        return processInputSplit(context, ts, std::move(input), outputFilename, outputFile);
    }


    HLSCore::logging::runtime_log("Processing splitAndProcessBuffer");
    return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFilename, outputFile);
      },
      llvm::outs());
}

bool HLSTool::synthesise() {
    HLSCore::logging::runtime_log("Starting Synthesis");

    MLIRContext context(registry);
    // Create the timing manager we use to sample execution times.
    DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    auto ts = tm.getRootScope();


    HLSCore::logging::runtime_log("Parsing Input");
    // Parse the input into a memBuffer
    auto input = opt->getInputBuffer();

    HLSCore::logging::runtime_log<std::string>("Processing Buffer: ");
    HLSCore::logging::runtime_log<llvm::StringRef>(input->getBuffer());

    // process the input
    std::string errorMessage;
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    
    // create output file, if not already handled by CIRCT 
    if(opt->outputFormat != OutputSplitVerilog) {
        outputFile.emplace(mlir::openOutputFile(opt->getOutputFilename(), &errorMessage));

        // error handling
        if(!*outputFile) {
            llvm::errs() << "[HLSCore ERROR] :" << errorMessage << "\n";
            return false;
        }
    }

    // create outputFile

    HLSCore::logging::runtime_log("Processing input MLIR");
    if (failed(processInput(context, ts, std::move(input), opt->getOutputFilename(), outputFile)))
        return false;

    // close output file (clean-up)
    if (outputFile.has_value())
        (*outputFile)->keep();


    HLSCore::logging::runtime_log("Emitted Output");

    return true; 
}

LogicalResult HLSTool::writeSingleFileOutput(const mlir::ModuleOp& module, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>>& outputFile) {

    // if output files are not written via CIRCT passes
    if ((opt->outputFormat != OutputSplitVerilog && opt->outputFormat != HLSCore::OutputVerilog) || opt->irOutputLevel != SV) {
        HLSCore::logging::runtime_log<std::string>("Setting detected, HLSTool configured to write to a single file (including terminal)");

        // write to an output file if specified
        HLSCore::logging::runtime_log<std::string>("Writing to single file.");

        HLSCore::logging::runtime_log<std::string>("Writing output...");
        // print the output
        module->print((*outputFile)->os());
    }


    // return success
    return llvm::LogicalResult::success();
}


HLSTool::HLSTool() {
  // register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // register MLIR dialects.
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();

  /* registerAllDialects(registry); */
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);

  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
}

void HLSTool::setOptions(std::unique_ptr<Options>&& _opt) {
    opt = std::move(_opt);
}

bool HLSTool::targetAbstractionLayer(IRLevel currentLevel) {
    return currentLevel <= opt->irOutputLevel && currentLevel >= opt->irInputLevel; 
}


}

