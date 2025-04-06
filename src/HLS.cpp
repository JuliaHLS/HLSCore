#include "HLS.hpp"
#include <llvm/ADT/StringRef.h>
#include <string>
#include "logging.hpp"

namespace HLSCore {

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
    pm.enableVerifier(verifyPasses);
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

    if (!buffer) HLSCore::logging::runtime_log<std::string>("Error: Buffer empty");

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

    if (!verifyDiagnostics) {
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

    if (!splitInputFile) {
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
    if(outputFormat != OutputSplitVerilog) {
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

  // Register MLIR passes.
  mlir::tosa::registerTosaToLinalgPipelines();
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::bufferization::registerOneShotBufferizePass();

  // Register CIRCT dialects.
  registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  sv::SVDialect, handshake::HandshakeDialect, esi::ESIDialect,
                  calyx::CalyxDialect>();

}

void HLSTool::setOptions(std::unique_ptr<Options>&& _opt) {
    opt = std::move(_opt);
}

}
