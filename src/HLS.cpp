#include "HLS.hpp"
#include <llvm/ADT/StringRef.h>
#include <string>
#include "logging.hpp"

namespace HLSCore {

// --------------------------------------------------------------------------
// Tool driver code
// --------------------------------------------------------------------------
/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr, const std::string& outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

    // Parse the input
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

    if (failed(doHLSFlowDynamic(pm, module.get(), outputFilename, outputFile)))
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

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
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

static LogicalResult executeHlstool(MLIRContext &context, const std::string& inputMLIR) {
    return success();
}





HLSTool::HLSTool() {
  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

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

}

void HLSTool::setOptions(std::unique_ptr<Options>&& _opt) {
    opt = std::move(_opt);
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

    // Process the input.
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;

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

}
