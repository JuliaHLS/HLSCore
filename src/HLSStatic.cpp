#include "HLSStatic.hpp"
#include <mlir/Transforms/ViewOpGraph.h>

namespace HLSCore {


LogicalResult HLSToolStatic::runHLSFlow(
    PassManager &pm, ModuleOp module, const std::string &outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

    HLSCore::logging::runtime_log<std::string>("Running Statically Scheduled HLS Flow");

    // bool suppressLaterPasses = false;
    // auto notSuppressed = [&]() { return !suppressLaterPasses; };
    // auto addIfNeeded = [&](llvm::function_ref<bool()> predicate,
    //                      llvm::function_ref<void()> passAdder) {
    // if (predicate())
      // passAdder();
    // };

    auto addIRLevel = [&](HLSCore::IRLevel level,
                        llvm::function_ref<void()> passAdder) {
    // addIfNeeded(notSuppressed, [&]() {
      if (targetAbstractionLayer(level))
        passAdder();
    // });
    };

    // Resolve blocks with multiple predescessors

    HLSCore::logging::runtime_log<std::string>("Building passes");

    // Software lowering
    addIRLevel(PreCompile, [&]() {
        HLSCore::pipelines::TosaToAffinePipeline(pm);

        // pm.addPass(circt::createAffineToLoopSchedule());

        // lower affine to cf
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());

        // // allow merge multiple basic block sources
        pm.addPass(circt::createInsertMergeBlocksPass());
        //

        // log
        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to Precompile");
    });

    addIRLevel(Core, [&]() {
        // pm.addPass(mlir::createPrintOpGraphPass());
        pm.addPass(circt::createSCFToCalyxPass());
        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to Core");
    });

    addIRLevel(PostCompile, [&]() {
        pm.addPass(createSimpleCanonicalizerPass());

        // eliminate Calyx's comb group abstraction
        pm.addNestedPass<calyx::ComponentOp>(circt::calyx::createRemoveCombGroupsPass());
        pm.addPass(createSimpleCanonicalizerPass());

        // compile to FSM
        pm.addNestedPass<calyx::ComponentOp>(circt::createCalyxToFSMPass());
        pm.addPass(createSimpleCanonicalizerPass());
        pm.addNestedPass<calyx::ComponentOp>(circt::createMaterializeCalyxToFSMPass());
        pm.addPass(createSimpleCanonicalizerPass());

        // eliminate calyx's group abstraction
        pm.addNestedPass<calyx::ComponentOp>(circt::createRemoveGroupsFromFSMPass());
        pm.addPass(createSimpleCanonicalizerPass());

        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to PostCompile");
    });

    // HW path.

    addIRLevel(RTL, [&]() {
        // Compile to HW
        pm.addPass(circt::createCalyxToHWPass());
        pm.addPass(createSimpleCanonicalizerPass());

        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to RTL");
    });

    addIRLevel(SV, [&]() {
        pm.addPass(circt::createConvertFSMToSVPass());
        pm.addPass(circt::createLowerSeqToSVPass());

        // handle output
        if (opt->traceIVerilog)
            pm.addPass(circt::sv::createSVTraceIVerilogPass());

        if (opt->outputFormat == OutputVerilog) {
            pm.addPass(createExportVerilogPass((*outputFile)->os()));
        } else if (opt->outputFormat == OutputSplitVerilog) {
            pm.addPass(createExportSplitVerilogPass(outputFilename));
        }

        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to SV");
    });


    HLSCore::logging::runtime_log<std::string>("Trying to start MLIR Lowering Process");

    if (failed(pm.run(module))) {
        HLSCore::logging::runtime_log<std::string>("Failed to synthesise the program...");
        throw std::runtime_error("ERROR: Failed to synthesise");
    }

    HLSCore::logging::runtime_log<std::string>("Successfully lowered MLIR");

    return writeSingleFileOutput(module, outputFilename, outputFile);
}

} // namespace HLSCore
