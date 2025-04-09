#include "HLSStatic.hpp"
#include <circt/Conversion/CalyxToFSM.h>
#include <circt/Conversion/FSMToSV.h>
#include <circt/Conversion/SCFToCalyx.h>
#include <circt/Conversion/SeqToSV.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>

// PASSES:

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h.inc"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace HLSCore {


LogicalResult HLSToolStatic::runHLSFlow(
    PassManager &pm, ModuleOp module, const std::string &outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

    HLSCore::logging::runtime_log<std::string>("Running Statically Scheduled HLS Flow");

    bool suppressLaterPasses = false;
    auto notSuppressed = [&]() { return !suppressLaterPasses; };
    auto addIfNeeded = [&](llvm::function_ref<bool()> predicate,
                         llvm::function_ref<void()> passAdder) {
    if (predicate())
      passAdder();
    };

    auto addIRLevel = [&](HLSCore::IRLevel level,
                        llvm::function_ref<void()> passAdder) {
    addIfNeeded(notSuppressed, [&]() {
      if (targetAbstractionLayer(level))
        passAdder();
    });
    };

    // Resolve blocks with multiple predescessors

    HLSCore::logging::runtime_log<std::string>("Building passes");

    // Software lowering
    addIRLevel(PreCompile, [&]() {
        // // lower tosa to Linalg
        // pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

        // // generate buffers
        // pm.addPass(mlir::bufferization::createOneShotBufferizePass(
        //     generateBufferConfig()));
        // pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());

        // // legalise return types
        // pm.addNestedPass<mlir::func::FuncOp>(
        //     HLSPasses::createOutputMemrefPassByRef());

        // // lower linalg to affine in a CIRCT friendly manner
        // pm.addPass(HLSCore::passes::createLowerLinalgToAffineCirctFriendly());

        // // lower affine to cf
        // pm.addPass(mlir::createLowerAffinePass());
        // pm.addPass(mlir::createSCFToControlFlowPass());

        // // allow merge multiple basic block sources
        // pm.addPass(circt::createInsertMergeBlocksPass());

        // log
        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to Precompile");
    });

    addIRLevel(Core, [&]() {
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

    if (failed(pm.run(module)))

    HLSCore::logging::runtime_log<std::string>("Successfully lowered MLIR");

    return writeSingleFileOutput(module, outputFilename, outputFile);
}

} // namespace HLSCore
