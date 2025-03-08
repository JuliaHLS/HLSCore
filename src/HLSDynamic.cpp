#include "HLSDynamic.hpp"

namespace HLSCore {

/// Create a simple canonicalizer pass.
std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  return mlir::createCanonicalizerPass(config);
}



void loadDHLSPipeline(OpPassManager &pm) {
  // Memref legalization.
  pm.addPass(circt::createFlattenMemRefPass());
  pm.nest<func::FuncOp>().addPass(
      circt::handshake::createHandshakeLegalizeMemrefsPass());

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());

  // DHLS conversion
  pm.addPass(circt::createCFToHandshakePass(
      false,
      dynParallelism != Pipelining));
  pm.addPass(circt::handshake::createHandshakeLowerExtmemToHWPass(withESI));
  /* pm.addPass(circt::handshake::createHandshakeLowerExtmemToHWPass()); */
  /* pm.addPass(circt::createFlattenMemRefPass()); */
  /* pm.nest<func::FuncOp>().addPass( */
      /* circt::handshake::createHandshakeLegalizeMemrefsPass()); */


  pm.addNestedPass<circt::handshake::FuncOp>(circt::handshake::createHandshakeRemoveBuffersPass());

  if (dynParallelism == Locking) {
    pm.nest<handshake::FuncOp>().addPass(
        circt::handshake::createHandshakeLockFunctionsPass());
    // The locking pass does not adapt forks, thus this additional pass is
    // required
    pm.nest<handshake::FuncOp>().addPass(
        handshake::createHandshakeMaterializeForksSinksPass());
  }
}

void loadHandshakeTransformsPipeline(OpPassManager &pm) {
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeMaterializeForksSinksPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeInsertBuffersPass(bufferingStrategy,
                                                  bufferSize));
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
}

void loadESILoweringPipeline(OpPassManager &pm) {
  pm.addPass(circt::esi::createESIPortLoweringPass());
  pm.addPass(circt::esi::createESIPhysicalLoweringPass());
  pm.addPass(circt::esi::createESItoHWPass());
}

void loadHWLoweringPipeline(OpPassManager &pm) {
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<hw::HWModuleOp>().addPass(circt::seq::createLowerSeqHLMemPass());
  pm.addPass(seq::createHWMemSimImplPass());
  pm.addPass(circt::createLowerSeqToSVPass());
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWCleanupPass());

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Tidy up the IR to improve verilog emission quality.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createPrettifyVerilogPass());
}



LogicalResult doHLSFlowDynamic(
    PassManager &pm, ModuleOp module, const std::string& outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

  bool suppressLaterPasses = false;
  auto notSuppressed = [&]() { return !suppressLaterPasses; };
  auto addIfNeeded = [&](llvm::function_ref<bool()> predicate,
                         llvm::function_ref<void()> passAdder) {
    if (predicate())
      passAdder();
  };

  auto addIRLevel = [&](IRLevel level, llvm::function_ref<void()> passAdder) {
    addIfNeeded(notSuppressed, [&]() {
      // Add the pass if the input IR level is at least the current
      // abstraction.
      if (targetAbstractionLayer(level))
        passAdder();

      // Suppresses later passes if we're emitting IR and the output IR level is
      // the current level.
      if (outputFormat == OutputIR && irOutputLevel == level)
        suppressLaterPasses = true;
    });
  };

  // Resolve blocks with multiple predescessors
  /* pm.addPass(circt::createInsertMergeBlocksPass()); */

    /* pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg()); */

    /* auto buff_opts = mlir::bufferization::OneShotBufferizationOptions(); */
    /* buff_opts.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap); */
    /* buff_opts.bufferizeFunctionBoundaries = true; */

    /* pm.addPass(mlir::bufferization::createOneShotBufferizePass(buff_opts)); */

    /* pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass()); */
  /* mlir::tosa::addTosaToLinalgPasses(pm); */


  // Software lowering
  addIRLevel(PreCompile, [&]() {
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertSCFToCFPass());
  });


  addIRLevel(Core, [&]() { loadDHLSPipeline(pm); });
  addIRLevel(PostCompile,
             [&]() { loadHandshakeTransformsPipeline(pm); });

  // HW path.


pm.addNestedPass<mlir::func::FuncOp>(circt::handshake::createHandshakeLegalizeMemrefsPass());
  addIRLevel(RTL, [&]() {
    pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
    if (withDC) {
      pm.addPass(circt::createHandshakeToDC({"clock", "reset"}));
      // This pass sometimes resolves an error in the
      pm.addPass(createSimpleCanonicalizerPass());
      pm.nest<hw::HWModuleOp>().addPass(
          circt::dc::createDCMaterializeForksSinksPass());
      // TODO: We assert without a canonicalizer pass here. Debug.
      pm.addPass(createSimpleCanonicalizerPass());
      pm.addPass(circt::createDCToHWPass());
      pm.addPass(createSimpleCanonicalizerPass());
      pm.addPass(circt::createMapArithToCombPass());
      pm.addPass(createSimpleCanonicalizerPass());
    } else {
      pm.addPass(circt::createHandshakeToHWPass());
    }
    pm.addPass(createSimpleCanonicalizerPass());
    loadESILoweringPipeline(pm);
  });


  addIRLevel(SV, [&]() { loadHWLoweringPipeline(pm); });

  if(targetAbstractionLayer(RTL)) {
      if (traceIVerilog)
        pm.addPass(circt::sv::createSVTraceIVerilogPass());

      /*if (loweringOptions.getNumOccurrences())*/
      /*  loweringOptions.setAsAttribute(module);*/
      if (outputFormat == OutputVerilog) {
        pm.addPass(createExportVerilogPass((*outputFile)->os()));
      } else if (outputFormat == OutputSplitVerilog) {
        pm.addPass(createExportSplitVerilogPass(outputFilename));
      }
  }
  
  // Go execute!
  if (failed(pm.run(module)))
    return failure();

  module->print((*outputFile)->os());

  return success();
}


}
