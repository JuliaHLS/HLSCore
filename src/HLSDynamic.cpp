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
  pm.addPass(mlir::createSCFToControlFlowPass());
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

    auto addIRLevel = [&](HLSCore::IRLevel level, llvm::function_ref<void()> passAdder) {
        addIfNeeded(notSuppressed, [&]() {
        if (targetAbstractionLayer(level))
            passAdder();
        });
    };


    // Resolve blocks with multiple predescessors
    /* pm.addPass(circt::createInsertMergeBlocksPass()); */
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

    auto buff_opts = mlir::bufferization::OneShotBufferizationOptions();
    buff_opts.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
    buff_opts.bufferizeFunctionBoundaries = true;

    pm.addPass(mlir::bufferization::createOneShotBufferizePass(buff_opts));

    pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
    /* mlir::tosa::addTosaToLinalgPasses(pm); */
    pm.addNestedPass<mlir::func::FuncOp>(HLSPasses::createOutputMemrefPassByRef());

    HLSCore::logging::runtime_log<std::string>("Ran Pre-lowering passes");

    // Software lowering
    addIRLevel(PreCompile, [&]() {
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to Precompile");
    });

    addIRLevel(Core, [&]() { 
            loadDHLSPipeline(pm); 
            HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to Core");
    });

    addIRLevel(PostCompile,
             [&]() { 
             loadHandshakeTransformsPipeline(pm); 

            HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to PostCompile");
     });

    // HW path.

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

        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to RTL");
    });

    addIRLevel(SV, [&]() { 
        loadHWLoweringPipeline(pm); 

        // handle output
        if (traceIVerilog)
        pm.addPass(circt::sv::createSVTraceIVerilogPass());

        if (outputFormat == OutputVerilog) {
            pm.addPass(createExportVerilogPass((*outputFile)->os()));
        } else if (outputFormat == OutputSplitVerilog) {
            pm.addPass(createExportSplitVerilogPass(outputFilename));
        }

        HLSCore::logging::runtime_log<std::string>("Successfully added passes to lower to SV");
    });


    /*if (loweringOptions.getNumOccurrences())*/
    /*  loweringOptions.setAsAttribute(module);*/
    // Go execute!
    if (failed(pm.run(module)))

    HLSCore::logging::runtime_log<std::string>("Successfully lowered MLIR");

    return HLSCore::output::writeSingleFileOutput(module, outputFilename, outputFile);
}


}
