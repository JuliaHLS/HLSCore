#include "HLSDynamic.hpp"

#include "BindExternalIP.h"
#include "FloatToInt.h"
#include "DeleteUnusedMemory.h"
#include "circt/Support/LoweringOptions.h"


namespace HLSCore {

void HLSToolDynamic::loadDHLSPipeline(OpPassManager &pm) {

  // Memref legalization.
  pm.addPass(circt::createFlattenMemRefPass());

    // legalise return types
    pm.addNestedPass<mlir::func::FuncOp>(
        HLSPasses::createOutputMemrefPassByRef());


    HLSCore::logging::runtime_log<std::string>("Optimise Input is set to:");
    HLSCore::logging::runtime_log<bool>(std::move(opt->optimiseInput));

    if(opt->optimiseInput){
        HLSCore::logging::runtime_log<std::string>("Running Memory Optimisations");
        pm.addNestedPass<mlir::func::FuncOp>(HLSPasses::createDeleteUnusedMemory());
    }

  pm.nest<func::FuncOp>().addPass(
      circt::handshake::createHandshakeLegalizeMemrefsPass());


    // allow merge multiple basic block sources
    pm.addPass(circt::createInsertMergeBlocksPass());

  pm.addPass(mlir::createSCFToControlFlowPass());

  pm.addPass(
      circt::createCFToHandshakePass(false, opt->dynParallelism != Pipelining));
  pm.addPass(circt::handshake::createHandshakeLowerExtmemToHWPass(opt->withESI));

  if (opt->dynParallelism == Locking) {
      HLSCore::logging::runtime_log<std::string>("LOCKING");
    pm.nest<handshake::FuncOp>().addPass(
        circt::handshake::createHandshakeLockFunctionsPass());
    // The locking pass does not adapt forks, thus this additional pass is
    // required
    pm.nest<handshake::FuncOp>().addPass(
        handshake::createHandshakeMaterializeForksSinksPass());
  }
}

void HLSToolDynamic::loadHandshakeTransformsPipeline(OpPassManager &pm) {
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeMaterializeForksSinksPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeInsertBuffersPass(opt->bufferingStrategy,
                                                  opt->bufferSize));
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
}

void HLSToolDynamic::loadESILoweringPipeline(OpPassManager &pm) {
  pm.addPass(circt::esi::createESIPortLoweringPass());
  pm.addPass(circt::esi::createESIPhysicalLoweringPass());
  pm.addPass(circt::esi::createESItoHWPass());
}

void HLSToolDynamic::loadHWLoweringPipeline(OpPassManager &pm) {
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


LogicalResult HLSToolDynamic::runHLSFlow(
    PassManager &pm, ModuleOp module, const std::string &outputFilename,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

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
        HLSCore::pipelines::TosaToAffinePipeline(pm);

        pm.addNestedPass<mlir::func::FuncOp>(HLSPasses::createOutputMemrefPassByRef());

        if(opt->optimiseInput){
          pm.nest<func::FuncOp>().addPass(createNormalCanonicalizerPass());
        }


        // lower affine to cf
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());



        // pm.addNestedPass<mlir::func::FuncOp>(HLSPasses::createBindExternalIP());
        // pm.addPass(HLSPasses::createFloatToInt());

        // log
        HLSCore::logging::runtime_log<std::string>(
            "Successfully added passes to lower to Precompile");
    });

    addIRLevel(Core, [&]() {
        loadDHLSPipeline(pm);
        HLSCore::logging::runtime_log<std::string>(
            "Successfully added passes to lower to Core");
    });

    addIRLevel(PostCompile, [&]() {
    loadHandshakeTransformsPipeline(pm);

    HLSCore::logging::runtime_log<std::string>(
        "Successfully added passes to lower to PostCompile");
    });

    // HW path.

    addIRLevel(RTL, [&]() {
    pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
    if (opt->withDC) {
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
      pm.addPass(circt::dc::createDCMaterializeForksSinksPass());
      pm.addPass(createSimpleCanonicalizerPass());
    } else {
      // pm.addPass(circt::createMapArithToCombPass());
      pm.addPass(circt::createHandshakeToHWPass());
    }
    pm.addPass(createSimpleCanonicalizerPass());
    loadESILoweringPipeline(pm);

    HLSCore::logging::runtime_log<std::string>(
        "Successfully added passes to lower to RTL");
    });

    addIRLevel(SV, [&]() {
    loadHWLoweringPipeline(pm);




    // handle output
    if (opt->traceIVerilog)
      pm.addPass(circt::sv::createSVTraceIVerilogPass());

    if (opt->outputFormat == OutputVerilog) {
      pm.addPass(createExportVerilogPass((*outputFile)->os()));
    } else if (opt->outputFormat == OutputSplitVerilog) {
      pm.addPass(createExportSplitVerilogPass(outputFilename));
    }

    HLSCore::logging::runtime_log<std::string>(
        "Successfully added passes to lower to SV");
    });


    // auto loweringOptions = circt::LoweringOptions();

    // switch(opt->synthTarget) {
    // case SynthesisTarget::GENERIC: {
    //     break;
    // }
    // case SynthesisTarget::QUARTUS: {
    //     std::cout << "Running Quartus configuration" << std::endl;
    //     loweringOptions.emitWireInPorts = true; 
    //     break;
    // }
    // default: {
    //     logging::runtime_log("Invalid SynthesisTarget used, reverting to default settings...");
    //     break;
    // }

    // }


    // loweringOptions.setAsAttribute(module);

    HLSCore::logging::runtime_log<std::string>(
      "Trying to start MLIR Lowering Process");

    if (failed(pm.run(module))) {
        HLSCore::logging::runtime_log<std::string>("Failed, dumping IR:");
        module.dump();
    }

    HLSCore::logging::runtime_log<std::string>("Successfully lowered MLIR");

    return writeSingleFileOutput(module, outputFilename, outputFile);
}

} // namespace HLSCore
