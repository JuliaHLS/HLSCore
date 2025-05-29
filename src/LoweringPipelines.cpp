#include "LoweringPipelines.hpp"
#include "logging.hpp"

namespace {


[[nodiscard]] mlir::bufferization::OneShotBufferizePassOptions generateBufferConfig() {
    auto buff_opts = mlir::bufferization::OneShotBufferizePassOptions();
    buff_opts.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    buff_opts.bufferizeFunctionBoundaries = true;

    return buff_opts;
}

}

namespace HLSCore::pipelines {


void registerCoreDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
}

void registerPreCompileDialects(mlir::DialectRegistry& registry) {
  HLSCore::logging::runtime_log("Registering precompile dialects");
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();

  // register bufferization interfaces
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);

  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
}


// lower TOSA To Affine
void TosaToAffinePipeline(mlir::PassManager &pm) {
    // lower tosa to Linalg
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

    // generate buffers
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(generateBufferConfig()));
    pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());


    // lower linalg to affine in a CIRCT friendly manner
    pm.addPass(HLSCore::passes::createLowerLinalgToAffineCirctFriendly());
}

}
