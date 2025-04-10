#include "LoweringPipelines.hpp"

namespace {


[[nodiscard]] mlir::bufferization::OneShotBufferizePassOptions generateBufferConfig() {
    auto buff_opts = mlir::bufferization::OneShotBufferizePassOptions();
    buff_opts.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    buff_opts.bufferizeFunctionBoundaries = true;

    return buff_opts;
}

}

namespace HLSCore::pipelines {

// lower TOSA To Affine
void TosaToAffinePipeline(mlir::PassManager &pm) {
    // lower tosa to Linalg
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());

    // generate buffers
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(generateBufferConfig()));
    pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());

    // legalise return types
    pm.addNestedPass<mlir::func::FuncOp>(
        HLSPasses::createOutputMemrefPassByRef());

    // lower linalg to affine in a CIRCT friendly manner
    pm.addPass(HLSCore::passes::createLowerLinalgToAffineCirctFriendly());


}

}
