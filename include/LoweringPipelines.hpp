#pragma once

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

#include "CirctFriendlyLoops.hpp"
#include "OutputMemrefPassByRef.h"


namespace HLSCore::pipelines {


// lower TOSA To Affine
void TosaToAffinePipeline(mlir::PassManager &pm);

}
