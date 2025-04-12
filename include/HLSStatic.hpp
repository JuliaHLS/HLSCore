#pragma once

#include "mlir/Transforms/Passes.h"

// MLIR Imports
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

// calyx imports
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include <circt/Conversion/CalyxToFSM.h>
#include <circt/Conversion/FSMToSV.h>
#include <circt/Conversion/SCFToCalyx.h>
#include <circt/Conversion/SeqToSV.h>

// CIRCT Imports
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "circt/Conversion/AffineToLoopSchedule.h"

// HLSCore Imports
#include "IRLevel.hpp"
#include "Options.hpp"
#include "CirctFriendlyLoops.hpp"
#include "HLS.hpp"
#include "OutputMemrefPassByRef.h"
#include "logging.hpp"
#include "LoweringPipelines.hpp"


using namespace llvm;
using namespace mlir;
using namespace circt;


namespace HLSCore {

class HLSToolStatic : public HLSTool {
public:
    HLSToolStatic() {
        logging::runtime_log("Using Statically Scheduled HLS flow");

        // register high-level dialects
        HLSCore::pipelines::registerCoreDialects(registry);
        HLSCore::pipelines::registerPreCompileDialects(registry);

        // register CIRCT dialects
        registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect, 
            loopschedule::LoopScheduleDialect, sv::SVDialect, calyx::CalyxDialect>();
    }

protected:
    [[nodiscard]] virtual LogicalResult runHLSFlow(PassManager &pm, ModuleOp module, const std::string &outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) override final;

};



}
