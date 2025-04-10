#pragma once

#include <iostream>

// MLIR Imports
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

// CIRCT Imports
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"

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

class HLSToolDynamic : public HLSTool {
public:
    HLSToolDynamic() {
        logging::runtime_log("Using Dynamically Scheduled HLS flow");

        // register high-level dialects
        HLSCore::pipelines::registerCoreDialects(registry);
        HLSCore::pipelines::registerPreCompileDialects(registry);

        // register CIRCT dialects.
        registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
            sv::SVDialect, handshake::HandshakeDialect, esi::ESIDialect>();
    }

protected:
    [[nodiscard]] virtual LogicalResult runHLSFlow(PassManager &pm, ModuleOp module, const std::string &outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) override final;


private:
    void loadDHLSPipeline(OpPassManager &pm);
    void loadHandshakeTransformsPipeline(OpPassManager &pm);
    void loadESILoweringPipeline(OpPassManager &pm);
    void loadHWLoweringPipeline(OpPassManager &pm);
};



}
