#include "FloatToInt.h"
#include "logging.hpp"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
// #include "llvm/ADT/Optional.h"

// Then your MLIR includes:
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Replace every floating-point element type by an integer with identical
/// bit-width.  All other types are kept as-is.  Container types (memref,
/// tensor, vector) are rewritten recursively, and a helper converts
/// function signatures.
struct FloatToIntTypeConverter : public mlir::TypeConverter {
  FloatToIntTypeConverter() {
    // ───────────── 1a.  Scalar float → scalar int (same width) ────────────
    addConversion([](mlir::FloatType ft) -> mlir::Type {
        HLSCore::logging::runtime_log("add conversion for scalars");
      return mlir::IntegerType::get(ft.getContext(), ft.getWidth());
    });

    // ───────────── 1b.  Shaped types (tensor/memref/vector) ───────────────
    addConversion([&](mlir::ShapedType st,
                      llvm::SmallVectorImpl<mlir::Type> &out) {
      mlir::Type elem = st.getElementType();
      llvm::SmallVector<mlir::Type> converted;
      if (failed(convertType(elem, converted)))
        return mlir::failure();
      out.push_back(st.clone(converted.front()));
      return mlir::success();
    });

    // ───────────── 1c.  Function type ─────────────────────────────────────
    addConversion([&](mlir::FunctionType fn,
                      llvm::SmallVectorImpl<mlir::Type> &out) {
      llvm::SmallVector<mlir::Type> ins, res;
      if (failed(convertTypes(fn.getInputs(),  ins)) ||
          failed(convertTypes(fn.getResults(), res)))
        return mlir::failure();
      out.push_back(mlir::FunctionType::get(fn.getContext(), ins, res));
      return mlir::success();
    });

    // Keep all other types unchanged.
    addConversion([](mlir::Type t) { return t; });
  }
};


struct ConstantFloatToInt
    : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
  using mlir::OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Only touch constants that really *hold* a floating attribute
    auto fAttr = op.getValue().dyn_cast_or_null<mlir::FloatAttr>();
    if (!fAttr)
      return mlir::failure();

    mlir::APInt bits = fAttr.getValue().bitcastToAPInt();
    unsigned width  = bits.getBitWidth();
    auto intTy = mlir::IntegerType::get(op.getContext(), width);

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, mlir::IntegerAttr::get(intTy, bits));
    return mlir::success();
  }
};

void HLSPasses::FloatToInt::runOnOperation() 
{
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();

    FloatToIntTypeConverter typeConv;
    RewritePatternSet patterns(&ctx);

    // Interface patterns for func ops / call / return
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConv);
    populateCallOpTypeConversionPattern(patterns, typeConv);
    populateReturnOpTypeConversionPattern(patterns, typeConv);

    // Constant pattern
    patterns.add<ConstantFloatToInt>(typeConv, &ctx);

    // Conversion target
    ConversionTarget target(ctx);
    // All ops are legal iff their types & regions are legal...
    target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return typeConv.isLegal(op); });

    // …but make func.func illegal when its signature still has floats
    target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp fn) {
        return typeConv.isSignatureLegal(fn.getFunctionType()) &&
               typeConv.isLegal(&fn.getBody());
      });

    if (failed(applyFullConversion(module, target,
                                   std::move(patterns)))) {
      module.dump();
      signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> HLSPasses::createFloatToInt() {
    return std::make_unique<HLSPasses::FloatToInt>();
}
