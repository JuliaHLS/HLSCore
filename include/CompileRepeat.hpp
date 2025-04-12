#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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



#include "mlir/Transforms/FoldUtils.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <iostream>



#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <iostream>




#include "circt/Dialect/Calyx/Calyx.h.inc"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace circt::calyx;


/// Given some number of states, returns the necessary bit width
/// TODO(Calyx): Probably a better built-in operation?
static size_t getNecessaryBitWidth(size_t numStates) {
  APInt apNumStates(64, numStates);
  size_t log2 = apNumStates.ceilLogBase2();
  return log2 > 1 ? log2 : 1;
}



/// Pattern rewriting a RepeatOp into a lower-level while-loop form.
struct CompileRepeatPattern : public OpRewritePattern<RepeatOp> {
  using OpRewritePattern<RepeatOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(RepeatOp repeatOp, PatternRewriter &rewriter) const override;
};



struct CompileRepeatPass
    : public mlir::PassWrapper<CompileRepeatPass,
                         mlir::OperationPass<mlir::calyx::ComponentOp>> {
private:
  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "compile-repeat"; }

  mlir::StringRef getDescription() const final {
    return "compile-repeat";
  }

};

std::unique_ptr<mlir::Pass> createCompileRepeatPass();

