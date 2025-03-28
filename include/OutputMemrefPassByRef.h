#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <iostream>


namespace HLSPasses {

std::unique_ptr<mlir::Pass> createOutputMemrefPassByRef();

struct OutputMemrefPassByRef 
    : public mlir::PassWrapper<OutputMemrefPassByRef,
                         mlir::OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "output-memref-pass-by-ref"; }

  mlir::StringRef getDescription() const final {
    return "Replace returned memrefs with pass by reference";
  }

  // function state helpers 
  [[nodiscard]] bool returnIsMemRef();

  // pass state
  mlir::func::FuncOp func;
  mlir::FunctionType funcType;

};



}
