#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace HLSPasses {

std::unique_ptr<mlir::Pass> createOutputMemrefPassByRef();

struct OutputMemrefPassByRef 
    : public mlir::PassWrapper<OutputMemrefPassByRef,
                         mlir::OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;  // implemented in AffineFullUnroll.cpp

  mlir::StringRef getArgument() const final { return "output-memref-pass-by-ref"; }

  mlir::StringRef getDescription() const final {
    return "Replace returned memrefs with pass by reference";
  }
};



}
