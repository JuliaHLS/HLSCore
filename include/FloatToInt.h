#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <iostream>


namespace HLSPasses {

std::unique_ptr<mlir::Pass> createFloatToInt();

struct FloatToInt 
    : public mlir::PassWrapper<FloatToInt,
                         mlir::OperationPass<mlir::ModuleOp>> {
private:
  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "float-to-int"; }

  mlir::StringRef getDescription() const final {
    return "Float To Int (bit array)";
  }

};



}
