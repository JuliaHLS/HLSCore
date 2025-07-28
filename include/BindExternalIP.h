#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <iostream>


namespace HLSPasses {

std::unique_ptr<mlir::Pass> createBindExternalIP();

struct BindExternalIP 
    : public mlir::PassWrapper<BindExternalIP,
                         mlir::OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "bind-external-ip"; }

  mlir::StringRef getDescription() const final {
    return "BindExternalIP";
  }

};



}
