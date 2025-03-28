#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>
#include <iostream>


namespace HLSCore::passes {

std::unique_ptr<mlir::Pass> createLowerLinalgToAffineCirctFriendly();

}
