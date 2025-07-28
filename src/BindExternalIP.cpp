#include "BindExternalIP.h"
#include "logging.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

constexpr std::pair<StringRef, StringRef> kOps[] = {
    {arith::AddFOp::getOperationName(), "arith_addf"},
    // {arith::MulFOp::getOperationName(), "arith_mulf"},
};

void HLSPasses::BindExternalIP::runOnOperation() {
    HLSCore::logging::runtime_log<std::string>("Starting Pass BindExternalIP");

    auto func = getOperation();
    OpBuilder topBuilder(func.getContext());

    // Cache to avoid regenerating the same helper multiple times.
    llvm::StringMap<func::FuncOp> helpers;

    // Pre‑create the canonical f64 type once.
    auto f64Ty = topBuilder.getF64Type();

    func.walk([&](Operation *op) {
      for (auto [opName, helperBase] : kOps) {
        if (op->getName() != OperationName(opName, func.getContext()))
          continue;
        if (op->getNumResults() != 1 || op->getResult(0).getType() != f64Ty)
          return;

        Region &region = func.getBody();
        Operation *parentOp = region.getParentOp();        // should be the FuncOp
        ModuleOp module3 = parentOp->getParentOfType<ModuleOp>();


        std::string helperName = (helperBase + "_f64").str();
        func::FuncOp helper = helpers.lookup(helperName);
        if (!helper) {
          SmallVector<Type> argTys(op->getOperandTypes());
          FunctionType fnTy = topBuilder.getFunctionType(argTys, op->getResultTypes());

          OpBuilder::InsertionGuard g(topBuilder);

          topBuilder.setInsertionPointToStart(module3.getBody());

          helper = topBuilder.create<func::FuncOp>(op->getLoc(), helperName, fnTy);
          helper.setPrivate();

          helpers[helperName] = helper;
            HLSCore::logging::runtime_log<std::string>("created and populated function");
        }

        helper = helpers.lookup(helperName);
        if (!helper) {
            HLSCore::logging::runtime_log<std::string>("helper not created successfully");
            
        }

        OpBuilder repl(op);
        auto call = repl.create<func::CallOp>(op->getLoc(), helper, op->getOperands());
        op->replaceAllUsesWith(call.getResults());
        op->erase();
        break;
      }
    }
    );
}

std::unique_ptr<mlir::Pass> HLSPasses::createBindExternalIP() {
    return std::make_unique<HLSPasses::BindExternalIP>();
}
