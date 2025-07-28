#pragma once

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





namespace HLSPasses {


struct DeleteUnusedMemory 
    : public mlir::PassWrapper<DeleteUnusedMemory,
                         mlir::OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override {
    auto func = getOperation();
    auto funcType = func.getFunctionType();
    mlir::Block &entryBlock = func.front();

    HLSCore::logging::runtime_log<std::string>("Starting Pass");

    std::vector<mlir::Operation*> uses = {};

    func.walk([&](memref::AllocOp allocOp) {
        bool has_uses = false;

        std::vector<Operation*> thisOp;
        thisOp.push_back(allocOp);

        for(Operation *use : allocOp->getUsers()) {
            // uses.push_back(use);
            thisOp.push_back(use);

            if (!(mlir::isa<mlir::memref::AllocOp>(*use)
                    || mlir::isa<mlir::memref::CopyOp>(*use)))
                has_uses = true;

        }

        if (!has_uses) {
            for (auto used_op : thisOp)
                uses.push_back(used_op);
            
        }
    });

    std::set<Operation*> deleted;
    for(int i = uses.size() - 1; i >= 0; i--) {
        if((deleted.find(uses[i]) == deleted.end())) {
            deleted.insert(uses[i]);
            uses[i]->erase();
        }
    }
    

    }

  mlir::StringRef getArgument() const final { return "delete unused memory"; }

  mlir::StringRef getDescription() const final {
    return "Delete unused memory";
  }

};

std::unique_ptr<mlir::Pass> createDeleteUnusedMemory() {
    return std::make_unique<DeleteUnusedMemory>();
}


}
