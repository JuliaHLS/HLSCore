#include "OutputMemrefPassByRef.h"
#include "logging.hpp"

void HLSPasses::OutputMemrefPassByRef::runOnOperation() {
    func = getOperation();
    funcType = func.getFunctionType();
    mlir::Block &entryBlock = func.front();

    HLSCore::logging::runtime_log<std::string>("Starting Pass");
    

    // modify the function if it is returning a memref
    if (returnIsMemRef()) {
        HLSCore::logging::runtime_log<std::string>("Output is memref");

        // instantiate rewriter and OpBuilder
        mlir::IRRewriter rewriter(&getContext());
        mlir::OpBuilder builder(func.getContext());

        // function information
        mlir::func::ReturnOp returnOp;

        // extract return operation
        func.walk([&](mlir::func::ReturnOp op) {
            returnOp = op;
        });


        // extract old ssa value
        auto oldSSAValue = returnOp.getOperand(0);
        HLSCore::logging::runtime_log<mlir::Value>(returnOp.getOperand(0));
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(oldSSAValue.getType());
        auto functionType = func.getFunctionType();

        // rewrite function arg types
        mlir::SmallVector<mlir::Type, 6> newArgTypes (func.getFunctionType().getInputs());
        newArgTypes.push_back(memrefType);

        auto newFunctionType = mlir::FunctionType::get(func.getContext(), newArgTypes,
                                           builder.getI1Type());
        func.setType(newFunctionType);

        // rewrite mlir block
        mlir::Block &entryBlock = func.front();
        entryBlock.addArgument(oldSSAValue.getType(), func.getLoc());

        // replace with new arg
        mlir::BlockArgument newArg = entryBlock.getArguments().back();

        // remove old SSA values
        oldSSAValue.replaceAllUsesWith(newArg);

        if(!oldSSAValue.isa<mlir::BlockArgument>()) {
            oldSSAValue.getDefiningOp()->erase();
        }
        // HLSCore::logging::runtime_log<mlir::Operation>(oldSSAValue.getDefiningOp());

        // insert new return op (boolean)
        rewriter.setInsertionPoint(returnOp);

        mlir::Value newReturn = rewriter.create<mlir::arith::ConstantOp>(returnOp.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
        returnOp->setOperands(newReturn);
    }
}

    // check if return type is MemRef
[[nodiscard]] bool HLSPasses::OutputMemrefPassByRef::returnIsMemRef() {
    // store num results
    const auto numResults = funcType.getNumResults();

    // check if function returns simple array type
    if (numResults == 1) {
        // extract result type
        const mlir::Type retType = funcType.getResult(0);

        // compare and return result type
        return retType.isa<mlir::MemRefType>();

    } else if (numResults > 1) {
        // for each result, check if there is a return type to consider
        
        for(uint i = 0; i < numResults; i++) {
            // extract current result
            const mlir::Type retType = funcType.getResult(i);

            // print warning if MemRefType warning
            if (retType.isa<mlir::MemRefType>()) {
                // print warning
                
                llvm::outs() << "Warning: MemRefType found as return type. Not sanitised by OutputMemrefPassByRef. Please implement custom pass \n";

                // early exit
                break;
            }
        }

    }

    // return default output - false
    return false;
}


std::unique_ptr<mlir::Pass> HLSPasses::createOutputMemrefPassByRef() {
    return std::make_unique<HLSPasses::OutputMemrefPassByRef>();
}
