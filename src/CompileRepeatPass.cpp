#include "CompileRepeat.hpp"
#include "logging.hpp"


void CompileRepeatPass::runOnOperation() {
    HLSCore::logging::runtime_log<std::string>("Starting Custom Pass");
    mlir::func::FuncOp func = getOperation();
    
    mlir::RewritePatternSet patterns (func.getContext());
    patterns.insert<CompileRepeatPattern>(func.getContext());
    
    (void)applyPatternsGreedily(func, std::move(patterns));
}

std::unique_ptr<mlir::Pass> createCompileRepeatPass() { return std::make_unique<CompileRepeatPass>(); } ;



LogicalResult CompileRepeatPattern::matchAndRewrite(RepeatOp repeatOp, PatternRewriter &rewriter) const {
    HLSCore::logging::runtime_log<std::string>("RUNNING CUSTOM PASS");
    uint64_t numRepeats = repeatOp.getCount();

    // get the current location for new operations.
    Location loc = repeatOp.getLoc();

    // replace with empty control
    if (numRepeats == 0) {
      rewriter.eraseOp(repeatOp);
      return success();
    }

    if (numRepeats == 1) {
      auto parentBlock = repeatOp.getBodyBlock();

      rewriter.inlineRegionBefore(repeatOp.getBody(), parentBlock);

      rewriter.eraseOp(repeatOp);

      return success();
    }
    return success();
}
