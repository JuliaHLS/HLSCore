#pragma once

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/BuiltinOps.h"

#include <memory>
#include <string>


[[nodiscard]] llvm::LogicalResult writeSingleFileOutput(const mlir::ModuleOp& module, const std::string& outputFilename) {
    std::unique_ptr<llvm::ToolOutputFile> outputFile;
    std::string errorMessage;

    // write to an output file if specified
    if (outputFilename.size() == 0) {
        // if output files are not split
        if (outputFormat != OutputSplitVerilog) {
            // write to output
            outputFile.emplace(openOutputFile(outputFilename, &errorMessage));

            // error handling
            if(!*outputFile) {
                llvm::errs() << "[HLSCore ERROR] :" << errorMessage << "\n";
                return llvm::LogicalResult::failure();
            }
        }
    }

    module->print((*outputFile)->os());
    
    return llvm::LogicalResult::success();
}
