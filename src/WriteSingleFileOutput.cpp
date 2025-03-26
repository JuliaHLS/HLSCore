#include "WriteSingleFileOutput.hpp"


[[nodiscard]] llvm::LogicalResult HLSCore::output::writeSingleFileOutput(const mlir::ModuleOp& module, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>>& outputFile) {

    std::string errorMessage;

    // if output files are not written via CIRCT passes
    if (outputFormat != OutputSplitVerilog && outputFormat != HLSCore::OutputVerilog) {
        HLSCore::logging::runtime_log<std::string>("Setting detected, HLSTool configured to write to a single file (including terminal)");

        // write to an output file if specified
        if (outputFilename.size() == 0) {
            HLSCore::logging::runtime_log<std::string>("Writing to single file.");
            // write to output
            outputFile.emplace(mlir::openOutputFile(outputFilename, &errorMessage));

            // error handling
            if(!*outputFile) {
                llvm::errs() << "[HLSCore ERROR] :" << errorMessage << "\n";
                return llvm::LogicalResult::failure();
            }
        }

        HLSCore::logging::runtime_log<std::string>("Writing output...");
        // print the output
        module->print((*outputFile)->os());
    }


    // return success
    return llvm::LogicalResult::success();
}


