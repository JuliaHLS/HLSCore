#pragma once

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ToolOutputFile.h"
#include "Options.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "logging.hpp"

#include <memory>
#include <string>


namespace HLSCore::output {

[[nodiscard]] llvm::LogicalResult writeSingleFileOutput(const mlir::ModuleOp& module, const std::string& outputFilename, std::optional<std::unique_ptr<llvm::ToolOutputFile>>& outputFile);

}
