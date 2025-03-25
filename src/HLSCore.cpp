//===----------------------------------------------------------------------===//
//
// This file implements a cli for the HLS tool to realise HLS (High Level Synthesis)
//
//===----------------------------------------------------------------------===//

#include "IRLevel.hpp"
#include "HLS.hpp"
#include "HLSDynamic.hpp"
#include "Options.hpp"

#include <iostream>
#include "llvm/Support/CommandLine.h"

using namespace HLSCore;
using namespace llvm;

// Command Line Options
static cl::opt<IRLevel> inputLevelOpt(
    "inType",
    cl::desc("Choose Input Type:"),
    cl::values(
        clEnumValN(IRLevel::High, "High", "High Level Input"),
        clEnumValN(IRLevel::PreCompile, "PreCompile", "PreCompile level as Input"),
        clEnumValN(IRLevel::Core, "Core", "Core level as Input"),
        clEnumValN(IRLevel::PostCompile, "PostCompile", "PostCompile level as Input"),
        clEnumValN(IRLevel::RTL, "RTL", "RTL level Input"),
        clEnumValN(IRLevel::SV, "SV", "SV level Input")
    ),
    cl::init(IRLevel::SV)
);


static cl::opt<IRLevel> outputLevelOpt(
    "outType",
    cl::desc("Choose Output Type:"),
    cl::values(
        clEnumValN(IRLevel::High, "High", "High Level Output"),
        clEnumValN(IRLevel::PreCompile, "PreCompile", "PreCompile level as Output"),
        clEnumValN(IRLevel::Core, "Core", "Core level as Output"),
        clEnumValN(IRLevel::PostCompile, "PostCompile", "PostCompile level as Output"),
        clEnumValN(IRLevel::RTL, "RTL", "RTL level Output"),
        clEnumValN(IRLevel::SV, "SV", "SV level Output")
    ),
    cl::init(IRLevel::SV)
);


// driver program
int hls_driver(std::string& filename) {
    HLSTool hls;
    std::unique_ptr<Options> opt = std::make_unique<HLSCore::OptionsFile>(filename, "-");
    hls.setOptions(std::move(opt));
    auto result = hls.synthesise();

    return 1;
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "HLSCore");
    irOutputLevel = outputLevelOpt;

    /* // Input MLIR string */
    /* std::string inputMLIR = R"mlir(func.func @t2(%arg0: i64, %arg1: i64) -> i64 { */
    /*   %0 = arith.cmpi slt, %arg0, %arg1 : i64 */
    /*   cf.cond_br %0, ^bb1, ^bb2 */
    /* ^bb1:  // pred: ^bb0 */
    /*   %1 = arith.addi %arg0, %arg1 : i64 */
    /*   cf.br ^bb4(%1 : i64) */
    /* ^bb2:  // pred: ^bb0 */
    /*   %2 = arith.cmpi slt, %arg1, %arg0 : i64 */
    /*   %c0_i64 = arith.constant 0 : i64 */
    /*   cf.cond_br %2, ^bb3, ^bb4(%c0_i64 : i64) */
    /* ^bb3:  // pred: ^bb2 */
    /*   %3 = arith.addi %arg0, %arg1 : i64 */
    /*   cf.br ^bb4(%3 : i64) */
    /* ^bb4(%4: i64):  // 3 preds: ^bb1, ^bb2, ^bb3 */
    /*   return %4 : i64 */
    /* })mlir"; */
    
}
