//===----------------------------------------------------------------------===//
//
// This file implements a cli for the HLS tool to realise HLS (High Level Synthesis)
//
//===----------------------------------------------------------------------===//

#include "IRLevel.hpp"
#include "HLS.hpp"
#include "HLSDynamic.hpp"
#include "Options.hpp"
#include "logging.hpp"

#include <iostream>
#include <string>
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

static cl::opt<std::string> inputFilename(
    "input",
    cl::desc("Select Input Filename"),
    cl::value_desc("Default filename: input.mlir"),
    cl::init("input.mlir")
);

static cl::opt<bool> runtime_logging_flag (
    "runtime_log",
    cl::desc("Toggle Runtime Logging"),
    cl::value_desc("Default: false"),
    cl::init(false)
);


// driver program
int hls_driver(std::string& filename) {
    logging::runtime_log<std::string>("Starting HLS Tool");

    HLSTool hls;

    logging::runtime_log<std::string>("Instantiated HLS Tool");

    std::unique_ptr<Options> opt = std::make_unique<HLSCore::OptionsFile>(filename, "-");
    hls.setOptions(std::move(opt));
    
    logging::runtime_log<std::string>("Set up HLS Tool, starting synthesis");

    auto result = hls.synthesise();

    logging::runtime_log<std::string>("Successfully Synthesised Program");

    return 1;
}


int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "HLSCore");
    logging::runtime_logging_flag = runtime_logging_flag;
    irOutputLevel = outputLevelOpt;


    // start driver program
    return hls_driver(inputFilename);
   
}
