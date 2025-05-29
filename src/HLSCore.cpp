// This file implements a cli for the HLS tool to realise HLS (High Level Synthesis)

#include "IRLevel.hpp"
#include "HLSDynamic.hpp"
#include "HLSStatic.hpp"
#include "Options.hpp"
#include "logging.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include "llvm/Support/CommandLine.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

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
    cl::init(IRLevel::High)
);

static cl::opt<bool> inputOptimiseInput(
    "optimInput",
    cl::desc("Toggle input MLIR optimisation:"),
    cl::init(true)
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

static cl::opt<std::string> outputFilename(
    "output",
    cl::desc("Select Output Filename"),
    cl::value_desc("Default filename: {EMPTY}"),
    cl::init("")
);

static cl::opt<bool> runtime_logging_flag (
    "runtime_log",
    cl::desc("Toggle Runtime Logging"),
    cl::value_desc("Default: false"),
    cl::init(false)
);

static cl::opt<bool> split_verilog_flag(
    "split_verilog",
    cl::desc("Toggle whether or not outputted Verilog will split into source files"),
    cl::value_desc("Default: false"),
    cl::init(false)
);


static cl::opt<bool> withESIOpt(
    "enable_esi",
    cl::desc("Toggle whether or ESI will be used to generate memory interfaces"),
    cl::value_desc("Default: false"),
    cl::init(false)
);


static cl::opt<bool> withTraceIVerilog(
    "trace_iverilog",
    cl::desc("Trace IVerilog"),
    cl::value_desc("Default: false"),
    cl::init(false)
);


static cl::opt<bool> withDCOpt(
    "with_dc",
    cl::desc("Lower with DC"),
    cl::value_desc("Default: false"),
    cl::init(false)
);


static cl::opt<int> bufferSizeOpt(
    "buff_size",
    cl::desc("Select Buffer Size"),
    cl::value_desc("Default: 2"),
    cl::init(2)
);


static cl::opt<std::string> bufferingStrategyOpt(
    "buff_strategy",
    cl::desc("Buffering Strategy to apply"),
    cl::value_desc("Default: all"),
    cl::init("all")
);


static cl::opt<HLSCore::SchedulingKind> schedulingStrategyOpt(
        "scheduling_strategy",
        cl::desc("Scheduling Strategy to apply"),
        cl::values(
            clEnumValN(HLSCore::SchedulingKind::Static, "static", "Statically Scheduled HLS Flow"),
            clEnumValN(HLSCore::SchedulingKind::Dynamic, "dynamic", "Dynamically Scheduled HLS Flow")
        ),
        cl::value_desc("Default: dynamic"),
        cl::init(HLSCore::SchedulingKind::Dynamic)
);

static llvm::cl::list<std::string> externalIPDeclaraction(
        "custom_ip",
        cl::desc("Custom IP to link, format"),
        cl::ZeroOrMore,
        cl::CommaSeparated,
        cl::value_desc("Default: empty")
);



// driver program
int hls_driver(std::unique_ptr<Options> options) {
    logging::runtime_log<std::string>("Starting HLS Tool");

    std::unique_ptr<HLSTool> hls;

    switch(schedulingStrategyOpt) {
        case HLSCore::SchedulingKind::Static: {
            hls = std::make_unique<HLSToolStatic>(); 
            break;
        }
        case HLSCore::SchedulingKind::Dynamic: {
            hls = std::make_unique<HLSToolDynamic>(); 
            break;
        }
        default: {
            logging::runtime_log<std::string>("Got unregistered unrecognised scheduling strategy");
            throw std::runtime_error("Got unrecognised scheduling strategy");
        }
    }

    logging::runtime_log<std::string>("Instantiated HLS Tool");

    hls->setOptions(std::move(options));
    
    logging::runtime_log<std::string>("Set up HLS Tool, starting synthesis");

    auto result = hls->synthesise();

    logging::runtime_log<std::string>("Successfully Synthesised Program");

    return 1;
}


int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "HLSCore");
    logging::runtime_logging_flag = runtime_logging_flag;

    std::unique_ptr<Options> opt = std::make_unique<HLSCore::OptionsFile>(inputFilename, outputFilename);
    if (externalIPDeclaraction.size() > 0) {
        std::vector<std::vector<std::string>> ip_list = {externalIPDeclaraction};
        opt = std::make_unique<HLSCore::OptionsFile>(inputFilename, outputFilename, ip_list);
    }

    // initialise Options

    // set objects (based off of the CLI arguments)
    opt->irInputLevel = inputLevelOpt;
    opt->irOutputLevel = outputLevelOpt;
    opt->withESI = withESIOpt;
    opt->withDC = withDCOpt;

    opt->outputFormat = split_verilog_flag ? HLSCore::OutputSplitVerilog : HLSCore::OutputVerilog;

    opt->traceIVerilog = withTraceIVerilog; 

    opt->bufferSize = bufferSizeOpt;
    opt->bufferingStrategy = bufferingStrategyOpt;

    opt->optimiseInput = inputOptimiseInput;
    
    if (split_verilog_flag && opt->irOutputLevel != SV)
        throw std::runtime_error("Error: Invalid flags, cannot have split_verilog_flag set while outType != SV");

    // start driver program
    return hls_driver(std::move(opt));
   
}
