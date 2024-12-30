#pragma once

#include "IRLevel.hpp"
#include "Options.hpp"
#include <string>

namespace HLSCore {

enum DynamicParallelismKind { None, Locking, Pipelining };
enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };

struct Options {
    const std::string inputMlir;
    const std::string outputFilename;

    Options(const std::string& _inputMlir, const std::string& _outputFilename) :
        inputMlir (_inputMlir),
        outputFilename (_outputFilename)
    {}
};

extern DynamicParallelismKind dynParallelism;
extern bool withESI;
extern std::string bufferingStrategy;
extern unsigned bufferSize;
extern IRLevel irInputLevel;
extern IRLevel irOutputLevel;
extern bool splitInputFile;

extern OutputFormatKind outputFormat;
extern bool traceIVerilog;
extern bool withDC;
extern bool verifyPasses;
extern bool verifyDiagnostics;

}
