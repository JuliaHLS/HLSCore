#pragma once

#include "IRLevel.hpp"
#include "Options.hpp"
#include <string>

enum DynamicParallelismKind { None, Locking, Pipelining };
enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };

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
extern std::string inputFilename;
extern std::string outputFilename;
extern bool verifyDiagnostics;
