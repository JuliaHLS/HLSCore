#pragma once


#include "HLSDynamic.hpp"
#include "Options.hpp"


extern "C" {

// Note: only expose OptionsString, because we want to use this to interoperate between the front end and back end

enum DynamicParallelismKind { None, Locking, Pipelining };
enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };

enum IRLevel {
  // high-level dialects (above "standard")
  High,
  // IR before the core lowering dialect
  PreCompile,
  // core dialect
  Core,
  // lowest form of core IR
  PostCompile,
  // lowering IR to RTL representation
  RTL,
  // System Verilog representation
  SV
};


struct HLSConfig {
    bool withESI;
    DynamicParallelismKind dynParallelism;

    char* bufferingStrategy;
    unsigned bufferSize;

    IRLevel irInputLevel;
    IRLevel irOutputLevel;
    bool splitInputFile; 

    OutputFormatKind outputFormat;
    bool traceIVerilog;
    bool withDC;
    bool verifyPasses;
    bool verifyDiagnostics;

    bool runtime_logs;
};


// HLSOptions* HLSOptions_create(char* inputMLIR, char* outputFilename);
// void HLSOptions_destroy(HLSOptions* options);

// expose a pointer to the HLS tool
struct HLSTool;

HLSTool* HLSTool_create();
void HLSTool_destroy(HLSTool* _tool);

void HLSTool_setOptions(HLSTool* tool, HLSConfig* options, char* inputMLIR, char* output);
bool HLSTool_synthesise(HLSTool* tool);


}
