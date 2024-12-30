#include "Options.hpp"

namespace HLSCore {

DynamicParallelismKind dynParallelism = Pipelining; // Assuming Pipelining is defined in another header or source
bool withESI = false;
std::string bufferingStrategy = "all";
unsigned bufferSize = 2;
IRLevel irInputLevel = High;
IRLevel irOutputLevel = SV;
bool splitInputFile = false;

OutputFormatKind outputFormat = OutputVerilog;
bool traceIVerilog = false;
bool withDC = false;
bool verifyPasses = true;
std::string outputFilename = "-";
bool verifyDiagnostics = false;

}
