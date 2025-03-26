#pragma once

#include "IRLevel.hpp"
#include "Options.hpp"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include "mlir/Support/FileUtilities.h"

namespace HLSCore {

enum DynamicParallelismKind { None, Locking, Pipelining };
enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };

class Options {
public:
    // extract buffer ref (regardless of input type
    [[nodiscard]] virtual std::unique_ptr<llvm::MemoryBuffer> getInputBuffer() const = 0;
    [[nodiscard]] const std::string getOutputFilename() const { return outputFilename; }

    Options()
    {
    }

protected:
    std::string outputFilename;
};

class OptionsString : public Options {
public:
    const std::string inputMlir;

    [[nodiscard]] virtual std::unique_ptr<llvm::MemoryBuffer> getInputBuffer() const override { return llvm::MemoryBuffer::getMemBuffer(inputMlir); }

    OptionsString(const std::string& _inputMlir, const std::string& _outputFilename) :
        inputMlir (_inputMlir)
    {
        // write to console if name not entered
        if (_outputFilename.size() == 0) outputFilename = "-";
        else outputFilename = _outputFilename;
    }
};

class OptionsFile : public Options {
public:
    const std::string inputFilename;

    [[nodiscard]] virtual std::unique_ptr<llvm::MemoryBuffer> getInputBuffer() const override {
        // try to extract memory buffer
        auto buffer = mlir::openInputFile(inputFilename);

        // check if error was received when opening the file
        if (!buffer) throw std::runtime_error("[HLSCore ERROR]: MLIR input file returned nullptr \n");
        
        // return buffer unique pointer (rval is implicit on return)
        return buffer;
    }

    OptionsFile(const std::string& _inputFilename, const std::string& _outputFilename) :
        inputFilename (_inputFilename)
    {
        // write to console if name not entered
        if (_outputFilename.size() == 0) outputFilename = "-";
        else outputFilename = _outputFilename;
    }   
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
