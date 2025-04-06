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

    // getters and setters (allows for better debugging and access control)
    // void setWithESI(bool _withESI) { withESI = _withESI; }
    // [[nodiscard]] bool getWithESI() const { return withESI; }

    // void setBufferSize(unsigned _bufferSize) { bufferSize = _bufferSize; }
    // [[nodiscard]] unsigned getBufferSize() const { return bufferSize; }

    // void setIrInputLevel (IRLevel _irInputLevel) { irInputLevel = _irInputLevel; }
    // [[nodiscard]] IRLevel getIrInputLevel () const { return irInputLevel; }

    // void setIrOutputLevel (IRLevel _irOutputLevel) { irOutputLevel = _irOutputLevel; }
    // [[nodiscard]] IRLevel setIrOutputLevel () const { return irOutputLevel; }

    // void set 


    Options()
    {
        // initialise default Options
        withESI = false;
        bufferSize = 2;

        irInputLevel = High;
        irOutputLevel = SV;
        splitInputFile = false;

        outputFormat = OutputVerilog;
        traceIVerilog = false;
        withDC = false;
        verifyPasses = true;
        verifyDiagnostics = false;
    }

    // public access to simplify C-API interaction
    bool withESI;

    std::string bufferingStrategy;
    unsigned bufferSize;

    IRLevel irInputLevel;
    IRLevel irOutputLevel;
    bool splitInputFile; 

    OutputFormatKind outputFormat;
    bool traceIVerilog;
    bool withDC;
    bool verifyPasses;
    bool verifyDiagnostics;


protected:
    std::string outputFilename;

    // Required Options
    // bool withESI;

    // std::string bufferingStrategy;
    // unsigned bufferSize;

    // IRLevel irInputLevel;
    // IRLevel irOutputLevel;
    // bool splitInputFile; 

    // OutputFormatKind outputFormat;
    // bool traceIVerilog;
    // bool withDC;
    // bool verifyPasses;
    // bool verifyDiagnostics;
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


// determine if the current level is within the target range
[[nodiscard]] bool targetAbstractionLayer(IRLevel currentLevel);

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
