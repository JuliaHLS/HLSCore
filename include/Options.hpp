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
enum SchedulingKind { Static, Dynamic };

class Options {
public:
    // extract buffer ref (regardless of input type
    [[nodiscard]] virtual std::unique_ptr<llvm::MemoryBuffer> getInputBuffer() const = 0;
    [[nodiscard]] const std::string getOutputFilename() const { return outputFilename; }

    Options()
    {
        // initialise default Options
        withESI = false;
        bufferSize = 2;

        dynParallelism = Pipelining;

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
    DynamicParallelismKind dynParallelism; 

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
};

class OptionsString : public Options {
public:
    std::string inputMlir;

    [[nodiscard]] virtual std::unique_ptr<llvm::MemoryBuffer> getInputBuffer() const override { return llvm::MemoryBuffer::getMemBuffer(inputMlir); }

    OptionsString()=delete;

    OptionsString(const std::string& _inputMlir, const std::string& _outputFilename) :
        inputMlir (_inputMlir)
    {
        // write to console if name not entered
        if (_outputFilename.size() == 0) outputFilename = "-";
        else outputFilename = _outputFilename;
    }

    // copy ctr (ptr) implemented as a deep copy
    OptionsString(const OptionsString* other) {
        // check if it is a nullptr
        if (other) {
            this->inputMlir = other->inputMlir;
            this->outputFilename = other->outputFilename;

            this->withESI = other->withESI;
            this->dynParallelism = other->dynParallelism;

            this->bufferingStrategy = other->bufferingStrategy;
            this->bufferSize = other->bufferSize;

            this->irInputLevel = other->irInputLevel;
            this->irOutputLevel = other->irOutputLevel;
            this->splitInputFile = other->splitInputFile;

            this->outputFormat = other->outputFormat;
            this->traceIVerilog = other->traceIVerilog;
            this->withDC = other->withDC;
            this->verifyPasses = other->verifyPasses;
            this->verifyDiagnostics = other->verifyDiagnostics;
        }
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

    OptionsFile()=delete;

    OptionsFile(const std::string& _inputFilename, const std::string& _outputFilename) :
        inputFilename (_inputFilename)
    {
        // write to console if name not entered
        if (_outputFilename.size() == 0) outputFilename = "-";
        else outputFilename = _outputFilename;
    }   
};

}
