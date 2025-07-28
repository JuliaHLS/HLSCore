#pragma once

#include "IRLevel.hpp"
#include "Options.hpp"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include "mlir/Support/FileUtilities.h"
#include "logging.hpp"

#include <unordered_map>
#include <vector>

namespace HLSCore {

enum DynamicParallelismKind { None, Locking, Pipelining };
enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };
enum SchedulingKind { Static, Dynamic };

class CustomIP {
public:
    CustomIP(std::vector<std::vector<std::string>> input) {
        if (input.size() > 0) {
            for (const auto& string_entry : input) {
                logging::runtime_log("Added new CustomIP entry");

                std::string name = string_entry.front();
                std::string mlir_def = string_entry.at(2);

                logging::runtime_log<std::string>(std::move(name));
                logging::runtime_log<std::string>(std::move(mlir_def));

                std::vector<std::string> p_names;

                for (int i = 2; i < string_entry.size(); i++){
                    p_names.push_back(string_entry.at(i));
                    // logging::runtime_log(p_names);
                }

                // fill input
                ip_names.push_back(name);
                mlir_definitions[name] = mlir_def;
                port_names[name] = p_names;
            }
        }
    }

    CustomIP(){
        
    };

private:
    std::vector<std::string> ip_names;
    std::unordered_map<std::string, std::string> mlir_definitions;
    std::unordered_map<std::string, std::vector<std::string>> port_names;
};

enum SynthesisTarget {
    GENERIC,
    QUARTUS
};

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

        optimiseInput = true;
        synthTarget = SynthesisTarget::GENERIC;

        custom_ip = CustomIP();
    }

    // public access to simplify C-API interaction
    bool withESI;
    DynamicParallelismKind dynParallelism; 

    std::string bufferingStrategy;
    unsigned bufferSize;

    IRLevel irInputLevel;
    IRLevel irOutputLevel;
    bool splitInputFile; 

    SynthesisTarget synthTarget;

    OutputFormatKind outputFormat;
    bool traceIVerilog;
    bool withDC;
    bool verifyPasses;
    bool verifyDiagnostics;
    bool optimiseInput;

    CustomIP custom_ip;

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

    OptionsString(const std::string& _inputMlir, const std::string& _outputFilename, std::vector<std::vector<std::string>> _custom_ip) :
        inputMlir (_inputMlir)
    {
        // write to console if name not entered
        if (_outputFilename.size() == 0) outputFilename = "-";
        else outputFilename = _outputFilename;

        custom_ip = CustomIP(_custom_ip);
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

    OptionsFile(const std::string& _inputFilename, const std::string& _outputFilename, std::vector<std::vector<std::string>> _custom_ip) :
        inputFilename (_inputFilename)
    {
        // write to console if name not entered
        if (_outputFilename.size() == 0) outputFilename = "-";
        else outputFilename = _outputFilename;

        custom_ip = CustomIP(_custom_ip);
    }   

};

}
