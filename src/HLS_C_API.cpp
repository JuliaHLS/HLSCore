#include "HLS_C_API.hpp"
#include "logging.hpp"
#include <memory>

// anonymous namespace to hide conversion functions from the global scope
namespace {

HLSCore::DynamicParallelismKind toDynamicParallelismKind(DynamicParallelismKind dyn) {
    switch (dyn) {
        case DynamicParallelismKind::None:        return HLSCore::DynamicParallelismKind::None;
        case DynamicParallelismKind::Locking :    return HLSCore::DynamicParallelismKind::Locking;
        case DynamicParallelismKind::Pipelining : return HLSCore::DynamicParallelismKind::Pipelining;
        default: { 
            HLSCore::logging::runtime_log("No valid DynamicParallelismKind found, defaulting to Locking");
            return HLSCore::DynamicParallelismKind::Locking;
        }
    }
}



HLSCore::OutputFormatKind toOutputFormatKind(OutputFormatKind outKind) {
    switch (outKind) {
        case OutputFormatKind::OutputIR :           return HLSCore::OutputFormatKind::OutputIR;
        case OutputFormatKind::OutputVerilog :      return  HLSCore::OutputFormatKind::OutputVerilog;
        case OutputFormatKind::OutputSplitVerilog : return HLSCore::OutputFormatKind::OutputSplitVerilog;
        default: { 
            HLSCore::logging::runtime_log("No valid OutputFormatKind found, defaulting to Verilog");
            return HLSCore::OutputFormatKind::OutputVerilog;
        }
    }
}


HLSCore::IRLevel toIRLevel(IRLevel level) {
    switch (level) {
        case IRLevel::High :            return HLSCore::IRLevel::High;
        case IRLevel::PreCompile:       return  HLSCore::IRLevel::PreCompile;
        case IRLevel::Core :            return HLSCore::IRLevel::Core;
        case IRLevel::PostCompile :     return HLSCore::IRLevel::PostCompile;
        case IRLevel::RTL :             return HLSCore::IRLevel::RTL;
        case IRLevel::SV :              return HLSCore::IRLevel::SV;
        default: { 
            HLSCore::logging::runtime_log("No valid IRLevel found, defaulting to SV");
            return HLSCore::IRLevel::SV;
        }
    }
}

std::ostream& operator<<(std::ostream& os, HLSCore::IRLevel level) {
    switch (level) {
        case HLSCore::IRLevel::High:         return os << "High";
        case HLSCore::IRLevel::PreCompile:   return os << "PreCompile";
        case HLSCore::IRLevel::Core:         return os << "Core";
        case HLSCore::IRLevel::PostCompile:  return os << "PostCompile";
        case HLSCore::IRLevel::RTL:          return os << "RTL";
        case HLSCore::IRLevel::SV:           return os << "SV";
        default:                  return os << "ERROR: <Unknown IRLevel>";
    }
}



std::ostream& operator<<(std::ostream& os, IRLevel level) {
    switch (level) {
        case IRLevel::High:         return os << "High";
        case IRLevel::PreCompile:   return os << "PreCompile";
        case IRLevel::Core:         return os << "Core";
        case IRLevel::PostCompile:  return os << "PostCompile";
        case IRLevel::RTL:          return os << "RTL";
        case IRLevel::SV:           return os << "SV";
        default:                  return os << "ERROR: <Unknown IRLevel>";
    }
}


}

extern "C" {

struct HLSTool {
   HLSCore::HLSTool* tool;
};

HLSTool* HLSTool_create(SchedulingKind* _schedulingKind) {
    HLSTool* c_obj = new HLSTool();

    if (_schedulingKind == nullptr) {
        HLSCore::logging::runtime_log<std::string>("error: nullptr");
        return nullptr;
    }

    if (*_schedulingKind == SchedulingKind::Static) {
        HLSCore::logging::runtime_log<std::string>("Static Scheduling");
        c_obj->tool = new HLSCore::HLSToolStatic();
    } else if (*_schedulingKind == SchedulingKind::Dynamic) {
        HLSCore::logging::runtime_log<std::string>("Dynamic Scheduling");
        c_obj->tool = new HLSCore::HLSToolDynamic();
    } else {
        HLSCore::logging::runtime_log<std::string>("ERROR: incorrect scheduling type");
    }

    return c_obj;
}

void HLSTool_destroy(HLSTool* _tool) {
    delete _tool;
}

void HLSTool_setOptions(HLSTool* tool, HLSConfig* options, char* inputMLIR, char* output) {
    // deep copy (to avoid undefined behaviour in case the options are changed
    // deallocated at runtime).
    // Use a smart-ptr to automatically handle the destructor without a memory
    // leak
    auto opt = std::make_unique<HLSCore::OptionsString>(inputMLIR, output);
    
    // copy settings
    if (options) {
            opt->withESI = options->withESI;
            opt->dynParallelism = toDynamicParallelismKind(options->dynParallelism);

            opt->bufferingStrategy = options->bufferingStrategy;
            opt->bufferSize = options->bufferSize;

            opt->irInputLevel = toIRLevel(options->irInputLevel);
            opt->irOutputLevel = toIRLevel(options->irOutputLevel);
            opt->splitInputFile = options->splitInputFile;

            opt->outputFormat = toOutputFormatKind(options->outputFormat);
            opt->traceIVerilog = options->traceIVerilog;
            opt->withDC = options->withDC;
            opt->verifyPasses = options->verifyPasses;
            opt->verifyDiagnostics = options->verifyDiagnostics;

            HLSCore::logging::runtime_logging_flag = options->runtime_logs;

            HLSCore::logging::runtime_log<std::string>("Received the following MLIR from the config: \n" + opt->inputMlir);

            HLSCore::logging::runtime_log<std::string>("Original Input Level");
            HLSCore::logging::runtime_log<IRLevel>(std::forward<IRLevel>(options->irInputLevel));

            HLSCore::logging::runtime_log<std::string>("Original Output Level");
            HLSCore::logging::runtime_log<IRLevel>(std::forward<IRLevel>(options->irOutputLevel));

            HLSCore::logging::runtime_log<std::string>("Translated Input Level");
            HLSCore::logging::runtime_log<HLSCore::IRLevel>(std::forward<HLSCore::IRLevel>(opt->irInputLevel));

            HLSCore::logging::runtime_log<std::string>("Translated Output Level");
            HLSCore::logging::runtime_log<HLSCore::IRLevel>(std::forward<HLSCore::IRLevel>(opt->irOutputLevel));
    }


    tool->tool->setOptions(std::move(opt));
}

bool HLSTool_synthesise(HLSTool* tool) {
    return tool->tool->synthesise();
}

}
