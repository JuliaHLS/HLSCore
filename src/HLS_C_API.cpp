#include "HLS_C_API.hpp"
#include "HLSDynamic.hpp"
#include <memory>


extern "C" {

struct HLSOptions {
    HLSCore::OptionsString* options;

    HLSOptions() = delete;
    HLSOptions(char* inputMLIR, char* outputFilename) {
        options = new HLSCore::OptionsString(inputMLIR, outputFilename);
    }
};

HLSOptions* HLSOptions_create(char* inputMLIR, char* outputFilename) {
    return new HLSOptions(inputMLIR, outputFilename);
}

struct HLSTool {
   HLSCore::HLSTool* tool;
};

HLSTool* HLSTool_create() {
    HLSTool* c_obj = new HLSTool();

    c_obj->tool = new HLSCore::HLSToolDynamic();

    return c_obj;
}

void HLSTool_destroy(HLSTool* _tool) {
    delete _tool;
}

void HLSTool_setOptions(HLSTool* tool, HLSOptions* options) {
    // deep copy (to avoid undefined behaviour in case the options are changed
    // deallocated at runtime).
    // Use a smart-ptr to automatically handle the destructor without a memory
    // leak
    auto opt = std::make_unique<HLSCore::OptionsString>(options->options);
    tool->tool->setOptions(std::move(opt));
}

bool HLSTool_synthesise(HLSTool* tool) {
    return tool->tool->synthesise();
}
}
