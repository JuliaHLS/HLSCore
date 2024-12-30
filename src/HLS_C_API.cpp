#include "HLS_C_API.hpp"
#include "HLS.hpp"
#include <memory>


struct HLSTool_C {
    std::unique_ptr<HLSCore::HLSTool> tool;
};

HLSTool_C* HLSTool_C_create() {
    HLSTool_C* c_obj;

    c_obj->tool = std::make_unique<HLSCore::HLSTool>();

    return c_obj;
}

void HLSTool_C_destroy(HLSTool_C* _tool) {
    delete _tool;
}

void HLSTool_C_setOptions(HLSTool_C* tool, char* inputMlir, char* outputFilename) {
    auto opt = std::make_unique<HLSCore::Options>(std::string(inputMlir), std::string(outputFilename));
    tool->tool->setOptions(std::move(opt));
}

bool HLSTool_C_synthesise(HLSTool_C* tool) {
    return tool->tool->synthesise();
}
