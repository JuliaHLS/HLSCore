#include "HLS_C_API.hpp"
#include "HLS.hpp"
#include <memory>


extern "C" {
struct HLSTool {
   HLSCore::HLSTool* tool;
};

HLSTool* HLSTool_create() {
    HLSTool* c_obj = new HLSTool();

    c_obj->tool = new HLSCore::HLSTool();

    return c_obj;
}

void HLSTool_destroy(HLSTool* _tool) {
    delete _tool;
}

void HLSTool_setOptions(HLSTool* tool, char* inputMlir, char* outputFilename) {
    auto opt = std::make_unique<HLSCore::Options>(std::string(inputMlir), std::string(outputFilename));
    tool->tool->setOptions(std::move(opt));
}

bool HLSTool_synthesise(HLSTool* tool) {
    return tool->tool->synthesise();
}
}
