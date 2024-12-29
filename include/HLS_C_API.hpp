#pragma once

#include "Options.hpp"


extern "C" {

typedef struct HLSTool_C HLCTool_C;

HLSTool_C* HLSTool_C_create();
void HLSTool_C_destroy(HLSTool_C* _tool);

void HLSTool_C_setOptions(HLSTool_C* tool, char* inputMlir, char* outputFilename);
bool HLSTool_C_synthesise(HLSTool_C*);

}
