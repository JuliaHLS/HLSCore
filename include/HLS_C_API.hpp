#pragma once


extern "C" {

typedef struct HLSTool HLSTool;

HLSTool* HLSTool_create();
void HLSTool_destroy(HLSTool* _tool);

void HLSTool_setOptions(HLSTool* tool, char* inputMlir, char* outputFilename);
bool HLSTool_synthesise(HLSTool*);

}
