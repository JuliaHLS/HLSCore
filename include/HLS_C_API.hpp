#pragma once


extern "C" {

// Note: only expose OptionsString, because we want to use this to interoperate between the front end and back end
struct HLSOptions;

HLSOptions* HLSOptions_create(char* inputMLIR, char* outputFilename);
void HLSOptions_destroy(HLSOptions* options);

// expose a pointer to the HLS tool
struct HLSTool;

HLSTool* HLSTool_create();
void HLSTool_destroy(HLSTool* _tool);

void HLSTool_setOptions(HLSTool* tool, HLSOptions* options);
bool HLSTool_synthesise(HLSTool* tool);

}
