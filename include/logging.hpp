#pragma once

#include "llvm/Support/raw_ostream.h"


namespace HLSCore::logging {
extern bool runtime_logging_flag;


// template for generating runtime logs
template<typename a> 
void runtime_log(a&& msg) {
    if (runtime_logging_flag) {
        outs() << "[HLSCore LOG]: " << msg << "\n";
    }
}

}
