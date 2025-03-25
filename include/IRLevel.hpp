#pragma once

namespace HLSCore {

enum IRLevel {
  // A high-level dialect like affine or scf
  High,
  // The IR right before the core lowering dialect
  PreCompile,
  // The IR in core dialect
  Core,
  // The lowest form of core IR (i.e. after all passes have run)
  PostCompile,
  // The IR after lowering is performed
  RTL,
  // System verilog representation
  SV
};


}
