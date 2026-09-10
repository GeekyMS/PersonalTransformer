#pragma once
// Project constants, fixed across all three implementations. Do not let
// these drift — see CLAUDE.md. Mirrors np_impl/model.py's module-level
// globals (B, T, d, H, d_head, L).

constexpr int kB = 32;       // batch size
constexpr int kT = 256;      // context length
constexpr int kV = 65;       // vocab (character-level), confirmed against data/tinyshakespeare.txt
constexpr int kD = 256;      // d_model
constexpr int kH = 4;        // attention heads
constexpr int kDHead = kD / kH;
constexpr int kL = 6;        // layers
