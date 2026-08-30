#pragma once
// Binary dump/load for cross-language correctness checks (see cpp/test/README.md).
// Deliberately generic: operates on raw float buffers, not on your Tensor type,
// so it carries no opinion about how you design shape/strides/allocation.

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

inline void dump_binary(const std::string& path, const float* data, size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("dump_binary: could not open " + path);
    size_t written = std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
    if (written != n) throw std::runtime_error("dump_binary: short write to " + path);
}

inline std::vector<float> load_binary(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("load_binary: could not open " + path);
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
    size_t read = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    if (read != out.size()) throw std::runtime_error("load_binary: short read from " + path);
    return out;
}
