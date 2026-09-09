// Port-protocol test binary for softmax (roadmap 5.4 / test/README.md).
// Shape is rank-3 (not 2D) deliberately: exercises the reshape-to-(rows,d)
// path for arbitrary input rank, not just the trivial 2D case.
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> s_data = load_binary(dump_dir + "/softmax_input_S.bin");
    Tensor S(s_data.data(), {3, 4, 5});

    std::vector<float> out_data(3 * 4 * 5);
    Tensor out(out_data.data(), {3, 4, 5});

    softmax(S, out);

    dump_binary(dump_dir + "/softmax_output.bin", out_data.data(), out_data.size());
    return 0;
}
