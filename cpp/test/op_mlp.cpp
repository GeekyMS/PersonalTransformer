// Port-protocol test binary for mlp (roadmap 5.4 / test/README.md).
#include "arena.h"
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data  = load_binary(dump_dir + "/mlp_input_x.bin");
    std::vector<float> W1_data = load_binary(dump_dir + "/mlp_input_W1.bin");
    std::vector<float> b1_data = load_binary(dump_dir + "/mlp_input_b1.bin");
    std::vector<float> W2_data = load_binary(dump_dir + "/mlp_input_W2.bin");
    std::vector<float> b2_data = load_binary(dump_dir + "/mlp_input_b2.bin");

    Tensor x(x_data.data(), {5, 4});
    Tensor W1(W1_data.data(), {4, 16});
    Tensor b1(b1_data.data(), {16});
    Tensor W2(W2_data.data(), {16, 4});
    Tensor b2(b2_data.data(), {4});

    std::vector<float> out_data(5 * 4);
    Tensor out(out_data.data(), {5, 4});

    Arena arena(2000);
    mlp(arena, x, W1, b1, W2, b2, out);

    dump_binary(dump_dir + "/mlp_output.bin", out_data.data(), out_data.size());
    return 0;
}
