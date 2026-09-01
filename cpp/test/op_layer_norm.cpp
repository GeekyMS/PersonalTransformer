// Port-protocol test binary for layer_norm (roadmap 5.4 / test/README.md).
// Loads x, g, b; runs layer_norm; dumps the output for compare.py to diff
// against np_impl/ops_ref.py.
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data = load_binary(dump_dir + "/layer_norm_input_x.bin");
    std::vector<float> g_data = load_binary(dump_dir + "/layer_norm_input_g.bin");
    std::vector<float> b_data = load_binary(dump_dir + "/layer_norm_input_b.bin");

    Tensor x(x_data.data(), {5, 4});
    Tensor g(g_data.data(), {4});
    Tensor b(b_data.data(), {4});

    std::vector<float> out_data(5 * 4);
    Tensor out(out_data.data(), {5, 4});

    layer_norm(x, g, b, out);

    dump_binary(dump_dir + "/layer_norm_output.bin", out_data.data(), out_data.size());
    return 0;
}
