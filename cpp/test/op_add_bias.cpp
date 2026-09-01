// Port-protocol test binary for add_bias (roadmap 5.4 / test/README.md).
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data = load_binary(dump_dir + "/add_bias_input_x.bin");
    std::vector<float> b_data = load_binary(dump_dir + "/add_bias_input_b.bin");

    Tensor x(x_data.data(), {5, 4});
    Tensor b(b_data.data(), {4});

    std::vector<float> out_data(5 * 4);
    Tensor out(out_data.data(), {5, 4});

    add_bias(x, b, out);

    dump_binary(dump_dir + "/add_bias_output.bin", out_data.data(), out_data.size());
    return 0;
}
