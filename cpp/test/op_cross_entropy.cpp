// Port-protocol test binary for cross_entropy forward (roadmap 5.4 /
// test/README.md). N=6, V=5.
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> logits_data = load_binary(dump_dir + "/cross_entropy_input_logits.bin");
    std::vector<float> y_data = load_binary(dump_dir + "/cross_entropy_input_y.bin");

    Tensor logits(logits_data.data(), {6, 5});
    std::vector<int> targets(y_data.size());
    for (size_t i = 0; i < y_data.size(); i++) targets[i] = (int)y_data[i];

    std::vector<float> out_data(1);
    Tensor out(out_data.data(), {1});

    cross_entropy(logits, targets, out);

    dump_binary(dump_dir + "/cross_entropy_output.bin", out_data.data(), out_data.size());
    return 0;
}
