// Port-protocol test binary for cross_entropy's backward closure (roadmap
// 5.4 / test/README.md). cross_entropy is the terminal op of the graph, so
// unlike every other backward test here, there's no upstream dOut to seed --
// its own closure treats dL/dL = 1 implicitly (see np_impl/backward.py's
// cross_entropy_backward, which likewise takes no incoming-gradient param).
#include "dump.h"
#include "ops.h"
#include "tape.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> logits_data = load_binary(dump_dir + "/cross_entropy_backward_input_logits.bin");
    std::vector<float> y_data = load_binary(dump_dir + "/cross_entropy_backward_input_y.bin");

    std::vector<int> targets(y_data.size());
    for (size_t i = 0; i < y_data.size(); i++) targets[i] = (int)y_data[i];

    std::vector<float> logits_grad(logits_data.size(), 0.0f);
    Tensor logits(logits_data.data(), {6, 5});
    logits.grad = logits_grad.data();

    std::vector<float> out_data(1);
    Tensor out(out_data.data(), {1});

    tape_reset();
    cross_entropy(logits, targets, out);
    tape_backward();

    dump_binary(dump_dir + "/cross_entropy_backward_output.bin", logits_grad.data(), logits_grad.size());
    return 0;
}
