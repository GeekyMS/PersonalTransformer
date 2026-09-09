// Port-protocol test binary for softmax's backward closure (roadmap 5.4 /
// test/README.md). Seeds a fixed upstream gradient dP into out.grad, runs
// softmax forward + the tape backward, and dumps S.grad (dS) for compare.py
// to diff against np_impl.backward.softmax_vjp (already grad-checked there).
#include "dump.h"
#include "ops.h"
#include "tape.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> s_data = load_binary(dump_dir + "/softmax_backward_input_S.bin");
    std::vector<float> dp_data = load_binary(dump_dir + "/softmax_backward_input_dP.bin");

    std::vector<float> s_grad(s_data.size(), 0.0f);
    std::vector<float> out_data(s_data.size());

    Tensor S(s_data.data(), {3, 4, 5});
    S.grad = s_grad.data();

    Tensor out(out_data.data(), {3, 4, 5});
    out.grad = dp_data.data();

    tape_reset();
    softmax(S, out);
    tape_backward();

    dump_binary(dump_dir + "/softmax_backward_output.bin", s_grad.data(), s_grad.size());
    return 0;
}
