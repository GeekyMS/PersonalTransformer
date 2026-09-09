// Port-protocol test binary for attention_core's backward closures (roadmap
// 5.4 / test/README.md). Seeds a fixed upstream gradient dOut into out.grad,
// runs attention_core forward + tape_backward, and dumps
// concat(x.grad, Wq.grad, Wk.grad, Wv.grad) for compare.py to diff against
// np_impl.ops_ref.attention_core_grads (same concatenation order).
#include "arena.h"
#include "dump.h"
#include "ops.h"
#include "tape.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data    = load_binary(dump_dir + "/attention_core_backward_input_x.bin");
    std::vector<float> Wq_data   = load_binary(dump_dir + "/attention_core_backward_input_Wq.bin");
    std::vector<float> Wk_data   = load_binary(dump_dir + "/attention_core_backward_input_Wk.bin");
    std::vector<float> Wv_data   = load_binary(dump_dir + "/attention_core_backward_input_Wv.bin");
    std::vector<float> dout_data = load_binary(dump_dir + "/attention_core_backward_input_dOut.bin");

    std::vector<float> x_grad(x_data.size(), 0.0f);
    std::vector<float> Wq_grad(Wq_data.size(), 0.0f);
    std::vector<float> Wk_grad(Wk_data.size(), 0.0f);
    std::vector<float> Wv_grad(Wv_data.size(), 0.0f);
    std::vector<float> out_data(dout_data.size());

    Tensor x(x_data.data(), {2, 3, 4});
    x.grad = x_grad.data();
    Tensor Wq(Wq_data.data(), {4, 4});
    Wq.grad = Wq_grad.data();
    Tensor Wk(Wk_data.data(), {4, 4});
    Wk.grad = Wk_grad.data();
    Tensor Wv(Wv_data.data(), {4, 4});
    Wv.grad = Wv_grad.data();

    Tensor out(out_data.data(), {2, 3, 4});
    out.grad = dout_data.data();

    Arena arena(4096);
    tape_reset();
    attention_core(arena, x, Wk, Wq, Wv, out);
    tape_backward();

    std::vector<float> result;
    result.insert(result.end(), x_grad.begin(), x_grad.end());
    result.insert(result.end(), Wq_grad.begin(), Wq_grad.end());
    result.insert(result.end(), Wk_grad.begin(), Wk_grad.end());
    result.insert(result.end(), Wv_grad.begin(), Wv_grad.end());

    dump_binary(dump_dir + "/attention_core_backward_output.bin", result.data(), result.size());
    return 0;
}
