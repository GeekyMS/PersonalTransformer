// Port-protocol test binary for block's backward closures (roadmap 5.4 /
// test/README.md). Seeds a fixed upstream gradient dOut into out.grad, runs
// block forward + tape_backward, and dumps concat(x.grad, ln1_g.grad,
// ln1_b.grad, Wq.grad, Wk.grad, Wv.grad, Wo.grad, ln2_g.grad, ln2_b.grad,
// W1.grad, b1.grad, W2.grad, b2.grad) for compare.py to diff against
// np_impl.ops_ref.block_grads (same order -- see _BLOCK_PARAM_NAMES there).
#include "arena.h"
#include "dump.h"
#include "model.h"
#include "tape.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data     = load_binary(dump_dir + "/block_backward_input_x.bin");
    std::vector<float> ln1_g_data = load_binary(dump_dir + "/block_backward_input_ln1_g.bin");
    std::vector<float> ln1_b_data = load_binary(dump_dir + "/block_backward_input_ln1_b.bin");
    std::vector<float> Wq_data    = load_binary(dump_dir + "/block_backward_input_Wq.bin");
    std::vector<float> Wk_data    = load_binary(dump_dir + "/block_backward_input_Wk.bin");
    std::vector<float> Wv_data    = load_binary(dump_dir + "/block_backward_input_Wv.bin");
    std::vector<float> Wo_data    = load_binary(dump_dir + "/block_backward_input_Wo.bin");
    std::vector<float> ln2_g_data = load_binary(dump_dir + "/block_backward_input_ln2_g.bin");
    std::vector<float> ln2_b_data = load_binary(dump_dir + "/block_backward_input_ln2_b.bin");
    std::vector<float> W1_data    = load_binary(dump_dir + "/block_backward_input_W1.bin");
    std::vector<float> b1_data    = load_binary(dump_dir + "/block_backward_input_b1.bin");
    std::vector<float> W2_data    = load_binary(dump_dir + "/block_backward_input_W2.bin");
    std::vector<float> b2_data    = load_binary(dump_dir + "/block_backward_input_b2.bin");
    std::vector<float> dout_data  = load_binary(dump_dir + "/block_backward_input_dOut.bin");

    std::vector<float> x_grad(x_data.size(), 0.0f);
    std::vector<float> ln1_g_grad(ln1_g_data.size(), 0.0f);
    std::vector<float> ln1_b_grad(ln1_b_data.size(), 0.0f);
    std::vector<float> Wq_grad(Wq_data.size(), 0.0f);
    std::vector<float> Wk_grad(Wk_data.size(), 0.0f);
    std::vector<float> Wv_grad(Wv_data.size(), 0.0f);
    std::vector<float> Wo_grad(Wo_data.size(), 0.0f);
    std::vector<float> ln2_g_grad(ln2_g_data.size(), 0.0f);
    std::vector<float> ln2_b_grad(ln2_b_data.size(), 0.0f);
    std::vector<float> W1_grad(W1_data.size(), 0.0f);
    std::vector<float> b1_grad(b1_data.size(), 0.0f);
    std::vector<float> W2_grad(W2_data.size(), 0.0f);
    std::vector<float> b2_grad(b2_data.size(), 0.0f);
    std::vector<float> out_data(dout_data.size());

    Tensor x(x_data.data(), {2, 3, 4});
    x.grad = x_grad.data();

    Tensor ln1_g(ln1_g_data.data(), {4}); ln1_g.grad = ln1_g_grad.data();
    Tensor ln1_b(ln1_b_data.data(), {4}); ln1_b.grad = ln1_b_grad.data();
    Tensor Wq(Wq_data.data(), {4, 4});    Wq.grad = Wq_grad.data();
    Tensor Wk(Wk_data.data(), {4, 4});    Wk.grad = Wk_grad.data();
    Tensor Wv(Wv_data.data(), {4, 4});    Wv.grad = Wv_grad.data();
    Tensor Wo(Wo_data.data(), {4, 4});    Wo.grad = Wo_grad.data();
    Tensor ln2_g(ln2_g_data.data(), {4}); ln2_g.grad = ln2_g_grad.data();
    Tensor ln2_b(ln2_b_data.data(), {4}); ln2_b.grad = ln2_b_grad.data();
    Tensor W1(W1_data.data(), {4, 6});    W1.grad = W1_grad.data();
    Tensor b1(b1_data.data(), {6});       b1.grad = b1_grad.data();
    Tensor W2(W2_data.data(), {6, 4});    W2.grad = W2_grad.data();
    Tensor b2(b2_data.data(), {4});       b2.grad = b2_grad.data();

    BlockParams p{ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2};

    Tensor out(out_data.data(), {2, 3, 4});
    out.grad = dout_data.data();

    Arena arena(8192);
    tape_reset();
    block(arena, x, p, out);
    tape_backward();

    std::vector<float> result;
    auto append = [&](std::vector<float>& g) { result.insert(result.end(), g.begin(), g.end()); };
    append(x_grad);
    append(ln1_g_grad); append(ln1_b_grad);
    append(Wq_grad); append(Wk_grad); append(Wv_grad); append(Wo_grad);
    append(ln2_g_grad); append(ln2_b_grad);
    append(W1_grad); append(b1_grad); append(W2_grad); append(b2_grad);

    dump_binary(dump_dir + "/block_backward_output.bin", result.data(), result.size());
    return 0;
}
