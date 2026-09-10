// Port-protocol test binary for block forward (roadmap 5.4 / test/README.md).
// H=4 (block() calls attention() with kH from constants.h, hardcoded, not a
// parameter), d=4 so d_head=1.
#include "arena.h"
#include "dump.h"
#include "model.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data     = load_binary(dump_dir + "/block_input_x.bin");
    std::vector<float> ln1_g_data = load_binary(dump_dir + "/block_input_ln1_g.bin");
    std::vector<float> ln1_b_data = load_binary(dump_dir + "/block_input_ln1_b.bin");
    std::vector<float> Wq_data    = load_binary(dump_dir + "/block_input_Wq.bin");
    std::vector<float> Wk_data    = load_binary(dump_dir + "/block_input_Wk.bin");
    std::vector<float> Wv_data    = load_binary(dump_dir + "/block_input_Wv.bin");
    std::vector<float> Wo_data    = load_binary(dump_dir + "/block_input_Wo.bin");
    std::vector<float> ln2_g_data = load_binary(dump_dir + "/block_input_ln2_g.bin");
    std::vector<float> ln2_b_data = load_binary(dump_dir + "/block_input_ln2_b.bin");
    std::vector<float> W1_data    = load_binary(dump_dir + "/block_input_W1.bin");
    std::vector<float> b1_data    = load_binary(dump_dir + "/block_input_b1.bin");
    std::vector<float> W2_data    = load_binary(dump_dir + "/block_input_W2.bin");
    std::vector<float> b2_data    = load_binary(dump_dir + "/block_input_b2.bin");

    Tensor x(x_data.data(), {2, 3, 4});

    Tensor ln1_g(ln1_g_data.data(), {4});
    Tensor ln1_b(ln1_b_data.data(), {4});
    Tensor Wq(Wq_data.data(), {4, 4});
    Tensor Wk(Wk_data.data(), {4, 4});
    Tensor Wv(Wv_data.data(), {4, 4});
    Tensor Wo(Wo_data.data(), {4, 4});
    Tensor ln2_g(ln2_g_data.data(), {4});
    Tensor ln2_b(ln2_b_data.data(), {4});
    Tensor W1(W1_data.data(), {4, 6});
    Tensor b1(b1_data.data(), {6});
    Tensor W2(W2_data.data(), {6, 4});
    Tensor b2(b2_data.data(), {4});

    BlockParams p{ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2};

    std::vector<float> out_data(2 * 3 * 4);
    Tensor out(out_data.data(), {2, 3, 4});

    Arena arena(8192);
    block(arena, x, p, out);

    dump_binary(dump_dir + "/block_output.bin", out_data.data(), out_data.size());
    return 0;
}
