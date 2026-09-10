// Port-protocol test binary for attention (multi-head) forward (roadmap
// 5.4 / test/README.md). H=2 fixed, matching np_impl.ops_ref.attention_h2.
// B=2, d=4 (so d_head=2): small enough to hand-verify, big enough to
// exercise both the batch loop and the head loop.
#include "arena.h"
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data  = load_binary(dump_dir + "/attention_input_x.bin");
    std::vector<float> Wq_data = load_binary(dump_dir + "/attention_input_Wq.bin");
    std::vector<float> Wk_data = load_binary(dump_dir + "/attention_input_Wk.bin");
    std::vector<float> Wv_data = load_binary(dump_dir + "/attention_input_Wv.bin");
    std::vector<float> Wo_data = load_binary(dump_dir + "/attention_input_Wo.bin");

    Tensor x(x_data.data(), {2, 3, 4});
    Tensor Wq(Wq_data.data(), {4, 4});
    Tensor Wk(Wk_data.data(), {4, 4});
    Tensor Wv(Wv_data.data(), {4, 4});
    Tensor Wo(Wo_data.data(), {4, 4});

    std::vector<float> out_data(2 * 3 * 4);
    Tensor out(out_data.data(), {2, 3, 4});

    Arena arena(4096);
    attention(arena, x, Wk, Wq, Wv, Wo, 2, out);

    dump_binary(dump_dir + "/attention_output.bin", out_data.data(), out_data.size());
    return 0;
}
