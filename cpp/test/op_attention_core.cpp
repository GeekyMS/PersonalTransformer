// Port-protocol test binary for attention_core forward (roadmap 5.4 /
// test/README.md). B=2 (not 1) deliberately: attention_core loops over the
// batch internally, so this also exercises that loop, not just the math for
// a single sequence.
#include "arena.h"
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_data  = load_binary(dump_dir + "/attention_core_input_x.bin");
    std::vector<float> Wq_data = load_binary(dump_dir + "/attention_core_input_Wq.bin");
    std::vector<float> Wk_data = load_binary(dump_dir + "/attention_core_input_Wk.bin");
    std::vector<float> Wv_data = load_binary(dump_dir + "/attention_core_input_Wv.bin");

    Tensor x(x_data.data(), {2, 3, 4});
    Tensor Wq(Wq_data.data(), {4, 4});
    Tensor Wk(Wk_data.data(), {4, 4});
    Tensor Wv(Wv_data.data(), {4, 4});

    std::vector<float> out_data(2 * 3 * 4);
    Tensor out(out_data.data(), {2, 3, 4});

    Arena arena(4096);
    attention_core(arena, x, Wk, Wq, Wv, out);

    dump_binary(dump_dir + "/attention_core_output.bin", out_data.data(), out_data.size());
    return 0;
}
