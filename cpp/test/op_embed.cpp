// Port-protocol test binary for embed (roadmap 5.4 / test/README.md).
// Loads x, tok_emb, pos_emb; runs embed; dumps the output for compare.py to
// diff against np_impl/ops_ref.py.
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> x_float = load_binary(dump_dir + "/embed_input_x.bin");
    std::vector<int> x(x_float.begin(), x_float.end());

    std::vector<float> tok_emb_data = load_binary(dump_dir + "/embed_input_tok_emb.bin");
    std::vector<float> pos_emb_data = load_binary(dump_dir + "/embed_input_pos_emb.bin");

    Tensor tok_emb(tok_emb_data.data(), {6, 4});
    Tensor pos_emb(pos_emb_data.data(), {3, 4});

    std::vector<float> out_data(2 * 3 * 4);
    Tensor out(out_data.data(), {2, 3, 4});

    embed(x, tok_emb, pos_emb, out);

    dump_binary(dump_dir + "/embed_output.bin", out_data.data(), out_data.size());
    return 0;
}
