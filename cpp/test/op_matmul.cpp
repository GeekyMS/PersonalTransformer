// Port-protocol test binary for matmul (roadmap 5.4 / test/README.md).
// Loads A, B; runs matmul; dumps the output for compare.py to diff against
// np_impl/ops_ref.py.
#include "dump.h"
#include "ops.h"
#include "tensor.h"

int main() {
    const std::string dump_dir = "cpp/test/dumps";

    std::vector<float> a_data = load_binary(dump_dir + "/matmul_input_A.bin");
    std::vector<float> b_data = load_binary(dump_dir + "/matmul_input_B.bin");

    Tensor A(a_data.data(), {8, 16});
    Tensor B(b_data.data(), {16, 4});

    std::vector<float> out_data(8 * 4);
    Tensor out(out_data.data(), {8, 4});

    matmul(A, B, out);

    dump_binary(dump_dir + "/matmul_output.bin", out_data.data(), out_data.size());
    return 0;
}
