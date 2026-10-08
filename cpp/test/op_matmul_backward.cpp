// Checks forward, both gradients, and accumulation through the same 64
// combinations of contiguous, padded, transposed, and strided views.
#include "dump.h"
#include "ops.h"
#include "tape.h"
#include "tensor.h"

#include <limits>
#include <stdexcept>

static Tensor make_view(std::vector<float>& data, std::vector<float>& grad,
                        int rows, int cols, int layout) {
    std::vector<int> strides{cols, 1};
    if(layout == 1){
        strides = {cols + 3, 1};
    }else if(layout == 2){
        strides = {1, rows + 3};
    }else if(layout == 3){
        strides = {2 * cols + 3, 2};
    }

    int offset = 5;
    int size = offset + (rows - 1) * strides[0] + (cols - 1) * strides[1] + 6;
    data.resize(size, -1234.5f);
    grad.resize(size, -1234.5f);
    Tensor x(data.data(), {rows, cols}, strides, offset);
    x.grad = grad.data();
    return x;
}

static void check_padding(const Tensor& x, const std::vector<float>& buffer) {
    std::vector<bool> used(buffer.size(), false);
    for(int i = 0; i < x.shape[0]; i++){
        for(int j = 0; j < x.shape[1]; j++){
            used[x.offset + i * x.strides[0] + j * x.strides[1]] = true;
        }
    }
    for(int i = 0; i < (int)buffer.size(); i++){
        if(!used[i] && buffer[i] != -1234.5f){
            throw std::runtime_error("matmul: wrote outside the tensor view");
        }
    }
}

static void append_grad(std::vector<float>& result, const Tensor& x) {
    for(int i = 0; i < x.shape[0]; i++){
        for(int j = 0; j < x.shape[1]; j++){
            result.push_back(x.grad_at(std::array<int, 2>{i, j}));
        }
    }
}

int main() {
    const std::string dump_dir = "cpp/test/dumps";
    std::vector<float> input_A = load_binary(dump_dir + "/matmul_backward_input_A.bin");
    std::vector<float> input_B = load_binary(dump_dir + "/matmul_backward_input_B.bin");
    std::vector<float> input_dOut = load_binary(dump_dir + "/matmul_backward_input_dOut.bin");
    std::vector<float> input_dA = load_binary(dump_dir + "/matmul_backward_input_dA.bin");
    std::vector<float> input_dB = load_binary(dump_dir + "/matmul_backward_input_dB.bin");
    std::vector<float> result;

    for(int a_layout = 0; a_layout < 4; a_layout++){
        for(int b_layout = 0; b_layout < 4; b_layout++){
            for(int out_layout = 0; out_layout < 4; out_layout++){
                std::vector<float> a_data, a_grad, b_data, b_grad, out_data, out_grad;
                Tensor A = make_view(a_data, a_grad, 8, 16, a_layout);
                Tensor B = make_view(b_data, b_grad, 16, 4, b_layout);
                Tensor out = make_view(out_data, out_grad, 8, 4, out_layout);
                for(int i = 0; i < 8; i++){
                    for(int k = 0; k < 16; k++){
                        A.at(std::array<int, 2>{i, k}) = input_A[i * 16 + k];
                        A.grad_at(std::array<int, 2>{i, k}) = input_dA[i * 16 + k];
                    }
                }
                for(int k = 0; k < 16; k++){
                    for(int j = 0; j < 4; j++){
                        B.at(std::array<int, 2>{k, j}) = input_B[k * 4 + j];
                        B.grad_at(std::array<int, 2>{k, j}) = input_dB[k * 4 + j];
                    }
                }
                for(int i = 0; i < 8; i++){
                    for(int j = 0; j < 4; j++){
                        out.at(std::array<int, 2>{i, j}) = std::numeric_limits<float>::quiet_NaN();
                        out.grad_at(std::array<int, 2>{i, j}) = input_dOut[i * 4 + j];
                    }
                }

                tape_reset();
                matmul(A, B, out);
                tape_backward();
                if(tape.size() != 1){
                    throw std::runtime_error("matmul: backward recorded more tape nodes");
                }
                for(int i = 0; i < 8; i++){
                    for(int j = 0; j < 4; j++){
                        result.push_back(out.at(std::array<int, 2>{i, j}));
                    }
                }
                append_grad(result, A);
                append_grad(result, B);

                // Same upstream gradient again must add to the existing grads.
                tape_backward();
                append_grad(result, A);
                append_grad(result, B);
                check_padding(A, a_data);
                check_padding(A, a_grad);
                check_padding(B, b_data);
                check_padding(B, b_grad);
                check_padding(out, out_data);
                check_padding(out, out_grad);
            }
        }
    }
    tape_reset();
    dump_binary(dump_dir + "/matmul_backward_output.bin", result.data(), result.size());
    return 0;
}
