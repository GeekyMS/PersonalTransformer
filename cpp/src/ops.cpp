#include "ops.h"
#include "tape.h"

#include <stdexcept>

void matmul(const Tensor& A, const Tensor& B, Tensor& out) {
    if(A.shape[1] != B.shape[0]){
        throw std::runtime_error("matmul: shapes do not match");
    }
    
    for(int i = 0; i < A.shape[0]; i++){
        for(int j = 0; j < B.shape[1]; j++){
            float temp = 0.0f;
            for(int k = 0; k < B.shape[0]; k++){
                temp += A.at({i, k}) * B.at({k, j});
            }
            out.at({i, j}) = temp;
        }
    }

    tape_push([A, B, out]() {
        for(int i = 0; i < (int)A.shape[0]; i++){
            for(int k = 0; k < (int)A.shape[1]; k++){
                float dA = A.grad_at({i, k});
                for(int j = 0; j < (int)B.shape[1]; j++){
                    dA += out.grad_at({i, j}) * B.at({k, j});
                }
                A.grad_at({i, k}) = dA;
            }
        }

        for(int k = 0; k < (int)B.shape[0]; k++){
            for(int j = 0; j < (int)B.shape[1]; j++){
                float dB = B.grad_at({k, j});
                for(int i = 0; i < (int)A.shape[0]; i++){
                    dB += out.grad_at({i, j}) * A.at({i, k});
                }
                B.grad_at({k, j}) = dB;
            }
        }
    });

}
