#include "ops.h"
#include "tape.h"

#include <cmath>
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

void embed(const std::vector<int>& x, const Tensor& tok_emb, const Tensor& pos_emb, Tensor& out) {
    int B = out.shape[0];
    int T = out.shape[1];
    int d = out.shape[2];

    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            for(int k = 0; k < d; k++){
                out.at({b, t, k}) = tok_emb.at({x[b * T + t], k}) + pos_emb.at({t, k});
            }
        }
    }

    tape_push([x, tok_emb, pos_emb, out]() {
        int B = out.shape[0];
        int T = out.shape[1];
        int d = out.shape[2];

        for(int b = 0; b < B; b++){
            for(int t = 0; t < T; t++){
                for(int k = 0; k < d; k++){
                    tok_emb.grad_at({x[b * T + t], k}) += out.grad_at({b, t, k});
                    pos_emb.grad_at({t, k}) += out.grad_at({b, t, k});
                }
            }
        }
    });
}

void layer_norm(const Tensor& x, const Tensor& g, const Tensor& b, Tensor& out, float eps) {
    int N = x.shape[0];
    int d = x.shape[1];

    for(int i = 0; i < N; i++){
        float mu = 0.0f;
        for(int k = 0; k < d; k++){
            mu += x.at({i, k});
        }
        mu /= d;

        float var = 0.0f;
        for(int k = 0; k < d; k++){
            float diff = x.at({i, k}) - mu;
            var += diff * diff;
        }
        var /= d;

        float sigma = std::sqrt(var + eps);

        for(int k = 0; k < d; k++){
            float xhat = (x.at({i, k}) - mu) / sigma;
            out.at({i, k}) = g.at({k}) * xhat + b.at({k});
        }
    }

    tape_push([x, g, b, out, eps]() {
        int N = x.shape[0];
        int d = x.shape[1];

        for(int i = 0; i < N; i++){
            // Recompute this row's forward quantities rather than caching
            // them — cheap (O(d)) relative to the backward math below, and
            // avoids threading a separate cache tensor through the tape.
            float mu = 0.0f;
            for(int k = 0; k < d; k++){
                mu += x.at({i, k});
            }
            mu /= d;

            float var = 0.0f;
            for(int k = 0; k < d; k++){
                float diff = x.at({i, k}) - mu;
                var += diff * diff;
            }
            var /= d;
            float sigma = std::sqrt(var + eps);

            std::vector<float> xhat(d), dxhat(d);
            for(int k = 0; k < d; k++){
                xhat[k] = (x.at({i, k}) - mu) / sigma;
                dxhat[k] = out.grad_at({i, k}) * g.at({k});
            }

            float mean_dxhat = 0.0f, mean_dxhat_xhat = 0.0f;
            for(int k = 0; k < d; k++){
                mean_dxhat += dxhat[k];
                mean_dxhat_xhat += dxhat[k] * xhat[k];
            }
            mean_dxhat /= d;
            mean_dxhat_xhat /= d;

            for(int k = 0; k < d; k++){
                float dx = (dxhat[k] - mean_dxhat - xhat[k] * mean_dxhat_xhat) / sigma;
                x.grad_at({i, k}) += dx;
                g.grad_at({k}) += out.grad_at({i, k}) * xhat[k];
                b.grad_at({k}) += out.grad_at({i, k});
            }
        }
    });
}

void add_bias(const Tensor& x, const Tensor& b, Tensor& out) {
    int N = x.shape[0];
    int d = x.shape[1];

    for(int i = 0; i < N; i++){
        for(int k = 0; k < d; k++){
            out.at({i, k}) = x.at({i, k}) + b.at({k});
        }
    }

    tape_push([x, b, out]() {
        int N = x.shape[0];
        int d = x.shape[1];

        for(int i = 0; i < N; i++){
            for(int k = 0; k < d; k++){
                x.grad_at({i, k}) += out.grad_at({i, k});
                b.grad_at({k}) += out.grad_at({i, k});
            }
        }
    });
}

void gelu(const Tensor& x, Tensor& out) {
    const float c = std::sqrt(2.0f / (float)M_PI);
    int N = x.shape[0];
    int d = x.shape[1];

    for(int i = 0; i < N; i++){
        for(int k = 0; k < d; k++){
            float v = x.at({i, k});
            float u = c * (v + 0.044715f * v * v * v);
            out.at({i, k}) = 0.5f * v * (1.0f + std::tanh(u));
        }
    }

    tape_push([x, out, c]() {
        int N = x.shape[0];
        int d = x.shape[1];

        for(int i = 0; i < N; i++){
            for(int k = 0; k < d; k++){
                float v = x.at({i, k});
                float u = c * (v + 0.044715f * v * v * v);
                float t = std::tanh(u);
                float du_dv = c * (1.0f + 3.0f * 0.044715f * v * v);
                float dgelu_dv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du_dv;
                x.grad_at({i, k}) += out.grad_at({i, k}) * dgelu_dv;
            }
        }
    });
}

void mlp(Arena& arena, const Tensor& x, const Tensor& W1, const Tensor& b1,
          const Tensor& W2, const Tensor& b2, Tensor& out) {
    int N = x.shape[0];
    int d4 = W1.shape[1];

    Tensor h1  = make_tensor(arena, {N, d4});   // x @ W1
    Tensor h1b = make_tensor(arena, {N, d4});   // h1 + b1
    Tensor h   = make_tensor(arena, {N, d4});   // gelu(h1b)
    Tensor h2  = make_tensor(arena, {N, out.shape[1]});  // h @ W2

    matmul(x, W1, h1);
    add_bias(h1, b1, h1b);
    gelu(h1b, h);
    matmul(h, W2, h2);
    add_bias(h2, b2, out);
}
