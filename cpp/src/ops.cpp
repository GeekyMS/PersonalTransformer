#include "ops.h"
#include "tape.h"

#include <cmath>
#include <stdexcept>
#include <limits>

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

void softmax(const Tensor& S, Tensor& out){
    int totalRows = 1;
    for(int k = 0; k < (int)S.shape.size() - 1; k++){
        totalRows *= S.shape[k];
    }
    std::vector<int> newShape = {totalRows, S.shape[S.shape.size() - 1]};
    Tensor v = S.reshape(newShape);
    Tensor newOut = out.reshape(newShape);
    for(int i = 0; i < v.shape[0]; i++){
        float maxx = -std::numeric_limits<float>::infinity();
        for(int j = 0; j < v.shape[1]; j++){
            maxx = std::max(maxx, v.at({i, j}));
        }

        for(int j = 0; j < v.shape[1]; j++){
            newOut.at({i, j}) = v.at({i, j}) - maxx;
        }
        float total = 0.0f;
        for(int j = 0; j < v.shape[1]; j++){
            newOut.at({i, j}) = std::exp(newOut.at({i, j}));
            total += newOut.at({i, j});
        }

        for(int j = 0; j < v.shape[1]; j++){
            newOut.at({i, j}) /= total;
        }
    }

    tape_push([newOut, v]() {
        for(int i = 0; i < newOut.shape[0]; i++){
            float total = 0.0f;
            for(int j = 0; j < newOut.shape[1]; j++){
                total += newOut.at({i, j}) * newOut.grad_at({i, j});
            }

            for(int j = 0; j < newOut.shape[1]; j++){
                float P = newOut.at({i, j});
                float dP = newOut.grad_at({i, j});
                v.grad_at({i, j}) += P * dP - P * total;
            }
        }
    });
}

void attention_core(Arena& arena, Tensor& x, Tensor& Wk, Tensor& Wq, Tensor& Wv, Tensor& out){
    int B = x.shape[0];

    for(int b = 0; b < B; b++){
        Tensor xb = x.slice({b});
        Tensor outb = out.slice({b});

        Tensor Q = make_tensor(arena, {x.shape[1], Wq.shape[1]});
        Tensor K = make_tensor(arena, {x.shape[1], Wk.shape[1]});
        Tensor V = make_tensor(arena, {x.shape[1], Wv.shape[1]});
        std::vector newShape({x.shape[1], x.shape[1]});
        Tensor QKT = make_tensor(arena, newShape);
        Tensor S = make_tensor(arena, QKT.shape);
        
        matmul(xb, Wq, Q);
        matmul(xb, Wk, K);
        matmul(xb, Wv, V);

        Tensor KT = K.transpose(K.shape.size() - 1, K.shape.size() - 2);

        matmul(Q, KT, QKT);


        for(int i = 0; i < QKT.shape[0]; i++){
            for(int j = 0; j < QKT.shape[1]; j++){
                    S.at({i, j}) = QKT.at({i, j}) / std::sqrt(x.shape[x.shape.size() - 1]);
            }
        }

        tape_push([S, QKT, x](){
            for(int i = 0; i < QKT.shape[0]; i++){
                for(int j = 0; j < QKT.shape[1]; j++){
                        QKT.grad_at({i, j}) += S.grad_at({i, j}) / std::sqrt(x.shape[x.shape.size() - 1]);
                }
            }
        });

        Tensor P = make_tensor(arena, S.shape);

        for(int i = 0; i < S.shape[0]; i++){
            for(int j = i + 1; j < S.shape[1]; j++){
                S.at({i, j}) = -std::numeric_limits<float>::infinity();
            }
        }

        tape_push([S](){
            for(int i = 0; i < S.shape[0]; i++){
                for(int j = 0; j < S.shape[1]; j++){
                    if(j > i){
                        S.grad_at({i, j}) = 0;
                    }
                }
            }
        });

        softmax(S, P);

        matmul(P, V, outb);
    }

}

