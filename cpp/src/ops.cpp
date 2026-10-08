#include "ops.h"
#include "tape.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <limits>

#ifdef USE_BLAS
#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// BLAS can read row-major views and transposes directly. Pack only when
// neither axis is contiguous, keeping offsets and padded row strides intact.
static const float* blas_input(const Tensor& x, std::vector<float>& buffer,
                              CBLAS_TRANSPOSE& trans, int& ld) {
    if(x.strides[1] == 1 && x.strides[0] >= x.shape[1]){
        trans = CblasNoTrans;
        ld = x.strides[0];
        return x.data + x.offset;
    }
    if(x.strides[0] == 1 && x.strides[1] >= x.shape[0]){
        trans = CblasTrans;
        ld = x.strides[1];
        return x.data + x.offset;
    }

    buffer.resize((size_t)x.shape[0] * x.shape[1]);
    for(int i = 0; i < x.shape[0]; i++){
        for(int j = 0; j < x.shape[1]; j++){
            buffer[i * x.shape[1] + j] = x.at(std::array<int, 2>{i, j});
        }
    }
    trans = CblasNoTrans;
    ld = x.shape[1];
    return buffer.data();
}

// Only arithmetic -- backward calls this without recording more tape nodes.
// beta=0 overwrites forward output; beta=1 accumulates into existing grads.
static void matmul_blas(const Tensor& A, const Tensor& B, const Tensor& out, float beta) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    if(M == 0 || N == 0){
        return;
    }
    if(K == 0){
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                out.at(std::array<int, 2>{i, j}) = beta == 0.0f ? 0.0f : beta * out.at(std::array<int, 2>{i, j});
            }
        }
        return;
    }

    bool row_major = out.strides[1] == 1 && out.strides[0] >= N;
    if(!row_major && out.strides[0] == 1 && out.strides[1] >= M){
        // Write a transposed output directly: (A @ B)^T = B^T @ A^T.
        matmul_blas(B.transpose(0, 1), A.transpose(0, 1), out.transpose(0, 1), beta);
        return;
    }

    std::vector<float> a_buffer;
    std::vector<float> b_buffer;
    std::vector<float> out_buffer;
    CBLAS_TRANSPOSE transA, transB;
    int lda, ldb;
    const float* a_data = blas_input(A, a_buffer, transA, lda);
    const float* b_data = blas_input(B, b_buffer, transB, ldb);
    float* out_data = out.data + out.offset;
    int ldc = out.strides[0];
    if(!row_major){
        out_buffer.resize((size_t)M * N);
        if(beta != 0.0f){
            for(int i = 0; i < M; i++){
                for(int j = 0; j < N; j++){
                    out_buffer[i * N + j] = out.at(std::array<int, 2>{i, j});
                }
            }
        }
        out_data = out_buffer.data();
        ldc = N;
    }

    cblas_sgemm(CblasRowMajor, transA, transB, M, N, K,
                1.0f, a_data, lda, b_data, ldb, beta, out_data, ldc);

    if(!row_major){
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                out.at(std::array<int, 2>{i, j}) = out_buffer[i * N + j];
            }
        }
    }
}
#endif

void matmul(const Tensor& A, const Tensor& B, Tensor& out) {
    if(A.shape.size() != 2 || B.shape.size() != 2 || out.shape.size() != 2){
        throw std::runtime_error("matmul: expected 2D tensors");
    }
    if(A.shape[1] != B.shape[0] || out.shape[0] != A.shape[0] || out.shape[1] != B.shape[1]){
        throw std::runtime_error("matmul: shapes do not match");
    }

#ifdef USE_BLAS
    matmul_blas(A, B, out, 0.0f);
#else
    #pragma omp parallel for
    for(int i = 0; i < A.shape[0]; i++){
        for(int j = 0; j < B.shape[1]; j++){
            float temp = 0.0f;
            for(int k = 0; k < B.shape[0]; k++){
                temp += A.at(std::array<int, 2>{i, k}) * B.at(std::array<int, 2>{k, j});
            }
            out.at(std::array<int, 2>{i, j}) = temp;
        }
    }
#endif

    tape_push([A, B, out]() {
#ifdef USE_BLAS
        Tensor dOut = out;
        dOut.data = out.grad;
        Tensor dA = A;
        dA.data = A.grad;
        Tensor dB = B;
        dB.data = B.grad;

        matmul_blas(dOut, B.transpose(0, 1), dA, 1.0f);
        matmul_blas(A.transpose(0, 1), dOut, dB, 1.0f);
#else
        // One thread team for both loops instead of two -- they write disjoint
        // outputs (A.grad vs B.grad), so they can share a region with `nowait`
        // between them instead of paying a second fork/join.
        #pragma omp parallel
        {
            #pragma omp for nowait
            for(int i = 0; i < (int)A.shape[0]; i++){
                for(int k = 0; k < (int)A.shape[1]; k++){
                    float dA = A.grad_at(std::array<int, 2>{i, k});
                    for(int j = 0; j < (int)B.shape[1]; j++){
                        dA += out.grad_at(std::array<int, 2>{i, j}) * B.at(std::array<int, 2>{k, j});
                    }
                    A.grad_at(std::array<int, 2>{i, k}) = dA;
                }
            }

            #pragma omp for
            for(int k = 0; k < (int)B.shape[0]; k++){
                for(int j = 0; j < (int)B.shape[1]; j++){
                    float dB = B.grad_at(std::array<int, 2>{k, j});
                    for(int i = 0; i < (int)A.shape[0]; i++){
                        dB += out.grad_at(std::array<int, 2>{i, j}) * A.at(std::array<int, 2>{i, k});
                    }
                    B.grad_at(std::array<int, 2>{k, j}) = dB;
                }
            }
        }
#endif
    });

}

void embed(const std::vector<int>& x, const Tensor& tok_emb, const Tensor& pos_emb, Tensor& out) {
    int B = out.shape[0];
    int T = out.shape[1];
    int d = out.shape[2];

    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            for(int k = 0; k < d; k++){
                out.at(std::array<int, 3>{b, t, k}) = tok_emb.at(std::array<int, 2>{x[b * T + t], k}) + pos_emb.at(std::array<int, 2>{t, k});
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
                    tok_emb.grad_at(std::array<int, 2>{x[b * T + t], k}) += out.grad_at(std::array<int, 3>{b, t, k});
                    pos_emb.grad_at(std::array<int, 2>{t, k}) += out.grad_at(std::array<int, 3>{b, t, k});
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
            mu += x.at(std::array<int, 2>{i, k});
        }
        mu /= d;

        float var = 0.0f;
        for(int k = 0; k < d; k++){
            float diff = x.at(std::array<int, 2>{i, k}) - mu;
            var += diff * diff;
        }
        var /= d;

        float sigma = std::sqrt(var + eps);

        for(int k = 0; k < d; k++){
            float xhat = (x.at(std::array<int, 2>{i, k}) - mu) / sigma;
            out.at(std::array<int, 2>{i, k}) = g.at(std::array<int, 1>{k}) * xhat + b.at(std::array<int, 1>{k});
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
                mu += x.at(std::array<int, 2>{i, k});
            }
            mu /= d;

            float var = 0.0f;
            for(int k = 0; k < d; k++){
                float diff = x.at(std::array<int, 2>{i, k}) - mu;
                var += diff * diff;
            }
            var /= d;
            float sigma = std::sqrt(var + eps);

            std::vector<float> xhat(d), dxhat(d);
            for(int k = 0; k < d; k++){
                xhat[k] = (x.at(std::array<int, 2>{i, k}) - mu) / sigma;
                dxhat[k] = out.grad_at(std::array<int, 2>{i, k}) * g.at(std::array<int, 1>{k});
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
                x.grad_at(std::array<int, 2>{i, k}) += dx;
                g.grad_at(std::array<int, 1>{k}) += out.grad_at(std::array<int, 2>{i, k}) * xhat[k];
                b.grad_at(std::array<int, 1>{k}) += out.grad_at(std::array<int, 2>{i, k});
            }
        }
    });
}

void add_bias(const Tensor& x, const Tensor& b, Tensor& out) {
    int N = x.shape[0];
    int d = x.shape[1];

    for(int i = 0; i < N; i++){
        for(int k = 0; k < d; k++){
            out.at(std::array<int, 2>{i, k}) = x.at(std::array<int, 2>{i, k}) + b.at(std::array<int, 1>{k});
        }
    }

    tape_push([x, b, out]() {
        int N = x.shape[0];
        int d = x.shape[1];

        for(int i = 0; i < N; i++){
            for(int k = 0; k < d; k++){
                x.grad_at(std::array<int, 2>{i, k}) += out.grad_at(std::array<int, 2>{i, k});
                b.grad_at(std::array<int, 1>{k}) += out.grad_at(std::array<int, 2>{i, k});
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
            float v = x.at(std::array<int, 2>{i, k});
            float u = c * (v + 0.044715f * v * v * v);
            out.at(std::array<int, 2>{i, k}) = 0.5f * v * (1.0f + std::tanh(u));
        }
    }

    tape_push([x, out, c]() {
        int N = x.shape[0];
        int d = x.shape[1];

        for(int i = 0; i < N; i++){
            for(int k = 0; k < d; k++){
                float v = x.at(std::array<int, 2>{i, k});
                float u = c * (v + 0.044715f * v * v * v);
                float t = std::tanh(u);
                float du_dv = c * (1.0f + 3.0f * 0.044715f * v * v);
                float dgelu_dv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du_dv;
                x.grad_at(std::array<int, 2>{i, k}) += out.grad_at(std::array<int, 2>{i, k}) * dgelu_dv;
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
            maxx = std::max(maxx, v.at(std::array<int, 2>{i, j}));
        }

        for(int j = 0; j < v.shape[1]; j++){
            newOut.at(std::array<int, 2>{i, j}) = v.at(std::array<int, 2>{i, j}) - maxx;
        }
        float total = 0.0f;
        for(int j = 0; j < v.shape[1]; j++){
            newOut.at(std::array<int, 2>{i, j}) = std::exp(newOut.at(std::array<int, 2>{i, j}));
            total += newOut.at(std::array<int, 2>{i, j});
        }

        for(int j = 0; j < v.shape[1]; j++){
            newOut.at(std::array<int, 2>{i, j}) /= total;
        }
    }

    tape_push([newOut, v]() {
        for(int i = 0; i < newOut.shape[0]; i++){
            float total = 0.0f;
            for(int j = 0; j < newOut.shape[1]; j++){
                total += newOut.at(std::array<int, 2>{i, j}) * newOut.grad_at(std::array<int, 2>{i, j});
            }

            for(int j = 0; j < newOut.shape[1]; j++){
                float P = newOut.at(std::array<int, 2>{i, j});
                float dP = newOut.grad_at(std::array<int, 2>{i, j});
                v.grad_at(std::array<int, 2>{i, j}) += P * dP - P * total;
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
                    S.at(std::array<int, 2>{i, j}) = QKT.at(std::array<int, 2>{i, j}) / std::sqrt(x.shape[x.shape.size() - 1]);
            }
        }

        tape_push([S, QKT, x](){
            for(int i = 0; i < QKT.shape[0]; i++){
                for(int j = 0; j < QKT.shape[1]; j++){
                        QKT.grad_at(std::array<int, 2>{i, j}) += S.grad_at(std::array<int, 2>{i, j}) / std::sqrt(x.shape[x.shape.size() - 1]);
                }
            }
        });

        Tensor P = make_tensor(arena, S.shape);

        for(int i = 0; i < S.shape[0]; i++){
            for(int j = i + 1; j < S.shape[1]; j++){
                S.at(std::array<int, 2>{i, j}) = -std::numeric_limits<float>::infinity();
            }
        }

        tape_push([S](){
            for(int i = 0; i < S.shape[0]; i++){
                for(int j = 0; j < S.shape[1]; j++){
                    if(j > i){
                        S.grad_at(std::array<int, 2>{i, j}) = 0;
                    }
                }
            }
        });

        softmax(S, P);

        matmul(P, V, outb);
    }

}

void attention(Arena& arena, Tensor& x, Tensor& Wk, Tensor& Wq, Tensor& Wv, Tensor& Wo, int H, Tensor& out){
    Tensor temp_out = make_tensor(arena, out.shape);
    int B = x.shape[0];

    for(int b = 0; b < B; b++){
        Tensor xb = x.slice({b});
        Tensor reshaped_out = temp_out.reshape({out.shape[0], out.shape[1], H, out.shape[2] / H});
        Tensor outb = reshaped_out.slice({b});

        Tensor Q = make_tensor(arena, {x.shape[1], Wq.shape[1]});
        Tensor K = make_tensor(arena, {x.shape[1], Wk.shape[1]});
        Tensor V = make_tensor(arena, {x.shape[1], Wv.shape[1]});

        matmul(xb, Wq, Q);
        matmul(xb, Wk, K);
        matmul(xb, Wv, V);

        Tensor reshaped_Q = Q.reshape({Q.shape[0], H, Q.shape[1] / H});
        Tensor reshaped_K = K.reshape({K.shape[0], H, K.shape[1] / H});
        Tensor reshaped_V = V.reshape({V.shape[0], H, V.shape[1] / H});

        Tensor transposed_Q = reshaped_Q.transpose(0, 1);
        Tensor transposed_K = reshaped_K.transpose(0, 1);
        Tensor transposed_V = reshaped_V.transpose(0, 1);

        for(int h = 0; h < H; h++){
            Tensor outbh = outb.transpose(0, 1).slice({h});
            std::vector newShape({x.shape[1], x.shape[1]});
            Tensor QKT = make_tensor(arena, newShape);
            Tensor S = make_tensor(arena, QKT.shape);

            Tensor Qh = transposed_Q.slice({h});
            Tensor Kh = transposed_K.slice({h});
            Tensor Vh = transposed_V.slice({h});

            Tensor KT = Kh.transpose(Kh.shape.size() - 1, Kh.shape.size() - 2);

            matmul(Qh, KT, QKT);


            for(int i = 0; i < QKT.shape[0]; i++){
                for(int j = 0; j < QKT.shape[1]; j++){
                        S.at(std::array<int, 2>{i, j}) = QKT.at(std::array<int, 2>{i, j}) / std::sqrt(x.shape[x.shape.size() - 1] / H);
                }
            }

            tape_push([S, QKT, x, H](){
                for(int i = 0; i < QKT.shape[0]; i++){
                    for(int j = 0; j < QKT.shape[1]; j++){
                            QKT.grad_at(std::array<int, 2>{i, j}) += S.grad_at(std::array<int, 2>{i, j}) / std::sqrt(x.shape[x.shape.size() - 1] / H);
                    }
                }
            });

            Tensor P = make_tensor(arena, S.shape);

            for(int i = 0; i < S.shape[0]; i++){
                for(int j = i + 1; j < S.shape[1]; j++){
                    S.at(std::array<int, 2>{i, j}) = -std::numeric_limits<float>::infinity();
                }
            }

            tape_push([S](){
                for(int i = 0; i < S.shape[0]; i++){
                    for(int j = 0; j < S.shape[1]; j++){
                        if(j > i){
                            S.grad_at(std::array<int, 2>{i, j}) = 0;
                        }
                    }
                }
            });

            softmax(S, P);

            matmul(P, Vh, outbh);

        }
    }
    Tensor out2d = out.reshape({out.shape[0] * out.shape[1], out.shape[2]});
    matmul(temp_out.reshape({temp_out.shape[0] * temp_out.shape[1], temp_out.shape[2]}), Wo, out2d);
}

void cross_entropy(const Tensor& logits, const std::vector<int>& targets, Tensor& out) {
    int N = logits.shape[0];
    int V = logits.shape[1];

    float total_loss = 0.0f;
    for (int i = 0; i < N; i++) {
        float maxx = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < V; j++) {
            maxx = std::max(maxx, logits.at(std::array<int, 2>{i, j}));
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < V; j++) {
            sum_exp += std::exp(logits.at(std::array<int, 2>{i, j}) - maxx);
        }
        float logsumexp = maxx + std::log(sum_exp);

        total_loss += logsumexp - logits.at(std::array<int, 2>{i, targets[i]});
    }
    out.at(std::array<int, 1>{0}) = total_loss / N;

    tape_push([logits, targets, N, V]() {
        for (int i = 0; i < N; i++) {
            float maxx = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < V; j++) {
                maxx = std::max(maxx, logits.at(std::array<int, 2>{i, j}));
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < V; j++) {
                sum_exp += std::exp(logits.at(std::array<int, 2>{i, j}) - maxx);
            }

            for (int j = 0; j < V; j++) {
                float p = std::exp(logits.at(std::array<int, 2>{i, j}) - maxx) / sum_exp;
                float onehot = (j == targets[i]) ? 1.0f : 0.0f;
                logits.grad_at(std::array<int, 2>{i, j}) += (p - onehot) / N;
            }
        }
    });
}

void add(const Tensor& x, const Tensor& y, Tensor& out){
    for(int i = 0; i < x.shape[0]; i++){
        for(int j = 0; j < x.shape[1]; j++){
            out.at(std::array<int, 2>{i, j}) = x.at(std::array<int, 2>{i, j}) + y.at(std::array<int, 2>{i, j});
        }
    }

    tape_push([x, y, out]() {
        for(int i = 0; i < x.shape[0]; i++){
            for(int j = 0; j < x.shape[1]; j++){
                x.grad_at(std::array<int, 2>{i, j}) += out.grad_at(std::array<int, 2>{i, j}); 
                y.grad_at(std::array<int, 2>{i, j}) += out.grad_at(std::array<int, 2>{i, j});
        }
    }
    });
}
