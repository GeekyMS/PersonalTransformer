#pragma once
// Forward ops. Each computes its forward result into `out` and pushes its
// own backward closure onto the tape (see tape.h). See docs/roadmap.md
// Phase 5.4 for the port protocol: implement, dump, diff against NumPy,
// only then move to the next op.

#include "tensor.h"

// out = A @ B. A: (M,K), B: (K,N), out: (M,N). Caller allocates out.
void matmul(const Tensor& A, const Tensor& B, Tensor& out);

// out[b][t] = tok_emb[x[b*T+t]] + pos_emb[t]. x: flattened token ids, length
// B*T. tok_emb: (V,d), pos_emb: (T,d). out: (B,T,d) — B, T, d read from
// out.shape. Caller allocates out.
void embed(const std::vector<int>& x, const Tensor& tok_emb, const Tensor& pos_emb, Tensor& out);

// Normalizes each row of x (N,d) over its d features: out = g * xhat + b,
// xhat = (x - mean) / sqrt(var + eps). g, b: (d,). out: (N,d).
void layer_norm(const Tensor& x, const Tensor& g, const Tensor& b, Tensor& out, float eps = 1e-5f);

// out = x + b, broadcasting b (d,) over every row of x (N,d).
void add_bias(const Tensor& x, const Tensor& b, Tensor& out);

// Elementwise GELU (tanh approximation), same shape in and out.
void gelu(const Tensor& x, Tensor& out);

// out = gelu(x @ W1 + b1) @ W2 + b2. x: (N,d), W1: (d,4d), b1: (4d,),
// W2: (4d,d), b2: (d,), out: (N,d). Pure composition of the ops above —
// allocates its own intermediates from arena.
void mlp(Arena& arena, const Tensor& x, const Tensor& W1, const Tensor& b1,
          const Tensor& W2, const Tensor& b2, Tensor& out);

void softmax(const Tensor& S, Tensor& out);

void attention_core(Arena& arena, Tensor& x, Tensor& Wk, Tensor& Wq, Tensor& Wv, Tensor& out);

void attention(Arena& arena, Tensor& x, Tensor& Wk, Tensor& Wq, Tensor& Wv, Tensor& Wo, int H, Tensor& out);

// Fused softmax + cross-entropy, mirroring np_impl.model.cross_entropy's
// numerically-stable logsumexp form (and np_impl.backward.cross_entropy_backward
// for the shortcut backward: dlogits = (softmax(logits) - onehot(y)) / N).
// logits: (N,V) where N = B*T (flatten the batch/time axes before calling).
// targets: flat token ids, length N. out: scalar loss, shape {1}.
// TODO(you): implement forward (logsumexp - true_class_logit, mean over N)
// and push the fused backward closure onto the tape.
void cross_entropy(const Tensor& logits, const std::vector<int>& targets, Tensor& out);

void add(const Tensor& x, const Tensor& y, Tensor& out);
