"""Reference functions for cpp/test/compare.py to diff the C++ op ports
against. Thin wrappers, not implementations — see docs/roadmap.md Phase 5.4.
"""
import numpy as np

from np_impl import model as np_model


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def embed(x: np.ndarray, tok_emb: np.ndarray, pos_emb: np.ndarray) -> np.ndarray:
    # x: flat (B*T,) float-encoded ints, row-major over (B, T) — i.e. index
    # b*T+t. T is recovered from pos_emb's shape, B from x's length.
    T = pos_emb.shape[0]
    B = x.shape[0] // T
    x_int = x.astype(np.int64)
    tok = tok_emb[x_int]                 # (B*T, d)
    pos = np.tile(pos_emb, (B, 1))       # (B*T, d), row i == pos_emb[i % T]
    return (tok + pos).reshape(B, T, -1)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    out, _cache = np_model.layer_norm(x, g, b)
    return out


def add_bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x + b


def gelu(x: np.ndarray) -> np.ndarray:
    return np_model.gelu(x)


def mlp(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np_model.mlp(x, W1, b1, W2, b2)


def softmax(S: np.ndarray) -> np.ndarray:
    return np_model.safe_softmax(S)


def softmax_backward(S: np.ndarray, dP: np.ndarray) -> np.ndarray:
    # Cross-checks the C++ analytic VJP against the already grad-checked
    # np_impl.backward.softmax_vjp (see backward.py:290-296), rather than
    # re-deriving a finite-difference check on the C++ side.
    from np_impl.backward import softmax_vjp
    P = np_model.safe_softmax(S)
    return softmax_vjp(P, dP)


def attention_core(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> np.ndarray:
    return np_model.attention_single(x, Wq, Wk, Wv)


def attention_core_grads(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                          dOut: np.ndarray) -> np.ndarray:
    # Reuses the already grad-checked attention_core_forward/backward (which
    # are written for the 4D (B,H,T,dh) shape) by threading a singleton head
    # axis through -- mathematically identical to the single-head (B,T,dh)
    # case attention_single computes, without re-deriving a 3D-shaped
    # backward from scratch. Then finishes the chain out to dWq/dWk/dWv/dx
    # the same way np_impl.backward.attention_backward does in its steps
    # 3 and 5, minus the multi-head merge/Wo (not present here).
    from np_impl.backward import attention_core_forward, attention_core_backward
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    Q4, K4, V4 = Q[:, None], K[:, None], V[:, None]
    _, cache = attention_core_forward(Q4, K4, V4)
    dOut4 = dOut[:, None]
    dQ4, dK4, dV4 = attention_core_backward(dOut4, cache)
    dQ, dK, dV = dQ4[:, 0], dK4[:, 0], dV4[:, 0]

    B_, T_, d_ = x.shape
    x_flat = x.reshape(-1, d_)
    dQ_flat = dQ.reshape(-1, dQ.shape[-1])
    dK_flat = dK.reshape(-1, dK.shape[-1])
    dV_flat = dV.reshape(-1, dV.shape[-1])
    dWq = x_flat.T @ dQ_flat
    dWk = x_flat.T @ dK_flat
    dWv = x_flat.T @ dV_flat
    dx = (dQ_flat @ Wq.T + dK_flat @ Wk.T + dV_flat @ Wv.T).reshape(B_, T_, d_)

    return np.concatenate([dx.ravel(), dWq.ravel(), dWk.ravel(), dWv.ravel()])


def attention(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
              Wo: np.ndarray, H: int) -> np.ndarray:
    # np_model.attention() reads the head count off a module-level constant
    # (model.py's H=4) rather than taking it as a parameter -- temporarily
    # override it so this test can use small shapes instead of the project's
    # real (B=32,T=256,d=256,H=4).
    orig_H = np_model.H
    np_model.H = H
    try:
        O, _cache = np_model.attention(x, Wq, Wk, Wv, Wo)
        return O
    finally:
        np_model.H = orig_H


def attention_grads(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                     Wo: np.ndarray, H: int, dOut: np.ndarray) -> np.ndarray:
    # Reuses the already grad-checked np_impl.backward.attention_backward.
    # It derives (B,H,T,dh) from the cache's own shape, not the module-level
    # H, so no monkeypatching needed here (unlike the forward wrapper above).
    from np_impl.backward import attention_core_forward, attention_backward
    B_, T_, d_ = x.shape
    dh = d_ // H
    Q, K, V = x @ Wq, x @ Wk, x @ Wv

    def split_heads(t):
        return t.reshape(B_, T_, H, dh).transpose(0, 2, 1, 3)

    Q4, K4, V4 = split_heads(Q), split_heads(K), split_heads(V)
    _, cache = attention_core_forward(Q4, K4, V4)

    dx, dWq, dWk, dWv, dWo = attention_backward(dOut, x, Wq, Wk, Wv, Wo, cache)

    return np.concatenate([dx.ravel(), dWq.ravel(), dWk.ravel(), dWv.ravel(), dWo.ravel()])


# contract.json / gen_inputs.py only knows how to generate array-shaped
# inputs, so H (a plain int, not a tensor) can't be a contract "input" --
# fix it here instead, matching the H=2 baked into the C++ test binary.
def attention_h2(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                  Wo: np.ndarray) -> np.ndarray:
    return attention(x, Wq, Wk, Wv, Wo, H=2)


def attention_h2_grads(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                        Wo: np.ndarray, dOut: np.ndarray) -> np.ndarray:
    return attention_grads(x, Wq, Wk, Wv, Wo, H=2, dOut=dOut)


# block()/forward() use H=4 (not H=2 like the standalone attention test) --
# the C++ block() calls attention() with kH from constants.h directly
# (hardcoded, not a parameter), and kH=4 there.
_BLOCK_PARAM_NAMES = [
    "b0.ln1.g", "b0.ln1.b",
    "b0.attn.Wq", "b0.attn.Wk", "b0.attn.Wv", "b0.attn.Wo",
    "b0.ln2.g", "b0.ln2.b",
    "b0.mlp.W1", "b0.mlp.b1", "b0.mlp.W2", "b0.mlp.b2",
]


def _block_params(ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2):
    values = [ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2]
    return dict(zip(_BLOCK_PARAM_NAMES, values))


def block(x: np.ndarray, ln1_g: np.ndarray, ln1_b: np.ndarray,
          Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
          ln2_g: np.ndarray, ln2_b: np.ndarray,
          W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    p = _block_params(ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2)
    orig_H = np_model.H
    np_model.H = 4
    try:
        fin, _cache = np_model.block(p, x, 0)
        return fin
    finally:
        np_model.H = orig_H


def block_grads(x: np.ndarray, ln1_g: np.ndarray, ln1_b: np.ndarray,
                 Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
                 ln2_g: np.ndarray, ln2_b: np.ndarray,
                 W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                 dOut: np.ndarray) -> np.ndarray:
    from np_impl.backward import block_backward
    p = _block_params(ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2)
    orig_H = np_model.H
    np_model.H = 4
    try:
        _fin, cache = np_model.block(p, x, 0)
        dx, grads = block_backward(dOut, p, 0, cache)
    finally:
        np_model.H = orig_H

    parts = [dx.ravel()] + [grads[k].ravel() for k in _BLOCK_PARAM_NAMES]
    return np.concatenate(parts)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    # np_model.cross_entropy reads V off a module-level import (common.data's
    # real vocab size, 65), not off logits.shape -- override it temporarily,
    # same reasoning as the H monkeypatch above.
    orig_V = np_model.V
    np_model.V = logits.shape[-1]
    try:
        return np.array([np_model.cross_entropy(logits, y.astype(np.int64))])
    finally:
        np_model.V = orig_V


def cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    # cross_entropy_backward unpacks logits.shape into exactly 3 values
    # (B_,T_,V_), so thread a dummy leading axis through -- the actual math
    # only depends on flattened N=B_*T_ and V, not how N splits.
    from np_impl.backward import cross_entropy_backward as ce_backward
    N, V = logits.shape
    dlogits3 = ce_backward(logits.reshape(N, 1, V), y.reshape(N, 1).astype(np.int64))
    return dlogits3.reshape(N, V)
