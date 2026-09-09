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
