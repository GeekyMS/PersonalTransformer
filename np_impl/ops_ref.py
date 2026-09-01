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
