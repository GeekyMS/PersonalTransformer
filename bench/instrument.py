import numpy as np
from pathlib import Path

from np_impl.model import init_params, d, H, L, V, T as BASE_T

BYTES_PER_FLOAT = 4  # fp32


def total_params(T):
    # every param except pos_emb is independent of T, so compute the real
    # count once at BASE_T and adjust just the pos_emb term algebraically
    # rather than re-deriving every shape by hand.
    base = init_params(np.random.default_rng(0))
    base_count = sum(v.size for v in base.values())
    return base_count + (T - BASE_T) * d


def activation_bytes(B, T):
    # rough estimate of the dominant *cached* activations per block --
    # the residual stream, Q/K/V, the attention weights P, the two LayerNorm
    # caches, and the MLP's widened hidden layer.
    residual = B * T * d
    qkv = 3 * B * T * d
    attn_weights = B * H * T * T
    ln_caches = 2 * B * T * d
    mlp_hidden = B * T * 4 * d
    per_layer = residual + qkv + attn_weights + ln_caches + mlp_hidden

    embed_out = B * T * d
    logits = B * T * V
    total_elems = L * per_layer + embed_out + logits
    return total_elems * BYTES_PER_FLOAT


def s_matrix_bytes(B, T):
    return B * H * T * T * BYTES_PER_FLOAT


def format_bytes(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.2f} {unit}'
        n /= 1024
    return f'{n:.2f} PB'


def main():
    B = 32
    Ts = [256, 1024, 4096]

    lines = []
    lines.append('# Phase 1.9 -- Instrumentation\n')
    lines.append(f'B = {B}, d = {d}, H = {H}, L = {L}, V = {V}\n')
    lines.append('| T | total params | activation memory | S matrix size |')
    lines.append('|---|---|---|---|')

    for T in Ts:
        p = total_params(T)
        act = activation_bytes(B, T)
        s = s_matrix_bytes(B, T)
        lines.append(f'| {T} | {p:,} | {format_bytes(act)} | {format_bytes(s)} |')

    out = '\n'.join(lines) + '\n'
    print(out)

    out_path = Path(__file__).resolve().parent / 'phase1_instrumentation.md'
    out_path.write_text(out)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    main()
