import numpy as np

from common.data import V

B = 32          # batch size
T = 256         # context length
d = 256         # d_model
H = 4           # attention heads
d_head = d // H
L = 6           # layers

def init_params(rng):
    p = {}
    p['tok_emb'] = rng.normal(0, 0.02, (V, d))
    p['pos_emb'] = rng.normal(0, 0.02, (T, d))

    for l in range(L):
        p[f"b{l}.ln1.g"] = np.ones(d); p[f"b{l}.ln1.b"] = np.zeros(d)
        p[f"b{l}.attn.Wq"] = rng.normal(0, 0.02, (d, d))
        p[f"b{l}.attn.Wk"] = rng.normal(0, 0.02, (d, d))
        p[f"b{l}.attn.Wv"] = rng.normal(0, 0.02, (d, d))
        p[f"b{l}.attn.Wo"] = rng.normal(0, 0.02, (d, d)) / np.sqrt(2 * L)


        p[f"b{l}.ln2.g"] = np.ones(d); p[f"b{l}.ln2.b"] = np.zeros(d)
        p[f"b{l}.mlp.W1"] = rng.normal(0, 0.02, (d, 4 * d))
        p[f"b{l}.mlp.b1"] = np.zeros(4 * d)

        p[f"b{l}.mlp.W2"] = rng.normal(0, 0.02, (4 * d, d)) / np.sqrt(2 * L)
        p[f"b{l}.mlp.b2"] = np.zeros(d)

    p["lnf.g"] = np.ones(d); p["lnf.b"] = np.zeros(d)
    return p