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

def embed(p, x):

    h = p['tok_emb'][x] + p['pos_emb'][:x.shape[1]]

    return h

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)

    xhat = (x - mu)/np.sqrt(var + eps)

    return  g * xhat + b, (xhat, np.sqrt(var + eps))

def safe_softmax(v):
    v_new = v - v.max(axis=-1, keepdims=True)
    res = np.exp(v_new) / np.exp(v_new).sum(axis=-1, keepdims=True)
    return res

def attention_single(x, Wq, Wk, Wv):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv

    S = (Q @ np.transpose(K, axes=(0, 2, 1))) / np.sqrt(x.shape[-1])

    remove = np.triu(np.ones(S.shape, dtype=bool), k = 1)
    masked = np.where(remove, -np.inf, S)

    P = safe_softmax(masked)

    res = P @ V

    return res

def attention(x, Wq, Wk, Wv, Wo):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv

    Q, K, V = np.reshape(Q, (Q.shape[0], Q.shape[1], H, Q.shape[2] // H)), np.reshape(K, (Q.shape[0], Q.shape[1], H, Q.shape[2] // H)), np.reshape(V, (Q.shape[0], Q.shape[1], H, Q.shape[2] // H))

    Q, K, V = np.transpose(Q, (0, 2,  1, 3)), np.transpose(K, (0, 2,  1, 3)), np.transpose(V, (0, 2,  1, 3))

    S = (Q @ np.transpose(K, (0, 1, 3, 2))) / np.sqrt(Q.shape[3])

    remove = np.triu(np.ones(S.shape, dtype=bool), k = 1)
    masked = np.where(remove, -np.inf, S)

    P = safe_softmax(masked)

    res = P @ V

    res = np.transpose(res, (0, 2, 1, 3))
    res = np.reshape(res, (res.shape[0], res.shape[1], x.shape[-1]))

    O = res @ Wo

    return O, (Q, K, V, P)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))

def mlp(x, W1, b1, W2, b2):
    h = gelu(x @ W1 + b1)
    return h @ W2 + b2

def block(p, x, l):
    m, cache = layer_norm(x, p[f"b{l}.ln1.g"], p[f"b{l}.ln1.b"])

    attn, attn_cache = attention(m, p[f"b{l}.attn.Wq"], p[f"b{l}.attn.Wk"], p[f"b{l}.attn.Wv"], p[f"b{l}.attn.Wo"])

    res = x + attn

    m2, cache2 = layer_norm(res, p[f"b{l}.ln2.g"], p[f"b{l}.ln2.b"])

    res2 = mlp(m2, p[f"b{l}.mlp.W1"], p[f"b{l}.mlp.b1"], p[f"b{l}.mlp.W2"], p[f"b{l}.mlp.b2"])

    fin = res + res2

    return fin, (cache, cache2, attn_cache)

def forward(p, x):
    h = embed(p, x)
    block_caches = []

    for l in range(L):
        h, cache = block(p, h, l)
        block_caches.append(cache)

    h, fin_cache = layer_norm(h, p["lnf.g"], p["lnf.b"])
    logits = h @ p['tok_emb'].T

    return logits, (block_caches, fin_cache)

def cross_entropy(logits, y):
    flattened_logits = logits.reshape(-1, V)
    flattened_y = y.reshape(-1)

    maxi = flattened_logits.max(axis=-1, keepdims=True)
    logsumexp = maxi + np.log(np.exp(flattened_logits - maxi).sum(axis=-1, keepdims=True))

    true_class_logit = flattened_logits[np.arange(len(flattened_y)), flattened_y]

    return (logsumexp.reshape(-1) - true_class_logit).mean()
