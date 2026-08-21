import numpy as np
from common.grad_check import check_grad
from np_impl.model import layer_norm, gelu, safe_softmax, cross_entropy, embed, mlp, block
import np_impl.model as _model
from np_impl.model import L as L_full


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def analytic_jacobian_row(p, i):
    # p: the already-computed softmax output vector (cached from forward)
    # returns J[j] = d p_i / d s_j, for every j -- one row of the Jacobian
    T = len(p)
    J = np.zeros(T)
    for j in range(T):
        if i == j:
            J[j] = p[i] * (1 - p[j])   # <- your i==j formula, in terms of p[i] / p[j]
        else:
            J[j] = -1 * p[i] * p[j]   # <- your i!=j formula, in terms of p[i] / p[j]
    return J

def softmax_vjp(p, dP):
    return p * dP - p * (p * dP).sum(axis=-1, keepdims=True)

def pv_backward(P, V, dO):
    dP = dO @ V.transpose(0, 1, 3, 2)   # <- your formula
    dV = P.transpose(0, 1, 3, 2) @ dO   # <- your formula
    return dP, dV


def qk_backward(Q, K, dS, dh):
    d_raw_S = dS / np.sqrt(dh)   # undo the scale first
    dQ = d_raw_S @ K   # <- your formula, in terms of d_raw_S and K
    dK = (Q.transpose(0, 1, 3, 2) @ d_raw_S).transpose(0, 1, 3, 2)   # <- your formula, in terms of d_raw_S and Q
    return dQ, dK

def mask_backward(d_masked, remove):
    dS = np.where(remove, 0, d_masked)   # <- your formula
    return dS


def attention_core_backward(dO, cache):
    # cache = (Q, K, V, P), same order attention() returns from its own cache
    Q, K, V, P = cache
    dh = Q.shape[-1]
    T = Q.shape[-2]
    remove = np.triu(np.ones((T, T), dtype=bool), k=1)

    dP, dV = pv_backward(P, V, dO)
    d_masked = softmax_vjp(P, dP)
    dS = mask_backward(d_masked, remove)
    dQ, dK = qk_backward(Q, K, dS, dh)

    return dQ, dK, dV


def attention_core_forward(Q, K, V):
    dh = Q.shape[-1]
    S = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    remove = np.triu(np.ones(S.shape, dtype=bool), k=1)
    masked = np.where(remove, -np.inf, S)
    P = safe_softmax(masked)
    O = P @ V
    return O, (Q, K, V, P)


def layer_norm_backward(dout, g, xhat, std):
    # dout: (..., d), g: (d,), xhat/std: cached from layer_norm's forward
    reduce_axes = tuple(range(xhat.ndim - 1))   # every axis except the feature axis
    dg = (dout * xhat).sum(axis=reduce_axes)
    db = dout.sum(axis=reduce_axes)

    dxhat = dout * g
    dx = (1 / std) * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dg, db


def gelu_backward(dout, x):
    k = np.sqrt(2 / np.pi)
    c = 0.044715
    u = k * (x + c * x**3)
    tanh_u = np.tanh(u)
    du_dx = k * (1 + 3 * c * x**2)
    dgelu_dx = 0.5 * (1 + tanh_u) + 0.5 * x * (1 - tanh_u**2) * du_dx
    return dout * dgelu_dx


def cross_entropy_backward(logits, y):
    # fused softmax + cross-entropy shortcut: dlogits = (softmax(logits) - onehot(y)) / N
    B_, T_, V_ = logits.shape
    N = B_ * T_
    probs = safe_softmax(logits)

    flat_y = y.reshape(-1)
    onehot = np.zeros((N, V_))
    onehot[np.arange(N), flat_y] = 1
    onehot = onehot.reshape(B_, T_, V_)

    return (probs - onehot) / N


def embed_backward(dout, x, V_, d_, T_full):
    # dout: (B, T, d) -- gradient flowing into embed()'s output
    # T_full: pos_emb's actual row count -- forward only ever uses pos_emb[:T_actual],
    # so rows T_actual..T_full-1 never participated and must get exactly zero gradient,
    # not be missing from the array entirely.
    dtok_emb = np.zeros((V_, d_))
    np.add.at(dtok_emb, x, dout)   # scatter-add -- += silently drops repeated indices
    dpos_emb = np.zeros((T_full, d_))
    dpos_emb[:dout.shape[1]] = dout.sum(axis=0)    # sum over the broadcast batch axis
    return dtok_emb, dpos_emb

def attention_backward(dout, x, Wq, Wk, Wv, Wo, cache):
    Q, K, V, P = cache
    B_, H_, T_, dh_ = Q.shape
    d_ = H_ * dh_

    def merge_heads(g):
        return g.transpose(0, 2, 1, 3).reshape(B_, T_, d_)

    def split_heads(g):
        return g.reshape(B_, T_, H_, dh_).transpose(0, 2, 1, 3)

    # recompute res (merged-heads output, pre-Wo) -- not cached, cheap to redo
    res = merge_heads(P @ V)

    # step 1: backward through Wo
    res_flat = res.reshape(-1, d_)
    dout_flat = dout.reshape(-1, d_)
    dWo =  res_flat.T @ dout_flat  # <- flatten res and dout first, then matmul
    dres = dout_flat @ Wo.T  # <- (B,T,d)

    # step 2: undo the head-merge reshape to feed attention_core_backward
    dO_core = split_heads(dres)

    # step 3: the chain you already built and verified
    dQ, dK, dV = attention_core_backward(dO_core, cache)

    # step 4: merge each back to (B,T,d)
    dQ_m, dK_m, dV_m = merge_heads(dQ), merge_heads(dK), merge_heads(dV)

    # step 5: backward through Q=x@Wq, K=x@Wk, V=x@Wv
    x_flat, dQ_m_flat, dK_m_flat, dV_m_flat = x.reshape(-1, d_), dQ_m.reshape(-1, d_), dK_m.reshape(-1, d_), dV_m.reshape(-1, d_)
    dWq = x_flat.T @ dQ_m_flat
    dWk = x_flat.T @ dK_m_flat
    dWv = x_flat.T @ dV_m_flat

    dx = (dQ_m_flat @ Wq.T + dK_m_flat @ Wk.T + dV_m_flat @ Wv.T).reshape(B_, T_, d_)   # <- sum of three separate contributions

    return dx, dWq, dWk, dWv, dWo

def mlp_backward(dout, x, W1, b1, W2):
    z = x @ W1 + b1
    h = gelu(z)

    # step 1: backward through out = h@W2 + b2
    reduce_axes = tuple(range(h.ndim - 1))
    db2 = dout.sum(axis = reduce_axes)   # <- sum dout over every axis except the feature axis
    h_flat = h.reshape(-1, h.shape[-1])
    dout_flat = dout.reshape(-1, dout.shape[-1])
    dW2 = h_flat.T @ dout_flat   # <- flatten h and dout first, then matmul (same pattern as dWo)
    dh = dout @ W2.T    # <- (B,T,4d), same pattern as dres = dout @ Wo.T

    # step 2: backward through h = gelu(z)
    dz = gelu_backward(dh, z)    # <- you already have a function for exactly this

    # step 3: backward through z = x@W1 + b1
    dz_flat = dz.reshape(-1, dz.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    db1 = dz.sum(axis = reduce_axes)
    dW1 = x_flat.T @ dz_flat
    dx = dz @ W1.T

    return dx, dW1, db1, dW2, db2

def block_backward(dfin, p, l, cache):
    cache1, cache2, attn_cache = cache
    xhat1, std1 = cache1
    xhat2, std2 = cache2

    g1, b1_ln = p[f"b{l}.ln1.g"], p[f"b{l}.ln1.b"]
    g2, b2_ln = p[f"b{l}.ln2.g"], p[f"b{l}.ln2.b"]
    Wq, Wk, Wv, Wo = p[f"b{l}.attn.Wq"], p[f"b{l}.attn.Wk"], p[f"b{l}.attn.Wv"], p[f"b{l}.attn.Wo"]
    W1, b1_mlp, W2, b2_mlp = p[f"b{l}.mlp.W1"], p[f"b{l}.mlp.b1"], p[f"b{l}.mlp.W2"], p[f"b{l}.mlp.b2"]

    m = g1 * xhat1 + b1_ln    # ln1's output, recomputed -- needed as attention's input
    m2 = g2 * xhat2 + b2_ln   # ln2's output, recomputed -- needed as mlp's input

    # step 1: residual2 split -- fin = res + res2
    dres_direct = dfin
    dres2 = dfin

    # step 2: mlp backward
    dm2, dW1, db1_mlp, dW2, db2_mlp = mlp_backward(dres2, m2, W1, b1_mlp, W2)

    # step 3: ln2 backward
    dres_from_ln2, dg2, db2_ln = layer_norm_backward(dm2, g2, xhat2, std2)

    # step 4: sum -- res was used in TWO places
    dres_total = dres_direct + dres_from_ln2

    # step 5: residual1 split -- res = x + attn
    dx_direct = dres_total
    dattn = dres_total

    # step 6: attention backward
    dm, dWq, dWk, dWv, dWo = attention_backward(dattn, m, Wq, Wk, Wv, Wo, attn_cache)

    # step 7: ln1 backward
    dx_from_ln1, dg1, db1_ln = layer_norm_backward(dm, g1, xhat1, std1)

    # step 8: sum -- x was used in TWO places
    dx_total = dx_direct + dx_from_ln1

    grads = {
        f"b{l}.ln1.g": dg1, f"b{l}.ln1.b": db1_ln,
        f"b{l}.attn.Wq": dWq, f"b{l}.attn.Wk": dWk, f"b{l}.attn.Wv": dWv, f"b{l}.attn.Wo": dWo,
        f"b{l}.ln2.g": dg2, f"b{l}.ln2.b": db2_ln,
        f"b{l}.mlp.W1": dW1, f"b{l}.mlp.b1": db1_mlp, f"b{l}.mlp.W2": dW2, f"b{l}.mlp.b2": db2_mlp,
    }
    return dx_total, grads

def backward(p, x, y, logits, caches):
    block_caches, fin_cache = caches
    xhat_fin, std_fin = fin_cache

    h_final = p['lnf.g'] * xhat_fin + p['lnf.b']   # lnf's output, recomputed

    dlogits = cross_entropy_backward(logits, y)

    # backward through logits = h_final @ tok_emb.T (the tied output projection)
    dh_final = dlogits @ p['tok_emb']                          # pass-through, natural shape
    dlogits_flat = dlogits.reshape(-1, dlogits.shape[-1])
    h_final_flat = h_final.reshape(-1, h_final.shape[-1])
    dtok_emb_from_output = dlogits_flat.T @ h_final_flat        # weight-gradient pattern

    # backward through lnf
    d_current, dg_fin, db_fin = layer_norm_backward(dh_final, p['lnf.g'], xhat_fin, std_fin)

    grads = {'lnf.g': dg_fin, 'lnf.b': db_fin}

    # loop over all blocks -- what order, and using what as each call's incoming gradient?
    for l in range(L_full - 1, -1, -1):
        d_current, block_grads = block_backward(d_current, p, l, block_caches[l])
        grads.update(block_grads)

    # backward through embed
    dtok_emb_from_input, dpos_emb = embed_backward(d_current, x, p['tok_emb'].shape[0], p['tok_emb'].shape[1], p['pos_emb'].shape[0])

    # weight tying: both paths into tok_emb need to be combined -- how?
    grads['tok_emb'] = dtok_emb_from_input + dtok_emb_from_output
    grads['pos_emb'] = dpos_emb

    return grads

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    B_, H_, T_, dh_ = 1, 1, 4, 3
    P = rng.random(size=(B_, H_, T_, T_))
    P = P / P.sum(axis=-1, keepdims=True)   # rows sum to 1, like a real softmax output
    V = rng.normal(size=(B_, H_, T_, dh_))
    dO = rng.normal(size=(B_, H_, T_, dh_))

    # composite check: the FULL assembled attention_core_backward chain,
    # end-to-end (dO -> dQ,dK,dV), not testing pieces in isolation this time
    Q_ac = rng.normal(size=(B_, H_, T_, dh_))
    K_ac = rng.normal(size=(B_, H_, T_, dh_))
    V_ac = rng.normal(size=(B_, H_, T_, dh_))
    dO_ac = rng.normal(size=(B_, H_, T_, dh_))

    O_ac, cache_ac = attention_core_forward(Q_ac, K_ac, V_ac)
    dQ_ac, dK_ac, dV_ac = attention_core_backward(dO_ac, cache_ac)

    g_Qc = lambda Q: np.sum(dO_ac * attention_core_forward(Q, K_ac, V_ac)[0])
    print('attention_core dQ max relative error:', check_grad(g_Qc, Q_ac, dQ_ac))

    g_Kc = lambda K: np.sum(dO_ac * attention_core_forward(Q_ac, K, V_ac)[0])
    print('attention_core dK max relative error:', check_grad(g_Kc, K_ac, dK_ac))

    g_Vc = lambda V: np.sum(dO_ac * attention_core_forward(Q_ac, K_ac, V)[0])
    print('attention_core dV max relative error:', check_grad(g_Vc, V_ac, dV_ac))

    # verify the generalized (axis=-1) softmax_vjp on real attention-shaped input
    S_sv = rng.normal(size=(B_, H_, T_, T_))
    P_sv = safe_softmax(S_sv)
    dP_sv = rng.normal(size=(B_, H_, T_, T_))
    d_masked_analytic = softmax_vjp(P_sv, dP_sv)
    g_sv = lambda S: np.sum(dP_sv * safe_softmax(S))
    print('softmax_vjp (generalized) max relative error:', check_grad(g_sv, S_sv, d_masked_analytic))

    dP_analytic, dV_analytic = pv_backward(P, V, dO)

    # check dP: hold V fixed, vary P
    g_P = lambda P: np.sum(dO * (P @ V))
    err_P = check_grad(g_P, P, dP_analytic)
    print('dP max relative error:', err_P)

    # check dV: hold P fixed, vary V
    g_V = lambda V: np.sum(dO * (P @ V))
    err_V = check_grad(g_V, V, dV_analytic)
    print('dV max relative error:', err_V)

    dh_ = 3
    Q = rng.normal(size=(B_, H_, T_, dh_))
    K = rng.normal(size=(B_, H_, T_, dh_))
    dS = rng.normal(size=(B_, H_, T_, T_))

    dQ_analytic, dK_analytic = qk_backward(Q, K, dS, dh_)

    def raw_S_fn(Q, K):
        return (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh_)

    g_Q = lambda Q: np.sum(dS * raw_S_fn(Q, K))
    err_Q = check_grad(g_Q, Q, dQ_analytic)
    print('dQ max relative error:', err_Q)

    g_K = lambda K: np.sum(dS * raw_S_fn(Q, K))
    err_K = check_grad(g_K, K, dK_analytic)
    print('dK max relative error:', err_K)

    T_ = 4
    S = rng.normal(size=(1, 1, T_, T_))
    remove = np.triu(np.ones(S.shape, dtype=bool), k=1)
    d_masked = rng.normal(size=(1, 1, T_, T_))

    dS_analytic = mask_backward(d_masked, remove)

    def masked_fn(S):
        return np.where(remove, -1e2, S)   # finite stand-in for -inf, just for this numerical check

    g = lambda S: np.sum(d_masked * masked_fn(S))
    err = check_grad(g, S, dS_analytic)
    print('mask backward max relative error:', err)

    # --- 2.3: LayerNorm backward ---
    x_ln = rng.normal(size=(2, 3, 4))
    g_ln = rng.normal(size=(4,))
    b_ln = rng.normal(size=(4,))
    out_ln, (xhat_ln, std_ln) = layer_norm(x_ln, g_ln, b_ln)
    dout_ln = rng.normal(size=(2, 3, 4))

    dx_ln, dg_ln, db_ln = layer_norm_backward(dout_ln, g_ln, xhat_ln, std_ln)

    f_x = lambda x: np.sum(dout_ln * layer_norm(x, g_ln, b_ln)[0])
    print('layer_norm dx max relative error:', check_grad(f_x, x_ln, dx_ln))

    f_g = lambda g: np.sum(dout_ln * layer_norm(x_ln, g, b_ln)[0])
    print('layer_norm dg max relative error:', check_grad(f_g, g_ln, dg_ln))

    f_b = lambda b: np.sum(dout_ln * layer_norm(x_ln, g_ln, b)[0])
    print('layer_norm db max relative error:', check_grad(f_b, b_ln, db_ln))

    # --- 2.3: GELU backward ---
    x_gelu = rng.normal(size=(2, 3, 4))
    dout_gelu = rng.normal(size=(2, 3, 4))
    dx_gelu = gelu_backward(dout_gelu, x_gelu)

    f_gelu = lambda x: np.sum(dout_gelu * gelu(x))
    print('gelu backward max relative error:', check_grad(f_gelu, x_gelu, dx_gelu))

    # --- 2.3: fused cross-entropy + softmax backward ---
    from common.data import V as V_real
    logits_ce = rng.normal(size=(2, 3, V_real))
    y_ce = rng.integers(0, V_real, size=(2, 3))
    dlogits_ce = cross_entropy_backward(logits_ce, y_ce)

    f_ce = lambda logits: cross_entropy(logits, y_ce)
    print('cross_entropy backward max relative error:', check_grad(f_ce, logits_ce, dlogits_ce))

    # --- 2.3: embedding backward ---
    # T_e < T_full_e deliberately, to exercise the zero-padding fix -- pos_emb is
    # only ever partially used, and the unused rows must still come back correctly zeroed.
    V_e, d_e, B_e, T_e, T_full_e = 5, 4, 2, 3, 7
    tok_emb_e = rng.normal(size=(V_e, d_e))
    pos_emb_e = rng.normal(size=(T_full_e, d_e))
    p_e = {'tok_emb': tok_emb_e, 'pos_emb': pos_emb_e}
    x_e = rng.integers(0, V_e, size=(B_e, T_e))
    dout_e = rng.normal(size=(B_e, T_e, d_e))

    dtok_emb_e, dpos_emb_e = embed_backward(dout_e, x_e, V_e, d_e, T_full_e)
    print('dpos_emb shape:', dpos_emb_e.shape, 'expected:', pos_emb_e.shape)
    print('unused pos_emb rows are exactly zero:', np.all(dpos_emb_e[T_e:] == 0))

    f_tok = lambda tok_emb: np.sum(dout_e * embed({'tok_emb': tok_emb, 'pos_emb': pos_emb_e}, x_e))
    print('embed dtok_emb max relative error:', check_grad(f_tok, tok_emb_e, dtok_emb_e))

    f_pos = lambda pos_emb: np.sum(dout_e * embed({'tok_emb': tok_emb_e, 'pos_emb': pos_emb}, x_e))
    print('embed dpos_emb max relative error:', check_grad(f_pos, pos_emb_e, dpos_emb_e))

    # --- full attention_backward composite check ---
    Ba, Ha, Ta, dha = 1, 2, 4, 3
    da = Ha * dha
    x_a = rng.normal(size=(Ba, Ta, da))
    Wq_a = rng.normal(size=(da, da))
    Wk_a = rng.normal(size=(da, da))
    Wv_a = rng.normal(size=(da, da))
    Wo_a = rng.normal(size=(da, da))
    dout_a = rng.normal(size=(Ba, Ta, da))

    def split_heads_a(g):
        return g.reshape(Ba, Ta, Ha, dha).transpose(0, 2, 1, 3)

    def merge_heads_a(g):
        return g.transpose(0, 2, 1, 3).reshape(Ba, Ta, da)

    def attention_fn(x, Wq, Wk, Wv, Wo):
        Q, K, V = split_heads_a(x @ Wq), split_heads_a(x @ Wk), split_heads_a(x @ Wv)
        O, cache = attention_core_forward(Q, K, V)
        out = merge_heads_a(O) @ Wo
        return out, cache

    out_a, cache_a = attention_fn(x_a, Wq_a, Wk_a, Wv_a, Wo_a)
    dx_a, dWq_a, dWk_a, dWv_a, dWo_a = attention_backward(dout_a, x_a, Wq_a, Wk_a, Wv_a, Wo_a, cache_a)

    g_x = lambda x: np.sum(dout_a * attention_fn(x, Wq_a, Wk_a, Wv_a, Wo_a)[0])
    print('attention_backward dx max relative error:', check_grad(g_x, x_a, dx_a))

    g_wq = lambda Wq: np.sum(dout_a * attention_fn(x_a, Wq, Wk_a, Wv_a, Wo_a)[0])
    print('attention_backward dWq max relative error:', check_grad(g_wq, Wq_a, dWq_a))

    g_wk = lambda Wk: np.sum(dout_a * attention_fn(x_a, Wq_a, Wk, Wv_a, Wo_a)[0])
    print('attention_backward dWk max relative error:', check_grad(g_wk, Wk_a, dWk_a))

    g_wv = lambda Wv: np.sum(dout_a * attention_fn(x_a, Wq_a, Wk_a, Wv, Wo_a)[0])
    print('attention_backward dWv max relative error:', check_grad(g_wv, Wv_a, dWv_a))

    g_wo = lambda Wo: np.sum(dout_a * attention_fn(x_a, Wq_a, Wk_a, Wv_a, Wo)[0])
    print('attention_backward dWo max relative error:', check_grad(g_wo, Wo_a, dWo_a))

    # --- mlp_backward composite check ---
    B_m, T_m, d_m = 2, 3, 4
    x_m = rng.normal(size=(B_m, T_m, d_m))
    W1_m = rng.normal(size=(d_m, 4 * d_m))
    b1_m = rng.normal(size=(4 * d_m,))
    W2_m = rng.normal(size=(4 * d_m, d_m))
    b2_m = rng.normal(size=(d_m,))
    dout_m = rng.normal(size=(B_m, T_m, d_m))

    dx_m, dW1_m, db1_m, dW2_m, db2_m = mlp_backward(dout_m, x_m, W1_m, b1_m, W2_m)

    g_x_m = lambda x: np.sum(dout_m * mlp(x, W1_m, b1_m, W2_m, b2_m))
    print('mlp dx max relative error:', check_grad(g_x_m, x_m, dx_m))

    g_W1 = lambda W1: np.sum(dout_m * mlp(x_m, W1, b1_m, W2_m, b2_m))
    print('mlp dW1 max relative error:', check_grad(g_W1, W1_m, dW1_m))

    g_b1 = lambda b1: np.sum(dout_m * mlp(x_m, W1_m, b1, W2_m, b2_m))
    print('mlp db1 max relative error:', check_grad(g_b1, b1_m, db1_m))

    g_W2 = lambda W2: np.sum(dout_m * mlp(x_m, W1_m, b1_m, W2, b2_m))
    print('mlp dW2 max relative error:', check_grad(g_W2, W2_m, dW2_m))

    g_b2 = lambda b2: np.sum(dout_m * mlp(x_m, W1_m, b1_m, W2_m, b2))
    print('mlp db2 max relative error:', check_grad(g_b2, b2_m, db2_m))

    # --- block_backward composite check ---
    orig_H, orig_dh = _model.H, _model.d_head
    _model.H, _model.d_head = 2, 2   # small test scale: d_b=4 -> d_head = 4/2 = 2

    d_b, T_b, B_b = 4, 3, 2
    rng_b = np.random.default_rng(1)
    p_b = {
        'b0.ln1.g': np.ones(d_b), 'b0.ln1.b': np.zeros(d_b),
        'b0.attn.Wq': rng_b.normal(0, 0.02, (d_b, d_b)),
        'b0.attn.Wk': rng_b.normal(0, 0.02, (d_b, d_b)),
        'b0.attn.Wv': rng_b.normal(0, 0.02, (d_b, d_b)),
        'b0.attn.Wo': rng_b.normal(0, 0.02, (d_b, d_b)),
        'b0.ln2.g': np.ones(d_b), 'b0.ln2.b': np.zeros(d_b),
        'b0.mlp.W1': rng_b.normal(0, 0.02, (d_b, 4 * d_b)),
        'b0.mlp.b1': np.zeros(4 * d_b),
        'b0.mlp.W2': rng_b.normal(0, 0.02, (4 * d_b, d_b)),
        'b0.mlp.b2': np.zeros(d_b),
    }
    x_b = rng_b.normal(size=(B_b, T_b, d_b))

    fin_b, cache_b = block(p_b, x_b, 0)
    dfin_b = rng_b.normal(size=(B_b, T_b, d_b))

    dx_b, grads_b = block_backward(dfin_b, p_b, 0, cache_b)

    g_x_b = lambda x: np.sum(dfin_b * block(p_b, x, 0)[0])
    print('block dx max relative error:', check_grad(g_x_b, x_b, dx_b))

    for key in p_b:
        def make_g(key):
            def g(val):
                p_test = dict(p_b)
                p_test[key] = val
                return np.sum(dfin_b * block(p_test, x_b, 0)[0])
            return g
        err = check_grad(make_g(key), p_b[key], grads_b[key])
        print(f'block d[{key}] max relative error:', err)

    _model.H, _model.d_head = orig_H, orig_dh

    # demonstrate why np.add.at matters: naive += silently drops repeated indices
    naive = np.zeros((V_e, d_e))
    naive[x_e] += dout_e
    correct = np.zeros((V_e, d_e))
    np.add.at(correct, x_e, dout_e)
    print('naive += matches np.add.at (should be False if indices repeat):', np.allclose(naive, correct))
