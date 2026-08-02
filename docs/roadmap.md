# Transformer From Scratch — Detailed Roadmap

**Goal:** a decoder-only transformer, built three times (NumPy → C++ → CUDA), that trains a
character-level language model and generates text — with enough instrumentation that
FlashAttention reads as an obvious consequence rather than a clever trick.

**Written assuming:** you understand the architecture conceptually, but your practical ML
experience is scikit-learn. That matters, so this document explains the machinery `model.fit()`
was hiding, not just the transformer parts.

---

## What's actually different from scikit-learn

Worth reading before Phase 0, because these are the assumptions everything below rests on.

**1. There is no `.fit()`. You write the loop.**

```
for step in range(max_steps):
    x, y   = get_batch('train')      # sample data
    logits, cache = forward(params, x)
    loss   = cross_entropy(logits, y)
    grads  = backward(cache, y)      # dLoss/dParam for every parameter
    params = adam_step(params, grads, step)
```

Five lines, and you own all of them. Every phase in this roadmap is filling in one of them.

**2. "Parameters" are a dict of arrays, and gradients mirror it exactly.**

In sklearn the model is an opaque object. Here:

```python
params = {
    'tok_emb':      ndarray (V, d),
    'pos_emb':      ndarray (T, d),
    'block0.ln1.g': ndarray (d,),
    'block0.attn.Wq': ndarray (d, d),
    ...
}
```

And `grads` has **exactly the same keys and exactly the same shapes**. `grads['block0.attn.Wq']`
is `dLoss/dWq`, same `(d, d)` shape as `Wq`. This structural identity is your best sanity check —
assert it every step.

**3. Your input is not a feature matrix.**

sklearn: `X` is `(n_samples, n_features)` of floats you scaled with `StandardScaler`. Here `x` is
`(B, T)` of **integers** — indices into a lookup table. No scaling, no normalization. `x[0,5] = 41`
means "the 6th character of the 1st sequence in this batch is character #41." Feeding it to the
model is `tok_emb[41]`, a row lookup.

**4. You never see the whole dataset. There are no epochs.**

sklearn fits on all of `X`. Here you sample random windows forever. "Step 3000" means 3000 random
batches, some data seen twice, some never. That's normal and fine.

**5. Everything is 4D and broadcasting will silently lie to you.**

A `(B, H, T, d)` array where you meant `(B, T, H, d)` often still runs. It will just train badly.
Annotate every line with the shape. This is not optional advice.

**6. Randomness is a dependency you control.**

Weight init, batch sampling, and generation sampling all draw from an RNG. Seed it explicitly.
Same seed must give the identical loss curve every run, or you can't compare phases.

---

## Target model — fixed across all three implementations

| Symbol | Meaning | Value |
|---|---|---|
| `B` | batch size (sequences per step) | 32 |
| `T` | context length (chars per sequence) | 256 |
| `V` | vocab size (distinct chars) | ~65 |
| `d` | model width (`d_model`) | 256 |
| `H` | attention heads | 4 |
| `d_head` | `d / H` | 64 |
| `L` | layers | 6 |

~5M parameters. Pre-LN, GELU, 4× MLP expansion, weights tied between input embedding and output
projection.

Keep these constant across NumPy, C++, and CUDA. Every phase must reproduce the same loss curve
from the same seed. That identity is your regression test.

**Out of scope:** BPE tokenizer, distributed training, mixed precision, dropout, beam search,
encoder-decoder. KV cache is deliberately deferred to Phase 6.

---

# Phase 0 — Data and test harness

*Estimated: 3–5 sessions. Do not skip. Everything downstream is defined as "matches the previous
phase," which requires machinery to check that.*

## 0.1 The dataset

**What it is:** `tiny-shakespeare` — a single plain-text file, ~1.1 MB, ~1.1M characters of
Shakespeare plays. Downloadable as one `.txt`. No labels, no CSV, no features. Just text.

**Why this one:** small enough to train in minutes on CPU, structured enough that you can *see*
learning happen — first it learns character frequencies, then spaces and word lengths, then
line breaks and `ROMEO:` speaker labels, then vaguely English-shaped words. That progression is
your debugging signal.

## 0.2 Tokenizer (character level)

"Tokenizing" here means: map each distinct character to an integer.

```python
chars = sorted(set(text))          # ['\n', ' ', '!', ..., 'z']  → ~65 entries
V     = len(chars)
stoi  = {c: i for i, c in enumerate(chars)}   # 'a' → 39
itos  = {i: c for c, i in stoi.items()}       # 39  → 'a'

encode = lambda s: [stoi[c] for c in s]       # str  → list[int]
decode = lambda l: ''.join(itos[i] for i in l) # list[int] → str
```

Then encode the *entire* file once into a flat array:

```python
data = np.array(encode(text), dtype=np.int64)   # shape (1_115_394,)
n     = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

**Why character-level:** a real BPE tokenizer is a genuinely separate project (merge tables,
greedy longest-match, byte fallback) and teaches you nothing about attention. Character-level
gives you `V=65`, which makes the output projection tiny and the whole thing fast.

**Sanity check:** `decode(encode("Hello")) == "Hello"`.

## 0.3 The batch sampler — *this is the "dataset loader"*

This is the part that has no sklearn analogue, so in detail.

```python
def get_batch(split, B, T, rng):
    data = train_data if split == 'train' else val_data
    ix = rng.integers(0, len(data) - T - 1, size=B)     # B random start positions
    x  = np.stack([data[i     : i + T    ] for i in ix])  # (B, T)
    y  = np.stack([data[i + 1 : i + T + 1] for i in ix])  # (B, T)
    return x, y
```

**What's happening:** you pick `B` random positions in the text. For each, you slice out `T`
consecutive characters as the input, and the *same window shifted right by one* as the target.

Concretely, if the text is `"To be or not"` and `T=5`:

```
x = "To be"      → [30, 66, 1, 40, 43]
y = "o be "      → [66, 1, 40, 43, 1]
```

Position 0 sees `T` and must predict `o`. Position 1 sees `To` and must predict ` `. Position 4
sees `To be` and must predict ` `.

**The key insight — and this trips up everyone coming from sklearn:** a single `(B, T)` batch
does **not** give you `B` training examples. It gives you `B × T` of them, because the causal
mask means position `t` can only see positions `≤ t`, so each position is an independent
next-character prediction problem. `B=32, T=256` → 8,192 gradient signals per step. That's why
the causal mask isn't just about "not cheating" — it's what makes training efficient.

**Shapes contract:** `x` and `y` are both `(B, T)`, dtype int64. Values in `[0, V)`.

**Sanity checks before moving on:**
- `decode(x[0])` prints readable Shakespeare
- `decode(y[0])` prints the same thing shifted one character left
- `(x[0][1:] == y[0][:-1]).all()` is True
- Same seed → identical batches across runs

## 0.4 Gradient checker

You're going to hand-derive gradients in Phase 2. This is how you find out you got them wrong.

The idea: the gradient is a derivative, and you can approximate any derivative numerically by
nudging the input and seeing how the output changes.

```python
def check_grad(f, x, analytic_grad, h=1e-5):
    """f: ndarray -> scalar.  analytic_grad: same shape as x."""
    num_grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + h; fp = f(x)
        x[i] = old - h; fm = f(x)
        x[i] = old
        num_grad[i] = (fp - fm) / (2 * h)
        it.iternext()
    rel = np.abs(num_grad - analytic_grad) / (np.abs(num_grad) + np.abs(analytic_grad) + 1e-12)
    return rel.max()
```

**Use central difference**, not `(f(x+h) - f(x)) / h` — the two-sided version has error `O(h²)`
instead of `O(h)` and it matters here.

**Threshold:** max relative error `< 1e-5` in float64 is a pass. `~1e-2` means a real bug.
`~1e-7` on some elements and `~1` on others usually means those elements are legitimately zero.

**Critical caveat:** it is `O(number of elements)` forward passes. Check ops in **isolation** on
tiny inputs (`B=2, T=4, d=8`), never the whole model. Also run it in float64 — float32 noise
swamps `h=1e-5`.

**Second caveat:** never gradient-check through a ReLU-like kink or through the causal mask's
`-inf`. Finite differences are undefined there. GELU is smooth so it's fine.

## 0.5 The PyTorch oracle

Install PyTorch. Write the same architecture using `nn.Module` — 60 lines, using `nn.Linear`,
`nn.LayerNorm`, and `F.scaled_dot_product_attention`. You will not learn anything from writing it,
and that's the point: it's a reference, not an exercise.

Then write a weight-transfer function that copies your `params` dict into the PyTorch model
(watch out: `nn.Linear` stores its weight transposed, as `(out, in)`).

**This gives you:** for any input, ground-truth logits and ground-truth `.grad` for every
parameter. Every "done when" below is `np.allclose(mine, oracle, atol=1e-5)`.

**Debugging technique this unlocks:** when your logits are wrong, hook intermediate PyTorch
activations and binary-search the model for the first layer where you diverge. Without this you
are guessing.

## 0.6 The FLOP/byte counter

A global tally that ops register into:

```python
COUNTER = {'flops': 0, 'bytes': 0}

def count(flops, bytes_moved):
    COUNTER['flops'] += flops
    COUNTER['bytes'] += bytes_moved
```

For each op, call it with the analytic cost. A matmul `(M,K) @ (K,N)`: `2*M*K*N` FLOPs (multiply
+ add), `4*(M*K + K*N + M*N)` bytes for fp32. Softmax over `(R, C)`: roughly `5*R*C` FLOPs and
`4*2*R*C*3` bytes (three passes: max, exp+sum, divide).

Build this in Phase 0 so you're not retrofitting it in Phase 4.

**Phase 0 done when:** you can pull a batch, print it as readable text, verify the shift
property, gradient-check a hand-written `f(x) = sum(x**2)` to `1e-6`, and load your (empty)
params into the PyTorch oracle without a shape error.

---

# Phase 1 — NumPy forward pass

*Estimated: 5–8 sessions. Build bottom-up. Don't write any backward code yet.*

Every function here is a plain function taking arrays and returning arrays. No classes yet.
Annotate every single line with the resulting shape as a comment.

## 1.1 Initialization

```python
def init_params(rng):
    p = {}
    p['tok_emb'] = rng.normal(0, 0.02, (V, d))
    p['pos_emb'] = rng.normal(0, 0.02, (T, d))
    for l in range(L):
        p[f'b{l}.ln1.g'] = np.ones(d);  p[f'b{l}.ln1.b'] = np.zeros(d)
        p[f'b{l}.attn.Wq'] = rng.normal(0, 0.02, (d, d))
        ...  # Wk, Wv, Wo
        p[f'b{l}.ln2.g'] = np.ones(d);  p[f'b{l}.ln2.b'] = np.zeros(d)
        p[f'b{l}.mlp.W1'] = rng.normal(0, 0.02, (d, 4*d))
        p[f'b{l}.mlp.b1'] = np.zeros(4*d)
        ...  # W2, b2
    p['lnf.g'] = np.ones(d); p['lnf.b'] = np.zeros(d)
    return p
```

**Why `std=0.02` and not `1.0`:** if weights are too large, activations grow layer over layer and
the softmax saturates. Too small and the signal dies. 0.02 is what GPT-2 used and it works at
this scale. Also scale the residual-path output projections (`Wo`, `W2`) by `1/sqrt(2*L)` — with
`L` residual additions the variance accumulates linearly, and this compensates.

**No bias on the attention projections.** Biases on `Wq/Wk/Wv` are provably redundant with the
LayerNorm before them. MLP biases are conventional; keep them.

**Sanity check:** with random init and no training, your loss should be `≈ ln(V) ≈ 4.17` nats.
The model predicts uniformly, and `-ln(1/65) = 4.17`. **If your initial loss isn't ~4.17,
something is wrong before you've trained anything.** This is the single best early check in the
whole project.

## 1.2 Embedding lookup

```python
def embed(p, x):           # x: (B, T) int
    tok = p['tok_emb'][x]              # (B, T, d)  ← fancy indexing
    pos = p['pos_emb'][:x.shape[1]]    # (T, d)
    return tok + pos                   # (B, T, d)  broadcasts over B
```

`p['tok_emb'][x]` with an integer array `x` is NumPy advanced indexing: it produces an array of
shape `x.shape + (d,)`. This is a gather, and in Phase 5 you'll write it as an explicit loop.

## 1.3 LayerNorm

```python
def layer_norm(x, g, b, eps=1e-5):     # x: (..., d)
    mu  = x.mean(-1, keepdims=True)          # (..., 1)
    var = x.var(-1, keepdims=True)           # (..., 1)
    xh  = (x - mu) / np.sqrt(var + eps)      # (..., d)
    return g * xh + b, (xh, np.sqrt(var+eps))  # cache for backward
```

Normalizes **across the feature dimension, per token** — not across the batch. Every one of your
`B*T` tokens gets its own mean and variance over its `d=256` features. This is why it works with
any batch size and needs no running statistics (unlike BatchNorm, which you may know from sklearn-
adjacent tooling).

Return the cache now even though Phase 2 is where you use it — retrofitting caches later is
tedious.

## 1.4 Attention — do single-head first

Write this for one head, `(B, T, d_head)`, verify it, *then* generalize. Debugging a 4D shape bug
and a math bug simultaneously is miserable.

```python
def attention_single(x, Wq, Wk, Wv):     # x: (B, T, dh)
    Q = x @ Wq                            # (B, T, dh)
    K = x @ Wk                            # (B, T, dh)
    V = x @ Wv                            # (B, T, dh)

    S = Q @ K.transpose(0, 2, 1)          # (B, T, T)  ← THE quadratic object
    S = S / np.sqrt(dh)

    mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # True above diagonal
    S = np.where(mask, -np.inf, S)        # (B, T, T)

    # softmax, numerically stable
    S_max = S.max(-1, keepdims=True)      # (B, T, 1)
    P     = np.exp(S - S_max)             # (B, T, T)
    P     = P / P.sum(-1, keepdims=True)  # (B, T, T)

    O = P @ V                             # (B, T, dh)
    return O, (Q, K, V, P)
```

**`S` is `(B, T, T)`. Write down its size in bytes at `T` = 256, 1024, 4096.** At `B=32, H=4,
T=4096`, that's `32*4*4096*4096*4` bytes = **8.6 GB** for a single layer's scores. This number is
the entire motivation for Phase 4 and 6.

**The mask.** Use true `-inf`, not `-1e9`. Two reasons: `-1e9` in fp16 overflows to `-inf`
anyway so you may as well be honest, and in fp32 a row that's *entirely* masked (never happens
with causal masking, but happens with padding masks) gives a nonzero softmax with `-1e9` and a
correct NaN with `-inf`. **Use `np.where`, not multiplication** — `0 * -inf = NaN`.

**The `1/sqrt(dh)` scale.** For `q, k` with unit-variance entries, `q·k` is a sum of `dh` products,
so it has variance `dh`. At `dh=64` that's a std of 8, which pushes softmax into saturation where
gradients vanish. Dividing by `sqrt(dh)` restores unit variance. **Empirically verify this:**
remove the scale, compute the entropy of `P`, and watch it collapse toward one-hot.

**Subtracting the row max.** `exp(800)` overflows to `inf`. Subtracting the max makes the largest
exponent 0, so `exp` maxes at 1. Mathematically it changes nothing — the constant cancels in the
ratio.

> **Leave a comment on that line.** Max → exp → sum → divide is the seed of FlashAttention. FA's
> core trick is computing this *incrementally*: keep a running max and running sum as tiles of `K`
> stream past, and rescale the accumulator whenever the max updates. You will implement that in
> Phase 6.3. Come back here.

## 1.5 Multi-head

```python
def attention(x, Wq, Wk, Wv, Wo):        # x: (B, T, d)
    B_, T_, _ = x.shape
    Q = (x @ Wq).reshape(B_, T_, H, dh).transpose(0, 2, 1, 3)   # (B, H, T, dh)
    K = (x @ Wk).reshape(B_, T_, H, dh).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(B_, T_, H, dh).transpose(0, 2, 1, 3)

    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)               # (B, H, T, T)
    ... mask, softmax ...
    O = P @ V                                                    # (B, H, T, dh)
    O = O.transpose(0, 2, 1, 3).reshape(B_, T_, d)              # (B, T, d)
    return O @ Wo, cache
```

**The reshape/transpose order is the whole lesson.** `reshape(B,T,H,dh)` then
`transpose(0,2,1,3)` is *not* the same as `reshape(B,H,T,dh)`. The first splits the `d` axis into
`(H, dh)` — meaning head `h` owns columns `h*dh : (h+1)*dh` of the projection — then moves `H`
forward. The second reinterprets the raw buffer and mixes tokens into heads. It runs. It produces
garbage.

**Pick `(B, H, T, dh)` as your canonical layout, write it on a sticky note, and check it at every
reshape.** In Phase 5, when you're managing strides by hand, this is where the `contiguous()`
question comes from: the transpose is free (metadata only), but the reshape after it requires a
real copy because the memory is no longer laid out the way the new shape claims.

## 1.6 GELU and MLP

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def mlp(x, W1, b1, W2, b2):     # x: (B, T, d)
    h = gelu(x @ W1 + b1)        # (B, T, 4d)
    return h @ W2 + b2           # (B, T, d)
```

Use the tanh approximation, not the exact `x * Φ(x)` — it's what every real implementation uses,
so your PyTorch oracle must be set to `approximate='tanh'` to match.

## 1.7 Block and full forward

```python
def block(x, p, l):
    a, c1 = attention(layer_norm(x, p[f'b{l}.ln1.g'], p[f'b{l}.ln1.b']), ...)
    x = x + a                                     # residual
    m, c2 = mlp(layer_norm(x, p[f'b{l}.ln2.g'], ...), ...)
    x = x + m
    return x, (c1, c2)
```

**Pre-LN** (normalize *before* the sublayer, add the raw input) rather than post-LN. Post-LN needs
a learning-rate warmup to train at all at this depth. Pre-LN also gives a clean gradient highway
straight from the loss to the embeddings through the residual adds.

Final: `x = layer_norm(x, lnf.g, lnf.b)`, then `logits = x @ tok_emb.T` → `(B, T, V)`.

**Weight tying:** the output projection reuses the transposed token embedding. Halves your
parameters and improves quality at this scale. In Phase 2 this means `tok_emb` receives gradient
from **two** paths — the embedding lookup *and* the output projection — and you must accumulate
both. Forgetting this is a classic bug that manifests as "trains, but worse than it should."

## 1.8 Cross-entropy loss

```python
def cross_entropy(logits, y):        # logits (B,T,V), y (B,T)
    logits = logits.reshape(-1, V)   # (B*T, V)
    y      = y.reshape(-1)           # (B*T,)
    m      = logits.max(-1, keepdims=True)
    logsumexp = m.squeeze(-1) + np.log(np.exp(logits - m).sum(-1))
    return (logsumexp - logits[np.arange(len(y)), y]).mean()
```

Flattening `(B, T)` into `B*T` independent classification problems is the concrete form of "each
position is its own training example."

Compute this in **log space** via logsumexp — never `log(softmax(x))`, which underflows.

**Units:** this is in nats (natural log). `4.17` = random. `~1.5` = a trained char-level model on
Shakespeare. If you want bits, divide by `ln(2)`.

## 1.9 Instrument

Print, for `T` in `[256, 1024, 4096]`: total params, activation memory, and specifically the size
of all `S` matrices. Save this table — Phase 4 needs it.

**Phase 1 done when:** with weights transferred to the PyTorch oracle, `np.allclose(my_logits,
torch_logits, atol=1e-5)` passes, and untrained loss is `4.17 ± 0.05`.

---

# Phase 2 — Backward pass, by hand

*Estimated: 8–12 sessions. The hardest phase, and the one that pays off in Phase 6.*

## 2.0 The mental model you need first

Every op gets a `backward` with this contract:

> Given `dL/d(my output)`, return `dL/d(my input)` and `dL/d(my parameters)`.

That's a **vector-Jacobian product**. You never build the Jacobian matrix — for softmax over 256
elements the Jacobian is `256×256` per row, and you'd need `B*T*H` of them. You always find the
algebraic simplification that computes the product directly.

Naming convention that will save you: for any variable `Z`, call its gradient `dZ`, and **`dZ`
always has exactly the same shape as `Z`.** If your `dWq` isn't `(d, d)`, stop and find the bug.

Rules you'll use constantly:
- `C = A @ B` → `dA = dC @ B.T`, `dB = A.T @ dC`
- `C = A + B` (broadcast) → `dA = dC`, `dB = dC.sum(over broadcast axes)`
- `y = f(x)` elementwise → `dx = dy * f'(x)`
- Two ops consumed the same tensor → **add** their gradients

## 2.1 Derive yourself: the softmax Jacobian

Do this on paper. It's four lines and it's the most important derivation in the project.

For `p_i = exp(s_i) / Σ_k exp(s_k)`:

```
∂p_i/∂s_j = p_i (δ_ij − p_j)
```

Now form the VJP. Given `dP`:

```
dS_j = Σ_i dP_i · p_i (δ_ij − p_j)
     = dP_j p_j − p_j Σ_i dP_i p_i
```

Vectorized over rows:

```python
dS = P * (dP - (dP * P).sum(-1, keepdims=True))
```

**One elementwise multiply and one row-sum.** No `T×T` Jacobian. Understanding *why* this
collapse happens is what lets you follow FlashAttention's backward, where the same expression
appears but `P` has to be recomputed rather than read.

## 2.2 Derive yourself: attention backward

Given `dO`, with cached `Q, K, V, P`:

```python
dV = P.transpose(0,1,3,2) @ dO          # (B,H,T,dh)
dP = dO @ V.transpose(0,1,3,2)          # (B,H,T,T)
dS = P * (dP - (dP*P).sum(-1, keepdims=True))
dS = np.where(mask, 0, dS) / np.sqrt(dh)
dQ = dS @ K                             # (B,H,T,dh)
dK = dS.transpose(0,1,3,2) @ Q          # (B,H,T,dh)
```

**Stop and notice:** computing `dQ` requires `dS`, which requires `P`, which requires `S`. The
`(B,H,T,T)` matrix you needed in the forward pass is needed *again* in the backward pass.

Standard attention solves this by keeping `P` in HBM from forward to backward. That's the second
reason attention eats memory — not just the transient cost, but the *retained* cost across the
whole forward pass of every layer.

**FlashAttention's backward recomputes `S` from `Q` and `K` inside the kernel instead of storing
it** — trading FLOPs (which are cheap, you're bandwidth-bound) for HBM traffic (which is not).
When you read that section of the paper, you'll already know exactly what it's avoiding. This
derivation is your Phase 6.5 spec.

**Mask handling:** zero the masked positions of `dS`, don't `-inf` them. Masked entries got
`P = 0` in the forward pass, so they contributed nothing and receive nothing.

## 2.3 Look these up (low information density, just implement)

- **LayerNorm backward.** The `(1/σ)(dy_normalized − mean(...) − x̂·mean(... · x̂))` form. The
  subtlety is that `μ` and `σ` both depend on every element of `x`, so there are three gradient
  paths. Get it from a reference; gradient-check it hard.
- **GELU backward.** Differentiate the tanh approximation. Tedious, mechanical.
- **Cross-entropy + softmax fused.** `dlogits = (softmax(logits) − onehot(y)) / N`. Fusing them
  avoids a numerically nasty division; also it's beautiful.
- **Embedding backward.** A scatter-add: `np.add.at(dtok_emb, x, dout)`. Note `np.add.at` and not
  `dtok_emb[x] += dout` — the latter silently drops repeated indices, and in char-level text
  indices repeat constantly. This bug is very hard to see and will just make training worse.

## 2.4 Structure

Explicit manual chaining — each function gets `forward` and `backward`, and you call them in
reverse order by hand. **Do not build a general autograd graph here.** That's Phase 5's job, and
building it now hides the activation-cache memory cost you need to see in Phase 4.

## 2.5 Verification protocol

Bottom-up. Never move to the next op until the current one passes.

1. Gradient-check each op **in isolation**, float64, tiny shapes (`B=2, T=4, d=8, H=2`)
2. Then check composites (a full block)
3. Then diff the full model's grads against PyTorch's `.grad`

**When something fails**, the fastest debugging move is: run the forward pass in PyTorch with
`retain_grad()` on intermediates, and binary-search for the first tensor where your gradient
diverges. The bug is in that op's backward.

**Phase 2 done when:** every op passes `check_grad` at `<1e-5`, and every entry of your `grads`
dict matches PyTorch's `.grad` at `atol=1e-5`.

---

# Phase 3 — Training loop

*Estimated: 3–5 sessions. Mechanically easy; mostly about learning what a broken loss curve
looks like.*

## 3.1 AdamW

Write it. Ten lines, and you need it again in C++.

```python
def adam_step(p, g, m, v, t, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    for k in p:
        m[k] = b1*m[k] + (1-b1)*g[k]
        v[k] = b2*v[k] + (1-b2)*g[k]**2
        mh = m[k] / (1 - b1**t)          # bias correction
        vh = v[k] / (1 - b2**t)
        p[k] -= lr * (mh / (np.sqrt(vh) + eps) + wd * p[k])
```

`m` is a running mean of the gradient (momentum), `v` a running mean of its square. Dividing by
`sqrt(v)` gives each parameter its own effective step size. **Bias correction** matters because
`m` and `v` start at zero, so early estimates are biased toward zero — without it your first
few hundred steps take wildly wrong step sizes.

**AdamW vs Adam:** weight decay applied to the parameter directly, not folded into the gradient.
Don't decay LayerNorm gains, biases, or embeddings — only matmul weights.

`β2 = 0.95` rather than the 0.999 default; shorter runs need faster adaptation.

## 3.2 Schedule

- Linear warmup over the first ~100 steps from 0 to `lr = 3e-4`
- Cosine decay to `lr/10` over the remaining steps
- Gradient clipping at global norm 1.0: compute `sqrt(Σ ||g||²)` across *all* params, and if it
  exceeds 1.0, scale everything down by the same factor

Warmup exists because Adam's `v` estimate is garbage for the first few dozen steps; taking full-
size steps then can push you somewhere unrecoverable.

## 3.3 The loop

```python
for step in range(5000):
    x, y = get_batch('train', B, T, rng)
    logits, cache = forward(params, x)
    loss  = cross_entropy(logits, y)
    grads = backward(cache, y)
    clip_(grads, 1.0)
    params = adam_step(params, grads, m, v, step+1, lr_at(step))

    if step % 100 == 0:
        val = evaluate(params, 'val', n_batches=20)
        print(f"{step}: train {loss:.4f}  val {val:.4f}")
    if step % 500 == 0:
        print(generate(params, prompt="\n", n=300))
```

## 3.4 Generation

```python
def generate(p, prompt, n, temp=0.8, top_k=40):
    idx = encode(prompt)
    for _ in range(n):
        ctx = idx[-T:]                              # crop to context window
        logits, _ = forward(p, np.array([ctx]))     # (1, len(ctx), V)
        logits = logits[0, -1] / temp               # last position only
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
        probs = softmax(logits)
        idx.append(rng.choice(V, p=probs))
    return decode(idx)
```

**Only the last position's logits matter** — you're predicting one next character. You recompute
the entire forward pass for every generated character, which is `O(n·T²)` and absurdly wasteful.
Leave it wasteful. Phase 6.6 fixes it with a KV cache, and the waste is what makes that fix
feel earned.

## 3.5 What a healthy run looks like

| Step | Loss | Sample |
|---|---|---|
| 0 | 4.17 | `qX;jZ vv!W?k` — uniform noise |
| 100 | ~2.6 | `the aoun the sa an` — character frequencies, spaces |
| 500 | ~2.1 | `And hor the wors of ther` — word shapes |
| 2000 | ~1.7 | `MENENIUS: I will not the man` — speaker labels, line breaks |
| 5000 | ~1.5 | mostly-real words, plausible dialogue structure |

**Failure signatures:**
- **Loss stuck at 4.17:** gradients aren't reaching parameters. Check that `params` is actually
  being mutated, and that `grads` has all the keys.
- **Loss → NaN:** almost always the mask (`0 * -inf`), or a missing max-subtraction in softmax,
  or LR too high. Add a `np.isfinite(loss)` assert and bisect.
- **Loss drops to ~0 immediately:** you're leaking the answer. Your mask is wrong, or your `y` is
  misaligned and position `t` can see character `t+1`.
- **Train ≪ val:** overfitting. Expected on 1MB of text with 5M params. Not a bug.

**Phase 3 done when:** val loss ~1.5, generated text has speaker names and line structure, and a
fixed seed reproduces the identical loss curve twice. **Save that curve and a fixed-seed sample
to a file — they are Phase 5's acceptance test.**

---

# Phase 4 — Roofline analysis

*Estimated: 2–4 sessions. Output is a writeup, not code. This is the phase the whole project
exists for.*

## 4.1 The per-op table

For each op at `B=32, T=1024`: FLOPs, bytes moved, arithmetic intensity (FLOPs/byte), measured
wall time.

Expected shape of the result:

| Op | FLOPs | AI (FLOP/byte) | Bound by |
|---|---|---|---|
| QKV projections | high | ~50+ | compute |
| `Q @ K.T` | `2·B·H·T²·dh` | ~`dh/2` | compute |
| mask + softmax | `~5·B·H·T²` | **< 1** | **memory** |
| `P @ V` | `2·B·H·T²·dh` | ~`dh/2` | compute |
| MLP | high | ~50+ | compute |

## 4.2 The ridge point

Compute your GPU's ridge point: `peak FLOP/s ÷ peak GB/s`. For an A100 that's roughly
`19.5e12 / 1555e9 ≈ 12.5 FLOP/byte`. Any op with arithmetic intensity below that is
**bandwidth-bound** — it will never approach peak compute no matter how good your kernel is,
because it's waiting on memory.

Plot every op against it. Matmuls land right of the ridge. Softmax, the mask, and the scale land
far to the left. **Attention as a composite is memory-bound, and structurally it shouldn't be** —
it's `O(T²·dh)` FLOPs on `O(T·dh)` of input data.

## 4.3 Count the HBM round-trips

In your implementation, for one attention head:

1. write `S` (T² floats)
2. read `S`, write `S_scaled`
3. read `S_scaled`, write `S_masked`
4. read `S_masked` (max pass), read again (exp+sum pass), write `P`
5. read `P` for `P @ V`
6. **and retain `P` in memory until the backward pass**

Roughly **6–8 full passes over a `T²` array**, for an operation whose inputs are only `T×dh`. At
`T=1024, dh=64` that's 16× more data movement than the inputs justify.

## 4.4 Then, and only then, read the FlashAttention paper

You will have independently derived its motivation. The paper stops being a clever trick and
becomes a description of the solution to a problem you personally measured.

This is the same roofline reasoning that justified halo tiling in your heat-diffusion stencil —
you fused passes to avoid re-reading global memory. Attention is that problem one level up, with
a softmax in the middle that makes the fusion non-obvious.

**Phase 4 done when:** you have the AI table, the roofline plot, the round-trip count, and a
paragraph in your own words predicting what an optimal attention kernel would do — written
*before* you read the paper. Then read it and diff your prediction against theirs.

---

# Phase 4.5 — JAX / XLA: watching a compiler fail to invent FlashAttention

*Optional but strongly recommended. Estimated: 3–5 sessions. Do this AFTER Phase 4 and BEFORE
Phase 5.*

**Framing:** you are not using JAX as a nicer NumPy. You are using it as a compiler you can
inspect, applied to a model you wrote yourself and understand completely. That combination is
rare and it's the whole value here.

**Deliberately NOT used in Phases 1–3.** `jax.grad` would gut Phase 2 — the point of hand-deriving
the softmax Jacobian is that FlashAttention's backward is unreadable without it. And JAX's
friction (functional purity, immutable arrays via `.at[].set()`, tracer leaks, retracing,
`jit` compile latency) teaches you nothing about attention while you're still learning your
first training loop.

## 4.5.1 Port the forward pass

`jax.numpy` is close to a drop-in for `numpy`. Port **the Phase 1 forward pass only** — no
backward, no training loop. Mostly mechanical.

Two things that will bite:
- Arrays are immutable. `x[i] = v` becomes `x = x.at[i].set(v)`.
- Everything under `jit` must be traceable — no data-dependent Python control flow, no `.item()`,
  no printing values (use `jax.debug.print`).

## 4.5.2 Read the IR at every level

```python
# Level 1 — the traced IR (jaxpr)
print(jax.make_jaxpr(forward)(params, x))

# Level 2 — HLO, before XLA optimizes
print(jax.jit(forward).lower(params, x).compiler_ir())

# Level 3 — HLO, AFTER optimization. This is where fusion decisions live.
print(jax.jit(forward).lower(params, x).compile().as_text())
```

For PTX and the full dump of every pass:

```bash
XLA_FLAGS="--xla_dump_to=./xla_dump --xla_dump_hlo_pass_re=.*" python model.py
```

## 4.5.3 What to look for

Work through the optimized HLO and answer these in writing:

1. **Which ops got fused?** Look for `fusion` instructions and read their contents. Expect the
   scale, the mask, and parts of the softmax to be fused together.
2. **Did the `T×T` score matrix survive?** Find a buffer of shape `(B, H, T, T)` in the optimized
   HLO. **It will be there.**
3. **How many kernels does attention become?** Count them. Compare against the 6–8 HBM round-trips
   you counted by hand in Phase 4.3.
4. **What layouts did XLA choose?** Layout assignment is a real pass with real consequences —
   look for transposes XLA inserted that you didn't write.

## 4.5.4 The insight this phase exists for

**XLA fuses aggressively and still materializes `S`. It cannot derive FlashAttention.**

Fusion is a *local rewrite over the dataflow graph* — it changes the schedule, not the algorithm.
FlashAttention requires online softmax: maintaining a running max and running sum across tiles and
**rescaling an already-accumulated output** when the max updates. That's an algebraic
reassociation across a reduction. It changes what is computed, not just when, and no general-
purpose compiler has license to make that transformation.

This is why FlashAttention had to be hand-written, and why it was eventually **pattern-matched**
into cuDNN and XLA as a recognized subgraph rather than emerging from the optimizer. Write that
argument up with your own HLO dumps as evidence.

For a compiler-engineering target, this is a better interview artifact than the transformer
itself: it's a concrete, measured statement about the boundary of what fusion passes can reach.

## 4.5.5 Optional extras

- Diff the HLO with `--xla_gpu_enable_triton_softmax_fusion` on and off
- Compare XLA's emitted kernel against your hand-written CUDA in Phase 6.1 — same op, one written
  by a compiler, one by you
- `jax.grad` your forward pass and diff against your Phase 2 hand-derived gradients. Free extra
  oracle, and now it isn't a crutch because you've already done the work.

**Phase 4.5 done when:** you have optimized HLO dumps, an annotated fusion map for attention, and
a written argument for why the `T×T` materialization survives every optimization pass.

---

# Phase 5 — C++ port with your own autograd

*Estimated: 10–15 sessions. This is where the preallocated-buffer thing you noticed becomes
something you can't avoid.*

## 5.1 Tensor type

```cpp
struct Tensor {
    float*              data;     // not owned
    std::vector<int>    shape;
    std::vector<int>    strides;  // in elements
    int                 offset;
};
```

Row-major: for shape `(B,H,T,d)`, strides are `(H*T*d, T*d, d, 1)`.

**The payoff:** `transpose` swaps two entries in `shape` and `strides` and touches zero bytes of
data. `reshape` only works if the memory is contiguous in the new order — otherwise you need a
real copy. **This is where `contiguous()` comes from, and you'll understand it in about ten
minutes of implementing it.**

## 5.2 Allocator — the thing you spotted

**Target: zero `malloc` inside the steady-state training loop.**

Every op signature takes its output buffer:

```cpp
void matmul(const Tensor& A, const Tensor& B, Tensor& out);
void softmax(const Tensor& in, Tensor& out);
```

Not `Tensor matmul(A, B)`. The caller owns the memory.

Implementation: a bump allocator over one big preallocated arena. `alloc(n)` returns
`base + offset` and advances `offset`. Reset the offset to zero at the start of each step. All
your shapes are known and fixed after step 1, so a single arena sized once at startup covers
every step forever.

**This is exactly the `out=` pattern you noticed in the high-level interface.** Now you're on the
other side of it: you're the one who has to decide where every byte lives, when it's reused, and
whether a transpose costs anything. Profile before and after — the allocation overhead in a naive
version is not small.

## 5.3 Tape-based autograd

```cpp
struct Node {
    std::function<void()> backward;  // closure capturing the tensors it needs
};
std::vector<Node> tape;
```

Forward pass pushes a closure per op. Backward walks the tape in reverse calling each. ~200 lines.

**Why now and not Phase 2:** in Phase 2 the manual chaining forced you to see every cached
activation explicitly, which is what let you count activation memory in Phase 4. Now that you've
seen it, the abstraction is safe.

## 5.4 Port protocol

Op by op, diffing against the NumPy reference at each step:
1. Write the C++ op
2. Dump its output to a binary file
3. Load in Python, `np.allclose` against your Phase 1/2 implementation
4. Only then move to the next op

Reuse the blocked/tiled matmul from your `cuda-kernels` Phase 1 CPU baselines.

**Phase 5 done when:** same seed → identical loss curve → identical generated text as Phase 3
(bit-identical is unrealistic across languages; match to 1e-4 and eyeball the sample), and a
profiler confirms no allocations in the training loop.

---

# Phase 6 — CUDA

*Estimated: 12–20 sessions. Folds into the `cuda-kernels` repo as its capstone.*

Each step gets a benchmark row in the same format as your stencil and matmul results:
GB/s achieved, % of peak, speedup over previous.

**6.1 Naive attention.** Separate kernels for QKᵀ, scale, mask, softmax, PV. This is your baseline
and it should be embarrassing. Measure it honestly — it's what everything else is compared against.

**6.2 Fused softmax kernel.** One block per row, warp-level shuffle reductions for max and sum,
then a broadcast divide. Self-contained lesson in reductions. Expect a large speedup — you just
went from 3 HBM passes to 1.

**6.3 Online softmax, standalone.** Single pass with a running max `m` and running sum `l`. When a
new element exceeds the running max, rescale the accumulated sum by `exp(m_old − m_new)`.
**Implement and test this in isolation before putting it inside anything.** It's the load-bearing
idea and you don't want to debug it inside a tiled kernel.

**6.4 FlashAttention forward.** Load a `Q` tile into shared memory. Stream `K`/`V` tiles past it.
For each `K` tile: compute the partial scores, update the online softmax state, and rescale the
output accumulator. **`S` is never written to HBM.** You now have every prerequisite — shared-
memory tiling from the stencil, reductions from 6.2, online softmax from 6.3.

**6.5 FlashAttention backward.** Recompute `S` from `Q` and `K` in-kernel rather than reading a
stored `P`. Your Phase 2.2 derivation is the spec. Harder than the forward — budget for it.

**6.6 KV cache for inference.** A structurally different memory-bound problem: at generation time
you process one token at a time, so there's no batch dimension to amortize the weight loads. It's
purely bandwidth-limited on reading the weights. Good contrast, and it fixes the `O(n·T²)`
generation from Phase 3.4.

**6.7 (optional) Pallas comparison.** Reimplement 6.4 in Pallas, JAX's Triton-like kernel DSL.
Roughly 40 lines against your 400. Benchmark both and write up the gap. Good "when is the
abstraction worth it" argument for a repo README. **Strictly after the raw CUDA, never instead of
it** — the hand-written kernel is the artifact that matters for an NVIDIA target.

**Phase 6 done when:** your CUDA attention matches the NumPy reference to fp32 tolerance, and the
benchmark table shows naive → fused softmax → tiled with real numbers.

---

# Consolidated gotchas

- **`-inf` vs `-1e9`.** Use `-inf`. Use `np.where`, never multiplication — `0 * -inf = NaN`.
- **`np.add.at` for embedding gradients**, not `+=`. Repeated indices are silently dropped
  otherwise, and in char-level text indices repeat constantly.
- **Weight tying means two gradient paths into `tok_emb`.** Accumulate both.
- **`(B,H,T,d)` vs `(B,T,H,d)`.** Pick one, write it down, check it at every reshape.
- **Initial loss must be `ln(V) ≈ 4.17`.** Best early check in the project.
- **Gradient-check in float64 on tiny shapes, op by op.** Never through the whole model, never
  through a `-inf`.
- **Don't decay LayerNorm gains, biases, or embeddings** in AdamW.
- **Seed everything.** Init, batch sampling, generation. Reproducibility is the acceptance
  criterion for three separate phases.
- **Scale residual output projections by `1/sqrt(2L)`.** Variance accumulates over residual adds.
- **PyTorch `nn.Linear` stores weights transposed** as `(out, in)`. Your oracle transfer will hit
  this.
- **Use `approximate='tanh'` for GELU** in the oracle to match your implementation.

---

# Appendix — Pairing with *How to Scale Your Model*

Austin et al., Google DeepMind, 2025. https://jax-ml.github.io/scaling-book/

## Why the pairing works

The book states roofline analysis as a **premise** and builds on it. This roadmap makes you
**derive** it on a model you wrote. Read cold, Chapter 1 is a plausible claim you accept on
authority. Read after Phase 4, it's recognition.

The book's own stated prerequisites are: basic understanding of the Transformer architecture, the
basics of LLM training, and some familiarity with JAX. **Phases 0–3 give you the second, Phase 4.5
gives you the third.** You'd be building the entry ticket without meaning to.

One connection worth holding onto: the book's intro notes that matmul is unusual in using far more
FLOPs per byte than almost any other algorithm, and that TPUs were designed around exactly that
property. That's the flip side of your Phase 4 finding — matmuls land right of the ridge; the
softmax, mask, and LayerNorm that surround them do not. **Attention is slow because of the
non-matmul parts, and that's a hardware-design fact, not an implementation detail.**

## Chapter → phase map

| Chapter | Phase | Relationship |
|---|---|---|
| **1. Roofline Analysis** | **Phase 4** | Direct. Their formalism, your measurements. Do their worked problems using your own numbers. |
| 2. How to Think About TPUs | Phase 6 (background) | Read paired with Ch. 12, not alone. Systolic array/MXU vs SM + tensor core is the useful contrast. |
| 3. Sharded Matrices | — | Above your layer. Single-device project. |
| **4. All the Transformer Math** | **Phases 1.9 + 4.1** | **Tightest match in the book.** They count params/FLOPs/KV-cache analytically; your `@counted` decorator measures the same things empirically. Run both and diff. |
| 5. Parallelizing for Training | — | Above your layer (FSDP, Megatron, pipeline). |
| 6. Training LLaMA 3 on TPUs | — | Above your layer. |
| **7. Transformer Inference** | **Phases 3.4 + 6.6** | Direct. Their KV-cache and latency discussion explains exactly why your `O(n·T²)` generation is bad and what fixes it. Read before writing 6.6. |
| 8. Serving LLaMA 3 | — | Above your layer. |
| **9. How to Profile TPU Code** | **Phase 4.5** | Their explanation of the JAX + XLA stack is the reference for your HLO reading. Read alongside 4.5.2. |
| **10. Programming TPUs in JAX** | **Phase 4.5 + 6.7** | Fusion, sharding APIs, worked problems. Their "compiler take the wheel vs. explicit control" framing is exactly the question 4.5.4 answers with evidence. |
| 11. Conclusions | — | Its further-reading list is worth mining. |
| **12. How to Think About GPUs** | **Phase 6 + `cuda-kernels`** | Bonus chapter, and for an NVIDIA target it's the most important one. Rooflines, networking, how they differ from TPUs. |

## Suggested reading order

1. **Nothing before Phase 4.** The book is extremely readable, and readable books about
   performance produce a strong, unearned sense of understanding. Build first.
2. **During Phase 4:** Chapters 1, 4, 12. This is the sweet spot — your own AI table open next to
   their analytic derivations, checking each other.
3. **During Phase 4.5:** Chapters 9, 10. Their XLA stack explanation while you're reading your own
   HLO dumps.
4. **Before Phase 6.6:** Chapter 7.
5. **Whenever:** Chapters 2, 3, 5, 6, 8. Genuinely new territory — this roadmap doesn't prepare you
   for multi-chip parallelism beyond giving you the vocabulary, and that's fine. It's the book's
   actual subject, not yours.

## Caveats

- **TPU-first.** Chapter 12 is a bonus GPU chapter, not the spine. For an NVIDIA target, read 2 and
  12 as a pair and treat the TPU material as instructive contrast rather than the target platform.
- **It's about the layer above you.** The book's center of gravity is multi-chip: sharding,
  AllGather/ReduceScatter, when communication overtakes computation. Your project never touches
  that. The overlap is Chapters 1, 4, 7, and 12.
- **Do the exercises.** Chapters 1 and 4 have worked problems. You have a model you can check them
  against, which almost no other reader does. Use that.