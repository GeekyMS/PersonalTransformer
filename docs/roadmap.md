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

# How the paper exercises work

Every phase has **paper exercises**, marked `P<phase>.<n>`. Pencil, paper, *before* you run the
corresponding code. They are not optional and they are not busywork. The entire project is a bet
that you can predict machine behavior from first principles, and the only way to find out whether
you can is to write the prediction down somewhere you can't quietly revise it afterwards.

**The protocol is always the same:**

1. Derive **symbolically** first — in terms of `B, T, d, H, L, V, dh`. Never plug in numbers at the
   start. A symbolic result tells you how the cost *scales*; a number tells you almost nothing.
2. Then substitute, and get a number, with units, to two significant figures.
3. Write both into `docs/notes/paper/`, dated, **before** running anything.
4. Run the code / counter / profiler.
5. Write the measured number beside your prediction and compute the ratio. **Never erase the
   prediction.** The archive of wrong predictions is the record of what you learned.

**Two kinds of exercise, two different bars:**

| Kind | Example | Pass bar |
|---|---|---|
| **Counting** — exact combinatorics, no hardware in the loop | parameter count, bytes in `S`, FLOPs in a matmul | **exact.** Off by anything means you miscounted. |
| **Modeling** — hardware behavior enters | predicted step time, achieved bandwidth, kernel speedup | **within 2×**, *and* you can name the dominant term in the error |

Confusing the two is the commonest way to fool yourself. "Within 2×" is a triumph for a modeling
exercise and a failure for a counting one.

**Rules that make the numbers mean anything:**

- **Fix your conventions once (P0.1) and never silently change them.** A FLOP count that switches
  between counting an FMA as 1 and as 2 halfway through the project is worth nothing.
- **Carry units through every line.** Dimensional analysis catches most algebra errors before the
  arithmetic even starts. If a result should be in FLOP/byte and your line yields seconds, stop.
- **Two significant figures.** `12.53 FLOP/byte` implies precision you do not have and hides which
  term dominates. `~13` is the honest answer and the more useful one.
- **Estimate before you know.** Once you've seen the measurement you cannot un-see it, and the
  exercise is spent. There is no way to get it back.

When prediction and measurement disagree, **the gap is the lesson.** It is either a bug in the code
or a missing term in your model, and both are worth more than a matching pair of numbers. Chase
every gap over 2× until you can name its cause.

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

> **P0.1 — Costing conventions.** Do this *before* writing a single `count()` call, because every
> number in this project inherits from it.
>
> Decide and write down, in your notebook, permanently:
> - Is an FMA **1 FLOP or 2**? Pick one. Note which convention vendor peak-FLOP/s numbers use, so
>   you don't compare against a spec sheet in the other convention.
> - How do you count transcendentals — `exp`, `tanh`, `sqrt`, `rsqrt`? 1 FLOP each, or their real
>   cost? Look up the special-function-unit throughput ratio on your GPU and record it. Softmax and
>   GELU are made of these, so this choice moves your Phase 4 table.
> - **Bytes: you need two counters, not one.** *Compulsory* traffic — every input read once, every
>   output written once, infinite cache — and *actual* traffic, what your implementation really
>   moves, re-reads included. They answer different questions. Compulsory bytes give the arithmetic
>   intensity of the *algorithm*; actual bytes give the AI of your *code*. FlashAttention is the
>   story of closing the gap between them, so you must be able to see both.
> - Does an op's byte count include reading its own parameters? For a weight matrix reused across
>   the whole batch, how does that term scale with `B`?
>
> Then, symbolically and under **both** byte conventions, write the FLOP and byte formula for:
> matmul `(M,K)@(K,N)`; row-softmax over `(R,C)`; layernorm over `(N,d)`; GELU over `N` elements;
> elementwise add over `N`.
>
> **Gate:** your `count()` calls are literal transcriptions of formulas already on the page. If you
> find yourself deriving at the keyboard, close the editor and go back to paper.

> **P0.2 — Hardware sheet.** For every device you will run on (laptop CPU, GPU), fill in: peak fp32
> FLOP/s **and the derivation** (clock × cores × FMA width × 2, or whatever your machine's structure
> is — don't copy a marketing number you can't reconstruct), peak DRAM GB/s from clock × bus width ×
> channels, cache/SMEM capacities at each level, and the **ridge point** `FLOP/s ÷ GB/s`.
>
> Then measure both axes: a large square matmul for achieved FLOP/s, a large-vector triad
> (`a[i] = b[i] + s*c[i]`) for achieved GB/s. Recompute the **achieved** ridge point. That is the
> one that governs everything downstream; spec ridge is aspirational.
>
> **Gate:** you can state your machine's achieved ridge point from memory, and explain in one
> sentence why it differs from spec on each axis.

**Phase 0 done when:** you can pull a batch, print it as readable text, verify the shift
property, gradient-check a hand-written `f(x) = sum(x**2)` to `1e-6`, load your (empty)
params into the PyTorch oracle without a shape error, and `docs/notes/paper/00-conventions.md`
is filled in with P0.1 and P0.2 complete for every device you own.

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

> **P1.1 — Parameter count, by hand, before you run `init_params`.** One row per tensor: name,
> shape, count symbolically, count numerically. Sum per block, then the whole model, first as an
> expression in `V, d, L` and only then as a number.
>
> Then answer on paper: what fraction of the parameters are embeddings? At what `d`, holding
> everything else fixed, does the embedding fraction drop below 10%? Real models live on the far
> side of that crossover and yours does not — knowing which regime you're in explains a lot of
> otherwise-mysterious advice you'll read.
>
> **Gate: exact.** `sum(v.size for v in params.values())` must equal your hand number to the digit.
> This is a counting exercise; being close is being wrong.

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

> **P1.2 — What the transposes cost.** Free in metadata is not free in bytes. For the
> `(B,T,H,dh) → (B,H,T,dh)` pattern applied to `Q, K, V` on the way in and `O` on the way out:
> how many bytes physically move per layer, counting the read and the write? Multiply by `L`.
>
> Now compare that to the bytes in the score matrices `S`, symbolically. Find the `T` at which they
> cross. Then predict, **before computing**, which one dominates at `T=256` and which at `T=4096` —
> the two answers are not the same, and the fact that they aren't is why "attention is memory-bound"
> is a statement about a regime, not a law.
>
> **Gate: exact**, and you can state the crossover `T` as a formula in `H, dh` (equivalently `d`),
> not just as a number.

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

**Both exercises below get done on paper first, then the code prints the same table beside yours.**
The code is the grader here, not the source. If you write the script first you have thrown the
exercise away, and this is the one whose result you will use in every remaining phase.

> **P1.3 — Forward FLOPs, symbolically.** Derive the forward FLOP count for one block, then the full
> model including embeddings and the tied output projection. Keep the expression **split into two
> groups**: terms that scale as `T²` and terms that scale as `T`. Do not collapse them into one
> number — the split is the whole point.
>
> Then solve, algebraically, for the context length at which the `T²` terms equal the `T` terms for
> *this* model. That crossover is the single most important number in the project. You will cite it
> in Phase 4, again in Phase 6, and in every conversation you ever have about this repo. Express it
> symbolically in `d` (and any other constants that survive) before you evaluate it.
>
> Sanity question with a one-line answer: at `T=256`, is this model's compute dominated by attention
> or by the MLP? Most people's intuition about "transformers are attention" is wrong at short
> context, and the formula tells you exactly where the intuition starts being right.
>
> **Gate: exact** against your Phase 0.6 counter for a single forward pass, modulo your declared
> transcendental convention. Any mismatch is a bug in the counter or the derivation — find out
> which before continuing, because Phase 4 is built entirely on these two agreeing.

> **P1.4 — The memory table, at three context lengths.** For `B=32` and `T ∈ {256, 1024, 4096}`,
> by hand: parameter bytes, activation bytes retained across the forward pass, and the `S` matrices
> alone. Nine cells, symbolic then numeric.
>
> Then solve for `T` — algebraically, no binary search — at which `S` alone exceeds:
> (a) all parameters, (b) 16 GB, (c) the memory of the GPU you actually own.
>
> The roadmap already tells you the answer at `B=32, H=4, T=4096` is 8.6 GB for one layer. Use that
> as a check on your formula, not as a substitute for deriving it.
>
> **Gate: exact** against 1.9's output, and the three `T` values in closed form.

**Phase 1 done when:** with weights transferred to the PyTorch oracle, `np.allclose(my_logits,
torch_logits, atol=1e-5)` passes, untrained loss is `4.17 ± 0.05`, and P1.1–P1.4 are in
`docs/notes/paper/01-counting.md` with their measured columns filled in and every discrepancy
resolved.

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

> **P2.1 — Why backward costs what it costs.** For a single `C = A @ B`, count the FLOPs in the
> forward and in each of the two backward matmuls. State the backward:forward ratio as a small
> integer, derived, not recalled.
>
> Then find the ops in your model that **violate** that ratio and say why in one line each: the
> softmax backward, the embedding scatter-add, the LayerNorm backward, the fused cross-entropy.
> Some are cheaper than the rule predicts, some aren't matmuls at all.
>
> Finally predict total FLOPs for one full training step (forward + backward + optimizer) as a
> multiple of one forward pass, and as an absolute number at `B=32, T=256`.
>
> **Gate: within 10%** of the counter over a full step — modeling, not counting, because the
> non-matmul ops make the clean ratio an approximation. If you're off by more than that, the
> discrepancy is concentrated in one op; find which by diffing per-op counts.

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

> **P2.2 — Two adjacent lines, two different worlds.** Take `dS = P * (dP - (dP*P).sum(-1))` and
> the very next line, `dQ = dS @ K`. For each: FLOPs, compulsory bytes, arithmetic intensity, all
> in terms of `T` and `dh`. Compare both against your P0.2 achieved ridge point.
>
> **Predict the ratio between their two intensities before you compute either.** Then compute it.
> These two lines sit next to each other in the same function, operate on the same `(B,H,T,T)`
> tensor, and land on opposite sides of the ridge. That is the entire reason a fused attention
> kernel is worth writing, and you can see it here, in Phase 2, with no GPU involved.
>
> **Gate: exact** on both intensities, and a one-sentence written statement of which one you'd have
> to fix first to speed up the backward pass.

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

> **P2.3 — The retention bill.** Do this on paper *before you write a single backward function*,
> because the list you produce **is** the specification for what `forward` must return.
>
> Every tensor cached in the forward pass for use in the backward pass: name, shape, bytes, and the
> point in the step at which it is last read. Sum it, at `T=1024, B=32`. That sum is peak activation
> memory, and it is the reason activation checkpointing exists as a technique.
>
> Then the payoff question. Take the largest single cached tensor. If you *didn't* store it and
> recomputed it in the backward pass instead:
> - how many extra FLOPs does the recompute cost?
> - how many bytes of traffic does not storing it save?
> - form the ratio — extra FLOPs per byte saved — and compare it against your achieved ridge point
>   from P0.2.
>
> If that ratio sits below the ridge, the trade is free: you are paying in the resource you have
> spare to buy back the one you don't. **That comparison is the whole FlashAttention argument, and
> you just made it in Phase 2, on paper, months before you write the kernel.** Write the conclusion
> as one sentence with the two numbers in it, date it, and keep it — in Phase 4.6 you get to check
> it against the paper.
>
> **Gate:** bytes exact; the FLOP/byte verdict stated in one sentence with numbers, not adjectives.

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

**Phase 2 done when:** every op passes `check_grad` at `<1e-5`, every entry of your `grads`
dict matches PyTorch's `.grad` at `atol=1e-5`, and P2.1–P2.3 are written up with the retention
bill's recompute-vs-store verdict stated numerically.

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

> **P3.1 — The optimizer's roofline.** AdamW touches four arrays per parameter: `p, g, m, v`. Per
> step, symbolically: bytes moved, FLOPs performed, arithmetic intensity. Which side of the ridge?
>
> Then predict what fraction of wall-clock step time the optimizer should take if it ran at your
> achieved bandwidth from P0.2. Commit to the number *before* timing it. At 5M params with
> `B=32, T=256`, is it a rounding error or not? Guessing is not allowed here; the point is that the
> answer follows from two numbers you already have.
>
> Then time it and compare. If your measurement is far off your prediction, the likely causes are
> NumPy allocating a fresh temporary per line and the loop running per-key in Python — both of which
> you will eliminate in Phase 5, so record the number now.
>
> **Gate: within 2×**, plus the AI exactly.

## 3.2 Schedule

- Linear warmup over the first ~100 steps from 0 to `lr = 3e-4`
- Cosine decay to `lr/10` over the remaining steps
- Gradient clipping at global norm 1.0: compute `sqrt(Σ ||g||²)` across *all* params, and if it
  exceeds 1.0, scale everything down by the same factor

Warmup exists because Adam's `v` estimate is garbage for the first few dozen steps; taking full-
size steps then can push you somewhere unrecoverable.

> **P3.2 — The time budget, before your first real run.** You have per-step FLOPs from P2.1 and
> achieved FLOP/s from P0.2. Divide. Predict seconds per step, and total wall time for 5000 steps.
> Write it down, then run 10 steps and measure.
>
> `measured / predicted` is your **implementation efficiency**, and it will be embarrassing. That's
> expected and it's the point: Phases 5 and 6 are a campaign to move that one number, and a campaign
> needs a starting value.
>
> Then attribute the gap *before* profiling: list your top three suspected causes, ranked, with the
> fraction of the gap you think each accounts for. Candidates worth considering — NumPy materializing
> a temporary per line, single-threaded ops, memory-bound ops running at bandwidth rather than
> compute, Python loop overhead per op. You'll check this list against a real profile in Phase 4.
>
> **Gate: within 2× on the time, and your ranked attribution written down before you profile.**
> Being wrong about the ranking is fine and informative. Not committing to one is not.

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

> **P3.3 — Quantify the waste.** Count the FLOPs to generate 300 characters with this loop, at full
> context. Express it as a multiple of one training step — a training step processes `B·T = 8192`
> tokens, this generates 300, so the ratio should offend you.
>
> Now count what the same 300 characters would cost **with** a KV cache. You do not need to implement
> it to count it: per new token you compute one row of `Q`, attend against `T` cached keys, and skip
> every recomputation of the prefix. Write the FLOP expression for both and take the ratio.
>
> Then the more interesting half: with a KV cache, per generated token, how many **bytes** must be
> read (weights + cache) and how many FLOPs performed? What's the arithmetic intensity, and which
> side of the ridge is it on? Compare against the same quantity for a training step at `B=32`.
> **Training and single-stream inference are not the same problem on the same hardware**, and this
> exercise is where that stops being a slogan. Chapter 7 of the scaling book covers it; do the
> arithmetic yourself first.
>
> **Gate: exact** on both FLOP counts; the AI comparison stated with numbers on both sides.

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
to a file — they are Phase 5's acceptance test.** P3.1–P3.3 written up, with the implementation
efficiency number recorded somewhere you'll find it again in Phase 5.

---

# Phase 4 — Roofline analysis

*Estimated: 3–6 sessions. Output is a writeup, not code. This is the phase the whole project
exists for, and it is almost entirely done with a pencil.*

## 4.0 Ground rule for this entire phase: pencil first

Everything in 4.1–4.3 gets filled in **by hand, symbolically then numerically, before you run the
counter even once.** Then you run it and add a second column, and a third column with the ratio.

This is not a stylistic preference. By Phase 4 you have a counter that will produce the whole table
in about four seconds, and if you let it, you will read the numbers, nod, and retain nothing. The
counter's job in this phase is to **grade you**, and a grader you consult before answering is not a
grader. You've already done the hard version of every derivation in this phase during Phases 1–3;
this is where they get assembled.

Any row where hand and counter disagree by more than 5% is a bug in one of them. Find out which one
before you move on — a wrong counter poisons every remaining phase, and a wrong derivation poisons
you.

## 4.1 The per-op table

For each op at `B=32, T=1024`: FLOPs, bytes moved, arithmetic intensity (FLOPs/byte), measured
wall time.

> **P4.1 — Fill this in blank, in pencil, before running anything.** One row per op in a full
> forward+backward step. Columns: FLOPs (symbolic), FLOPs (numeric), compulsory bytes, actual bytes
> in *your* implementation, AI under each byte convention, **your predicted bound-by**, predicted
> time at achieved peak. Commit the bound-by column before measuring — that's the falsifiable part.
>
> The template is in `docs/notes/paper/04-roofline.md`. Note the two byte columns: compulsory bytes
> tell you what the *algorithm* demands, actual bytes what *your code* does. Ops where those two
> differ by a lot are exactly the ops a fused kernel would fix, and reading that column is how you
> generate the Phase 6 work list rather than being handed it.
>
> **Gate:** every FLOP and byte cell within 5% of the counter, every bound-by call correct, and for
> any you got wrong, a written sentence on what you mis-modeled.

Once your own table exists, here is the qualitative shape it should have. If one of your rows
disagrees with this, one of the two is wrong — find out which, don't assume it's you:

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

> **P4.2 — Draw the roofline by hand, on log-log graph paper, before you touch matplotlib.**
> Two axes, one sloped segment, one flat segment, one corner. Place every op from P4.1 on it as a
> point. Doing this once with a pencil is what makes the axes mean something; after that, plotting
> it is a rendering step and you can generate the pretty version however you like.
>
> Then compute the ridge point by hand for **three** devices: the GPU you own, an A100, and a
> current-generation part (H100, B200, or MI300X — pick one and cite the spec sheet). Tabulate
> peak FLOP/s, peak GB/s, and ridge for each.
>
> Then the question that matters: **has the ridge point moved up or down across those generations,
> and what does that imply for attention's memory-boundedness over time?** Answer in one sentence
> with the trend number in it. This is the single most quoted fact in performance engineering and
> almost everyone quoting it has never computed it. You will have computed it three times.
>
> **Gate:** three ridge points exact, the trend stated as a ratio, and a hand-drawn plot you'd be
> willing to photograph and put in the repo.

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

> **P4.3 — Count your own round-trips, separately for forward and backward.** Don't take the "6–8"
> above on faith; it's an estimate of a *typical* implementation, and yours is sitting in
> `np_impl/`. Read it line by line and count the passes it actually makes over a `T²` array.
> Forward and backward get separate counts — the backward touches `P` and `dP` and `dS`, and its
> number is larger.
>
> Then convert to bytes at `B=32, H=4, T=1024`, and divide by the bytes that a *perfect* kernel
> would move — one read of `Q, K, V`, one write of `O`, and nothing else. That quotient is the
> factor a fused kernel is playing for, and it is the number your Phase 6 benchmark table is
> ultimately compared against.
>
> **Gate: exact** on the pass counts, and you can point at the specific line of `np_impl/model.py`
> that causes each one.

## 4.4 The closed-book estimation exam

Everything up to here you did with your own code open. This is the same material with nothing open:
no notes, no calculator beyond arithmetic, no code. **Ten minutes per question, written answers,
two significant figures.**

The format matters. This is how these questions actually arrive — in an interview, in a design
review, in your own head at 2am deciding whether a kernel is worth three days. The skill being
tested is not the algebra, which you've already done. It's whether the model is *loaded*, close
enough to hand that you can run it without a reference.

1. **Score memory at long context.** Model constants as above, `T = 8192`. How much HBM do the
   score matrices need for one forward pass across all `L` layers, if nothing is freed? Now the
   same question if each layer frees its `S` before the next one starts. Which of those two numbers
   describes your NumPy implementation, and why?

2. **The attention fraction.** What fraction of total model FLOPs is attention at `T = 256`,
   `1024`, `8192`? Sketch the curve on the back of the page, label the asymptotes, and mark the
   crossover you derived in P1.2.

3. **Reshaping the model.** You double `d` and halve `L`. For each of parameter count, forward
   FLOPs, and peak activation memory: up, down, or flat, and by what factor? Three answers, one
   line of reasoning each. Then: which of those three would you have gotten wrong a month ago?

4. **A machine you've never used.** 3 TB/s HBM, 1000 TFLOP/s fp16 dense. Ridge point? Your `T=1024`
   attention layer, on a kernel achieving 40% of peak *bandwidth* — how long does it take? Same
   kernel achieving 40% of peak *compute* — how long? Which of those two numbers is meaningful for
   this workload, and why is the other one a category error?

5. **Amdahl on your own table.** Your softmax kernel reaches 60% of peak bandwidth. Using your P4.1
   table, what is the maximum possible remaining speedup for the *whole attention block* if softmax
   became instantaneous? What does that tell you about where to spend Phase 6?

6. **The fusion prize.** Fusing QKᵀ + scale + mask + softmax + PV into one kernel at
   `B=32, H=4, T=1024, dh=64`: how many bytes of HBM traffic does that eliminate? Express it as a
   multiple of the traffic that remains, not as an absolute — absolutes don't transfer between
   machines and ratios do.

7. **Batch size and the regime boundary.** At what batch size does your training step stop being
   dominated by fixed per-kernel overhead and start being bandwidth-bound on your hardware? You'll
   need a per-kernel launch/dispatch overhead estimate; state the one you're assuming and where it
   came from.

8. **Tile sizes, from capacity alone.** An SM has 48 KB of usable shared memory. fp32. You need a
   `Q` tile of `Bq × dh`, `K` and `V` tiles of `Bk × dh`, the running softmax state (`m` and `l`,
   one each per row of the `Q` tile), and the output accumulator. Write the capacity inequality and
   solve for the largest `Bq = Bk`. Now redo it in fp16. Now redo it with double buffering on the
   `K`/`V` tiles.

   **That last answer is the tile size you will use in Phase 6.4**, and you just derived it from
   nothing but a capacity constraint, months before writing the kernel. When you eventually read
   FlashAttention's tile-size discussion, you will recognize the inequality rather than learn it.

**Gate: 6 of 8 within 2×, closed book.** For any you miss, you must be able to reconstruct the
answer after seeing only the setup line again — if you can't, the model isn't loaded and the fix is
to redo the corresponding paper exercise, not to reread the answer.

Retake it before Phase 6 starts. The second sitting should be noticeably easier, and if it isn't,
that's worth knowing before you spend twenty sessions writing CUDA.

## 4.6 Then, and only then, read the FlashAttention paper

*(numbered 4.6, not 4.5, so it's never confused with Phase 4.5 below.)*

You will have independently derived its motivation. The paper stops being a clever trick and
becomes a description of the solution to a problem you personally measured.

This is the same roofline reasoning that justified halo tiling in your heat-diffusion stencil —
you fused passes to avoid re-reading global memory. Attention is that problem one level up, with
a softmax in the middle that makes the fusion non-obvious.

> **P4.6 — The prediction must contain numbers.** Before opening the paper, your written prediction
> of what an optimal attention kernel does has to commit to at least four quantities:
>
> 1. HBM traffic of the fused kernel, as a fraction of your Phase 4.3 count
> 2. Predicted speedup on the forward pass at `T=1024`, and separately at `T=4096` — the scaling
>    with `T` is the interesting half, and a single number hides it
> 3. Arithmetic intensity of the fused kernel, and which side of the ridge it lands on
> 4. What the backward pass has to do differently, and what that costs in extra FLOPs (you derived
>    this in P2.3 — go get the sentence you wrote and date-check yourself)
>
> A prediction with no numbers in it is a vibe, and vibes are not falsifiable. The whole value of
> this phase is that you can be *measurably* wrong here and then find out exactly where.
>
> Then read the paper and fill in a third column from their reported figures. Anything you missed by
> more than 2×: write a paragraph on what your model was missing. Anything you got right: note that
> too, because in Phase 6 you'll be tempted to believe the paper over your own analysis, and the
> record of where your analysis was already correct is what will stop you.

**Phase 4 done when:** you have the AI table (hand column and measured column, reconciled), the
hand-drawn roofline plus the plotted one, the round-trip count, a passing score on the closed-book
exam, and a paragraph in your own words predicting what an optimal attention kernel would do —
written *before* you read the paper, with numbers in it. Then read it and diff your prediction
against theirs.

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

> **P4.5.1 — Predict the compiler, on paper, before you dump anything.** Write down, in advance:
> how many kernels attention becomes after XLA optimizes it; which specific ops fuse into which;
> the shape and size of the largest buffer that survives; and total HBM traffic for the fused
> version, as a fraction of your Phase 4.3 hand count.
>
> Then dump the HLO and score yourself line by line. **Predicting a compiler's output is a distinct
> skill from predicting hardware**, and it's the one a compiler-engineering role actually tests. Two
> sessions of doing this badly is worth more than a month of reading about fusion passes.
>
> **Gate: kernel count within ±2, and every fusion you predicted either confirmed or explained.**
> Where XLA fused something you didn't expect, work out what property of the dataflow graph made it
> legal — that's the reusable lesson, not the specific fusion.

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

**Phase 4.5 done when:** you have optimized HLO dumps, an annotated fusion map for attention, your
scored P4.5.1 prediction, and a written argument for why the `T×T` materialization survives every
optimization pass.

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

> **P5.1 — Size the arena on paper.** Every buffer that must be simultaneously live at the moment
> of peak usage — which is somewhere in the backward pass, and part of the exercise is working out
> exactly where. Sum the bytes. That number is your arena size.
>
> You already did most of this as P2.3's retention bill; this adds the transient buffers that only
> exist within one op. Get it wrong low and you segfault; get it wrong high and you've quietly
> wasted memory you'll want at `T=1024`.
>
> Then: what is the **theoretical minimum** arena, if you were willing to aggressively reuse
> buffers whose last read has passed? Express the gap as a percentage. That gap is what a real
> memory planner in a compiler does for you, and having the number makes the next paragraph of any
> XLA buffer-assignment documentation you read actually mean something.
>
> **Gate: exact** on the naive sum; the tight bound within 10%, checked against your allocator's
> high-water mark.

> **P5.2 — Cache blocking, from capacity.** Same exercise as exam question 8, one level down the
> memory hierarchy and on hardware you own. Using your P0.2 cache sizes: what is the largest tile
> of the `(T,dh) × (dh,T)` matmul whose working set fits in L1? In L2? Write the capacity
> inequality, solve it, and predict the resulting GFLOP/s from the tile's arithmetic intensity and
> your achieved bandwidth at that level of the hierarchy.
>
> Then implement and measure. This is exactly the analysis behind the tiled matmul in your
> `cuda-kernels` repo — same inequality, different constants. You have benchmark numbers from that
> project; check whether the same reasoning reproduces them.
>
> **Gate: within 2× on predicted GFLOP/s**, and if you're outside that, the profiler tells you
> which term you dropped.

> **P5.3 — Predict the port's speedup before you run it.** Where will C++ beat NumPy, where will it
> tie, and where might it *lose*? Give a factor for each of: matmuls, elementwise ops, the softmax,
> the optimizer step, per-op Python overhead. Remember that NumPy's matmul calls into BLAS, which
> is not code you're going to beat by hand — being honest about that in advance is the point.
>
> Then check against P3.2's implementation-efficiency number: how much of that gap did the port
> actually close?
>
> **Gate: within 2× per category**, with the categories where you predicted "no gain" correctly
> identified. Predicting where you *won't* win is the harder and more valuable half.

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
(bit-identical is unrealistic across languages; match to 1e-4 and eyeball the sample), a
profiler confirms no allocations in the training loop, and P5.1–P5.3 are scored.

---

# Phase 6 — CUDA

*Estimated: 12–20 sessions. Folds into the `cuda-kernels` repo as its capstone.*

Each step gets a benchmark row in the same format as your stencil and matmul results:
GB/s achieved, % of peak, speedup over previous.

> **P6.0 — Every kernel gets a prediction column. No exceptions, and the prediction is written
> before the kernel is.** For each of 6.1–6.6, on paper, before you open the editor: FLOPs,
> compulsory bytes, arithmetic intensity, which side of the ridge, predicted runtime at your
> device's *achieved* peak (from P0.2), and predicted speedup over the previous step.
>
> Then implement, measure, and add the achieved columns. Your benchmark table has a **predicted**
> column beside every measured one, all the way down.
>
> **A kernel you cannot predict within 2× before writing is a kernel you do not understand yet.**
> When you're outside 2×, the difference between "my model was wrong" and "my kernel is bad" is
> itself the finding — occupancy, launch overhead, uncoalesced access, and bank conflicts all live
> in that gap, and they are much easier to recognize when you arrived with a number in hand.
>
> This is also the deliverable that makes the repo interesting to read. Anyone can post a benchmark
> table; a benchmark table with a prediction column and honest post-mortems on the misses is
> evidence of a mental model, which is the thing being hired for.

**6.1 Naive attention.** Separate kernels for QKᵀ, scale, mask, softmax, PV. This is your baseline
and it should be embarrassing. Measure it honestly — it's what everything else is compared against.

> **P6.1** — before writing it, predict its total HBM traffic and its runtime. You counted the
> round-trips for the NumPy version in P4.3; this is the same count on different hardware. That
> number is the target 6.2–6.4 spend the rest of the phase chipping away at, so it needs to exist
> on paper first, not be discovered afterwards.

**6.2 Fused softmax kernel.** One block per row, warp-level shuffle reductions for max and sum,
then a broadcast divide. Self-contained lesson in reductions. Expect a large speedup — you just
went from 3 HBM passes to 1.

**6.3 Online softmax, standalone.** Single pass with a running max `m` and running sum `l`. When a
new element exceeds the running max, rescale the accumulated sum by `exp(m_old − m_new)`.
**Implement and test this in isolation before putting it inside anything.** It's the load-bearing
idea and you don't want to debug it inside a tiled kernel.

> **P6.3 — Prove it on paper before you code it.** Two parts, and neither takes more than half a
> page:
> 1. **Exactness.** Show that processing the row in tiles, carrying `(m, l)` and rescaling on every
>    max update, gives *identically* the same result as the single-pass version in exact arithmetic.
>    Induction over tiles. If you can't write this proof, you will not be able to debug the kernel,
>    because you won't know whether a mismatch is a bug or expected drift.
> 2. **Error.** In fp32, bound the relative error of the online version against the two-pass one.
>    Which is actually more accurate, and why? The answer is not the one most people guess, and it
>    determines what tolerance you should be testing at in 6.4.
>
> **Gate:** the induction written out, and a numerically justified test tolerance — a number you
> derived, not one you tuned until the test passed.

**6.4 FlashAttention forward.** Load a `Q` tile into shared memory. Stream `K`/`V` tiles past it.
For each `K` tile: compute the partial scores, update the online softmax state, and rescale the
output accumulator. **`S` is never written to HBM.** You now have every prerequisite — shared-
memory tiling from the stencil, reductions from 6.2, online softmax from 6.3.

> **P6.4 — Go get exam question 8.** You derived the tile size from the shared-memory capacity
> inequality back in Phase 4. Redo it now with the real constraints you didn't have then: your
> actual SMEM per SM, your register budget, and a target occupancy. Does the answer move?
>
> Then predict, before measuring: what happens to runtime if you **halve** the tile size? There are
> two competing effects — one about how many times `K` and `V` get re-read from HBM, one about
> occupancy and latency hiding — and your prediction has to name both and say which wins. Then
> build both versions and measure.
>
> Being wrong here is common and instructive. Being wrong *without having predicted* teaches you
> nothing, because you'll retrofit an explanation to whichever number came out larger.

**6.5 FlashAttention backward.** Recompute `S` from `Q` and `K` in-kernel rather than reading a
stored `P`. Your Phase 2.2 derivation is the spec. Harder than the forward — budget for it.

**6.6 KV cache for inference.** A structurally different memory-bound problem: at generation time
you process one token at a time, so there's no batch dimension to amortize the weight loads. It's
purely bandwidth-limited on reading the weights. Good contrast, and it fixes the `O(n·T²)`
generation from Phase 3.4.

> **P6.6 — Predict tokens/second from bandwidth alone.** Per generated token you must read every
> weight once plus the KV cache. Bytes ÷ achieved GB/s = seconds per token, and that is the *entire*
> model — no FLOPs in it anywhere. Compute it, then measure, then explain the gap.
>
> Then: how does it change at batch size 32 instead of 1? Weight bytes stay fixed, cache bytes and
> FLOPs scale with batch. Write throughput and per-token latency as functions of batch size and find
> where the curve bends. **Batching is free until it isn't, and the bend is where.**
>
> You did the small version of this in P3.3. This is the same calculation on real hardware, and it's
> the one that generalizes to every inference-serving argument you'll ever read.
>
> **Gate: within 2×** on single-stream tokens/sec, and the batching curve sketched with the knee
> located.

**6.7 (optional) Pallas comparison.** Reimplement 6.4 in Pallas, JAX's Triton-like kernel DSL.
Roughly 40 lines against your 400. Benchmark both and write up the gap. Good "when is the
abstraction worth it" argument for a repo README. **Strictly after the raw CUDA, never instead of
it** — the hand-written kernel is the artifact that matters for an NVIDIA target.

**Phase 6 done when:** your CUDA attention matches the NumPy reference to fp32 tolerance, and the
benchmark table shows naive → fused softmax → tiled with **a predicted column beside every measured
one**, plus a written post-mortem for every prediction that missed by more than 2×.

---

# Appendix — Paper exercise index

Every exercise, its gate, and what it feeds. Nothing here takes more than a session; several take
twenty minutes. The `docs/notes/paper/` worksheets hold the blank templates.

| # | Phase | Exercise | Kind | Gate | Feeds |
|---|---|---|---|---|---|
| P0.1 | 0.6 | Costing conventions: FMA, transcendentals, compulsory vs actual bytes | counting | conventions fixed before any `count()` call | everything |
| P0.2 | 0.6 | Hardware sheet + ridge point, spec and achieved | modeling | ridge point from memory | P2.3, P3.2, P4.2, P6.0 |
| P1.1 | 1.1 | Parameter count per tensor, symbolic then numeric | counting | exact | P1.4, exam Q3 |
| P1.2 | 1.5 | Transpose traffic vs score-matrix traffic; find the crossover | counting | exact | P4.3 |
| P1.3 | 1.9 | Forward FLOPs split into `T²` and `T` terms; solve the crossover | counting | exact | P2.1, P4.1, exam Q2 |
| P1.4 | 1.9 | Memory table at three context lengths; solve for `T` at three thresholds | counting | exact | P4.1, exam Q1 |
| P2.1 | 2.0 | Backward:forward FLOP ratio, derived; the ops that violate it | modeling | within 10% | P3.2, P4.1 |
| P2.2 | 2.1 | AI of softmax backward vs `dS @ K`, two adjacent lines | counting | exact | P4.1 |
| P2.3 | 2.2 | Retention bill; recompute-vs-store FLOP/byte vs ridge | counting | bytes exact, verdict numeric | **P4.6, 6.5** |
| P3.1 | 3.1 | AdamW's arithmetic intensity and its share of step time | modeling | within 2× | P5.1 |
| P3.2 | 3.2 | Predicted step time; implementation efficiency; ranked attribution | modeling | within 2× | P5.3 |
| P3.3 | 3.4 | Generation FLOPs, with and without a KV cache; the AI comparison | counting | exact | P6.6 |
| P4.1 | 4.1 | The full per-op table, hand column before counter column | counting | 5% per cell, bound-by correct | all of Phase 6 |
| P4.2 | 4.2 | Hand-drawn roofline; ridge point for three devices; the trend | modeling | ridge exact, trend as ratio | P6.0 |
| P4.3 | 4.3 | Round-trip count over `T²`, forward and backward separately | counting | exact, line-attributed | P6.1 |
| **P4.4** | **4.4** | **Closed-book estimation exam, 8 questions** | mixed | **6/8 within 2×** | retake before Phase 6 |
| P4.6 | 4.6 | Numeric FlashAttention prediction, then diff against the paper | modeling | 4 numbers committed pre-read | Phase 6 |
| P4.5.1 | Ph 4.5.3 | Predict XLA's fusion output before dumping HLO | modeling | kernel count ±2 | the 4.5.4 writeup |
| P5.1 | 5.2 | Arena sizing; naive sum and the tight bound | counting | exact / 10% | the allocator |
| P5.2 | 5.2 | Cache-blocking inequality per level; predicted GFLOP/s | modeling | within 2× | P6.4 |
| P5.3 | 5.4 | Where C++ beats NumPy, where it ties, where it loses | modeling | within 2× per category | — |
| P6.0 | 6 | A prediction column for every kernel, written before the kernel | modeling | within 2× or a post-mortem | the benchmark table |
| P6.1 | 6.1 | Naive kernel HBM traffic and runtime | modeling | within 2× | 6.2–6.4 targets |
| P6.3 | 6.3 | Prove online softmax exact by induction; bound the fp32 error | counting | proof written, tolerance derived | 6.4 test suite |
| P6.4 | 6.4 | Tile size from real SMEM/register/occupancy limits; halve-the-tile | modeling | both effects named | the kernel |
| P6.6 | 6.6 | Tokens/sec from bandwidth alone; the batching knee | modeling | within 2× | — |

**If you only do five of these:** P0.2, P1.3, P2.3, P4.1, P4.4. That subset alone gets you to a
defensible answer for "why is attention slow and what would you do about it," which is the question
the whole project is training you to answer without looking anything up.

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
- **Never run the counter before the pencil.** Once you've seen a number you can't un-see it, and
  the exercise is gone. There is no way to recover it later.
- **Don't silently change a costing convention mid-project.** If you do change one, redo every
  affected number and date the change in `00-conventions.md`.
- **Counting exercises are exact; modeling exercises are 2×.** Applying the wrong bar in either
  direction is how you end up either chasing float noise or accepting a real bug.

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
  against, which almost no other reader does. Use that — work each of their problems with your own
  constants substituted in, and when their analytic result and your measured counter disagree, that
  disagreement is a better exercise than the problem was. Their Chapter 4 is essentially P1.3 and
  P1.4 done by someone else; do yours first, then read theirs as a marking scheme.