import pickle
import csv

import numpy as np

from common.data import get_batch, encode, decode
from np_impl.model import init_params, forward, cross_entropy, safe_softmax, T, V
from np_impl.backward import backward


def save_checkpoint(p, path):
    with open(path, 'wb') as f:
        pickle.dump(p, f)


def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def should_decay(key):
    # only matmul weight matrices get weight decay -- not LayerNorm gains/biases,
    # not embeddings, not MLP biases. Matmul weights are the only ones named
    # "W..." at the end of their key (Wq, Wk, Wv, Wo, W1, W2).
    return key.split('.')[-1].startswith('W')


def adam_step(p, g, m, v, t, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
        mh = m[k] / (1 - b1 ** t)      # bias correction
        vh = v[k] / (1 - b2 ** t)
        decay = wd if should_decay(k) else 0.0
        p[k] -= lr * (mh / (np.sqrt(vh) + eps) + decay * p[k])


def lr_at(step, max_steps, warmup=100, lr_max=3e-4):
    if step < warmup:
        return lr_max * (step + 1) / warmup          # linear warmup, 0 -> lr_max
    progress = min((step - warmup) / max(1, max_steps - warmup), 1.0)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))     # 1 -> 0 over the remaining steps
    lr_min = lr_max / 10
    return lr_min + (lr_max - lr_min) * cosine


def clip_(grads, max_norm=1.0):
    total_norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for k in grads:
            grads[k] *= scale


def evaluate(p, split, B, T_, rng, n_batches=20):
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(split, B, T_, rng)
        logits, _ = forward(p, x)
        losses.append(cross_entropy(logits, y))
    return float(np.mean(losses))


def generate(p, prompt, n, rng, temp=0.8, top_k=40):
    idx = encode(prompt)
    for _ in range(n):
        ctx = idx[-T:]                                    # crop to context window
        logits, _ = forward(p, np.array([ctx]))            # (1, len(ctx), V)
        logits = logits[0, -1] / temp                       # last position only
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
        probs = safe_softmax(logits)
        idx.append(int(rng.choice(V, p=probs)))
    return decode(idx)


def train(max_steps=5000, B=32, T_=256, seed=0,
          ckpt_path='checkpoint.pkl', log_path='loss_log.csv'):
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'train_loss', 'val_loss'])

    for step in range(max_steps):
        x, y = get_batch('train', B, T_, rng)
        logits, caches = forward(p, x)
        loss = cross_entropy(logits, y)
        grads = backward(p, x, y, logits, caches)
        clip_(grads, 1.0)
        adam_step(p, grads, m, v, step + 1, lr_at(step, max_steps))

        if step % 100 == 0:
            val_loss = evaluate(p, 'val', B, T_, rng)
            print(f"{step}: train {loss:.4f}  val {val_loss:.4f}", flush=True)
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, float(loss), val_loss])
        if step % 500 == 0:
            print(generate(p, prompt="\n", n=300, rng=rng), flush=True)
            save_checkpoint(p, ckpt_path)   # periodic checkpoint -- crash/interrupt safety net

    save_checkpoint(p, ckpt_path)   # final save

    # fixed-seed sample -- this is explicitly your Phase 5 acceptance test per the roadmap
    fixed_rng = np.random.default_rng(1234)
    sample = generate(p, prompt="\n", n=500, rng=fixed_rng)
    with open('fixed_seed_sample.txt', 'w') as f:
        f.write(sample)

    return p


if __name__ == '__main__':
    train()
