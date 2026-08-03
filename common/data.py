from pathlib import Path
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
text = open(DATA_PATH).read()

chars = sorted(set(text))

V = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = np.array(encode(text), dtype=np.int64)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split, B, T, rng):
    data = train_data if split == 'train' else val_data
    ix = rng.integers(0, len(data) - T - 1, B)
    x = np.stack([data[i: i + T] for i in ix])
    y = np.stack([data[i + 1: i + T + 1] for i in ix])
    return x, y
