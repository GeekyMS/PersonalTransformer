import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import V

B = 32          # batch size
T = 256         # context length
d = 256         # d_model
H = 4           # attention heads
d_head = d // H
L = 6           # layers

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(4 * d, d)
        )

    def forward(self, x):
        B_, T_, _ = x.shape

        a = self.ln1(x)
        q = self.Wq(a).view(B_, T_, H, d_head).transpose(1,2)
        k = self.Wk(a).view(B_, T_, H, d_head).transpose(1,2)
        v = self.Wv(a).view(B_, T_, H, d_head).transpose(1,2)

        o = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        o = o.transpose(1, 2).contiguous().view(B_, T_, d)
        x = x + self.Wo(o)

        m = self.ln2(x)
        x = x + self.mlp(m)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(d)

    def forward(self, x):
        B_, T_, = x.shape
        pos = torch.arange(T_, device = (x.device))
        h = self.tok_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            h = block(h)

        h = self.lnf(h)
        logits = F.linear(h, self.tok_emb.weight)
        return logits