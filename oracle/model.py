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

def load_params(model, p):
    # nn.Linear stores weight as (out_features, in_features) -- transposed
    # relative to the numpy convention (x @ W with W as (in, out)) -- so every
    # Linear weight needs a .T on the way in.
    with torch.no_grad():
        model.tok_emb.weight.copy_(torch.from_numpy(p['tok_emb']))
        model.pos_emb.weight.copy_(torch.from_numpy(p['pos_emb']))

        for l, block in enumerate(model.blocks):
            block.ln1.weight.copy_(torch.from_numpy(p[f'b{l}.ln1.g']))
            block.ln1.bias.copy_(torch.from_numpy(p[f'b{l}.ln1.b']))

            block.Wq.weight.copy_(torch.from_numpy(p[f'b{l}.attn.Wq'].T.copy()))
            block.Wk.weight.copy_(torch.from_numpy(p[f'b{l}.attn.Wk'].T.copy()))
            block.Wv.weight.copy_(torch.from_numpy(p[f'b{l}.attn.Wv'].T.copy()))
            block.Wo.weight.copy_(torch.from_numpy(p[f'b{l}.attn.Wo'].T.copy()))

            block.ln2.weight.copy_(torch.from_numpy(p[f'b{l}.ln2.g']))
            block.ln2.bias.copy_(torch.from_numpy(p[f'b{l}.ln2.b']))

            block.mlp[0].weight.copy_(torch.from_numpy(p[f'b{l}.mlp.W1'].T.copy()))
            block.mlp[0].bias.copy_(torch.from_numpy(p[f'b{l}.mlp.b1']))
            block.mlp[2].weight.copy_(torch.from_numpy(p[f'b{l}.mlp.W2'].T.copy()))
            block.mlp[2].bias.copy_(torch.from_numpy(p[f'b{l}.mlp.b2']))

        model.lnf.weight.copy_(torch.from_numpy(p['lnf.g']))
        model.lnf.bias.copy_(torch.from_numpy(p['lnf.b']))