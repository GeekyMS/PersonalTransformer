#pragma once
// One transformer block and the full forward pass. Pure composition of the
// ops in ops.h -- mirrors np_impl/model.py's block()/forward() structure.
// See docs/roadmap.md Phase 1.7 / 5.4.

#include "arena.h"
#include "tensor.h"

// Weights for one block, matching np_impl/model.py's init_params() naming
// (p[f"b{l}.ln1.g"], p[f"b{l}.attn.Wq"], etc.) -- one BlockParams per layer.
struct BlockParams {
    Tensor ln1_g, ln1_b;
    Tensor Wq, Wk, Wv, Wo;
    Tensor ln2_g, ln2_b;
    Tensor W1, b1, W2, b2;
};

// Full model weights: token/position embeddings, one BlockParams per layer,
// final layer norm. tok_emb doubles as the (tied) output projection weight.
struct ModelParams {
    Tensor tok_emb, pos_emb;
    std::vector<BlockParams> blocks;   // size kL
    Tensor lnf_g, lnf_b;
};

// One pre-LN block: x + attention(layer_norm(x)), then x + mlp(layer_norm(x)).
// x, out: (B,T,d) -- kept 3D, NOT flattened to (B*T,d), because attention()
// needs real sequence boundaries for causal masking (a flattened view would
// let batch b's tokens attend across into batch b+1's). layer_norm/mlp/add
// don't care about that boundary (row-independent), so reshape to (B*T,d)
// locally, right before calling each of those, then treat the result as
// (B,T,d) again afterward -- same free reshape trick used elsewhere.
// Allocates its own intermediates from arena.
// TODO(you): implement -- pure composition of layer_norm, attention, add
// (need an elementwise add op -- do you have one yet?), and mlp.
void block(Arena& arena, Tensor& x, BlockParams& p, Tensor& out);

// Full forward: embed -> kL blocks -> final layer_norm -> output projection
// (x @ tok_emb.transpose(0,1), tied weights, no separate output matrix).
// x: token ids, length B*T (flattened, matching embed()'s convention).
// out: logits, (B,T,V) -- 3D for the same reason block()'s x/out are: keep
// the sequence boundary visible through the whole stack, only flatten
// locally around the row-independent ops. TODO(you): implement.
void forward(Arena& arena, const std::vector<int>& x, ModelParams& p, Tensor& out);
