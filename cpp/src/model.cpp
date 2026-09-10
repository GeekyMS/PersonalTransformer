#include "model.h"
#include "ops.h"
#include "constants.h"

void block(Arena& arena, Tensor& x, BlockParams& p, Tensor& out) {
    Tensor flattened_x = x.reshape({x.shape[0] * x.shape[1], x.shape[2]});
    Tensor m = make_tensor(arena, flattened_x.shape);
    layer_norm(flattened_x, p.ln1_g, p.ln1_b, m);
    Tensor unflattened_m = m.reshape(x.shape);
    Tensor a = make_tensor(arena, x.shape);
    attention(arena, unflattened_m, p.Wk, p.Wq, p.Wv, p.Wo, kH, a);
    Tensor flattened_a = a.reshape(flattened_x.shape);
    Tensor res = make_tensor(arena, x.shape);
    Tensor flattened_res = res.reshape(flattened_x.shape);
    add(flattened_x, flattened_a, flattened_res);

    Tensor m2 = make_tensor(arena, flattened_x.shape);
    layer_norm(flattened_res, p.ln2_g, p.ln2_b, m2);
    Tensor a2 = make_tensor(arena, flattened_x.shape);
    mlp(arena, m2, p.W1, p.b1, p.W2, p.b2, a2);
    Tensor flattened_out = out.reshape(flattened_x.shape);
    add(flattened_res, a2, flattened_out);
}

void forward(Arena& arena, const std::vector<int>& x, ModelParams& p, Tensor& out) {
    Tensor h = make_tensor(arena, {kB, kT, kD});
    embed(x, p.tok_emb, p.pos_emb, h);

    for(int l = 0; l < kL; l++){
        Tensor h_next = make_tensor(arena, h.shape);
        block(arena, h, p.blocks[l], h_next);
        h = h_next;
    }

    Tensor flattened_h = h.reshape({h.shape[0] * h.shape[1], h.shape[2]});
    Tensor normed_h = make_tensor(arena, flattened_h.shape);
    layer_norm(flattened_h, p.lnf_g, p.lnf_b, normed_h);

    Tensor transposed_tok_emb = p.tok_emb.transpose(0, 1);
    Tensor flattened_out = out.reshape({out.shape[0] * out.shape[1], out.shape[2]});

    matmul(normed_h, transposed_tok_emb, flattened_out);
}
