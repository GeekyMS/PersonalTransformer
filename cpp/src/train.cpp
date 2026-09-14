// Standalone C++ trainer -- no Python at runtime. Mirrors np_impl/train.py.
// See docs/roadmap.md Phase 5.4 "done when": same seed -> identical loss
// curve -> identical generated text as Phase 3, and a profiler confirming
// zero allocations in the steady-state loop (the whole point of Arena).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arena.h"
#include "constants.h"
#include "model.h"
#include "ops.h"
#include "tape.h"
#include "tensor.h"

// ---- data / tokenizer (file handling -- mechanical, filled in) ----

struct Dataset {
    std::string text;
    std::vector<char> itos;
    int stoi[256];                 // char -> id, -1 if not in vocab
    std::vector<int> train_ids, val_ids;
};

Dataset load_dataset(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("load_dataset: could not open '" + path +
                                  "' (cwd matters -- this is relative to wherever the binary is run from)");
    }
    std::stringstream ss;
    ss << f.rdbuf();

    Dataset d;
    d.text = ss.str();
    if (d.text.empty()) {
        throw std::runtime_error("load_dataset: '" + path + "' opened but is empty");
    }

    bool seen[256] = {false};
    for (unsigned char c : d.text) seen[c] = true;
    for (int i = 0; i < 256; i++) d.stoi[i] = -1;
    for (int i = 0; i < 256; i++) {
        if (seen[i]) {
            d.stoi[i] = (int)d.itos.size();
            d.itos.push_back((char)i);
        }
    }
    if ((int)d.itos.size() != kV) {
        std::fprintf(stderr, "warning: dataset vocab size %d != kV (%d)\n",
                     (int)d.itos.size(), kV);
    }

    std::vector<int> ids;
    ids.reserve(d.text.size());
    for (unsigned char c : d.text) ids.push_back(d.stoi[c]);

    size_t n = (size_t)(0.9 * ids.size());
    d.train_ids.assign(ids.begin(), ids.begin() + n);
    d.val_ids.assign(ids.begin() + n, ids.end());
    return d;
}

std::string decode(const Dataset& d, const std::vector<int>& ids) {
    std::string out;
    out.reserve(ids.size());
    for (int id : ids) out.push_back(d.itos[id]);
    return out;
}

std::vector<int> encode(const Dataset& d, const std::string& s) {
    std::vector<int> ids;
    ids.reserve(s.size());
    for (unsigned char c : s) ids.push_back(d.stoi[c]);
    return ids;
}

// Mirrors common/data.py's get_batch: B random windows of length T, x is the
// window, y is the same window shifted one position (next-token targets).
void get_batch(const std::vector<int>& data, std::mt19937& rng,
                std::vector<int>& x, std::vector<int>& y) {
    if (data.size() < (size_t)kT + 2) {
        // data.size() - kT - 2 is size_t (unsigned) -- if data is too small
        // this underflows to a huge number instead of going negative, and
        // uniform_int_distribution silently accepts it, handing out
        // wildly-out-of-bounds indices later. Fail loudly here instead.
        throw std::runtime_error("get_batch: dataset too small for kT=" + std::to_string(kT));
    }
    std::uniform_int_distribution<size_t> pick(0, data.size() - kT - 2);
    x.resize(kB * kT);
    y.resize(kB * kT);
    for (int b = 0; b < kB; b++) {
        size_t i = pick(rng);
        for (int t = 0; t < kT; t++) {
            x[b * kT + t] = data[i + t];
            y[b * kT + t] = data[i + t + 1];
        }
    }
}

// ---- params (mirrors np_impl/model.py's init_params) ----

// Every param tensor with its np_impl-style dotted name, in a fixed order --
// used for init, AdamW moment buffers, weight decay, and (de)serialization.
struct Param {
    std::string name;
    Tensor* t;
};

std::vector<Param> collect_params(ModelParams& p) {
    std::vector<Param> ps;
    ps.push_back({"tok_emb", &p.tok_emb});
    ps.push_back({"pos_emb", &p.pos_emb});
    for (int l = 0; l < kL; l++) {
        std::string pre = "b" + std::to_string(l) + ".";
        BlockParams& b = p.blocks[l];
        ps.push_back({pre + "ln1.g", &b.ln1_g});
        ps.push_back({pre + "ln1.b", &b.ln1_b});
        ps.push_back({pre + "attn.Wq", &b.Wq});
        ps.push_back({pre + "attn.Wk", &b.Wk});
        ps.push_back({pre + "attn.Wv", &b.Wv});
        ps.push_back({pre + "attn.Wo", &b.Wo});
        ps.push_back({pre + "ln2.g", &b.ln2_g});
        ps.push_back({pre + "ln2.b", &b.ln2_b});
        ps.push_back({pre + "mlp.W1", &b.W1});
        ps.push_back({pre + "mlp.b1", &b.b1});
        ps.push_back({pre + "mlp.W2", &b.W2});
        ps.push_back({pre + "mlp.b2", &b.b2});
    }
    ps.push_back({"lnf.g", &p.lnf_g});
    ps.push_back({"lnf.b", &p.lnf_b});
    return ps;
}

int numel(const Tensor& t) {
    int n = 1;
    for (int s : t.shape) n *= s;
    return n;
}

void fill_normal(Tensor& t, std::mt19937& rng, float mean, float std) {
    std::normal_distribution<float> dist(mean, std);
    int n = numel(t);
    for (int i = 0; i < n; i++) t.data[i] = dist(rng);
}

void fill_const(Tensor& t, float v) {
    int n = numel(t);
    for (int i = 0; i < n; i++) t.data[i] = v;
}

// Matches model.py:12-33 exactly: normal(0, 0.02) everywhere, except Wo and
// mlp.W2 get an extra /sqrt(2*kL) (residual-stream variance control across
// depth), LayerNorm gains start at 1, biases at 0.
ModelParams init_params(Arena& arena, std::mt19937& rng) {
    Tensor tok_emb = make_tensor(arena, {kV, kD});
    Tensor pos_emb = make_tensor(arena, {kT, kD});
    fill_normal(tok_emb, rng, 0.0f, 0.02f);
    fill_normal(pos_emb, rng, 0.0f, 0.02f);

    std::vector<BlockParams> blocks;
    blocks.reserve(kL);
    for (int l = 0; l < kL; l++) {
        Tensor ln1_g = make_tensor(arena, {kD}); fill_const(ln1_g, 1.0f);
        Tensor ln1_b = make_tensor(arena, {kD}); fill_const(ln1_b, 0.0f);

        Tensor Wq = make_tensor(arena, {kD, kD}); fill_normal(Wq, rng, 0.0f, 0.02f);
        Tensor Wk = make_tensor(arena, {kD, kD}); fill_normal(Wk, rng, 0.0f, 0.02f);
        Tensor Wv = make_tensor(arena, {kD, kD}); fill_normal(Wv, rng, 0.0f, 0.02f);
        Tensor Wo = make_tensor(arena, {kD, kD});
        fill_normal(Wo, rng, 0.0f, 0.02f / std::sqrt(2.0f * kL));

        Tensor ln2_g = make_tensor(arena, {kD}); fill_const(ln2_g, 1.0f);
        Tensor ln2_b = make_tensor(arena, {kD}); fill_const(ln2_b, 0.0f);

        Tensor W1 = make_tensor(arena, {kD, 4 * kD}); fill_normal(W1, rng, 0.0f, 0.02f);
        Tensor b1 = make_tensor(arena, {4 * kD}); fill_const(b1, 0.0f);
        Tensor W2 = make_tensor(arena, {4 * kD, kD});
        fill_normal(W2, rng, 0.0f, 0.02f / std::sqrt(2.0f * kL));
        Tensor b2 = make_tensor(arena, {kD}); fill_const(b2, 0.0f);

        blocks.push_back(BlockParams{ln1_g, ln1_b, Wq, Wk, Wv, Wo, ln2_g, ln2_b, W1, b1, W2, b2});
    }

    Tensor lnf_g = make_tensor(arena, {kD}); fill_const(lnf_g, 1.0f);
    Tensor lnf_b = make_tensor(arena, {kD}); fill_const(lnf_b, 0.0f);

    return ModelParams{tok_emb, pos_emb, blocks, lnf_g, lnf_b};
}

void zero_grads(std::vector<Param>& params) {
    for (Param& pr : params) {
        int n = numel(*pr.t);
        for (int i = 0; i < n; i++) pr.t->grad[i] = 0.0f;
    }
}

// ---- AdamW (mirrors np_impl/train.py's adam_step/lr_at/clip_) ----

// Only matmul weight matrices get weight decay -- not LayerNorm gains/
// biases, not embeddings, not MLP biases. Matches train.py:21-25: the last
// dot-separated component of the name starts with 'W'.
bool should_decay(const std::string& name) {
    size_t dot = name.find_last_of('.');
    std::string last = (dot == std::string::npos) ? name : name.substr(dot + 1);
    return !last.empty() && last[0] == 'W';
}

float lr_at(int step, int max_steps, int warmup = 100, float lr_max = 3e-4f) {
    if (step < warmup) return lr_max * (step + 1) / warmup;
    float progress = std::min((float)(step - warmup) / std::max(1, max_steps - warmup), 1.0f);
    float cosine = 0.5f * (1.0f + std::cos((float)M_PI * progress));
    float lr_min = lr_max / 10.0f;
    return lr_min + (lr_max - lr_min) * cosine;
}

void clip_grads(std::vector<Param>& params, float max_norm = 1.0f) {
    double total_sq = 0.0;
    for (Param& pr : params) {
        int n = numel(*pr.t);
        for (int i = 0; i < n; i++) total_sq += (double)pr.t->grad[i] * pr.t->grad[i];
    }
    float total_norm = (float)std::sqrt(total_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (Param& pr : params) {
            int n = numel(*pr.t);
            for (int i = 0; i < n; i++) pr.t->grad[i] *= scale;
        }
    }
}

struct AdamState {
    std::vector<std::vector<float>> m, v;   // one buffer per param, same order as `params`
};

AdamState make_adam_state(std::vector<Param>& params) {
    AdamState s;
    s.m.resize(params.size());
    s.v.resize(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        s.m[i].assign(numel(*params[i].t), 0.0f);
        s.v[i].assign(numel(*params[i].t), 0.0f);
    }
    return s;
}

void adam_step(std::vector<Param>& params, AdamState& s, int t, float lr,
               float b1 = 0.9f, float b2 = 0.95f, float eps = 1e-8f, float wd = 0.1f) {
    float bias1 = 1.0f - std::pow(b1, (float)t);
    float bias2 = 1.0f - std::pow(b2, (float)t);
    for (size_t k = 0; k < params.size(); k++) {
        Tensor& pt = *params[k].t;
        std::vector<float>& m = s.m[k];
        std::vector<float>& v = s.v[k];
        float decay = should_decay(params[k].name) ? wd : 0.0f;
        int n = numel(pt);
        for (int i = 0; i < n; i++) {
            float g = pt.grad[i];
            m[i] = b1 * m[i] + (1 - b1) * g;
            v[i] = b2 * v[i] + (1 - b2) * g * g;
            float mh = m[i] / bias1;
            float vh = v[i] / bias2;
            pt.data[i] -= lr * (mh / (std::sqrt(vh) + eps) + decay * pt.data[i]);
        }
    }
}

// ---- checkpointing: flat concatenation of every param, in collect_params
// order -- simple, not a fancy format, matches pickle's role of "crash/
// interrupt safety net", not a portable serialization scheme. ----

void save_checkpoint(std::vector<Param>& params, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    for (Param& pr : params) {
        f.write(reinterpret_cast<const char*>(pr.t->data), numel(*pr.t) * sizeof(float));
    }
}

void load_checkpoint(std::vector<Param>& params, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    for (Param& pr : params) {
        f.read(reinterpret_cast<char*>(pr.t->data), numel(*pr.t) * sizeof(float));
    }
}

// ---- eval / generation ----

float evaluate(ModelParams& p, Arena& scratch, const std::vector<int>& val_ids,
                std::mt19937& rng, int n_batches = 20) {
    float total = 0.0f;
    for (int i = 0; i < n_batches; i++) {
        std::vector<int> x, y;
        get_batch(val_ids, rng, x, y);

        scratch.reset();
        tape_reset();
        Tensor logits = make_tensor(scratch, {kB, kT, kV}, /*needs_grad=*/false);
        forward(scratch, x, p, logits);

        Tensor flat_logits = logits.reshape({kB * kT, kV});
        Tensor loss = make_tensor(scratch, {1}, /*needs_grad=*/false);
        cross_entropy(flat_logits, y, loss);
        total += loss.at({0});
    }
    return total / n_batches;
}

// Generation is constrained by forward()'s fixed (kB,kT) shape -- it has no
// path for a single variable-length sequence the way np_impl's generate()
// does. Workaround: replicate the (right-aligned, left-padded-with-token-0)
// prompt across all kB rows and run one full forward, then read row 0's
// last-position logits. Wasteful (kB x redundant compute) and not identical
// to train.py's growing-context generate(), but needs no changes to
// forward()'s shape contract. Revisit if this turns out to matter.
std::string generate(ModelParams& p, Arena& scratch, const Dataset& d,
                      const std::string& prompt, int n, std::mt19937& rng,
                      float temp = 0.8f, int top_k = 40) {
    std::vector<int> ids = encode(d, prompt);

    for (int step = 0; step < n; step++) {
        std::vector<int> ctx(kT, 0);
        int have = std::min((int)ids.size(), kT);
        for (int t = 0; t < have; t++) {
            ctx[kT - have + t] = ids[ids.size() - have + t];
        }
        std::vector<int> x(kB * kT);
        for (int b = 0; b < kB; b++)
            for (int t = 0; t < kT; t++) x[b * kT + t] = ctx[t];

        scratch.reset();
        tape_reset();
        Tensor logits = make_tensor(scratch, {kB, kT, kV}, /*needs_grad=*/false);
        forward(scratch, x, p, logits);

        int last_t = kT - 1;
        std::vector<float> row(kV);
        for (int j = 0; j < kV; j++) row[j] = logits.at({0, last_t, j}) / temp;

        int k = std::min(top_k, kV);
        std::vector<float> sorted_row = row;
        std::nth_element(sorted_row.begin(), sorted_row.end() - k, sorted_row.end());
        float kth = sorted_row[sorted_row.size() - k];
        for (int j = 0; j < kV; j++) if (row[j] < kth) row[j] = -std::numeric_limits<float>::infinity();

        float maxx = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        for (int j = 0; j < kV; j++) { row[j] = std::exp(row[j] - maxx); sum += row[j]; }
        for (int j = 0; j < kV; j++) row[j] /= sum;

        std::discrete_distribution<int> dist(row.begin(), row.end());
        ids.push_back(dist(rng));
    }
    return decode(d, ids);
}

// ---- training loop ----

int main() {
    const int kMaxSteps = 5000;
    const unsigned kSeed = 0;

    Dataset d = load_dataset("data/tinyshakespeare.txt");
    std::mt19937 rng(kSeed);

    // Params live in the front of the arena and must survive every step;
    // scratch (everything forward()/block() allocate) goes right after and
    // gets thrown away every step. arena.mark() right after init_params()
    // records that boundary -- arena.reset() from then on always rewinds to
    // just past the params, never touching them.
    // Capacity is a generous over-estimate, not the tight P5.1 accounting
    // (that exercise is explicitly deferred per project notes) -- if you
    // hit "alloc: no memory available", raise this.
    Arena arena(1'300'000'000);
    ModelParams p = init_params(arena, rng);
    arena.mark();

    std::vector<Param> params = collect_params(p);
    AdamState adam = make_adam_state(params);

    // Separate output dir from np_impl/train.py's (bench/numpy_run) so a run
    // of one implementation never clobbers the other's loss_log/checkpoint/
    // sample -- lets bench/plot_loss.py (or a manual diff) compare them.
    const std::string out_dir = "bench/cpp_run";
    std::filesystem::create_directories(out_dir);
    const std::string log_path = out_dir + "/loss_log.csv";
    const std::string ckpt_path = out_dir + "/checkpoint.bin";
    {
        std::ofstream f(log_path);
        f << "step,train_loss,val_loss,elapsed_sec\n";
    }

    using clock = std::chrono::steady_clock;
    auto train_start = clock::now();
    auto last_log = train_start;

    for (int step = 0; step < kMaxSteps; step++) {
        std::vector<int> x, y;
        get_batch(d.train_ids, rng, x, y);

        zero_grads(params);       // params persist across steps -- must re-zero before each backward
        arena.reset();            // rewind scratch to just past params (see mark() above)
        tape_reset();

        Tensor logits = make_tensor(arena, {kB, kT, kV});
        forward(arena, x, p, logits);

        Tensor flat_logits = logits.reshape({kB * kT, kV});
        Tensor loss = make_tensor(arena, {1});
        cross_entropy(flat_logits, y, loss);
        // No explicit loss.grad seeding: cross_entropy's backward closure
        // treats dL/dL = 1 implicitly (it's always the terminal op) -- see
        // ops.cpp's cross_entropy.
        tape_backward();

        clip_grads(params, 1.0f);
        adam_step(params, adam, step + 1, lr_at(step, kMaxSteps));

        if (step % 100 == 0) {
            auto now = clock::now();
            double sec_per_step = std::chrono::duration<double>(now - last_log).count() / (step > 0 ? 100 : 1);
            double total_elapsed = std::chrono::duration<double>(now - train_start).count();
            last_log = now;

            float val_loss = evaluate(p, arena, d.val_ids, rng);
            std::printf("%d: train %.4f  val %.4f  (%.3f s/step, %.1fs elapsed)\n",
                        step, loss.at({0}), val_loss, sec_per_step, total_elapsed);
            std::fflush(stdout);   // printf is fully buffered when redirected to a file
                                    // (slurm-%j.out) -- flush so progress shows up live,
                                    // matching train.py's print(..., flush=True)
            std::ofstream f(log_path, std::ios::app);
            f << step << "," << loss.at({0}) << "," << val_loss << "," << total_elapsed << "\n";
        }
        if (step % 500 == 0) {
            std::string sample = generate(p, arena, d, "\n", 300, rng);
            std::printf("%s\n", sample.c_str());
            std::fflush(stdout);
            save_checkpoint(params, ckpt_path);
        }
    }

    save_checkpoint(params, ckpt_path);

    double total_elapsed = std::chrono::duration<double>(clock::now() - train_start).count();
    {
        std::ofstream f(out_dir + "/timing.txt");
        f << "max_steps=" << kMaxSteps << "\n";
        f << "total_elapsed_sec=" << total_elapsed << "\n";
        f << "sec_per_step=" << (total_elapsed / kMaxSteps) << "\n";
    }
    std::printf("total training time: %.1fs (%.4f s/step)\n", total_elapsed, total_elapsed / kMaxSteps);

    // Fixed-seed sample -- the actual Phase 5 acceptance test per CLAUDE.md.
    std::mt19937 fixed_rng(1234);
    std::string sample = generate(p, arena, d, "\n", 500, fixed_rng);
    std::ofstream out(out_dir + "/fixed_seed_sample.txt");
    out << sample;

    return 0;
}
