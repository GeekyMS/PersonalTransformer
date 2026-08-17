# Paper exercises

Worksheets for the `P<phase>.<n>` exercises in [`docs/roadmap.md`](../../roadmap.md). Blank on
purpose.

**The rule:** fill in the *predicted* column with a pencil, off a piece of paper, before running
anything. Then run the code and fill in the *measured* column. Never edit a prediction after seeing
a measurement — add a note underneath instead. The archive of wrong predictions is the point.

Every exercise is one of two kinds, and they have different pass bars:

- **counting** — exact combinatorics, no hardware. Off by anything = miscounted.
- **modeling** — hardware in the loop. Within 2×, *and* you can name the dominant error term.

Log format for each: date locked → predicted → measured → ratio → what I got wrong.

| File | Phase |
|---|---|
| `00-conventions.md` | 0 — costing conventions, hardware sheet |
| `01-counting.md` | 1 — params, FLOPs, memory, transpose traffic |
| `02-backward.md` | 2 — backward cost, retention bill, softmax AI |
| `03-training.md` | 3 — time budget, optimizer, generation |
| `04-roofline.md` | 4 + 4.5 — per-op table, roofline, round-trips, exam, FA prediction |
| `05-cpp.md` | 5 — arena, cache blocking, port speedup |
| `06-cuda.md` | 6 — per-kernel predictions |
