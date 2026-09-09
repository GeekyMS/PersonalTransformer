"""
Manim visualization of safe_softmax (model.py:49-52)

Renders one row of v[b,h,t,:] moving through the five steps:
  max -> subtract -> exp -> sum -> divide

The point of the animation is the thing that matters for your C++ port:
every row is a fully independent 1-D problem. There's no cross-row state
at any point, so the outer (B,H,T) loop and the inner reduction loop are
cleanly separable.

Requires: pip install manim  (plus system deps: ffmpeg, a LaTeX install
is NOT required here since we only use Text, not MathTex/Tex).

Render with:
  manim -pql safe_softmax_manim.py SafeSoftmaxRow
(-pql = preview, quality low, fast; use -pqh for a high quality render)
"""

from manim import *
import numpy as np


class SafeSoftmaxRow(Scene):
    def construct(self):
        title = Text(
            "safe_softmax — one row (fixed b, h, query t)", font_size=28
        )
        title.to_edge(UP)
        self.play(Write(title))

        # ---- raw scores for this one row: v[b, h, t, :] ----
        values = [2.0, 5.0, 1.0, 3.0]
        row = self.make_row(values, YELLOW)
        row.move_to(UP * 2.2)
        row_label = Text("v[b,h,t,:]", font_size=22).next_to(row, LEFT)
        self.play(FadeIn(row), Write(row_label))
        self.wait(0.5)

        code = self.make_code_box(
            [
                "for b, h, t in outer_dims:",
                "    row = v[b][h][t];  // T floats",
            ]
        )
        code.to_corner(DR)
        self.play(FadeIn(code))
        self.wait(1)

        # ---- step 1: row-wise max (the reduction) ----
        step_label = self.step_text("1. max = row.max()  — a REDUCTION over this row")
        self.play(Write(step_label))
        max_idx = int(np.argmax(values))
        self.play(Indicate(row[max_idx], color=RED, scale_factor=1.35))
        self.replace_code(
            code,
            [
                "float m = -INFINITY;",
                "for (i = 0; i < T; i++)",
                "    m = max(m, row[i]);",
            ],
        )
        self.wait(1)
        self.play(FadeOut(step_label))

        # ---- step 2: subtract max (broadcast, still elementwise) ----
        step_label = self.step_text(
            "2. v_new[i] = v[i] - max  — every element, same scalar subtracted"
        )
        self.play(Write(step_label))
        new_values = [v - values[max_idx] for v in values]
        new_row = self.make_row(new_values, ORANGE)
        new_row.move_to(row.get_center() + DOWN * 1.3)
        self.play(TransformFromCopy(row, new_row))
        self.replace_code(
            code,
            [
                "for (i = 0; i < T; i++)",
                "    row[i] = row[i] - m;",
            ],
        )
        self.wait(1)
        self.play(FadeOut(step_label))

        # ---- step 3: exp (purely elementwise, no reduction at all) ----
        step_label = self.step_text(
            "3. exp(v_new[i])  — elementwise, zero interaction between i's"
        )
        self.play(Write(step_label))
        exp_values = [float(np.exp(v)) for v in new_values]
        exp_row = self.make_row([round(v, 2) for v in exp_values], GREEN)
        exp_row.move_to(new_row.get_center() + DOWN * 1.3)
        self.play(TransformFromCopy(new_row, exp_row))
        self.replace_code(
            code,
            [
                "for (i = 0; i < T; i++)",
                "    row[i] = expf(row[i]);",
            ],
        )
        self.wait(1)
        self.play(FadeOut(step_label))

        # ---- step 4: sum (the second reduction) ----
        step_label = self.step_text("4. denom = sum(row)  — a REDUCTION again")
        self.play(Write(step_label))
        total = float(sum(exp_values))
        sum_cell = self.make_cell(round(total, 2), BLUE)
        sum_cell.move_to(exp_row.get_center() + DOWN * 1.3 + LEFT * 2.2)
        sum_label = Text("sum", font_size=20).next_to(sum_cell, LEFT)
        self.play(TransformFromCopy(exp_row, sum_cell), Write(sum_label))
        self.replace_code(
            code,
            [
                "float s = 0.0f;",
                "for (i = 0; i < T; i++)",
                "    s += row[i];",
            ],
        )
        self.wait(1)
        self.play(FadeOut(step_label))

        # ---- step 5: divide (broadcast the scalar sum back out) ----
        step_label = self.step_text(
            "5. out[i] = row[i] / denom  — same scalar divided into every i"
        )
        self.play(Write(step_label))
        out_values = [round(v / total, 2) for v in exp_values]
        out_row = self.make_row(out_values, PURPLE)
        out_row.move_to(exp_row.get_center() + DOWN * 1.3)
        self.play(TransformFromCopy(exp_row, out_row))
        self.replace_code(
            code,
            [
                "for (i = 0; i < T; i++)",
                "    row[i] = row[i] / s;",
            ],
        )
        self.wait(1.5)
        self.play(FadeOut(step_label))

        outro = Text(
            "Every row runs this exact 4-pass loop independently.\n"
            "Outer loop over (B,H,T) is embarrassingly parallel;\n"
            "the inner max/sub/exp/sum/div passes can fuse into one loop\n"
            "over i if you keep a running max+sum (online softmax).",
            font_size=22,
            line_spacing=1.2,
        )
        outro.to_edge(DOWN)
        self.play(Write(outro))
        self.wait(3)

    # ---------- helpers ----------

    def make_cell(self, value, color):
        sq = Square(side_length=1.0, color=color)
        txt = Text(str(value), font_size=26)
        txt.move_to(sq.get_center())
        return VGroup(sq, txt)

    def make_row(self, values, color):
        cells = VGroup(*[self.make_cell(v, color) for v in values])
        cells.arrange(RIGHT, buff=0.15)
        return cells

    def step_text(self, s):
        t = Text(s, font_size=24)
        t.to_edge(DOWN, buff=1.3)
        return t

    def make_code_box(self, lines):
        txt = Text("\n".join(lines), font_size=18, font="Monospace")
        box = SurroundingRectangle(txt, buff=0.25, color=GREY)
        return VGroup(box, txt)

    def replace_code(self, code_group, lines):
        new_txt = Text("\n".join(lines), font_size=18, font="Monospace")
        new_box = SurroundingRectangle(new_txt, buff=0.25, color=GREY)
        new_group = VGroup(new_box, new_txt)
        new_group.move_to(code_group.get_center())
        self.play(Transform(code_group, new_group))
