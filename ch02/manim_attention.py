"""2장 어텐션 애니메이션: softmax(QKᵀ/√d + M) V

실행 (manim은 프로젝트 의존성과 분리해 `uv tool install --python 3.12 manim`으로 설치):
    manim -pql ch02/manim_attention.py MaskedAttention     # 저화질 미리보기
    manim -qh  ch02/manim_attention.py MaskedAttention     # 1080p
"""
import numpy as np
from manim import (
    MoveToTarget, Arrow, GrowArrow,
    Scene, VGroup, Rectangle, Text, MathTex, SurroundingRectangle,
    Create, FadeIn, FadeOut, Write, Transform, ReplacementTransform,
    LaggedStart, Indicate, AnimationGroup,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, GREY_B, GREY_D, YELLOW, BLUE, GREEN, RED, ORANGE, TEAL,
    interpolate_color, ManimColor,
)

FONT = "NanumGothic"
TOKENS = ["A", "cat", "sat", "on", "the"]
C, D, E = 5, 4, 3          # 토큰 수, 쿼리·키 차원, 밸류 차원
FOCUS = 2                  # 자세히 볼 쿼리 행: " sat"

# ── 예시 숫자: 화면에 보이는 값끼리 계산이 맞도록 반올림한 값으로 다시 계산 ──
rng = np.random.default_rng(7)
Qm = np.round(rng.normal(0, 1.2, (C, D)), 1)
Km = np.round(rng.normal(0, 1.2, (C, D)), 1)
Vm = np.round(rng.normal(0, 1.0, (C, E)), 1)
S = np.round(Qm @ Km.T, 1)                       # QKᵀ
S_scaled = np.round(S / np.sqrt(D), 2)            # ÷√d
mask = np.tril(np.ones((C, C), dtype=bool))
masked = np.where(mask, S_scaled, -np.inf)
W = np.exp(masked - masked.max(axis=1, keepdims=True))
W = np.round(W / W.sum(axis=1, keepdims=True), 2)
Out = np.round(W @ Vm, 2)


def txt(s, size=22, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def grid(values, cw, ch, fmt, size=24, stroke=GREY_B):
    """values(2D) → VGroup[행][열], 각 칸은 VGroup(사각형, 숫자)."""
    rows = VGroup()
    for i, row in enumerate(values):
        r = VGroup()
        for j, v in enumerate(row):
            box = Rectangle(width=cw, height=ch, stroke_color=stroke, stroke_width=1.5)
            box.move_to(RIGHT * j * cw + DOWN * i * ch)
            r.add(VGroup(box, txt(fmt(v), size).move_to(box)))
        rows.add(r)
    return rows


def f1(v):
    return f"{round(v, 1) + 0.0:.1f}"


def f2(v):
    return f"{round(v, 2) + 0.0:.2f}"


def row_labels(g, color=GREY_B, size=22):
    return VGroup(*[txt(t, size, color).next_to(g[i][0], LEFT, buff=0.15) for i, t in enumerate(TOKENS)])


def col_labels(g, color=GREY_B, size=22):
    return VGroup(*[txt(t, size, color).next_to(g[0][j], UP, buff=0.12) for j, t in enumerate(TOKENS)])


def heat(w):
    return interpolate_color(ManimColor("#1d2b3a"), ManimColor(ORANGE), float(w) ** 0.7)


class MaskedAttention(Scene):
    def construct(self):
        # ── 0. 수식 ────────────────────────────────────────────
        formula = MathTex(r"\mathrm{softmax}\Big(", r"QK^\top", r"/\sqrt{d}", r"+M",
                          r"\Big)", r"V", font_size=46).to_edge(UP, buff=0.3)
        title = txt("인과 마스크가 있는 셀프 어텐션", 34).next_to(formula, DOWN, buff=0.5)
        sub = txt('토큰 5개 "A cat sat on the",  d = 4', 24, GREY_B).next_to(title, DOWN, buff=0.25)
        self.play(Write(formula), FadeIn(title, shift=UP * 0.2))
        self.play(FadeIn(sub))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(sub))

        def focus(part_idx, color=YELLOW):
            return SurroundingRectangle(formula[part_idx], color=color, buff=0.08)

        # ── 1. QKᵀ ─────────────────────────────────────────────
        box = focus(1)
        self.play(Create(box))

        Sg = grid(S, 0.85, 0.55, f1)
        Sg.move_to(DOWN * 1.35 + LEFT * 0.3)
        Kg = grid(Km.T, 0.85, 0.45, f1, stroke=TEAL)
        Kg.next_to(Sg, UP, buff=0.5).align_to(Sg, LEFT)
        Qg = grid(Qm, 0.75, 0.55, f1, stroke=BLUE)
        Qg.next_to(Sg, LEFT, buff=0.35).align_to(Sg, UP)
        q_rows = row_labels(Qg, BLUE)
        k_cols = col_labels(Kg, TEAL)
        q_name = MathTex("Q", color=BLUE, font_size=36).next_to(Qg, DOWN, buff=0.15)
        k_name = MathTex(r"K^\top", color=TEAL, font_size=36).next_to(Kg, LEFT, buff=0.25)
        q_shape = txt("이전 (C, d) = (5, 4)", 18, GREY_B).next_to(q_name, RIGHT, buff=0.2)
        k_shape = txt("이전 (d, C) = (4, 5)", 18, GREY_B).next_to(k_name, DOWN, buff=0.15).align_to(k_name, RIGHT)

        self.play(FadeIn(Qg), FadeIn(q_rows), FadeIn(q_name), FadeIn(q_shape))
        self.play(FadeIn(Kg), FadeIn(k_cols), FadeIn(k_name), FadeIn(k_shape))

        # 칸 하나: sat 쿼리 · cat 키
        qi, kj = FOCUS, 1
        hq = SurroundingRectangle(VGroup(*Qg[qi]), color=YELLOW, buff=0.03)
        hk = SurroundingRectangle(VGroup(*[Kg[r][kj] for r in range(D)]), color=YELLOW, buff=0.03)
        prods = [f"({f1(Qm[qi, t])})({f1(Km[kj, t])})" for t in range(D)]
        calc = VGroup(
            txt(f'"{TOKENS[qi]}"의 쿼리 · "{TOKENS[kj]}"의 키', 22, YELLOW),
            txt(" + ".join(prods[:2]) + " +", 20),
            txt(" + ".join(prods[2:]), 20),
            txt(f"= {f1(S[qi, kj])}", 24, YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(Sg, RIGHT, buff=0.5).align_to(Sg, UP)
        self.play(Create(hq), Create(hk))
        self.play(FadeIn(calc, shift=LEFT * 0.2))
        self.play(FadeIn(Sg[qi][kj][0]), ReplacementTransform(calc[3].copy(), Sg[qi][kj][1]))
        self.wait(0.8)

        # 나머지 칸 모두
        rest = [Sg[i][j] for i in range(C) for j in range(C) if (i, j) != (qi, kj)]
        self.play(FadeOut(hq), FadeOut(hk), FadeOut(calc))
        self.play(LaggedStart(*[FadeIn(c) for c in rest], lag_ratio=0.04), run_time=2)
        self.play(FadeOut(q_name), FadeOut(q_shape))
        s_name = txt("이후 점수 (C, C): 행 = 쿼리(보는 쪽), 열 = 키(보이는 쪽)", 20, YELLOW)
        s_name.next_to(Sg, DOWN, buff=0.25)
        self.play(FadeIn(s_name))
        self.wait(1)

        # S를 왼쪽(이전 자리)으로 옮기고 토큰 라벨 붙이기
        self.play(FadeOut(VGroup(Qg, Kg, k_name, k_shape, k_cols)), FadeOut(s_name))
        s_rows = q_rows
        Lc, Rc = LEFT * 3.5 + DOWN * 0.3, RIGHT * 3.7 + DOWN * 0.3        # 이전 / 이후 행렬 중심
        KS = 0.85                                                          # 이전·이후를 나란히 놓으려고 축소
        FS = 24 * KS
        self.play(VGroup(Sg, s_rows).animate.scale(KS).shift(Lc - Sg.get_center()))
        self.play(Transform(s_rows, row_labels(Sg, size=20)), FadeIn(col_labels(Sg, size=20)))
        cur = VGroup(Sg, s_rows, self.mobjects[-1])     # (칸, 행 라벨, 열 라벨)

        def dim_anim(m, k=0.45):
            m.generate_target()
            for sub in m.target.family_members_with_points():
                sub.set_fill(opacity=sub.get_fill_opacity() * k)
                sub.set_stroke(opacity=sub.get_stroke_opacity() * k)
            return MoveToTarget(m)

        def note(*lines):
            g = VGroup(*[txt(l, 24, c) for l, c in lines]).arrange(DOWN, buff=0.15)
            return g.to_edge(DOWN, buff=0.35)

        def advance(cur, op):
            """cur(이전)를 흐리게 두고, 오른쪽에 복사본(이후)을 만든다. 화살표·라벨 포함."""
            new = cur.copy()
            self.play(new.animate.shift(Rc - cur[0].get_center()), run_time=1)
            self.play(dim_anim(cur))
            arrow = Arrow([cur[0].get_right()[0], Lc[1], 0], [new[1].get_left()[0], Lc[1], 0],
                          buff=0.2, color=GREY_B)
            op_t = txt(op, 22, WHITE).next_to(arrow, UP, buff=0.1)
            tags = VGroup(txt("이전", 22, GREY_B).next_to(cur[2], UP, buff=0.2),
                          txt("이후", 22, YELLOW).next_to(new[2], UP, buff=0.2))
            self.play(GrowArrow(arrow), FadeIn(op_t), FadeIn(tags))
            return new, VGroup(arrow, op_t, tags)

        def pair_box(cur, new, i, j, color):
            return VGroup(SurroundingRectangle(cur[0][i][j], color=color, buff=0.03, stroke_width=4),
                          SurroundingRectangle(new[0][i][j], color=color, buff=0.03, stroke_width=4))

        def retire(cur, new, deco):
            """이후 행렬을 이전 자리로 옮겨 다음 단계를 준비한다."""
            self.play(FadeOut(cur), FadeOut(deco))
            self.play(new.animate.shift(Lc - new[0].get_center()))
            return new

        # ── 2. ÷√d ─────────────────────────────────────────────
        self.play(Transform(box, focus(2)))
        new, deco = advance(cur, "÷ √d  (= ÷ 2)")
        g = new[0]
        self.play(*[Transform(g[i][j][1], txt(f2(S_scaled[i, j]), FS).move_to(g[i][j][0]))
                    for i in range(C) for j in range(C)], run_time=1.5)
        bi, bj = np.unravel_index(np.argmax(S), S.shape)        # 가장 큰 점수 칸
        hl = pair_box(cur, new, bi, bj, YELLOW)
        n2 = note((f"예: {f1(S[bi, bj])} → {f2(S_scaled[bi, bj])}   모든 칸을 √4 = 2로 나눈다", YELLOW),
                  ("내적의 분산을 1 근처로 맞춰 softmax 포화를 막는다", GREY_B))
        self.play(Create(hl), FadeIn(n2))
        self.wait(2)
        self.play(FadeOut(hl), FadeOut(n2))
        cur = retire(cur, new, deco)

        # ── 3. 인과 마스크 ──────────────────────────────────────
        self.play(Transform(box, focus(3, RED)))
        new, deco = advance(cur, "+ M")
        g = new[0]
        future = [(i, j) for i in range(C) for j in range(C) if j > i]
        self.play(LaggedStart(*[
            AnimationGroup(
                g[i][j][0].animate.set_fill(GREY_D, opacity=0.9),
                Transform(g[i][j][1], txt("−∞", FS, RED).move_to(g[i][j][0])),
            ) for i, j in future], lag_ratio=0.08), run_time=2)
        mi, mj = FOCUS, FOCUS + 1                                # sat 쿼리 → on 키 (미래)
        hl = pair_box(cur, new, mi, mj, RED)
        n3 = note((f'예: "{TOKENS[mi]}"가 "{TOKENS[mj]}"를 보는 칸  {f2(S_scaled[mi, mj])} → −∞', RED),
                  ("i번째 쿼리는 1 ~ i번째 키만 본다: 미래(오른쪽 위 삼각형)는 −∞", GREY_B))
        self.play(Create(hl), FadeIn(n3))
        self.wait(2)
        self.play(FadeOut(hl), FadeOut(n3))
        cur = retire(cur, new, deco)

        # ── 4. softmax (행마다) ─────────────────────────────────
        self.play(Transform(box, focus(0, ORANGE)))
        new, deco = advance(cur, "softmax (행마다)")
        g = new[0]

        def soft_cell(i, j):
            if j > i:   # 마스크된 칸: e^(−∞) = 0
                return Transform(g[i][j][1], txt("0", FS, GREY_B).move_to(g[i][j][0]))
            return AnimationGroup(
                g[i][j][0].animate.set_fill(heat(W[i, j]), opacity=1),
                Transform(g[i][j][1], txt(f2(W[i, j]), FS, WHITE if W[i, j] < 0.6 else "#1d2b3a")
                          .move_to(g[i][j][0])),
            )

        hrow = VGroup(SurroundingRectangle(VGroup(*cur[0][FOCUS]), color=ORANGE, buff=0.04, stroke_width=4),
                      SurroundingRectangle(VGroup(*g[FOCUS]), color=ORANGE, buff=0.04, stroke_width=4))
        self.play(Create(hrow))
        self.play(*[soft_cell(FOCUS, j) for j in range(C)], run_time=1.5)
        row_sum = " + ".join(f2(W[FOCUS, j]) for j in range(FOCUS + 1)) + " = 1"
        n4 = note((f'"{TOKENS[FOCUS]}" 행: −∞는 e^(−∞) = 0,  남은 칸끼리 {row_sum}', ORANGE),
                  ("행마다 따로 정규화한다", GREY_B))
        self.play(FadeIn(n4))
        self.wait(2)
        self.play(FadeOut(hrow), FadeOut(n4))
        self.play(*[soft_cell(i, j) for i in range(C) if i != FOCUS for j in range(C)], run_time=1.5)
        n5 = note(("어텐션 맵 (C, C): 행마다 합이 1,  오른쪽 위(미래)는 모두 0", WHITE),)
        self.play(FadeIn(n5))
        self.wait(1.5)
        self.play(FadeOut(n5))
        self.play(FadeOut(cur), FadeOut(deco))
        Sg, s_rows, s_cols = new

        # ── 5. × V ─────────────────────────────────────────────
        self.play(Transform(box, focus(5, GREEN)))
        attn = VGroup(Sg, s_rows, s_cols)
        k = 0.8
        self.play(attn.animate.scale(k / KS).to_edge(LEFT, buff=0.5).shift(DOWN * 0.2))
        times = MathTex(r"\times", font_size=44).next_to(Sg, RIGHT, buff=0.3)
        Vg = grid(Vm, 0.75 * k, 0.55 * k, f1, size=24 * k, stroke=GREEN).next_to(times, RIGHT, buff=0.3)
        Vg.align_to(Sg, UP)
        v_rows = VGroup(*[txt(t, 18, GREEN).next_to(Vg[i][-1], RIGHT, buff=0.1) for i, t in enumerate(TOKENS)])
        v_name = MathTex("V", color=GREEN, font_size=36).next_to(Vg, UP, buff=0.15)
        eq = MathTex("=", font_size=44).next_to(v_rows, RIGHT, buff=0.3)
        Og = grid(np.zeros_like(Out), 0.95 * k, 0.55 * k, lambda v: "", stroke=GREY_B)
        Og.next_to(eq, RIGHT, buff=0.3).align_to(Sg, UP)
        for r in Og:          # 빈 글자는 쓰지 않는다(숫자는 나중에 FadeIn)
            for cell in r:
                cell.remove(cell[1])
        o_name = txt("출력 (C, E)", 22).next_to(Og, UP, buff=0.15)
        self.play(FadeIn(times), FadeIn(Vg), FadeIn(v_rows), FadeIn(v_name))
        self.play(FadeIn(eq), FadeIn(Og), FadeIn(o_name))
        tags5 = VGroup(txt("이전", 22, GREY_B).next_to(VGroup(s_cols, v_name), UP, buff=0.2),
                       txt("이후", 22, YELLOW).next_to(o_name, UP, buff=0.2))
        self.play(FadeIn(tags5))

        def out_num(i, j, color=WHITE):
            return txt(f2(Out[i, j]), 24 * k, color).move_to(Og[i][j][0])

        # sat 행 = 0.xx·A + 0.xx·cat + 0.xx·sat
        hW = SurroundingRectangle(VGroup(*Sg[FOCUS]), color=YELLOW, buff=0.03)
        hV = VGroup(*[SurroundingRectangle(VGroup(*Vg[r]), color=YELLOW, buff=0.03) for r in range(FOCUS + 1)])
        ws = " + ".join(f"{W[FOCUS, r]:.2f}·{TOKENS[r]}" for r in range(FOCUS + 1))
        expl = txt(f'"{TOKENS[FOCUS]}"의 출력 = {ws}', 26, YELLOW).to_edge(DOWN, buff=0.6)
        expl2 = txt("on, the의 밸류는 가중치 0이라 섞이지 않는다", 22, GREY_B).next_to(expl, DOWN, buff=0.15)
        self.play(Create(hW), Create(hV), FadeIn(expl))
        focus_out = VGroup(*[out_num(FOCUS, j, YELLOW) for j in range(E)])
        self.play(LaggedStart(*[FadeIn(m, scale=1.3) for m in focus_out], lag_ratio=0.2))
        self.play(FadeIn(expl2))
        self.wait(1.5)
        self.play(FadeOut(hW), FadeOut(hV), FadeOut(expl), FadeOut(expl2))
        self.play(focus_out.animate.set_color(WHITE),
                  LaggedStart(*[FadeIn(out_num(i, j)) for i in range(C) if i != FOCUS for j in range(E)],
                              lag_ratio=0.05), run_time=1.5)

        # ── 6. 정리 ────────────────────────────────────────────
        self.play(FadeOut(box))
        summary = txt("입력 5토큰 → 출력 5행: 각 행은 자기와 과거 토큰의 밸류만 섞은 결과", 26, YELLOW)
        summary.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(summary, shift=UP * 0.2), Indicate(formula, color=YELLOW))
        self.wait(2.5)
