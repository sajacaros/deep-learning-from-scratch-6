"""2장 멀티헤드 어텐션 애니메이션: 텐서 형상 흐름

실행 (manim은 `uv tool install --python 3.12 manim`으로 따로 설치):
    manim -pql ch02/manim_multihead.py MultiHeadShapes     # 저화질 미리보기
    manim -qh  ch02/manim_multihead.py MultiHeadShapes     # 1080p
"""
import numpy as np
from manim import (
    Scene, VGroup, Square, Rectangle, Text, MathTex, Arrow, SurroundingRectangle,
    Create, FadeIn, FadeOut, Write, Transform, ReplacementTransform, LaggedStart,
    Indicate, GrowArrow, MoveToTarget,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    WHITE, GREY_B, GREY_D, YELLOW, BLUE_C, GREEN_C, ORANGE, PINK,
    ManimColor, interpolate_color,
)

FONT = "NanumGothic"
MONO = "DejaVu Sans Mono"
TOKENS = ["A", "cat", "sat", "on", "the"]
C, H, D, E = 5, 3, 2, 6          # 토큰 수, 헤드 수, 헤드 차원, 임베딩 차원
HEAD_COLORS = [BLUE_C, GREEN_C, ORANGE]
FOCUS_ROW = 2               # 자세히 볼 행: " sat"
CELL = 0.36


def txt(s, size=24, color=WHITE):
    return Text(s, font=FONT, font_size=size, color=color)


def code(s, size=24, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def blocks(rows, cols, color_of, cell=CELL, opacity_of=None):
    """rows×cols 칸 VGroup[행][열]. color_of(i, j) → 색, opacity_of(i, j) → 채움 투명도."""
    g = VGroup()
    for i in range(rows):
        r = VGroup()
        for j in range(cols):
            op = 0.85 if opacity_of is None else opacity_of(i, j)
            sq = Square(cell, stroke_color=GREY_B, stroke_width=1.2)
            sq.set_fill(color_of(i, j), opacity=op)
            sq.move_to(RIGHT * j * cell + DOWN * i * cell)
            r.add(sq)
        g.add(r)
    return g


def cols_of(g, j0, j1):
    """g의 j0 ≤ j < j1 열만 모은 VGroup (행 순서 유지)."""
    return VGroup(*[VGroup(*[g[i][j] for j in range(j0, j1)]) for i in range(len(g))])


def head_matrix(h, cell=CELL):
    return blocks(C, D, lambda i, j: HEAD_COLORS[h], cell)


def token_labels(g, size=20):
    return VGroup(*[txt(t, size, GREY_B).next_to(g[i][0], LEFT, buff=0.12) for i, t in enumerate(TOKENS)])


CHANGED = PINK   # 바뀐 축


def shape_text(axes, hi, color, size=28):
    """(B, C, H·D) 문자열에서 hi 위치의 축만 CHANGED 색으로."""
    s, spans, pos = "(", [], 1
    for k, a in enumerate(axes):
        if k:
            s += ", "
            pos += 2
        if k in hi:
            spans.append((pos, pos + len(a)))
        s += a
        pos += len(a)
    s += ")"
    return Text(s, font=MONO, font_size=size, color=color,
                t2c={f"[{i}:{j}]": CHANGED for i, j in spans})


def shape_bar(before, after=None, op="", hb=(), ha=()):
    """아래쪽 형상 줄. after가 없으면 현재 형상 하나만."""
    if after is None:
        bar = VGroup(shape_text(before, (), YELLOW))
    else:
        left = shape_text(before, hb, GREY_B)
        right = shape_text(after, ha, YELLOW)
        arrow = Arrow(LEFT, RIGHT, buff=0, color=GREY_B, stroke_width=3,
                      max_tip_length_to_length_ratio=0.15).set_length(max(1.6, len(op) * 0.17 + 0.6))
        body = VGroup(left, arrow, right).arrange(RIGHT, buff=0.3)
        op_t = Text(op, font=MONO, font_size=20, color=GREY_B).next_to(arrow, UP, buff=0.08)
        tag_b = Text("이전", font=FONT, font_size=18, color=GREY_B).next_to(left, UP, buff=0.1)
        tag_a = Text("이후", font=FONT, font_size=18, color=YELLOW).next_to(right, UP, buff=0.1)
        bar = VGroup(body, op_t, tag_b, tag_a)
    return bar.to_edge(DOWN, buff=0.35)


def before_after_arrow(left, right, label):
    y = left.get_center()[1]
    arrow = Arrow([left.get_right()[0], y, 0], [right.get_left()[0], y, 0], buff=0.25, color=GREY_B)
    lab = Text(label, font=MONO, font_size=18, color=GREY_B).next_to(arrow, UP, buff=0.08)
    return VGroup(arrow, lab)


def stage_tags(left, right):
    return VGroup(Text("이전", font=FONT, font_size=20, color=GREY_B).next_to(left, UP, buff=0.55),
                  Text("이후", font=FONT, font_size=20, color=YELLOW).next_to(right, UP, buff=0.55))


def ghost(m, opacity=0.3):
    g = m.copy()
    for sub in g.family_members_with_points():
        sub.set_fill(opacity=sub.get_fill_opacity() * opacity)
        sub.set_stroke(opacity=opacity)
    return g


# 헤드마다 다른 관점: ① 자기 자신 ② 바로 앞 토큰 ③ 첫 토큰
def attention_pattern(h):
    rng = np.random.default_rng(h)
    s = rng.normal(0, 0.4, (C, C))
    for i in range(C):
        if h == 0:
            s[i, i] += 2.5
        elif h == 1:
            s[i, max(i - 1, 0)] += 2.5
        else:
            s[i, 0] += 2.5
    s = np.where(np.tril(np.ones((C, C), bool)), s, -np.inf)
    w = np.exp(s - s.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


class MultiHeadShapes(Scene):
    def construct(self):
        # ── 0. 제목 ────────────────────────────────────────────
        title = txt("멀티헤드 어텐션: 텐서 형상 흐름", 40)
        sub = txt(f"C = {C} 토큰,  E = {E},  H = {H} 헤드,  D = {D}   (배치 B는 하나만 그린다)", 24, GREY_B)
        VGroup(title, sub).arrange(DOWN, buff=0.35)
        legend = VGroup(*[
            VGroup(Square(0.3).set_fill(c, 0.85).set_stroke(GREY_B, 1), txt(f"head {h + 1}", 22, c)).arrange(RIGHT, buff=0.15)
            for h, c in enumerate(HEAD_COLORS)
        ]).arrange(RIGHT, buff=0.6).next_to(sub, DOWN, buff=0.5)
        self.play(FadeIn(title, shift=UP * 0.2))
        self.play(FadeIn(sub), FadeIn(legend))
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, sub, legend)))

        # 위: 코드 한 줄, 아래: 형상 변화  "이전 ──연산──▶ 이후"
        code_line = code("x", 26).to_edge(UP, buff=0.35)

        def set_code(s):
            return Transform(code_line, code(s, 26).to_edge(UP, buff=0.35))

        shape = shape_bar(["B", "C", "E"])

        def set_shape(before, after, op, hb, ha):
            return Transform(shape, shape_bar(before, after, op, hb, ha))

        # ── 1. x → W_q(x): 모든 헤드의 Q를 한 번에 ──────────────
        x = blocks(C, E, lambda i, j: GREY_D).move_to(LEFT * 4.2)
        x_tok = token_labels(x)
        x_name = MathTex("x", font_size=40).next_to(x, UP, buff=0.2)
        self.play(FadeIn(code_line), FadeIn(shape), FadeIn(x), FadeIn(x_tok), FadeIn(x_name))
        self.wait(0.5)

        Q = blocks(C, H * D, lambda i, j: HEAD_COLORS[j // D]).move_to(RIGHT * 1.6)
        q_tok = token_labels(Q)
        arrow = Arrow(x.get_right(), q_tok.get_left(), buff=0.2, color=GREY_B)
        w_label = code("nn.Linear(E, H*D)", 20, GREY_B).next_to(arrow, UP, buff=0.1)
        q_name = MathTex("Q", font_size=40).next_to(Q, UP, buff=0.2)
        note1 = txt("열 6개 = 헤드 3개 × D 2개", 24).next_to(Q, RIGHT, buff=0.6)
        self.play(set_code("Q = W_q(x)"), GrowArrow(arrow), FadeIn(w_label))
        self.play(FadeIn(Q, shift=RIGHT * 0.3), FadeIn(q_tok), FadeIn(q_name), set_shape(["B", "C", "E"], ["B", "C", "H·D"], "W_q", [2], [2]))
        self.play(FadeIn(note1))
        self.wait(1.5)

        # ── 2. view: 마지막 축을 (H, D)로 쪼갠다 ─────────────────
        self.play(FadeOut(VGroup(x, x_tok, x_name, arrow, w_label, note1)))
        L, R = LEFT * 3.7 + UP * 0.3, RIGHT * 2.9 + UP * 0.3      # 이전 / 이후 자리
        self.play(VGroup(Q, q_tok).animate.move_to(L), FadeOut(q_name))
        prev = ghost(VGroup(Q, q_tok))
        self.add(prev)
        parts = [cols_of(Q, h * D, (h + 1) * D) for h in range(H)]
        self.play(VGroup(Q, q_tok).animate.move_to(R), run_time=1)
        self.play(set_code("Q = W_q(x).view(B, C, H, D)"),
                  *[p.animate.shift(RIGHT * 0.35 * (h - 1)) for h, p in enumerate(parts)],
                  q_tok.animate.shift(LEFT * 0.35),
                  set_shape(["B", "C", "H·D"], ["B", "C", "H", "D"], ".view", [2], [2, 3]))
        head_tags = VGroup(*[txt(f"head {h + 1}", 20, HEAD_COLORS[h]).next_to(p, UP, buff=0.15)
                             for h, p in enumerate(parts)])
        ba = before_after_arrow(prev, VGroup(*parts, q_tok), ".view(B, C, H, D)")
        st = stage_tags(prev, VGroup(*parts))
        self.play(FadeIn(head_tags), GrowArrow(ba[0]), FadeIn(ba[1]), FadeIn(st))
        note2 = txt("값은 그대로, 마지막 축 6을 (3, 2)로 묶어 볼 뿐", 24, GREY_B).move_to(DOWN * 2.0)
        self.play(FadeIn(note2))
        self.wait(1.5)

        # ── 3. transpose(1, 2): H를 앞으로 → 헤드가 배치처럼 ─────
        after2 = VGroup(*parts, q_tok, head_tags)
        self.play(FadeOut(prev), FadeOut(ba), FadeOut(st), FadeOut(note2))
        self.play(after2.animate.shift(L - R))
        prev = ghost(after2)
        self.add(prev)
        offset = np.array([0.45, 0.4, 0])
        front = R + LEFT * 0.6 + DOWN * 0.45
        targets = [front + offset * h for h in range(H)]   # head 1이 맨 앞(왼쪽 아래)
        for h, p in enumerate(parts):
            p.set_z_index(H - h)
        self.play(set_code("Q = W_q(x).view(B, C, H, D).transpose(1, 2)"),
                  FadeOut(head_tags), FadeOut(q_tok),
                  *[p.animate.move_to(targets[h]) for h, p in enumerate(parts)],
                  set_shape(["B", "C", "H", "D"], ["B", "H", "C", "D"], ".transpose(1, 2)", [1, 2], [1, 2]),
                  run_time=2)
        q_tok = token_labels(parts[0])
        deck_right = VGroup(*parts).get_right()[0]
        deck_tags = VGroup(*[txt(f"head {h + 1}", 20, HEAD_COLORS[h])
                             .move_to([deck_right + 0.55, parts[h].get_top()[1] - 0.12, 0]) for h in range(H)])
        ba = before_after_arrow(prev, VGroup(*parts, q_tok), ".transpose(1, 2)")
        st = stage_tags(prev, VGroup(*parts))
        self.play(FadeIn(q_tok), FadeIn(deck_tags), GrowArrow(ba[0]), FadeIn(ba[1]), FadeIn(st))
        note3 = txt("H가 C 앞으로 → 헤드 3개가 (C, D) 행렬 3장으로, 배치처럼 쌓인다", 24, GREY_B).move_to(DOWN * 2.0)
        self.play(FadeIn(note3))
        self.wait(2)

        # K, V도 같은 방식
        q_deck = VGroup(*parts)
        self.play(FadeOut(note3), FadeOut(deck_tags), FadeOut(prev), FadeOut(ba), FadeOut(st))
        self.play(VGroup(q_deck, q_tok).animate.scale(0.8).move_to(LEFT * 4.2 + DOWN * 0.2),
                  set_code("Q, K, V  각각 (B, H, C, D)"))

        def deck(cell=CELL * 0.8):
            ms = [head_matrix(h, cell) for h in range(H)]
            for h, m in enumerate(ms):
                m.move_to(offset * 0.8 * h).set_z_index(H - h)
            return VGroup(*ms)

        k_deck = deck().move_to(LEFT * 0.4 + DOWN * 0.2)
        v_deck = deck().move_to(RIGHT * 3.4 + DOWN * 0.2)
        names = VGroup(
            MathTex("Q", font_size=40).next_to(q_deck, UP, buff=0.25),
            MathTex("K", font_size=40).next_to(k_deck, UP, buff=0.25),
            MathTex("V", font_size=40).next_to(v_deck, UP, buff=0.25),
        )
        self.play(FadeIn(k_deck, shift=LEFT * 0.3), FadeIn(v_deck, shift=LEFT * 0.3), FadeIn(names))
        self.wait(1.2)

        # ── 4. 헤드마다 따로 어텐션 맵 ───────────────────────────
        self.play(FadeOut(VGroup(q_deck, q_tok, k_deck, v_deck, names)),
                  set_code("scores = Q @ K.transpose(-2, -1) / D**0.5   # + mask, softmax"),
                  set_shape(["B", "H", "C", "D"], ["B", "H", "C", "C"], "@ K^T", [3], [3]))
        mcell = 0.42
        maps = VGroup()
        for h in range(H):
            w = attention_pattern(h)
            col = ManimColor(HEAD_COLORS[h])
            m = blocks(C, C,
                       lambda i, j, w=w, col=col: col if j <= i else GREY_D,
                       mcell,
                       lambda i, j, w=w: (0.08 + 0.92 * w[i, j]) if j <= i else 0.5)
            maps.add(m)
        maps.arrange(RIGHT, buff=1.0).move_to(UP * 0.4)
        map_tags = VGroup(*[txt(f"head {h + 1}", 22, HEAD_COLORS[h]).next_to(maps[h], UP, buff=0.2)
                            for h in range(H)])
        views = ["자기 자신을 본다", "바로 앞 토큰을 본다", "첫 토큰을 본다"]
        view_tags = VGroup(*[txt(v, 20, GREY_B).next_to(maps[h], DOWN, buff=0.2) for h, v in enumerate(views)])
        m_tok = token_labels(maps[0], 18)
        self.play(LaggedStart(*[FadeIn(VGroup(maps[h], map_tags[h])) for h in range(H)], lag_ratio=0.3),
                  FadeIn(m_tok))
        sample = txt("(설명용 예시 패턴 · 학습된 모델의 값이 아님)", 18, GREY_B).next_to(view_tags, DOWN, buff=0.15)
        self.play(LaggedStart(*[FadeIn(v) for v in view_tags], lag_ratio=0.3), FadeIn(sample))
        why = txt("같은 입력이라도 헤드마다 W_q, W_k가 달라서 보는 곳이 다르다", 24, WHITE)
        why.next_to(sample, DOWN, buff=0.3)
        self.play(FadeIn(why))
        self.wait(2)
        note4 = txt("4차원 matmul은 B와 H를 배치 축으로 → 헤드 3개를 한 번에 계산", 24, YELLOW)
        note4.move_to(why)
        self.play(FadeOut(why), FadeIn(note4))
        self.wait(2)

        # ── 5. weights @ V → 헤드별 결과 R^i ─────────────────────
        # 헤드마다  맵 (C, C) × V^i (C, D) = R^i (C, D)  를 나란히 보여 준다
        self.play(FadeOut(view_tags), FadeOut(sample), FadeOut(note4), FadeOut(m_tok),
                  set_code("hidden = weights @ V"))
        k = 0.68                                    # 맵·V·R을 한 줄에 넣으려고 축소
        scell = mcell * k
        centers = [LEFT * 4.45, ORIGIN, RIGHT * 4.45]
        Vs, Rs, ops = VGroup(), VGroup(), VGroup()
        for h in range(H):
            m_target = maps[h].copy().scale(k)
            times = MathTex(r"\times", font_size=30)
            v = blocks(C, D, lambda i, j, h=h: HEAD_COLORS[h], scell, lambda i, j: 0.35)
            eq = MathTex("=", font_size=30)
            r = head_matrix(h, scell)
            row = VGroup(m_target, times, v, eq, r).arrange(RIGHT, buff=0.18).move_to(centers[h] + UP * 0.4)
            maps[h].generate_target()
            maps[h].target = m_target
            Vs.add(v)
            Rs.add(r)
            ops.add(VGroup(times, eq))
        self.play(*[MoveToTarget(maps[h]) for h in range(H)],
                  *[map_tags[h].animate.next_to(maps[h].target, UP, buff=0.2) for h in range(H)])
        m_tok = token_labels(maps[0], 16)
        v_names = VGroup(*[MathTex(f"V^{h + 1}", font_size=30, color=HEAD_COLORS[h]).next_to(Vs[h], UP, buff=0.2)
                           for h in range(H)])
        self.play(FadeIn(m_tok), *[FadeIn(VGroup(ops[h][0], Vs[h], v_names[h])) for h in range(H)])
        dims = VGroup(*[
            VGroup(code("(C,C)", 16, GREY_B).next_to(maps[h], DOWN, buff=0.15),
                   code("(C,D)", 16, GREY_B).next_to(Vs[h], DOWN, buff=0.15))
            for h in range(H)])
        self.play(FadeIn(dims))
        r_names = VGroup(*[MathTex(f"R^{h + 1}", font_size=30, color=HEAD_COLORS[h]).next_to(Rs[h], UP, buff=0.2)
                           for h in range(H)])
        r_dims = VGroup(*[code("(C,D)", 16, YELLOW).next_to(Rs[h], DOWN, buff=0.15) for h in range(H)])
        self.play(*[FadeIn(ops[h][1]) for h in range(H)],
                  *[FadeIn(Rs[h], shift=RIGHT * 0.2) for h in range(H)],
                  FadeIn(r_names), FadeIn(r_dims),
                  set_shape(["B", "H", "C", "C"], ["B", "H", "C", "D"], "@ V", [3], [3]))
        note5 = VGroup(
            txt("(C, C) @ (C, D) → (C, D):  가로가 키 5개에서 D 2개로 줄어든다", 24, YELLOW),
            txt("각 행 = 그 토큰이 보는 밸류들의 가중합", 22, GREY_B),
        ).arrange(DOWN, buff=0.15).next_to(dims, DOWN, buff=0.45)
        self.play(FadeIn(note5))
        # sat 행 하나: 맵의 한 행 × V 전체 → R의 한 행
        hl = VGroup(
            SurroundingRectangle(maps[0][FOCUS_ROW], color=YELLOW, buff=0.02),
            SurroundingRectangle(Vs[0], color=YELLOW, buff=0.02),
            SurroundingRectangle(Rs[0][FOCUS_ROW], color=YELLOW, buff=0.02),
        )
        self.play(Create(hl))
        self.wait(2)
        self.play(FadeOut(hl), FadeOut(note5), FadeOut(dims), FadeOut(r_dims), FadeOut(m_tok),
                  FadeOut(VGroup(*maps)), FadeOut(map_tags), FadeOut(Vs), FadeOut(v_names), FadeOut(ops))
        self.play(*[Rs[h].animate.scale(1 / k).move_to(centers[h] * 0.75 + UP * 0.4) for h in range(H)],
                  *[r_names[h].animate.next_to(Rs[h].copy().scale(1 / k).move_to(centers[h] * 0.75 + UP * 0.4),
                                               UP, buff=0.2) for h in range(H)])
        m_tok = token_labels(Rs[0], 18)
        self.play(FadeIn(m_tok))
        self.wait(0.5)

        # ── 6. transpose + view: 헤드 이어 붙이기 ────────────────
        self.play(set_code("hidden.transpose(1, 2).contiguous().view(B, C, H*D)"))
        sep_x = [L[0] - 1.2, L[0], L[0] + 1.2]
        self.play(*[Rs[h].animate.move_to([sep_x[h], L[1], 0]) for h in range(H)],
                  *[r_names[h].animate.move_to([sep_x[h], L[1] + C * mcell / 2 + 0.35, 0]) for h in range(H)],
                  FadeOut(m_tok))
        m_tok = token_labels(Rs[0], 18)
        self.add(m_tok)
        prev = ghost(VGroup(*Rs, r_names, m_tok))
        self.add(prev)
        concat_at = R + RIGHT * 0.3
        moves = [Rs[h].animate.move_to(concat_at + RIGHT * (h - 1) * D * mcell) for h in range(H)]
        self.play(*moves, FadeOut(m_tok), FadeOut(r_names),
                  set_shape(["B", "H", "C", "D"], ["B", "C", "H·D"], ".transpose(1, 2).view", [1, 2, 3], [1, 2]),
                  run_time=1.5)
        concat = VGroup(*Rs)
        c_tok = token_labels(Rs[0], 18)
        c_name = MathTex(r"[\,R^1\;R^2\;R^3\,]", font_size=36).next_to(concat, UP, buff=0.2)
        ba = before_after_arrow(prev, VGroup(concat, c_tok), ".transpose(1, 2).view")
        st = stage_tags(prev, VGroup(concat, c_name))
        self.play(FadeIn(c_tok), FadeIn(c_name), GrowArrow(ba[0]), FadeIn(ba[1]), FadeIn(st))
        note6 = txt("헤드 축을 C 뒤로 되돌리고, (H, D)를 다시 H·D 하나로 합친다", 24, GREY_B).move_to(DOWN * 2.0)
        self.play(FadeIn(note6))
        self.wait(2)
        self.play(FadeOut(prev), FadeOut(ba), FadeOut(st), FadeOut(note6))
        concat_left = LEFT * 3.4 + UP * 0.2
        self.play(VGroup(concat, c_tok, c_name).animate.shift(concat_left - concat.get_center()))

        # ── 7. W_o: (H·D, E) → (B, C, E) ─────────────────────────
        self.play(set_code("output = W_o(hidden)    # nn.Linear(H*D, E)"))
        times = MathTex(r"\times", font_size=40).next_to(concat, RIGHT, buff=0.3)
        Wo = blocks(H * D, E, lambda i, j: HEAD_COLORS[i // D], mcell, lambda i, j: 0.45)
        Wo.next_to(times, RIGHT, buff=0.3).align_to(concat, UP)
        wo_name = MathTex(r"W_o", font_size=36).next_to(Wo, UP, buff=0.2)
        wo_shape = code("(H·D, E)", 20, GREY_B).next_to(Wo, DOWN, buff=0.15)
        eq = MathTex("=", font_size=40).next_to(Wo, RIGHT, buff=0.3)
        out = blocks(C, E, lambda i, j: WHITE, mcell, lambda i, j: 0.25).next_to(eq, RIGHT, buff=0.3)
        out.align_to(concat, UP)
        out_name = txt("출력", 24).next_to(out, UP, buff=0.2)
        self.play(FadeIn(times), FadeIn(Wo), FadeIn(wo_name), FadeIn(wo_shape))
        self.play(FadeIn(eq), FadeIn(out), FadeIn(out_name), set_shape(["B", "C", "H·D"], ["B", "C", "E"], "W_o", [2], [2]))
        same = txt("입력 x와 같은 형상", 22, YELLOW).next_to(out, DOWN, buff=0.2)
        self.play(FadeIn(same))
        self.wait(1)

        # [식 2.3]: 헤드 i의 열 묶음 ↔ W_o의 i번째 행 묶음
        formula = MathTex(r"\sum_{i=1}^{3} R^i W_o^i", "=",
                          r"\begin{bmatrix} R^1 & R^2 & R^3 \end{bmatrix}"
                          r"\begin{bmatrix} W_o^1 \\ W_o^2 \\ W_o^3 \end{bmatrix}",
                          font_size=34).next_to(VGroup(concat, out), DOWN, buff=0.55)
        formula.shift(LEFT * (formula.get_center()[0]))
        self.play(FadeOut(wo_shape), FadeOut(same), FadeOut(shape), Write(formula))
        for h in range(H):
            rows = VGroup(*[Wo[r] for r in range(h * D, (h + 1) * D)])
            self.play(Indicate(Rs[h], color=HEAD_COLORS[h], scale_factor=1.08),
                      Indicate(rows, color=HEAD_COLORS[h], scale_factor=1.08), run_time=0.8)
        note7 = txt("헤드마다 따로 변환해 더하기 = 이어 붙여서 큰 행렬곱 한 번", 24, YELLOW)
        note7.next_to(formula, DOWN, buff=0.3)
        self.play(FadeIn(note7))
        self.wait(2)

        # sat 행 하나: 세 헤드의 관점이 이어 붙여져 한 출력이 된다
        self.play(FadeOut(formula), FadeOut(note7))
        seg_boxes = VGroup(*[SurroundingRectangle(Rs[h][FOCUS_ROW], color=WHITE, buff=0.03, stroke_width=5)
                             for h in range(H)])
        out_box = SurroundingRectangle(out[FOCUS_ROW], color=YELLOW, buff=0.03, stroke_width=4)
        src = [TOKENS[FOCUS_ROW], TOKENS[FOCUS_ROW - 1], TOKENS[0]]     # 맵에서 본 곳: 자기 / 바로 앞 / 첫 토큰
        parts_txt = [f"head {h + 1}: {src[h]}의 정보" for h in range(H)]
        line = "   |   ".join(parts_txt)
        spans, pos = {}, 0
        for h, t in enumerate(parts_txt):
            spans[f"[{pos}:{pos + len(t)}]"] = HEAD_COLORS[h]
            pos += len(t) + len("   |   ")
        persp = VGroup(
            txt(f'"{TOKENS[FOCUS_ROW]}" 행 = 세 헤드의 관점을 이어 붙인 것', 26, WHITE),
            Text(line, font=FONT, font_size=24, color=GREY_B, t2c=spans),
            txt(f'→ W_o가 섞어서 "{TOKENS[FOCUS_ROW]}"의 출력 한 행이 된다', 24, YELLOW),
        ).arrange(DOWN, buff=0.2).next_to(VGroup(concat, out), DOWN, buff=0.55)
        persp.shift(LEFT * persp.get_center()[0])
        self.play(LaggedStart(*[Create(b) for b in seg_boxes], lag_ratio=0.3), FadeIn(persp[0]))
        self.play(FadeIn(persp[1]))
        self.play(Create(out_box), FadeIn(persp[2]))
        self.wait(2.5)
        self.play(FadeOut(seg_boxes), FadeOut(out_box), FadeOut(persp))

        # ── 8. 정리: 형상 흐름 ──────────────────────────────────
        self.play(FadeOut(VGroup(concat, c_tok, c_name, times, Wo, wo_name, eq, out, out_name,
                                 code_line)))
        steps = [
            ("x", "(B, C, E)"),
            ("W_q(x)", "(B, C, H·D)"),
            (".view(B, C, H, D)", "(B, C, H, D)"),
            (".transpose(1, 2)", "(B, H, C, D)"),
            ("Q @ K^T", "(B, H, C, C)"),
            ("weights @ V", "(B, H, C, D)"),
            (".transpose(1, 2).view(B, C, H*D)", "(B, C, H·D)"),
            ("W_o", "(B, C, E)"),
        ]
        rows = VGroup(*[
            VGroup(code(a, 22), code(b, 22, YELLOW)) for a, b in steps
        ])
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.4)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        col_x = max(r[0].get_right()[0] for r in rows) + 0.6
        for r in rows:
            r[1].move_to([col_x + r[1].width / 2, r[0].get_center()[1], 0])
        rows.move_to(ORIGIN)
        head_rows = VGroup(*[rows[i] for i in (3, 4, 5)])
        box = SurroundingRectangle(head_rows, color=ORANGE, buff=0.12)
        box_tag = txt("헤드가 배치 축", 22, ORANGE).next_to(box, RIGHT, buff=0.25)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.15), run_time=2.5)
        self.play(Create(box), FadeIn(box_tag))
        self.wait(3)
