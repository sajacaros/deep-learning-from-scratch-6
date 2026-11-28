# 1장 노트의 그림 "텍스트가 토큰 ID열이 되기까지"를 그린다.
# 입력은 codebot/tiny_codes.txt의 첫 문서 경계 부분(실제 텍스트)이고,
# 토크나이저는 1장에서 학습한 codebot/merge_rules.pkl(어휘 1,000)이다.
# 실행(저장소 루트에서): .venv/bin/python ch01/flow_graph.py → docs/ch01/images/fig01-flow-a.png, fig01-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from codebot.tokenizer import BPETokenizer, pretokenize, merge
import regex as re

# ---------------------------------------------------------------- 데이터
tok = BPETokenizer.load_from(str(ROOT / "codebot/merge_rules.pkl"))
END = tok.end_token
text_all = open(ROOT / "codebot/tiny_codes.txt", encoding="utf-8").read()
k = text_all.find(END)
SAMPLE = text_all[k - 46:k + len(END) + 20]     # "\nprint(f'Your ... {tasks}')\n<|endoftext|>\nshopping_list = {}\n"
PIECES = re.split("(" + re.escape(END) + ")", SAMPLE)
ENC = tok.encode(SAMPLE)
BIN = np.fromfile(ROOT / "codebot/tiny_codes.bin", dtype=np.uint16)
BIN_POS = next(i for i in range(len(BIN)) if BIN[i:i + len(ENC)].tolist() == ENC)
N_BYTES = len(SAMPLE.encode("utf-8"))
assert tok.decode(ENC) == SAMPLE


def show(s):
    """보이지 않는 글자를 보이게: 공백은 ␣, 줄바꿈은 ↵."""
    return s.replace(" ", "␣").replace("\n", "↵")


# 조각(사전 토큰)마다 (조각 문자열, [(토큰 ID, 토큰 문자열), ...])
CHUNKS = []        # 문서별 리스트
for p in PIECES:
    if p == END:
        CHUNKS.append("SPECIAL")
        continue
    doc = []
    for pt in pretokenize(p):
        ids = tok._encode_text(pt)
        doc.append((pt, [(i, tok.id_to_bytes[i].decode("utf-8")) for i in ids]))
    CHUNKS.append(doc)


def trace(word):
    """word에 병합 규칙을 학습 순서대로 적용하면서, 실제로 바뀐 단계만 기록한다."""
    ids = list(word.encode("utf-8"))
    steps = [(None, ids)]
    for pair, nid in tok.merge_rules.items():
        new = merge(ids, pair, nid)
        if new != ids:
            steps.append(((pair, nid), new))
            ids = new
    return steps


# ---------------------------------------------------------------- 스타일
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
MONO = "DejaVu Sans Mono"

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_SPEC, C_BYTE, C_STEP = "#F6D6D6", "#EFEFEF", "#F7F5EE"

# 좌표 단위: 1단위 = 0.5인치 = 36pt
U = 36.0
W = 16.0


def new_fig(h_units):
    fig = plt.figure(figsize=(W / 2, h_units / 2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, h_units); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec=INK, lw=1.0, ls="-", r=0.08, z=3):
    """왼쪽 아래가 (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, ms=12, cs="arc3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def mono_w(s, fs):
    return len(s) * 0.602 * fs / U


def step_label(ax, y, num, title, sub=None):
    ax.text(0.35, y, num, fontsize=13, fontweight="bold", color=RED, va="center")
    ax.text(0.95, y, title, fontsize=11.5, color=INK, va="center")
    if sub:
        ax.text(W - 0.35, y, sub, fontsize=9.5, color=GRAY, va="center", ha="right")


# =====================================================================
# (a) 인코딩 전체 흐름
# =====================================================================
HA = 27.0
fig, ax = new_fig(HA)
ax.text(0.35, HA - 0.6, f"(a)  텍스트 {N_BYTES}바이트가 토큰 ID {len(ENC)}개가 되기까지",
        fontsize=14, fontweight="bold", color=INK, va="center")

# 원문
y = HA - 1.55
ax.text(0.35, y, "원문", fontsize=11.5, color=INK, va="center")
ax.text(W - 0.35, y, "tiny_codes.txt의 첫 문서 경계 부분 (␣ 공백, ↵ 줄바꿈)", fontsize=9.5, color=GRAY,
        va="center", ha="right")
FS_RAW = 10.5
rbox(ax, 0.35, y - 1.05, W - 0.7, 0.75, "white", ec=GRAY)
xx = 0.55
for p in PIECES:
    s = show(p)
    col = RED if p == END else INK
    ax.text(xx, y - 0.67, s, fontsize=FS_RAW, family=MONO, color=col, va="center",
            fontweight="bold" if p == END else "normal", zorder=5)
    xx += mono_w(s, FS_RAW)

# ① 특수 토큰으로 나누기
y1 = y - 1.75
arrow(ax, W / 2, y - 1.1, W / 2, y1 + 0.3)
step_label(ax, y1, "①", "특수 토큰(문서 경계)으로 나눈다", "re.split — 패턴을 괄호로 감싸 구분자도 남긴다")
y1b = y1 - 0.55
piece_x = []
widths = [7.7, 2.9, 4.3]
gap = (W - 0.7 - sum(widths)) / 2
xx = 0.35
for p, w in zip(PIECES, widths):
    if p == END:
        rbox(ax, xx, y1b - 1.2, w, 1.2, C_SPEC, ec=RED, lw=1.3)
        ax.text(xx + w / 2, y1b - 0.38, "<|endoftext|>", fontsize=10, family=MONO, color=RED, ha="center",
                va="center", fontweight="bold", zorder=5)
        ax.text(xx + w / 2, y1b - 0.85, f"BPE 없이 ID {tok.end_token_id}", fontsize=10, color=INK,
                ha="center", va="center", zorder=5)
    else:
        rbox(ax, xx, y1b - 1.2, w, 1.2, "white", ec=INK)
        ax.text(xx + w / 2, y1b - 0.38, "문서 1" if xx < 1 else "문서 2", fontsize=10, color=GRAY,
                ha="center", va="center", zorder=5)
        s = show(p)
        fs = 9 if len(s) > 40 else 10
        ax.text(xx + w / 2, y1b - 0.85, s, fontsize=fs, family=MONO, ha="center", va="center", zorder=5)
    piece_x.append((xx, w))
    xx += w + gap

# ② 사전 토큰화 + ③④ 바이트 → 병합 : 조각 칩과 그 아래 토큰을 같은 열에 놓는다
y2 = y1b - 1.95
arrow(ax, W / 2, y1b - 1.25, W / 2, y2 + 0.3)
step_label(ax, y2, "②", "문서마다 사전 토큰화 (GPT-2 정규식)", "조각 경계를 넘는 병합은 일어나지 않는다")
ax.text(1.35, y2 - 0.55, "조각마다 UTF-8 바이트로 바꾸고, 병합 규칙 743개를 학습 순서대로 적용한다",
        fontsize=11.5, color=INK, va="center")
ax.text(0.35, y2 - 0.55, "③④", fontsize=13, fontweight="bold", color=RED, va="center")

FS_CH, FS_ID = 10, 8.5
BYTE_W = mono_w("x", FS_CH)
MIN_TOK = 0.62           # ID 세 자리가 들어가는 최소 폭
CH_H, TOK_H = 0.55, 0.72
ROW_H = CH_H + 0.25 + TOK_H + 0.55


def tok_w(t):
    return max(mono_w(show(t), FS_CH) + 0.16, MIN_TOK)


def draw_doc(doc, x0, ytop, xmax, ci0):
    """문서 하나의 조각들을 줄바꿈하며 그린다. 반환: 다음 y, 조각 색 인덱스"""
    xx, yy, ci = x0, ytop, ci0
    for pt, toks in doc:
        ws = [tok_w(t) for _, t in toks]
        cw = max(sum(ws), mono_w(show(pt), FS_CH) + 0.16)
        scale = cw / sum(ws)
        ws = [w * scale for w in ws]
        if xx + cw > xmax:
            xx, yy = x0, yy - ROW_H
        c = ci % len(POS)
        # 조각 칩
        rbox(ax, xx, yy - CH_H, cw, CH_H, "white", ec=POS[c], lw=1.2)
        ax.text(xx + cw / 2, yy - CH_H / 2, show(pt), fontsize=FS_CH, family=MONO, ha="center", va="center", zorder=5)
        # 토큰
        ty = yy - CH_H - 0.25
        ax.plot([xx + cw / 2, xx + cw / 2], [yy - CH_H, ty], color=POS[c], lw=0.9, zorder=2)
        tx = xx
        for (tid, t), w in zip(toks, ws):
            rbox(ax, tx, ty - TOK_H, w, TOK_H, POSL[c], ec=POS[c], lw=1.0, r=0.05)
            ax.text(tx + w / 2, ty - 0.24, show(t), fontsize=FS_CH, family=MONO, ha="center", va="center", zorder=5)
            ax.text(tx + w / 2, ty - 0.54, str(tid), fontsize=FS_ID, color="#444444", ha="center", va="center", zorder=5)
            tx += w
        xx += cw + 0.14
        ci += 1
    return yy - ROW_H, ci


y3 = y2 - 1.15
ax.text(0.35, y3 - 0.27, "문서 1", fontsize=10, color=GRAY, va="center")
ynext, ci = draw_doc(CHUNKS[0], 1.45, y3, W - 0.3, 0)
ax.text(0.35, ynext - 0.27, "특수", fontsize=10, color=GRAY, va="center")
rbox(ax, 1.45, ynext - CH_H, 3.0, CH_H, C_SPEC, ec=RED, lw=1.2)
ax.text(1.45 + 1.5, ynext - CH_H / 2, "<|endoftext|>", fontsize=FS_CH, family=MONO, color=RED, ha="center", va="center", zorder=5)
ax.plot([2.95, 2.95], [ynext - CH_H, ynext - CH_H - 0.25], color=RED, lw=0.9)
rbox(ax, 1.45, ynext - CH_H - 0.25 - TOK_H, 3.0, TOK_H, C_SPEC, ec=RED, lw=1.0, r=0.05)
ax.text(2.95, ynext - CH_H - 0.25 - 0.36, f"{tok.end_token_id} (어휘의 마지막 번호)", fontsize=FS_ID, color=RED,
        ha="center", va="center", zorder=5)
ax.text(4.75, ynext - 0.27, "문서 2", fontsize=10, color=GRAY, va="center")
ynext2, ci = draw_doc(CHUNKS[2], 5.85, ynext, W - 0.3, ci)

# 범례 같은 설명
yl = ynext2 + 0.05
ax.text(0.35, yl, "테두리 칸 = 사전 토큰(조각),  아래 색 칸 = 병합 후 남은 토큰과 그 ID.  "
        "한 조각이 여러 토큰으로 남기도 한다(␣today → ␣to · d · ay).",
        fontsize=9.5, color=GRAY, va="center")

# ⑤ 이어 붙이기 → bin
y5 = yl - 0.75
step_label(ax, y5, "⑤", f"모든 토큰 ID를 순서대로 이어 붙인다 → {len(ENC)}개", "09_bpe_encode.py")
rbox(ax, 0.35, y5 - 1.75, W - 0.7, 1.3, "white", ec=GRAY)
s_ids = ", ".join(str(i) for i in ENC)
half = s_ids.find(", ", len(s_ids) // 2)
for j, line in enumerate([s_ids[:half + 1], s_ids[half + 2:]]):
    ax.text(0.55, y5 - 0.78 - j * 0.48, line, fontsize=9.5, family=MONO, color=INK, va="center", zorder=5)
y6 = y5 - 2.6
arrow(ax, W / 2, y5 - 1.8, W / 2, y6 + 0.32)
rbox(ax, W / 2 - 4.2, y6 - 0.45, 8.4, 0.85, "#F3F0C4", ec=INK)
ax.text(W / 2, y6 - 0.03, "np.uint16으로 tiny_codes.bin에 저장 → 2장 이후 학습 데이터", fontsize=11, ha="center",
        va="center", zorder=5)
ax.text(W / 2, y6 - 1.0, f"실제 tiny_codes.bin의 {BIN_POS}~{BIN_POS + len(ENC) - 1}번째 값과 똑같다.   "
        f"압축률 = {N_BYTES} 바이트 / {len(ENC)} 토큰 = {N_BYTES / len(ENC):.2f}배",
        fontsize=10, color=GRAY, ha="center", va="center")

# 디코딩 되돌림
ax.text(0.35, y6 - 1.75, "디코딩은 거꾸로: ID마다 id_to_bytes[ID]를 꺼내 이어 붙여 UTF-8로 풀면 원문이 그대로 나온다.", fontsize=10, color=RED, va="center")

HA_USED = y6 - 2.2
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch01/images"
out_a = OUT / "fig01-flow-a.png"
from matplotlib.transforms import Bbox
fig.savefig(out_a, dpi=200, facecolor="white", bbox_inches=Bbox([[0, max(HA_USED, 0) / 2], [W / 2, HA / 2]]))
plt.close(fig)

# =====================================================================
# (b) 조각 하나에 병합 규칙이 적용되는 순서
# =====================================================================
HB = 22.0
fig, ax = new_fig(HB)
ax.text(0.35, HB - 0.6, "(b)  조각 하나 안에서 병합 규칙이 적용되는 순서", fontsize=14, fontweight="bold",
        color=INK, va="center")
ax.text(0.35, HB - 1.25, "규칙 번호 = 학습된 순서(= 새 ID). 743개를 256번부터 차례로 훑으며, 맞는 쌍이 있는 규칙에서만 ID열이 바뀐다.",
        fontsize=10, color=GRAY, va="center")

BW = 0.78       # 바이트 한 칸 폭
RH = 0.78       # 단계 한 줄 높이


def ladder(word, x0, ytop, title):
    steps = trace(word)
    nb = len(word.encode("utf-8"))
    ax.text(x0, ytop, title, fontsize=12, fontweight="bold", color=INK, va="center")
    yy = ytop - 0.5
    for si, (rule, ids) in enumerate(steps):
        # 각 토큰이 차지하는 바이트 수
        xx = x0 + 2.55
        for tid in ids:
            n = len(tok.id_to_bytes[tid])
            merged = tid >= 256
            just = rule is not None and tid == rule[1]
            fc = POSL[0] if merged else C_BYTE
            ec = RED if just else (POS[0] if merged else "#999999")
            rbox(ax, xx + 0.03, yy - RH + 0.06, n * BW - 0.06, RH - 0.12, fc, ec=ec, lw=1.6 if just else 0.9, r=0.06)
            ax.text(xx + n * BW / 2, yy - 0.27, show(tok.id_to_bytes[tid].decode()), fontsize=10, family=MONO,
                    ha="center", va="center", zorder=5)
            ax.text(xx + n * BW / 2, yy - 0.56, str(tid), fontsize=8.5, color=RED if just else "#444444",
                    ha="center", va="center", zorder=5)
            xx += n * BW
        if rule is None:
            lab = "UTF-8 바이트"
        else:
            (a, b), nid = rule
            lab = f"규칙 {nid}"
            sa, sb = show(tok.id_to_bytes[a].decode()), show(tok.id_to_bytes[b].decode())
            ax.text(x0 + 2.55 + nb * BW + 0.2, yy - RH / 2, f"{sa} + {sb}", fontsize=10, family=MONO,
                    color=RED, va="center")
        ax.text(x0 + 2.35, yy - RH / 2, lab, fontsize=10, color=RED if rule else GRAY, va="center", ha="right")
        yy -= RH
    ax.text(x0 + 2.35, yy - 0.3, "결과", fontsize=10.5, fontweight="bold", color=INK, va="center", ha="right")
    res = str(steps[-1][1])
    ax.text(x0 + 2.55, yy - 0.3, res, fontsize=10, family=MONO, color=INK, va="center")
    ax.text(x0 + 2.55 + mono_w(res, 10) + 0.3, yy - 0.3, f"({nb}바이트 → {len(steps[-1][1])}토큰)",
            fontsize=10, color=INK, va="center")
    return yy - 0.3


yb = HB - 2.1
yend1 = ladder("print", 0.35, yb, "print — 규칙 4개를 거쳐 토큰 하나가 된다")
ax.text(9.0, yb - 1.3, "앞 규칙이 만든 토큰(in, rin, rint)을\n뒤 규칙이 재료로 쓴다.\n"
        "그래서 학습 순서대로 적용해야 한다.", fontsize=10, color=INK, va="top", linespacing=1.5)

yb2 = yend1 - 1.0
yend2 = ladder("shopping", 0.35, yb2, "shopping — 순서가 결과를 정한다")
pp = tok.merge_rules[(112, 112)]
ax.text(0.35, yend2 - 0.85,
        f"pp 규칙({pp}번)도 있지만, op 규칙(499번)이 먼저 와서 첫 p를 가져갔다. 남은 p는 짝이 없어 혼자 남는다.\n"
        f"ping 규칙은 없으므로 결과는 sh · op · p · ing 4토큰. 규칙 순서를 바꾸면 다른 ID열이 된다.",
        fontsize=10, color=INK, va="center", linespacing=1.5)

# 공백 포함 여부
yb3 = yend2 - 2.0
ax.text(0.35, yb3, "같은 글자라도 앞 공백이 붙으면 다른 토큰", fontsize=12, fontweight="bold", color=INK, va="center")
for j, (w, note) in enumerate([(" list", "(a) 문서 1의 ␣list"), ("list", "(a) 문서 2의 shopping_list 속 list")]):
    ids = tok._encode_text(w)
    yy = yb3 - 0.75 - j * 0.9
    xx = 0.6
    for b in w.encode():
        rbox(ax, xx, yy - 0.33, 0.6, 0.66, C_BYTE, ec="#999999", lw=0.9, r=0.05)
        ax.text(xx + 0.3, yy, show(chr(b)), fontsize=10, family=MONO, ha="center", va="center", zorder=5)
        xx += 0.6
    arrow(ax, xx + 0.15, yy, xx + 0.95, yy, lw=1.0, ms=10)
    rbox(ax, xx + 1.1, yy - 0.33, 2.3, 0.66, POSL[0], ec=POS[0], lw=1.0, r=0.05)
    ax.text(xx + 2.25, yy, f"{show(w)} = {ids[0]}", fontsize=10, family=MONO, ha="center", va="center", zorder=5)
    ax.text(xx + 3.7, yy, note, fontsize=10, color=GRAY, va="center")
ax.text(0.35, yb3 - 2.65, "디코딩: id_to_bytes[350] = id_to_bytes[112] + id_to_bytes[306] = b'p' + b'rint' = b'print'\n"
        "병합 규칙만 있으면 어휘표(id_to_bytes)를 다시 만들 수 있다.", fontsize=10, color=RED, va="center",
        linespacing=1.5)

HB_USED = yb3 - 3.3
out_b = OUT / "fig01-flow-b.png"
fig.savefig(out_b, dpi=200, facecolor="white", bbox_inches=Bbox([[0, max(HB_USED, 0) / 2], [W / 2, HB / 2]]))
plt.close(fig)

# 256색 팔레트로 줄인다(그라데이션이 없으므로 충분)
for out in (out_a, out_b):
    im = Image.open(out).convert("RGB")
    im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out, im.size)
print("sample:", repr(SAMPLE))
print("ids:", ENC, "bin pos:", BIN_POS)
