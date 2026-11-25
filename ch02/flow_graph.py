# 2장 노트의 그림 "배치 하나가 GPT를 지나 학습되는 흐름"을 그린다.
# 입력은 CodeBot 학습 데이터(codebot/tiny_codes.bin)에서 3장 TokenDataset과 같은 방식으로 잘라 낸 실제 토큰이다.
# 실제 학습은 B=32, C=256(ch03/01_pretrain.py)이지만, 그림에서는 B=2, C=5로 줄였다.
# 실행(저장소 루트에서): python ch02/flow_graph.py → docs/ch02/images/fig02-flow-a.png, fig02-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from codebot.tokenizer import BPETokenizer

# ---------------------------------------------------------------- 데이터
B, C = 2, 5
STARTS = [103, 53]           # tiny_codes.bin 안의 시작 위치(문장 0, 문장 1)
tok = BPETokenizer.load_from(str(ROOT / "codebot/merge_rules.pkl"))
ids = np.fromfile(ROOT / "codebot/tiny_codes.bin", dtype=np.uint16)
X_IDS = [ids[s:s + C].tolist() for s in STARTS]            # x: (B, C)
Y_IDS = [ids[s + 1:s + C + 1].tolist() for s in STARTS]    # y: 한 칸 민 것


def show(i):
    """토큰 하나를 보이는 글자로. 앞 공백은 ␣로."""
    t = tok.decode([int(i)])
    return t.replace(" ", "␣").replace("\n", "↵")


X_TOK = [[show(i) for i in row] for row in X_IDS]
Y_TOK = [[show(i) for i in row] for row in Y_IDS]

# ---------------------------------------------------------------- 스타일
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]       # 위치(열) 색
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_EMB, C_ATT, C_FFN, C_LN, C_OUT, C_BLK = "#F6D6D6", "#F9E3C2", "#CFE3F3", "#F3F0C4", "#E2DAF0", "#F4F4F4"

W_UNITS, H_UNITS = 34, 33
fig = plt.figure(figsize=(17, 16.5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_UNITS); ax.set_ylim(0, H_UNITS); ax.axis("off")


def box(x, y, w, h, text, fc, fs=12, bold=False, ec=INK, lw=1.1, color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=4)


def arrow(x1, y1, x2, y2, color=INK, lw=1.2, cs="arc3", ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def shape(x, y, s, ha="left", fs=10.5):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=GRAY, zorder=4)


def plus(x, y, r=0.28):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.1, zorder=4))
    ax.plot([x - r * .6, x + r * .6], [y, y], color=INK, lw=1.1, zorder=5)
    ax.plot([x, x], [y - r * .6, y + r * .6], color=INK, lw=1.1, zorder=5)


def token_matrix(x, y, ncols, cw=0.3, rh=0.34, rows=C, label=None, dots=True):
    """행 = 위치(토큰). 왼쪽 위가 (x, y)."""
    for r in range(rows):
        for c in range(ncols):
            if dots and c == ncols - 2:
                ax.text(x + c * cw + cw / 2, y - r * rh - rh / 2, "···", ha="center", va="center",
                        fontsize=7, color=POS[r], zorder=4)
                continue
            ax.add_patch(Rectangle((x + c * cw, y - (r + 1) * rh), cw, rh, fc=POSL[r], ec=POS[r], lw=.7, zorder=3))
    w, h = ncols * cw, rows * rh
    if label:
        ax.text(x + w / 2, y - h - 0.3, label, ha="center", va="center", fontsize=11, color=INK)
    return w, h


# =====================================================================
# (a) 전체 흐름
# =====================================================================
ax.text(0.5, 32.55, f"(a)  배치 하나(B={B}, C={C})가 학습되는 흐름", fontsize=15, fontweight="bold", color=INK, va="center")
X = 7.0

# 학습 전 준비: 토크나이징은 학습 루프 밖에서 한 번만 한다(1장)
ax.add_patch(FancyBboxPatch((0.4, 30.45), 14.4, 1.3, boxstyle="round,pad=0,rounding_size=0.2",
                            fc="none", ec=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(14.8, 32.0, "학습 전에 한 번만 (1장)", fontsize=10, color=GRAY, va="center", ha="right")
for cx, t1, t2 in [(2.55, "tiny_codes.txt", "파이썬 코드 원문"),
                   (7.45, "BPE 토크나이저", "ch01/09_bpe_encode.py"),
                   (12.35, "tiny_codes.bin", f"토큰 ID {len(ids):,}개")]:
    box(cx, 31.05, 3.9, 0.85, "", "white" if cx != 7.45 else C_EMB)
    ax.text(cx, 31.2, t1, fontsize=11.5, ha="center", va="center", zorder=5)
    ax.text(cx, 30.85, t2, fontsize=9, color=GRAY, ha="center", va="center", zorder=5)
arrow(4.5, 31.05, 5.5, 31.05)
arrow(9.4, 31.05, 10.4, 31.05)
arrow(12.35, 30.62, 12.35, 29.45)
ax.text(12.1, 29.75, "TokenDataset(3장) — 무작위 위치에서 C=5개씩 잘라 B=2개를 쌓는다. y는 한 칸 민 것",
        fontsize=10.5, color=INK, va="center", ha="right")

# 입력 배치: 문장 2개 × 토큰 5개
ax.text(0.5, 28.9, "입력 x  (2, 5)  — 모델에 들어가는 것은 토큰 ID", fontsize=11.5, color=INK, va="center")
shape(14.9, 28.9, "열 = 위치 0~4", ha="right")
for b in range(B):
    yy = 28.05 - b * 0.98
    ax.text(0.5, yy, f"문장 {b}", fontsize=11, color=INK, va="center")
    for c in range(C):
        cx = 3.0 + c * 2.25
        box(cx, yy, 2.05, 0.82, "", POSL[c], ec=POS[c], lw=1.3)
        ax.text(cx, yy + 0.15, X_TOK[b][c], fontsize=11.5, ha="center", va="center", zorder=5)
        ax.text(cx, yy - 0.22, str(X_IDS[b][c]), fontsize=8.5, color=GRAY, ha="center", va="center", zorder=5)
arrow(X, 26.62, X, 25.45)

# 임베딩: 표 크기를 함께 적는다
box(X - 2.0, 24.95, 4.3, 0.95, "", C_EMB)
ax.text(X - 2.0, 25.15, "토큰 임베딩", fontsize=12, ha="center", va="center", zorder=5)
ax.text(X - 2.0, 24.72, "표 (1000, 384) — 행 수 = 어휘 크기", fontsize=9.5, color=GRAY, ha="center", va="center", zorder=5)
box(X + 2.6, 24.95, 4.3, 0.95, "", C_EMB)
ax.text(X + 2.6, 25.15, "위치 임베딩 (0~4행)", fontsize=12, ha="center", va="center", zorder=5)
ax.text(X + 2.6, 24.72, "표 (256, 384) — 행 수 = 최대 길이", fontsize=9.5, color=GRAY, ha="center", va="center", zorder=5)
plus(X, 23.75)
arrow(X - 2.0, 24.47, X - 0.25, 23.9, cs="arc3,rad=0.15")
arrow(X + 2.6, 24.47, X + 0.25, 23.9, cs="arc3,rad=-0.15")
shape(X + 0.45, 23.3, "(2, 5, 384)   위치 벡터는 두 문장에 똑같이 더함")

# ---------------- 블록 ----------------
BT, BB = 22.9, 12.1
ax.add_patch(FancyBboxPatch((X - 3.4, BB), 6.8, BT - BB, boxstyle="round,pad=0,rounding_size=0.3",
                            fc=C_BLK, ec=INK, lw=1.3, zorder=1))
ax.text(X + 3.7, 15.0, "× 6", fontsize=17, fontweight="bold", va="center", color=INK)
ax.text(X - 3.2, BB + 0.3, "트랜스포머 블록", fontsize=11, color=GRAY, va="center", ha="left")
arrow(X, 23.47, X, 21.95)
box(X, 21.6, 3.4, 0.7, "LayerNorm", C_LN, fs=11.5)
arrow(X, 21.25, X, 20.7)
box(X, 19.9, 4.6, 1.6, "", C_ATT)
ax.text(X, 20.3, "마스크드 멀티헤드 어텐션", fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=5)
ax.text(X, 19.62, "헤드 6개 × 64차원\n문장끼리는 섞지 않음", fontsize=9.5, color=GRAY, ha="center", va="center",
        zorder=5, linespacing=1.3)
arrow(X, 19.1, X, 18.33)

# 어텐션 안의 형상 (codebot/model.py MultiHeadAttention 순서, B=2, C=5, H=6, D=64)
NX0, NX1, NY0, NY1 = X + 3.65, 14.95, 16.45, 22.45
ax.add_patch(FancyBboxPatch((NX0, NY0), NX1 - NX0, NY1 - NY0, boxstyle="round,pad=0,rounding_size=0.15",
                            fc="white", ec=C_ATT.replace("F9E3C2", "E0B97A"), lw=1.1, zorder=3))
ax.plot([X + 2.3, NX0], [19.9, 19.9], color="#E0B97A", lw=1.0, ls=(0, (3, 2)), zorder=2)
ax.text(NX0 + 0.2, NY1 - 0.35, "어텐션 안의 형상", fontsize=10.5, fontweight="bold", color=INK, va="center", zorder=5)
rows_qkv = [(r"$Q, K, V = xW$", "(2, 5, 384)"),
            ("헤드로 나눔", "(2, 6, 5, 64)"),
            (r"점수 $QK^\top/\sqrt{64}$", "(2, 6, 5, 5)"),
            (r"가중치 $\times\, V$", "(2, 6, 5, 64)"),
            (r"이어 붙임 $\to W_o$", "(2, 5, 384)")]
for k, (lab, shp) in enumerate(rows_qkv):
    yy = NY1 - 0.95 - k * 0.68
    ax.text(NX0 + 0.2, yy, lab, fontsize=9.5, color=INK, va="center", zorder=5)
    ax.text(NX1 - 0.15, yy, shp, fontsize=9.5, color=GRAY, va="center", ha="right", zorder=5)
ax.text(NX0 + 0.35, NY1 - 0.95 - 4 * 0.68 - 0.62, r"$W_o$ (384, 384): 헤드 6개를 섞음" "\n"
        "H·D = E라 형상은 그대로", fontsize=9, color=RED, va="center", zorder=5, linespacing=1.4)
ax.text(NX0 + 0.2, NY0 + 0.5, "가운데 세 줄은 (B, H, C, ·) 순서\n(b)는 이 중 문장 0·헤드 1개", fontsize=9, color=GRAY,
        va="center", zorder=5, linespacing=1.4)
plus(X, 18.05)
ax.plot([X, X - 2.9, X - 2.9], [22.45, 22.45, 18.05], color=INK, lw=1.1, zorder=2)
arrow(X - 2.9, 18.05, X - 0.28, 18.05)
shape(X + 0.45, 18.05, "(2, 5, 384)")
arrow(X, 17.77, X, 17.15)
box(X, 16.8, 3.4, 0.7, "LayerNorm", C_LN, fs=11.5)
arrow(X, 16.45, X, 15.9)
box(X, 14.8, 5.2, 2.2, "", C_FFN)
ax.text(X, 15.65, "FFN", fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=5)
ax.text(X, 14.85, r"Linear 384$\to$1536  $\to$  GELU" "\n" r"$\to$  Linear 1536$\to$384",
        fontsize=10.5, ha="center", va="center", zorder=5, linespacing=1.5)
ax.text(X, 14.0, "토큰(행)마다 따로 계산", fontsize=10.5, color=GRAY, ha="center", va="center", zorder=5)
arrow(X, 13.7, X, 13.08)
plus(X, 12.8)
ax.plot([X, X - 2.9, X - 2.9], [17.45, 17.45, 12.8], color=INK, lw=1.1, zorder=2)
arrow(X - 2.9, 12.8, X - 0.28, 12.8)
shape(X + 0.45, 12.8, "(2, 5, 384)")

# ---------------- 출력 ----------------
arrow(X, 12.52, X, 11.75)
box(X, 11.4, 3.4, 0.7, "LayerNorm", C_LN, fs=11.5)
arrow(X, 11.05, X, 10.45)
box(X, 10.05, 5.0, 0.8, r"Linear 384$\to$1000  (unembed)", C_OUT, fs=11.5)
arrow(X, 9.65, X, 9.1)
shape(X + 0.3, 9.4, "logits (2, 5, 1000)")
ax.plot([X - 4.3, X - 4.3 - 0.4, X - 4.3 - 0.4], [24.95, 24.95, 10.05], color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=0)
ax.plot([X - 4.7, X - 2.5], [10.05, 10.05], color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=0)
ax.text(X - 4.9, 17.5, "가중치 공유", rotation=90, fontsize=10.5, color=GRAY, ha="right", va="center")

# 위치마다 다음 토큰 예측 (문장 0)
ax.text(0.9, 8.75, "문장 0 — 위치마다 다음 토큰 예측", fontsize=11, color=INK, va="center")
ax.text(13.2, 8.75, "정답 y", fontsize=10.5, color=GRAY, va="center", ha="center")
for i in range(C):
    yy = 8.1 - i * 0.68
    ctx = "".join(X_TOK[0][: i + 1]).replace("␣", " ").strip()
    ax.add_patch(Rectangle((0.9, yy - 0.27), 0.2, 0.54, fc=POS[i], ec="none", zorder=3))
    ax.text(1.3, yy, f"위치 {i}   P( · | {ctx} )", fontsize=10.5, va="center", color=INK)
    box(13.2, yy, 1.7, 0.54, Y_TOK[0][i], POSL[i], fs=11, ec=POS[i], lw=1.3)
ax.text(0.9, 4.55, f"문장 1도 같은 방식으로 5개 → 배치 전체 B×C = {B * C}개 예측을 한 번에", fontsize=10.5, color=GRAY, va="center")
arrow(X, 4.2, X, 3.7)
box(X, 3.15, 8.4, 1.0, r"loss $= -\frac{1}{B\,C}\sum_{b}\sum_{i}\log P(y_{b,i})$    (cross-entropy, 10개 평균)", "white", fs=11.5)
arrow(X, 2.65, X, 2.05, color=RED)
ax.text(X, 1.7, r"역전파 → 모든 가중치(임베딩, $W_q, W_k, W_v, W_o$, FFN, LayerNorm) 갱신",
        fontsize=11, ha="center", va="center", color=RED)

# =====================================================================
# (b) 블록 안의 어텐션 (문장 0, 헤드 하나)
# =====================================================================
OX = 16.3
ax.plot([15.4, 15.4], [0.6, 32.2], color="#DDDDDD", lw=1.0)
ax.text(OX, 32.55, "(b)  마스크드 어텐션 내부 — 문장 0, 헤드 하나", fontsize=15, fontweight="bold", color=INK, va="center")

xx, xy = OX + 1.0, 27.6
w, h = token_matrix(xx, xy, 12, cw=0.34)
for i, t in enumerate(X_TOK[0]):
    ax.text(xx - 0.15, xy - i * 0.34 - 0.17, t, ha="right", va="center", fontsize=10, color=POS[i])
ax.text(xx + w / 2, xy + 0.35, r"$X$  (5, 384)  — LayerNorm 출력", ha="center", fontsize=11.5, color=INK)
ax.text(xx + w + 0.3, xy - h / 2, "행 = 위치(토큰) 하나\n칸 = 숫자 하나", fontsize=10, color=GRAY, va="center")

qy = 24.0
for cx, name, wname in [(OX + 0.6, "Q", r"$W_q$"), (OX + 4.6, "K", r"$W_k$"), (OX + 8.6, "V", r"$W_v$")]:
    arrow(xx + w / 2, xy - h - 0.1, cx + 0.9, qy + 0.1, lw=1.0)
    ax.text(cx + 1.15, qy + 0.45, r"$\times$ " + wname, fontsize=11.5, ha="left", color=INK,
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    token_matrix(cx, qy, 6, cw=0.3, label=f"${name}$  (5, 64)")

# 점수와 가중치 (설명용 예시 값)
SC = np.array([[1.8, 0, 0, 0, 0],
               [0.6, 1.4, 0, 0, 0],
               [-0.3, 0.9, 1.7, 0, 0],
               [0.2, -0.5, 1.1, 1.3, 0],
               [-0.8, 0.4, 0.7, 1.6, 0.9]])
WT = np.zeros_like(SC)
for r in range(C):
    e = np.exp(SC[r, : r + 1]); WT[r, : r + 1] = e / e.sum()
WT_TXT = []
for r in range(C):   # 반올림한 값의 합이 정확히 1.00이 되도록 가장 큰 칸에서 보정
    v = np.round(WT[r, : r + 1], 2); v[np.argmax(v)] += round(1 - v.sum(), 2)
    WT_TXT.append([f"{x:.2f}" for x in v] + ["0"] * (C - r - 1))
SC_TXT = [[f"{SC[r, c]:.1f}" if c <= r else r"$-\infty$" for c in range(C)] for r in range(C)]
cs = 0.5


def grid(gx, gy, vals):
    for r in range(C):
        for c in range(C):
            m = c > r
            ax.add_patch(Rectangle((gx + c * cs, gy - (r + 1) * cs), cs, cs, fc="#E3E3E3" if m else POSL[r],
                                   ec="#BBBBBB" if m else POS[r], lw=.8, zorder=3))
            ax.text(gx + c * cs + cs / 2, gy - r * cs - cs / 2, vals[r][c], ha="center", va="center",
                    fontsize=8, color=GRAY if m else INK, zorder=4)
    for c in range(C):
        ax.text(gx + c * cs + cs / 2, gy + 0.08, X_TOK[0][c], ha="left", va="bottom", fontsize=8,
                color=POS[c], rotation=50, rotation_mode="anchor")
    for r in range(C):
        ax.text(gx - 0.1, gy - r * cs - cs / 2, X_TOK[0][r], ha="right", va="center", fontsize=8.5, color=POS[r])


sx, sy = OX + 1.2, 19.3
ax.text(sx + 2.5 * cs, sy + 1.45, "점수 " r"$QK^\top/\sqrt{64}$" "\n+ 인과 마스크  (5, 5)",
        fontsize=11, color=INK, ha="center", va="center", linespacing=1.4)
grid(sx, sy, SC_TXT)
ax.text(sx + 2.5 * cs, sy - C * cs - 0.45, "행 = 쿼리(보는 쪽)\n열 = 키(보이는 쪽)", fontsize=9.5, ha="center", color=GRAY, va="center")
arrow(OX + 1.5, 21.75, sx + 2.0 * cs, sy + 1.95, lw=1.0)
arrow(OX + 5.5, 21.75, sx + 3.0 * cs, sy + 1.95, lw=1.0)

wx = sx + 5.6
my = sy - 2.5 * cs
arrow(sx + C * cs + 0.2, my, wx - 0.85, my)
ax.text((sx + C * cs + wx - 0.65) / 2, my + 0.15, "softmax\n(행마다)", ha="center", va="bottom", fontsize=10.5, color=INK)
grid(wx, sy, WT_TXT)
ax.text(wx + 2.5 * cs, sy + 1.45, "어텐션 가중치  (5, 5)\n각 행의 합 = 1, 미래 칸 = 0",
        fontsize=11, color=INK, ha="center", va="center", linespacing=1.4)

hy = 14.4
hx0 = wx + 1.2
arrow(wx + 2.5 * cs, sy - C * cs - 0.1, hx0 + 0.55, hy + 0.15, lw=1.0)
ax.plot([OX + 10.4, OX + 10.4], [22.15, hy + 1.2], color=INK, lw=1.0, zorder=2)
arrow(OX + 10.4, hy + 1.2, hx0 + 1.5, hy + 0.15, lw=1.0)
ax.text(OX + 10.55, 18.0, r"$V$", fontsize=11, color=INK, ha="left", va="center")
ax.text(hx0 + 1.0, hy + 1.0, r"가중치 $\times\ V$", fontsize=11.5, color=INK, ha="center",
        bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
token_matrix(hx0, hy, 6, cw=0.3, label="헤드 출력  (5, 64)")
ax.text(hx0 - 0.4, hy - 0.85, "행 i = 위치 0~i 밸류의 가중합\n(미래 위치는 섞이지 않음)",
        fontsize=10, color=GRAY, va="center", ha="right")

cy = 10.9
arrow(hx0 + 0.9, hy - 2.1, hx0 + 0.9, cy + 0.15, lw=1.0)
ax.text(hx0 + 1.15, (hy - 2.1 + cy + 0.15) / 2, "헤드 6개가 각자의\n" r"$W_q, W_k, W_v$로 같은 계산을 동시에",
        fontsize=10, color=GRAY, va="center")
hx = OX + 2.2
for k in range(6):
    token_matrix(hx + k * 1.45, cy - 0.25, 4, cw=0.3, dots=False)
    ax.text(hx + k * 1.45 + 0.6, cy - 0.05, f"헤드{k + 1}", fontsize=9, ha="center", color=GRAY)
ax.text(hx + 4.35, cy - 2.25, "이어 붙임  (5, 6×64 = 384)", fontsize=11, ha="center", color=INK)
arrow(hx + 4.35, cy - 2.55, hx + 4.35, cy - 3.25)
ax.text(hx + 4.55, cy - 2.9, r"$\times\, W_o$", fontsize=12, color=INK, va="center")
ax.text(hx + 5.6, cy - 2.72, r"$W_o$ (384, 384) — 헤드별로 따로 놓인 값을 섞는다" "\n"
        "(형상은 그대로, 잔차에 더할 E차원 출력)", fontsize=9.5, color=RED, va="center", linespacing=1.4)
token_matrix(hx + 2.4, cy - 3.35, 12, cw=0.34, label="어텐션 출력  (5, 384)  → (a)의 잔차 연결(⊕)로")

ax.text(OX, 3.0, "• 문장 1도 같은 계산을 따로 한다. 배치 축 B와 헤드 축 H는 행렬곱의 배치 축이라 동시에 계산되고,\n"
        "   문장끼리는 정보가 섞이지 않는다. 실제 텐서는 (B, H, C, C) = (2, 6, 5, 5).\n"
        "• 토큰은 tiny_codes.bin의 실제 BPE 토큰(␣는 앞 공백). 점수·가중치 값은 설명용 예시.\n"
        "• 실제 학습은 B=32, C=256 (ch03/01_pretrain.py). 그림은 B=2, C=5로 줄인 것.",
        fontsize=10.5, color=INK, va="center", linespacing=1.7)

# (a)·(b) 두 패널을 각각 따로 저장한다(노트 본문 폭에서도 글자가 읽히도록).
outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch02/images"
for name, (x0, x1) in {"a": (0.0, 7.6), "b": (7.75, 17.0)}.items():
    out = outdir / f"fig02-flow-{name}.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=Bbox([[x0, 0.0], [x1, 16.5]]))
    print(out)
print("x:", X_TOK, "\ny:", Y_TOK)
