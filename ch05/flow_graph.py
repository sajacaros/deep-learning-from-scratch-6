# 5장 노트의 그림 "개선된 GPT의 순전파와 KV 캐시 생성 흐름"을 그린다.
# 모델 설정은 노트 5.4.4(어휘 10,000, E=384, 헤드 6×64, 블록 6, SwiGLU 중간 1,024)를 따른다.
# StoryBot의 학습된 가중치·토크나이저는 저장소에 없으므로, 토큰은 그림 5-7의 예("A", " cat", " sat", " on")를 쓰고,
# 형상·RoPE 각도·점수·캐시 일치 여부는 ch05/05_kvcache.py의 클래스를 무작위 초기화(seed 0)로 실제로 돌려 얻은 값을 쓴다.
# 실행(저장소 루트에서): .venv/bin/python ch05/flow_graph.py → docs/ch05/images/fig05-flow-a.png, fig05-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 데이터 (05_kvcache.py의 클래스를 그대로 씀)
ns = {}
exec((ROOT / "ch05/05_kvcache.py").read_text().split("# 테스트")[0], ns)
GPT, RoPE = ns["GPT"], ns["RoPE"]
VOCAB, CTX, E, H_, L, FF, THETA = 10000, 256, 384, 6, 6, int(384 * 8 / 3), 10000
D = E // H_
torch.manual_seed(0)
model = GPT(VOCAB, CTX, E, H_, L, FF, THETA).eval()
N_PARAMS = sum(p.numel() for p in model.parameters())
FFN_GPT2 = 2 * E * (4 * E)
FFN_SWI = 3 * E * FF

TOKS = ["A", "␣cat", "␣sat", "␣on"]
IDS = torch.tensor([[11, 22, 33, 44]])      # 무작위 모델용 임의 ID (형상·일치 검사에만 쓴다)
STEPS = []                                  # (입력 길이, offset, 캐시 형상, 점수 형상, logits 형상)
with torch.no_grad():
    attn0 = model.blocks[0].attn

    model.clear_cache()
    for inp in (IDS[:, :2], IDS[:, 2:3], IDS[:, 3:4]):
        off = attn0.cache_offset
        logits = model(inp, use_cache=True)
        STEPS.append((inp.shape[1], off, tuple(attn0.k_cache.shape), (1, H_, inp.shape[1], attn0.k_cache.shape[2]),
                      tuple(logits.shape)))
    last_cache = logits[0, -1]
    full = model(IDS)                       # 캐시 없이 4토큰 전체
    DIFF = (full[0, -1] - last_cache).abs().max().item()
    # offset을 넘기지 않은 경우: 디코드 토큰이 0번 위치로 회전된다
    model.clear_cache()
    model(IDS[:, :2], use_cache=True)
    for b in model.blocks:
        b.attn.cache_offset = 0
    bad = model(IDS[:, 2:3], use_cache=True)
    DIFF_BAD = (full[0, 2] - bad[0, -1]).abs().max().item()
    model.clear_cache()

rope = RoPE(THETA, D, CTX)
INV = 1.0 / (THETA ** (torch.arange(D // 2) * 2 / D))
KS = [0, 1, 15, 31]
torch.manual_seed(1)
q, k = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)


def at(x, pos):
    return rope(x.repeat(1, 1, pos + 1, 1))[:, :, pos]


SC = [(m, n, (at(q, m) * at(k, n)).sum().item()) for m, n in [(4, 2), (5, 3), (100, 98), (4, 3)]]
SC_RAW = (q * k).sum().item()

# ---------------------------------------------------------------- 스타일 (2장 흐름도와 같음)
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_EMB, C_ATT, C_FFN, C_LN, C_OUT, C_BLK = "#F6D6D6", "#F9E3C2", "#CFE3F3", "#F3F0C4", "#E2DAF0", "#F4F4F4"
W_IN = 9.0                       # 그림 폭(인치). 820px로 보이면 1pt ≈ 1.27px


def new_fig(h_units):
    fig = plt.figure(figsize=(W_IN, h_units / 2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18); ax.set_ylim(0, h_units); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc, fs=11, bold=False, ec=INK, lw=1.1, color=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=3, ls=ls))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=4, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, cs="arc3", ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def plus(ax, x, y, r=0.26):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.1, zorder=4))
    ax.plot([x - r * .6, x + r * .6], [y, y], color=INK, lw=1.1, zorder=5)
    ax.plot([x, x], [y - r * .6, y + r * .6], color=INK, lw=1.1, zorder=5)


def shape(ax, x, y, s, ha="left", fs=10):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=GRAY, zorder=4)


def save(fig, name):
    out = OUTDIR / name
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) <= 10000:
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(out, optimize=True)
    print(out)


OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch05/images"

# =====================================================================
# (a) 개선된 GPT 순전파
# =====================================================================
HA = 22.6
fig, ax = new_fig(HA)
ax.text(0.3, HA - 0.7, "(a)  개선된 GPT — 토큰 3개가 지나가는 길 (학습·프리필)", fontsize=15, fontweight="bold",
        color=INK, va="center")
ax.text(0.3, HA - 1.45, f"5.4.4 설정: 어휘 {VOCAB:,}, E={E}, 헤드 {H_}×{D}, 블록 {L}, SwiGLU 중간 {FF:,}  ·  "
        f"파라미터 {N_PARAMS:,}개", fontsize=10, color=GRAY, va="center")
X = 4.6

# 입력
y = HA - 2.9
ax.text(X + 3.4, y, "입력 (1, 3)", fontsize=10.5, va="center", color=GRAY)
for i in range(3):
    box(ax, X - 2.1 + i * 2.1, y, 1.9, 0.75, TOKS[i], POSL[i], fs=11, ec=POS[i], lw=1.3)
arrow(ax, X, y - 0.4, X, y - 1.0)
y = y - 1.45
box(ax, X, y, 4.6, 0.85, "토큰 임베딩  표 (10000, 384)", C_EMB, fs=11)
shape(ax, X + 0.3, y - 0.75, "(1, 3, 384)")
# 사라진 위치 임베딩
box(ax, X + 5.0, y, 3.2, 0.75, "", "white", ec="#BBBBBB", ls=(0, (4, 3)))
ax.text(X + 5.0, y, "+ 위치 임베딩", fontsize=10.5, color="#AAAAAA", ha="center", va="center", zorder=5)
ax.plot([X + 3.5, X + 6.5], [y - 0.3, y + 0.3], color=RED, lw=1.3, zorder=6)
ax.text(X + 6.8, y, "  없음 — RoPE가 대신", fontsize=10, color=RED, ha="left", va="center")

# 블록
BT, BB = y - 1.15, y - 13.2
ax.add_patch(FancyBboxPatch((X - 3.3, BB), 6.6, BT - BB, boxstyle="round,pad=0,rounding_size=0.3",
                            fc=C_BLK, ec=INK, lw=1.3, zorder=1))
ax.text(X - 3.1, BB + 0.3, "Block", fontsize=10.5, color=GRAY, va="center")
ax.text(X - 4.05, (BT + BB) / 2, f"× {L}", fontsize=16, fontweight="bold", va="center", ha="center")
arrow(ax, X, y - 0.43, X, BT - 0.55)
yy = BT - 0.9
box(ax, X, yy, 3.4, 0.65, "RMSNorm (γ만)", C_LN, fs=10.5)
arrow(ax, X, yy - 0.33, X, yy - 0.8)
yA = yy - 2.2
box(ax, X, yA, 5.4, 2.8, "", C_ATT)
ax.text(X, yA + 1.05, "멀티헤드 어텐션", fontsize=12, fontweight="bold", ha="center", va="center", zorder=5)
ax.text(X, yA - 0.1, r"$Q, K, V = xW$" "  (1, 6, 3, 64)\n" r"RoPE($Q$), RoPE($K$)  — $V$는 그대로" "\n"
        r"인과 마스크 → softmax → $\times V$ → $W_o$",
        fontsize=10, ha="center", va="center", zorder=5, linespacing=1.6)
ax.text(X, yA - 1.15, "bias 없음, 드롭아웃 없음", fontsize=9.5, color=GRAY, ha="center", va="center", zorder=5)
arrow(ax, X, yA - 1.42, X, yA - 1.93)
yP1 = yA - 2.2
plus(ax, X, yP1)
ax.plot([X, X - 3.0, X - 3.0], [BT - 0.35, BT - 0.35, yP1], color=INK, lw=1.1, zorder=2)
arrow(ax, X - 3.0, yP1, X - 0.26, yP1)
shape(ax, X + 0.4, yP1, "(1, 3, 384)")
arrow(ax, X, yP1 - 0.26, X, yP1 - 0.75)
yy = yP1 - 1.08
box(ax, X, yy, 3.4, 0.65, "RMSNorm (γ만)", C_LN, fs=10.5)
arrow(ax, X, yy - 0.33, X, yy - 0.8)
yS = yy - 2.1
box(ax, X, yS, 5.4, 2.6, "", C_FFN)
ax.text(X, yS + 0.95, "SwiGLU", fontsize=12, fontweight="bold", ha="center", va="center", zorder=5)
ax.text(X, yS - 0.05, r"SiLU($xW$) $\odot$ $xV$" f"   ({E}→{FF:,})\n" r"$\to$ $O$" f"   ({FF:,}→{E})",
        fontsize=10.5, ha="center", va="center", zorder=5, linespacing=1.6)
ax.text(X, yS - 0.95, "토큰(행)마다 따로 계산", fontsize=9.5, color=GRAY, ha="center", va="center", zorder=5)
arrow(ax, X, yS - 1.32, X, yS - 1.78)
yP2 = yS - 2.05
plus(ax, X, yP2)
ax.plot([X, X - 3.0, X - 3.0], [yP1 - 0.55, yP1 - 0.55, yP2], color=INK, lw=1.1, zorder=2)
arrow(ax, X - 3.0, yP2, X - 0.26, yP2)
shape(ax, X + 0.4, yP2, "(1, 3, 384)")

# 출력
arrow(ax, X, yP2 - 0.26, X, BB - 0.45)
yy = BB - 0.8
box(ax, X, yy, 3.4, 0.65, "RMSNorm", C_LN, fs=10.5)
arrow(ax, X, yy - 0.33, X, yy - 0.8)
yy -= 1.15
box(ax, X, yy, 5.4, 0.8, f"Linear {E}→{VOCAB:,} (unembed)", C_OUT, fs=11)
ax.text(X, yy - 0.68, "bias 없음 · 임베딩과 가중치 공유 안 함", fontsize=9.5, color=GRAY, ha="center", va="center")
arrow(ax, X, yy - 0.95, X, yy - 1.45)
shape(ax, X + 0.3, yy - 1.75, f"logits (1, 3, {VOCAB:,})  — 위치마다 다음 토큰 분포", fs=10)

# ---------------- 오른쪽: RoPE 안
RX0, RX1 = 9.9, 17.7
rt = BT + 0.3
ax.plot([X + 2.7, RX0], [yA + 0.2, yA + 0.2], color="#E0B97A", lw=1.0, ls=(0, (3, 2)), zorder=2)
ty = rt - 0.45
ax.text(RX0 + 0.25, ty, "RoPE 안 — 64차원을 32쌍으로, 위치 m만큼 회전", fontsize=10.5, fontweight="bold",
        va="center", zorder=5)
ty -= 0.7
ax.text(RX0 + 0.25, ty, r"각도 $m\theta_k$ (라디안),  $\theta_k = 10000^{-2k/64}$", fontsize=10, va="center", zorder=5)
ty -= 0.62
hdr = ["k", r"$\theta_k$", "m=0", "m=1", "m=2"]
cx = [RX0 + 0.45, RX0 + 1.5, RX0 + 3.35, RX0 + 4.85, RX0 + 6.35]
for c, h in zip(cx, hdr):
    ax.text(c, ty, h, fontsize=10, fontweight="bold", va="center", ha="center", zorder=5)
for j, kk in enumerate(KS):
    yr = ty - 0.52 * (j + 1)
    th = INV[kk].item()
    vals = [str(kk), f"{th:.4f}" if th > 1e-3 else f"{th:.1e}"] + \
           [f"{m * th:.4f}" if m * th < 0.1 and m else f"{m * th:.2f}" for m in (0, 1, 2)]
    for c, v in zip(cx, vals):
        ax.text(c, yr, v, fontsize=9.5, va="center", ha="center", zorder=5)
ty = ty - 0.52 * len(KS) - 0.85
ax.text(RX0 + 0.25, ty, "앞쪽 쌍은 한 칸에 1라디안(57°)씩 빠르게,\n뒤쪽 쌍은 거의 돌지 않는다.",
        fontsize=9.5, color=GRAY, va="center", zorder=5, linespacing=1.4)
# k=0 쌍의 회전 그림
cxx, cyy, rr = RX0 + 6.5, ty - 0.55, 0.7
ax.add_patch(Circle((cxx, cyy), rr, fc="none", ec="#CCCCCC", lw=0.8, zorder=4))
for m in range(3):
    a = 0.35 + m * 1.0
    ax.add_patch(FancyArrowPatch((cxx, cyy), (cxx + rr * np.cos(a), cyy + rr * np.sin(a)), arrowstyle="-|>",
                                 mutation_scale=9, color=POS[m], lw=1.3, zorder=5, shrinkA=0, shrinkB=0))
    ax.text(cxx + (rr + 0.3) * np.cos(a), cyy + (rr + 0.3) * np.sin(a), f"m={m}", fontsize=9, color=POS[m],
            ha="center", va="center", zorder=5)
ty -= 1.95
ax.text(RX0 + 0.25, ty, "같은 q, k를 위치만 바꿔 놓은 점수 q·k", fontsize=10, fontweight="bold", va="center", zorder=5)
for j, (m, n, s) in enumerate(SC):
    yr = ty - 0.5 * (j + 1)
    ax.text(RX0 + 0.45, yr, f"(m, n) = ({m}, {n})   m−n = {m - n}", fontsize=9.5, va="center", zorder=5)
    ax.text(RX1 - 0.3, yr, f"{s:.4f}", fontsize=9.5, va="center", ha="right", zorder=5,
            color=RED if m - n == 2 else INK, fontweight="bold" if m - n == 2 else "normal")
yr = ty - 0.5 * (len(SC) + 1)
ax.text(RX0 + 0.45, yr, "회전 전 (위치 없음)", fontsize=9.5, va="center", color=GRAY, zorder=5)
ax.text(RX1 - 0.3, yr, f"{SC_RAW:.4f}", fontsize=9.5, va="center", ha="right", color=GRAY, zorder=5)
ax.text(RX0 + 0.25, yr - 0.75, "m−n이 같으면 점수가 같다(float32 오차 수준).\ncos/sin 표 (256, 32) 하나를 6블록이 공유",
        fontsize=9.5, color=RED, va="center", zorder=5, linespacing=1.4)
rb = yr - 1.35
ax.add_patch(FancyBboxPatch((RX0, rb), RX1 - RX0, rt - rb, boxstyle="round,pad=0,rounding_size=0.15",
                            fc="white", ec="#E0B97A", lw=1.1, zorder=3))

# ---------------- 오른쪽 아래: SwiGLU 파라미터
sy0 = rb - 0.4
ax.add_patch(FancyBboxPatch((RX0, sy0 - 3.3), RX1 - RX0, 3.3, boxstyle="round,pad=0,rounding_size=0.15",
                            fc="white", ec="#7FA7CF", lw=1.1, zorder=3))
ax.plot([X + 2.7, RX0], [yS, sy0 - 1.6], color="#7FA7CF", lw=1.0, ls=(0, (3, 2)), zorder=2)
ax.text(RX0 + 0.25, sy0 - 0.45, "FFN 파라미터 수 (bias 제외)", fontsize=10.5, fontweight="bold", va="center", zorder=5)
ax.text(RX0 + 0.25, sy0 - 1.15, f"GPT-2   2 × {E} × {4 * E:,}", fontsize=10, va="center", zorder=5)
ax.text(RX1 - 0.3, sy0 - 1.15, f"{FFN_GPT2:,}", fontsize=10, va="center", ha="right", zorder=5)
ax.text(RX0 + 0.25, sy0 - 1.75, f"SwiGLU  3 × {E} × {FF:,}", fontsize=10, va="center", zorder=5)
ax.text(RX1 - 0.3, sy0 - 1.75, f"{FFN_SWI:,}", fontsize=10, va="center", ha="right", zorder=5)
ax.text(RX0 + 0.25, sy0 - 2.6, f"int({E}×8/3) = {FF:,}이라 두 값이 정확히 같다.\nLinear가 3개라 중간 차원을 줄여 맞춘 것",
        fontsize=9.5, color=RED, va="center", zorder=5, linespacing=1.4)
save(fig, "fig05-flow-a.png")

# =====================================================================
# (b) KV 캐시로 생성
# =====================================================================
HB = 19.9
fig, ax = new_fig(HB)
ax.text(0.3, HB - 0.7, "(b)  KV 캐시로 생성 — 프리필 1번, 디코드 2번 (블록 하나의 어텐션)", fontsize=15,
        fontweight="bold", color=INK, va="center")
ax.text(0.3, HB - 1.45, "토큰은 그림 5-7의 예. 형상은 05_kvcache.py를 실제로 실행해 확인한 값 (B=1, 헤드 6, D=64)",
        fontsize=10, color=GRAY, va="center")
COLX = [3.0, 9.0, 15.0]
names = ["① 프리필", "② 디코드", "③ 디코드"]
inputs = [[0, 1], [2], [3]]
preds = ["␣sat", "␣on", "…"]
row_y = {"head": HB - 2.6, "in": HB - 3.55, "off": HB - 4.65, "kv": HB - 6.6, "sc": HB - 9.6, "out": HB - 11.6}
ax.text(0.15, row_y["in"], "입력", fontsize=10, color=GRAY, va="center")
ax.text(0.15, row_y["kv"] + 0.6, "K 캐시", fontsize=10, color=GRAY, va="center")
ax.text(0.15, row_y["sc"] + 0.6, "점수", fontsize=10, color=GRAY, va="center")
ax.text(0.15, row_y["out"], "출력", fontsize=10, color=GRAY, va="center")
for c, (cx, nm, inp, st, pr) in enumerate(zip(COLX, names, inputs, STEPS, preds)):
    C, off, kshape, sshape, lshape = st
    if c:
        ax.plot([cx - 3.0, cx - 3.0], [row_y["out"] - 1.2, row_y["head"] + 0.4], color="#DDDDDD", lw=1.0)
    ax.text(cx, row_y["head"], nm, fontsize=12.5, fontweight="bold", ha="center", va="center")
    # 입력 토큰
    w = 1.5
    x0 = cx - w * len(inp) / 2
    for j, t in enumerate(inp):
        box(ax, x0 + w * j + w / 2, row_y["in"], w - 0.1, 0.65, TOKS[t], POSL[t], fs=10.5, ec=POS[t], lw=1.3)
    shape(ax, cx, row_y["in"] - 0.55, f"입력 (1, {C})", ha="center", fs=9.5)
    ax.text(cx, row_y["off"], f"RoPE offset = {off}  → 위치 {', '.join(str(off + j) for j in range(C))}",
            fontsize=10, ha="center", va="center", color=RED if c else INK)
    # K 캐시: 칸 = 위치
    n = kshape[2]
    cw = 1.15
    gx = cx - cw * n / 2
    for j in range(n):
        new = j >= off
        ax.add_patch(Rectangle((gx + j * cw, row_y["kv"]), cw - 0.08, 1.2, fc=POSL[j] if new else "#EEEEEE",
                               ec=POS[j] if new else "#BBBBBB", lw=1.3 if new else 0.8, zorder=3))
        ax.text(gx + j * cw + cw / 2 - 0.04, row_y["kv"] + 0.82, f"K{j + 1}", fontsize=10, ha="center",
                va="center", zorder=4)
        ax.text(gx + j * cw + cw / 2 - 0.04, row_y["kv"] + 0.38, "새로" if new else "캐시", fontsize=9,
                ha="center", va="center", color=INK if new else GRAY, zorder=4)
    shape(ax, cx, row_y["kv"] - 0.4, f"k_cache {kshape}", ha="center", fs=9.5)
    shape(ax, cx, row_y["kv"] - 0.85, "V 캐시도 같은 모양", ha="center", fs=9.5)
    # 점수 격자: 행 = 쿼리, 열 = 키
    cs = 0.62
    sx0 = cx - cs * n / 2 + 0.2
    sy0 = row_y["sc"] + 1.2
    for r in range(C):
        qpos = off + r
        ax.text(sx0 - 0.12, sy0 - r * cs - cs / 2, f"Q{qpos + 1}", fontsize=9.5, ha="right", va="center",
                color=POS[qpos])
        for j in range(n):
            masked = j > qpos
            ax.add_patch(Rectangle((sx0 + j * cs, sy0 - (r + 1) * cs), cs, cs, fc="#E3E3E3" if masked else POSL[qpos],
                                   ec="#BBBBBB" if masked else POS[qpos], lw=0.8, zorder=3))
            ax.text(sx0 + j * cs + cs / 2, sy0 - r * cs - cs / 2, r"$-\infty$" if masked else "", fontsize=9,
                    ha="center", va="center", color=GRAY, zorder=4)
    for j in range(n):
        ax.text(sx0 + j * cs + cs / 2, sy0 + 0.22, f"K{j + 1}", fontsize=9, ha="center", va="center", color=GRAY)
    yb = sy0 - C * cs - 0.4
    shape(ax, cx, yb, f"scores {sshape}", ha="center", fs=9.5)
    ax.text(cx, yb - 0.5, "인과 마스크 필요" if c == 0 else "마스크 없음 — 키가 모두 과거·현재",
            fontsize=9.5, ha="center", va="center", color=INK if c == 0 else RED)
    # 출력
    shape(ax, cx, row_y["out"] + 0.45, f"logits {lshape}", ha="center", fs=9.5)
    ax.text(cx, row_y["out"] - 0.1, "마지막 행만 사용 → 다음 토큰", fontsize=9.5, ha="center", va="center")
    if pr != "…":
        box(ax, cx, row_y["out"] - 0.85, 1.6, 0.62, pr, POSL[c + 2], fs=10.5, ec=POS[c + 2], lw=1.3)
        ax.text(cx + 0.95, row_y["out"] - 0.85, "→ 다음 입력", fontsize=9.5, color=GRAY, va="center")
    else:
        ax.text(cx, row_y["out"] - 0.85, "…", fontsize=12, ha="center", va="center")

# 아래: 캐시 없이 했다면 / 실제 검증
yb = row_y["out"] - 1.7
ax.add_patch(Rectangle((0.4, yb - 5.6), 17.2, 5.6, fc="#FAFAFA", ec="#DDDDDD", lw=0.8, zorder=0))
ax.text(0.7, yb - 0.5, "③에서 캐시가 없다면", fontsize=11, fontweight="bold", va="center")
rows = [("", "캐시 없음", "KV 캐시"),
        ("입력", "4토큰 전체 (1, 4)", "새 토큰 1개 (1, 1)"),
        ("새로 계산하는 Q·K·V", "4토큰분", "1토큰분"),
        ("점수", "(1, 6, 4, 4) — 16칸", "(1, 6, 1, 4) — 4칸"),
        ("FFN·RMSNorm·unembed", "4행", "1행")]
for j, (a, b_, c_) in enumerate(rows):
    yr = yb - 1.15 - j * 0.55
    fw = "bold" if j == 0 else "normal"
    ax.text(0.9, yr, a, fontsize=10, va="center", fontweight=fw)
    ax.text(7.2, yr, b_, fontsize=10, va="center", fontweight=fw)
    ax.text(12.4, yr, c_, fontsize=10, va="center", fontweight=fw, color=RED if j else INK)
ax.text(0.7, yb - 4.4,
        f"• 실제 검증(무작위 초기화 모델): ③의 마지막 logits와 캐시 없이 4토큰을 넣은 logits의 최대 차이 {DIFF:.1e}\n"
        f"• offset을 넘기지 않으면(②의 토큰을 0번 위치로 회전) ②의 logits가 캐시 없는 계산과 최대 {DIFF_BAD:.3f} 달라진다\n"
        f"• 캐시는 블록마다 K, V 하나씩 — 블록 {L}개 × 2개가 생성 길이에 비례해 늘어난다",
        fontsize=10, va="center", linespacing=1.8)
save(fig, "fig05-flow-b.png")
print("steps", STEPS, "diff", DIFF, "bad", DIFF_BAD, "SC", SC, SC_RAW, "params", N_PARAMS, FFN_GPT2, FFN_SWI)
