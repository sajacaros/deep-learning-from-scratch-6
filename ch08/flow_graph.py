# 8장 노트의 그림 "WebBot 모델 한눈에 보기"를 그린다.
# (a) 친칠라 법칙으로 크기를 정하고 webbot/model.py로 실제 파라미터 수를 센 결과
# (b) 어텐션 한 층 안의 GQA(쿼리 헤드 12 : KV 헤드 4)와 Flash Attention(블록 처리 + 온라인 소프트맥스)
# 파라미터 수는 webbot/model.py의 GPT를 ch09/01_pretrain.py 설정으로 만들어 실제로 센 값이다.
# 온라인 소프트맥스의 점수 8개는 설명용 예시 값이고, m·ℓ·가중치는 그 값으로 실제 계산한 것이다.
# 실행(저장소 루트에서): .venv/bin/python ch08/flow_graph.py → docs/ch08/images/fig08-flow-a.png, fig08-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.transforms import Bbox
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from webbot.model import GPT

# ---------------------------------------------------------------- 데이터 (실제 계산)
CFG = dict(vocab_size=50000, max_context_len=1024, embed_dim=768, n_head=12, n_kv_head=4, n_layer=12, ff_dim=2048)
TOKENS = 2.6e9                                     # 7장에서 준비한 학습 데이터


def count(cfg):
    m = GPT(**cfg)
    parts = {"embed": 0, "attn": 0, "mlp": 0, "unembed": 0, "norm": 0}
    for n, p in m.named_parameters():
        key = next(k for k in ("unembed", "norm", "attn", "mlp", "embed") if k in n)
        parts[key] += p.numel()
    return sum(parts.values()), parts, m


N, PARTS, model = count(CFG)
N_MHA, _, _ = count({**CFG, "n_kv_head": 12})
blk = model.blocks[0]
L_ATT = {k: getattr(blk.attn, k).weight.numel() for k in ("W_q", "W_k", "W_v", "W_o")}
L_MLP = sum(p.numel() for p in blk.mlp.parameters())
D_HEAD = CFG["embed_dim"] // CFG["n_head"]
KV_BYTES = lambda kv: CFG["n_layer"] * CFG["max_context_len"] * 2 * kv * D_HEAD * 2   # K·V, bf16
FLOP = 6 * N * TOKENS

# 온라인 소프트맥스 예시: 쿼리 하나의 점수 8개를 블록 2개(4개씩)로 나눠 처리
S = np.array([1.2, -0.4, 2.1, 0.3, 3.0, 0.8, -1.1, 1.5])
m1 = S[:4].max(); l1 = np.exp(S[:4] - m1).sum()
m2 = max(m1, S[4:].max()); l1c = l1 * np.exp(m1 - m2); lb2 = np.exp(S[4:] - m2).sum(); l2 = l1c + lb2
W_ON = np.exp(S - m2) / l2
W_FULL = np.exp(S - S.max()) / np.exp(S - S.max()).sum()
DIFF = np.abs(W_ON - W_FULL).max()

# ---------------------------------------------------------------- 스타일 (2장 그림과 같은 색·폰트)
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_EMB, C_ATT, C_FFN, C_LN, C_OUT, C_BLK = "#F6D6D6", "#F9E3C2", "#CFE3F3", "#F3F0C4", "#E2DAF0", "#F4F4F4"
WU = 17.0   # 가로 단위 수 (그림 폭 8.5인치 → 1단위 = 0.5인치)


def new_fig(h_units):
    fig = plt.figure(figsize=(WU / 2, h_units / 2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WU); ax.set_ylim(0, h_units); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc, ec=INK, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))


def text(ax, x, y, s, fs=11, color=INK, ha="center", bold=False, va="center", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5,
            fontweight="bold" if bold else "normal", **kw)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, cs="arc3", ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def mb(x):
    return f"{x / 1e6:.1f}M"


def save(fig, name, outdir, y_min=0.0):
    out = outdir / name
    h = fig.get_size_inches()[1]
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=Bbox([[0, y_min / 2], [WU / 2, h]]))
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(maxcolors=1 << 24)) > 10000:
        im.save(out, optimize=True)
    else:
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out)


outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch08/images"

# =====================================================================
# (a) 계산 예산 → 모델 크기 → 실제 파라미터 수
# =====================================================================
H = 21.0
fig, ax = new_fig(H)
text(ax, 0.4, H - 0.55, "(a)  WebBot의 크기가 정해지는 과정", fs=15, bold=True, ha="left")

# 1줄: 데이터 → 친칠라 → 목표 크기
y = H - 2.0
for cx, t1, t2, fc in [(2.6, "학습 데이터", "2.6B 토큰 (7장)", "white"),
                       (8.5, "친칠라 법칙", "토큰 ≈ 파라미터 × 20", C_LN),
                       (14.4, "목표 크기", "2.6B ÷ 20 ≈ 130M", "white")]:
    box(ax, cx, y, 4.4, 1.15, fc)
    text(ax, cx, y + 0.22, t1, fs=12)
    text(ax, cx, y - 0.25, t2, fs=10.5, color=GRAY)
arrow(ax, 4.8, y, 6.3, y); arrow(ax, 10.7, y, 12.2, y)

# 2줄: 표 8-1 설정
arrow(ax, 14.4, y - 0.6, 14.4, y - 1.35)
text(ax, 14.6, y - 0.95, "조금 크게", fs=10, color=GRAY, ha="left")
y2 = y - 2.55
box(ax, 8.5, y2, 16.2, 2.3, C_BLK)
text(ax, 0.75, y2 + 0.8, "표 8-1 설정 (ch09/01_pretrain.py)", fs=11.5, bold=True, ha="left")
cfg_items = [("어휘", "50,000"), ("층", "12"), ("임베딩", "768"), ("헤드", "12"),
             ("KV 헤드", "4"), ("FFN", "2,048"), ("컨텍스트", "1,024")]
for k, (a, b) in enumerate(cfg_items):
    cx = 1.6 + k * 2.3
    box(ax, cx, y2 - 0.25, 2.05, 1.05, "#FBE9D0" if a == "KV 헤드" else "white",
        ec=RED if a == "KV 헤드" else INK)
    text(ax, cx, y2 - 0.05, a, fs=10, color=GRAY)
    text(ax, cx, y2 - 0.45, b, fs=12, color=RED if a == "KV 헤드" else INK)

# 3줄: 실제로 센 파라미터 수
arrow(ax, 8.5, y2 - 1.15, 8.5, y2 - 1.85)
text(ax, 8.75, y2 - 1.5, "webbot/model.py의 GPT를 이 설정으로 만들어 셈", fs=10, color=GRAY, ha="left")
y3 = y2 - 2.5
text(ax, 8.5, y3, f"실제 파라미터 수  {N:,}개  ≈ {N / 1e6:.0f}M", fs=14, bold=True)
text(ax, 8.5, y3 - 0.6, f"파라미터당 {TOKENS / N:.1f}토큰 — 친칠라 기준 20에 가깝다", fs=11, color=GRAY)

# 누적 막대: 어디에 파라미터가 있나
yb = y3 - 1.95
bx0, bx1 = 0.8, 16.2
scale = (bx1 - bx0) / N
segs = [("embed", "토큰 임베딩", C_EMB), ("attn", "어텐션×12", C_ATT),
        ("mlp", "SwiGLU × 12층", C_FFN), ("unembed", "출력층 unembed", C_OUT)]
xx = bx0
for key, lab, col in segs:
    w = PARTS[key] * scale
    ax.add_patch(Rectangle((xx, yb - 0.45), w, 0.9, fc=col, ec=INK, lw=1.0, zorder=3))
    text(ax, xx + w / 2, yb + 0.12, lab, fs=10.5)
    text(ax, xx + w / 2, yb - 0.22, f"{mb(PARTS[key])}  ({PARTS[key] / N * 100:.0f}%)", fs=10, color=GRAY)
    xx += w
text(ax, bx0, yb + 0.85, "파라미터가 있는 곳", fs=11.5, bold=True, ha="left")
text(ax, bx1, yb + 0.85, f"RMSNorm {PARTS['norm']:,}개는 너무 작아 생략", fs=10, color=GRAY, ha="right")
text(ax, bx0, yb - 0.85, "임베딩과 출력층(가중치 공유 안 함)이 절반 — 어휘 50,000의 비용", fs=10.5, color=RED, ha="left")

# 층 하나 펼치기
yl = yb - 2.25
text(ax, bx0, yl + 0.55, "층 하나 펼치기", fs=11.5, bold=True, ha="left")
att_w = [("W_q", L_ATT["W_q"], "768→768"), ("W_k", L_ATT["W_k"], "768→256"),
         ("W_v", L_ATT["W_v"], "768→256"), ("W_o", L_ATT["W_o"], "768→768")]
lscale = (bx1 - bx0) / (sum(L_ATT.values()) + L_MLP)
xx = bx0
for nm, v, shp in att_w:
    w = v * lscale
    red = nm in ("W_k", "W_v")
    ax.add_patch(Rectangle((xx, yl - 0.9), w, 0.9, fc=C_ATT, ec=RED if red else INK, lw=1.4 if red else 1.0, zorder=3 + red))
    text(ax, xx + w / 2, yl - 0.45, f"$W_{nm[-1]}$", fs=11.5, color=RED if red else INK)
    xx += w
text(ax, bx0, yl - 1.25, r"$W_q, W_o$: 768×768 = 0.59M      " r"$W_k, W_v$: 768×256 = 0.20M  (KV 헤드 4 × 64차원)",
     fs=10.5, color=RED, ha="left")
for nm in ("W", "V", "O"):
    w = L_MLP / 3 * lscale
    ax.add_patch(Rectangle((xx, yl - 0.9), w, 0.9, fc=C_FFN, ec=INK, lw=1.0, zorder=3))
    text(ax, xx + w / 2, yl - 0.45, f"SwiGLU {nm}  ({L_MLP / 3 / 1e6:.2f}M)", fs=10)
    xx += w
text(ax, bx0, yl - 1.75, f"어텐션 {mb(sum(L_ATT.values()))}  +  SwiGLU {mb(L_MLP)}  =  층당 {mb(sum(L_ATT.values()) + L_MLP)}",
     fs=10.5, color=GRAY, ha="left")

# GQA 효과: MHA였다면
yg = yl - 3.15
text(ax, bx0, yg + 0.55, "GQA(KV 헤드 4)를 쓰지 않고 MHA(KV 헤드 12)였다면", fs=11.5, bold=True, ha="left")
rows = [("파라미터", N_MHA, N, lambda v: f"{v / 1e6:.1f}M", 170e6),
        ("KV 캐시 (배치 1, 1,024토큰, bf16)", KV_BYTES(12), KV_BYTES(4), lambda v: f"{v / 1e6:.1f}MB", 40e6)]
for k, (lab, vm, vg, fmt, vmax) in enumerate(rows):
    yy = yg - 0.35 - k * 1.55
    text(ax, bx0, yy - 0.05, lab, fs=10.5, ha="left")
    for j, (nm, v, col) in enumerate([("MHA", vm, "#E3E3E3"), ("GQA", vg, "#FBE9D0")]):
        ry = yy - 0.45 - j * 0.45
        w = v / vmax * 8.5
        text(ax, bx0 + 0.95, ry, nm, fs=10, color=GRAY, ha="right")
        ax.add_patch(Rectangle((bx0 + 1.1, ry - 0.18), w, 0.36, fc=col, ec=RED if nm == "GQA" else GRAY, lw=1.0, zorder=3))
        text(ax, bx0 + 1.25 + w, ry, fmt(v), fs=10.5, color=RED if nm == "GQA" else INK, ha="left")
text(ax, bx1, yg - 0.4, f"파라미터는 {(N_MHA - N) / 1e6:.1f}M(약 {(N_MHA - N) / N_MHA * 100:.0f}%)만 줄지만",
     fs=10.5, color=GRAY, ha="right")
text(ax, bx1, yg - 1.95, "KV 캐시는 1/3 — 생성 때 쌓이는 메모리", fs=10.5, color=RED, ha="right")

# 학습 계산량
yc = yg - 4.0
box(ax, 8.5, yc, 16.2, 1.2, "white")
text(ax, 8.5, yc + 0.22, r"학습 계산량  $C \approx 6ND$" f" = 6 × {N / 1e6:.0f}M × 2.6B ≈ {FLOP / 1e18:.1f}" r"$\times 10^{18}$ FLOP", fs=12)
text(ax, 8.5, yc - 0.27, "토큰 하나당 순전파 2N(곱셈·덧셈) + 역전파 4N", fs=10.5, color=GRAY)
save(fig, "fig08-flow-a.png", outdir, y_min=yc - 1.0)

# =====================================================================
# (b) 어텐션 한 층: GQA + Flash Attention
# =====================================================================
H = 22.0
fig, ax = new_fig(H)
text(ax, 0.4, H - 0.55, "(b)  WebBot 어텐션 한 층 — GQA와 Flash Attention", fs=15, bold=True, ha="left")

# 입력
y0 = H - 1.6
box(ax, 8.5, y0, 6.0, 0.8, C_LN)
text(ax, 8.5, y0, "RMSNorm 출력 x   (B, C, 768)", fs=11.5)

# Q, K, V 헤드
GC = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
GCL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#DAD4EC"]
yq = y0 - 2.6
hw, hh = 0.62, 1.0
qx0 = 0.9
q_cx = []
for h in range(12):
    g = h // 3
    x = qx0 + h * 0.72 + g * 0.18
    ax.add_patch(Rectangle((x, yq - hh / 2), hw, hh, fc=GCL[g], ec=GC[g], lw=1.2, zorder=3))
    q_cx.append(x + hw / 2)
text(ax, (q_cx[0] + q_cx[-1]) / 2, yq + 0.85, r"$Q$ = $xW_q$   (B, 12, C, 64)", fs=11.5)
text(ax, (q_cx[0] + q_cx[-1]) / 2, yq - 0.8, "쿼리 헤드 12개", fs=10, color=GRAY, bbox=dict(fc="white", ec="none", pad=1))
kv_cx = {}
for name, x0 in (("K", 10.3), ("V", 13.6)):
    kv_cx[name] = []
    for g in range(4):
        x = x0 + g * 0.72
        ax.add_patch(Rectangle((x, yq - hh / 2), hw, hh, fc=GCL[g], ec=GC[g], lw=1.2, zorder=3))
        kv_cx[name].append(x + hw / 2)
    text(ax, x0 + 1.4, yq + 0.85, f"${name}$ = $xW_{name.lower()}$", fs=11.5)
    text(ax, x0 + 1.4, yq - 0.8, "KV 헤드 4개 (B, 4, C, 64)" if name == "V" else "KV 헤드 4개", fs=10, color=GRAY,
         bbox=dict(fc="white", ec="none", pad=1))
for x1 in (4.2, 11.7, 15.0):
    arrow(ax, 8.5 + (x1 - 8.5) * 0.25, y0 - 0.42, x1, yq + 1.1, lw=1.0)

# 공유 연결: 쿼리 헤드 3개 → KV 헤드 1개
yS = yq - 1.95
for g in range(4):
    for h in range(3 * g, 3 * g + 3):
        ax.plot([q_cx[h], q_cx[h]], [yq - hh / 2, yS + 0.05], color=GC[g], lw=1.0, zorder=2)
    ax.plot([q_cx[3 * g], q_cx[3 * g + 2]], [yS + 0.05, yS + 0.05], color=GC[g], lw=1.6, zorder=2)
    ax.plot([kv_cx["K"][g], kv_cx["K"][g]], [yq - hh / 2, yS - 0.15], color=GC[g], lw=1.0, zorder=2)
    yd = yS - 0.2 - g * 0.16
    ax.plot([q_cx[3 * g + 1], q_cx[3 * g + 1]], [yS + 0.05, yd], color=GC[g], lw=1.6, zorder=2)
    ax.plot([kv_cx["K"][g], kv_cx["K"][g]], [yS - 0.15, yd], color=GC[g], lw=1.0, zorder=2)
    ax.plot([q_cx[3 * g + 1], kv_cx["K"][g]], [yd, yd], color=GC[g], lw=1.0, zorder=2, ls=(0, (3, 2)))
text(ax, 8.5, yS - 1.0, "같은 색 = 한 그룹: 쿼리 헤드 3개가 KV 헤드 1개를 함께 본다  (enable_gqa=True)", fs=10.5)
text(ax, 8.5, yS - 1.5, r"$W_k, W_v$는 768→256(KV 헤드 4 × 64)이라 KV 캐시도 MHA의 1/3", fs=10.5, color=RED)

# Flash Attention 상자
yF1 = yS - 2.0
text(ax, 0.7, yF1 - 0.45, "F.scaled_dot_product_attention(Q, K, V, is_causal=True)  — Flash Attention",
     fs=11.5, bold=True, ha="left")
text(ax, 0.7, yF1 - 0.95, "그룹 하나·쿼리 헤드 하나의 점수 맵 (C × C)을 블록으로 잘라 SRAM 안에서 처리",
     fs=10.5, color=GRAY, ha="left")

# 블록 타일 그리드
nb, ts = 6, 0.78
gx, gy = 1.3, yF1 - 1.9
for r in range(nb):
    for c in range(nb):
        fut = c > r
        hl = r == 3 and c <= r
        fc = "#E3E3E3" if fut else ("#F9E3C2" if hl else "white")
        ax.add_patch(Rectangle((gx + c * ts, gy - (r + 1) * ts), ts, ts, fc=fc,
                               ec=RED if hl else "#BBBBBB", lw=1.4 if hl else 0.8, zorder=3 + hl))
        if c == r:
            ax.add_patch(plt.Polygon([(gx + c * ts, gy - r * ts), (gx + (c + 1) * ts, gy - r * ts),
                                      (gx + (c + 1) * ts, gy - (r + 1) * ts)], fc="#E3E3E3", ec="none", zorder=3.5))
for c in range(4):
    text(ax, gx + c * ts + ts / 2, gy - 3.5 * ts, f"{c + 1}", fs=10, color=RED, bold=True)
text(ax, gx + nb * ts / 2, gy + 0.3, "열 = 키 블록 →", fs=10, color=GRAY)
text(ax, gx - 0.25, gy - nb * ts / 2, "행 = 쿼리 블록", fs=10, color=GRAY, rotation=90)
text(ax, gx + nb * ts / 2, gy - nb * ts - 0.35, "회색 = 미래(인과 마스크)", fs=10, color=GRAY)
gb = gy - nb * ts - 0.95
text(ax, 0.7, gb, "쿼리 블록 하나를 SRAM에 두고, 키·밸류 블록을\n1→2→3→4 순서로 읽으며 결과를 갱신한다.\n"
     "C×C 맵 전체를 DRAM에 쓰지 않는다.", fs=10.5, ha="left", va="top", linespacing=1.5)

yF0 = gb - 1.55
ax.add_patch(FancyBboxPatch((0.4, yF0), 16.2, yF1 - yF0, boxstyle="round,pad=0,rounding_size=0.25",
                            fc=C_BLK, ec=INK, lw=1.3, zorder=1))
# 온라인 소프트맥스 예시
ox = 7.2
text(ax, ox, gy + 0.3, "온라인 소프트맥스 — 쿼리 하나, 점수 8개를 블록 2개로", fs=11, bold=True, ha="left")
cw = 0.98
for j, s in enumerate(S):
    x = ox + j * cw + (0.25 if j >= 4 else 0)
    ax.add_patch(Rectangle((x, gy - 0.75), cw - 0.06, 0.6, fc="#FBE9D0" if j < 4 else "#CFE3F3", ec=GRAY, lw=0.8, zorder=3))
    text(ax, x + cw / 2 - 0.03, gy - 0.45, f"{s:.1f}", fs=10.5)
text(ax, ox + 2 * cw, gy - 1.0, "블록 1", fs=10, color=GRAY)
text(ax, ox + 6 * cw + 0.25, gy - 1.0, "블록 2", fs=10, color=GRAY)
steps = [("블록 1 후", rf"$m = {m1:.2f}$,   $\ell = \sum e^{{x-m}} = {l1:.3f}$", INK),
         ("블록 2 후", rf"$m: {m1:.2f} \to {m2:.2f}$   (최댓값이 바뀜)", INK),
         ("", rf"$\ell = {l1:.3f}\times e^{{{m1:.2f}-{m2:.2f}}} + {lb2:.3f} = {l2:.3f}$", RED),
         ("", "  ↑ 이전 합을 보정        ↑ 블록 2의 합", GRAY)]
for k, (a, b, col) in enumerate(steps):
    yy = gy - 1.6 - k * 0.55
    text(ax, ox, yy, a, fs=10.5, ha="left", color=GRAY)
    text(ax, ox + 1.45, yy, b, fs=10.5 if col != GRAY else 9.5, ha="left", color=col)
yy = gy - 1.6 - 4 * 0.55 - 0.2
text(ax, ox, yy, r"가중치 $e^{x-m}/\ell$", fs=10.5, ha="left")
for j, w in enumerate(W_ON):
    x = ox + j * cw + (0.25 if j >= 4 else 0)
    ax.add_patch(Rectangle((x, yy - 0.95), cw - 0.06, 0.55, fc="white", ec=GRAY, lw=0.8, zorder=3))
    text(ax, x + cw / 2 - 0.03, yy - 0.67, f"{w:.3f}", fs=9.5)
text(ax, ox, yy - 1.4, f"전체를 한 번에 softmax한 값과 최대 차이 {DIFF:.0e} → 근사가 아니라 같은 값",
     fs=10.5, color=RED, ha="left")
text(ax, ox, yy - 1.9, "(점수는 설명용 예시, m·ℓ·가중치는 그 값으로 계산)", fs=9.5, color=GRAY, ha="left")

# 출력
arrow(ax, 8.5, yF0, 8.5, yF0 - 0.6)
box(ax, 8.5, yF0 - 1.1, 11.0, 0.85, C_ATT)
text(ax, 8.5, yF0 - 1.1, r"헤드 출력 (B, 12, C, 64) → 이어 붙임 (B, C, 768) → $W_o$ → (B, C, 768)", fs=11)
text(ax, 0.6, yF0 - 2.4, "• 학습·프리필: is_causal=True.  디코드(Q 길이 1, K 길이 t): is_causal=False — True면 왼쪽 위 기준\n"
     "   마스크가 되어 새 쿼리가 첫 키만 본다(8.3절).  • 블록 6×6은 그림용, 실제 블록 크기는 파이토치가 정한다.",
     fs=10, color=INK, ha="left", linespacing=1.6)
save(fig, "fig08-flow-b.png", outdir, y_min=yF0 - 3.1)

print(f"N={N:,}  parts={PARTS}  N_MHA={N_MHA:,}  KV MHA/GQA={KV_BYTES(12) / 1e6:.2f}/{KV_BYTES(4) / 1e6:.2f}MB  FLOP={FLOP:.3e}")
print(f"m1={m1} l1={l1:.4f} m2={m2} l2={l2:.4f} w={np.round(W_ON, 3)} diff={DIFF:.1e}")
