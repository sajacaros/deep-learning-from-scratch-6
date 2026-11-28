# 9장 노트의 그림 "WebBot 학습 한 스텝"을 그린다.
# (a) 옵티마이저 스텝 하나: DDP(GPU 2장 예) + 기울기 누적 4번 + no_sync + All-Reduce + clip + AdamW
# (b) GPU 한 장의 메모리: 실제 WebBot 파라미터 수(webbot/model.py, ch09/01_pretrain.py 설정)로 계산한
#     파라미터·기울기·옵티마이저 상태. AdamW vs Muon 하이브리드, ZeRO 단계별(GPU 4장).
# 하이퍼파라미터는 ch09/01_pretrain.py와 같다. 학습 데이터(owt_train.bin)는 필요 없다.
# 실행(저장소 루트에서): .venv/bin/python ch09/flow_graph.py → docs/ch09/images/fig09-flow-a.png, fig09-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from webbot.model import GPT

# ---------------------------------------------------------------- 데이터 (ch09/01_pretrain.py 설정)
vocab_size, context_len, embed_dim, n_head, n_kv_head, n_layer, ff_dim = 50000, 1024, 768, 12, 4, 12, 2048
micro_batch_size, accumulation_steps, max_iters, grad_clip = 32, 4, 100000, 1.0
model = GPT(vocab_size, context_len, embed_dim, n_head, n_kv_head, n_layer, ff_dim)
PSI = sum(p.numel() for p in model.parameters())                       # 152,316,672
PSI_2D = sum(p.numel() for p in model.parameters() if p.ndim == 2)     # Muon이 맡는 행렬
PSI_1D = PSI - PSI_2D                                                  # RMSNorm 가중치
GB = 1e9
TOK_MICRO = micro_batch_size * context_len                             # 32,768
TOK_STEP = TOK_MICRO * accumulation_steps                              # 131,072 (GPU 1장)

# ---------------------------------------------------------------- 스타일 (2장 flow_graph.py와 같은 색·폰트)
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_EMB, C_ATT, C_FFN, C_LN, C_OUT, C_BLK = "#F6D6D6", "#F9E3C2", "#CFE3F3", "#F3F0C4", "#E2DAF0", "#F4F4F4"


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc, ec=INK, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, cs="arc3", ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def txt(ax, x, y, s, fs=11, color=INK, ha="center", va="center", bold=False, **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight="bold" if bold else "normal",
            zorder=5, **kw)


def save(fig, name):
    out = OUTDIR / name
    fig.savefig(out, dpi=200, facecolor="white")
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) > 10000:
        im.save(out, optimize=True)
    else:
        im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out)


OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch09/images"

# =====================================================================
# (a) 옵티마이저 스텝 하나 — GPU 2장(랭크 0, 1), 기울기 누적 4번
# =====================================================================
W, H = 10.0, 9.4
fig, ax = new_fig(W, H)
txt(ax, 0.25, H - 0.35, "(a)  옵티마이저 스텝 하나 — DDP(GPU 2장 예) + 기울기 누적 4번", fs=14, ha="left", bold=True)

# 데이터
box(ax, W / 2, H - 1.0, 6.6, 0.62, "white")
txt(ax, W / 2, H - 0.9, "owt_train.bin  (np.memmap, uint16 토큰 ID)", fs=11.5)
txt(ax, W / 2, H - 1.15, "get_batch: 무작위 위치에서 길이 1,024를 32개 잘라 x, 한 칸 민 y", fs=10, color=GRAY)

LX0, CW, GAP = 1.55, 1.6, 0.12          # 마이크로 스텝 칸 위치
COLX = [LX0 + i * (CW + GAP) + CW / 2 for i in range(4)]
LANE_R = LX0 + 4 * (CW + GAP) + 0.42    # 랭크 틀의 오른쪽 끝 = All-Reduce 막대 바로 오른쪽
LANES = [(H - 2.55, 0, "랭크 0", "시드 42"), (H - 4.55, 1, "랭크 1", "시드 43")]
for ly, r, name, seed in LANES:
    ax.add_patch(FancyBboxPatch((0.2, ly - 0.95), LANE_R - 0.2, 1.75, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=C_BLK, ec=GRAY, lw=0.9, zorder=1))
    txt(ax, 0.85, ly + 0.25, name, fs=12, bold=True)
    txt(ax, 0.85, ly - 0.08, "GPU " + str(r), fs=10, color=GRAY)
    txt(ax, 0.85, ly - 0.38, seed, fs=10, color=GRAY)
    for k, cx in enumerate(COLX):
        last = k == accumulation_steps - 1
        box(ax, cx, ly + 0.12, CW, 1.15, POSL[k], ec=POS[k], lw=1.3)
        txt(ax, cx, ly + 0.5, f"마이크로 {k + 1}", fs=11, bold=True, color=POS[k])
        txt(ax, cx, ly + 0.22, "x (32, 1024)", fs=10)
        txt(ax, cx, ly - 0.04, "순전파 → loss/4", fs=10)
        txt(ax, cx, ly - 0.28, "→ backward", fs=10)
        txt(ax, cx, ly - 0.7, "All-Reduce 함께" if last else "no_sync: 통신 없음",
            fs=10, color=RED if last else GRAY, bold=last)
        if k:
            arrow(ax, cx - CW / 2 - GAP, ly + 0.15, cx - CW / 2, ly + 0.15, lw=1.0, ms=9)
    if r == 0:
        arrow(ax, COLX[1], H - 1.31, COLX[1], ly + 0.8, color=GRAY, lw=1.0)
    else:   # 랭크 1은 랭크 0 칸을 지나지 않게 왼쪽 여백으로 돌린다
        ax.plot([W / 2 - 3.3, 0.08, 0.08], [H - 1.0, H - 1.0, ly], color=GRAY, lw=1.0, zorder=2)
        arrow(ax, 0.08, ly, 0.2, ly, color=GRAY, lw=1.0, ms=9)

# 누적되는 .grad
gy = H - 5.95
txt(ax, 0.3, gy, "각 GPU의 .grad", fs=11, ha="left")
txt(ax, 1.75, gy, r"$=\ \frac{1}{4}\nabla L_1 + \frac{1}{4}\nabla L_2 + \frac{1}{4}\nabla L_3 + \frac{1}{4}\nabla L_4$"
    "    (backward는 .grad에 더한다)", fs=12, ha="left")
txt(ax, 1.75, gy - 0.42, "→ 마이크로배치 4개 = 유효 배치 128개의 평균 손실의 기울기 (메모리는 32개분 활성값만)",
    fs=10, color=GRAY, ha="left")

# All-Reduce 막대 (마지막 마이크로 스텝 열)
BX = COLX[3] + CW / 2 + 0.12
ax.add_patch(Rectangle((BX, H - 4.55 - 0.5), 0.34, 2.75, fc=RED, ec="none", alpha=0.85, zorder=3))
txt(ax, BX + 0.17, H - 3.55, "All-Reduce", fs=10.5, color="white", rotation=90, bold=True)
txt(ax, LANE_R + 0.1, H - 3.55, "두 GPU의\n.grad를\n평균해\n양쪽에\n나눠 줌", fs=10, color=RED, ha="left",
    linespacing=1.4)   # 랭크 틀 밖 오른쪽 여백

# 갱신
uy = H - 7.0
arrow(ax, 1.75, gy - 0.6, 1.75, uy + 0.33)   # 누적된 .grad → clip_grad_norm_ (첫 상자)
steps = [("clip_grad_norm_", f"전체 노름 ≤ {grad_clip}"), ("AdamW.step()", "lr = get_lr(step)  (D2Z)"),
         ("가중치 갱신", "두 GPU가 같은 값으로")]
for i, (t1, t2) in enumerate(steps):
    cx = 1.75 + i * 3.25
    box(ax, cx, uy, 2.9, 0.66, C_OUT if i == 1 else "white")
    txt(ax, cx, uy + 0.13, t1, fs=11.5)
    txt(ax, cx, uy - 0.15, t2, fs=10, color=GRAY)
    if i:
        arrow(ax, cx - 3.25 + 1.45, uy, cx - 1.45, uy)
txt(ax, W / 2, uy - 0.62, "→ 다음 스텝. 로그·평가·저장은 랭크 0만 (is_main)", fs=10.5, color=GRAY)

# 숫자
ny = 1.0
ax.plot([0.25, W - 0.25], [ny + 0.55, ny + 0.55], color="#DDDDDD", lw=1.0)
txt(ax, 0.3, ny + 0.2,
    f"• 마이크로배치 하나 = 32 × 1,024 = {TOK_MICRO:,} 토큰.  GPU 한 장의 스텝 하나 = × 4 = {TOK_STEP:,} 토큰 (GPU 2장이면 {2 * TOK_STEP:,})",
    fs=10.5, ha="left")
txt(ax, 0.3, ny - 0.22,
    f"• GPU 1장으로 {max_iters:,} 스텝 = 약 {TOK_STEP * max_iters / 1e9:.1f}B(131억) 토큰.  통신은 스텝마다 1번 (no_sync가 없으면 4번)",
    fs=10.5, ha="left")
txt(ax, 0.3, ny - 0.64, "• 랭크마다 시드가 달라(42 + RANK) 서로 다른 배치를 뽑는다. 저장소 코드는 GPU 1장이면 DDP 없이 같은 루프를 돈다.",
    fs=10.5, ha="left")
save(fig, "fig09-flow-a.png")

# =====================================================================
# (b) GPU 한 장의 메모리 — WebBot 파라미터 수로 계산
# =====================================================================
W, H = 10.0, 7.4
fig, ax = new_fig(W, H)
txt(ax, 0.25, H - 0.35, f"(b)  GPU 한 장의 메모리 — WebBot  Ψ = {PSI:,}개", fs=14, ha="left", bold=True)
txt(ax, 0.25, H - 0.75, "막대 = 파라미터 + 기울기 + 옵티마이저 상태 (GB = 10⁹바이트).  활성값(중간 출력)은 따로 — 기울기 누적·체크포인트가 줄이는 몫",
    fs=10, color=GRAY, ha="left")
C_P, C_G, C_O = "#4C72B0", "#DD8452", "#55A868"
X0, SCALE = 3.55, 2.25          # 1GB = 2.25인치
N_D = 4
rows = [
    ("저장소 설정: AdamW", "FP32 4+4+8 바이트", [4 * PSI, 4 * PSI, 8 * PSI]),
    ("Muon + AdamW 하이브리드", "상태 = 1차 모멘트만", [4 * PSI, 4 * PSI, 4 * PSI_2D + 8 * PSI_1D]),
    None,
    ("Baseline (데이터 병렬)", r"$(2+2+K)\Psi$", [2 * PSI, 2 * PSI, 12 * PSI]),
    ("ZeRO-1", "옵티마이저 상태 ÷ 4", [2 * PSI, 2 * PSI, 12 * PSI / N_D]),
    ("ZeRO-2", "+ 기울기 ÷ 4", [2 * PSI, 2 * PSI / N_D, 12 * PSI / N_D]),
    ("ZeRO-3", "+ 파라미터 ÷ 4", [2 * PSI / N_D, 2 * PSI / N_D, 12 * PSI / N_D]),
]
y = H - 1.45
for row in rows:
    if row is None:
        ax.plot([0.25, W - 0.25], [y + 0.1, y + 0.1], color="#DDDDDD", lw=1.0)
        txt(ax, 0.3, y - 0.2, f"ZeRO, GPU {N_D}장 — 책(그림 9-2)의 혼합 정밀도 가정: 파라미터·기울기 FP16 2바이트, K = 12",
            fs=10.5, ha="left", bold=True)
        y -= 0.75
        continue
    name, sub, parts = row
    txt(ax, 0.3, y + 0.12, name, fs=11, ha="left")
    txt(ax, 0.3, y - 0.18, sub, fs=10, color=GRAY, ha="left")
    x = X0
    for v, c in zip(parts, (C_P, C_G, C_O)):
        w = v / GB * SCALE
        ax.add_patch(Rectangle((x, y - 0.22), w, 0.44, fc=c, ec="white", lw=0.8, zorder=3))
        if w > 0.42:
            txt(ax, x + w / 2, y, f"{v / GB:.2f}", fs=10, color="white")
        x += w
    txt(ax, x + 0.1, y, f"{sum(parts) / GB:.2f} GB", fs=11, ha="left", bold=True)
    y -= 0.72
# 범례
ly = 0.55
for i, (c, t) in enumerate([(C_P, "파라미터"), (C_G, "기울기"), (C_O, "옵티마이저 상태")]):
    ax.add_patch(Rectangle((X0 + i * 1.75, ly - 0.12), 0.3, 0.24, fc=c, ec="none"))
    txt(ax, X0 + i * 1.75 + 0.4, ly, t, fs=10.5, ha="left")
txt(ax, 0.3, ly, "막대 안 숫자 = GB", fs=10, color=GRAY, ha="left")
save(fig, "fig09-flow-b.png")

print(f"PSI={PSI:,}  2D={PSI_2D:,}  1D={PSI_1D:,}")
for row in rows:
    if row:
        print(row[0], [round(v / GB, 3) for v in row[2]], round(sum(row[2]) / GB, 3))
