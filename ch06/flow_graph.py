# 6장 노트의 그림 "한눈에 보기"를 그린다.
#   (a) StoryBot 사전 학습 한 스텝(ch06/05_pretrain.py): 6.1~6.5의 기법이 루프 어디에 들어가는지
#   (b) DPO 한 스텝(ch06/08_dpo.py): 선호 쌍 하나가 손실이 되기까지
# 수치는 이 스크립트가 직접 계산한다.
#   - 학습률: 05_pretrain.py의 get_lr를 그대로 실행
#   - 파라미터 수·초기 손실·기울기 노름: 05_pretrain.py와 같은 설정의 StoryBot GPT를 무작위 초기화하고,
#     무작위 토큰 ID 배치(32, 256)로 실제 순전파·역전파를 한 번 실행한 값
#     (storybot/merge_rules.pkl, tiny_stories_train.bin은 저장소에 없어 실제 학습 데이터는 쓰지 못했다)
#   - DPO 선호 쌍: storybot/tiny_stories_dpo.json의 첫 번째 쌍(단어 단위로 표시)
#   - DPO의 로그 확률 네 값은 설명용 예시(학습된 모델이 저장소에 없음). 그 값으로 구한 logit·손실은 실제 계산
# 실행(저장소 루트에서): .venv/bin/python ch06/flow_graph.py → docs/ch06/images/fig06-flow-a.png, fig06-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.transforms import Bbox
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from storybot.model import GPT

# ---------------------------------------------------------------- 데이터 (05_pretrain.py 설정)
context_len, vocab_size, batch_size = 256, 10000, 32
max_lr, warmup_iters, max_iters = 1e-3, 200, 40000
embed_dim, n_head, n_layer, ff_dim, theta = 512, 16, 4, 1344, 10000
grad_clip = 1.0


def get_lr(it, max_lr, warmup_iters, max_iters):     # 05_pretrain.py와 같은 함수
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)
    return 0.0


torch.manual_seed(0)
model = GPT(vocab_size, context_len, embed_dim, n_head, n_layer, ff_dim, theta)
N_PARAMS = sum(p.numel() for p in model.parameters())
ids = torch.randint(vocab_size, (batch_size, context_len + 1))
bx, by = ids[:, :-1], ids[:, 1:]
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    logits = model(bx)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.reshape(-1))
LOGITS_DTYPE, LOSS_DTYPE, LOSS0 = str(logits.dtype).replace("torch.", ""), str(loss.dtype).replace("torch.", ""), loss.item()
loss.backward()
GRAD_DTYPE = str(next(model.parameters()).grad.dtype).replace("torch.", "")
GNORM = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
MB = N_PARAMS * 4 / 1e6

dpo = json.load(open(ROOT / "storybot/tiny_stories_dpo.json"))
PAIR = dpo[0]
N_PAIRS, DPO_BATCH, DPO_ITERS, BETA = len(dpo), 8, 1000, 0.1
# 설명용 예시 로그 확률 (학습이 어느 정도 진행된 뒤를 가정)
LP = {"pc": -112.0, "pr": -135.0, "rc": -120.0, "rr": -115.0}
DPO_LOGIT = BETA * ((LP["pc"] - LP["pr"]) - (LP["rc"] - LP["rr"]))
DPO_LOSS = -F.logsigmoid(torch.tensor(DPO_LOGIT)).item()
LOSS_AT0 = -F.logsigmoid(torch.tensor(0.0)).item()

# ---------------------------------------------------------------- 스타일 (ch02/flow_graph.py와 같은 색·폰트)
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_EMB, C_ATT, C_FFN, C_LN, C_OUT, C_BLK = "#F6D6D6", "#F9E3C2", "#CFE3F3", "#F3F0C4", "#E2DAF0", "#F4F4F4"
C_BF16 = "#FBEFD9"

ax = None


def new_fig(w_in, h_in):
    global ax
    fig = plt.figure(figsize=(w_in, h_in), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in * 2); ax.set_ylim(0, h_in * 2); ax.axis("off")   # 1단위 = 0.5인치
    return fig


def box(x, y, w, h, text, fc, fs=11, bold=False, ec=INK, lw=1.1, color=INK, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=z + 1)


def two_line(x, y, w, h, t1, t2, fc, fs1=11, fs2=9, ec=INK, lw=1.1, bold=False):
    box(x, y, w, h, "", fc, ec=ec, lw=lw)
    ax.text(x, y + h * 0.18, t1, fontsize=fs1, ha="center", va="center", zorder=5, fontweight="bold" if bold else "normal")
    ax.text(x, y - h * 0.22, t2, fontsize=fs2, color=GRAY, ha="center", va="center", zorder=5)


def arrow(x1, y1, x2, y2, color=INK, lw=1.2, cs="arc3", ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def note(x, y, s, fs=9, color=GRAY, ha="left", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center", zorder=5, linespacing=1.4, **kw)


def step_label(x, y, s):
    ax.text(x, y, s, fontsize=12, fontweight="bold", color=RED, ha="center", va="center", zorder=6)


def save(fig, name, crop=0.0):
    out = ROOT / "docs/ch06/images" / name
    w_, h_ = fig.get_size_inches()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches=Bbox([[0, crop], [w_, h_]]))
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) > 10000:     # 그라데이션이 많으면 무손실만
        im.save(out, optimize=True)
    else:
        im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out)


# =====================================================================
# (a) 사전 학습 한 스텝
# =====================================================================
fig = new_fig(8.6, 15.2)
W = 17.2
ax.text(0.4, 29.9, "(a)  StoryBot 사전 학습 — 스텝 i 하나에서 일어나는 일", fontsize=14.5, fontweight="bold",
        color=INK, va="center")
X, BW = 6.3, 7.6          # 가운데 기둥
RX = 11.0                 # 오른쪽 설명 열

# 학습 전 준비
ax.add_patch(FancyBboxPatch((0.3, 27.35), 16.6, 1.75, boxstyle="round,pad=0,rounding_size=0.2",
                            fc="none", ec=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(16.8, 28.85, "학습 전에 한 번만 (4장)", fontsize=9, color=GRAY, va="center", ha="right")
for cx, t1, t2, fc in [(2.7, "tiny_stories_train.txt", "TinyStories 원문", "white"),
                       (8.4, "BPE 토크나이저", "어휘 10,000개", C_EMB),
                       (14.1, "tiny_stories_train.bin", "토큰 ID, uint16 (2바이트)", "white")]:
    two_line(cx, 28.0, 4.6, 0.95, t1, t2, fc, fs1=10.5, fs2=8.5)
arrow(5.0, 28.0, 6.1, 28.0)
arrow(10.7, 28.0, 11.8, 28.0)
arrow(14.1, 27.52, 14.1, 26.75)
two_line(14.1, 26.25, 5.0, 0.95, "np.memmap (6.5.1)", "파일은 디스크에, 읽은 부분만 메모리로", "white", fs1=10.5, fs2=8.5)

# 루프 테두리
LT, LB = 26.95, 6.1
ax.add_patch(FancyBboxPatch((1.6, LB), 9.4, LT - LB, boxstyle="round,pad=0,rounding_size=0.3",
                            fc="none", ec=GRAY, lw=1.1, zorder=0))
ax.text(1.85, LT - 0.35, f"for i in range({max_iters}):", fontsize=10.5, color=GRAY, va="center", family="DejaVu Sans Mono")

# ① 학습률
y = 25.6
step_label(2.05, y - 0.2, "①")
two_line(X, y, BW, 1.05, "lr = get_lr(i)  →  param_group['lr']", "워밍업 200스텝 동안 0→1e-3, 이후 0까지 직선 (6.3)", "white", fs1=10.5, fs2=9)
arrow(X, y - 0.53, X, y - 0.95)

# lr 미니 그래프 (오른쪽)
lax = fig.add_axes([RX / W + 0.04, 20.4 / 30.4, 0.27, 2.75 / 30.4])
it = np.arange(0, max_iters + 1, 50)
lax.plot(it, [get_lr(i, max_lr, warmup_iters, max_iters) * 1e3 for i in it], color=POS[0], lw=1.6)
lax.axvspan(0, warmup_iters, color="#DDDDDD")
marks = [100, 200, 20000, 39999]
for m_ in marks:
    v = get_lr(m_, max_lr, warmup_iters, max_iters)
    lax.plot([m_], [v * 1e3], "o", color=RED, ms=4)
lax.set_xlim(-800, max_iters + 800); lax.set_ylim(0, 1.12)
lax.set_xticks([0, 20000, 40000]); lax.set_xticklabels(["0", "20,000", "40,000"], fontsize=8.5)
lax.set_yticks([0, 0.5, 1.0]); lax.set_yticklabels(["0", "5e-4", "1e-3"], fontsize=8.5)
lax.set_title("D2Z 학습률 (실제 get_lr 값)", fontsize=9.5, pad=3)
for s in ("top", "right"):
    lax.spines[s].set_visible(False)
tbl = [(m_, get_lr(m_, max_lr, warmup_iters, max_iters)) for m_ in marks]
note(RX + 0.35, 18.95, "\n".join(f"i = {m_:>6,}  →  lr = {v:.3g}" for m_, v in tbl), fs=9, color=INK,
     family="DejaVu Sans Mono")

# ② 배치
y = 24.05
step_label(2.25, y, "②")
two_line(X, y, BW, 1.05, "get_batch — 무작위 시작 위치 32개",
         f"x: ({batch_size}, {context_len})   y: 한 칸 민 것   uint16 → int64", "white", fs1=10.5, fs2=9)
ax.plot([14.1, 14.1], [25.77, y], color=GRAY, lw=1.0, ls=(0, (3, 2)), zorder=1)
arrow(14.1, y, X + BW / 2, y, color=GRAY, lw=1.0)
note(14.25, 24.6, "읽기", fs=9)
arrow(X, y - 0.53, X, y - 0.95)

# ③ zero_grad
y = 22.75
step_label(2.25, y, "③")
box(X, y, BW, 0.75, "optimizer.zero_grad()  — 지난 스텝의 기울기를 지움", "white", fs=10.5)
arrow(X, y - 0.38, X, y - 0.8)

# ④ autocast 블록
AT, AB = 21.95, 13.05
ax.add_patch(FancyBboxPatch((X - BW / 2 - 0.1, AB), BW + 0.2, AT - AB, boxstyle="round,pad=0,rounding_size=0.25",
                            fc=C_BF16, ec="#D9A441", lw=1.4, zorder=1))
step_label(2.25, AT - 0.4, "④")
ax.text(X - BW / 2 + 0.15, AT - 0.38, "with autocast(dtype=bfloat16):  (6.4)", fontsize=10, color="#9A6A12",
        va="center", family="DejaVu Sans Mono")
yy = AT - 1.15
two_line(X, yy, 5.2, 0.85, f"토큰 임베딩 ({vocab_size:,}, {embed_dim})", f"x ({batch_size}, {context_len}) → ({batch_size}, {context_len}, {embed_dim})",
         C_EMB, fs1=10.5, fs2=8.5)
arrow(X, yy - 0.43, X, yy - 0.8)
# 블록
bt, bb = yy - 0.8, yy - 4.75
ax.add_patch(FancyBboxPatch((X - 3.0, bb), 6.0, bt - bb, boxstyle="round,pad=0,rounding_size=0.2",
                            fc=C_BLK, ec=INK, lw=1.1, zorder=2))
ax.text(X + 3.15, (bt + bb) / 2, f"× {n_layer}", fontsize=15, fontweight="bold", va="center", color=INK)
box(X, bt - 0.5, 3.0, 0.55, "RMSNorm", C_LN, fs=10)
two_line(X, bt - 1.45, 5.0, 0.95, "어텐션  16헤드 × 32차원 + RoPE", "Q·K·V·O 행렬곱 → BF16", C_ATT, fs1=10, fs2=8.5)
box(X, bt - 2.45, 3.0, 0.55, "RMSNorm", C_LN, fs=10)
two_line(X, bt - 3.4, 5.0, 0.95, f"SwiGLU  {embed_dim} → {ff_dim:,} → {embed_dim}", "W, V, O 행렬곱 → BF16", C_FFN, fs1=10, fs2=8.5)
for a_, b_ in [(bt - 0.78, bt - 0.97), (bt - 1.93, bt - 2.17), (bt - 2.73, bt - 2.92)]:
    arrow(X, a_, X, b_, lw=1.0, ms=9)
arrow(X, bb, X, bb - 0.42)
yy = bb - 0.85
two_line(X, yy, 5.2, 0.85, f"RMSNorm → unembed {embed_dim} → {vocab_size:,}", f"logits ({batch_size}, {context_len}, {vocab_size:,})  dtype = {LOGITS_DTYPE}",
         C_OUT, fs1=10.5, fs2=8.5)
arrow(X, yy - 0.43, X, yy - 0.8)
yy -= 1.35
box(X, yy, 6.6, 0.95, "", "white")
ax.text(X, yy + 0.2, "F.cross_entropy  (32×256 = 8,192개 위치 평균)", fontsize=10.5, ha="center", va="center", zorder=5)
ax.text(X, yy - 0.22, f"loss = {LOSS0:.2f}  dtype = {LOSS_DTYPE}", fontsize=10, color=RED, ha="center", va="center", zorder=5)

note(RX + 0.35, 16.9, "행렬곱처럼 빨라야 하는 연산은 BF16,\n"
     "softmax·합계·손실처럼 오차가 쌓이는\n연산은 FP32 — autocast가 연산마다\n자동으로 고른다.", fs=9.5, color=INK)
note(RX + 0.35, 14.5, f"초기화 직후 loss {LOSS0:.2f}  ≈  ln 10,000 = {math.log(vocab_size):.2f}\n"
     "(어휘 1만 개에 고르게 찍는 수준.\n그림 6-3의 출발점 약 9.3과 같다)", fs=9.5, color=INK)
ax.plot([X + 3.3, RX + 0.2], [yy, 14.5], color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=1)

# ⑤ backward
arrow(X, AB, X, 12.55)
y = 12.15
step_label(2.25, y, "⑤")
two_line(X, y, BW, 0.85, "loss.backward()  — autocast 밖", f"기울기 dtype = {GRAD_DTYPE} (파라미터와 같은 FP32)", "white", fs1=10.5, fs2=9)
arrow(X, y - 0.43, X, y - 0.85)

# ⑥ clip
y = 10.85
step_label(2.25, y, "⑥")
two_line(X, y, BW, 1.05, "clip_grad_norm_(params, 1.0)  (6.5.2)",
         r"$\Vert g\Vert > 1$이면 $g \leftarrow g/\Vert g\Vert$ : 방향 유지, 크기만 1로", "white", fs1=10.5, fs2=9.5)
note(RX + 0.35, y, f"초기화 직후 실제 노름 {GNORM:.3f}\n≤ 1.0 이라 이 스텝에서는 그대로.\n"
     "반환값은 자르기 전 노름이다.", fs=9.5, color=INK)
arrow(X, y - 0.53, X, y - 0.95)

# ⑦ AdamW
y = 9.2
step_label(2.25, y, "⑦")
box(X, y, BW, 1.3, "", C_ATT)
ax.text(X, y + 0.33, "optimizer.step()  — AdamW (6.1~6.2)", fontsize=10.5, ha="center", va="center", zorder=5)
ax.text(X, y - 0.22, r"$m, v$ 갱신 → $\theta \leftarrow \theta - lr\cdot\hat m/(\sqrt{\hat v}+\epsilon) - lr\cdot\lambda\theta$",
        fontsize=10, ha="center", va="center", zorder=5)
note(RX + 0.35, y, "lr은 ①에서 정한 값.\n분리된 감쇠 " r"$\lambda$" " = 0.01 (기본값)", fs=9.5, color=INK)
arrow(X, y - 0.65, X, y - 1.05)

# ⑧ 평가·저장
y = 7.55
step_label(2.25, y, "⑧")
two_line(X, y, BW, 1.05, "500스텝마다 evaluate() → 검증 손실 (6.5.3)",
         "no_grad · 'sum' ÷ 토큰 수 · i = 500, 5,000에서 저장", "white", fs1=10.5, fs2=8.5)
# 루프 되돌림
ax.plot([X, X, 1.15, 1.15], [y - 0.53, 6.55, 6.55, 25.85], color=INK, lw=1.1, zorder=2)
arrow(1.15, 25.85, X - BW / 2, 25.85)
ax.text(0.85, 16.0, "i += 1  (총 40,000번, GPU에서 약 40분)", rotation=90, fontsize=9.5, color=GRAY, ha="center", va="center")

# 메모리 표
ax.text(0.4, 5.5, f"파라미터 {N_PARAMS:,}개(약 {N_PARAMS / 1e6:.1f}M, 실제 집계)가 FP32로 차지하는 메모리", fontsize=10.5,
        color=INK, va="center")
rows = [("파라미터 θ", f"{MB:.1f} MB", "FP32 그대로 유지 (autocast는 연산만 BF16)"),
        ("기울기 g", f"{MB:.1f} MB", "⑤에서 채워지고 ③에서 지워짐"),
        ("AdamW 상태 m, v", f"{2 * MB:.1f} MB", "파라미터마다 2개 — 스텝을 넘어 유지 (self.state)")]
for k, (a_, b_, c_) in enumerate(rows):
    yy = 4.65 - k * 0.72
    ax.add_patch(Rectangle((0.4, yy - 0.33), 16.4, 0.66, fc="#F7F7F7" if k % 2 == 0 else "white", ec="none", zorder=1))
    ax.text(0.6, yy, a_, fontsize=10, va="center", color=INK)
    ax.text(5.6, yy, b_, fontsize=10, va="center", ha="right", color=RED)
    ax.text(6.1, yy, c_, fontsize=9.5, va="center", color=GRAY)
ax.text(0.4, 2.0, "수치: 05_pretrain.py와 같은 설정의 GPT를 무작위 초기화하고 무작위 토큰 배치 (32, 256)로 한 번 실행한 값.\n"
        "실제 학습 데이터(tiny_stories_train.bin)는 저장소에 없어 쓰지 않았다. 활성값 메모리는 표에서 뺐다.",
        fontsize=8.5, color=GRAY, va="center", linespacing=1.5)
save(fig, "fig06-flow-a.png", crop=0.45)

# =====================================================================
# (b) DPO 한 스텝
# =====================================================================
fig = new_fig(8.6, 13.4)
ax.text(0.4, 26.1, "(b)  DPO — 선호 쌍 하나가 손실이 되기까지", fontsize=14.5, fontweight="bold", color=INK, va="center")


def wrap(s, n):
    out, line = [], ""
    for w_ in s.split():
        if len(line) + len(w_) + 1 > n:
            out.append(line); line = w_
        else:
            line = (line + " " + w_).strip()
    out.append(line)
    return out


# 선호 쌍
ax.text(0.4, 25.2, f"storybot/tiny_stories_dpo.json — {N_PAIRS}쌍 중 첫 번째", fontsize=10.5, color=INK, va="center")
items = [("prompt  x", PAIR["prompt"], "#EEEEEE", GRAY),
         ("chosen  y_w", PAIR["chosen"], POSL[2], POS[2]),
         ("rejected  y_l", PAIR["rejected"], POSL[3], POS[3])]
yy = 24.55
for lab, txt, fc, ec in items:
    lines = wrap(txt, 118)
    if len(lines) > 2:
        lines = lines[:2]; lines[1] = lines[1] + " …"
    h = 0.62 + 0.42 * len(lines)
    ax.add_patch(FancyBboxPatch((0.4, yy - h), 16.4, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(0.6, yy - 0.3, lab, fontsize=9.5, fontweight="bold", color=ec if ec != GRAY else INK, va="center", zorder=3)
    for k, ln in enumerate(lines):
        ax.text(0.6, yy - 0.72 - k * 0.42, ln, fontsize=9, color=INK, va="center", zorder=3)
    yy -= h + 0.2
arrow(8.6, yy + 0.1, 8.6, yy - 0.35)

# DPODataset: ids와 mask
Y0 = yy - 0.75
ax.text(0.4, Y0, f"DPODataset — 프롬프트 + 응답을 잇고 길이 {context_len}까지 0으로 패딩, 마스크는 응답만 1", fontsize=10.5, color=INK, va="center")
pw = len(PAIR["prompt"].split())
segs_w = {"c": len(PAIR["chosen"].split()), "r": len(PAIR["rejected"].split())}
SX, SW = 2.6, 14.0
scale = SW / 120.0          # 그림 폭 = 단어 120개 분량 (나머지 패딩은 생략 표시)
for k, (key, lab, fc, ec) in enumerate([("c", "chosen", POSL[2], POS[2]), ("r", "rejected", POSL[3], POS[3])]):
    ry = Y0 - 0.85 - k * 1.55
    ax.text(SX - 0.2, ry, f"{lab} ids", fontsize=9.5, ha="right", va="center", color=ec)
    ax.text(SX - 0.2, ry - 0.58, "mask", fontsize=9.5, ha="right", va="center", color=GRAY)
    x0 = SX
    for w_, fc_, t_, mval in [(pw, "#E6E6E6", f"x {pw}단어", 0), (segs_w[key], fc, f"응답 {segs_w[key]}단어", 1),
                              (120 - pw - segs_w[key], "white", "패딩 0 … (256까지)", 0)]:
        ww = w_ * scale
        ax.add_patch(Rectangle((x0, ry - 0.25), ww, 0.5, fc=fc_, ec=ec if mval else "#AAAAAA", lw=0.9, zorder=3,
                               ls="-" if fc_ != "white" else (0, (3, 2))))
        ax.text(x0 + ww / 2, ry, t_, fontsize=9, ha="center", va="center", zorder=4, color=INK)
        ax.add_patch(Rectangle((x0, ry - 0.8), ww, 0.42, fc=POSL[2] if mval else "#F4F4F4", ec="#AAAAAA", lw=0.7, zorder=3))
        ax.text(x0 + ww / 2, ry - 0.59, "1 1 1 … 1" if mval else "0 0 … 0", fontsize=9, ha="center", va="center", zorder=4,
                color=INK if mval else GRAY)
        x0 += ww
yb = Y0 - 0.85 - 1.55 - 1.15
ax.text(0.4, yb, "프롬프트는 두 줄이 같은 토큰. 칸 길이는 단어 수 비율(실제 단위는 BPE 토큰). 둘 다 256보다 훨씬 짧아 대부분이 패딩이다.",
        fontsize=9, color=GRAY, va="center")
arrow(8.6, yb - 0.3, 8.6, yb - 0.75)

# 순전파 4번
G = yb - 1.25
ax.text(0.4, G, r"get_sequence_logprobs — 응답 토큰의 로그 확률 합  $\log\pi(y\mid x) = \sum_n \log\pi(y_n\mid x, y_{<n})$",
        fontsize=10.5, color=INK, va="center")
ax.text(0.4, G - 0.55, "logits[:, :-1]과 ids[:, 1:]로 한 칸 시프트 → gather로 정답 토큰 값만 → mask[:, 1:]를 곱해 응답만 남기고 합",
        fontsize=9, color=GRAY, va="center")
gx0, gy0 = 4.4, G - 1.35
cw, ch = 5.4, 1.15
ax.text(gx0 + cw / 2, gy0 + 0.05, "chosen  y_w", fontsize=10, ha="center", va="bottom", color=POS[2], fontweight="bold")
ax.text(gx0 + cw * 1.5 + 0.3, gy0 + 0.05, "rejected  y_l", fontsize=10, ha="center", va="bottom", color=POS[3], fontweight="bold")
cells = [[("pc", r"$\log\pi_\theta(y_w)$"), ("pr", r"$\log\pi_\theta(y_l)$")],
         [("rc", r"$\log\pi_{\mathrm{ref}}(y_w)$"), ("rr", r"$\log\pi_{\mathrm{ref}}(y_l)$")]]
rlabels = [("model  " r"$\pi_\theta$", "학습 대상, 기울기 O"), ("ref_model  " r"$\pi_{\mathrm{ref}}$", "고정, torch.no_grad()")]
for r in range(2):
    cy = gy0 - 0.2 - r * (ch + 0.2) - ch / 2
    ax.text(gx0 - 0.25, cy + 0.18, rlabels[r][0], fontsize=10, ha="right", va="center", color=INK)
    ax.text(gx0 - 0.25, cy - 0.25, rlabels[r][1], fontsize=8.5, ha="right", va="center", color=RED if r == 0 else GRAY)
    for c in range(2):
        key, lab = cells[r][c]
        cx = gx0 + c * (cw + 0.3) + cw / 2
        box(cx, cy, cw, ch, "", "white" if r else C_ATT, ec=POS[2] if c == 0 else POS[3], lw=1.2)
        ax.text(cx, cy + 0.22, lab, fontsize=10.5, ha="center", va="center", zorder=5)
        ax.text(cx, cy - 0.25, f"{LP[key]:.0f}", fontsize=10.5, ha="center", va="center", zorder=5, color=RED)
note(gx0 + 2 * cw + 0.45, gy0 - 0.2 - ch - 0.1, "순전파\n4번", fs=9.5, color=INK)
gy_end = gy0 - 0.2 - 2 * ch - 0.2
ax.text(0.4, gy_end - 0.3, "빨간 수는 설명용 예시(학습이 진행된 뒤를 가정). 첫 스텝에는 model = ref_model이라 위아래 값이 같다.",
        fontsize=9, color=GRAY, va="center")
arrow(8.6, gy_end - 0.6, 8.6, gy_end - 1.0)

# 결합
L = gy_end - 1.55
box(8.6, L, 16.2, 1.0, "", "white")
ax.text(8.6, L + 0.2, r"logits $= \beta\,[\,(\log\pi_\theta(y_w) - \log\pi_\theta(y_l)) - (\log\pi_{\mathrm{ref}}(y_w) - \log\pi_{\mathrm{ref}}(y_l))\,]$",
        fontsize=10.5, ha="center", va="center", zorder=5)
ax.text(8.6, L - 0.25,
        f"= {BETA} × [ ({LP['pc']:.0f} − ({LP['pr']:.0f})) − ({LP['rc']:.0f} − ({LP['rr']:.0f})) ] = {BETA} × [ {LP['pc'] - LP['pr']:.0f} − ({LP['rc'] - LP['rr']:.0f}) ] = {DPO_LOGIT:.1f}",
        fontsize=10, ha="center", va="center", zorder=5, color=RED)
arrow(8.6, L - 0.5, 8.6, L - 0.9)
L2 = L - 1.4
box(5.2, L2, 7.6, 0.95, "", "white")
ax.text(5.2, L2 + 0.18, r"loss $= -\log\sigma(\mathrm{logits})$, 배치 8쌍 평균", fontsize=10.5, ha="center", va="center", zorder=5)
ax.text(5.2, L2 - 0.24, f"예시: −log σ({DPO_LOGIT:.1f}) = {DPO_LOSS:.3f}", fontsize=10, ha="center", va="center", color=RED, zorder=5)
box(13.1, L2, 7.0, 0.95, "", "#F7F7F7", ec=GRAY)
ax.text(13.1, L2 + 0.18, "첫 스텝: model = ref_model", fontsize=10.5, ha="center", va="center", zorder=5)
ax.text(13.1, L2 - 0.24, f"logits = 0 → −log σ(0) = {LOSS_AT0:.3f} (그림 6-6 출발점)", fontsize=9.5, ha="center", va="center", zorder=5, color=GRAY)
arrow(5.2, L2 - 0.48, 5.2, L2 - 0.9, color=RED)
ax.text(5.2, L2 - 1.2, "역전파 → model만 AdamW로 갱신 (lr 5e-6)", fontsize=10.5, ha="center", va="center", color=RED)
ax.text(0.4, L2 - 2.1,
        f"• y_w의 로그 확률은 올리고 y_l의 것은 내리는 방향으로 기울기가 흐른다. 기준은 ref_model 대비 변화량이다.\n"
        f"• {N_PAIRS}쌍 ÷ 배치 {DPO_BATCH} = {N_PAIRS // DPO_BATCH}스텝이 1에폭. {DPO_ITERS:,}스텝이면 같은 쌍을 약 {DPO_ITERS / (N_PAIRS / DPO_BATCH):.0f}번 본다.\n"
        f"• β = {BETA}: 괄호 안 차이가 10 커져야 logits가 1 커진다. β가 크면 같은 손실에 필요한 변화가 작아 ref 가까이 머문다.",
        fontsize=9.5, color=INK, va="center", linespacing=1.7)
save(fig, "fig06-flow-b.png", crop=2.1)
print("params", N_PARAMS, "loss0", LOSS0, "gnorm", GNORM, "dpo", DPO_LOGIT, DPO_LOSS)
