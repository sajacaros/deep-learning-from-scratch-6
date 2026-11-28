# 3.10절 그림: 어휘 1,000 vs 2,000 학습 비교 (책 밖 보충)
# 입력: ch03/10_vocab_compare.py가 만든 codebot/vocab_compare.json
# 실행(저장소 루트에서): .venv/bin/python ch03/vocab_compare_fig.py
#   → docs/ch03/images/fig03-vocab-a.png (사전 학습 검증 손실), fig03-vocab-b.png (GRPO 정답률)
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/ch03/images"
R = json.load(open(ROOT / "codebot/vocab_compare.json"))

for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": ["NanumMyeongjo", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "font.size": 10,
})
INK, GRAY = "#222222", "#777777"
STYLE = {"1000": dict(color="#8A8A8A", ls="--", label="어휘 1,000 (책)"),
         "2000": dict(color="#2F5D8A", ls="-", label="어휘 2,000 (이 저장소)")}


def save(fig, out):
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) > 10000:
        im.save(out, optimize=True)
    else:
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out, Image.open(out).size)


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- (a) 검증 손실
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
for v in ("1000", "2000"):
    r = R[v]
    it, loss = zip(*r["pretrain_log"]["val_losses"])
    it, loss = np.array(it), np.array(loss)
    to_bpb = r["val"]["n_tokens"] / r["val"]["n_bytes"] / math.log(2)   # nat/토큰 → bit/바이트
    axes[0].plot(it, loss, lw=1.3, **STYLE[v])
    axes[1].plot(it, loss * to_bpb, lw=1.3, **STYLE[v])
    k = int(np.argmin(loss))
    axes[1].plot(it[k], loss[k] * to_bpb, "o", ms=4.5, color=STYLE[v]["color"])
    axes[1].annotate(f"최저 {loss[k] * to_bpb:.3f}\n({it[k]:,}회)", (it[k], loss[k] * to_bpb),
                     xytext=(0, -26 if v == "1000" else 12), textcoords="offset points",
                     ha="center", fontsize=8.5, color=STYLE[v]["color"])

axes[0].set_title("(1) 토큰당 손실 — 어휘가 다르면 직접 비교할 수 없다", fontsize=10.5, loc="left")
axes[0].set_ylabel("검증 손실 (nat / 토큰)")
axes[1].set_title("(2) 바이트당 비트(BPB) — 같은 글자 기준", fontsize=10.5, loc="left")
axes[1].set_ylabel("검증 BPB (bit / 바이트)")
for ax in axes:
    style_ax(ax)
    ax.set_xlabel("반복 횟수")
    ax.set_xlim(0, 20500)
    ax.set_ylim(None, None)
axes[0].set_ylim(1.4, 2.8)
axes[1].set_ylim(0.85, 1.35)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(w_pad=2.5)
save(fig, OUT / "fig03-vocab-a.png")

# ---------------------------------------------------------------- (b) GRPO 정답률
fig, ax = plt.subplots(figsize=(6.4, 3.1))
for v in ("1000", "2000"):
    g = R[v]["grpo_log"]
    steps = np.arange(len(g["accuracies"])) * g["eval_interval"]
    ax.plot(steps, g["accuracies"], lw=1.2, **STYLE[v])
style_ax(ax)
ax.set_xlabel("GRPO 반복 횟수")
ax.set_ylabel("81문제 정답률 (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
save(fig, OUT / "fig03-vocab-b.png")
