# 3장 노트의 그림 "세 단계 학습을 한눈에"를 그린다.
# (a) 사전 학습·SFT·GRPO가 같은 "실제 다음 토큰의 확률"에 서로 다른 가중치를 주는 모습
# (b) GRPO 반복 한 번: 질문 하나(7+8=)에 대한 그룹 8개, 보상·어드밴티지, 배치, 클리핑
# 토큰은 codebot/merge_rules_2000.pkl 토크나이저(어휘 2,000)로 실제로 인코딩한 것이다. 책의 1,000으로 그리려면 VOCAB=1000.
# 사전 학습 입력은 codebot/tiny_codes_2000.bin의 실제 토큰(2장 흐름도와 같은 위치)이고,
# SFT 샘플은 codebot/tiny_codes_sft.json에 있는 "3+5=" → "8"이다.
# GRPO의 응답 8개는 설명용 예시다(보상·어드밴티지·패딩은 ch03/09_grpo.py와 같은 방식으로 계산).
# 실행(저장소 루트에서): .venv/bin/python ch03/flow_graph.py → docs/ch03/images/fig03-flow-a.png, fig03-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import json
import re
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
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_vocab_size, codebot_paths

PATHS = codebot_paths(get_vocab_size())
tok = BPETokenizer.load_from(str(ROOT / PATHS["tokenizer"]))
EOT = tok.end_token_id


def show(i):
    """토큰 하나를 보이는 글자로. 앞 공백은 ␣, 줄바꿈은 ↵."""
    if i == EOT:
        return "<eot>"
    return tok.decode([int(i)]).replace(" ", "␣").replace("\n", "↵")


# ---------------------------------------------------------------- 데이터
# ① 사전 학습: tiny_codes_2000.bin에서 길이 8을 잘라 x, 한 칸 민 y (TokenDataset과 같은 방식)
ids_all = np.fromfile(ROOT / PATHS["data"], dtype=np.uint16)
S = {1000: 103, 2000: 88}[get_vocab_size()]   # 2장 흐름도와 같은 원문 위치("␣item ␣= ␣input ...")
PC = 8
PT_X = ids_all[S:S + PC].tolist()
PT_Y = ids_all[S + 1:S + PC + 1].tolist()

# ② SFT: SFTDataset._create_sample과 같은 방식
sft = json.load(open(ROOT / "codebot/tiny_codes_sft.json"))
item = next(d for d in sft if d["instruction"] == "3+5=")
p_ids = tok.encode(f"### Instruction:\n{item['instruction']}\n\n### Response:\n")
r_ids = tok.encode(f"{item['response']}<|endoftext|>")
SFT_IDS = (p_ids + r_ids)[:-1]
SFT_LAB = ([-100] * len(p_ids) + r_ids)[1:]
SFT_PAD = 256 - len(SFT_IDS)
assert len(p_ids) == 22 and SFT_LAB.count(-100) == 21 and SFT_PAD == 233

# ③ GRPO: 질문 7+8=, 응답 8개(설명용 예시). 보상·어드밴티지·패딩은 09_grpo.py와 같은 계산
PROMPT = "### Instruction:\n7+8=\n\n### Response:\n"
GT = 15
RESP = ["15", "16", "15", "7+8=15", "15", "13", "15", "15"]


def reward(gt, r):
    m = re.findall(r"(-?\d+)", r)
    return 1.0 if m and int(m[-1]) == gt else 0.0


R = np.array([reward(GT, r) for r in RESP])
A = R - R.mean()
q_ids = tok.encode(PROMPT)
RESP_IDS = [tok.encode(r) for r in RESP]
MAX_LEN = max(len(q_ids) + len(r) for r in RESP_IDS)
assert len(q_ids) == 22 and MAX_LEN == 28 and R.sum() == 6

# ---------------------------------------------------------------- 스타일 (2장 흐름도와 같음)
for p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
BLUE, BLUEL = "#4C72B0", "#C9D6EA"
GREEN, GREENL = "#55A868", "#CDE6D3"
REDL = "#EECACB"
ORANGE, ORANGEL = "#DD8452", "#F3D6C2"
PURPLE, PURPLEL = "#8172B3", "#DAD4EC"
MUTE, MUTEE = "#E8E8E8", "#BBBBBB"
C_BLK = "#F4F4F4"


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def cell(ax, x, y, w, text, fc, ec, fs=11, h=0.42, color=INK, sub=None):
    """왼쪽 위가 아니라 왼쪽 가운데가 (x, y)."""
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=fc, ec=ec, lw=0.9, zorder=3))
    ty = y + (0.06 if sub else 0)
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs, color=color, zorder=4)
    if sub:
        ax.text(x + w / 2, y - 0.13, sub, ha="center", va="center", fontsize=8, color=GRAY, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, ms=12, cs="arc3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def panel(ax, x, y, w, h, fc=C_BLK, ec="#CCCCCC"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fc, ec=ec, lw=1.0, zorder=1))


def save(fig, out):
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) > 10000:
        im.save(out, optimize=True)
    else:
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out)


# =====================================================================
# (a) 세 단계의 손실 비교
# =====================================================================
W, H = 10.0, 11.9
fig, ax = new_fig(W, H)
ax.text(0.3, H - 0.4, "(a)  세 단계 모두 '실제 다음 토큰의 확률'을 쓴다 — 다른 것은 가중치뿐",
        fontsize=14, fontweight="bold", color=INK, va="center")

# 모델 사슬
N_PARAMS = {1000: "11.1M", 2000: "11.5M"}[get_vocab_size()]
GRPO_ACC = {1000: "덧셈 80%+", 2000: "덧셈 96%"}[get_vocab_size()]   # 2000은 3.10절 실험 결과(78/81)
chain = [("무작위 GPT", f"{N_PARAMS} 파라미터", "white"),
         (Path(PATHS["pretrain"]).name, "이어 쓰기", BLUEL),
         (Path(PATHS["sft"]).name, "대화 가능", GREENL),
         (Path(PATHS["grpo"]).name, GRPO_ACC, ORANGEL)]
steps = ["① 사전 학습\n20,000회 · lr 3e-4", "② SFT\n500회 · lr 3e-4", "③ GRPO\n500회 · lr 7e-6"]
cy, bw, gap = H - 1.6, 1.6, 0.95
for k, (t1, t2, fc) in enumerate(chain):
    x = 0.3 + k * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, cy - 0.33), bw, 0.66, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec=INK, lw=1.0, zorder=3))
    ax.text(x + bw / 2, cy + 0.1, t1, fontsize=10.5, ha="center", va="center", zorder=4)
    ax.text(x + bw / 2, cy - 0.17, t2, fontsize=9.5, color=GRAY, ha="center", va="center", zorder=4)
    if k < 3:
        arrow(ax, x + bw + 0.05, cy, x + bw + gap - 0.05, cy)
        ax.text(x + bw + gap / 2, cy + 0.42, steps[k], fontsize=9.5, ha="center", va="bottom",
                color=INK, linespacing=1.3)

LX, CX0 = 0.3, 1.6      # 행 이름의 x, 칸 시작 x
ROWH = 0.56


def row_label(y, s, sub=None):
    ax.text(LX, y + (0.08 if sub else 0), s, fontsize=10.5, va="center", color=INK)
    if sub:
        ax.text(LX, y - 0.17, sub, fontsize=9, va="center", color=GRAY)


def section_head(y, num, title, code, fc, ec):
    ax.add_patch(Rectangle((0.3, y - 0.17), 0.12, 0.34, fc=ec, ec="none", zorder=3))
    ax.text(0.55, y, f"{num}  {title}", fontsize=12.5, fontweight="bold", va="center", color=INK)
    ax.text(W - 0.3, y, code, fontsize=9.5, va="center", ha="right", color=GRAY)


# ---------------- ① 사전 학습
y0 = H - 2.6
section_head(y0, "①", "사전 학습 — 모든 위치가 정답", "ch03/01_pretrain.py · TokenDataset", BLUEL, BLUE)
cw = 0.98
yx, yy, yw = y0 - 0.55, y0 - 0.55 - ROWH, y0 - 0.55 - 2 * ROWH
row_label(yx, "입력 x")
row_label(yy, "정답 y", "한 칸 뒤")
row_label(yw, "가중치")
for i in range(PC):
    x = CX0 + i * cw
    cell(ax, x, yx, cw - 0.06, show(PT_X[i]), BLUEL, BLUE)
    cell(ax, x, yy, cw - 0.06, show(PT_Y[i]), "white", BLUE)
    cell(ax, x, yw, cw - 0.06, "1", "white", MUTEE)
    if i < PC - 1:
        ax.plot([x + cw - 0.06 + 0.02, x + cw + 0.18], [yx - 0.21, yy + 0.21], color="#AAAAAA", lw=0.8, zorder=2)
ax.text(CX0, yw - 0.5, r"loss $= -\,$평균$\,\log p(y)$  — 배치 32 × 256 = 8,192개 위치 전부 (F.cross_entropy)",
        fontsize=10.5, va="center", color=INK)

# ---------------- ② SFT
# 칸 묶음: (입력 글자, 토큰 수, 레이블 글자, 가중치 글자, 폭, 색)
y0 = yw - 1.25
section_head(y0, "②", "SFT — 응답 위치만 정답, 나머지 레이블은 -100", "ch03/03_sft.py · SFTDataset", GREENL, GREEN)
yx, yy, yw = y0 - 0.55, y0 - 0.55 - ROWH, y0 - 0.55 - 2 * ROWH
row_label(yx, "입력 ids")
row_label(yy, "레이블", "한 칸 뒤")
row_label(yw, "가중치")


def strip(groups, yx, yy, yw, lab_c, wt_c):
    x = CX0
    for g in groups:
        txt, n, lab, wt, w, kind = g
        on = kind == "on"
        pad = kind == "pad"
        cell(ax, x, yx, w - 0.06, txt, MUTE if pad else (lab_c[0] if on else "white"),
             MUTEE if pad else lab_c[1], sub=(f"{n}토큰" if n > 1 else None), fs=10.5)
        cell(ax, x, yy, w - 0.06, lab, lab_c[0] if on else MUTE, lab_c[1] if on else MUTEE,
             color=INK if on else GRAY, fs=10.5)
        cell(ax, x, yw, w - 0.06, wt, wt_c if on else "white", lab_c[1] if on else MUTEE,
             color=INK if on else GRAY, fs=10.5)
        x += w
    return x


GA = "### Instruction:↵"
GB = "↵↵### Response:"
sft_groups = [(GA, 9, "-100 ×9", "0", 1.45, "off"),
              ("3", 1, "-100", "0", 0.48, "off"), ("+", 1, "-100", "0", 0.48, "off"),
              ("5", 1, "-100", "0", 0.48, "off"), ("=", 1, "-100", "0", 0.48, "off"),
              (GB, 8, "-100 ×8", "0", 1.4, "off"),
              ("↵", 1, show(SFT_LAB[21]), "1", 0.62, "on"),
              (show(SFT_IDS[22]), 1, show(SFT_LAB[22]), "1", 0.8, "on"),
              (f"0 ×{SFT_PAD}", SFT_PAD, f"-100 ×{SFT_PAD}", "0", 1.35, "pad")]
xe = strip(sft_groups, yx, yy, yw, (GREENL, GREEN), GREENL)
ax.text(CX0, yw - 0.5, "loss = −평균 log p(레이블)  — -100이 아닌 위치만 (ignore_index). 이 샘플은 256칸 중 2칸",
        fontsize=10.5, va="center", color=INK)
ax.text(CX0, yw - 0.85, f"프롬프트의 마지막 ↵ 위치가 첫 응답 토큰 '8'을, '8' 위치가 <eot>({EOT})를 예측한다. 회색 칸 = 고정 길이 256까지 채운 패딩",
        fontsize=9.5, va="center", color=GRAY)

# ---------------- ③ GRPO
y0 = yw - 1.6
section_head(y0, "③", "GRPO — 응답 위치에 어드밴티지 A를 곱한다", "ch03/09_grpo.py · get_batch, grpo_loss", ORANGEL, ORANGE)
yx, yy, yw = y0 - 0.55, y0 - 0.55 - ROWH, y0 - 0.55 - 2 * ROWH
row_label(yx, "입력 ids", "응답 '15'")
row_label(yy, "다음 토큰", "ids[:, 1:]")
row_label(yw, "가중치", "mask × A")
r0 = RESP_IDS[0]
npad = MAX_LEN - len(q_ids) - len(r0)
a0 = A[0]
grpo_groups = [(GA, 9, "…", "0", 1.45, "off"),
               ("7", 1, "+", "0", 0.48, "off"), ("+", 1, "8", "0", 0.48, "off"),
               ("8", 1, "=", "0", 0.48, "off"), ("=", 1, "↵", "0", 0.48, "off"),
               (GB, 8, "…", "0", 1.4, "off"),
               ("↵", 1, show(r0[0]), f"{a0:+.2f}", 0.62, "on"),
               (show(r0[0]), 1, show(r0[1]), f"{a0:+.2f}", 0.8, "on"),
               (show(r0[1]), 1, "0", "0", 0.62, "off"),
               (f"0 ×{npad}", npad, f"0 ×{npad - 1}", "0", 0.9, "pad")]
xe = strip(grpo_groups, yx, yy, yw, (ORANGEL, ORANGE), ORANGEL)
ax.text(CX0, yw - 0.5, r"목적 $=$ (응답 토큰의 $\min(r\,A,\ \mathrm{clip}(r, 0.8, 1.2)\,A)$ 합) ÷ $B\cdot G$,"
        r"  $r = \pi_\theta / \pi_{\theta_\mathrm{old}}$  — loss는 여기에 마이너스",
        fontsize=10.5, va="center", color=INK)
ax.text(CX0, yw - 0.75, "회색 칸 = 배치에서 가장 긴 응답(길이 28)에 맞춘 패딩. 마지막 칸은 다음 토큰이 없어 확률을 구하지 않는다(C−1).\n"
        "응답에는 <eot>가 없다(generate가 종료 토큰을 붙이지 않음). "
        "A는 응답마다 하나라 그 응답의 모든 토큰에 같은 값이\n곱해진다. 이 응답은 정답이라 A = +0.25 — 계산은 (b)",
        fontsize=9.5, va="top", color=GRAY, linespacing=1.4)

save(fig, ROOT / "docs/ch03/images/fig03-flow-a.png")

# =====================================================================
# (b) GRPO 반복 한 번
# =====================================================================
W, H = 10.0, 12.2
fig, ax = new_fig(W, H)
ax.text(0.3, H - 0.4, "(b)  GRPO 반복 한 번 — 질문 하나(7+8=)를 펼친 것", fontsize=14, fontweight="bold",
        color=INK, va="center")

# 단계 칩
chips = ["❶ 질문 32개 뽑기", "❷ old_model ← model", "❸ 질문마다 8개 생성\n(old_model, T = 1.0)",
         "❹ 보상 R · 어드밴티지 A", "❺ get_batch\n(256행, mask)", "❻ grpo_loss로 2번 갱신"]
cw_, cg = 1.47, 0.12
for k, t in enumerate(chips):
    x = 0.3 + k * (cw_ + cg)
    ax.add_patch(FancyBboxPatch((x, H - 1.55), cw_, 0.72, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=ORANGEL if k in (2, 3, 4) else "white", ec=INK, lw=0.9, zorder=3))
    ax.text(x + cw_ / 2, H - 1.19, t, fontsize=9.5, ha="center", va="center", zorder=4, linespacing=1.25)
    if k < 5:
        arrow(ax, x + cw_ + 0.01, H - 1.19, x + cw_ + cg - 0.01, H - 1.19, ms=8, lw=1.0)
ax.text(W - 0.3, H - 1.85, "10회마다 81문제를 T = 0으로 풀어 정답률을 잰다", fontsize=9.5, color=GRAY,
        ha="right", va="center")

# 프롬프트
py = H - 2.45
ax.text(0.3, py, "프롬프트", fontsize=10.5, va="center", color=INK)
ax.add_patch(FancyBboxPatch((1.3, py - 0.25), 4.3, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                            fc="white", ec=INK, lw=1.0, zorder=3))
ax.text(3.45, py, r"### Instruction:↵7+8=↵↵### Response:↵", fontsize=10.5, ha="center", va="center", zorder=4)
ax.text(5.8, py, "22토큰 · mask 0 · 정답 15", fontsize=10, color=GRAY, va="center")

# 표 머리
ty = py - 0.75
cols = {"i": 0.45, "text": 1.4, "tok": 2.75, "last": 6.35, "R": 7.55, "A": 8.6}
heads = {"i": "", "text": "응답 (8개)", "tok": "응답 토큰 → 배치 행 (프롬프트 뒤, 길이 28까지 패딩)",
         "last": "마지막 정수", "R": "보상 R", "A": "A = R − 0.75"}
for k, v in heads.items():
    ax.text(cols[k], ty, v, fontsize=10, color=GRAY, va="center", ha="left" if k == "tok" else "center")
ax.plot([0.3, W - 0.3], [ty - 0.25, ty - 0.25], color="#CCCCCC", lw=0.8)

TW = 0.55
for i, (txt, rid, r, a) in enumerate(zip(RESP, RESP_IDS, R, A)):
    y = ty - 0.55 - i * 0.44
    good = r > 0
    fc, ec = (GREENL, GREEN) if good else (REDL, RED)
    ax.text(cols["i"], y, f"{i + 1}", fontsize=10, color=GRAY, ha="center", va="center")
    ax.text(cols["text"], y, f'"{txt}"', fontsize=11, ha="center", va="center", color=INK)
    for j in range(6):
        x = cols["tok"] + j * TW
        if j < len(rid):
            cell(ax, x, y, TW - 0.05, show(rid[j]), fc, ec, h=0.38, fs=10.5)
        else:
            cell(ax, x, y, TW - 0.05, "0", MUTE, MUTEE, h=0.38, fs=10, color=GRAY)
    last = re.findall(r"(-?\d+)", txt)[-1]
    ax.text(cols["last"], y, last, fontsize=11, ha="center", va="center", color=INK)
    ax.text(cols["R"], y, f"{r:.0f}", fontsize=11, ha="center", va="center", color=GREEN if good else RED,
            fontweight="bold")
    ax.text(cols["A"], y, f"{a:+.2f}", fontsize=11, ha="center", va="center", color=GREEN if good else RED,
            fontweight="bold")
yb = ty - 0.55 - 7 * 0.44 - 0.32
ax.plot([0.3, W - 0.3], [yb, yb], color="#CCCCCC", lw=0.8)
ax.text(cols["R"], yb - 0.3, f"평균 {R.mean():.2f}", fontsize=10.5, ha="center", va="center", color=INK)
ax.text(cols["tok"], yb - 0.3, "초록 = 확률을 올릴 토큰, 빨강 = 내릴 토큰, 회색 = mask 0(패딩)",
        fontsize=9.5, va="center", color=GRAY)
ax.text(cols["tok"], yb - 0.62, "4번은 7, 8, 15 중 마지막 정수 15로 채점된다 (calculate_reward)",
        fontsize=9.5, va="center", color=GRAY)

# 클리핑 그래프 두 개 (실제 함수 min(rA, clip(r)A))
gy = yb - 1.0
ax.text(0.3, gy, "❻ 토큰 하나의 목적 함수  min(r·A, clip(r, 0.8, 1.2)·A)  — r = 확률비",
        fontsize=11, fontweight="bold", va="center", color=INK)
rr = np.linspace(0.4, 1.6, 400)
for k, (a, name, col) in enumerate([(A[0], "정답 응답", GREEN), (A[1], "오답 응답", RED)]):
    gx0 = 0.75 + k * 4.75
    gax = fig.add_axes([gx0 / W, 0.65 / H, 3.85 / W, (gy - 1.35) / H])
    obj = np.minimum(rr * a, np.clip(rr, 0.8, 1.2) * a)
    gax.axvspan(0.8, 1.2, color="#F2F2F2", zorder=0)
    gax.plot(rr, rr * a, color="#BBBBBB", lw=1.0, ls=(0, (4, 3)), label="r·A (클리핑 없음)")
    gax.plot(rr, obj, color=col, lw=2.2, label="min(…)")
    gax.axvline(1.0, color=INK, lw=0.8, ls=":")
    gax.plot([1.0], [a], "o", color=INK, ms=5, zorder=5)
    gax.set_ylim((-0.05, 0.55) if a > 0 else (-1.4, -0.05))
    gax.annotate("1회차 갱신: r = 1", (1.0, a), xytext=(1.27, 0.12) if a > 0 else (1.05, -1.2),
                 fontsize=9.5, color=INK, va="center",
                 arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
    edge = 1.2 if a > 0 else 0.8
    gax.axvline(edge, color=col, lw=0.8, ls="--")
    msg = ("r > 1.2: 더 올려도\n이득 없음 (기울기 0)" if a > 0 else "r < 0.8: 더 내려도\n이득 없음 (기울기 0)")
    gax.text(1.23 if a > 0 else 0.77, 0.5 if a > 0 else -0.16, msg, fontsize=9.5, color=col,
             ha="left" if a > 0 else "right", va="top", linespacing=1.3)
    gax.set_title(f"{name}  A = {a:+.2f}", fontsize=11, color=INK, pad=6)
    gax.set_xlabel("확률비 r", fontsize=10)
    gax.set_xticks([0.4, 0.8, 1.0, 1.2, 1.6])
    gax.tick_params(labelsize=9.5)
    for s in ("top", "right"):
        gax.spines[s].set_visible(False)
    gax.legend(fontsize=9, frameon=False, loc="upper left" if a > 0 else "lower left")

save(fig, ROOT / "docs/ch03/images/fig03-flow-b.png")
print("pretrain x:", [show(i) for i in PT_X], "\ny:", [show(i) for i in PT_Y])
print("sft labels tail:", SFT_LAB[19:], "pad", SFT_PAD)
print("R", R, "A", A, "max_len", MAX_LEN, [len(r) for r in RESP_IDS])
