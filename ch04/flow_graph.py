# 4장 노트의 그림 "개선된 BPE의 학습과 인코딩 흐름"을 그린다.
# StoryBot 데이터(2GB)는 저장소에 없으므로, 같은 코드(storybot/tokenizer.py)를 TinyCodes(codebot/tiny_codes.txt)에
# 4청크로 돌려 얻은 실제 값을 쓴다. 실제 StoryBot은 64청크·8프로세스·어휘 10,000이다.
# 실행(저장소 루트에서): .venv/bin/python ch04/flow_graph.py → docs/ch04/images/fig04-flow-a.png, fig04-flow-b.png
# 사전 토큰화와 전체 인코딩을 실제로 돌리므로 30초 정도 걸린다. 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
import pickle
from collections import defaultdict
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
from storybot.tokenizer import (find_chunk_boundaries, pretoken_chunk, count_pairs, merge,
                                BPETokenizer)

# ---------------------------------------------------------------- 데이터
FP = str(ROOT / "codebot/tiny_codes.txt")
END = "<|endoftext|>"
NCH = 4
FILE_SIZE = Path(FP).stat().st_size
EVEN = [i * (FILE_SIZE // NCH) for i in range(NCH)] + [FILE_SIZE]
BOUND = find_chunk_boundaries(FP, NCH)

# 4.4 청크별 사전 토큰화 (그림에서는 순서대로 돌리지만 결과는 Pool.imap과 같다)
CHUNK_COUNTS = [pretoken_chunk((FP, BOUND[i], BOUND[i + 1], END)) for i in range(len(BOUND) - 1)]
pretoken_counts = defaultdict(int)
for r in CHUNK_COUNTS:
    for k, v in r.items():
        pretoken_counts[k] += v
N_UNIQ, N_ALL = len(pretoken_counts), sum(pretoken_counts.values())

# 4.1 + 4.2 학습 루프 앞 5스텝
ids_counts = {tuple(p.encode("utf-8")): c for p, c in pretoken_counts.items()}
pair_counts, pair_to_ids = defaultdict(int), defaultdict(set)
for ids, c in ids_counts.items():
    count_pairs(ids, c, pair_counts)
    for p in zip(ids, ids[1:]):
        pair_to_ids[p].add(ids)
N_PAIRS0 = len(pair_counts)
id_bytes = {i: bytes([i]) for i in range(256)}
STEPS = []
for step in range(5):
    best = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1]))
    new_id = 256 + step
    id_bytes[new_id] = id_bytes[best[0]] + id_bytes[best[1]]
    aff = pair_to_ids.pop(best)
    STEPS.append((best, new_id, pair_counts[best], len(aff), len(ids_counts)))
    for ids in list(aff):
        n = ids_counts.pop(ids)
        new = tuple(merge(ids, best, new_id))
        ids_counts[new] = n
        for p, c in count_pairs(ids).items():
            pair_counts[p] -= c * n
            if pair_counts[p] <= 0:
                del pair_counts[p]
            pair_to_ids[p].discard(ids)
        for p, c in count_pairs(new).items():
            pair_counts[p] += c * n
            pair_to_ids[p].add(new)

tok = BPETokenizer.load_from(str(ROOT / "codebot/merge_rules.pkl"))
R = tok.merge_rules
FIRST_RULES = list(R.items())[:5]
assert [(s[0], s[1]) for s in STEPS] == FIRST_RULES   # 1장 규칙과 같은지


def show_b(b):
    return b.decode("utf-8", "replace").replace(" ", "␣").replace("\n", "↵")


def show(i):
    return show_b(tok.id_to_bytes[i])


# 4.6 손으로 따라가기
def trace(word):
    ids = list(word.encode("utf-8"))
    rows = []
    while len(ids) > 1:
        cands = list(count_pairs(ids))
        best = min(cands, key=lambda p: R.get(p, float("inf")))
        rows.append((list(ids), [(p, R.get(p)) for p in cands], best if best in R else None))
        if best not in R:
            break
        ids = merge(ids, best, R[best])
    rows.append((list(ids), None, None))
    return rows


TR_INPUT = trace(" input")
TR_APPEND = trace(" append")

# 4.7 청크별 인코딩 → 이어 붙이기
CHUNK_TOK = []
for i in range(len(BOUND) - 1):
    with open(FP, "rb") as f:
        f.seek(BOUND[i])
        t = f.read(BOUND[i + 1] - BOUND[i]).decode("utf-8", errors="ignore")
    CHUNK_TOK.append(np.array(tok.encode(t), dtype=np.uint16))
ALL = np.concatenate(CHUNK_TOK)
REF = np.fromfile(ROOT / "codebot/tiny_codes.bin", dtype=np.uint16)
SAME_BIN = len(ALL) == len(REF) and bool((ALL == REF).all())

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


def tag(ax, x, y, s):
    """절 번호 꼬리표"""
    ax.text(x, y, s, fontsize=10, color="white", fontweight="bold", va="center", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec="none"))


def save(fig, name):
    out = OUTDIR / name
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) <= 10000:
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(out, optimize=True)
    print(out)


OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch04/images"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# (a) BPE 학습
# =====================================================================
H = 25.3
fig, ax = new_fig(H)
ax.text(0.3, H - 0.7, "(a)  BPE 학습 — 파일 하나가 병합 규칙이 되기까지", fontsize=15, fontweight="bold",
        color=INK, va="center")
ax.text(0.3, H - 1.45, f"TinyCodes({FILE_SIZE:,}바이트)를 {NCH}청크로 나눠 실제로 돌린 값. "
        "StoryBot은 2GB · 64청크 · 8프로세스 · 어휘 10,000", fontsize=10, color=GRAY, va="center")

# --- 4.3 청크 경계
y0 = H - 3.1
tag(ax, 0.3, y0 + 0.55, "4.3")
ax.text(1.35, y0 + 0.55, "find_chunk_boundaries — 균등 위치에서 앞으로 <|endoftext|>를 찾아 경계를 옮긴다",
        fontsize=10.5, color=INK, va="center")
BX0, BX1 = 0.6, 17.4
sx = lambda b: BX0 + (BX1 - BX0) * b / FILE_SIZE
for i in range(NCH):
    ax.add_patch(Rectangle((sx(BOUND[i]), y0 - 0.75), sx(BOUND[i + 1]) - sx(BOUND[i]), 0.6,
                           fc=POSL[i], ec=POS[i], lw=1.1, zorder=3))
    ax.text((sx(BOUND[i]) + sx(BOUND[i + 1])) / 2, y0 - 0.45, f"청크 {i}  ·  {BOUND[i + 1] - BOUND[i]:,}B",
            fontsize=9.5, ha="center", va="center", zorder=4)
for i in range(1, NCH):
    x = sx(BOUND[i])
    ax.plot([x, x], [y0 - 0.95, y0 - 0.1], color=RED, lw=1.6, zorder=5)
    ax.text(x, y0 - 1.25, f"{EVEN[i]:,} → {BOUND[i]:,}  (+{BOUND[i] - EVEN[i]})",
            fontsize=9, ha="center", va="center", color=RED)
ax.text(BX0, y0 - 1.25, "0", fontsize=9, ha="center", va="center", color=GRAY)
ax.text(BX1, y0 - 1.25, f"{FILE_SIZE:,}", fontsize=9, ha="right", va="center", color=GRAY)

# --- 4.4 청크별 사전 토큰화
y1 = y0 - 2.4
tag(ax, 0.3, y1, "4.4")
ax.text(1.35, y1, "Pool.imap(pretoken_chunk, …) — 프로세스마다 (경로, 시작, 끝)만 받아 청크를 직접 읽고 센다",
        fontsize=10.5, color=INK, va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=1.5))
PW = 4.05
for i, r in enumerate(CHUNK_COUNTS):
    cx = 0.55 + PW / 2 + i * (PW + 0.25)
    arrow(ax, cx, y0 - 1.55, cx, y1 - 0.45, lw=1.0)
    cy = y1 - 2.15
    box(ax, cx, cy, PW, 3.2, "", POSL[i], ec=POS[i], lw=1.3)
    ax.text(cx, cy + 1.2, f"프로세스 {i + 1}", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(cx, cy + 0.65, f"고유 {len(r):,} / 전체 {sum(r.values()):,}", fontsize=9.5, ha="center",
            va="center", zorder=5)
    for k, (w, n) in enumerate(sorted(r.items(), key=lambda kv: -kv[1])[:3]):
        ax.text(cx - PW / 2 + 0.45, cy + 0.05 - k * 0.5, f"'{show_b(w.encode())}'", fontsize=9.5,
                va="center", zorder=5)
        ax.text(cx + PW / 2 - 0.35, cy + 0.05 - k * 0.5, f"{n:,}", fontsize=9.5, va="center", ha="right", zorder=5)

# --- 합산 → pretoken_counts (4.1)
y2 = y1 - 5.0
for i in range(NCH):
    cx = 0.55 + PW / 2 + i * (PW + 0.25)
    arrow(ax, cx, y1 - 3.8, 9.0 + (cx - 9.0) * 0.25, y2 + 0.85, lw=1.0)
tag(ax, 0.3, y2 + 0.2, "4.1")
box(ax, 9.0, y2, 10.6, 1.6, "", "white")
ax.text(9.0, y2 + 0.45, f"pretoken_counts  —  고유 사전 토큰 {N_UNIQ:,}개 (전체 {N_ALL:,}개)",
        fontsize=11.5, fontweight="bold", ha="center", va="center", zorder=5)
ex = [" print", " input", " item"]
ax.text(9.0, y2 - 0.3, "   ".join(f"'{show_b(w.encode())}': {pretoken_counts[w]:,}" for w in ex),
        fontsize=10, ha="center", va="center", zorder=5)
ax.text(14.45, y2 + 0.05, f"처리할 ID열이\n약 {N_ALL / N_UNIQ:.0f}분의 1로", fontsize=10, color=RED, va="center",
        ha="left", linespacing=1.35)

# --- ids_counts
y3 = y2 - 2.6
arrow(ax, 9.0, y2 - 0.82, 9.0, y3 + 0.8)
box(ax, 9.0, y3, 15.0, 1.5, "", C_LN)
ax.text(9.0, y3 + 0.4, "ids_counts  —  ID열(튜플) → 횟수", fontsize=11.5, fontweight="bold", ha="center",
        va="center", zorder=5)
ax.text(9.0, y3 - 0.3, "   ".join(f"{tuple(w.encode())}: {pretoken_counts[w]:,}" for w in ex[:2]) + "   …",
        fontsize=10, ha="center", va="center", zorder=5)

# --- 초기화: pair_counts + pair_to_ids
y4 = y3 - 2.4
arrow(ax, 9.0, y3 - 0.77, 9.0, y4 + 0.7)
ax.text(9.25, (y3 - 0.77 + y4 + 0.7) / 2, "처음 한 번만 전체를 센다", fontsize=9.5, color=GRAY, va="center")
box(ax, 4.6, y4, 8.2, 1.3, f"pair_counts — 쌍 → 횟수 (처음 {N_PAIRS0:,}쌍)\ncount_pairs(ids, weight=횟수)", C_FFN,
    fs=10)
box(ax, 13.4, y4, 8.2, 1.3, "pair_to_ids (캐시) — 쌍 → 그 쌍이 든 ID열들\n예: (105, 110) → {' input', ' print', …}",
    C_ATT, fs=10)

# --- 4.2 병합 루프
y5 = y4 - 1.6
tag(ax, 0.3, y5 - 0.3, "4.2")
ax.text(1.35, y5 - 0.3, "병합 루프 — 매 스텝 영향받는 ID열만 꺼내 차감/가산한다 (앞 5스텝, 실제 값)",
        fontsize=10.5, color=INK, va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=1.5))
arrow(ax, 9.0, y4 - 0.67, 9.0, y5 + 0.05)
cols = [(1.0, "스텝"), (2.6, "best_pair"), (6.0, "바이트"), (8.6, "횟수"), (11.6, "영향받는 ID열"), (15.4, "새 ID")]
ty = y5 - 1.15
ax.add_patch(Rectangle((0.6, ty - 0.33), 16.8, 0.66, fc="#EEEEEE", ec="none", zorder=1))
for x, t in cols:
    ax.text(x, ty, t, fontsize=10, fontweight="bold", va="center", color=INK)
for k, (best, nid, cnt, aff, tot) in enumerate(STEPS):
    yy = ty - 0.72 * (k + 1)
    ax.text(1.25, yy, str(k), fontsize=10, va="center")
    ax.text(2.6, yy, str(best), fontsize=10, va="center")
    ax.text(6.0, yy, f"'{show_b(id_bytes[best[0]])}' + '{show_b(id_bytes[best[1]])}'", fontsize=10, va="center")
    ax.text(8.6, yy, f"{cnt:,}", fontsize=10, va="center")
    ax.add_patch(Rectangle((11.6, yy - 0.2), 3.2 * aff / tot, 0.4, fc=RED, ec="none", zorder=3))
    ax.add_patch(Rectangle((11.6, yy - 0.2), 3.2, 0.4, fc="none", ec="#BBBBBB", lw=0.7, zorder=3))
    ax.text(11.6 + 3.2 * aff / tot + 0.12, yy, f"{aff:,} / {tot:,}", fontsize=9.5, va="center", zorder=4)
    ax.text(15.4, yy, f"{nid}", fontsize=10, va="center")
yb = ty - 0.72 * 5 - 0.55
ax.text(0.6, yb, f"스텝 0에서 다시 계산하는 ID열은 {STEPS[0][3]}개({STEPS[0][3] / STEPS[0][4] * 100:.2f}%)뿐. "
        f"4.1은 매 스텝 {STEPS[0][4]:,}개를 모두 다시 센다.", fontsize=10, color=RED, va="center")

# --- 결과
y6 = yb - 1.5
arrow(ax, 9.0, yb - 0.35, 9.0, y6 + 0.55)
box(ax, 9.0, y6, 12.5, 1.1, "", C_EMB)
ax.text(9.0, y6 + 0.18, "merge_rules  {(32, 32): 256, (256, 256): 257, (105, 110): 258, …}", fontsize=10.5,
        ha="center", va="center", zorder=5)
ax.text(9.0, y6 - 0.27, "1장 방식으로 학습한 codebot/merge_rules.pkl의 앞 규칙과 같다", fontsize=9.5,
        color=GRAY, ha="center", va="center", zorder=5)
save(fig, "fig04-flow-a.png")

# =====================================================================
# (b) BPE 인코딩
# =====================================================================
H = 25.4
fig, ax = new_fig(H)
ax.text(0.3, H - 0.7, "(b)  BPE 인코딩 — 단어 하나, 그리고 파일 전체", fontsize=15, fontweight="bold",
        color=INK, va="center")
ax.text(0.3, H - 1.45, f"CodeBot 규칙(codebot/merge_rules.pkl, {len(R)}개)으로 실제로 인코딩한 값",
        fontsize=10, color=GRAY, va="center")

tag(ax, 0.3, H - 2.6, "4.6")
ax.text(1.35, H - 2.6, "' input' — 지금 있는 쌍 중 새 ID가 가장 작은(가장 먼저 배운) 규칙부터 병합",
        fontsize=10.5, color=INK, va="center")
ax.text(0.6, H - 3.35, "반복", fontsize=10, fontweight="bold", va="center")
ax.text(1.8, H - 3.35, "ID열", fontsize=10, fontweight="bold", va="center")
ax.text(7.9, H - 3.35, "후보 쌍 → 우선순위(= 새 ID, 규칙에 없으면 ∞)", fontsize=10, fontweight="bold", va="center")
TW = 1.0


def trace_rows(rows, ytop, row_h=0.95, start=0):
    for k, (ids, cands, best) in enumerate(rows):
        yy = ytop - k * row_h
        ax.text(0.85, yy, str(start + k + 1) if cands else "끝", fontsize=10, va="center", ha="center")
        x = 1.6
        for i in ids:
            s = show(i)
            w = max(TW, 0.33 * len(s) + 0.35)
            box(ax, x + w / 2, yy, w - 0.08, 0.66, "", "white" if i < 256 else C_EMB, ec=GRAY if i < 256 else INK)
            ax.text(x + w / 2, yy + 0.1, s, fontsize=10, ha="center", va="center", zorder=5)
            ax.text(x + w / 2, yy - 0.2, str(i), fontsize=9, ha="center", va="center", color=GRAY, zorder=5)
            x += w
        if cands is None:
            continue
        for j, (p, pr) in enumerate(cands):
            cx = 7.9 + j * 1.9
            s = f"({show(p[0])},{show(p[1])}) {pr if pr is not None else '∞'}"
            chosen = p == best
            ax.text(cx, yy, s, fontsize=9.5, va="center", color=RED if chosen else (GRAY if pr is None else INK),
                    fontweight="bold" if chosen else "normal",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#FBE9E7" if chosen else "none",
                              ec=RED if chosen else "none", lw=0.9))
        if best is None:
            ax.text(7.9 + len(cands) * 1.9, yy, "← 모두 ∞ → 종료", fontsize=9.5, color=RED, va="center")


trace_rows(TR_INPUT, H - 4.2)
yA = H - 4.2 - len(TR_INPUT) * 0.95
ax.text(0.6, yA + 0.1, f"반복 {len(TR_INPUT) - 1}번으로 끝난다. 1장 방식은 이 단어에도 규칙 {len(R)}개를 모두 돈다.\n"
        "(␣,i) 274는 후보였지만 258이 먼저 적용되면서 사라졌다 — 1장 순서와 같은 결과",
        fontsize=10, color=RED, va="center", linespacing=1.5)
ax.text(0.6, yA - 1.0, "' append' — 규칙에 있는 쌍이 바닥나면 한 토큰이 되지 못하고 멈춘다", fontsize=10.5,
        color=INK, va="center")
trace_rows(TR_APPEND[-2:], yA - 1.85, start=len(TR_APPEND) - 2)

# --- 4.7 파일 전체
yC = yA - 4.6
tag(ax, 0.3, yC, "4.7")
ax.text(1.35, yC, "encode_file — 청크마다 인코딩해 임시 파일에 쓰고, memmap 하나에 순서대로 이어 붙인다",
        fontsize=10.5, color=INK, va="center")
PW = 4.05
offs = np.cumsum([0] + [len(c) for c in CHUNK_TOK])
yP, yF, yM = yC - 1.3, yC - 3.3, yC - 6.3
for i, c in enumerate(CHUNK_TOK):
    cx = 0.55 + PW / 2 + i * (PW + 0.25)
    box(ax, cx, yP, PW, 1.2, "", POSL[i], ec=POS[i], lw=1.3)
    ax.text(cx, yP + 0.25, f"프로세스 {i + 1}: _encode_chunk", fontsize=10, ha="center", va="center", zorder=5)
    ax.text(cx, yP - 0.25, f"청크 {i} ({BOUND[i + 1] - BOUND[i]:,}B)", fontsize=9.5, ha="center", va="center",
            color=GRAY, zorder=5)
    arrow(ax, cx, yP - 0.62, cx, yF + 0.55, lw=1.0)
    box(ax, cx, yF, PW, 1.05, "", "white", ec=POS[i], lw=1.1, ls=(0, (4, 2)))
    ax.text(cx, yF + 0.2, f"bpe_cache/chunk_{i:05d}.npy", fontsize=9.5, ha="center", va="center", zorder=5)
    ax.text(cx, yF - 0.22, f"토큰 {len(c):,}개", fontsize=9.5, ha="center", va="center", zorder=5)
    # memmap 속 자리
    bx0 = 0.6 + 16.8 * offs[i] / offs[-1]
    bw = 16.8 * len(c) / offs[-1]
    arrow(ax, cx, yF - 0.55, bx0 + bw / 2, yM + 0.4, lw=1.0)
    ax.add_patch(Rectangle((bx0, yM - 0.3), bw, 0.7, fc=POSL[i], ec=POS[i], lw=1.1, zorder=3))
    ax.text(bx0 + 0.08, yM - 0.65, f"arr[{offs[i]:,}:]", fontsize=9, va="center", color=GRAY)
ax.text(9.0, yM + 0.05, f"np.memmap(uint16, shape=({offs[-1]:,},))  →  tiny_codes.bin  {offs[-1] * 2:,}B",
        fontsize=10.5, ha="center", va="center", zorder=5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
ax.text(0.6, yM - 1.9,
        "• 프로세스는 ID열을 직접 돌려주지 않고 파일에 쓴 뒤 (파일 이름, 토큰 수)만 돌려준다.\n"
        "• 크기를 알아야 memmap을 만들 수 있으므로, 토큰 수 합계를 먼저 구한 뒤 파일을 하나씩 읽어 채운다.\n"
        "• 메모리에는 한 번에 청크 하나만 올라간다. 다 쓰면 bpe_cache 폴더를 지운다.\n"
        f"• 이어 붙인 결과는 저장소의 codebot/tiny_codes.bin과 {'완전히 같다' if SAME_BIN else '다르다'} "
        "(청크로 나눠도 결과가 바뀌지 않는다).",
        fontsize=10, color=INK, va="center", linespacing=1.7)
save(fig, "fig04-flow-b.png")
print("steps", STEPS, "same_bin", SAME_BIN)
