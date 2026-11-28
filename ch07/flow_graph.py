# 7장 노트의 그림 "WebBot 토크나이저가 만들어지는 흐름"을 그린다.
# (a) owt_train.txt → 청크 분할 → 병렬 사전 토큰화 → 결과 통합 → BPE 학습 49,743회 → merge_rules.pkl
#     → encode_file → owt_train.bin, 그리고 50,000개 어휘의 ID 배정(uint16 범위 안).
# (b) 같은 흐름을 책 7.1절의 OpenWebText 예시 문서 한 개(683바이트)로 실제로 돌린 결과.
#     storybot/tokenizer.py의 train_bpe()를 그대로 호출해 병합 30번을 학습하고, 그 규칙으로 인코딩한다.
#     (OWT 12GB와 WebBot의 merge_rules.pkl은 저장소에 없으므로, 실제 값은 이 작은 예시로 보여 준다.)
# 실행(저장소 루트에서): .venv/bin/python ch07/flow_graph.py → docs/ch07/images/fig07-flow-a.png, fig07-flow-b.png
# 나눔명조(NanumMyeongjo) 폰트가 필요하다.
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from storybot.tokenizer import train_bpe, BPETokenizer, pretokenize

# ---------------------------------------------------------------- 데이터
# 책 7.1절(2, 3부.pdf)에 실린 OpenWebText 문서 한 개. 끝에 문서 구분 토큰이 붙어 있다.
SAMPLE = (
    "When Cara Jones let her black-and-white pit bull, “Creature,” out Thursday night to relieve herself "
    "at the end of William Street, she heard something unusual coming from the woods. “I kept hearing "
    "sticks breaking,” said Jones, 23, a hairstylist who works in Bridgewater. “I thought it was a fox.” "
    "Suddenly, her 2-year-old dog began barking wildly. “She was standing there, barking and looking back "
    "at me,” Jones said. Jones ran back into her home, grabbed a flashlight and cellphone and ran into the "
    "woods. There, in the brush she spotted 89-year-old Carmen Mitchell, lying on the ground. “She was "
    "covered in mud and she was barefoot,” Jones said....<|endoftext|>"
)
END = "<|endoftext|>"
N_MERGES = 30

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "owt_sample.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    MERGES = train_bpe(str(p), 256 + 1 + N_MERGES, num_processes=1, num_chunks=1)
tok = BPETokenizer(MERGES)

N_BYTES = len(SAMPLE.encode("utf-8"))
PRETOKENS = [t for s in SAMPLE.split(END) for t in pretokenize(s)]
TOP = Counter(PRETOKENS).most_common(6)
N_TOKENS = len(tok.encode(SAMPLE))

# WebBot 설정 (ch07/01_bpe_train.py, storybot/tokenizer.py)
VOCAB = 50000
NUM_MERGES = VOCAB - 256 - 1           # 49,743
LAST_MERGE_ID = 256 + NUM_MERGES - 1   # 49,998
END_ID = 256 + NUM_MERGES              # 49,999


def vis(b: bytes) -> str:
    """바이트열을 보이는 글자로. 공백은 ␣, ASCII가 아닌 바이트는 16진수."""
    try:
        s = b.decode("ascii")
        return s.replace(" ", "␣")
    except UnicodeDecodeError:
        out = []
        for x in b:
            out.append("␣" if x == 32 else (chr(x) if x < 128 else f"{x:02X}"))
        return " ".join(out) if any(x >= 128 for x in b) else "".join(out)


# ---------------------------------------------------------------- 스타일 (2장 flow_graph.py와 같음)
for fp in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(fp))
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY, RED = "#222222", "#777777", "#B03A2E"
POS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
POSL = ["#C9D6EA", "#F3D6C2", "#CDE6D3", "#EECACB", "#DAD4EC"]
C_EMB, C_ATT, C_FFN, C_LN, C_OUT, C_BLK = "#F6D6D6", "#F9E3C2", "#CFE3F3", "#F3F0C4", "#E2DAF0", "#F4F4F4"


def new_fig(w, h):
    fig = plt.figure(figsize=(w / 2, h / 2), dpi=200)   # 1 단위 = 0.5인치
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc, ec=INK, lw=1.1, ls="-", z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2, cs="arc3", ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, connectionstyle=cs, zorder=2, shrinkA=0, shrinkB=0))


def tag(ax, x, y, s, color=RED):
    ax.text(x, y, s, fontsize=10, color=color, va="center", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.9))


def save(fig, out):
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    im = Image.open(out).convert("RGB")
    if len(im.getcolors(1 << 24) or []) > 10000:
        im.save(out, optimize=True)
    else:
        im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, optimize=True)
    print(out)


outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ch07/images"
outdir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# (a) WebBot 토크나이저 파이프라인
# =====================================================================
W, H = 18, 22.6
fig, ax = new_fig(W, H)
ax.text(0.5, H - 0.6, "(a)  WebBot 토크나이저가 만들어지는 흐름 — 학습 → 인코딩",
        fontsize=15, fontweight="bold", color=INK, va="center")
X, BW = 4.6, 7.4          # 단계 상자의 가운데 x와 폭
TX = X + BW / 2 + 0.5     # 오른쪽 설명 시작 x


def stage(y, h, title, sub, fc, code=None):
    box(ax, X, y, BW, h, fc)
    dy = 0.2 if sub else 0
    ax.text(X, y + dy, title, fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=5)
    if sub:
        ax.text(X, y - 0.33, sub, fontsize=10, color=GRAY, ha="center", va="center", zorder=5)
    if code:
        ax.text(X - BW / 2 + 0.15, y + h / 2 - 0.02, code, fontsize=9.5, color=GRAY, ha="left", va="bottom",
                family="DejaVu Sans Mono")


def note(y, s, color=INK, fs=10.5):
    ax.text(TX, y, s, fontsize=fs, color=color, va="center", ha="left", linespacing=1.45)


# 학습 단계 묶음
ax.add_patch(FancyBboxPatch((0.35, 9.3), W - 0.7, H - 1.4 - 9.3, boxstyle="round,pad=0,rounding_size=0.25",
                            fc="none", ec=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(W - 0.55, H - 1.65, "학습  ch07/01_bpe_train.py → train_bpe()", fontsize=10.5, color=GRAY,
        ha="right", va="center")

y = H - 2.4
stage(y, 1.1, "owt_train.txt  (약 12GB)", f"문서마다 {END} 로 구분", "white")
note(y, "OpenWebText — 뉴스·리뷰·블로그가 섞인 웹 문서\n어휘만 50,000으로 바꾸고 storybot 코드를 그대로 쓴다")
arrow(ax, X, y - 0.55, X, y - 1.05)

y -= 1.85
stage(y, 1.1, "청크 분할  find_chunk_boundaries()", f"64등분한 뒤 경계를 가장 가까운 {END}로 민다", C_LN)
note(y, "문서 중간에서 자르지 않으므로\n청크끼리 사전 토큰이 끊기지 않는다")
arrow(ax, X, y - 0.55, X, y - 1.0)

# 병렬 사전 토큰화: 프로세스 8개
y -= 2.05
box(ax, X, y, BW, 1.9, C_FFN)
ax.text(X, y + 0.6, "사전 토큰화  pretoken_chunk()", fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=5)
for k in range(8):
    cx = X - BW / 2 + 0.55 + k * 0.9
    box(ax, cx, y - 0.25, 0.72, 0.62, "white", ec=POS[k % 5], lw=1.1, z=4)
    ax.text(cx, y - 0.25, f"P{k}", fontsize=10, ha="center", va="center", color=POS[k % 5], zorder=5)
ax.text(X, y - 0.78, "Pool(8): 청크 64개를 프로세스 8개가 나눠 처리", fontsize=10, color=GRAY, ha="center",
        va="center", zorder=5)
note(y + 0.25, "청크마다 정규식으로 단어 단위(사전 토큰)로 자르고\n빈도를 센다 → pretoken_counts 64개")
tag(ax, TX, y - 0.6, "7.2 병목 후보: regex 매칭·딕셔너리 카운트")
arrow(ax, X, y - 0.95, X, y - 1.4)

y -= 2.05
stage(y, 1.1, "결과 통합", "64개 dict를 하나로 더함 → 고유 사전 토큰을 바이트 ID열로", "white")
note(y + 0.2, "부모 프로세스 하나가 처리한다(병렬 아님)")
tag(ax, TX, y - 0.35, "7.2 병목 후보: 단일 프로세스 + 메모리")
arrow(ax, X, y - 0.55, X, y - 1.0)

# BPE 학습 루프
y -= 2.25
box(ax, X, y, BW, 2.3, C_ATT)
ax.text(X, y + 0.75, f"BPE 학습 루프  × {NUM_MERGES:,}회", fontsize=12.5, fontweight="bold", ha="center",
        va="center", zorder=5)
for k, s in enumerate(["① max(pair_counts): 가장 많은 쌍 고르기", "② 새 ID = 256 + step 으로 병합 규칙 등록",
                       "③ 그 쌍이 든 ID열만 다시 세기 (pair_to_ids)"]):
    ax.text(X - BW / 2 + 0.35, y + 0.2 - k * 0.48, s, fontsize=10.5, ha="left", va="center", zorder=5)
ax.add_patch(FancyArrowPatch((X + BW / 2 - 0.1, y - 0.75), (X + BW / 2 - 0.1, y + 0.4), arrowstyle="-|>",
                             mutation_scale=10, color=GRAY, lw=1.0, connectionstyle="arc3,rad=-0.9", zorder=5))
note(y + 0.6, f"병합 횟수 = 50,000 − 256(바이트) − 1(특수 토큰) = {NUM_MERGES:,}\n"
              "StoryBot(9,743회)의 약 5배")
tag(ax, TX, y - 0.3, "7.2 병목 후보: ①이 매번 모든 쌍을 훑는다 → 힙?")
tag(ax, TX, y - 0.95, "7.3 알고리즘으로 더 못 줄이면 러스트/C++로", color=POS[0])
arrow(ax, X, y - 1.15, X, y - 1.6)

y -= 2.05
stage(y, 0.9, "webbot/merge_rules.pkl", None, C_EMB)
note(y, f"{{(ID1, ID2): 새 ID}} 사전 {NUM_MERGES:,}개 — 학습의 결과물은 이것뿐")

# 인코딩 묶음
ax.add_patch(FancyBboxPatch((0.35, 5.0), W - 0.7, 4.1, boxstyle="round,pad=0,rounding_size=0.25",
                            fc="none", ec=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(W - 0.55, 8.85, "인코딩  ch07/02_bpe_encode.py → encode_file()", fontsize=10.5, color=GRAY,
        ha="right", va="center")
arrow(ax, X, y - 0.45, X, 8.6)
y = 8.05
box(ax, X, y, BW, 1.1, C_FFN)
ax.text(X, y + 0.2, "청크 64개 × 프로세스 8개로 encode()", fontsize=12.5, fontweight="bold", ha="center",
        va="center", zorder=5)
ax.text(X, y - 0.33, "사전 토큰마다 병합 규칙을 우선순위(작은 ID) 순으로 적용", fontsize=10, color=GRAY,
        ha="center", va="center", zorder=5)
note(y + 0.2, "owt_train.txt와 owt_valid.txt를 각각 처리")
tag(ax, TX, y - 0.35, "7.2 병목 후보: 같은 단어를 매번 다시 인코딩 → 캐시?")
arrow(ax, X, y - 0.55, X, y - 1.0)
y -= 1.75
stage(y, 1.1, "owt_train.bin / owt_valid.bin", "토큰 ID를 uint16(2바이트)으로 이어 붙인 파일", C_EMB)
note(y, "8장·9장 학습이 np.memmap으로 읽는 입력")

# ID 배정 막대: 0 ~ 65,535 (uint16)
ax.text(0.5, 4.25, f"어휘 {VOCAB:,}개의 ID 배정 — uint16 한 칸(0~65,535)에 들어간다", fontsize=12, fontweight="bold",
        color=INK, va="center")
BX0, BX1, BY, BH = 0.7, W - 0.7, 3.1, 0.75
sc = (BX1 - BX0) / 65536


def seg(a, b, fc, ec=INK, hatch=None):
    ax.add_patch(Rectangle((BX0 + a * sc, BY - BH / 2), (b - a) * sc, BH, fc=fc, ec=ec, lw=0.9, hatch=hatch, zorder=3))


seg(0, 256, POS[0])
seg(256, END_ID, POSL[1], ec=POS[1])
seg(END_ID, END_ID + 1, RED, ec=RED)
seg(END_ID + 1, 65536, "#F2F2F2", ec="#BBBBBB")
ax.text(BX0 + (256 + END_ID) / 2 * sc, BY, f"병합으로 생긴 토큰  256 ~ {LAST_MERGE_ID:,}  ({NUM_MERGES:,}개)",
        fontsize=11, ha="center", va="center", zorder=5)
ax.text(BX0 + (END_ID + 65536) / 2 * sc, BY, "비어 있음 (15,536칸)", fontsize=10.5, color=GRAY, ha="center",
        va="center", zorder=5)
ax.text(BX0, BY - 0.75, "0~255: 바이트 256개", fontsize=10.5, color=POS[0], ha="left", va="center")
ax.annotate(f"{END_ID:,}: {END}", xy=(BX0 + END_ID * sc, BY - BH / 2), xytext=(BX0 + END_ID * sc - 0.2, BY - 0.8),
            fontsize=10.5, color=RED, ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
ax.annotate("50,000~: 7.4의 특수 토큰을 늘릴 자리\n(임베딩 행 수도 함께 늘려야 함)", xy=(BX0 + 50300 * sc, BY - BH / 2),
            xytext=(BX1, BY - 1.6), fontsize=10.5, color=INK, ha="right", va="center",
            linespacing=1.4, arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
ax.text(BX1, BY + 0.62, "65,535", fontsize=10.5, color=GRAY, ha="right", va="center")
ax.text(BX0, BY + 0.62, "0", fontsize=10.5, color=GRAY, ha="left", va="center")
ax.text(0.5, 0.4, "65,536개 이상으로 키우면 encode_file()의 dtype을 uint32로 바꿔야 한다(파일 크기도 2배).",
        fontsize=10.5, color=GRAY, va="center")

save(fig, outdir / "fig07-flow-a.png")

# =====================================================================
# (b) 작은 예시로 실제로 돌려 보기
# =====================================================================
W, H = 18, 21.4
fig, ax = new_fig(W, H)
ax.text(0.5, H - 0.6, f"(b)  같은 흐름을 OWT 문서 한 개({N_BYTES}바이트)로 실제로 돌린 결과",
        fontsize=15, fontweight="bold", color=INK, va="center")
ax.text(0.5, H - 1.3, f"storybot/tokenizer.py의 train_bpe(vocab_size=256+1+{N_MERGES})를 그대로 호출 — 병합 {N_MERGES}번만 학습",
        fontsize=10.5, color=GRAY, va="center")

# ① 원문
y0 = H - 2.1
ax.text(0.5, y0, "① 원문 (책 7.1절의 예시 문서)", fontsize=12, fontweight="bold", color=INK, va="center")
box(ax, W / 2, y0 - 1.15, W - 1.0, 1.55, "white", ec=GRAY)
ax.text(0.75, y0 - 1.15,
        "When Cara Jones let her black-and-white pit bull, “Creature,” out Thursday night to relieve herself at\n"
        "the end of William Street, she heard something unusual coming from the woods. “I kept hearing ...\n"
        f"... “She was covered in mud and she was barefoot,” Jones said....{END}",
        fontsize=10.5, color=INK, va="center", ha="left", linespacing=1.55, zorder=5)

# ② 사전 토큰화
y1 = y0 - 2.5
ax.text(0.5, y1, f"② {END}로 나눈 뒤 사전 토큰화 → {len(PRETOKENS)}개 (고유 {len(set(PRETOKENS))}개)",
        fontsize=12, fontweight="bold", color=INK, va="center")
x = 0.6
for k, t in enumerate(PRETOKENS[:16]):
    s = t.replace(" ", "␣")
    w = 0.32 + 0.205 * len(s)
    if x + w > W - 0.5:
        break
    box(ax, x + w / 2, y1 - 0.75, w, 0.62, POSL[k % 5], ec=POS[k % 5], z=4)
    ax.text(x + w / 2, y1 - 0.75, s, fontsize=10.5, ha="center", va="center", zorder=5)
    x += w + 0.12
ax.text(0.6, y1 - 1.55, "빈도 상위: " + "   ".join(f"{t.replace(' ', '␣')} ×{c}" for t, c in TOP),
        fontsize=10.5, color=INK, va="center")
ax.text(0.6, y1 - 2.1, "␣는 앞 공백. 공백은 뒤 단어에 붙고, 문장 부호는 따로 떨어진다. 병합은 사전 토큰 안에서만 일어난다.",
        fontsize=10, color=GRAY, va="center")

# ③ 병합 규칙
y2 = y1 - 2.85
ax.text(0.5, y2, f"③ BPE 학습 — 앞쪽 병합 규칙 {N_MERGES}개 중 16개 (새 ID = 256 + step)", fontsize=12,
        fontweight="bold", color=INK, va="center")
items = list(MERGES.items())[:16]
for k, ((a, b), n) in enumerate(items):
    col, row = divmod(k, 8)
    cx = 0.8 + col * 8.7
    yy = y2 - 0.7 - row * 0.62
    nonascii = any(v >= 128 for v in tok.id_to_bytes[n])
    color = RED if nonascii else INK
    ax.text(cx, yy, f"{n}", fontsize=10.5, color=GRAY, va="center", ha="left", family="DejaVu Sans Mono")
    ax.text(cx + 1.0, yy, f"{vis(tok.id_to_bytes[a])}  +  {vis(tok.id_to_bytes[b])}", fontsize=10.5, color=color,
            va="center", ha="left", family="DejaVu Sans Mono")
    ax.text(cx + 3.4, yy, f"→  {vis(tok.id_to_bytes[n])}", fontsize=10.5, color=color, va="center", ha="left",
            family="DejaVu Sans Mono")
yq = y2 - 0.7 - 7 * 0.62 - 0.75
ax.text(0.6, yq, "빨간 줄: 곡선 따옴표 “ ”는 UTF-8로 3바이트(E2 80 9C / E2 80 9D) — 바이트 단위 BPE는 이 바이트들도 같은 방식으로 합친다.\n"
        "번호가 작을수록 먼저 학습된(더 자주 나온) 쌍이고, 인코딩할 때도 작은 번호부터 적용한다.",
        fontsize=10, color=GRAY, va="center", linespacing=1.5)

# ④ 인코딩
y3 = yq - 1.05
ax.text(0.5, y3, "④ 학습한 규칙으로 인코딩 — 토큰 하나 = 상자 하나, 아래 숫자가 토큰 ID", fontsize=12,
        fontweight="bold", color=INK, va="center")


def token_row(text, yy):
    ids = tok.encode(text)
    x = 0.6
    for k, i in enumerate(ids):
        s = vis(tok.id_to_bytes[i])
        w = max(0.62, 0.3 + 0.19 * len(s))
        merged = i >= 256
        box(ax, x + w / 2, yy, w, 0.62, POSL[1] if merged else "white", ec=POS[1] if merged else "#999999", z=4)
        ax.text(x + w / 2, yy, s, fontsize=10.5, ha="center", va="center", zorder=5,
                family="DejaVu Sans Mono")
        ax.text(x + w / 2, yy - 0.55, str(i), fontsize=9, ha="center", va="center", color=GRAY, zorder=5)
        x += w + 0.08
    nb = len(text.encode("utf-8"))
    ax.text(W - 0.5, yy, f"{nb}바이트 → {len(ids)}토큰", fontsize=10.5, color=INK, ha="right", va="center")
    return ids


token_row("When Cara Jones let her", y3 - 0.85)
token_row(" “Creature,”", y3 - 2.1)
ax.add_patch(Rectangle((0.6, y3 - 3.45), 0.45, 0.32, fc=POSL[1], ec=POS[1], lw=1.0))
ax.text(1.15, y3 - 3.29, "병합으로 생긴 토큰(ID ≥ 256)", fontsize=10, color=INK, va="center")
ax.add_patch(Rectangle((5.6, y3 - 3.45), 0.45, 0.32, fc="white", ec="#999999", lw=1.0))
ax.text(6.15, y3 - 3.29, "합쳐지지 못하고 남은 바이트(ID < 256)", fontsize=10, color=INK, va="center")

# 정리
y4 = y3 - 3.95
box(ax, W / 2, y4 - 0.85, W - 1.0, 1.75, C_BLK, ec=GRAY)
ax.text(0.8, y4 - 0.85,
        f"• 문서 전체: {N_BYTES}바이트 → {N_TOKENS}토큰 (병합 {N_MERGES}번만 배웠으므로 대부분 아직 바이트)\n"
        f"• 마지막 토큰은 {END} = ID {tok.end_token_id} (= 256 + 병합 {N_MERGES}개). WebBot에서는 {END_ID:,}\n"
        f"• WebBot은 이 과정을 12GB에서 병합 {NUM_MERGES:,}번 반복한다 — 그래서 (a)의 병목 후보가 문제가 된다",
        fontsize=10.5, color=INK, va="center", linespacing=1.7, zorder=5)

save(fig, outdir / "fig07-flow-b.png")
