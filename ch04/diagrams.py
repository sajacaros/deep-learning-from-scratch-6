# 4장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch04/diagrams.py
# 강조(fill)는 '최종 산출물'과 각 그림의 핵심 단계 하나에만 쓴다.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram
import matplotlib.pyplot as plt

# ①②③④ 글리프가 나눔명조·DejaVu Serif에 없어서 DejaVu Sans를 마지막 대체 글꼴로 더한다.
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Serif", "DejaVu Sans"]

OUT = ROOT / "docs/ch04/images"


def bus_down(d, src, dsts, y_bus):
    """src 아래에서 가로 버스로 내려가 dsts 각각의 위로 화살표."""
    d.line([src.bottom, (src.x, y_bus)])
    xs = [n.x for n in dsts] + [src.x]
    d.line([(min(xs), y_bus), (max(xs), y_bus)])
    for n in dsts:
        d.arrow((n.x, y_bus), n.top)


def bus_up(d, srcs, dst, y_bus):
    """srcs 각각의 아래에서 가로 버스로 모여 dst 위로 화살표 하나."""
    for n in srcs:
        d.line([n.bottom, (n.x, y_bus)])
    xs = [n.x for n in srcs] + [dst.x]
    d.line([(min(xs), y_bus), (max(xs), y_bus)])
    d.arrow((dst.x, y_bus), dst.top)


# ---------------------------------------------------------------- d01: 4장 전체 흐름
d = Diagram(17.6, 9.7)
W, H = 2.75, 1.3
xs = [1.5 + i * 3.65 for i in range(5)]
y1, y2, y3 = 7.3, 4.75, 1.15
d.group(0.0, y1 - H / 2 - 0.3, 17.6, y1 + H / 2 + 1.55, "BPE 학습 개선 (train_bpe)")
row1 = [("1장 구현", "TinyCodes 약 15분"), ("4.1 중복 제거", "약 30초"), ("4.2 부분 갱신(캐시)", "약 1초"),
        ("4.3 분할 처리", "메모리 대폭 감소"), ("4.4 병렬화", "약 30초")]
n1 = [d.box(x, y1, t, sub=s, w=W, h=H) for x, (t, s) in zip(xs, row1)]
lab1 = ["같은 사전 토큰을\n매번 따로 처리", "병합마다 전체를\n다시 셈", "2GB 통째로 읽어\n메모리 10GB+", "사전 토큰화가\n병목(약 3분)"]
for a, b, lab in zip(n1, n1[1:], lab1):
    d.arrow(a, b, side="h")
    d.text((a.x + b.x) / 2, y1 + H / 2 + 0.15, lab, style="italic", va="bottom", linespacing=1.2)
ev = d.box(xs[4], y2, "4.5 평가", sub="압축률 4.18배", w=W, h=H)
d.arrow(n1[4], ev, side="v")
# 인코딩 묶음
g_top = y3 + H / 2 + 1.55
d.group(0.0, y3 - H / 2 - 0.3, xs[2] + W / 2 + 0.35, g_top, "BPE 인코딩 개선 (encode)")
d.arrow(ev.left, (xs[0], g_top), path=[(xs[0], y2)])
row3 = [("1장 구현", "TinyCodes 약 5분"), ("4.6 고속 병합", "약 5초"), ("4.7 청크 병렬 + memmap", "TinyStories 약 20분")]
n3 = [d.box(x, y3, t, sub=s, w=W + (0.5 if i == 2 else 0), h=H) for i, (x, (t, s)) in enumerate(zip(xs, row3))]
lab3 = ["규칙 10,000개를\n전부 순회", "2GB를 한 번에\n메모리에"]
for a, b, lab in zip(n3, n3[1:], lab3):
    d.arrow(a, b, side="h")
    d.text((a.x + b.x) / 2, y3 + H / 2 + 0.15, lab, style="italic", va="bottom", linespacing=1.2)
binn = d.box((xs[3] + xs[4]) / 2 + 0.6, y3, "tiny_stories_train.bin", sub="(5장·6장 학습 입력)", w=3.6, h=H,
             fill=True)
d.arrow(n3[2], binn, side="h")
d.save(OUT / "fig04-d01.png")

# ---------------------------------------------------------------- d02: 부분 갱신 (캐시)
d = Diagram(17.6, 6.3)
yt, yb = 5.65, 1.2
bp = d.box(2.0, yt, "best_pair = (7, 8)", w=3.6, h=1.0)
af = d.box(8.4, yt, "영향받는 ID열", sub="{(7, 8, 9)}", w=3.4, h=1.2, fill=True)
m = d.box(14.6, yt, "merge → (256, 9)", w=3.6, h=1.0)
d.arrow(bp, af, side="h", label="pair_to_ids 조회")
d.arrow(af, m, side="h")
ub = [d.box(3.0, yb, "① ids_counts", sub="del (7,8,9)\nadd (256,9): 200", w=4.6, h=1.9),
      d.box(8.8, yb, "② pair_counts", sub="(7,8) −200 → 삭제\n(8,9) −200 → 삭제\n(256,9) +200", w=4.6, h=1.9),
      d.box(14.6, yb, "③ pair_to_ids", sub="(7,8)·(8,9)에서 (7,8,9) 제거\n(256,9) → {(256,9)} 추가", w=5.4, h=1.9)]
bus_down(d, m, ub, (yt - 0.5 + yb + 0.95) / 2)
d.save(OUT / "fig04-d02.png")

# ---------------------------------------------------------------- d03: 청크 경계 찾기
d = Diagram(17.6, 5.5)
yr, ye, yl = 4.3, 1.9, 0.35
a = d.box(1.6, yr, "파일 크기 / num_chunks", sub="= 균등 간격 경계", w=3.2, h=1.3)
b = d.box(5.6, yr, "각 경계에서", sub="4096바이트 읽기", w=2.7, h=1.3)
c = d.box(9.4, yr, "<|endoftext|>\n발견?", w=3.3, h=1.9, shape="diamond")
dd = d.box(13.0, yr, "그 위치를", sub="새 경계로", w=2.5, h=1.3, fill=True)
g = d.box(16.1, yr, "sorted(set(경계))", sub="중복 제거 후 반환", w=2.9, h=1.3)
e = d.box(9.4, ye, "다음 4096바이트로", w=3.0, h=0.9)
f = d.box(16.1, ye, "경계 = 파일 끝", w=2.9, h=0.9)
d.arrow(a, b, side="h")
d.arrow(b, c, side="h")
d.arrow(c, dd, side="h", label="예")
d.arrow(dd, g, side="h")
d.arrow(c, e, side="v", label="아니오")
d.arrow(e.left, (b.x + 0.5, b.y - 0.65), path=[(b.x + 0.5, ye)])
d.arrow((b.x - 0.5, b.y - 0.65), f.bottom, path=[(b.x - 0.5, yl), (f.x, yl)], label="파일 끝")
d.arrow(f, g, side="v")
d.save(OUT / "fig04-d03.png")

# ---------------------------------------------------------------- d04: 병렬 사전 토큰화
d = Diagram(17.6, 8.9)
y1, y2, y3 = 8.0, 4.9, 1.0
f = d.box(2.4, y1, "tiny_stories_train.txt", sub="2GB", w=3.6, h=1.8, shape="db")
b = d.box(8.0, y1, "find_chunk_boundaries", sub="64청크", w=3.6, h=1.2)
l = d.box(14.2, y1, "chunk_info_list", sub="(path, start, end, end_token) × 64", w=5.6, h=1.2)
d.arrow(f, b, side="h")
d.arrow(b, l, side="h")
px = [2.4, 6.6, 10.8, 15.0]
ps = [d.box(px[0], y2, "프로세스 1", sub="pretoken_chunk", w=3.2, h=1.2),
      d.box(px[1], y2, "프로세스 2", sub="pretoken_chunk", w=3.2, h=1.2),
      d.box(px[2], y2, "...", w=3.2, h=1.2),
      d.box(px[3], y2, "프로세스 8", sub="pretoken_chunk", w=3.2, h=1.2)]
bus_down(d, l, ps, (y1 - 0.6 + y2 + 0.6) / 2)
m = d.box(2.4, y3, "결과 합산", sub="pretoken_counts", w=3.2, h=1.2)
t = d.box(8.4, y3, "BPE 학습 루프", sub="(4.2와 동일, 단일 프로세스)", w=4.2, h=1.2)
r = d.box(14.2, y3, "merge_rules.pkl", w=3.2, h=1.0, fill=True)
bus_up(d, ps, m, (y2 - 0.6 + y3 + 0.6) / 2)
d.arrow(m, t, side="h")
d.arrow(t, r, side="h")
d.save(OUT / "fig04-d04.png")

# ---------------------------------------------------------------- d05: 고속 인코딩 루프
d = Diagram(17.6, 7.9)
y0, y1, y2 = 7.0, 4.2, 1.2
s = d.box(1.5, y1, "ids = 바이트열", w=2.7, h=0.9)
c = d.box(5.4, y1, "len(ids) > 1 ?", w=3.2, h=1.6, shape="diamond")
e = d.box(5.4, y0, "return ids", w=2.6, h=0.9)
p = d.box(9.6, y1, "count_pairs(ids)", sub="현재 있는 인접 쌍만", w=3.4, h=1.2)
q = d.box(14.5, y1, "min(쌍, key=merge_rules\n.get(쌍, inf))", sub="= 가장 먼저 학습된 쌍", w=5.0, h=1.6, fill=True)
r = d.box(14.5, y2, "규칙에 있음?", w=3.0, h=1.5, shape="diamond")
mm = d.box(9.6, y2, "merge(ids, best_pair, new_id)", w=4.6, h=0.9)
d.arrow(s, c, side="h")
d.arrow(c, p, side="h", label="예")
d.arrow(c, e, side="v", label="아니오")
d.arrow(p, q, side="h")
d.arrow(q, r, side="v")
d.arrow(r, mm, side="h", label="예")
d.arrow(mm.left, c.bottom, path=[(c.x, y2)])
d.arrow(r.right, e.right, path=[(17.4, y2), (17.4, y0)], label="아니오", label_pos=0.5)
d.save(OUT / "fig04-d05.png")

# ---------------------------------------------------------------- d06: 청크 병렬 인코딩 + memmap
d = Diagram(17.6, 11.0)
ya, yw, yc, ym = 9.9, 7.0, 4.6, 1.3
xs = [3.0, 8.8, 14.6]
f = d.box(3.0, ya, "train.txt", sub="2GB", w=2.8, h=1.8, shape="db")
b = d.box(8.8, ya, "① 파일 분할", sub="find_chunk_boundaries", w=3.8, h=1.2)
d.arrow(f, b, side="h")
ws = [d.box(xs[0], yw, "② 프로세스 1", sub="_encode_chunk", w=3.4, h=1.2),
      d.box(xs[1], yw, "② 프로세스 2", sub="_encode_chunk", w=3.4, h=1.2),
      d.box(xs[2], yw, "② 프로세스 8", sub="...", w=3.4, h=1.2)]
bus_down(d, b, ws, (ya - 0.6 + yw + 0.6) / 2)
cs = [d.box(x, yc, "③ bpe_cache/", sub=f"chunk_{n}.npy", w=3.4, h=1.2)
      for x, n in zip(xs, ["00000", "00001", "000NN"])]
for w_, c_ in zip(ws, cs):
    d.arrow(w_, c_, side="v")
mmp = d.box(8.8, ym, "④ np.memmap", sub="total_tokens 크기", w=3.4, h=1.2)
bus_up(d, cs, mmp, (yc - 0.6 + ym + 0.6) / 2)
o = d.box(14.6, ym, "train.bin", sub="uint16", w=2.8, h=1.8, shape="db", fill=True)
d.arrow(mmp, o, side="h")
dl = d.box(3.0, ym, "shutil.rmtree", sub="(bpe_cache)", w=3.0, h=1.2)
d.arrow(mmp, dl, side="h", dashed=True, label="끝나면")
d.save(OUT / "fig04-d06.png")
