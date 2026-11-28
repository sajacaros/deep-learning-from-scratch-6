# 1장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch01/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch01/images"

# ---------------------------------------------------------------- d01: 토크나이저의 발전
d = Diagram(17.6, 5.3)
W, H = 3.6, 1.45
y1, y2 = 4.25, 1.05
xs = [2.0, 8.8, 15.6]
row1 = [
    ("문자 단위", "ord / chr", False),
    ("바이트 단위", "UTF-8", False),
    ("BPE", "자주 나오는 쌍 병합", True),
]
row2 = [
    ("+ 특수 토큰", " ", False),   # 부제는 아래에서 고정폭 글꼴로 따로 쓴다
    ("+ 사전 토큰화", "GPT-2 정규식", False),
    ("TinyCodes 학습", "vocab = 1,000", True),
]
n1 = [d.box(x, y1, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row1)]
n2 = [d.box(x, y2, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row2)]
# 나눔명조는 < > 를 꺾쇠(〈 〉)처럼 그리므로 특수 토큰 문자열은 고정폭 글꼴로
d.text(xs[0], y2 - 0.21, "<|endoftext|>", family="DejaVu Sans Mono")
lab1 = ["어휘 약 15만 개\n너무 큼", "토큰열이\n너무 김"]
for a, b, lab in zip(n1, n1[1:], lab1):
    d.arrow(a, b, side="h")
    d.text((a.x + b.x) / 2, y1 + 0.2, lab, style="italic", va="bottom")
ymid = (y1 + y2) / 2
e, f = n1[-1], n2[0]
d.arrow(e, f, path=[(e.x, ymid), (f.x, ymid)], side="v")
d.text((e.x + f.x) / 2, ymid + 0.22, "문서 경계 문제", style="italic", va="bottom")
lab2 = ["hello! / hello?\n따로 학습됨", None]
for a, b, lab in zip(n2, n2[1:], lab2):
    d.arrow(a, b, side="h")
    if lab:
        d.text((a.x + b.x) / 2, y2 + 0.2, lab, style="italic", va="bottom")
d.save(OUT / "fig01-d01.png")
