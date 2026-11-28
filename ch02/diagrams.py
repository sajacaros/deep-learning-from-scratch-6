# 2장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch02/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch02/images"

# ---------------------------------------------------------------- d01: 장 전체 흐름
d = Diagram(17.6, 5.6)
W, H = 2.75, 1.45
y1, y2 = 4.15, 1.05
xs = [1.5 + i * 3.65 for i in range(5)]
row1 = [
    ("소프트 딕셔너리", "유사도 → softmax\n→ 가중합", False),
    ("행렬 수식", r"softmax$(QK^{\top})V$", False),
    ("스케일링", r"$\div\sqrt{d}$", False),
    ("트랜스포머 구조", "Embed → 블록×N\n→ Linear", False),
    ("위치 임베딩", "학습 가능한 벡터", False),
]
row2 = [
    ("인과 마스크", r"$-\infty$ + softmax", True),
    (r"$W_v$ / $W_o$ 분리", "저랭크 분해", False),
    ("멀티헤드", "(B, H, C, D)", False),
    ("트랜스포머 블록", "LayerNorm · FFN\nGELU · 잔차", False),
    ("GPT-2", "가중치 공유 · 초기화", True),
]
n1 = [d.box(x, y1, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row1)]
n2 = [d.box(x, y2, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row2)]
labels1 = [None, "포화 문제", None, "어순을 모름"]
for a, b, lab in zip(n1, n1[1:], labels1):
    d.arrow(a, b, side="h")
    if lab:
        d.text((a.x + b.x) / 2, y1 + H / 2 + 0.25, lab, style="italic")
# 1행 끝 → 2행 처음 (꺾인 화살표)
ymid = (y1 + y2) / 2
e, f = n1[-1], n2[0]
d.arrow(e, f, path=[(e.x, ymid), (f.x, ymid)], side="v")
d.text((e.x + f.x) / 2, ymid + 0.22, "미래 토큰을 봄", style="italic")
labels2 = [r"$W_v$가 너무 큼", None, None, None]
for a, b, lab in zip(n2, n2[1:], labels2):
    d.arrow(a, b, side="h")
    if lab:
        d.text((a.x + b.x) / 2, y2 - H / 2 - 0.25, lab, style="italic")
d.save(OUT / "fig02-d01.png")
