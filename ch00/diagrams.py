# 0장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch00/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch00/images"

# ---------------------------------------------------------------- d01: LLM이 만들어지기까지
d = Diagram(17.6, 6.2)
W, H = 3.3, 1.45
y1, y2 = 4.35, 1.05
xs = [1.85 + i * 4.6 for i in range(4)]
row1 = [
    ("대량의 텍스트", None, False),
    ("토크나이저 학습", "BPE", False),
    ("사전 학습", "다음 토큰 예측", False),
    ("베이스 모델", "텍스트 이어 쓰기", True),
]
row2 = [
    ("SFT", "답변 방식 학습", False),
    ("강화 학습", "RLHF / 검증 가능한 보상", False),
    ("AI 비서 / 추론 모델", None, True),
]
n1 = [d.box(x, y1, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row1)]
n2 = [d.box(x, y2, t, sub=s, w=W + (0.6 if i == 1 else 0), h=H, fill=f)
      for i, (x, (t, s, f)) in enumerate(zip(xs, row2))]
for a, b in zip(n1, n1[1:]):
    d.arrow(a, b, side="h")
ymid = (y1 + y2) / 2
e, f = n1[-1], n2[0]
d.arrow(e, f, path=[(e.x, ymid), (f.x, ymid)], side="v")
for a, b in zip(n2, n2[1:]):
    d.arrow(a, b, side="h")
# 묶음: 사전 학습까지 / 사후 학습
d.group(xs[2] - W / 2 - 0.3, y1 - H / 2 - 0.3, xs[3] + W / 2 + 0.3, y1 + H / 2 + 0.75, "사전 학습 (2절)")
d.group(xs[0] - W / 2 - 0.3, y2 - H / 2 - 0.3, xs[1] + (W + 0.6) / 2 + 0.3, y2 + H / 2 + 0.75, "사후 학습 (3절)")
d.save(OUT / "fig00-d01.png")
