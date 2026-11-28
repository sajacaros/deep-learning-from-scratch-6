# 5장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch05/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch05/images"

# ---------------------------------------------------------------- d01: 장 전체 흐름
d = Diagram(17.6, 5.1)
W, H = 3.5, 1.45
y1, y2 = 4.0, 1.0
xs = [1.85 + i * 4.62 for i in range(4)]
row1 = [("GPT-2", "(2장 · codebot/model.py)", False),
        ("5.1 RoPE", "상대 위치를 회전으로", False),
        ("5.2 SwiGLU", "게이트가 있는 FFN", False),
        ("5.3 RMSNorm +", "드롭아웃·편향·\n가중치 공유 제거", False)]
row2 = [("5.4 개선된 GPT", "ch05/04_improved_gpt.py", True),
        ("5.5 KV 캐시", "생성 시 중복 계산 제거", False),
        ("storybot/model.py", "→ 6장 StoryBot 학습", True)]
n1 = [d.box(x, y1, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row1)]
n2 = [d.box(x, y2, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row2)]
for a, b in zip(n1, n1[1:]):
    d.arrow(a, b, side="h")
ymid = (y1 + y2) / 2
e, f = n1[-1], n2[0]
d.arrow(e, f, path=[(e.x, ymid), (f.x, ymid)], side="v")
for a, b in zip(n2, n2[1:]):
    d.arrow(a, b, side="h")
d.save(OUT / "fig05-d01.png")
