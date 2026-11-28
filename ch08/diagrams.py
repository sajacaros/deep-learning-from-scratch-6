# 8장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch08/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch08/images"

# ---------------------------------------------------------------- d01: 8장 전체 흐름
# 본선: 5장 GPT → 8.1 → 8.2 → 8.3 → webbot/model.py, 곁가지(점선): 8.1 → 8.4~8.6 → model.py
d = Diagram(17.6, 5.2)
W, H = 3.0, 1.55
y1 = 3.9
xs = [1.6 + i * 3.6 for i in range(5)]
a = d.box(xs[0], y1, "5장 개선된 GPT", sub="(StoryBot, 약 20M)", w=W, h=H)
b = d.box(xs[1], y1, "8.1 모델 크기 선택", sub="친칠라:\n토큰 ≈ 20 × 파라미터", w=W, h=H)
c = d.box(xs[2], y1, "8.2 Flash Attention", sub="SRAM 블록 처리", w=W, h=H)
dd = d.box(xs[3], y1, "8.3 GQA", sub="쿼리 헤드가\nKV 헤드 공유", w=W, h=H)
e = d.box(xs[4], y1, "webbot/model.py", sub="(약 150M)", w=W, h=H, fill=True)
for p, q in zip([a, b, c, dd], [b, c, dd, e]):
    d.arrow(p, q, side="h")
f = d.box(xs[2], 1.0, "8.4 MoE · 8.5 torch.compile · 8.6 PI · YaRN", w=7.4, h=0.95)
d.arrow(b, f, path=[(b.x, f.y)], dashed=True)
d.arrow(f, e, path=[(e.x, f.y)], dashed=True)
d.text((f.x + f.w / 2 + e.x) / 2, f.y + 0.3, "여유가 있으면 도전", style="italic")
d.save(OUT / "fig08-d01.png")
