# 6장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch06/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch06/images"

# ---------------------------------------------------------------- d01: 장 전체 흐름
d = Diagram(17.6, 6.4)
W, H = 3.5, 1.45
y1, y2 = 4.45, 1.05
# 1행: 학습 기법 3개(묶음) → 사전 학습(개선)
xa = [2.15 + i * 3.85 for i in range(3)]
opt = [d.box(x, y1, t, sub=s, w=W, h=H) for x, (t, s) in zip(xa, [
    ("AdamW", "적응적 lr + 분리된 감쇠"),
    ("학습률 스케줄", "워밍업 → D2Z"),
    ("혼합 정밀도", "BF16 + autocast"),
])]
d.group(0.2, y1 - H / 2 - 0.35, xa[-1] + W / 2 + 0.2, y1 + H / 2 + 0.75, "학습 기법")
pre = d.box(15.05, y1, "사전 학습 (개선)", sub="memmap 데이터\n기울기 클리핑 · 검증 손실", w=4.2, h=1.85, fill=True)
d.arrow((xa[-1] + W / 2 + 0.2, y1), pre, side="h")
# 2행: 텍스트 생성 → DPO 이론 → DPO 구현
xb = [2.6, 8.8, 15.05]
e = d.box(xb[0], y2, "텍스트 생성 +\nLLM-as-a-Judge", w=4.2, h=H)
f = d.box(xb[1], y2, "RLHF → DPO 이론", sub="보상 모델 없이 선호 학습", w=4.6, h=H)
g = d.box(xb[2], y2, "DPO 구현", sub="해피엔딩 StoryBot", w=4.2, h=H, fill=True)
ymid = (y1 - 1.85 / 2 + y2 + H / 2) / 2
d.arrow(pre, e, path=[(pre.x, ymid), (e.x, ymid)], side="v")
d.arrow(e, f, side="h")
d.arrow(f, g, side="h")
d.save(OUT / "fig06-d01.png")
