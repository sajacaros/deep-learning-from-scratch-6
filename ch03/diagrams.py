# 3장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch03/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch03/images"

# ---------------------------------------------------------------- d01: 장 전체 흐름
d = Diagram(17.6, 5.9)
W, H = 3.45, 1.75
y1, y2 = 4.25, 1.05
xs = [1.85 + i * 4.6 for i in range(4)]
row1 = [
    ("tiny_codes_2000.bin", "토큰 ID 약 231만 개", False),
    ("3.1 사전 학습", "다음 토큰 예측\n20,000회", True),
    ("model_pretrain_2000.pt", "텍스트 이어 쓰기", False),
    ("3.3 SFT", "알파카 포맷 + 마스크\n500회", True),
]
row2 = [
    ("model_sft_2000.pt", "대화 가능", False),
    ("3.5~3.8 이론", "정책 경사 → 베이스라인\n→ 중요도 샘플링 → 클리핑", False),
    ("3.9 GRPO", "그룹 평균 = 베이스라인\n500회", True),
    ("model_grpo_2000.pt", "덧셈 정답률 96%", False),
]
n1 = [d.box(x, y1, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row1)]
n2 = [d.box(x, y2, t, sub=s, w=W, h=H, fill=f) for x, (t, s, f) in zip(xs, row2)]
labels1 = [None, None, "지시를 못 따름"]
for a, b, lab in zip(n1, n1[1:], labels1):
    d.arrow(a, b, side="h")
    if lab:
        d.text((a.x + b.x) / 2, y1 + H / 2 + 0.25, lab, style="italic")
# 1행 끝 → 2행 처음 (꺾인 화살표)
ymid = (y1 + y2) / 2
e, f = n1[-1], n2[0]
d.arrow(e, f, path=[(e.x, ymid), (f.x, ymid)], side="v")
labels2 = ["덧셈 정답률 절반 정도", None, None]
for a, b, lab in zip(n2, n2[1:], labels2):
    d.arrow(a, b, side="h")
    if lab:
        d.text((a.x + b.x) / 2, y2 - H / 2 - 0.25, lab, style="italic")
d.save(OUT / "fig03-d01.png")
