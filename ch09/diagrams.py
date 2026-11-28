# 9장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch09/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch09/images"

# ---------------------------------------------------------------- d01: 9장 전체 흐름
# 네 갈래(GPU 1장 절약, 여러 GPU, Muon, W&B)가 WebBot 사전 학습으로 모이고, 생성 결과로 이어진다.
d = Diagram(17.6, 8.6)
BW, BH = 3.9, 1.3
GX0, GX1 = 0.2, 9.0
c1, c2 = 2.45, 6.75                      # 묶음 안 두 상자의 x
# 묶음 1: GPU 1장의 메모리 절약
d.group(GX0, 6.1, GX1, 8.4, label="GPU 1장의 메모리 절약")
a = d.box(c1, 7.05, "기울기 누적", sub="작은 배치 × n번 = 큰 배치", w=BW, h=BH)
b = d.box(c2, 7.05, "기울기 체크포인트", sub="중간 출력 대신 재계산", w=BW, h=BH)
# 묶음 2: 여러 GPU 활용
d.group(GX0, 3.5, GX1, 5.8, label="여러 GPU 활용")
c = d.box(c1, 4.45, "데이터 병렬 (DDP)", sub="All-Reduce로 기울기 공유", w=BW, h=BH)
dd = d.box(c2, 4.45, "ZeRO", sub="중복 저장 제거", w=BW, h=BH)
e = d.box(c1, 2.15, "Muon", sub="행렬 단위 직교화 갱신", w=BW, h=BH)
f = d.box(c1, 0.7, "W&B", sub="학습 기록·시각화", w=BW, h=1.15)

g = d.box(12.0, 4.3, "WebBot 사전 학습", sub="150M, 10만 스텝", w=3.6, h=1.4, fill=True)
h = d.box(15.75, 4.3, "생성 결과", sub="잘하는 것 / 못하는 것", w=3.3, h=1.4)
JX = 9.6                                 # 네 갈래가 모이는 세로선
srcs = [(GX1, 7.25), (GX1, 4.65), e.right, f.right]
for x, y in srcs:
    d.line([(x, y), (JX, y)])
d.line([(JX, srcs[0][1]), (JX, srcs[-1][1])])
d.arrow((JX, g.y), g.left)
d.arrow(g, h, side="h")
d.save(OUT / "fig09-d01.png")
