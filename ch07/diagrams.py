# 7장 노트의 다이어그램(논문 스타일). mermaid를 대신한다.
# 실행(저장소 루트에서): .venv/bin/python ch07/diagrams.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools/paperfig"))
from paperfig import Diagram

OUT = ROOT / "docs/ch07/images"

# ---------------------------------------------------------------- d01: 7장 전체 흐름
# 7.1에서 네 갈래: 7.2 → 7.3(고속화), 7.4, 7.5, 7.6(더 넓은 주제)
d = Diagram(17.6, 7.4)
BW, BH = 3.9, 1.3
ys = [6.6, 4.75, 2.9, 1.05]
a = d.box(2.2, (ys[0] + ys[-1]) / 2, "7.1 WebBot 토크나이저", sub="storybot 코드 재사용\nvocab 50,000",
          w=4.0, h=1.7, fill=True)
TX = 5.0                       # 갈래가 나뉘는 세로선
BX = 9.6
b = d.box(BX, ys[0], "7.2 병목 찾기", sub="cProfile", w=BW, h=BH)
c = d.box(15.2, ys[0], "7.3 러스트·C++로 병목 재작성", sub="PyO3 / pybind11", w=4.6, h=BH)
dd = d.box(BX, ys[1], "7.4 특수 토큰 추가", sub="하모니 포맷 등", w=BW, h=BH)
e = d.box(BX, ys[2], "7.5 다른 토크나이저", sub="SentencePiece / WordPiece", w=BW, h=BH)
f = d.box(BX, ys[3], "7.6 토크나이저 없는 접근", sub="바이트 수준 모델", w=BW, h=BH)
d.line([a.right, (TX, a.y)])
d.line([(TX, ys[0]), (TX, ys[-1])])
for n in (b, dd, e, f):
    d.arrow((TX, n.y), n.left)
d.text((TX + b.x - BW / 2) / 2, ys[0] + 0.42, "12GB는\n여전히 오래 걸림", style="italic", va="bottom",
       linespacing=1.2)
d.arrow(b, c, side="h")
d.save(OUT / "fig07-d01.png")

# ---------------------------------------------------------------- d02: 7.3 확장 모듈
d = Diagram(15.0, 3.6)
p = d.box(2.2, 1.8, "파이썬 코드", sub="(전체 흐름)", w=3.6, h=1.3)
m = d.box(7.6, 1.8, "확장 모듈", sub="(병목 함수만)", w=3.6, h=1.3, fill=True)
r = d.box(12.8, 2.8, "러스트 + PyO3", w=3.6, h=0.9)
cc = d.box(12.8, 0.8, "C++ + pybind11", w=3.6, h=0.9)
d.arrow(p, m, side="h", label="import")
JX = 10.25
d.line([m.right, (JX, m.y)])
d.line([(JX, r.y), (JX, cc.y)])
d.line([(JX, r.y), r.left])
d.line([(JX, cc.y), cc.left])
d.save(OUT / "fig07-d02.png")

# ---------------------------------------------------------------- d03: 7.6 2단계 파이프라인 vs 종단간
d = Diagram(17.6, 6.2)
d.group(0.2, 3.3, 17.4, 6.0, label="현재 주류: 2단계 파이프라인")
t1 = d.box(2.8, 4.45, "텍스트", w=3.2, h=1.1)
k1 = d.box(8.8, 4.45, "토크나이저", sub="(고정 규칙, 따로 학습)", w=4.4, h=1.3)
m1 = d.box(14.6, 4.45, "모델", w=3.2, h=1.1)
d.arrow(t1, k1, side="h")
d.arrow(k1, m1, side="h")
d.group(0.2, 0.2, 17.4, 2.9, label="연구 중: 종단간(end-to-end)")
t2 = d.box(2.8, 1.35, "텍스트", sub="(바이트)", w=3.2, h=1.3)
m2 = d.box(14.6, 1.35, "모델", sub="(분할 방식도 함께 학습)", w=4.4, h=1.3, fill=True)
d.arrow(t2, m2, side="h")
d.save(OUT / "fig07-d03.png")
