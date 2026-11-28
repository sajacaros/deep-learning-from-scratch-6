"""노트용 '논문 스타일' 다이어그램 공용 모듈.

mermaid 대신 쓰는 흐름도·구조도를 모든 장에서 같은 모양으로 그리기 위한 도구다.

스타일 규칙
- 흰 바탕, 나눔명조(세리프), 검은 얇은 선(0.8pt), 각진 상자, 그림자 없음.
- 색은 아껴 쓴다: 기본 상자는 흰색, 강조할 상자만 연한 색(FILL) 하나.
- 화살표 위 설명(엣지 라벨)은 작은 회색 기울임꼴.
- 묶음(subgraph)은 점선 테두리 + 왼쪽 위 작은 대문자 라벨.

크기 규칙
- 좌표는 '단위(unit)' = 0.5인치. dpi=200으로 저장한다.
- 노트에는 표시 폭 = PNG 픽셀 폭 / 2 (최대 820)로 넣는다 → `display_width()`.
  이렇게 하면 본문 글자 크기(10pt)가 화면에서 약 14px, 작은 글자(8.5pt)가 약 12px가 된다.

사용 예
    from paperfig import Diagram
    d = Diagram(16, 4)                         # 가로 16단위, 세로 4단위
    a = d.box(2, 2, "소프트 딕셔너리", sub="유사도 → softmax → 가중합")
    b = d.box(7, 2, "행렬 수식", fill=True)
    d.arrow(a, b, label="포화 문제")
    d.save("docs/ch02/images/fig02-d01.png")    # → (픽셀 폭, 권장 표시 폭) 출력
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Ellipse
from PIL import Image

for _p in (Path.home() / ".local/share/fonts").glob("NanumMyeongjo*.ttf"):
    fm.fontManager.addfont(str(_p))
# ①②③ 같은 원문자는 나눔명조·DejaVu Serif에 없어 DejaVu Sans로 대체한다.
plt.rcParams["font.family"] = ["NanumMyeongjo", "DejaVu Serif", "DejaVu Sans"]
# 나눔명조는 < > 를 꺾쇠(〈 〉)처럼 그린다. < > 가 든 글자는 라틴 문자를 DejaVu Serif로,
# 한글은 나눔명조로 그린다(matplotlib 글리프 단위 대체).
_ANGLE_FAMILY = ["DejaVu Serif", "NanumMyeongjo", "DejaVu Sans"]


def _family(s, family=None):
    if family is not None:
        return family
    return _ANGLE_FAMILY if ("<" in s or ">" in s) else None
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK = "#1a1a1a"      # 선·본문
GRAY = "#6b6b6b"     # 보조 글자·엣지 라벨
LIGHT = "#bdbdbd"    # 점선 묶음 테두리
FILL = "#e8eef6"     # 강조 상자 (연한 청회색) — 한 그림에 한 종류만
FILL2 = "#f6ece1"    # 두 번째 강조가 꼭 필요할 때만 (연한 베이지)
ACCENT = "#9c2f2f"   # 아주 드물게: 핵심 한 곳의 글자·화살표

UNIT = 0.5           # 인치
DPI = 200
FS = 10              # 상자 제목
FS_SUB = 8.5         # 상자 부제·엣지 라벨·묶음 라벨


@dataclass
class Node:
    x: float
    y: float
    w: float
    h: float

    # 상자 변의 가운데 점
    @property
    def top(self): return (self.x, self.y + self.h / 2)
    @property
    def bottom(self): return (self.x, self.y - self.h / 2)
    @property
    def left(self): return (self.x - self.w / 2, self.y)
    @property
    def right(self): return (self.x + self.w / 2, self.y)


class Diagram:
    def __init__(self, width: float, height: float):
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width * UNIT, height * UNIT), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    # ------------------------------------------------------------ 상자
    def box(self, x, y, title, sub=None, w=None, h=None, fill=False, color=None,
            fs=FS, mono=False, shape="rect") -> Node:
        """가운데 (x, y)에 상자. fill=True면 FILL, 문자열이면 그 색. shape: rect | round | db | diamond."""
        lines = [title] + ([sub] if sub else [])
        if w is None:
            longest = max(len(s) for s in "\n".join(lines).split("\n"))
            w = max(2.2, 0.36 * longest * fs / 10 + 0.8)
        if h is None:
            n = title.count("\n") + 1 + (sub.count("\n") + 1 if sub else 0)
            h = 0.55 + 0.42 * n
        fc = FILL if fill is True else (fill if isinstance(fill, str) else "white")
        ec = INK
        if shape == "diamond":
            pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
            self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.8, zorder=3))
        elif shape == "db":
            e = 0.18
            self.ax.add_patch(Rectangle((x - w / 2, y - h / 2 + e), w, h - 2 * e, fc=fc, ec="none", zorder=3))
            self.ax.plot([x - w / 2, x - w / 2], [y - h / 2 + e, y + h / 2 - e], color=ec, lw=0.8, zorder=3)
            self.ax.plot([x + w / 2, x + w / 2], [y - h / 2 + e, y + h / 2 - e], color=ec, lw=0.8, zorder=3)
            self.ax.add_patch(Ellipse((x, y - h / 2 + e), w, 2 * e, fc=fc, ec=ec, lw=0.8, zorder=3))
            self.ax.add_patch(Ellipse((x, y + h / 2 - e), w, 2 * e, fc=fc, ec=ec, lw=0.8, zorder=4))
        else:
            style = "round,pad=0,rounding_size=0.12" if shape == "round" else "square,pad=0"
            self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                             fc=fc, ec=ec, lw=0.8, zorder=3))
        family = "DejaVu Sans Mono" if mono else None
        tcolor = color or INK
        sfam = _family(sub) if sub else None
        if sub:
            nt = title.count("\n") + 1
            ns = sub.count("\n") + 1
            total = 0.42 * (nt + ns)
            ty = y + total / 2 - 0.42 * nt / 2
            sy = y - total / 2 + 0.42 * ns / 2
            self.ax.text(x, ty, title, ha="center", va="center", fontsize=fs, color=tcolor,
                         zorder=5, family=_family(title, family), linespacing=1.25)
            self.ax.text(x, sy, sub, ha="center", va="center", fontsize=FS_SUB, color=GRAY,
                         zorder=5, family=sfam, linespacing=1.25)
        else:
            self.ax.text(x, y, title, ha="center", va="center", fontsize=fs, color=tcolor,
                         zorder=5, family=_family(title, family), linespacing=1.25)
        return Node(x, y, w, h)

    # ------------------------------------------------------------ 화살표
    def arrow(self, a, b, label=None, side=None, dashed=False, color=INK, path=None,
              label_pos=0.5, label_offset=(0, 0)):
        """a → b. a, b는 Node 또는 (x, y).
        side: 'h'(가로) | 'v'(세로) | None(자동) — Node끼리 이을 때 붙을 변을 고른다.
        path: 꺾이는 점 목록 [(x, y), ...] — 직각으로 돌아가는 화살표."""
        p1, p2 = self._ends(a, b, side, path)
        pts = [p1] + (path or []) + [p2]
        ls = (0, (4, 3)) if dashed else "solid"
        for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([x1, x2], [y1, y2], color=color, lw=0.8, ls=ls, zorder=2, solid_capstyle="butt")
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9,
                                          color=color, lw=0.8, linestyle=ls, zorder=2, shrinkA=0, shrinkB=0))
        if label:
            # 가장 긴 구간의 label_pos 지점에 라벨
            segs = list(zip(pts[:-1], pts[1:]))
            (x1, y1), (x2, y2) = max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
            lx = x1 + (x2 - x1) * label_pos + label_offset[0]
            ly = y1 + (y2 - y1) * label_pos + label_offset[1]
            horizontal = abs(y2 - y1) < 1e-6
            if horizontal:
                self.ax.text(lx, ly + 0.12, label, ha="center", va="bottom", fontsize=FS_SUB,
                             color=GRAY, style="italic", zorder=5, linespacing=1.2, family=_family(label))
            else:
                self.ax.text(lx + 0.15, ly, label, ha="left", va="center", fontsize=FS_SUB,
                             color=GRAY, style="italic", zorder=5, linespacing=1.2, family=_family(label))

    def line(self, pts, dashed=False, color=INK):
        ls = (0, (4, 3)) if dashed else "solid"
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=0.8, ls=ls, zorder=2)

    def _ends(self, a, b, side, path):
        if not isinstance(a, Node) and not isinstance(b, Node):
            return a, b
        ta = path[0] if path else (b.x, b.y) if isinstance(b, Node) else b
        tb = path[-1] if path else (a.x, a.y) if isinstance(a, Node) else a
        return self._anchor(a, ta, side), self._anchor(b, tb, side)

    @staticmethod
    def _anchor(n, toward, side):
        if not isinstance(n, Node):
            return n
        dx, dy = toward[0] - n.x, toward[1] - n.y
        if side is None:
            side = "h" if abs(dx) / max(n.w, 1e-6) >= abs(dy) / max(n.h, 1e-6) else "v"
        if side == "h":
            return n.right if dx > 0 else n.left
        return n.top if dy > 0 else n.bottom

    # ------------------------------------------------------------ 묶음·글자
    def group(self, x0, y0, x1, y1, label=None):
        """점선 묶음(subgraph). (x0, y0) 왼쪽 아래, (x1, y1) 오른쪽 위."""
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=LIGHT, lw=0.8,
                                    ls=(0, (4, 3)), zorder=1))
        if label:
            self.ax.text(x0 + 0.15, y1 - 0.15, label, ha="left", va="top", fontsize=FS_SUB,
                         color=GRAY, zorder=5)

    def text(self, x, y, s, fs=FS_SUB, color=GRAY, ha="center", va="center", **kw):
        kw["family"] = _family(s, kw.get("family"))
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=5, **kw)

    def caption_label(self, x, y, s):
        """그림 안 패널 이름: (a), (b) 같은 것."""
        self.ax.text(x, y, s, ha="left", va="top", fontsize=FS, color=INK, fontweight="bold", zorder=5)

    # ------------------------------------------------------------ 저장
    def save(self, path):
        """PNG로 저장하고 256색 팔레트로 줄인다. (픽셀 폭, 권장 표시 폭)을 돌려준다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        self.fig.savefig(buf, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.08)
        plt.close(self.fig)
        im = Image.open(buf).convert("RGB")
        q = im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q.save(path, optimize=True)
        w = im.size[0]
        print(f"{path}  {im.size[0]}x{im.size[1]}  표시 폭 {display_width(w)}")
        return w, display_width(w)


def display_width(pixel_width: int) -> int:
    """노트의 <img width>: 픽셀 폭의 절반, 최대 820."""
    return min(820, round(pixel_width / 2))
