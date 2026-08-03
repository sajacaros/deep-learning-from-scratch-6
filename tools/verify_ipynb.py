#!/usr/bin/env python3
"""변환된 노트북의 코드가 원본 .py와 동일한지 검증한다.

경로 설정 헤더(__file__ 사용부 / PATH_SETUP)를 제외한 나머지 코드 줄이
순서까지 그대로인지 비교한다. 빈 줄만 무시한다.
"""
import ast
import os
import re
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from py2ipynb import find_path_header  # noqa: E402


def py_body_lines(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    lines = src.splitlines()
    nodes = ast.parse(src).body
    if not nodes:
        return []
    header = find_path_header(nodes, lines)
    start = 0 if header is None else nodes[header[1]].end_lineno
    return [l for l in lines[start:] if l.strip()]


def nb_body_lines(path):
    nb = nbf.read(path, as_version=4)
    out = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        if cell.source.startswith('import os, sys\n\n# 노트북에는 __file__이 없으므로'):
            continue
        out.extend(l for l in cell.source.splitlines() if l.strip())
    return out


bad = 0
total = 0
for ch in sorted(d for d in os.listdir(ROOT)
                 if re.fullmatch(r'ch\d+', d) and os.path.isdir(os.path.join(ROOT, d))):
    for name in sorted(os.listdir(os.path.join(ROOT, ch))):
        if not name.endswith('.ipynb'):
            continue
        nb_path = os.path.join(ROOT, ch, name)
        py_path = nb_path[:-6] + '.py'
        if not os.path.exists(py_path):
            continue
        total += 1
        a, b = py_body_lines(py_path), nb_body_lines(nb_path)
        if a != b:
            bad += 1
            print(f'✗ {ch}/{name}: 원본 {len(a)}줄 vs 노트북 {len(b)}줄')
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f'   첫 불일치 {i}행:\n     py: {x!r}\n     nb: {y!r}')
                    break

print(f'\n검증: {total}개 중 {total - bad}개 일치, {bad}개 불일치')
sys.exit(1 if bad else 0)
