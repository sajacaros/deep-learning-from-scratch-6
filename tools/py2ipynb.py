#!/usr/bin/env python3
"""dlfs6 실습용 .py 파일을 주피터 노트북(.ipynb)으로 변환한다.

- 코드는 그대로 옮긴다. 빈 줄로 나뉜 최상위 블록 단위로 셀을 쪼개되,
  함수·클래스·for·with 같은 복합문 내부에서는 절대 쪼개지 않는다.
- 노트북에는 __file__이 없으므로 맨 앞의 경로 설정 부분만 노트북용으로 바꾼다.

사용법:
    .venv/bin/python tools/py2ipynb.py             # ch01~ch09 전체 변환
    .venv/bin/python tools/py2ipynb.py ch01 ch02   # 특정 장만 변환
    .venv/bin/python tools/py2ipynb.py --force     # 기존 노트북 덮어쓰기
"""

import argparse
import ast
import os
import re
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# __file__을 쓰는 원본 헤더를 대신할 노트북용 경로 설정 셀
PATH_SETUP = """import os, sys

# 노트북에는 __file__이 없으므로 pyproject.toml이 있는 폴더(저장소 루트)를 찾아 이동한다
_dir = os.path.abspath('.')
while not os.path.exists(os.path.join(_dir, 'pyproject.toml')) and _dir != os.path.dirname(_dir):
    _dir = os.path.dirname(_dir)
os.chdir(_dir)
if '.' not in sys.path:
    sys.path.append('.')
print('작업 폴더:', os.getcwd())"""

DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def find_path_header(nodes, lines):
    """맨 앞의 경로 설정 블록이 차지하는 노드 구간 [시작, 끝]을 반환한다."""
    chdir_at = None
    for i, node in enumerate(nodes[:6]):  # 헤더는 항상 파일 맨 앞에 있다
        src = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        if 'os.chdir' in src and '__file__' in src:
            chdir_at = i
            break
    if chdir_at is None:
        return None

    # 앞쪽: import os / import sys / import os, sys 만 흡수
    start = chdir_at
    while start > 0 and isinstance(nodes[start - 1], ast.Import):
        names = {a.name for a in nodes[start - 1].names}
        if not names <= {'os', 'sys'}:
            break
        start -= 1

    # 뒤쪽: 바로 뒤따르는 sys.path.append(...) 흡수
    end = chdir_at
    while end + 1 < len(nodes):
        src = '\n'.join(lines[nodes[end + 1].lineno - 1:nodes[end + 1].end_lineno])
        if 'sys.path' not in src:
            break
        end += 1

    return start, end


def node_start(node):
    """데코레이터까지 포함한 노드의 시작줄."""
    start = node.lineno
    for dec in getattr(node, 'decorator_list', []):
        start = min(start, dec.lineno)
    return start


def to_cells(source):
    """소스를 셀 문자열 리스트로 쪼갠다.

    최상위에서 빈 줄을 만나면 셀을 나눈다. 함수·클래스·for 문 등 복합문
    '안'의 빈 줄은 분할점으로 보지 않으므로 코드가 잘리지 않는다.
    주석만 있는 줄도 한 줄도 빠뜨리지 않고 그대로 옮긴다.
    """
    lines = source.splitlines()
    n = len(lines)
    nodes = ast.parse(source).body
    if not nodes:
        return []

    cells = []
    skip_until = 0  # 경로 설정 헤더로 대체돼 건너뛸 마지막 줄 (1-based)
    header = find_path_header(nodes, lines)
    if header:
        cells.append(PATH_SETUP)
        skip_until = nodes[header[1]].end_lineno

    # 최상위 문장이 차지하는 줄 표시 — 이 안에서는 절대 쪼개지 않는다
    inside = [False] * (n + 2)
    def_starts, def_ends = set(), set()
    for node in nodes:
        start = node_start(node)
        for i in range(start, node.end_lineno + 1):
            inside[i] = True
        if isinstance(node, DEF_TYPES):
            def_starts.add(start)
            def_ends.add(node.end_lineno)

    cur = []

    def flush(carry=0):
        """현재까지 모은 줄을 셀로 확정한다. carry개의 끝줄은 다음 셀로 넘긴다."""
        nonlocal cur
        cut = len(cur) - carry
        text = '\n'.join(cur[:cut]).strip('\n')
        if text.strip():
            cells.append(text)
        cur = cur[cut:]

    prev_was_def_end = False
    for i in range(skip_until + 1, n + 1):
        line = lines[i - 1]
        if not line.strip() and not inside[i]:
            flush()
            prev_was_def_end = False
            continue
        if cur and (i in def_starts or prev_was_def_end):
            # 함수·클래스는 앞뒤로 셀을 나눈다. 바로 위 주석은 함수 쪽에 딸려 보낸다
            carry = 0
            if i in def_starts:
                while carry < len(cur) and cur[-1 - carry].lstrip().startswith('#'):
                    carry += 1
            flush(carry)
        cur.append(line)
        prev_was_def_end = i in def_ends
    flush()

    return cells


def build_notebook(py_path, rel_path):
    with open(py_path, encoding='utf-8') as f:
        source = f.read()

    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3',
    }
    nb.metadata['language_info'] = {'name': 'python'}

    title = os.path.basename(py_path)[:-3]
    nb.cells.append(nbf.v4.new_markdown_cell(
        f'# {title}\n\n'
        f'『밑바닥부터 시작하는 딥러닝 ❻』 실습 코드 — 원본: `{rel_path}`\n\n'
        f'셀을 위에서부터 차례대로 실행하세요.'
    ))
    for cell in to_cells(source):
        nb.cells.append(nbf.v4.new_code_cell(cell))
    return nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('chapters', nargs='*', help='변환할 장 폴더 (기본: ch로 시작하는 전체)')
    parser.add_argument('--force', action='store_true', help='기존 노트북 덮어쓰기')
    args = parser.parse_args()

    chapters = args.chapters or sorted(
        d for d in os.listdir(ROOT) if re.fullmatch(r'ch\d+', d)
        and os.path.isdir(os.path.join(ROOT, d))
    )

    made = skipped = 0
    for ch in chapters:
        ch_dir = os.path.join(ROOT, ch)
        for name in sorted(os.listdir(ch_dir)):
            if not name.endswith('.py'):
                continue
            py_path = os.path.join(ch_dir, name)
            ipynb_path = py_path[:-3] + '.ipynb'
            rel_path = os.path.relpath(py_path, ROOT)

            if os.path.exists(ipynb_path) and not args.force:
                print(f'  건너뜀 (이미 있음): {os.path.relpath(ipynb_path, ROOT)}')
                skipped += 1
                continue

            nb = build_notebook(py_path, rel_path)
            with open(ipynb_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            print(f'  {rel_path} -> {os.path.relpath(ipynb_path, ROOT)} '
                  f'({len(nb.cells) - 1}개 코드 셀)')
            made += 1

    print(f'\n완료: {made}개 생성, {skipped}개 건너뜀')


if __name__ == '__main__':
    main()
