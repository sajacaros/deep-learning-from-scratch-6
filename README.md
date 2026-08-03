『밑바닥부터 시작하는 딥러닝 ❻』 <br>: 토크나이저부터 추론 모델까지 직접 만드는 LLM
=============================

<!-- [<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-5/images/cover.png" width="200px">](https://www.amazon.co.jp/dp/4814400594/) -->


Coming soon


## 파일 구성

|폴더명 |설명                             |
|:--        |:--                              |
|`ch01`〜`ch09`|각 장에서 사용하는 코드|
|`codebot`   |CodeBot에서 사용하는 코드와 데이터 |
|`storybot`   |StoryBot에서 사용하는 코드와 데이터 |
|`webbot`   |WebBot에서 사용하는 코드와 데이터 |
<!-- |`notebooks`   |1章〜6章までのコード（Jupyter Notebook形式）| -->

## 파이썬과 외부 라이브러리

소스 코드를 실행하려면 다음 라이브러리가 필요합니다.

* NumPy
* Matplotlib
* PyTorch 2.x
* tqdm

※ 파이썬은 3.10 이상을 사용합니다.


## 환경 구축

### uv를 사용하는 경우

[uv](https://docs.astral.sh/uv/)는 빠른 파이썬 패키지 관리자입니다.`uv.lock`을 사용하면 운영체제와 관계없이 동일한 환경을 재현할 수 있습니다.
```bash
# uv 설치(설치되어 있지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Mac/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

실행할 때는 `uv run`을 사용합니다. 필요한 환경도 자동으로 구축됩니다.
```bash
uv run python ch01/01_char_tokenizer.py
```

### pip를 사용하는 경우
```bash
pip install -r requirements.txt
python ch01/01_char_tokenizer.py
```


## 실행 방법

각 장의 폴더로 이동하여 실행하거나, 상위 폴더에서 실행합니다.
```bash
# 각 장의 폴더에서 실행
$ cd ch01
$ python 01_char_tokenizer.py

# 상위 폴더에서 실행
$ python ch02/10_gpt.py
```

uv를 사용하는 경우에는 `python`을 `uv run python`으로 바꾸어 실행하세요.

## 주피터 노트북으로 실습하기

`ch01`~`ch09`의 각 `.py` 파일은 같은 이름의 `.ipynb` 노트북으로도 준비돼 있습니다.
한 셀씩 직접 쳐보며 따라 하고 싶다면 노트북 쪽을 사용하세요.

```bash
uv run jupyter lab   # 또는 .venv/bin/jupyter lab
```

노트북은 `tools/py2ipynb.py`가 `.py`에서 자동 생성한 것이라 코드는 원본과 같습니다.
`__file__`을 쓰는 맨 앞의 경로 설정 부분만 노트북용으로 바뀌어 있어서, 어느 장 폴더에서
열든 저장소 루트를 기준으로 동작합니다.

원본 `.py`가 갱신되면 아래 명령으로 노트북을 다시 만들 수 있습니다.

```bash
uv run python tools/py2ipynb.py --force   # 전체 다시 생성
uv run python tools/py2ipynb.py ch01      # 특정 장만 (기존 노트북은 건너뜀)
uv run python tools/verify_ipynb.py       # 노트북 코드가 .py와 같은지 검증
```

## 라이선스

이 리포지터리의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)로 제공됩니다. 상업적·비상업적 용도와 관계없이 자유롭게 이용할 수 있습니다.