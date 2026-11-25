# note-html — 노트를 단독 배포용 HTML로

`docs/chXX/README.md` 노트를 **파일 하나만으로 동작하는 HTML**(`docs/chXX/chXX.html`)로 만든다.
수식, mermaid 다이어그램, 그림을 모두 파일 안에 넣기 때문에 인터넷 연결 없이 열 수 있고,
HTML 파일 하나만 따로 보내도 그대로 보인다.

## 처음 한 번

```bash
cd tools/note-html
npm install
npx playwright install chromium   # mermaid 렌더링과 검사에 쓰는 헤드리스 브라우저
```

## 만들기와 검사

```bash
node build.cjs ch02            # → docs/ch02/ch02.html
node build.cjs ch02 ch03       # 여러 챕터를 한 번에
node verify.cjs ch02           # 단독 배포 검사
node verify.cjs ch02 --shots   # 장마다 전체 화면 스크린샷도 남김 (shots/, git 제외)
node build.cjs ch04 --out-dir /tmp/x && node verify.cjs /tmp/x/ch04.html   # docs/를 건드리지 않고 시험해 보기
```

노트(README.md)를 고친 뒤에는 `build.cjs`를 다시 실행하면 된다. HTML은 직접 고치지 않는다.

`verify.cjs`는 HTML을 빈 임시 폴더에 하나만 복사하고 **네트워크를 모두 막은 상태**로 연 뒤 다음을 확인한다.
하나라도 걸리면 `FAIL`을 출력하고 종료 코드 1을 돌려준다.

- 깨진 그림, 차단된 외부 요청, 스크립트 오류가 없는지
- 수식이나 마크다운 기호(`$`, `**`, `\frac` 등)가 글자로 남지 않았는지 (코드 블록은 제외)
- 코드 블록에 코드 상자 스타일이 붙었는지
- 400px 폭(모바일)에서 페이지가 가로로 넘치지 않는지

## 변환 규칙

| 노트(README.md) | HTML |
|---|---|
| `## 제목` | 장 하나(한 화면). 왼쪽 차례·상단 레일에 표시 |
| `## 2.2 제목 — \`ch02/02_x.py\`` | 장 제목은 "2.2 제목", 코드 파일은 장 첫 줄 "코드: …"로 분리 |
| `$…$`, `` $`…`$ ``, `$$ … $$` | MathJax로 미리 렌더한 SVG |
| ` ```mermaid ` | 일직선 사슬(A --> B --> C …)은 줄바꿈되는 상자, 그 밖(subgraph 등)은 미리 렌더한 SVG |
| `<img src="images/…">` + 다음 줄 `*캡션*` | data URI 그림 + 캡션 (`width` 속성은 최대 폭) |
| `> 인용` | 콜아웃 상자 |
| 표 | 가로 스크롤 상자 안의 표 |
| `<details><summary>정답 보기</summary> … </details>` | "정답 보기" 버튼, 누르면 아래로 펼쳐짐 |

## 챕터별 설정 — `chapters/chXX.json`

없어도 빌드는 된다. 표지 문구는 README의 `# 제목`에서, 차례 이름과 레일 이름은 각 `##` 제목에서 추정한다.
다만 장이 7개를 넘으면 차례가 길어지고 레일 이름이 겹치기 쉬우므로, 설정 파일로 그룹과 짧은 레일 이름을 주는 편이 좋다.
`chapters/ch02.json`이 예시다.

```json
{
  "title": "2장 모델 기본편",                 // 브라우저 탭 제목
  "description": "…",                          // 페이지 설명
  "headline": "2장. 모델 (기본편)",            // 표지 큰 제목 (기본값: README의 # 제목)
  "meta": "소프트 딕셔너리 → … · 13개 절",     // 표지 아래 한 줄
  "searchHint": "예: 인과 마스크, LayerNorm",  // 검색창 안내 문구
  "sections": [                                 // 장마다: 제목 앞부분으로 찾아 그룹·레일 이름을 준다
    { "match": "2.1", "group": "어텐션", "rail": "기본 원리" },
    { "match": "복습 질문", "group": "GPT-2와 정리", "rail": "복습" }
  ]
}
```

(실제 JSON 파일에는 `//` 주석을 쓰지 않는다.)

- `match`: 장 제목이 이 문자열과 같거나, 이 문자열 + 공백으로 시작하면 해당 장으로 본다.
- `group`: 같은 값이 이어지는 장들이 차례에서 접이식 그룹 하나로 묶인다. 그룹은 2~5개, 그룹당 장 3~6개가 적당하다.
- `rail`: 상단 레일에 붙는 짧은 이름. 8자 안쪽.

## 파일

| 파일 | 내용 |
|---|---|
| `build.cjs` | 노트 → HTML 변환 |
| `verify.cjs` | 단독 배포 검사 (+ 스크린샷) |
| `template/page.html`, `template/deck.css` | 덱 셸(장 단위 화면, 차례, 레일, 검색). deck 스킬의 템플릿을 복사해 둔 것 |
| `template/note.css` | 노트용 추가 스타일(수식, 코드 글꼴, 정답 버튼, 흐름 상자) |
| `chapters/*.json` | 챕터별 설정 |
