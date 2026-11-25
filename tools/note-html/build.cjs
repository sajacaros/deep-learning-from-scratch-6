// docs/chXX/README.md 노트 → docs/chXX/chXX.html (단독 배포용 HTML 한 파일)
//
//   node build.cjs ch02            # 한 챕터
//   node build.cjs ch02 ch03 ...   # 여러 챕터
//   node build.cjs ch04 --out-dir /tmp/x   # docs/ 대신 다른 폴더에 써 보기
//
// - 수식($…$, $`…`$, $$…$$)은 MathJax로 미리 SVG로 렌더한다.
// - mermaid 다이어그램은 헤드리스 Chromium에서 mermaid로 미리 SVG로 렌더한다.
// - 그림(images/…)은 data URI로 넣는다. 외부 요청이 없으므로 파일 하나만 배포해도 된다.
// - 장(h2)마다 한 화면씩 보여 주는 덱 셸(template/page.html + deck.css)을 쓴다.
// - 챕터별 설정(표지 문구, 차례 그룹, 레일 이름)은 chapters/chXX.json. 없으면 README에서 추정한다.
const fs = require("fs");
const path = require("path");
const MarkdownIt = require("markdown-it");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "../..");

// ---------------------------------------------------------------- MathJax
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mj = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "local" }),
});
let stats;
function tex(src, display) {
  stats.math++;
  const html = adaptor.outerHTML(mj.convert(src, { display }));
  if (html.includes("data-mjx-error") || html.includes("merror")) stats.mathErrors.push(src);
  return html;
}

// ---------------------------------------------------------------- markdown-it + 수식 규칙
const md = new MarkdownIt({ html: true, linkify: false, typographer: false });

md.block.ruler.before("fence", "math_block", (state, startLine, endLine, silent) => {
  const first = state.src.slice(state.bMarks[startLine] + state.tShift[startLine], state.eMarks[startLine]);
  if (first.trim() !== "$$") return false;
  let line = startLine + 1;
  const body = [];
  while (line < endLine) {
    const s = state.src.slice(state.bMarks[line] + state.tShift[line], state.eMarks[line]);
    if (s.trim() === "$$") break;
    body.push(s);
    line++;
  }
  if (line >= endLine) return false;
  if (silent) return true;
  const tok = state.push("math_block", "div", 0);
  tok.content = body.join("\n");
  tok.map = [startLine, line + 1];
  state.line = line + 1;
  return true;
}, { alt: ["paragraph", "blockquote", "list"] });
md.renderer.rules.math_block = (t, i) => `<div class="math-display">${tex(t[i].content, true)}</div>\n`;

md.inline.ruler.before("escape", "math_inline", (state, silent) => {
  const src = state.src, pos = state.pos;
  if (src[pos] !== "$") return false;
  let content, end, display = false;
  if (src.startsWith("$$", pos)) {                       // 한 줄 안의 $$...$$
    const close = src.indexOf("$$", pos + 2);
    if (close < 0) return false;
    content = src.slice(pos + 2, close); end = close + 2; display = true;
  } else if (src[pos + 1] === "`") {                     // $`...`$ (GitHub 문법)
    const close = src.indexOf("`$", pos + 2);
    if (close < 0) return false;
    content = src.slice(pos + 2, close); end = close + 2;
  } else {                                               // $...$
    let close = pos + 1;
    while (close < src.length) {
      close = src.indexOf("$", close);
      if (close < 0) return false;
      if (src[close - 1] !== "\\") break;
      close++;
    }
    content = src.slice(pos + 1, close);
    if (!content || /^\s|\s$/.test(content)) return false;
    end = close + 1;
  }
  if (!silent) {
    const tok = state.push(display ? "math_inline_display" : "math_inline", "span", 0);
    tok.content = content;
  }
  state.pos = end;
  return true;
});
md.renderer.rules.math_inline = (t, i) => `<span class="math-inline">${tex(t[i].content, false)}</span>`;
md.renderer.rules.math_inline_display = (t, i) => `<span class="math-display">${tex(t[i].content, true)}</span>`;

// ---------------------------------------------------------------- mermaid (헤드리스 Chromium)
async function renderMermaid(codes) {
  if (!codes.length) return [];
  const { chromium } = require("playwright");
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.setContent("<!doctype html><html><body></body></html>");
    await page.addScriptTag({ path: require.resolve("mermaid/dist/mermaid.min.js") });
    return await page.evaluate(async (list) => {
      mermaid.initialize({
        startOnLoad: false, theme: "neutral", securityLevel: "strict",
        fontFamily: '"Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif',
        htmlLabels: false, flowchart: { htmlLabels: false, useMaxWidth: false },   // 순수 SVG 글자: 페이지 CSS의 줄 높이에 영향받지 않게
      });
      const out = [];
      for (let i = 0; i < list.length; i++) out.push((await mermaid.render(`mmd${i}`, list[i])).svg);
      return out;
    }, codes);
  } finally {
    await browser.close();
  }
}

// 일직선 사슬(A --> B --> C …)이면 줄바꿈되는 HTML 상자로. 아니면 null(→ mermaid SVG).
function chainHtml(code) {
  const lines = code.split("\n").map(l => l.trim()).filter(Boolean);
  if (!/^(flowchart|graph)\s+(LR|RL|TB|TD|BT)$/.test(lines[0])) return null;
  const nodes = {}, edges = [];
  for (const l of lines.slice(1)) {
    const m = l.match(/^(\w+)(?:\s*\["([^"]*)"\]|\s*\[([^\]]*)\])?\s*-->(?:\|"?([^"|]*)"?\|)?\s*(\w+)(?:\s*\["([^"]*)"\]|\s*\[([^\]]*)\])?\s*$/);
    if (!m) return null;                                   // subgraph·스타일 등 → mermaid로
    const [, a, aq, ab, label, b, bq, bb] = m;
    if (aq || ab) nodes[a] = aq || ab;
    if (bq || bb) nodes[b] = bq || bb;
    edges.push({ a, b, label: label || "" });
  }
  if (!edges.length) return null;
  for (let i = 1; i < edges.length; i++) if (edges[i].a !== edges[i - 1].b) return null;   // 사슬이 아님
  const order = [edges[0].a, ...edges.map(e => e.b)];
  if (new Set(order).size !== order.length) return null;
  let h = '<div class="flow-chain">';
  order.forEach((id, k) => {
    if (k) h += `<span class="flow-arrow">${edges[k - 1].label ? `<span class="flow-label">${edges[k - 1].label}</span>` : ""}→</span>`;
    h += `<span class="flow-node">${nodes[id] ?? id}</span>`;
  });
  return h + "</div>";
}

// ---------------------------------------------------------------- 도우미
const MIME = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", svg: "image/svg+xml", webp: "image/webp" };
const strip = s => s.replace(/<[^>]+>/g, "").replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
const attr = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

function loadConfig(ch, src) {
  const file = path.join(HERE, "chapters", `${ch}.json`);
  const cfg = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  const h1 = (src.match(/^# (.*)$/m) || [])[1] || ch;
  const num = parseInt(ch.replace(/\D/g, ""), 10);
  return {
    title: cfg.title || h1.replace(/[.()]/g, "").replace(/\s+/g, " ").trim(),
    brand: cfg.brand || "밑바닥부터 시작하는 딥러닝 6",
    brandSub: cfg.brandSub || "학습 노트",
    description: cfg.description || `${h1} 학습 노트`,
    kicker: cfg.kicker || `CHAPTER ${num} · 학습 노트`,
    headline: cfg.headline || h1,
    meta: cfg.meta || "",
    searchHint: cfg.searchHint || "예: 검색어",
    sections: cfg.sections || [],
  };
}

// ---------------------------------------------------------------- 챕터 하나 빌드
async function build(ch) {
  stats = { math: 0, mathErrors: [], chains: 0, mermaid: 0, h2: 0, h3: 0, figures: 0, bareImg: 0 };
  const dir = path.join(ROOT, "docs", ch);
  let src = fs.readFileSync(path.join(dir, "README.md"), "utf8");
  const cfg = loadConfig(ch, src);

  src = src.replace(/^# .*\n/, "");                       // h1은 표지로 보낸다

  // mermaid: 자리표시자로 바꿔 두고, 렌더한 SVG를 나중에 끼운다
  const mermaidCodes = [];
  src = src.replace(/```mermaid\n([\s\S]*?)```/g, (_, code) => {
    const chain = chainHtml(code);
    if (chain) { stats.chains++; return `\n${chain}\n`; }
    mermaidCodes.push(code);
    return `\n<div class="mermaid-figure">@@MERMAID${mermaidCodes.length - 1}@@</div>\n`;
  });

  let html = md.render(src);

  const svgs = await renderMermaid(mermaidCodes);
  stats.mermaid = svgs.length;
  html = html.replace(/@@MERMAID(\d+)@@/g, (_, i) => svgs[+i]);

  // 그림: data URI + figure/figcaption, width 속성은 최대 폭으로
  html = html.replace(/<img src="(images\/[^"]+)"([^>]*)>/g, (_, p, rest) => {
    const buf = fs.readFileSync(path.join(dir, p));
    const mime = MIME[path.extname(p).slice(1).toLowerCase()] || "application/octet-stream";
    const w = (rest.match(/width="(\d+)"/) || [])[1];
    rest = rest.replace(/\s*width="\d+"/, "");
    return `<img src="data:${mime};base64,${buf.toString("base64")}"${rest}${w ? ` style="width:${w}px"` : ""}>`;
  });
  html = html.replace(/(<img [^>]*>)\s*<p><em>([\s\S]*?)<\/em><\/p>/g, (_, img, cap) => {
    stats.figures++;
    return `<figure>${img}<figcaption>${cap}</figcaption></figure>`;
  });
  stats.bareImg = (html.match(/^<img /gm) || []).length;
  const missing = html.match(/<img src="(?!data:)[^"]*"/g) || [];
  if (missing.length) throw new Error("data URI로 바꾸지 못한 그림: " + missing.join(", "));

  html = html.replace(/<pre><code/g, '<pre class="code-block"><code');                 // 덱의 코드 상자
  html = html.replace(/<table>/g, '<div class="table-scroll"><table>').replace(/<\/table>/g, "</table></div>");
  html = html.replace(/<blockquote>/g, '<div class="callout">').replace(/<\/blockquote>/g, "</div>");
  html = html.replace(/<hr>\n?/g, "");

  // h2 → 장. 차례 이름(data-title)·레일(data-rail)·그룹(data-group)
  html = html.replace(/<h2>([\s\S]*?)<\/h2>/g, (_, inner) => {
    const text = strip(inner);
    const title = text.split(" — ")[0].trim();
    const sec = cfg.sections.find(s => text === s.match || text.startsWith(s.match + " ")) || {};
    const rail = sec.rail || title.replace(/^\d+(\.\d+)*\s*/, "").slice(0, 8);
    const grp = sec.group ? ` data-group="${attr(sec.group)}"` : "";
    const id = `s${++stats.h2}`;
    const head = `<h2 id="${id}"${grp} data-title="${attr(title)}" data-rail="${attr(rail)}">`;
    // "제목 — `코드 파일`" → 장 제목은 이름만, 코드 파일은 장 첫 줄에 따로
    const fm = inner.match(/^([\s\S]*?) — <code>([^<]+)<\/code>$/);
    if (fm) return `${head}${fm[1]}</h2>\n<p class="chapter-code">코드: <code>${fm[2]}</code></p>`;
    return `${head}${inner}</h2>`;
  });
  html = html.replace(/<h3>/g, () => `<h3 id="h${++stats.h3}">`);

  const mathCss = adaptor.textContent(mj.outputJax.styleSheet(mj));
  const extraCss = fs.readFileSync(path.join(HERE, "template", "note.css"), "utf8");
  const content = `<style>${mathCss}\n${extraCss}</style>\n${html}`;

  let page = fs.readFileSync(path.join(HERE, "template", "page.html"), "utf8");
  const vals = {
    DECK_CSS: fs.readFileSync(path.join(HERE, "template", "deck.css"), "utf8"),   // 템플릿의 <style> 안에 들어간다
    TITLE: cfg.title, BRAND: cfg.brand, BRAND_SUB: cfg.brandSub, DESCRIPTION: attr(cfg.description),
    KICKER: cfg.kicker, HEADLINE: cfg.headline, META: cfg.meta, SIDEBAR_LABEL: "차례",
    SEARCH_HINT: attr(cfg.searchHint),
  };
  for (const [k, v] of Object.entries(vals)) page = page.split(`{{${k}}}`).join(v);
  const left = page.match(/\{\{(?!CONTENT\}\})[A-Z_]+\}\}/g);
  if (left) throw new Error("남은 자리표시자: " + left.join(", "));
  page = page.replace("{{CONTENT}}", () => content);

  const out = path.join(OUT_DIR || dir, `${ch}.html`);
  fs.writeFileSync(out, page);
  console.log(`${path.relative(ROOT, out)}  ${Math.round(page.length / 1024)} KB  ` +
    JSON.stringify({ ...stats, mathErrors: stats.mathErrors.length }));
  if (stats.mathErrors.length) console.warn("  수식 오류:", stats.mathErrors);
}

const argv = process.argv.slice(2);
const oi = argv.indexOf("--out-dir");
const OUT_DIR = oi >= 0 ? path.resolve(argv.splice(oi, 2)[1]) : null;   // 시험 빌드용: 다른 폴더에 쓰기

(async () => {
  const chapters = argv;
  if (!chapters.length) { console.error("사용: node build.cjs ch02 [ch03 ...]"); process.exit(1); }
  for (const ch of chapters) await build(ch);
})().catch(e => { console.error(e); process.exit(1); });
