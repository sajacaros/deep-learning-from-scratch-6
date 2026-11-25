// build.cjs로 만든 HTML이 파일 하나만으로 동작하는지 검사한다.
//
//   node verify.cjs ch02 [ch03 ...]
//   node verify.cjs ch02 --shots    # 장마다 스크린샷도 남긴다(shots/ 폴더)
//   node verify.cjs /tmp/x/ch04.html  # 다른 곳에 만든 HTML 검사
//
// 파일을 빈 임시 폴더에 하나만 복사하고, 네트워크 요청을 모두 막은 상태로 연다.
// 검사: 장 수, 깨진 그림, 차단된 요청, 스크립트 오류, 글자로 남은 $·마크다운 기호,
//       코드 상자 스타일, 400px 폭에서 가로 넘침.
const fs = require("fs");
const os = require("os");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "../..");
const args = process.argv.slice(2);
const SHOTS = args.includes("--shots");
const chapters = args.filter(a => !a.startsWith("--"));
if (!chapters.length) { console.error("사용: node verify.cjs ch02 [ch03 ...] [--shots]"); process.exit(1); }

(async () => {
  const browser = await chromium.launch();
  let failed = false;
  for (const ch of chapters) {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "note-html-"));
    const srcFile = ch.endsWith(".html") ? path.resolve(ch) : path.join(ROOT, "docs", ch, `${ch}.html`);
    const file = path.join(tmp, path.basename(srcFile));
    fs.copyFileSync(srcFile, file);
    const url = "file://" + file;

    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const blocked = [];
    await ctx.route("**/*", r => {
      const u = r.request().url();
      if (u.startsWith("data:") || u.split("#")[0] === url) return r.continue();
      blocked.push(u); return r.abort();
    });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(String(e)));
    await page.goto(url); await page.waitForTimeout(500);
    const n = await page.evaluate(() => document.querySelectorAll(".chapter-link").length);

    const r = { chapters: n, images: 0, brokenImages: 0, code: 0, codeUnstyled: 0, rawText: [] };
    for (let i = 1; i <= n; i++) {
      await page.goto(`${url}#s${i}`); await page.waitForTimeout(200);
      const s = await page.evaluate(() => {
        const pg = document.querySelector(".chapter-page.is-on");
        const imgs = [...pg.querySelectorAll("img")];
        const pres = [...pg.querySelectorAll("pre")];
        const codeText = pres.map(e => e.innerText).join("\n");
        const raw = pg.innerText.split("\n")
          .filter(l => /\$|\*\*|\\(mathrm|frac|top|sqrt)\b/.test(l) && !codeText.includes(l))
          .map(l => l.slice(0, 80));
        return {
          images: imgs.length, broken: imgs.filter(x => !x.complete || x.naturalWidth === 0).length,
          code: pres.length, unstyled: pres.filter(x => !x.classList.contains("code-block")).length, raw,
        };
      });
      r.images += s.images; r.brokenImages += s.broken; r.code += s.code; r.codeUnstyled += s.unstyled;
      r.rawText.push(...s.raw.map(l => `s${i}: ${l}`));
      if (SHOTS) {
        const dir = path.join(__dirname, "shots", path.basename(file, ".html")); fs.mkdirSync(dir, { recursive: true });
        await page.screenshot({ path: path.join(dir, `s${String(i).padStart(2, "0")}.png`), fullPage: true });
      }
    }

    const mobile = await ctx.newPage();
    await mobile.setViewportSize({ width: 400, height: 860 });
    const overflow = [];
    for (let i = 1; i <= n; i++) {
      await mobile.goto(`${url}#s${i}`); await mobile.waitForTimeout(150);
      const o = await mobile.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      if (o > 0) overflow.push(`s${i}:${o}px`);
    }
    await ctx.close();
    fs.rmSync(tmp, { recursive: true, force: true });

    const ok = n > 0 && !r.brokenImages && !blocked.length && !errors.length && !r.rawText.length && !r.codeUnstyled && !overflow.length;
    failed ||= !ok;
    console.log(`${ok ? "OK  " : "FAIL"} ${ch}`, JSON.stringify({ ...r, rawText: r.rawText.length, blocked: blocked.length, errors: errors.length, mobileOverflow: overflow }));
    for (const l of r.rawText.slice(0, 10)) console.log("     글자로 남음 →", l);
    for (const u of blocked.slice(0, 5)) console.log("     차단된 요청 →", u);
    for (const e of errors.slice(0, 5)) console.log("     스크립트 오류 →", e);
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
