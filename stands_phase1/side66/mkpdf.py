"""md2pdf — простий друкарський PDF із нашого markdown.

Кирилиця: DejaVu Serif · емодзі: Noto Color Emoji · код: DejaVu Sans Mono.
Рендер через headless Chrome, бо жоден інший рушій у системі не вміє
кольорові емодзі разом із кирилицею.
"""
import html
import os
import io
import re
import subprocess
import sys

import markdown

SRC = sys.argv[1]
OUT_HTML = sys.argv[2]
OUT_PDF = sys.argv[3]
TITLE = sys.argv[4] if len(sys.argv) > 4 else "White-Box"
SUB = sys.argv[5] if len(sys.argv) > 5 else ""
LANG = sys.argv[6] if len(sys.argv) > 6 else "uk"

src = io.open(SRC, encoding="utf-8").read()

# --- титул: перший H1 і підзаголовок беремо з тексту, з тіла прибираємо ----
lines = src.split("\n")
h1 = next((l[2:].strip() for l in lines if l.startswith("# ")), TITLE)

md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "attr_list"],
                       extension_configs={"toc": {"title": None, "toc_depth": "2-3"}})
body = md.convert(src)
toc = md.toc

CSS = r"""
@page { size: A4; margin: 17mm 15mm 17mm 15mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 10.2pt; line-height: 1.52; color: #111; margin: 0;
  orphans: 3; widows: 3; text-align: justify; hyphens: auto;
}
h1, h2, h3, h4 {
  font-family: "DejaVu Sans", "Liberation Sans", sans-serif;
  line-height: 1.25; text-align: left; page-break-after: avoid;
  text-wrap: balance;
}
h1 { font-size: 19pt; margin: 0 0 .5em; }
h2 {
  font-size: 15pt; margin: 1.9em 0 .55em; padding-bottom: .22em;
  border-bottom: 1.6pt solid #1a3a5c; color: #10233a;
  page-break-before: always;
}
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 12.2pt; margin: 1.5em 0 .4em; color: #1a3a5c; }
h4 { font-size: 10.8pt; margin: 1.2em 0 .35em; color: #333; }
p { margin: .52em 0; }
strong { color: #000; }
code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
  background: #f2f3f5; padding: .5pt 2pt; border-radius: 2px;
  word-break: break-word;
}
pre {
  font-family: "DejaVu Sans Mono", monospace; font-size: 7.7pt;
  line-height: 1.34; background: #f6f7f9; border: .5pt solid #dcdfe4;
  border-left: 2.2pt solid #1a3a5c; border-radius: 2px;
  padding: 5pt 7pt; margin: .7em 0; white-space: pre; overflow: hidden;
  page-break-inside: avoid; text-align: left;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; font-size: 8.4pt;
  margin: .8em 0; page-break-inside: avoid; text-align: left;
}
th, td { border: .5pt solid #c9ced6; padding: 3.2pt 5pt; vertical-align: top; }
th { background: #eef1f5; font-family: "DejaVu Sans", sans-serif; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote {
  margin: .8em 0; padding: .3em 0 .3em 10pt; border-left: 2.5pt solid #b9c2cc;
  color: #333; font-style: italic;
}
ul, ol { margin: .5em 0 .5em 1.1em; padding-left: .6em; }
li { margin: .22em 0; }
hr { border: 0; border-top: .6pt solid #d5d9df; margin: 1.4em 0; }
a { color: #1a3a5c; text-decoration: none; }

/* титул */
.title-page { page-break-after: always; text-align: left; padding-top: 55mm; }
.title-page .t { font-family: "DejaVu Sans", sans-serif; font-size: 25pt;
                 line-height: 1.2; font-weight: bold; color: #10233a; }
.title-page .s { font-family: "DejaVu Sans", sans-serif; font-size: 12.5pt;
                 color: #3a4a5c; margin-top: 9mm; line-height: 1.45; }
.title-page .m { font-size: 10pt; color: #555; margin-top: 22mm;
                 font-family: "DejaVu Sans", sans-serif; }
.rule { height: 2.5pt; background: #1a3a5c; width: 42mm; margin: 7mm 0; }

/* зміст */
.toc-page { page-break-after: always; }
.toc-page h2 { page-break-before: avoid; }
.toc ul { list-style: none; margin-left: 0; padding-left: 0; }
.toc ul ul { margin-left: 5mm; }
.toc li { margin: .18em 0; font-size: 9.4pt;
          font-family: "DejaVu Sans", sans-serif; }
.toc > ul > li { margin-top: .5em; font-weight: bold; font-size: 10pt; }
.toc > ul > li > ul > li { font-weight: normal; }
"""

doc = f"""<!doctype html><html lang="{LANG}"><head><meta charset="utf-8">
<title>{html.escape(h1)}</title><style>{CSS}</style></head><body>
<div class="title-page">
  <div class="rule"></div>
  <div class="t">{html.escape(h1)}</div>
  <div class="s">{html.escape(SUB)}</div>
  <div class="m">{html.escape(TITLE)}</div>
</div>
<div class="toc-page"><h2>Зміст</h2><div class="toc">{toc}</div></div>
{body}
</body></html>"""

# --- 🚨 АВТО-ПІДГІН КЕГЛЯ КОД-БЛОКІВ ------------------------------------------
# `pre { overflow: hidden }` ріже задовгі рядки МОВЧКИ — на око у PDF це
# виглядає як нормальна таблиця з відрізаною правою колонкою. Тому ширину
# кожного блоку рахуємо і кегль підбираємо, а не сподіваємось.
USABLE_PT = 496.0          # 180mm смуги − 5mm на padding/бордюр pre
ADVANCE = 0.602            # ширина гліфа DejaVu Sans Mono, часток em
BASE_PT = 7.7


def _fit(m):
    body_ = m.group(2)
    txt = re.sub(r"<[^>]+>", "", body_)
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"')):
        txt = txt.replace(a, b)
    mx = max((len(l) for l in txt.split("\n")), default=0)
    if mx == 0:
        return m.group(0)
    fs = min(BASE_PT, USABLE_PT / (ADVANCE * mx))
    if fs >= BASE_PT - 0.01:
        return m.group(0)
    _fit.shrunk.append((mx, round(fs, 2)))
    return f'<pre style="font-size:{fs:.2f}pt"{m.group(1)}>{body_}</pre>'


_fit.shrunk = []
doc = re.sub(r"<pre([^>]*)>(.*?)</pre>", _fit, doc, flags=re.S)
if _fit.shrunk:
    print(f"  підігнано {len(_fit.shrunk)} широких блоків, "
          f"найширший {max(x[0] for x in _fit.shrunk)} симв. → "
          f"{min(x[1] for x in _fit.shrunk)}pt")
else:
    print("  усі код-блоки влазять у смугу без підгону")

io.open(OUT_HTML, "w", encoding="utf-8").write(doc)
print(f"HTML {len(doc)/1024:.0f} КБ → {OUT_HTML}")

cmd = ["/usr/bin/google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
       "--no-pdf-header-footer",
       f"--print-to-pdf={OUT_PDF}",
       # 🚨 ЛИШЕ АБСОЛЮТНИЙ ШЛЯХ: із відносним Chrome мовчки віддає 1 порожню
       #    сторінку і код виходу 0 — прогін виглядає успішним.
       "file://" + os.path.abspath(OUT_HTML)]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)
print("PDF готовий")
