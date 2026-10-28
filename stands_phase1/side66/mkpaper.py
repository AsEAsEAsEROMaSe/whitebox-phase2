"""mkpaper — markdown у PDF, зверстаний як академічна стаття.

Відмінності від mkpdf.py (який робив «звіт»):
  · гарнітура P052/Palladio — класична паперова, не DejaVu
  · титул по центру, як у статті; анотація вужчою смугою
  · заголовки нумеровані, без кольорових лінійок і без синього
  · таблиці у стилі booktabs: лише горизонтальні лінійки, без заливки
  · верстатка без плашки — як \\verbatim у LaTeX
  · НОМЕРИ СТОРІНОК і ЗМІСТ ІЗ НОМЕРАМИ — двопрохідний рендер:
      прохід 1 → знайти, на якій сторінці опинився кожен заголовок (fitz)
      прохід 2 → підставити номери у зміст і надрукувати заново
  · закладки PDF (outline), щоб читач мав навігацію збоку

🚨 Chrome не підтримує CSS margin boxes, тож номери сторінок штампуються
   через fitz ПІСЛЯ рендеру, а не засобами CSS.
"""
import html
import io
import os
import re
import subprocess
import sys

import fitz
import markdown

SRC, OUT_PDF = sys.argv[1], sys.argv[2]
TITLE = sys.argv[3]
SUB = sys.argv[4]
BYLINE = sys.argv[5] if len(sys.argv) > 5 else ""
LANG = sys.argv[6] if len(sys.argv) > 6 else "en"
DATE = "17 серпня 2026" if LANG == "uk" else "17 August 2026"
ABSH = "Анотація" if LANG == "uk" else "Abstract"
TOCH = "Зміст" if LANG == "uk" else "Contents"
TMP = os.path.splitext(OUT_PDF)[0] + "_work"

BODY = '"P052", "URW Palladio L", "Nimbus Roman", Georgia, serif'
MONO = '"DejaVu Sans Mono", monospace'

CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: %(body)s; font-size: 10.5pt; line-height: 1.44;
       color: #000; margin: 0; text-align: justify; hyphens: auto;
       orphans: 3; widows: 3; }

/* ── титул ───────────────────────────────────────────────── */
.tp { text-align: center; padding-top: 26mm; page-break-after: always; }
.tp .t  { font-size: 21pt; line-height: 1.22; font-weight: bold;
          margin: 0 auto 7mm; max-width: 150mm; }
.tp .s  { font-size: 12pt; font-style: italic; margin: 0 auto 12mm;
          max-width: 140mm; line-height: 1.4; }
.tp .by { font-size: 11pt; margin-bottom: 3mm; }
.tp .dt { font-size: 10.5pt; }
.tp .ab { text-align: justify; max-width: 132mm; margin: 16mm auto 0;
          font-size: 9.8pt; line-height: 1.4; }
.tp .ab h2 { text-align: center; font-size: 11pt; margin: 0 0 3mm;
             border: 0; padding: 0; page-break-before: avoid; }

/* ── заголовки ───────────────────────────────────────────── */
h1 { font-size: 15pt; margin: 0 0 4mm; text-align: left; }
h2 { font-size: 13pt; font-weight: bold; text-align: left;
     margin: 8mm 0 3mm; page-break-after: avoid; }
h3 { font-size: 11.2pt; font-weight: bold; text-align: left;
     margin: 6mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 10.5pt; font-style: italic; margin: 4mm 0 1.5mm;
     page-break-after: avoid; }
h2:first-of-type { page-break-before: avoid; }

p { margin: 0 0 2.4mm; }
p + p { text-indent: 0; }
strong { font-weight: bold; }
em { font-style: italic; }
a { color: #000; text-decoration: none; }

/* ── верстатка: як \\verbatim, без плашки ─────────────────── */
pre { font-family: %(mono)s; font-size: 7.9pt; line-height: 1.32;
      background: none; border: 0; border-left: 0.6pt solid #999;
      padding: 1.2mm 0 1.2mm 3.2mm; margin: 3mm 0 3.4mm 2mm;
      white-space: pre; overflow: hidden; page-break-inside: avoid;
      text-align: left; }
code { font-family: %(mono)s; font-size: 8.9pt; background: none; }
pre code { font-size: inherit; }

/* ── таблиці: booktabs ───────────────────────────────────── */
table { border-collapse: collapse; width: 100%%; font-size: 9pt;
        margin: 3mm 0 4mm; page-break-inside: avoid; text-align: left; }
th, td { border: 0; padding: 1.4mm 2.4mm 1.4mm 0; vertical-align: top; }
thead th { border-top: 1pt solid #000; border-bottom: 0.5pt solid #000;
           font-weight: bold; }
tbody tr:last-child td { border-bottom: 1pt solid #000; }
tbody td { background: none; }

blockquote { margin: 3mm 0 3.4mm 6mm; padding: 0; font-size: 9.6pt;
             line-height: 1.38; border: 0; }
ul, ol { margin: 2mm 0 2.6mm 6mm; padding-left: 2mm; }
li { margin: 0.8mm 0; }
hr { display: none; }   /* у статті розділи не відбивають лінійкою */

/* ── зміст ───────────────────────────────────────────────── */
.toc { page-break-after: always; }
.toc h2 { page-break-before: avoid; text-align: center; margin: 0 0 4mm; }
.toc table { font-size: 9.6pt; width: 100%%; line-height: 1.2; }
.toc td { padding: 0.55mm 0; border: 0; }
.toc td.n { text-align: right; width: 12mm; font-variant-numeric: tabular-nums; }
.toc tr.l1 td { font-weight: bold; padding-top: 1.9mm; }
.toc tr.l2 td.t { padding-left: 7mm; font-weight: normal; }
.toc td.t { position: relative; }
.toc td.t::before { content: ""; position: absolute; left: 0; right: 0;
                    bottom: 0.32em; border-bottom: 0.5pt dotted #777; }
.toc td.t span { position: relative; background: #fff; padding-right: 3pt; }
.toc { page-break-inside: auto; }
""" % {"body": BODY, "mono": MONO}

_W = 470.0
USABLE_PT, ADVANCE, BASE_PT = 452.0, 0.602, 7.9   # 452pt = 159.5mm, запас 5mm
ABS_PT = 342.0        # смуга анотації 132mm мінус відступ і запас


def build_html(src, toc_rows):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    body = md.convert(src)
    # анотацію перенести на титул
    m = re.search(rf"<h2[^>]*>{ABSH}</h2>(.*?)(?=<h2)", body, re.S)
    abstract = ""
    if m:
        abstract = f"<h2>{ABSH}</h2>" + m.group(1)
        body = body[:m.start()] + body[m.end():]
    body = re.sub(r"<h1.*?</h1>", "", body, flags=re.S)

    rows = []
    for lvl, num, txt, page in toc_rows:
        clean = html.escape(txt.replace("`", "").replace('"', "\u201c", 1))
        rows.append(f'<tr class="l{lvl-1}"><td class="t"><span>{clean}</span></td>'
                    f'<td class="n">{page or ""}</td></tr>')
    toc = (f"<div class='toc'><h2>{TOCH}</h2><table>" + "".join(rows)
           + "</table></div>")

    def fit_all(frag, width):
        """підганяє кегль КОЖНОГО pre під задану ширину смуги."""
        def one(m):
            inner = m.group(2)
            t = re.sub(r"<[^>]+>", "", inner)
            for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                         ("&quot;", '"')):
                t = t.replace(a, b)
            mx = max((len(x) for x in t.split("\n")), default=0)
            if not mx:
                return m.group(0)
            fs = min(BASE_PT, width / (ADVANCE * mx))
            if fs >= BASE_PT - 0.01:
                return m.group(0)
            return (f'<pre style="font-size:{fs:.2f}pt"{m.group(1)}>'
                    f'{inner}</pre>')
        return re.sub(r"<pre([^>]*)>(.*?)</pre>", one, frag, flags=re.S)

    abstract = fit_all(abstract, ABS_PT)
    body = fit_all(body, USABLE_PT)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(TITLE)}</title><style>{CSS}</style></head><body>
<div class="tp"><div class="t">{html.escape(TITLE)}</div>
<div class="s">{html.escape(SUB)}</div>
<div class="by">{BYLINE}</div><div class="dt">{DATE}</div>
<div class="ab">{abstract}</div></div>
{toc}
{body}
</body></html>"""
    return doc


def render(htm, pdf):
    hp = TMP + ".html"
    io.open(hp, "w", encoding="utf-8").write(htm)
    r = subprocess.run(["/usr/bin/google-chrome", "--headless", "--disable-gpu",
                        "--no-sandbox", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", "file://" + os.path.abspath(hp)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        print(r.stderr[-1500:])
        sys.exit(1)


src = io.open(SRC, encoding="utf-8").read()
# 🚨 титульний блок (h1, підзаголовок, рядок автора, ---) переїхав на титульну
#    сторінку; лишити його ще й у тілі — здублювати на першій сторінці тексту
_i = src.find(f"## {ABSH}")
if _i > 0:
    src = src[_i:]
def _clean(t):
    """🚨 у PDF нема ні лапок, ні зірочок — пошук і закладки мусять іти
    ЧИСТИМ текстом, інакше заголовок «III.1 `κ` …» не знаходиться ніде."""
    return re.sub(r"[`*_]", "", t).strip()


heads = [(2 if h.startswith("## ") else 3, _clean(h.lstrip("# ")))
         for h in src.split("\n")
         if re.match(r"^#{2,3} ", h) and ABSH not in h]
# 🚨 підзаголовок статті — це `###` ДО першого `##`; він не є розділом і
#    ламає ієрархію закладок («item 0 must be level 1»)
while heads and heads[0][0] == 3:
    heads.pop(0)

# ── прохід 1: без номерів, аби дізнатись сторінки ────────────────────────
render(build_html(src, [(l, None, t, None) for l, t in heads]), TMP + "1.pdf")
d = fitz.open(TMP + "1.pdf")
# 🚨 сторінка змісту містить ТЕКСТ усіх заголовків, тож пошук знаходить їх
#    саме на ній. Шукаємо лише ПІСЛЯ неї.
toc_pno = 0
for pno in range(len(d)):
    if d[pno].search_for(TOCH, quads=False):
        toc_pno = pno
        break
pages, floor = [], toc_pno + 1
for lvl, txt in heads:
    probe = re.sub(r"\s+", " ", txt)[:34]
    found = None
    for pno in range(floor, len(d)):
        if d[pno].search_for(probe, quads=False):
            found = pno + 1
            break
    if found is None:                       # довгий заголовок міг перенестись
        short = " ".join(probe.split()[:5])
        for pno in range(floor, len(d)):
            if d[pno].search_for(short, quads=False):
                found = pno + 1
                break
    pages.append(found or floor + 1)
    floor = max(floor, (found or floor + 1) - 1)   # монотонність
d.close()
print(f"прохід 1: {len(heads)} заголовків розкладено по сторінках")

# ── прохід 2: зі змістом ────────────────────────────────────────────────
render(build_html(src, [(l, None, t, p) for (l, t), p in zip(heads, pages)]),
       TMP + "2.pdf")

# ── штамп номерів сторінок + закладки ──────────────────────────────────
d = fitz.open(TMP + "2.pdf")
W, H = d[0].rect.width, d[0].rect.height
for i, pg in enumerate(d):
    if i == 0:
        continue                                   # титул без номера
    pg.insert_text((W / 2 - 8, H - 34), str(i + 1), fontname="times-roman",
                   fontsize=10, color=(0, 0, 0))
d.set_toc([[min(lvl - 1, 2), t, p] for (lvl, t), p in zip(heads, pages)])
d.save(OUT_PDF, garbage=4, deflate=True)
print(f"{OUT_PDF}: {len(d)} стор., закладок {len(heads)}")
d.close()
for f in (TMP + "1.pdf", TMP + "2.pdf", TMP + ".html"):
    if os.path.exists(f):
        os.remove(f)
