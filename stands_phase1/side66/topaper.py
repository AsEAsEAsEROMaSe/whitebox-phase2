"""topaper — markdown зі звітними маркерами у markdown академічної статті.

Один прохід, ідемпотентний, і 🚨 НІКОЛИ не чіпає вміст код-блоків, крім
самих маркерів. Попередня версія робила глобальний `^[ \\t]+` і зрізала
відступи у 53 рядках верстатки — саме тому це тепер один файл із пломбою.

🔒 ПЛОМБА: кількість рядків з провідним пробілом усередині ``` мусить
   лишитись НЕЗМІННОЮ.
"""
import io
import re
import sys

SRC, DST = sys.argv[1], sys.argv[2]

LABEL = {"🔒": "**Seal.**", "🔑": "**Mechanism.**", "⚠": "**Caveat.**",
         "📏": "**Rule.**", "🎓": "**Lesson.**", "🚨": "**Warning.**",
         "🔬": "**Measurement.**", "🚦": "**Scope.**", "🛑": "**Ceiling.**",
         "⛔": "**Closed.**", "📎": "", "🔴": "", "🟢": "", "📌": "",
         "🖼": "", "🎯": "", "💰": "", "✅": "", "❌": ""}
VERB = {"🔒": "†", "✅": "y", "❌": "n", "🔑": " ", "⚠": "!", "📏": " ",
        "🎓": " ", "⛔": " ", "🛑": " ", "🟢": " ", "📌": " ", "🚨": "!"}
PAT = re.compile("([" + "".join(LABEL) + r"])(️?)(\s*)")
LB = r"(Seal|Mechanism|Caveat|Rule|Lesson|Warning|Measurement|Scope|Ceiling|Closed)"


def code_indents(text):
    """відступ кожного рядка верстатки, за позицією у файлі."""
    out, inb = [], False
    for l in text.split("\n"):
        if l.startswith("```"):
            inb = not inb
            out.append(None)
            continue
        out.append(len(l) - len(l.lstrip(" ")) if inb else None)
    return out


src = io.open(SRC, encoding="utf-8").read()
before = code_indents(src)

out, inb = [], False
for line in src.split("\n"):
    if line.startswith("```"):
        inb = not inb
        out.append(line)
        continue
    if inb:
        # ширина: емодзі ≈ 2 знаки моноширинного ⟹ віддаємо ту саму ширину
        out.append(PAT.sub(lambda m: (VERB.get(m.group(1), " ") + " ")[:2]
                           + m.group(3), line))
        continue

    def rep(m, _l=line):
        rest = _l[m.end():]
        if rest.startswith("**"):
            return ""                       # жирне вже несе наголос
        lab = LABEL.get(m.group(1), "")
        return (lab + (m.group(3) or " ")) if lab else ""
    out.append(PAT.sub(rep, line))

res = "\n".join(out)
res = re.sub(r"\*\*" + LB + r"\.\*\*\s+(?=[Ss]eals?\b)", "", res)   # тавтологія
res = re.sub(r"\*\*" + LB + r"\.\*\*\s+(?=And\b)", "", res)         # «And …»
res = re.sub(r"^(\*\*" + LB + r"\.\*\*) +", r"\1 ", res, flags=re.M)
res = re.sub(r"[ \t]+$", "", res, flags=re.M)
res = re.sub(r"\n{3,}", "\n\n", res)

after = code_indents(res)
left = sum(res.count(e) for e in LABEL)
print(f"емодзі лишилось {left} · міток {len(re.findall(r'[*][*]' + LB + r'[.][*][*]', res))}")
lost = [i for i, (x, y) in enumerate(zip(before, after))
        if x is not None and y is not None and y < x
        and not src.split("\n")[i][:1].isspace() is False]
lost = [i for i, (x, y) in enumerate(zip(before, after))
        if x is not None and y is not None and y < x and x > 0
        and src.split("\n")[i].lstrip(" ")[:1] not in "".join(LABEL)]
print(f"🔒 рядків верстатки, що ВТРАТИЛИ відступ: {len(lost)}")
assert left == 0, "лишились емодзі"
assert not lost, f"ЗЛАМАНО ВІДСТУПИ у рядках {lost[:8]}"
io.open(DST, "w", encoding="utf-8").write(res)
print("ok")
