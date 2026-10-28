"""pack90 — дедуплікація кутів + компактне кодування + збірка HTML.

Кути зберігаю як ДЕЛЬТИ індексів сітки (2^23) у base36 — це втричі менше за
десяткові радіани і дає ту саму точність, що й сама сітка.
"""
import base64
import json
import os
import sys

OUT = "/mnt/win_d/llm/White-Box/side66/conedata"
M = 1 << 23
KDUP = 3
TWO_PI = 6.283185307179586
D36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def b36(v):
    if v == 0:
        return "0"
    s = ""
    while v:
        s = D36[v % 36] + s
        v //= 36
    return s


src = json.load(open(f"{OUT}/cone2d.json"))
edges_out, counts_out = {}, {}
for d, angs in src["edges"].items():
    idx = sorted({int(round(a * M / TWO_PI)) % M for a in angs})
    ded, last = [], -10 ** 9
    for i in idx:
        if i - last > KDUP:
            ded.append(i)
        last = i
    if len(ded) > 1 and (ded[0] + M - ded[-1]) <= KDUP:
        ded.pop(0)
    counts_out[d] = len(ded)
    prev, parts = 0, []
    for i in ded:
        parts.append(b36(i - prev))
        prev = i
    edges_out[d] = ".".join(parts)

raw = sum(len(v) for v in edges_out.values())
print(f"кути після дедуплікації: {raw/1024:.0f} КБ (було "
      f"{os.path.getsize(f'{OUT}/cone2d.json')/1024:.0f} КБ)")
for d in sorted(counts_out, key=int):
    print(f"   шар {d:>2s}: {counts_out[d]:>6d} секторів")

imgs = {}
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png") and (f.startswith("sph_") or f.startswith("zoom_")):
        b = open(f"{OUT}/{f}", "rb").read()
        imgs[f[:-4]] = "data:image/png;base64," + base64.b64encode(b).decode()
print(f"зображень {len(imgs)}, base64 разом "
      f"{sum(len(v) for v in imgs.values())/1024/1024:.2f} МБ")

tpl = open(sys.argv[1]).read()
blob = json.dumps({"M": M, "edges": edges_out, "counts": counts_out,
                   "uniq": src["uniq"], "img": imgs}, separators=(",", ":"))
html = tpl.replace("/*__DATA__*/", "window.CONE=" + blob + ";")
open(sys.argv[2], "w").write(html)
print(f"\n✅ {sys.argv[2]}  {os.path.getsize(sys.argv[2])/1024/1024:.2f} МБ"
      f"  {'⛔ ПОНАД 16 МБ' if os.path.getsize(sys.argv[2]) > 16e6 else '✅ у межах 16 МБ'}")
