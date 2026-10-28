"""orac101_an — аналіз стенду orac101. Вибір `α` і `κ` тут і ВИДИМИЙ.

MSE_i(α) = (α²·Σm² − 2α·ΣmT + ΣT²)/256          ← з трьох збережених сум
глобальний оптимальний α = Σ_i ΣmT_i / Σ_i Σm²_i
агрегат = mean_i MSE_i        (саме так рахує грейдер: середнє по мережах)
похибка x = √(MSE_agg / mean(T²))

🔒 ПЛОМБИ, ЯКІ ДРУКУЮТЬСЯ РАЗОМ ІЗ ТАБЛИЦЕЮ:
  · `allm` мусить дати 0.10…0.20% (одношагова похибка формули)
  · quarter проти full: якщо стеля РОСТЕ з N, число не збіглось
  · крос A/B: κ, підібраний на половині A, застосований до B
"""
import json
import math
import sys

V_MC = 1.9227e-06        # [ЗМІРЯНО С96 грейдерним приладом, 50 мереж]
SRC = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
    "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/orac101_big.json")
MODES = ["base", "off", "diag", "full", "allm", "mm"]

st = json.load(open(SRC))
nets = sorted(st, key=int)
print(f"orac101_an · {len(nets)} мереж · {SRC.split('/')[-1]}")

TT = {n: st[n]["_TT"] for n in nets}
MT2 = sum(TT.values()) / len(nets) / 256.0        # mean(T²)


def mse_agg(sub, mode, k, sel):
    """агрегатна MSE по мережах `sel` при глобальному оптимальному α"""
    num = den = 0.0
    for n in sel:
        tab = st[n][sub][mode]
        if k not in tab:
            return None, None
        mm, mt = tab[k]
        num += mt
        den += mm
    if den <= 0:
        return None, None
    a = num / den
    tot = 0.0
    for n in sel:
        mm, mt = st[n][sub][mode][k]
        tot += (a * a * mm - 2.0 * a * mt + TT[n]) / 256.0
    return tot / len(sel), a


def best(sub, mode, sel):
    ks = sorted(st[sel[0]][sub][mode], key=float)
    out = []
    for k in ks:
        v, a = mse_agg(sub, mode, k, sel)
        if v is not None and math.isfinite(v):
            out.append((v, k, a))
    return min(out) if out else (float("nan"), "-", 0.0)


def score(e2):
    return V_MC * e2 / (V_MC + e2)


print(f"\nmean(T²) = {MT2:.6f}   [для довідки: записана константа 0.9097]")
print("=" * 78)
print(f"{'режим':<8s}{'κ*':>8s}{'α*':>10s}{'MSE замик.':>14s}"
      f"{'похибка':>10s}{'×замик.':>9s}{'×БАЛ':>9s}")
print("=" * 78)

b0, k0, a0 = best("full", "base", nets)
s0 = score(b0)
rows = {}
for mode in MODES:
    v, k, a = best("full", mode, nets)
    rows[mode] = (v, k, a)
    print(f"{mode:<8s}{k:>8s}{a:>10.5f}{v:>14.6e}"
          f"{100*math.sqrt(v/MT2):>9.4f}%{b0/v:>9.4f}{s0/score(v):>9.4f}")

print("\n🔒 ПЛОМБА МАСШТАБУВАННЯ (чверть N проти повного N):")
print(f"{'режим':<8s}{'чверть':>12s}{'повний':>12s}{'×зміна':>10s}")
for mode in MODES:
    vq, _, _ = best("quarter", mode, nets)
    vf = rows[mode][0]
    print(f"{mode:<8s}{100*math.sqrt(vq/MT2):>11.4f}%"
          f"{100*math.sqrt(vf/MT2):>11.4f}%{vq/vf:>10.4f}")

print("\n🔒 КРОС A/B (κ підібрано на A, застосовано до B — чи не перефіт):")
A = [n for i, n in enumerate(nets) if i % 2 == 0]
B = [n for i, n in enumerate(nets) if i % 2 == 1]
bB, _, _ = best("full", "base", B)
for mode in MODES:
    _, kA, _ = best("full", mode, A)
    vB, _ = mse_agg("full", mode, kA, B)
    vBown, kB, _ = best("full", mode, B)
    if vB is None:
        continue
    print(f"{mode:<8s}κ(A)={kA:>7s}  на B: {100*math.sqrt(vB/MT2):.4f}% "
          f"(×{bB/vB:.4f})   власний κ(B)={kB:>7s}: "
          f"{100*math.sqrt(vBown/MT2):.4f}%")

d0 = [st[n]["_dev0"] for n in nets if "_dev0" in st[n]]
if d0:
    dd = sum(x[0] for x in d0) / len(d0)
    do = sum(x[1] for x in d0) / len(d0)
    print(f"\n🔒 ПЛОМБА ШАРУ 0 (вхід гаусів ТОЧНО ⟹ чиста похибка формули кроку):")
    print(f"   діагональ {100*dd:.4f}%  ·  позадіагональ {100*do:.4f}%")
    print(f"   [для звірки: маскова розбіжність С100 = 4.7…4.9% на глибині]")
