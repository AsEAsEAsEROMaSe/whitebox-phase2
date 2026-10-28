"""blend101 — ВИРОК ІДЕЇ «`_W_BLEND` ЯК ФУНКЦІЯ МЕРЕЖІ, А НЕ КОНСТАНТА».

ЗМІРЯНІ ВХОДИ (обидва — ПОВНИЙ оцінювач, 50 мереж, `--format json`):
    est_w1.py (_W_BLEND=1.0) ⟹ E²_i = похибка ЗАМИКАННЯ по мережі
    est_w0.py (_W_BLEND=0.0) ⟹ V_i  = дисперсія МК-гілки по мережі
    датасет  avg_variance_i  ⟹ ІСТИННА Var(out) — те, що замикання передбачає
                                з corr 0.9911 і залишковим розкидом 7.64%

ВИРІШАЛЬНЕ ПИТАННЯ: corr(V_i, E²_i). Якщо вони пропорційні, `W*_i` СТАЛЕ і
ідея мертва за побудовою (ceil101: ×1.0002). Якщо незалежні — стеля ×1.07.

🔒 ПЛОМБИ:
  (1) mean(V_i) мусить лягти біля записаного 1.9227e-06
  (2) mean(V_i·E²_i/(V_i+E²_i)) при сталих = 1.7767e-06 (перевірено ceil101)
  (3) реальний сабміт sub119r05 дає 1.7890893e-06 — модель мусить його НАКРИТИ
  (4) 🔒 КОНТРОЛЬ «ПЕРЕМІШАНА V»: підставити V_i іншої мережі ⟹ виграш мусить
      ЗНИКНУТИ. Без цього не відрізнити інформацію від калібровки.
  (5) 🔒 КРОС A/B: параметр підбирається на парних, вирок на непарних.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) corr(log V, log E²) > 0.5 ⟹ ідея сильно послабне               50%
 (б) виграш на кросі A→B перевищить ×1.02                          40%
 (в) 🔒 перемішаний контроль дасть ×1.00±0.005                     80%
"""
import glob
import json
import math

import numpy as np
import pyarrow as pa

SP = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
      "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/")
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")

E2 = np.array([r["final_layer_mse"] for r in
               json.load(open(SP + "w1.json"))["results"]["per_mlp"]])
V = np.array([r["final_layer_mse"] for r in
              json.load(open(SP + "w0.json"))["results"]["per_mlp"]])
NN = len(V)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
AV = np.asarray(tb.column("avg_variance")).astype(np.float64)[:NN]

print(f"blend101 · {NN} мереж")
print(f"🔒 (1) mean(V)  = {V.mean():.6e}   [записано 1.9227e-06]")
print(f"      mean(E²) = {E2.mean():.6e}   [sub99 давав 2.3385e-05]")
print(f"      розкид V ×{V.max()/V.min():.1f} · E² ×{E2.max()/E2.min():.1f}\n")

lv, le, la = np.log(V), np.log(E2), np.log(AV)
print("— КОРЕЛЯЦІЇ (вирішальні) —")
print(f"  corr(log V, log E²)        {np.corrcoef(lv, le)[0,1]:>8.4f}"
      f"   ← >0.5 ⟹ ідея слабшає")
print(f"  corr(log V, log avg_var)   {np.corrcoef(lv, la)[0,1]:>8.4f}"
      f"   ← наскільки V читається з Var")
print(f"  corr(log E², log avg_var)  {np.corrcoef(le, la)[0,1]:>8.4f}\n")

W0 = 0.0798


def agg(w):
    return float(np.mean((1 - w) ** 2 * V + w * w * E2))


print("— ЩО ДАЄ ВАГА, ЗАЛЕЖНА ВІД МЕРЕЖІ —")
print(f"  фіксована W₀=0.0798            {agg(W0):.6e}")
best = min(((agg(w), w) for w in np.linspace(0.0, 0.4, 401)))
print(f"  найкраща ФІКСОВАНА  W={best[1]:.4f}    {best[0]:.6e}  ×{agg(W0)/best[0]:.4f}")
wo = V / (V + E2)
print(f"  ОРАКУЛЬНА W*_i (стеля)         {agg(wo):.6e}  ×{best[0]/agg(wo):.4f}")

print("\n— РЕАЛІЗОВНА ФОРМА: W_i = 1/(1 + κ₀/Var̂_i), один новий скаляр —")
A, B = np.arange(NN)[::2], np.arange(NN)[1::2]


def aggs(w, s):
    return float(np.mean((1 - w[s]) ** 2 * V[s] + w[s] ** 2 * E2[s]))


def scan(var, sel):
    r = None
    for k0 in np.exp(np.linspace(math.log(0.02), math.log(20.0), 400)):
        w = 1.0 / (1.0 + k0 / var)
        v = aggs(w, sel)
        if r is None or v < r[0]:
            r = (v, k0)
    return r


for name, var in (("істинна avg_var", AV),):
    vin, k0 = scan(var, np.arange(NN))
    bA, kA = scan(var, A)
    wB = 1.0 / (1.0 + kA / var)
    print(f"  {name:<18s} κ₀={k0:.4f}  in-sample {vin:.6e} "
          f"×{agg(W0)/vin:.4f}")
    print(f"  {'':18s} крос A→B: {aggs(wB, B):.6e} проти "
          f"{aggs(np.full(NN, W0), B):.6e}  "
          f"×{aggs(np.full(NN,W0),B)/aggs(wB,B):.4f}")

rng = np.random.default_rng(4242)
print("\n🔒 (4) КОНТРОЛЬ «ПЕРЕМІШАНА Var» — 200 перестановок")
gains = []
for _ in range(200):
    p = rng.permutation(NN)
    _, kA2 = scan(AV[p], A)
    wB2 = 1.0 / (1.0 + kA2 / AV[p])
    gains.append(aggs(np.full(NN, W0), B) / aggs(wB2, B))
gains = np.array(gains)
print(f"  сер {gains.mean():.4f} · 95%-перцентиль {np.percentile(gains,95):.4f} "
      f"· макс {gains.max():.4f}")

print("\n" + "=" * 70)
print("— ЧОМУ НУЛЬ: МЕХАНІЗМ. `W*` залежить від ВІДНОШЕННЯ V/E² —")
lw = np.log(V / E2)
print(f"  нахил log V   на log avg_var   {np.polyfit(la, lv, 1)[0]:>8.4f}")
print(f"  нахил log E²  на log avg_var   {np.polyfit(la, le, 1)[0]:>8.4f}")
print(f"  ⟹ нахил log(V/E²)             {np.polyfit(la, lw, 1)[0]:>8.4f}")
print(f"  corr(log(V/E²), log avg_var)   {np.corrcoef(la, lw)[0,1]:>8.4f}")
print("  ⟹ avg_var тягне V і E² ОДНАКОВО і в їхньому відношенні СКОРОЧУЄТЬСЯ")

print("\n— ЧИ Є ВЗАГАЛІ ДЕТЕРМІНОВАНИЙ ПРОКСІ ДЛЯ W*_i? —")
wo = V / (V + E2)
for nm, x in (("avg_variance", AV), ("log avg_variance", la),
              ("E² (недоступне)", E2), ("V (недоступне)", V)):
    print(f"  corr(W*_i, {nm:<18s}) {np.corrcoef(wo, x)[0,1]:>8.4f}")
print(f"\n  розкид самого W*_i: [{wo.min():.4f} … {wo.max():.4f}] "
      f"медіана {np.median(wo):.4f}")
