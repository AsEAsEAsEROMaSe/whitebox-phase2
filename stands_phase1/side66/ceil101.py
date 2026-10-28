"""ceil101 — СТЕЛЯ ІДЕЇ «W_BLEND ПО МЕРЕЖІ» при РІЗНИХ припущеннях про E²_i.

Модель злиття (виведена, не вгадана):
    ours = T + ε_v   (шум МК, дисперсія V_i)
    m·a  = T + ε_e   (похибка замикання, E²_i),  ε_v ⊥ ε_e
    out  = (1−W)·ours + W·(a·m)
    MSE_i(W) = (1−W)²V_i + W²E²_i        ⟹ W*_i = V_i/(V_i+E²_i)
                                            MSE*_i = V_i E²_i/(V_i+E²_i)
🔒 ПЛОМБА: при сталих V,E² формула дає рівно записане 1.7767e-06.

КЛЮЧОВЕ ПИТАННЯ — КОРЕЛЯЦІЯ. Якщо E²_i ∝ V_i, то W*_i СТАЛЕ і виграшу НЕМА
за побудовою. Тому рахую три сценарії; вирок дасть прогін оцінювача.
"""
import glob
import math
import sys

import numpy as np
import pyarrow as pa

W_ = 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
V_BAR, E2_BAR, W0 = 1.9227e-06, 2.3385e-05, 0.0798
NN = int(sys.argv[1]) if len(sys.argv) > 1 else 50

tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
AV = np.asarray(tb.column("avg_variance")).astype(np.float64)[:NN]
V = V_BAR * AV / AV.mean()          # V_i ∝ Var_i, нормовано на зміряне середнє
print(f"ceil101 · {NN} мереж · V_i ∝ avg_variance, розкид ×{AV.max()/AV.min():.1f}")
print(f"🔒 ПЛОМБА сталої моделі: {V_BAR*E2_BAR/(V_BAR+E2_BAR):.6e} "
      f"проти записаного 1.7767e-06\n")

rng = np.random.default_rng(20260809)
print(f"{'сценарій E²_i':<34s}{'фікс W₀':>12s}{'W*_i':>12s}{'×виграш':>10s}")
print("=" * 68)
for name, E2 in (
    ("стале E² (=2.3385e-05)", np.full(NN, E2_BAR)),
    ("незалежне, розкид 88.5%", E2_BAR * np.exp(rng.normal(0, 0.72, NN) - 0.26)),
    ("ПОВНІСТЮ ∝ V_i (найгірше)", E2_BAR * V / V.mean()),
    ("напів-∝ V_i (corr≈0.7)", E2_BAR * (V / V.mean()) ** 0.5),
):
    a = float(np.mean((1 - W0) ** 2 * V + W0 * W0 * E2))
    b = float(np.mean(V * E2 / (V + E2)))
    print(f"{name:<34s}{a:>12.5e}{b:>12.5e}{a/b:>10.4f}")

print("\n— ЧУТЛИВІСТЬ ДО ПОХИБКИ В `V_i` (наше замикання знає її з 7.64%) —")
E2 = np.full(NN, E2_BAR)
a = float(np.mean((1 - W0) ** 2 * V + W0 * W0 * E2))
for err in (0.0, 0.0764, 0.15, 0.30, 0.60):
    Vh = V * np.exp(rng.normal(0, err, NN)) if err else V
    Wi = np.clip(Vh / (Vh + E2), 0.0, 1.0)
    got = float(np.mean((1 - Wi) ** 2 * V + Wi * Wi * E2))
    print(f"  похибка V̂ {100*err:>5.1f}%   MSE {got:.5e}   ×{a/got:.4f}")
