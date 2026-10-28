"""alllayer101 — МИ ВИКИДАЄМО ГОТОВУ ВІДПОВІДЬ. Скільки коштує її НЕ викидати?

ФАКТ ІЗ КОДУ САБМІТУ (`estimator_sub119r05.py:408`):
    out.append(fnp.sum(x,axis=0)/denom if i == depth-1 else zrow)   # zrow = НУЛІ
⟹ у `all_layer_means` ми кладемо нулі для шарів 0..30, хоча наша рекурсія
   рахує `m` на КОЖНОМУ шарі і просто його не зберігає.

ЩО ЦЕ КОШТУЄ: `all_layers_mse` = mean по 32 шарах. З нулями він дорівнює
mean(true²) ≈ 0.75. Наша панель: 0.7537. Лідер: 0.7437 (він заповнює лише шар 0).

ЦІНА ЗАПОВНЕННЯ: НУЛЬ ФЛОПІВ — значення вже обчислені.
РИЗИК ДЛЯ ОСНОВНОГО БАЛУ: НУЛЬ — останній елемент не міняється.

🔒 ПЛОМБА: шар 0 має лягти РІВНО на аналітику `‖w‖/√(2π)`, тобто дати
   6.87e-10 — рівень шуму істини (зміряно leader101).

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) ALL-LAYERS впаде нижче 1e-3                                    80%
 (б) похибка `m` по шарах лишиться в смузі 0.4…1.0% усю глибину     70%
 (в) шар 0 дасть рівно підлогу шуму істини                          85%
"""
import glob
import math

import numpy as np
import pyarrow as pa

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG, KAPPA, ALPHA = 8, 0.9985, -0.00679
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = 50


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def ndtr(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
al = tb.column("all_layer_means").combine_chunks()
while not pa.types.is_primitive(al.type):
    al = al.flatten()
ALM = np.asarray(al).reshape(tb.num_rows, DEPTH, W_).astype(np.float64)

per = np.zeros((NNET, DEPTH))
rel = np.zeros((NNET, DEPTH))
for net in range(NNET):
    W = Wall[net].astype(np.float64)
    m = np.zeros(W_)
    C = np.eye(W_)
    for i in range(DEPTH):
        w = W[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = np.sqrt(np.maximum(np.diag(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        pred = mn if i == 0 else (1.0 + ALPHA) * mn
        tr = ALM[net, i]
        per[net, i] = ((pred - tr) ** 2).mean()
        rel[net, i] = math.sqrt(per[net, i] / (tr * tr).mean())
        if i == DEPTH - 1:
            break
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * np.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * np.outer(ph / s, ph / s)
        C = Co + np.diag(d - np.diag(Co))
        m = mn
    if (net + 1) % 25 == 0:
        print(f"  {net+1}/{NNET}", flush=True)

lvl = (ALM[:NNET] ** 2).mean(axis=2)
print(f"\n{'шар':>4s}{'per_layer_mse':>16s}{'рівень':>10s}{'віднос.похибка':>16s}")
for i in list(range(0, 8)) + [12, 16, 20, 24, 28, 30, 31]:
    print(f"{i:>4d}{per[:,i].mean():>16.6e}{lvl[:,i].mean():>10.4f}"
          f"{100*rel[:,i].mean():>15.4f}%")
print("=" * 46)
print(f"ALL-LAYERS з НАШИМ m      {per.mean():.6e}")
print(f"ALL-LAYERS з нулями       {lvl.mean():.6f}   [наша панель 0.7537]")
print(f"⟹ покращення             ×{lvl.mean()/per.mean():.0f}   ЗА НУЛЬ ФЛОПІВ")
print(f"\n🔒 шар 0: {per[:,0].mean():.4e}  [підлога шуму істини 6.87e-10]"
      f"  ×{per[:,0].mean()/6.871352e-10:.3f}")
print(f"   лідер ALL-LAYERS 0.7437 = заповнено ЛИШЕ шар 0 "
      f"⟹ {(lvl.mean()*32 - lvl[:,0].mean())/32:.4f} передбачено")
