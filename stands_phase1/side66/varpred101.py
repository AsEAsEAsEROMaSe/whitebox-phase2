"""varpred101 — ЧИ ПЕРЕДБАЧАЄ НАШЕ ЗАМИКАННЯ ДИСПЕРСІЮ ВИХОДУ `avg_variance`?

ЧОМУ ЦЕ ВАЖЛИВО (нове, С101). `avg_variance` = mean_j Var(out_j) лежить у
датасеті окремою колонкою і зміряно: сер **0.0495**, розкид між мережами
**×13.2** (mini) / **×44.3** (full). А наш `_W_BLEND` — ОДНА КОНСТАНТА на всі
мережі, хоча оптимальна вага є `W*_i = V_i/(V_i + E²_i)` і `V_i ∝ Var_i/N`.

⟹ ми зливаємо дві оцінки вагою, підібраною для СЕРЕДНЬОЇ мережі, на мережі,
   що відрізняються за `V_i` у 13 разів.

АРИФМЕТИКА ВИГРАШУ (до прогону, на реальних числах датасету):
    MSE_i(W₀) = (1−W₀)²V_i + W₀²E²    ⟹ mean = 1.778e-06
    🔒 ПЛОМБА: записане в CLAUDE.md значення моделі = 1.7767e-06 — збігається
    MSE_i(W*_i) = V_i E²/(V_i+E²)     ⟹ mean ≈ 1.687e-06
    ⟹ стеля ×1.054 при ЦІНІ НУЛЬ, якщо `V_i` відома

ЧИ ВІДОМА? `V_i = Var_i/N_eff`, а `Var_i` = `mean_j diag(Cov(out))_j` — це рівно
те, що наше замикання рахує на останньому шарі як `d`. Ціна = 0 (вже є).

🚨 ЦЕ ПЕРЕВІРКА ПРИДАТНОСТІ ПРИЛАДУ, А НЕ ВИРОК ІДЕЇ. Вирок — лише повним
   оцінювачем, парно, `--format json` (закон С101).

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) corr(log Var_pred, log Var_true) > 0.9                              70%
 (б) систематичний зсув буде (замикання ЗАНИЖУЄ σ на 10.6%) ⟹ нахил ≠ 1  85%
 (в) після зняття спільного множника залишковий розкид < 20%             55%
"""
import glob
import math
import sys

import numpy as np
import pyarrow as pa

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG, KAPPA = 8, 0.9985
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def ndtr(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


files = sorted(glob.glob(SRC + "/data-*.arrow"))
tb = pa.ipc.open_stream(pa.memory_map(files[0], "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
AV = np.asarray(tb.column("avg_variance")).astype(np.float64)
NNET = min(NNET, tb.num_rows)
print(f"varpred101 · {NNET} мереж · замикання fp64, κ={KAPPA} при i>{KSIG}",
      flush=True)

pred = np.zeros(NNET)
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
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        if i == DEPTH - 1:
            pred[net] = d.mean()          # ← передбачена avg_variance
            break
        Co = cpm * np.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * np.outer(ph / s, ph / s)
        C = Co + np.diag(d - np.diag(Co))
        m = mn
    if (net + 1) % 20 == 0:
        print(f"  {net+1}/{NNET}", flush=True)

t_ = AV[:NNET]
lp, lt = np.log(pred), np.log(t_)
r = np.corrcoef(lp, lt)[0, 1]
ratio = pred / t_
sl = np.polyfit(lt, lp, 1)[0]
res = ratio / np.median(ratio)
print(f"\n{'':>4s}{'істинна':>12s}{'передбач.':>12s}{'відн.':>9s}")
for k in range(0, NNET, max(1, NNET // 12)):
    print(f"{k:>4d}{t_[k]:>12.5f}{pred[k]:>12.5f}{ratio[k]:>9.3f}")
print("=" * 40)
print(f"corr(log,log)          {r:>10.4f}   [гейт (а): >0.90]")
print(f"нахил log-log          {sl:>10.4f}   [гейт (б): ≠1 ⟹ зсув є]")
print(f"медіана відношення     {np.median(ratio):>10.4f}")
print(f"розкид ПІСЛЯ множника  {100*res.std():>9.2f}%   [гейт (в): <20%]")
print(f"розкид істинної Var    ×{t_.max()/t_.min():>9.2f}")
