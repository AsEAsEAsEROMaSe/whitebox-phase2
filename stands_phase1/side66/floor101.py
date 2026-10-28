"""floor101 — ПІДЛОГА ШУМУ САМОЇ ІСТИНИ. Датасет згенеровано МК з N=1e9.

МЕХАНІЗМ. `final_means` у датасеті — НЕ аналітична істина, а емпіричне середнє
з `n_samples = 1e9` (metadata.json). Отже воно має власну дисперсію
`Var(out_j)/N`, і будь-який оцінювач, навіть ІДЕАЛЬНИЙ, дасть
    MSE = mean_j Var(out_j)/N = avg_variance / N
Це ПІДЛОГА задачі. `avg_variance` лежить прямо в датасеті окремою колонкою.

ЗВІДКИ ГІПОТЕЗА (не вгадана — виведена з панелі лідера):
    GROWTH L0-L31 = per_layer_mse[31]/per_layer_mse[0] = 5.36
    STEEPEST L1   = per_layer_mse[1]/per_layer_mse[0]  = 7.4e8
    ⟹ per_layer_mse[0] = 3.628e-9/5.36 = 6.769e-10 (він шар 0 знає ТОЧНО)
    ⟹ per_layer_mse[1] = 7.4e8 × 6.769e-10 = 0.501  (шари 1..30 у нього НУЛІ)
🔒 ТРЕТЯ, НЕЗАЛЕЖНА ПЕРЕВІРКА: ALL-LAYERS у нас 0.7537, у нього 0.7437.
   Різниця = mean(true_0²)/32 ⟹ mean(true_0²) = 0.320.
   А зі STEEPEST: 0.501/1.58 = 0.317.  Збіг 1%.
⟹ 6.769e-10 — це НЕ його похибка, це шум істини на шарі 0.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) підлога шару 0 = Var(ReLU(W0 z))/1e9 відтворить 6.77e-10 у межах 10%   75%
 (б) підлога ФІНАЛЬНОГО шару = avg_variance/1e9 лежатиме в 1.5e-9…2.5e-9    70%
 (в) ⟹ лідер 3.628e-9 стоїть у межах ×2.5 від підлоги                      65%
"""
import glob
import math

import numpy as np
import pyarrow as pa

N_TRUTH = 1_000_000_000
DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/")

for split in ("mini", "full"):
    files = sorted(glob.glob(SRC + split + "/data-*.arrow"))
    av, fm2, n = [], [], 0
    for f in files:
        tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        av.append(np.asarray(tb.column("avg_variance")))
        yy = tb.column("final_means").combine_chunks()
        while not pa.types.is_primitive(yy.type):
            yy = yy.flatten()
        FM = np.asarray(yy).reshape(tb.num_rows, W_).astype(np.float64)
        fm2.append((FM * FM).mean(axis=1))
        n += tb.num_rows
    av = np.concatenate(av).astype(np.float64)
    fm2 = np.concatenate(fm2)
    floor = av / N_TRUTH
    print(f"\n=== {split}: {n} мереж ===")
    print(f"avg_variance   сер {av.mean():.6f}  мед {np.median(av):.6f}  "
          f"[{av.min():.4f} … {av.max():.4f}]")
    print(f"mean(true²)    сер {fm2.mean():.6f}   ⟹ ПЛОМБА: записано 0.9097")
    print(f"ПІДЛОГА MSE    сер {floor.mean():.6e}  мед {np.median(floor):.6e}")
    print(f"               [{floor.min():.4e} … {floor.max():.4e}]  розкид "
          f"×{floor.max()/floor.min():.2f}")
    print(f"відносна похибка підлоги: {100*math.sqrt(floor.mean()/fm2.mean()):.5f}%")

print("\n" + "=" * 70)
print("ОРІЄНТИРИ БОРДУ (final-layer MSE):")
for name, v in (("#1 dpskv5", 3.628e-9), ("#2 huang", 9.00e-9),
                ("#3 joe_wanza", 4.00e-9), ("#4 ednacob", 9.11e-8),
                ("МИ sub119r05", 1.855e-6)):
    print(f"  {name:<16s} {v:.4e}")
