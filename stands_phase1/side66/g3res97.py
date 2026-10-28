"""g3res97 — ЗНЯТИ СУПЕРЕЧНІСТЬ ПО ПОНЕЙРОННОМУ γ₃ НА ОДНАКОВОМУ ЗРІЗІ.

════════════════ СУПЕРЕЧНІСТЬ ════════════════
`[ЗМІРЯНО gamtab97, 16 мереж, 2^18 семплів, ПЕРЕДактивації]` понейронний γ₃:
    медіана СТАТИСТИЧНО НУЛЬ на КОЖНОМУ шарі, |·| < 0.036
`[ЗМІРЯНО з датасету higher-moments, mlp_00000, шар 31]` медіана −0.1071,
    медіана |·| = 0.3887
Розрив ×3 у ПІДПИСАНІЙ медіані. Поки не знято — НІ ОДНЕ з двох не цитується,
і еталон із N=1e8 (шум ×16 нижчий) стоїть невикористаний.

════════════════ ЩО РОБИМО ════════════════
Той самий об'єкт, та сама МЕРЕЖА, той самий шар, два незалежні прилади:
  A. датасет keenanpepper: γ₃ з `pre_mean, pre_m2, pre_m3`, N = 1e8
  B. власна МК на ТИХ САМИХ вагах, float64-акумуляція, N = 2^18…2^20
Спершу ПЛОМБА ТОТОЖНОСТІ МЕРЕЖІ: `official_alm` датасету проти
`all_layer_means` публічного банку для рядка з тим самим `mlp_name`.
Без цієї пломби порівняння безглузде (три хвороби контролю: «не той об'єкт»).

════════════════ ГІПОТЕЗИ ПРО ПРИЧИНУ (ДО ЗАПУСКУ) ════════════════
 (H1) різні ШАРИ: «|·|<0.036» — це середнє по шарах, а на шарі 31
      справжня медіана більша                                     35%
 (H2) різні ОБ'ЄКТИ: датасет `pre_*` це передактивації ПЕРЕД шаром L,
      gamtab97 — ПІСЛЯ множення на W[L] ⟹ зсув індексу на 1        25%
 (H3) катастрофічне скорочення у float32: γ₃ = (m3−3μm2+2μ³)/σ³ при
      t≈3.5 має κ ≈ 2.8e3 ⟹ похибка ~3e-5, ЗАМАЛО. Тому 15%,
      але перевірити ПРЯМО (порахувати κ на реальних числах)       15%
 (H4) розкид МІЖ мережами: mlp_00000 нетипова                      15%
 (H5) моя помилка в першому читанні датасету (взяла λ₂₁-нормування
      замість γ₃)                                                  10%
Передбачення: A і B зійдуться в межах шуму на ОДНАКОВОМУ шарі, і винною
виявиться агрегація по шарах або зсув індексу — тобто H1/H2 разом 60%.
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

SCR = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NS = 1 << (int(sys.argv[1]) if len(sys.argv) > 1 else 18)
CH = 1 << 13

t0 = time.time()
d = np.load(f"{SCR}/mlp_00000.npz")
name = str(d["name"])
gidx = int(d["global_index"])
print(f"g3res97 · {name} · датасет N={int(d['n_samples']):,} · "
      f"власна МК N={NS:,}", flush=True)

# ─────────────────── ПЛОМБА ТОТОЖНОСТІ МЕРЕЖІ ───────────────────
W = None
alm_bank = None
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    ids = t.column("mlp_id").to_pylist()
    if gidx not in ids:
        continue
    i = ids.index(gidx)
    print(f"  мережа {name} = mlp_id {gidx} = "
          f"'{t.column('mlp_name').to_pylist()[i]}'", flush=True)
    x = t.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    W = np.asarray(x).reshape(t.num_rows, 32, 256, 256)[i].astype(np.float32)
    y = t.column("all_layer_means").combine_chunks()
    while not pa.types.is_primitive(y.type):
        y = y.flatten()
    alm_bank = np.asarray(y).reshape(t.num_rows, 32, 256)[i]
    break

assert W is not None, "мережу не знайдено в банку"
alm_ds = d["official_alm"]
rel = np.abs(alm_ds - alm_bank).max() / np.abs(alm_bank).max()
print(f"🔒 ПЛОМБА ТОТОЖНОСТІ: max|official_alm − all_layer_means|/max|alm| "
      f"= {rel:.3e}   {'✅ та сама мережа' if rel < 1e-6 else '❌ РІЗНІ'}")
if rel >= 1e-6:
    sys.exit("зупиняюсь: об'єкти різні, порівнювати нема сенсу")


def gam(mean, m2, m3, m4):
    """γ₃, γ₄ із СИРИХ нецентральних моментів + число обумовленості."""
    mean = mean.astype(np.float64)
    m2 = m2.astype(np.float64)
    m3 = m3.astype(np.float64)
    m4 = m4.astype(np.float64)
    var = m2 - mean ** 2
    s = np.sqrt(np.maximum(var, 1e-300))
    num3 = m3 - 3 * mean * m2 + 2 * mean ** 3
    num4 = m4 - 4 * mean * m3 + 6 * mean ** 2 * m2 - 3 * mean ** 4
    kap3 = np.abs(m3) + 3 * np.abs(mean * m2) + 2 * np.abs(mean) ** 3
    kap3 = kap3 / np.maximum(np.abs(num3), 1e-300)
    return num3 / s ** 3, num4 / s ** 4 - 3.0, kap3


# ─────────────────── A. ДАТАСЕТ (N = 1e8) ───────────────────
g3_ds, g4_ds, cond = gam(d["pre_mean"], d["pre_m2"], d["pre_m3"], d["pre_m4"])
g3_post, _, _ = gam(d["mean"], d["m2"], d["m3"], d["m4"])

# ─────────────────── B. ВЛАСНА МК НА ТИХ САМИХ ВАГАХ ───────────────────
S = np.zeros((32, 4, 256), dtype=np.float64)
rng = np.random.default_rng(970109)
for o in range(0, NS, CH):
    z = rng.standard_normal((CH, 256), dtype=np.float32)
    h = z
    for L in range(32):
        p = (h @ W[L]).astype(np.float64)
        S[L, 0] += p.sum(0)
        S[L, 1] += (p * p).sum(0)
        S[L, 2] += (p ** 3).sum(0)
        S[L, 3] += (p ** 4).sum(0)
        h = np.maximum(p, 0.0).astype(np.float32)
print(f"  МК зроблено ({time.time()-t0:.0f} с)", flush=True)
S /= NS
g3_mc, g4_mc, _ = gam(S[:, 0], S[:, 1], S[:, 2], S[:, 3])

# ─────────────────── ЗВІРКА ПОШАРОВО ───────────────────
print("\n" + "=" * 96)
print("ПОНЕЙРОННИЙ γ₃ ПЕРЕДАКТИВАЦІЙ — ДАТАСЕТ (1e8) ПРОТИ ВЛАСНОЇ МК, "
      "ОДНА Й ТА САМА МЕРЕЖА")
print("=" * 96)
print(f"{'шар':>4s}{'med g3 ДС':>12s}{'med g3 МК':>12s}{'Δ':>10s}"
      f"{'med|g3| ДС':>12s}{'med|g3| МК':>12s}"
      f"{'med g4 ДС':>11s}{'med g4 МК':>11s}{'κ(f32)':>10s}")
for L in range(32):
    print(f"{L:>4d}{np.median(g3_ds[L]):>12.4f}{np.median(g3_mc[L]):>12.4f}"
          f"{np.median(g3_ds[L])-np.median(g3_mc[L]):>10.4f}"
          f"{np.median(np.abs(g3_ds[L])):>12.4f}"
          f"{np.median(np.abs(g3_mc[L])):>12.4f}"
          f"{np.median(g4_ds[L]):>11.4f}{np.median(g4_mc[L]):>11.4f}"
          f"{np.median(cond[L]):>10.1f}")

print("\n" + "-" * 96)
print("ПОНЕЙРОННО, ШАР ЗА ШАРОМ: КОРЕЛЯЦІЯ ДВОХ ПРИЛАДІВ (тест «той самий "
      "об'єкт?»)")
for L in (1, 4, 8, 16, 24, 31):
    c = np.corrcoef(g3_ds[L], g3_mc[L])[0, 1]
    sd = float(np.std(g3_ds[L] - g3_mc[L]))
    print(f"  шар {L:>2d}: corr = {c:>7.4f} · std(різниці) = {sd:.4f} · "
          f"очік. шум МК = {np.sqrt(6.0/NS):.4f}")

print("\n" + "-" * 96)
print("ЗСУВ ІНДЕКСУ (H2): corr(g3_ds[L], g3_mc[L±1]) — чи не переплутано шар")
for L in (8, 16, 24, 30):
    print(f"  L={L:>2d}: [L]={np.corrcoef(g3_ds[L], g3_mc[L])[0,1]:>7.4f}"
          f"  [L-1]={np.corrcoef(g3_ds[L], g3_mc[L-1])[0,1]:>7.4f}"
          f"  [L+1]={np.corrcoef(g3_ds[L], g3_mc[L+1])[0,1]:>7.4f}")

print("\n" + "-" * 96)
print("ПОСТактивації (для повноти — інший об'єкт, НЕ порівнювати з gamtab97):")
print("  med g3(post):  " + " ".join(
    f"L{L}={np.median(g3_post[L]):.3f}" for L in (1, 8, 16, 24, 31)))

print("\n" + "=" * 96)
print("АГРЕГАТИ, ЩО ЇХ ЦИТУЮТЬ:")
allds = np.concatenate([g3_ds[L] for L in range(1, 32)])
allmc = np.concatenate([g3_mc[L] for L in range(1, 32)])
print(f"  по УСІХ шарах 1..31 разом:  med(g3) ДС = {np.median(allds):+.4f} · "
      f"МК = {np.median(allmc):+.4f}")
print(f"                              med|g3| ДС = "
      f"{np.median(np.abs(allds)):.4f} · МК = {np.median(np.abs(allmc)):.4f}")
print(f"  МАКСИМУМ |med(g3)| по шарах: ДС = "
      f"{max(abs(np.median(g3_ds[L])) for L in range(32)):.4f} · "
      f"МК = {max(abs(np.median(g3_mc[L])) for L in range(32)):.4f}")
print(f"\n  час {time.time()-t0:.0f} с")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/g3res97.npz",
         g3_ds=g3_ds, g3_mc=g3_mc, g4_ds=g4_ds, g4_mc=g4_mc, cond=cond)
