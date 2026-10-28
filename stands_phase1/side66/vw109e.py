"""vw109e — дві незміряні величини з розбору «однієї матриці не існує».

(1) ПРОФІЛЬ ФЛУКТУАЦІЇ ПО ГЛИБИНІ: `‖M−M̄‖/‖M̄‖` зміряно лише для повного ланцюга
    (9.98). Як воно росте з глибиною — не мірялось; був лише здогад, що на мілких
    шарах менше.

(2) НАСКІЛЬКИ `M̄` ВІДРІЗНЯЄТЬСЯ ВІД СЕРЕДНЬОПОЛЬОВОГО ЛАНЦЮГА. Наше замикання
    будує `B = W₀·diag(Φ₀)·W₁·diag(Φ₁)⋯`, тобто ДОБУТОК СЕРЕДНІХ, тоді як `M̄` є
    СЕРЕДНІМ ДОБУТКУ. Різниця між ними — це рівно накопичена кореляція затворів
    між шарами, і вона ніколи не мірялась.

    Два варіанти `Φ`, і різниця між ними теж інформативна:
      Φ_emp  — вибіркова частка відкриттів ⟹ ізолює САМУ кореляцію затворів
      Φ_clo  — з двомоментного замикання ⟹ додає власну похибку замикання

🔒 ПЛОМБИ
 E1 ШАР 0 МУСИТЬ ДАТИ ТОЧНИЙ НУЛЬ. `M⁽⁰⁾ = W₀D₀`, отже `E[M⁽⁰⁾] = W₀·diag(E[D₀])`
    — середнє добутку і добуток середніх збігаються, бо множник ОДИН.
    Розрив може з'явитись лише з шару 1.
 E2 ПЕРЕМІШУВАННЯ МАСОК ПОШАРОВО (маска кожного шару береться від свого
    випадкового перемішування семплів) знищує міжшарові кореляції затворів,
    лишаючи ті самі маргіналі ⟹ розрив мусить ВПАСТИ до шуму.
 E3 Φ_emp на шарі 0 мусить дорівнювати 0.5 (вхід симетричний).
 🆕 E4 ПЛОМБА МАСШТАБУВАННЯ (додано ПІСЛЯ першого прогону, ДО другого).
    Перемішування не може дати РІВНО нуль при скінченному `N`: незалежні
    перестановки лишають випадкову кореляцію порядку `1/√N`. Отже питання
    «чи розрив ПОВНІСТЮ з міжшарової кореляції» вирішує НЕ величина
    `gap_shuf`, а її ПОКАЗНИК по `N`.
       gap_shuf ∝ N^(−0.5)  ⟹  це шум ⟹ істинний перемішаний розрив = 0
                                 ⟹ реальний розрив ЦІЛКОМ із кореляції затворів
       показник ≈ 0         ⟹  перемішування лишає РЕАЛЬНИЙ розрив
                                 ⟹ частина розриву НЕ з міжшарової кореляції
    📣 передбачення ДО другого прогону: показник ляже в [−0.60, −0.40]   65%
       (за N 8192→2048 це `gap_shuf`(31) 0.2713 → 0.54)
    🔒 контроль у тому самому прогоні: `gap_emp` МУСИТЬ стояти (показник ≈ 0),
       інакше падає не шум, а сам об'єкт.

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ
    флуктуація росте монотонно з глибиною                        75%
    розрив M̄ проти середньопольового > 50% уже на шарі 8         60%
    перемішування масок валить розрив нижче 10% від реального    80%
"""
import glob
import json
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
NPROBE = 32
CK = (0, 1, 2, 4, 8, 16, 24, 31)
OUT = f"/home/trim/wblogs/vw109e_N{NSAMP}_P{NPROBE}.json"   # 🚨 ім'я НЕСЕ N

import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 ВИДНО НЕ ОДНУ КАРТУ — вихід.", flush=True)
    sys.exit(2)

Wall = []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall.append(np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_))
Wall = np.concatenate(Wall, 0)
print(f"vw109e · {NNET} мереж · N={NSAMP} · проб={NPROBE}", flush=True)


def closure_phi(W):
    """Φ(t) з двомоментного замикання; Φ на індексі i описує затвор D_i."""
    m = cp.zeros(W_, dtype=cp.float64)
    C = (W[0].T @ W[0]).astype(cp.float64)
    out = []
    for i in range(DEPTH):
        s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
        t = m / s
        Ph = ndtr(t)
        ph = cp.exp(-0.5 * t * t) / SQ2PI
        out.append(Ph.astype(cp.float32))
        ma = s * (ph + t * Ph)
        q = s * s * ((1.0 + t * t) * Ph + t * ph)
        rho = C / cp.maximum(cp.outer(s, s), 1e-300)
        Ca = cp.outer(Ph, Ph) * C + 0.5 * cp.outer(ph, ph) * rho * rho \
            * cp.outer(s, s)
        cp.fill_diagonal(Ca, q - ma * ma)
        if i == DEPTH - 1:
            break
        Wi = W[i + 1].astype(cp.float64)
        m = Wi.T @ ma
        C = Wi.T @ Ca @ Wi
    return out


def sweep(W, masks, U):
    """M̄ і флуктуація через проби, на контрольних шарах."""
    v = cp.broadcast_to(U[None], (NSAMP, NPROBE, W_)).copy()
    mb, fl = {}, {}
    for i in range(DEPTH):
        v = (v.reshape(-1, W_) @ W[i]).reshape(NSAMP, NPROBE, W_)
        v = v * masks[i][:, None, :]
        if i in CK:
            X = v.reshape(NSAMP, -1)
            xb = X.mean(0)
            mb[i] = xb.copy()
            fl[i] = float(cp.sqrt(cp.mean(cp.sum((X - xb) ** 2, 1)))
                          / cp.maximum(cp.sqrt(cp.sum(xb * xb)), 1e-30))
            del X
    del v
    return mb, fl


def meanfield(W, PH, U):
    """B = U·W₀·diag(Φ₀)·W₁·diag(Φ₁)⋯ — ДОБУТОК СЕРЕДНІХ."""
    v = U.copy()
    out = {}
    for i in range(DEPTH):
        v = (v @ W[i]) * PH[i][None, :]
        if i in CK:
            out[i] = v.reshape(-1).copy()
    return out


rows = []
for net in range(NNET):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
         for L in range(DEPTH)]
    rng = cp.random.default_rng(9109 + net)
    z = rng.standard_normal((NSAMP, W_), dtype=cp.float32)
    U = rng.standard_normal((NPROBE, W_), dtype=cp.float32)
    U /= cp.sqrt(cp.sum(U * U, 1, keepdims=True))

    masks, a = [], z
    for i in range(DEPTH):
        p = a @ W[i]
        mk = p > 0
        masks.append(mk)
        a = p * mk
    PHe = [mk.mean(0, dtype=cp.float32) for mk in masks]
    PHc = closure_phi([w.astype(cp.float64) for w in W])

    mb, fl = sweep(W, masks, U)
    Be = meanfield(W, PHe, U)
    Bc = meanfield(W, PHc, U)
    # E2: маски пошарово ПЕРЕМІШАНІ ⟹ міжшарові кореляції знищені
    pm = [cp.asarray(np.random.default_rng(700 + 13 * i + net)
                     .permutation(NSAMP)) for i in range(DEPTH)]
    mbs, _ = sweep(W, [masks[i][pm[i]] for i in range(DEPTH)], U)

    r = {"fluct": {}, "gap_emp": {}, "gap_clo": {}, "gap_shuf": {},
         "phi0": float(PHe[0].mean()),
         "phi_clo0": float(PHc[0].mean())}
    for i in CK:
        n = float(cp.sqrt(cp.sum(mb[i] ** 2)))
        r["fluct"][str(i)] = fl[i]
        r["gap_emp"][str(i)] = float(cp.sqrt(cp.sum((mb[i] - Be[i]) ** 2))) / n
        r["gap_clo"][str(i)] = float(cp.sqrt(cp.sum((mb[i] - Bc[i]) ** 2))) / n
        ns = float(cp.sqrt(cp.sum(mbs[i] ** 2)))
        r["gap_shuf"][str(i)] = (
            float(cp.sqrt(cp.sum((mbs[i] - Be[i]) ** 2))) / max(ns, 1e-30))
    rows.append(r)
    print(f"  ✅ {net}: шар0 розрив {r['gap_emp']['0']:.2e} · "
          f"шар8 {r['gap_emp']['8']:.3f} · шар31 {r['gap_emp']['31']:.3f} · "
          f"🔒переміш31 {r['gap_shuf']['31']:.3f} · Φ₀ {r['phi0']:.4f}",
          flush=True)
    del W, z, masks, mb, mbs
    cp.get_default_memory_pool().free_all_blocks()

json.dump(rows, open(OUT, "w"))
g = lambda k, i: np.array([r[k][str(i)] for r in rows])       # noqa: E731
print("\n шар   ‖M−M̄‖/‖M̄‖    розрив M̄ vs добуток середніх")
print("                      Φ вибіркова   Φ замикання   🔒маски переміш.")
for i in CK:
    print(f"  {i:>3}    {g('fluct',i).mean():8.2f}      "
          f"{g('gap_emp',i).mean():7.4f}      {g('gap_clo',i).mean():7.4f}"
          f"      {g('gap_shuf',i).mean():7.4f}")
print(f"\n🔒 E1 шар 0, розрив = {g('gap_emp',0).mean():.2e} (мусить бути 0)")
print(f"🔒 E3 Φ вибіркова на шарі 0 = "
      f"{np.mean([r['phi0'] for r in rows]):.5f} (мусить бути 0.5)")
