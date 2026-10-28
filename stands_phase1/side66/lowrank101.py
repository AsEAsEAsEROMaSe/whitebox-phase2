"""lowrank101 — СТЕЛЯ РАНГ-`r` НЕГАУСОВОГО ОПИСУ СТАНУ. ВИМІР ДО РЕАЛІЗАЦІЇ.

ПІДСТАВА (indep101): 96% негаусовості передактивацій походить із ЗАЛЕЖНОСТІ
між нейронами (γ₄ переміш 0.0142 проти реальної 0.3947), а ефективний ранг
Cov(h) падає 165.6 → 2.8. ⟹ co-skewness, який С85 закрила за ціною тензора
256³, у ранг-r підпросторі коштує r³/6 чисел.

ПИТАННЯ, І ВОНО ПРО СТЕЛЮ, А НЕ ПРО РЕАЛІЗАЦІЮ:
   якщо дати ОРАКУЛЬНО спільний розподіл у r головних напрямках,
   скільки з похибки ОДНОГО КРОКУ це знімає?
Записане: похибка кроку з ТОЧНИМИ (μ,σ,ρ) = 0.139% для E[ReLU].

СХЕМА (точна алгебра, не апроксимація):
   p = h @ w  розкладаємо як  p = mp + ξ·B + e
       ξ = r головних компонент центрованого h (НЕГАУСОВІ, беремо з вибірки)
       B = Uᵣᵀ w                      — проєкція ваг на підпростір
       e = залишок, ГАУСІВ за побудовою (сума 256−r напрямків)
   ⟹ E[ReLU(p)] = E_ξ[ σ_e · ψ((mp + ξB)/σ_e) ],   ψ(t) = φ(t) + t·Φ(t)
   тобто ВНУТРІШНЄ математичне сподівання аналітичне, ЗОВНІШНЄ — по r-вимірному
   ядру. При r=0 формула ТОТОЖНО збігається з гаусовим замиканням ⟹ вбудована
   пломба.

ПОРІВНЮЄМО ТРИ РЕЧІ НА ОДНИХ І ТИХ САМИХ ДАНИХ:
   G     повністю гаусова модель (= наше замикання при ТОЧНИХ моментах)
   R_r   ранг-r негаусове ядро + гаусів залишок
   🔒 CHU  ядро ξ від ІНШОЇ мережі (контроль «чужа») — має зламати

РОЗДІЛЕННЯ ВИБІРОК: базис і ядро будуються на половині A, істина міряється
на половині B ⟹ спільного шуму немає (закон «сіди істини й оракула — розділяти»).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) r=0 відтворить G до 4 знаків (пломба)                          95%
 (б) r=4 зніме >30% похибки кроку на шарі 31                        45%
 (в) виграш РОСТИМЕ з глибиною (там ранг найменший)                 70%
 (г) 🔒 чужа зламає (похибка ГІРША за G)                            85%
 (д) на шарі 1 виграшу не буде майже зовсім (ранг 165)              80%

ВИРОК: якщо r=4 знімає <20% на шарі 31 ⟹ гілка мертва до написання коду,
бо реалізація коштує щонайменше проєкцію + несення ядра через 32 шари.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")   # ПОСТІЙНИЙ шлях: /tmp чиститься

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
LS = [1, 8, 16, 24, 31]
RS = [0, 1, 2, 4, 8, 16]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"lowrank101 · {NNET} мереж · {NS} семплів · шари {LS} · ранги {RS}",
      flush=True)


def psi(t):
    """ψ(t) = φ(t) + t·Φ(t) ⟹ E[ReLU(σ·(t+ν))] = σ·ψ(t), ν~N(0,1)"""
    return cp.exp(-0.5 * t * t) / SQ2PI + t * ndtr(t)


def acts(Wg, layer, ns, seed):
    """реальні активації h ПЕРЕД шаром `layer` (антитетичні пари)"""
    rng = cp.random.default_rng(seed)
    out = []
    n = 0
    while n < ns:
        nb = min(32768, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        h = cp.concatenate([z, -z], axis=0)
        for i in range(layer):
            h = cp.maximum(h @ Wg[i], 0.0)
        out.append(h)
        n += nb
    return cp.concatenate(out, axis=0).astype(cp.float64)


def eval_model(hA, hB, w, r, U_foreign=None):
    """похибка ранг-r негаусової моделі, побудованої на A, проти істини на B.
    r=0 ⟹ повністю гаусова модель. Повертає RMS відносної похибки по нейронах."""
    mA = hA.mean(0)
    cA = hA - mA
    mp = mA @ w                                   # ТОЧНО, як у замиканні
    cov = (cA.T @ cA) / cA.shape[0]
    var_tot = cp.einsum("ij,ia,ja->a", cov, w, w)  # ТОЧНА повна дисперсія

    if r == 0:
        s = cp.sqrt(cp.maximum(var_tot, 1e-30))
        pred = s * psi(mp / s)
    else:
        if U_foreign is None:
            lam, V = cp.linalg.eigh(cov)
            U = V[:, -r:]                          # r головних напрямків
        else:
            U = U_foreign
        xi = cA @ U                                # (N, r) НЕГАУСОВЕ ядро
        B = U.T @ w                                # (r, 256)
        # дисперсія ядра рахується з РЕАЛЬНОЇ Cov(ξ), не з припущення
        cxi = (xi.T @ xi) / xi.shape[0]
        var_k = cp.einsum("kl,ka,la->a", cxi, B, B)
        var_e = cp.maximum(var_tot - var_k, 1e-12)  # залишок ГАУСІВ
        se = cp.sqrt(var_e)
        # E[ReLU] = E_ξ[ σ_e · ψ((mp + ξB)/σ_e) ] — зовнішнє по ядру
        acc = cp.zeros(W_, dtype=cp.float64)
        nb = 16384
        for k in range(0, xi.shape[0], nb):
            q = xi[k:k + nb] @ B                   # (nb, 256)
            acc += psi((mp + q) / se).sum(0)
        pred = se * (acc / xi.shape[0])

    truth = cp.maximum(hB @ w, 0.0).mean(0)
    nrm = cp.sqrt((truth * truth).mean())
    return float(cp.sqrt(((pred - truth) ** 2).mean()) / nrm)


res = {}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    Wf = [cp.asarray(np.ascontiguousarray(Wall[(net + 1) % NNET][i],
                                          dtype=np.float32))
          for i in range(DEPTH)]
    for L in LS:
        hA = acts(Wg, L, NS // 2, 4242 + net)
        hB = acts(Wg, L, NS // 2, 999983 + net)
        w = Wg[L].astype(cp.float64)
        for r in RS:
            res.setdefault((L, r), []).append(eval_model(hA, hB, w, r))
        # 🔒 чужа: базис із мережі-сусіда, ядро з НАШИХ активацій у ньому
        hF = acts(Wf, L, NS // 2, 4242 + net)
        cF = hF - hF.mean(0)
        covF = (cF.T @ cF) / cF.shape[0]
        _, VF = cp.linalg.eigh(covF)
        res.setdefault((L, -4), []).append(
            eval_model(hA, hB, w, 4, U_foreign=VF[:, -4:]))
        del hA, hB, hF
        cp.get_default_memory_pool().free_all_blocks()
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'шар':>4s}" + "".join(f"{'r=' + str(r):>11s}" for r in RS)
      + f"{'🔒чужа r=4':>12s}{'×виграш r=4':>13s}")
print("=" * 96)
for L in LS:
    base = np.mean(res[(L, 0)])
    line = f"{L:>4d}"
    for r in RS:
        line += f"{100 * np.mean(res[(L, r)]):>10.4f}%"
    line += f"{100 * np.mean(res[(L, -4)]):>11.4f}%"
    line += f"{base / np.mean(res[(L, 4)]):>13.3f}"
    print(line)
print("\n🔒 ПЛОМБА: колонка r=0 — це РІВНО гаусове замикання при ТОЧНИХ моментах;")
print("   записане число для одного кроку = 0.139%")
print("🔒 ЧУЖА мусить бути ГІРШОЮ за r=0, інакше виграш — артефакт розмірності")
print("ВИРОК: r=4 знімає <20% на шарі 31 ⟹ гілка мертва до написання коду")
