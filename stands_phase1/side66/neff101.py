"""neff101 — `1/n`-ЧЛЕН ЯВНО З КОНКРЕТНОЇ `W`. КРОК `E` ПЛАНУ ДО 1e-8.

⚠️ СТАТУС: НАПИСАНО, НЕ ЗАПУЩЕНО (сесія 101 обірвалась на стисненні контексту).
   Це ПЕРША ДІЯ наступної сесії. Запуск — унизу файлу в докстрінгу.

ВИВЕДЕННЯ (з тотожності, не з голови). Перечитавши власний код замикання
(`estimator_sub119r05.py:257-279`), видно точний розподіл ролей:
    cp = einsum("ij,ia,jb->ab", C, w, w)   # ТОЧНО — лінійна алгебра
    mp = matmul(m, w)                      # ТОЧНО
    m  = mp*Φ(t) + s*φ(t)                  # ← гаусове припущення #1
    C  = cp*Φ⊗Φ + 0.5*cp²*(φ/s)⊗(φ/s)      # ← #2 (Мелер K=2)
    d  = (mp²+s²)Φ + mp*s*φ − m²           # ← #3
⟹ ПЕРЕНОС ТОЧНИЙ. Уся похибка — в тому, що `p_a = Σ_j W_ja h_j` вважається
ГАУСОВИМ. Для суми незалежних доданків
    κ₄(p_a) = Σ_j W_ja⁴ κ₄(h_j),   γ₄(p_a) = κ₄(p_a)/Var(p_a)²
при однакових `h_j` це дає РІВНО
    γ₄(p_a) = γ₄(h) / n_eff(a),   n_eff(a) = (Σ_j W_ja²)² / Σ_j W_ja⁴
тобто **participation ratio ваг** — величина, обчислювана ПРЯМО З `W`, ціною
2 матвектори на шар (~131k флопів = 0.00005% бюджету).

🔑 ЦЕ ПОЯСНЮЄ, ЧОМУ ТАБЛИЧНА `γ₄` ПРОВАЛИЛА КОНТРОЛЬ «ЧУЖА» (С101, gam101):
   таблиця дає СЕРЕДНЄ по мережах, а формула вимагає ПОНЕЙРОННОГО значення.
   І це узгоджується з [f]finding-gamma3-per-network-is-real-the-across-network-
   average-hid-it: понейронні кумулянти РЕАЛЬНІ, усереднені — артефакт.

🚨 ДВІ ВЕРСІЇ, БО `h_j` НЕ НЕЗАЛЕЖНІ:
   naive:  n_eff = (Σ W²)² / Σ W⁴                  — ігнорує C
   diag :  n_eff = (Σ W² C_jj)² / Σ W⁴ C_jj²       — враховує різні дисперсії
🚨 ОРІЄНТИР, ЯКИЙ ВИРІШУЄ: для He-ваг `naive` дає аналітично n/3 = 85.3 з
   розкидом ~6%, а ЗМІРЯНА ефективна ширина ЦГТ на глибині — **25…36** (С99,
   підтверджено С100 незалежно: 86.6 → 25.0). Якщо `naive` дасть 85 — він НЕ
   пояснює спостережене ⟹ вирішує `diag`, або жодна.

ПЕРЕДБАЧЕННЯ (ЗАПИСАНІ ДО ПРОГОНУ):
 (а) `naive` дасть ≈85 і майже не варіюватиме між нейронами        80%
 (б) `diag` дасть 25…50 на глибині                                 50%
 (в) corr(γ₄ реальна, 1/n_eff_diag) по нейронах перевищить +0.3    35%
 (г) 🔒 чужа W дасть |corr| < 0.05                                 85%

ЩО РОБИТИ З РЕЗУЛЬТАТОМ:
 · якщо (в) справдиться ⟹ маємо ДЕТЕРМІНОВАНИЙ понейронний предиктор
   негаусовості з `W` ⟹ наступний крок: вставити `γ₄(a) = γ₄_табл/n_eff(a)`
   в Еджворт-поправку рекурсії і зміряти ПОВНИМ оцінювачем ПАРНО (`--format json`).
   Орієнтир зверху: `edgm` (γ₄ поверх ІСТИННИХ m,C) дав ×22.8 = бал 8.16e-09.
 · якщо (в) провалиться ⟹ крок `E` мертвий, і разом із ним крок `D`
   (звільнення бюджету) втрачає сенс — витрачати ×16 бюджету НЕМА НА ЩО.

ЗАПУСК:
    SP=/tmp/claude-1000/-mnt-win-d-llm-White-Box/<session>/scratchpad
    CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 \
    nohup /usr/bin/python3 -u neff101.py 8 524288 > $SP/neff101.log 2>&1 &
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG, KAPPA = 8, 0.9985
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 524288
LS = [1, 4, 8, 16, 24, 31]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"neff101 · {NNET} мереж · {NS} семплів · шари {LS}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure_diag(Wg):
    """наше замикання; повертає diag(C) активацій ПЕРЕД кожним шаром"""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    dg = [cp.diagonal(C).copy()]
    for i in range(DEPTH - 1):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn
        dg.append(cp.diagonal(C).copy())
    return dg


def true_g4(Wg, layer, ns):
    """реальна γ₄ ПЕРЕДактивацій `p` шару `layer`, по нейронах"""
    rng = cp.random.default_rng(20260810)
    S = [cp.zeros(W_, dtype=cp.float64) for _ in range(4)]
    n = 0
    while n < ns:
        nb = min(16384, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(layer):
            a = cp.maximum(a @ Wg[i].astype(cp.float32), 0.0)
        p = (a @ Wg[layer].astype(cp.float32)).astype(cp.float64)
        S[0] += p.sum(0)
        S[1] += (p * p).sum(0)
        S[2] += (p ** 3).sum(0)
        S[3] += (p ** 4).sum(0)
        n += nb
    N = 2 * ns
    m1 = S[0] / N
    m2 = S[1] / N
    m3 = S[2] / N
    m4 = S[3] / N
    v = m2 - m1 * m1
    k4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 * m1 - 3 * m1 ** 4 - 3 * v * v
    return k4 / cp.maximum(v * v, 1e-30)


rows = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    dg = closure_diag(Wg)
    for L in LS:
        w = Wg[L]
        w2 = w * w
        w4 = w2 * w2
        naive = (w2.sum(0) ** 2) / cp.maximum(w4.sum(0), 1e-300)
        c = dg[L]
        diagn = ((w2 * c[:, None]).sum(0) ** 2) / \
            cp.maximum((w4 * (c * c)[:, None]).sum(0), 1e-300)
        g4 = true_g4(Wg, L, NS)
        rows.append((net, L, cp.asnumpy(naive), cp.asnumpy(diagn), cp.asnumpy(g4)))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'шар':>4s}{'n_eff naive':>13s}{'розкид':>9s}{'n_eff diag':>12s}"
      f"{'розкид':>9s}{'γ₄ сер':>9s}{'corr(γ₄,1/naive)':>18s}"
      f"{'corr(γ₄,1/diag)':>17s}{'🔒чужа':>9s}")
print("=" * 100)
for L in LS:
    R = [r for r in rows if r[1] == L]
    na = np.concatenate([r[2] for r in R])
    di = np.concatenate([r[3] for r in R])
    g4 = np.concatenate([r[4] for r in R])
    cn = np.mean([np.corrcoef(1.0 / r[2], r[4])[0, 1] for r in R])
    cd = np.mean([np.corrcoef(1.0 / r[3], r[4])[0, 1] for r in R])
    cf = np.mean([np.corrcoef(1.0 / R[(k + 1) % len(R)][3], R[k][4])[0, 1]
                  for k in range(len(R))]) if len(R) > 1 else float('nan')
    print(f"{L:>4d}{na.mean():>13.2f}{100*na.std()/na.mean():>8.1f}%"
          f"{di.mean():>12.2f}{100*di.std()/di.mean():>8.1f}%"
          f"{g4.mean():>9.4f}{cn:>18.4f}{cd:>17.4f}{cf:>9.4f}")
print("\n🚨 ОРІЄНТИР: `naive` для He-ваг = n/3 = 85.3 ⟹ якщо він і вийде 85,")
print("   він НЕ пояснює зміряну ефективну ширину ЦГТ 25…36 (записано С99)")
print("🔒 КОНТРОЛЬ «ЧУЖА»: n_eff сусідньої мережі проти нашої γ₄ ⟹ має бути ≈0")
