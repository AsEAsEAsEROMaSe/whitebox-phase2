"""spec101 — СПЕКТРАЛЬНИЙ БЛЕНД: вінерівська вага ПО НАПРЯМКАХ замість скаляра.

ЩО НЕ ТАК ІЗ ТЕПЕРІШНІМ. Наш бленд:
    out = ours + W_BLEND·(a·m_deep − ours)          W_BLEND = 0.0798, СКАЛЯР
тобто ОДНА вага на всі 256 напрямків. Але шум МК АНІЗОТРОПНИЙ: його коваріація
є `Cov(out)/N`, а `Cov(out)` має ефективний ранг ~25 з 256 ⟹ у різних напрямках
шум відрізняється на ПОРЯДКИ. Скаляр усереднює це в одне число.

ПРАВИЛЬНА ВАГА (Вінер / байєсівське злиття двох оцінок):
    m̂ = m_deep + K·(mc − m_deep),   K = Σ_e (Σ_e + Σ_n)^{-1}
    Σ_n = Cov(out)/N_eff   ← ЗНАЄМО: замикання рахує Cov(out)
    Σ_e = коваріація похибки замикання ← параметризуємо 2 числами
У власному базисі `Cov(out)` це діагональ:
    k_i = σ²_e(i) / (σ²_e(i) + λ_i/N_eff)

ДВІ ПАРАМЕТРИЗАЦІЇ `Σ_e` (обидві — 1–2 параметри, не 2209):
    (A) ІЗОТРОПНА:      σ²_e(i) = e²           ⟹ k_i = e²/(e² + λ_i/N)
    (B) ПРОПОРЦІЙНА:    σ²_e(i) = e²·λ_i^ρ     ⟹ форма похибки слідує за C
🔒 при `ρ = 1` і сталому `N` варіант (B) дає СТАЛЕ `k` ⟹ **вироджується в наш
   скалярний бленд** ⟹ теперішній метод є ЧАСТКОВИМ ВИПАДКОМ, і спектральний
   не може програти in-sample. Вирок — ЛИШЕ на кросі A/B.

🔒 ПЛОМБА: при (B), ρ=1 підібране `e` мусить відтворити W_BLEND≈0.08 і ту саму
   MSE, що скалярний бленд ⟹ якщо ні, реалізація зламана.
🔒 КРОС A/B по мережах · 🔒 LOO.
🚨 Це ВИБІРКОВА гілка — вона покращує `V` (91% балу), тобто б'є туди, де вага.

ЦІНА: один `eigh` на 256×256 = 1.51e8 флопів = 0.055% бюджету. Плюс два
матвектори. Порівняно з бюджетом 2.72e11 — нуль.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒 (B) з ρ=1 відтворить скалярний бленд у межах 1%                85%
 (б) (A) ізотропна дасть КРАЩЕ за скаляр                              55%
 (в) виграш у MSE перевищить ×1.10                                    35%
 (г) крос A→B збереже щонайменше половину виграшу                     50%
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3072      # ~наш робочий бюджет

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)
print(f"spec101 · {NNET} мереж · {2*NS} семплів МК (антитетика)", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(Wg):
    """наша рекурсія; повертає m_deep і Cov(out) останнього шару"""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = Co + cp.diag(d - cp.diagonal(Co))
    return m, C            # C — Cov(ReLU(p_31)) за нашою формулою


rows = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    T = cp.asarray(FM[net].astype(np.float64))
    md, Cout = closure(W64)
    # пряма МК з вибілюванням входу (як у сабміті) + антитетика
    rng = cp.random.default_rng(1234 + net)
    z = rng.standard_normal((NS, W_), dtype=cp.float32)
    zc = z - z.mean(0, keepdims=True)
    cov = (zc.T @ zc) / NS
    Lw = cp.linalg.cholesky(cov.astype(cp.float64) + 1e-9 * cp.eye(W_))
    Li = cp.linalg.inv(Lw)
    zw = (zc.astype(cp.float64) @ Li.T).astype(cp.float32)
    a = cp.concatenate([zw, -zw], axis=0)
    for i in range(DEPTH):
        p = a @ Wg[i]
        a = cp.maximum(p, 0.0) if i < DEPTH - 1 else p
    mc = cp.maximum(p, 0.0).mean(0, dtype=cp.float64)
    ev, U = cp.linalg.eigh(Cout)
    ev = cp.maximum(ev, 1e-30)
    dg = cp.maximum(cp.diagonal(Cout), 1e-30)   # 🆕 ДІАГОНАЛЬ — без eigh
    d = U.T @ (mc - md)                     # різниця у власному базисі
    tt = U.T @ (T - md)                     # істина у тому ж базисі
    rows.append((cp.asnumpy(ev), cp.asnumpy(d), cp.asnumpy(tt),
                 cp.asnumpy(dg), cp.asnumpy(mc - md), cp.asnumpy(T - md)))
    print(f"  мережа {net} готова", flush=True)

EV = np.array([r[0] for r in rows])
D = np.array([r[1] for r in rows])
TT = np.array([r[2] for r in rows])
NEFF = 2 * NS


def mse_of(k):
    """k — матриця ваг (nets × 256) у власному базисі"""
    return float(np.mean((k * D - TT) ** 2))


ALL = np.arange(len(rows))
A, B = ALL[::2], ALL[1::2]


def mse_sel(k, sel):
    return float(np.mean((k[sel] * D[sel] - TT[sel]) ** 2))


print("\n── СКАЛЯРНИЙ БЛЕНД (наш) ──")
best = None
for w in np.arange(0.80, 1.001, 0.005):
    k = np.full_like(D, w)
    v = mse_sel(k, ALL)
    if best is None or v < best[0]:
        best = (v, w)
v_sc, w_sc = best
print(f"   оптимум ваги МК = {w_sc:.3f}  (наш _W_BLEND = 1−0.0798 = 0.920)")
print(f"   MSE = {v_sc:.6e}")

print("\n── (A) ІЗОТРОПНА Σ_e:  k_i = e²/(e² + λ_i/N) ──")
bestA = None
for le in np.arange(-14, -2, 0.25):
    e2 = 10.0 ** le
    k = e2 / (e2 + EV / NEFF)
    v = mse_sel(k, ALL)
    if bestA is None or v < bestA[0]:
        bestA = (v, e2)
vA, e2A = bestA
print(f"   оптимум e² = {e2A:.3e} · MSE = {vA:.6e} · ×проти скаляра "
      f"{v_sc/vA:.4f}")

print("\n── (B) ПРОПОРЦІЙНА Σ_e = e²·λ^ρ ──")
bestB = None
for rho in np.arange(0.0, 1.51, 0.1):
    for le in np.arange(-14, 2, 0.25):
        e2 = 10.0 ** le
        k = (e2 * EV ** rho) / (e2 * EV ** rho + EV / NEFF)
        v = mse_sel(k, ALL)
        if bestB is None or v < bestB[0]:
            bestB = (v, e2, rho)
vB, e2B, rhoB = bestB
print(f"   оптимум e² = {e2B:.3e} · ρ = {rhoB:.2f} · MSE = {vB:.6e} · "
      f"×проти скаляра {v_sc/vB:.4f}")

print("\n🔒 КРОС A/B (підбір на A, вирок на B):")
for nm, fn, grid in (
        ("скаляр", lambda p: np.full_like(D, p[0]),
         [(w,) for w in np.arange(0.80, 1.001, 0.005)]),
        ("(A) ізотроп", lambda p: p[0] / (p[0] + EV / NEFF),
         [(10.0 ** le,) for le in np.arange(-14, -2, 0.25)]),
        ("(B) проп.", lambda p: (p[0] * EV ** p[1])
         / (p[0] * EV ** p[1] + EV / NEFF),
         [(10.0 ** le, r) for r in np.arange(0.0, 1.51, 0.1)
          for le in np.arange(-14, 2, 0.25)])):
    bA = min(grid, key=lambda p: mse_sel(fn(p), A))
    print(f"   {nm:<12s} на B: {mse_sel(fn(bA), B):.6e}", flush=True)
# 🔒 LOO по мережах для (B)
nnet = len(rows)
wins = 0
for i in range(nnet):
    sel = np.array([j for j in range(nnet) if j != i])
    bl = min([(10.0 ** le, r) for r in np.arange(0.0, 1.51, 0.1)
              for le in np.arange(-14, 2, 0.25)],
             key=lambda p: mse_sel((p[0] * EV ** p[1])
                                   / (p[0] * EV ** p[1] + EV / NEFF), sel))
    bs = min([(w,) for w in np.arange(0.80, 1.001, 0.005)],
             key=lambda p: mse_sel(np.full_like(D, p[0]), sel))
    kb = (bl[0] * EV ** bl[1]) / (bl[0] * EV ** bl[1] + EV / NEFF)
    one = np.array([i])
    if mse_sel(kb, one) < mse_sel(np.full_like(D, bs[0]), one):
        wins += 1
DG = np.array([r[3] for r in rows])
DS = np.array([r[4] for r in rows])
TS = np.array([r[5] for r in rows])


def mse_diag(k, sel):
    return float(np.mean((k[sel] * DS[sel] - TS[sel]) ** 2))


print("\n── (C) ДІАГОНАЛЬНА, БЕЗ eigh: k_j = e²·d_j^ρ/(e²d_j^ρ + d_j/N) ──")
bc = None
for rho in np.arange(0.0, 1.51, 0.1):
    for le in np.arange(-14, 2, 0.25):
        e2 = 10.0 ** le
        k = (e2 * DG ** rho) / (e2 * DG ** rho + DG / NEFF)
        v = mse_diag(k, ALL)
        if bc is None or v < bc[0]:
            bc = (v, e2, rho)
print(f"   оптимум e²={bc[1]:.3e} ρ={bc[2]:.2f} · MSE {bc[0]:.6e} · "
      f"×проти скаляра {v_sc/bc[0]:.4f}")
gC = [(10.0 ** le, r) for r in np.arange(0.0, 1.51, 0.1)
      for le in np.arange(-14, 2, 0.25)]
bA_ = min(gC, key=lambda p: mse_diag((p[0] * DG ** p[1])
                                     / (p[0] * DG ** p[1] + DG / NEFF), A))
kB_ = (bA_[0] * DG ** bA_[1]) / (bA_[0] * DG ** bA_[1] + DG / NEFF)
scB = min([(w,) for w in np.arange(0.80, 1.001, 0.005)],
          key=lambda p: mse_diag(np.full_like(DS, p[0]), A))
print(f"   🔒 крос A→B: {mse_diag(kB_, B):.6e}  проти скаляра "
      f"{mse_diag(np.full_like(DS, scB[0]), B):.6e}  ⟹ "
      f"×{mse_diag(np.full_like(DS, scB[0]), B)/mse_diag(kB_, B):.4f}")
print(f"   💰 ЦІНА: НУЛЬ (diag(Cout) замикання вже має, eigh не потрібен)")

print(f"\n🔒 LOO по мережах: спектральний кращий за скаляр у {wins}/{nnet} фолдах")
print(f"🔒 ρ=1 вироджується в скаляр — вбудована пломба, тому вирок ЛИШЕ на кросі")
