"""scal89 — ПЛОМБА МАСШТАБУВАННЯ: ZCA проти Холецького — це шум чи ЗМІЩЕННЯ?

ПРОБЛЕМА: `mse89` дав ×0.9346, але парний t = −1.18 ⟹ вимір нічого не
розсудив. А ризик реальний: вибілювання НЕЛІНІЙНЕ (матриця залежить від самої
вибірки), тож ZCA і Холецький дають РІЗНІ зміщення, а `b²` — це 17.5% MSE.

ІДЕЯ ПЛОМБИ: беремо ту саму вибірку `z` і два вибілювання. Якщо різниця
прогнозів — це МК-шум, вона мусить падати як `1/√N` (учетверо на кожному
×16). Якщо там різниця ЗМІЩЕНЬ — вона вийде на ПЛАТО.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (шанс 60%): вийде на плато на рівні ~1e-4 відносних,
бо вибілювання ортогональним поворотом НЕ комутує з `relu` — а саме
несиметричність `relu` і робить обидва зміщення різними.
Якщо ж падає як 1/√N — Холецький безпечний і його можна лишити.
"""
import numpy as np

import flopscope.numpy as fnp

W = np.load("fs_w.npy", mmap_mode="r")
n = 256
INV_SQRT_2PI = 0.3989422804014327


def whiten_zca(cov):
    ev, U = np.linalg.eigh(cov)
    return (U * (1.0 / np.sqrt(np.maximum(ev, 1e-8)))) @ U.T


def whiten_chol(cov):
    return np.linalg.inv(np.linalg.cholesky(cov)).T


def run(ws, npair, mode, seed):
    """Спрощений МК-прохід із антитетикою — та сама схема, що в сабміті."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((npair, n)).astype(np.float32)
    cov = (z.T @ z / npair).astype(np.float32)
    M = whiten_zca(cov) if mode == "zca" else whiten_chol(cov)
    w0 = M @ ws[0]
    y = z @ w0
    x = np.concatenate([np.maximum(y, 0.0), np.maximum(-y, 0.0)], axis=0)
    denom = float(2 * npair)
    for i in range(1, len(ws)):
        p = x @ ws[i]
        x = np.maximum(p, 0.0)
    return p.sum(axis=0) / denom            # E[передактивація] останнього шару


ws = [np.ascontiguousarray(W[0, k]).astype(np.float32) for k in range(W.shape[1])]

print("=" * 74)
print("ПЛОМБА МАСШТАБУВАННЯ: |ZCA − Холецький| проти N")
print("=" * 74)
print(f"{'N':>8s} {'|Δ|_rms':>13s} {'відносно':>11s} {'падіння':>9s} {'очік.1/√N':>10s}")
prev = None
for npair in (512, 2048, 8192, 32768):
    ds = []
    for rep in range(3):
        a = run(ws, npair, "zca", 100 + rep)
        b = run(ws, npair, "chol", 100 + rep)
        ds.append(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-30))
    d = float(np.mean(ds))
    drop = f"{prev / d:.2f}×" if prev else "  —"
    print(f"{npair:>8d} {d:>13.4e} {d:>11.4e} {drop:>9s} {'2.00×' if prev else '  —':>10s}")
    prev = d

print()
print("=" * 74)
print("ВИРОК")
print("=" * 74)
print("  падіння ≈2.00× на кожному ×4 ⟹ ЧИСТИЙ ШУМ, Холецький безпечний")
print("  вихід на плато        ⟹ РІЗНІ ЗМІЩЕННЯ, Холецького ВІДКОТИТИ")
