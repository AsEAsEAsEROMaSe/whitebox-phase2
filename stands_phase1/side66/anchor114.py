#!/usr/bin/env python3
"""anchor114 — КЛАС СУРОГАТІВ З ТОЧНО ВІДОМИМ СЕРЕДНІМ ТА ОПТИМІЗОВАНОЮ МАТРИЦЕЮ.

════════ ЧОМУ САМЕ ЦЕ ════════
🎯 У фазі 2 вузьке місце — **МК (0.240%), а не замикання (0.126%)**. Покращення
МК вдвічі дає +66% балу — вп'ятеро більше за все, що дала поправка форми.
Але всі відомі шляхи редукції дисперсії закриті: RQMC ×1.0156, сферизація
×0.74, 4-ті моменти ×1.00, лінійна КЗ ×1.0096 (вибілювання їх з'їдає).

🔑 **ПРИЧИНА, ЧОМУ ВОНИ ГИНУТЬ**: вибілювання робить `mean(z)=0` і `cov(z)=I`
ТОЧНО, антитетика зануляє всі НЕПАРНІ моменти ⟹ будь-яка КЗ, що є поліномом
від `z` до 3-го степеня, дає РІВНО НУЛЬ за побудовою.
⟹ потрібна КЗ, ЯКА НЕ Є ПОЛІНОМОМ НИЗЬКОГО СТЕПЕНЯ, і водночас має ВІДОМЕ
   середнє. Таких класів мало — і один із них ми ніколи не використовували:

    ĝ(z) = relu(z·W₀) @ A ,   A — БУДЬ-ЯКА матриця 1024×1024
    E[ĝ] = (‖W₀[:,j]‖/√(2π))_j @ A     ← ТОЧНО, бо вхід гаусів за побудовою

`A` вільна ⟹ її можна ОПТИМІЗУВАТИ під максимальну кореляцію з `out`.
Це лінійна регресія виходу на активації ПЕРШОГО шару; КЗ тоді виправляє саме
похибку `mean(x₁)`, спроєктовану через `A`. Ми цього не робили: λ-блок
використовує `s` і кілька скалярів, а не оптимальну повну матрицю.

════════ 🔒 ПЛОМБИ ════════
 A1 E[ĝ] аналітичне проти власної МК на великому N ⟹ збіг до шуму МК
 A2 A=0 ⟹ редукція РІВНО 1.0000
 A3 🔒 A з ЧУЖОЇ мережі ⟹ λ* → 0
 A4 🔒 A, підібрана на ПОЛОВИНІ семплів, оцінена на ДРУГІЙ (без самофіту)
 A5 масштабування N: редукція мусить стояти

════════ 📉 ПЕРЕДБАЧЕННЯ ════════
   R²(out ~ x₁) < 0.2 ⟹ КЗ слабка                        35%
   R² = 0.2…0.5 ⟹ редукція ×1.12…×1.41                   40%
   R² > 0.5 ⟹ редукція >×1.41, це +20% балу              15%
   самофіт роздує оцінку ⟹ чесна половинна нижча         70%
"""
import argparse
import math
import time

import numpy as np


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=8)
    ap.add_argument("--ntruth", type=int, default=1048576)
    ap.add_argument("--nwork", type=int, default=6144)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--ridge", type=float, default=1e-2)
    a = ap.parse_args()

    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_ = a.width, a.depth
    SQ2PI = math.sqrt(2.0 * math.pi)

    def cerr(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1.0 - c * c)))

    def fwd_all(Ws, x):
        """повертає (x₁ = relu(zW₀), out)."""
        h = cp.maximum(x @ Ws[0], 0.0)
        x1 = h
        for l in range(1, D_):
            h = cp.maximum(h @ Ws[l], 0.0)
        return x1, h

    def truth(Ws, seed, n):
        rng = cp.random.default_rng(seed)
        acc = cp.zeros(W_, cp.float64)
        done = 0
        while done < n // 2:
            nb = min(a.chunk, n // 2 - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            _, o = fwd_all(Ws, cp.concatenate([z, -z], axis=0))
            acc += o.sum(0, dtype=cp.float64)
            done += nb
        return acc / n

    def whiten(seed, n):
        rng = cp.random.default_rng(seed)
        z = rng.standard_normal((n // 2, W_), dtype=cp.float32)
        x = cp.concatenate([z, -z], axis=0)
        cov = (x.T.astype(cp.float64) @ x.astype(cp.float64)) / x.shape[0]
        Lc = cp.linalg.cholesky(cov + 1e-9 * cp.eye(W_, dtype=cp.float64))
        return (x.astype(cp.float64) @ cp.linalg.inv(Lc).T).astype(cp.float32)

    R = {"base": [], "self": [], "half": [], "fore": [], "r2": [], "a1": []}
    t0 = time.time()
    PREVA = None
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        T = truth(Ws, 990000 + net, a.ntruth)
        Ex1 = cp.linalg.norm(Ws[0].astype(cp.float64), axis=0) / SQ2PI   # ТОЧНО

        # 🔒 A1: аналітичне E[x₁] проти МК
        rng2 = cp.random.default_rng(880000 + net)
        acc = cp.zeros(W_, cp.float64); done = 0
        while done < 131072:
            nb = min(a.chunk, 131072 - done)
            z = rng2.standard_normal((nb, W_), dtype=cp.float32)
            acc += cp.maximum(cp.concatenate([z, -z], axis=0) @ Ws[0],
                              0.0).sum(0, dtype=cp.float64)
            done += nb
        a1 = float(cp.linalg.norm(acc / (2 * 131072) - Ex1) / cp.linalg.norm(Ex1))
        R["a1"].append(a1)

        x = whiten(5000 + net, a.nwork)
        x1, out = fwd_all(Ws, x)
        x1 = x1.astype(cp.float64); out = out.astype(cp.float64)
        N = out.shape[0]
        m0 = out.mean(0)
        R["base"].append(cerr(m0, T))
        d1 = x1 - x1.mean(0)
        dO = out - m0

        def fitA(idx):
            """A = регресія out на x₁ (центровані), гребенева."""
            P = d1[idx]; Q = dO[idx]
            G = P.T @ P / len(idx)
            G = G + a.ridge * float(cp.trace(G)) / W_ * cp.eye(W_, dtype=cp.float64)
            return cp.linalg.solve(G, P.T @ Q / len(idx))

        def apply(A):
            gm = x1.mean(0) @ A
            ge = Ex1 @ A
            gc = (x1 @ A) - gm
            cov = float((dO * gc).sum() / N)
            var = float((gc * gc).sum() / N)
            lam = cov / max(var, 1e-300)
            return cerr(m0 - lam * (gm - ge), T), lam

        A_self = fitA(cp.arange(N))
        e_s, _ = apply(A_self)
        R["self"].append(e_s)
        # 🔒 A4 чесна половина
        h = N // 2
        A_h = fitA(cp.arange(h))
        e_h, _ = apply(A_h)
        R["half"].append(e_h)
        # R² передбачення out із x₁ на ДРУГІЙ половині
        P2 = d1[h:]; pred = P2 @ A_h
        r2 = 1 - float(((dO[h:] - pred) ** 2).sum() / (dO[h:] ** 2).sum())
        R["r2"].append(r2)
        # 🔒 A3 чужа A
        if PREVA is not None:
            e_f, _ = apply(PREVA)
            R["fore"].append(e_f)
        PREVA = A_h
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)  база {100*R['base'][-1]:.4f}%  "
            f"самофіт {100*e_s:.4f}%  чесно {100*e_h:.4f}%  R² {r2:+.4f}  🔒A1 {a1:.2e}")
        cp.get_default_memory_pool().free_all_blocks()

    b = float(np.mean(R["base"]))
    log(f"\n📊 {a.nets} мереж · N={a.nwork} · 🔒A1 {np.mean(R['a1']):.3e} "
        f"(шум МК ≈{1/math.sqrt(262144):.2e})")
    log(f"{'база (вибілена МК)':>34}  {100*b:8.4f}%   ×1.0000")
    log(f"{'КЗ, A самофітом (роздуто)':>34}  {100*np.mean(R['self']):8.4f}%   "
        f"×{b/np.mean(R['self']):.4f}")
    log(f"{'🔒A4 КЗ, A з ПОЛОВИНИ семплів':>34}  {100*np.mean(R['half']):8.4f}%   "
        f"×{b/np.mean(R['half']):.4f}")
    if R["fore"]:
        log(f"{'🔒A3 КЗ, A з ЧУЖОЇ мережі':>34}  {100*np.mean(R['fore']):8.4f}%   "
            f"×{b/np.mean(R['fore']):.4f}")
    log(f"\n📐 R²(out ~ x₁) на невидимих семплах: {np.mean(R['r2']):+.4f}")
    log("📉 ПЕРЕДБАЧЕННЯ: R²<0.2 (35%) · 0.2–0.5 (40%) · >0.5 (15%)")


if __name__ == "__main__":
    main()
