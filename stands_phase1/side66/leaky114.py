#!/usr/bin/env python3
"""leaky114 — ГОМОТОПІЯ ПО НЕЛІНІЙНОСТІ + ПЛОМБА ЗАМИКАННЯ НА ЛІНІЙНІЙ ТОЧЦІ.

════════ ЧОМУ ЦЕ ІНША МЕТРИКА, НІЖ РУХ ПО ВАГАХ ════════
`hom114b`: масштаб декореляції по ВАГАХ τ*≈0.01 — бо зсув ваг МІНЯЄ МАСКИ D(z).
Тут ваги ті самі, міняється лише активація:
    h_α(x) = max(x, α·x) = (1−α)·relu(x) + α·x
⟹ **маски D(z) ФІКСОВАНІ на всьому шляху**. Є підстава чекати, що декореляція
на порядки повільніша, і тоді місток існує там, де по вагах його не було.

Кінці шляху:
    α = 0  наша задача
    α = 1  мережа ЛІНІЙНА по z ⟹ E[out] = 0 РІВНО (точний якір)

════════ ЩО МІРЯЄМО ════════
 A 🔒 ПЛОМБА: чи ТОЧНЕ наше замикання на мережі |W|? Там усі p ≥ 0 ⟹ t→+∞ ⟹
   Φ→1, φ→0 ⟹ крок вироджується в лінійний. Якщо збіг із аналітикою малий —
   це НОВА безкоштовна пломба коректності реалізації замикання.
 B ρ(α) = corr(out_relu, out_α) на ТИХ САМИХ семплах ⟹ є місток чи ні
 C стеля КЗ 1/√(1−ρ²) при α=1, де середнє відоме ТОЧНО (=0)

════════ 📉 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
   ρ(α) падає ПОВІЛЬНІШЕ, ніж по вагах (маски фіксовані)     75%
   ρ(α=1) > 0.3 (КЗ з точним нулем щось дає)                 30%
   🔒 замикання на |W| точне краще ніж 1e-3                   60%
   ρ(α=0.2) > 0.9 ⟹ є робочий місток                         45%
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
    ap.add_argument("--nets", type=int, default=6)
    ap.add_argument("--ntruth", type=int, default=262144)
    ap.add_argument("--nwork", type=int, default=6144)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--alphas", type=str,
                    default="0.05,0.1,0.2,0.4,0.7,1.0")
    a = ap.parse_args()

    import cupy as cp
    from cupyx.scipy.special import ndtr
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_ = a.width, a.depth
    AL = [float(x) for x in a.alphas.split(",")]
    SQ2PI = math.sqrt(2.0 * math.pi)

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1.0 - c * c)))

    def fwd(Ws, x, al=0.0):
        h = x
        for l in range(D_):
            p = h @ Ws[l]
            h = cp.maximum(p, 0.0) if al == 0.0 else cp.maximum(p, al * p)
        return h

    def truth(Ws, seed, n, al=0.0):
        rng = cp.random.default_rng(seed)
        acc = cp.zeros(W_, cp.float64)
        done = 0
        while done < n // 2:
            nb = min(a.chunk, n // 2 - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            h = fwd(Ws, cp.concatenate([z, -z], axis=0), al)
            acc += h.sum(0, dtype=cp.float64)
            done += nb
        return acc / n

    def closure(Ws):
        """НАШЕ замикання (форма фази 2, без κ) на повних вагах."""
        m = cp.zeros(W_, cp.float64)
        C = cp.eye(W_, dtype=cp.float64)
        for i in range(D_):
            w = Ws[i].astype(cp.float64)
            cpm = w.T @ C @ w
            s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-30))
            mp = m @ w
            t = mp / s
            Ph, ph = ndtr(t), cp.exp(-0.5 * t * t) / SQ2PI
            mn = mp * Ph + s * ph
            if i == D_ - 1:
                return mn
            Co = (cpm * cp.outer(Ph, Ph)
                  + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
            d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
            C = Co + cp.diag(d - cp.diagonal(Co))
            m = mn

    def exact_linear(As):
        m = cp.linalg.norm(As[0].astype(cp.float64), axis=0) / SQ2PI
        for l in range(1, D_):
            m = m @ As[l].astype(cp.float64)
        return m

    def whiten(seed, n):
        rng = cp.random.default_rng(seed)
        z = rng.standard_normal((n // 2, W_), dtype=cp.float32)
        x = cp.concatenate([z, -z], axis=0)
        cov = (x.T.astype(cp.float64) @ x.astype(cp.float64)) / x.shape[0]
        Lc = cp.linalg.cholesky(cov + 1e-9 * cp.eye(W_, dtype=cp.float64))
        return (x.astype(cp.float64) @ cp.linalg.inv(Lc).T).astype(cp.float32)

    RA, RB, RC, RS = [], [], [], []
    t0 = time.time()
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        As = [cp.abs(w) for w in Ws]

        # ── A 🔒 ПЛОМБА: замикання на |W| проти аналітики ──
        mcl = closure(As)
        mex = exact_linear(As)
        relA = float(cp.linalg.norm(mcl - mex) / cp.linalg.norm(mex))
        # для порівняння: похибка замикання на НАШІЙ мережі
        T = truth(Ws, 990000 + net, a.ntruth)
        eclo = cos_err(closure(Ws), T)
        RA.append(relA); RS.append(eclo)

        # ── B/C гомотопія по нелінійності ──
        x = whiten(5000 + net, a.nwork)
        out = fwd(Ws, x).astype(cp.float64)
        m0 = out.mean(0)
        e0 = cos_err(m0, T)
        do = out - m0
        v0 = float((do * do).sum() / do.shape[0])
        row_r, row_e = [], []
        for al in AL:
            g = fwd(Ws, x, al).astype(cp.float64)
            gm = g.mean(0)
            dg = g - gm
            cov = float((do * dg).sum() / do.shape[0])
            var = float((dg * dg).sum() / dg.shape[0])
            rho = cov / max(math.sqrt(var * v0), 1e-300)
            row_r.append(rho)
            if al == 1.0:
                lam = cov / max(var, 1e-300)
                row_e.append(cos_err(m0 - lam * gm, T))   # E[g]=0 ТОЧНО
            else:
                row_e.append(float("nan"))
        RB.append(row_r); RC.append([e0] + row_e)
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)  🔒A {relA:.3e}  "
            f"замик {100*eclo:.4f}%  ρ(α): " +
            " ".join(f"{v:.3f}" for v in row_r))
        cp.get_default_memory_pool().free_all_blocks()

    RA = np.array(RA); RB = np.array(RB); RC = np.array(RC)
    log(f"\n📊 {a.nets} мереж · N={a.nwork}")
    log(f"🔒 A ЗАМИКАННЯ НА |W| проти аналітики: {RA.mean():.3e} "
        f"(розкид {RA.min():.1e}…{RA.max():.1e})")
    log(f"   для порівняння похибка замикання на НАШІЙ мережі: "
        f"{100*np.array(RS).mean():.4f}%")
    log(f"   ⟹ {'✅ НОВА ПЛОМБА: замикання ТОЧНЕ на лінійній точці' if RA.mean()<1e-3 else '🟡 не точне — цікаво саме по собі'}")
    log(f"\n{'α':>7}{'ρ':>10}{'стеля КЗ':>11}")
    for k, al in enumerate(AL):
        r = RB[:, k].mean()
        log(f"{al:>7.2f}{r:>10.4f}{1/math.sqrt(max(1-r*r,1e-12)):>11.4f}")
    e0 = RC[:, 0].mean()
    e1 = RC[:, -1].mean()
    log(f"\n🎯 КЗ при α=1 (E[g]=0 ТОЧНО): база {100*e0:.4f}% → {100*e1:.4f}% "
        f"= ×{e0/e1:.4f}")
    log("\n📉 ПЕРЕДБАЧЕННЯ: ρ(α) падає повільніше ніж по вагах (75%) · "
        "ρ(α=1)>0.3 (30%) · пломба A <1e-3 (60%) · ρ(0.2)>0.9 (45%)")


if __name__ == "__main__":
    main()
