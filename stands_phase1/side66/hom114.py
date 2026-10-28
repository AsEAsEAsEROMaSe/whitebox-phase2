#!/usr/bin/env python3
"""hom114 — ГОМОТОПІЯ ДО ЛІНІЙНОЇ ТОЧКИ (розвиток ідеї ЮЗЕРА про рух у вагах).

════════ ЗНАХІДКА, НА ЯКІЙ ЦЕ СТОЇТЬ ════════
`W(τ) = (1−τ)·W + τ·|W|`. При τ=1 УСІ ваги додатні ⟹ після першого шару
активації невід'ємні ⟹ relu ТОТОЖНИЙ ⟹ мережа ЛІНІЙНА, і середнє БЕРЕТЬСЯ
АНАЛІТИЧНО:
    E[out(W(1))] = (‖A₀[:,j]‖/√(2π))_j @ A₁ @ … @ A₁₅ ,   A = |W|
🔑 Це ПʼЯТА точно відома істина в задачі (інвентар С102 знав лише чотири:
   E[zzᵀ]=I · E‖z‖ · E[(u·z)⁴] · E[h₁]=σ/√2π). І вона НЕЛІНІЙНА по z.

════════ КОНСТРУКЦІЯ ════════
Контрольна змінна з ВІДОМИМ середнім на ТИХ САМИХ семплах:
    m̂ = mean(out) − λ·(mean(g) − E[g]_ТОЧНЕ),   g(z) = out(z; |W|)
Стеля = 1/√(1−ρ²), ρ = кореляція похибок. Ціна = другий форвард (×2 семплів),
тож потрібно ρ > 0.71 лише щоб вийти в нуль — а на менших ρ КЗ вигідна тільки
якщо семпли не ділити (є запас бюджету).

════════ 🔒 ПЛОМБИ ════════
 H1 аналітичне E[out(|W|)] проти власної МК ⟹ мусить збігтись до шуму МК
    (це пломба ПРИЛАДУ: без неї весь стенд недійсний)
 H2 λ=0 відтворює базову вибілену МК ПОБІТОВО
 H3 🔒 КЗ із ЧУЖОЇ мережі ⟹ λ* → 0
 H4 карта ρ(τ) для проміжних τ — чи є гладкий місток

════════ 📉 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
   ρ(τ=1) < 0.3 (мережі надто різні, КЗ слабка)          40%
   ρ = 0.3…0.7 (є що брати, але ціна ×2 з'їдає)          35%
   ρ > 0.7 (КЗ окупається навіть із подвоєним форвардом) 15%
   ρ(τ) гладко росте при τ→0 ⟹ можливий ЛАНЦЮГ           50%
"""
import argparse
import json
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
    ap.add_argument("--taus", type=str, default="0.25,0.5,0.75,1.0")
    a = ap.parse_args()

    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_ = a.width, a.depth
    TAUS = [float(x) for x in a.taus.split(",")]
    SQ2PI = math.sqrt(2.0 * math.pi)

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1.0 - c * c)))

    def fwd(Ws, x):
        h = x
        for l in range(D_):
            h = cp.maximum(h @ Ws[l], 0.0)
        return h

    def truth(Ws, seed, n):
        rng = cp.random.default_rng(seed)
        acc = cp.zeros(W_, cp.float64)
        done = 0
        while done < n // 2:
            nb = min(a.chunk, n // 2 - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            h = fwd(Ws, cp.concatenate([z, -z], axis=0))
            acc += h.sum(0, dtype=cp.float64)
            done += nb
        return acc / n

    def exact_linear(As):
        """🔑 ТОЧНЕ E[out] для мережі з НЕВІД'ЄМНИМИ вагами."""
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

    res = {}
    t0 = time.time()
    PREV = None
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        As = [cp.abs(w) for w in Ws]
        T = truth(Ws, 990000 + net, a.ntruth)
        Eg = exact_linear(As)

        # 🔒 H1: аналітичне проти власної МК на тій самій мережі |W|
        Tg = truth(As, 880000 + net, 262144)
        h1 = float(cp.linalg.norm(Tg - Eg) / cp.linalg.norm(Eg))

        x = whiten(5000 + net, a.nwork)
        out = fwd(Ws, x).astype(cp.float64)
        m0 = out.mean(0)
        e0 = cos_err(m0, T)                       # 🔒 H2: база

        rec = {"e0": e0, "h1": h1, "tau": {}}
        for tau in TAUS:
            Wt = [((1.0 - tau) * Ws[l] + tau * As[l]) for l in range(D_)]
            g = fwd(Wt, x).astype(cp.float64)
            gm = g.mean(0)
            if tau == 1.0:
                gref = Eg                          # ТОЧНЕ середнє
            else:
                gref = truth(Wt, 770000 + net, 262144)   # опорне (лише карта)
            do = out - m0
            dg = g - gm
            cov = float((do * dg).sum() / do.shape[0])
            var = float((dg * dg).sum() / dg.shape[0])
            lam = cov / max(var, 1e-300)
            rho = cov / max(math.sqrt(var * float((do * do).sum() / do.shape[0])), 1e-300)
            mh = m0 - lam * (gm - gref)
            rec["tau"][str(tau)] = {"rho": rho, "lam": lam,
                                    "e": cos_err(mh, T)}
        # 🔒 H3: КЗ із ЧУЖОЇ мережі (її точне середнє теж відоме)
        if PREV is not None:
            Af, Ef = PREV
            gf = fwd(Af, x).astype(cp.float64)
            gmf = gf.mean(0)
            dgf = gf - gmf
            covf = float(((out - m0) * dgf).sum() / out.shape[0])
            varf = float((dgf * dgf).sum() / dgf.shape[0])
            lf = covf / max(varf, 1e-300)
            rec["foreign"] = {"lam": lf,
                              "e": cos_err(m0 - lf * (gmf - Ef), T)}
        PREV = (As, Eg)
        res[str(net)] = rec
        r1 = rec["tau"][str(TAUS[-1])]
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)  база {100*e0:.4f}%  "
            f"τ=1 ρ={r1['rho']:+.4f} → {100*r1['e']:.4f}%  🔒H1 {h1:.2e}")
        cp.get_default_memory_pool().free_all_blocks()

    json.dump(res, open("/home/trim/wblogs/hom114.json", "w"))
    e0 = float(np.mean([r["e0"] for r in res.values()]))
    h1 = float(np.mean([r["h1"] for r in res.values()]))
    log(f"\n📊 {len(res)} мереж · N={a.nwork} · база {100*e0:.4f}%")
    log(f"🔒 H1 аналітичне E[out(|W|)] проти МК: відн. розбіжність "
        f"{h1:.3e}  (шум МК при 262144 ≈ {1/math.sqrt(262144):.2e})  "
        f"{'✅' if h1 < 5e-3 else '🚨 ПРИЛАД НЕВАЛІДНИЙ'}")
    log(f"\n{'τ':>6}{'ρ':>10}{'λ':>10}{'похибка':>12}{'×база':>9}")
    for tau in TAUS:
        rr = [r["tau"][str(tau)] for r in res.values()]
        rho = float(np.mean([x["rho"] for x in rr]))
        lam = float(np.mean([x["lam"] for x in rr]))
        e = float(np.mean([x["e"] for x in rr]))
        log(f"{tau:>6.2f}{rho:>10.4f}{lam:>10.4f}{100*e:>11.4f}%{e0/e:>9.4f}")
    if "foreign" in list(res.values())[-1]:
        ff = [r["foreign"] for r in res.values() if "foreign" in r]
        ef = float(np.mean([x["e"] for x in ff]))
        lf = float(np.mean([x["lam"] for x in ff]))
        log(f"\n🔒 H3 КЗ із ЧУЖОЇ мережі: λ={lf:.4f}  похибка {100*ef:.4f}%  "
            f"×{e0/ef:.4f}  (мусить бути ×1.00)")
    log("\n📉 ПЕРЕДБАЧЕННЯ: ρ(τ=1)<0.3 (40%) · 0.3–0.7 (35%) · >0.7 (15%)")
    r1 = float(np.mean([r["tau"][str(TAUS[-1])]["rho"] for r in res.values()]))
    red = 1.0 / math.sqrt(max(1.0 - r1 * r1, 1e-12))
    log(f"🎯 ρ(τ=1) = {r1:.4f} ⟹ теоретична стеля КЗ ×{red:.4f} "
        f"(потрібно >×1.414, щоб окупити подвоєний форвард)")


if __name__ == "__main__":
    main()
