#!/usr/bin/env python3
"""feat113c — ЕТАП C: вкидання НАВЧЕНОЇ поправки в рекурсію, вирок на TEST.

R² сам по собі не є виграшем (закон корпусу). Тут модель, навчена на TRAIN
мережах (feat113, wA), працює ВСЕРЕДИНІ рекурсії: ознаки перераховуються на
ЗБУРЕНІЙ траєкторії на льоту — саме так, як це робив би сабміт.

    m ← крок(m,C) − λ · ŷ(ознаки) · s

🔒 ПЛОМБИ
 C1 λ=0 мусить відтворити базу step113 (та сама рекурсія, ті самі мережі)
 C2 ОРАКУЛЬНЕ вкидання (справжня b_l) мусить відтворити стелю step113
 C3 🔒ЧУЖА МОДЕЛЬ (ваги моделі перемішані) ⟹ оптимальний λ → 0
 C4 κ перепідбирається РАЗОМ із λ (закон С100)
"""
import argparse
import json
import math
import time

import numpy as np

SQ2PI = math.sqrt(2.0 * math.pi)


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=16)
    ap.add_argument("--npairs", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ksig", type=int, default=8)
    ap.add_argument("--kaps", type=str, default="0.9985,0.9993,1.0")
    ap.add_argument("--lams", type=str,
                    default="0.0,0.25,0.5,0.75,1.0,1.25,1.5")
    a = ap.parse_args()

    import cupy as cp
    from cupyx.scipy.special import ndtr
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_, KS = a.width, a.depth, a.ksig
    NP, CH = a.npairs, a.chunk
    KAPS = [float(x) for x in a.kaps.split(",")]
    LAMS = [float(x) for x in a.lams.split(",")]
    NF = 16
    wA = cp.asarray(np.load("/home/trim/wblogs/feat113_wA.npy"))
    rs = np.random.default_rng(4242)
    wF = cp.asarray(np.load("/home/trim/wblogs/feat113_wA.npy")[rs.permutation(NF)])

    def phi(x):
        return cp.exp(-0.5 * x * x) / SQ2PI

    def truth(Ws, seed):
        N = 2 * NP
        rng = cp.random.default_rng(seed)
        A1 = cp.zeros((D_, W_), cp.float64)
        A2 = cp.zeros((D_, W_, W_), cp.float64)
        O1 = cp.zeros(W_, cp.float64)
        done = 0
        while done < NP:
            nb = min(CH, NP - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            act = cp.concatenate([z, -z], axis=0)
            for i in range(D_):
                A1[i] += act.sum(0, dtype=cp.float64)
                A2[i] += (act.T @ act).astype(cp.float64)
                act = cp.maximum(act @ Ws[i], 0.0)
            O1 += act.sum(0, dtype=cp.float64)
            done += nb
        mt = A1 / N
        Ct = A2 / N - mt[:, :, None] * mt[:, None, :]
        return mt, 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1))), O1 / N

    def step(w, C, m, i, kap):
        cpm = w.T @ C @ w
        if i > KS:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        return mp * ndtr(t) + s * phi(t), mp, s, ndtr(t), phi(t), cpm

    def feats(cpm, s, t, Ph, ph, l):
        inv = 1.0 / s
        R = cpm * cp.outer(inv, inv)
        R = R - cp.diag(cp.diagonal(R))
        R2 = (R * R).sum(1)
        R3 = (R * R * R).sum(1)
        R4 = (R * R * R * R).sum(1)
        A1 = cp.abs(R).sum(1)
        PR = R2 * R2 / cp.maximum(R4, 1e-30)
        sn = s / cp.mean(s)
        tp = t * ph
        hp = (t * t - 1.0) * ph
        one = cp.ones_like(t)
        return cp.stack([one, t, Ph, ph, tp, hp,
                         R2, R3, R4, A1, cp.log(cp.maximum(PR, 1e-30)),
                         sn, one * (l / D_),
                         t * R2, ph * R2, hp * R2], axis=1)

    def closure(Ws, kap, mode, lam, B=None):
        m = cp.zeros(W_, cp.float64)
        C = cp.eye(W_, dtype=cp.float64)
        for i in range(D_):
            mn, mp, s, Ph, ph, cpm = step(Ws[i], C, m, i, kap)
            if lam != 0.0:
                if mode == "orac":
                    corr = B[i]
                elif mode in ("oracO", "oracP"):
                    # розклад ТОЧНОЇ b на компоненту ВЗДОВЖ сигналу (масштаб,
                    # який косинусна метрика і α беруть даром) і ОРТОГОНАЛЬНУ
                    g = mn / cp.maximum(cp.linalg.norm(mn), 1e-30)
                    par = float(cp.dot(B[i], g)) * g
                    corr = (B[i] - par) if mode == "oracO" else par
                elif mode in ("learn", "foreign"):
                    F = feats(cpm, s, mp / s, Ph, ph, i)
                    corr = (F @ (wA if mode == "learn" else wF)) * s
                else:
                    corr = 0.0
                mn = mn - lam * corr
            m = mn
            if i == D_ - 1:
                break
            Co = (cpm * cp.outer(Ph, Ph)
                  + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
            d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
            C = Co + cp.diag(d - cp.diagonal(Co))
        return m

    MODES = ["base", "learn", "foreign", "orac", "oracO", "oracP"]
    res = {}
    t0 = time.time()
    for net in range(a.nets // 2, a.nets):          # ЛИШЕ TEST-мережі
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        mt, Ct, T = truth(Ws, 113000 + net)
        Ws64 = [w.astype(cp.float64) for w in Ws]
        rec = {"_TT": float(T @ T)}
        for kap in KAPS:
            B = cp.zeros((D_, W_), cp.float64)
            for i in range(D_):
                Cin = cp.eye(W_, dtype=cp.float64) if i == 0 else Ct[i]
                min_ = cp.zeros(W_, cp.float64) if i == 0 else mt[i]
                mn, mp, s, Ph, ph, _ = step(Ws64[i], Cin, min_, i, kap)
                B[i] = mn - (T if i == D_ - 1 else mt[i + 1])
            for mode in MODES:
                for lam in ([0.0] if mode == "base" else LAMS):
                    mm = closure(Ws64, kap, mode, lam, B)
                    if bool(cp.all(cp.isfinite(mm))):
                        rec[f"{mode}|{kap}|{lam}"] = [float(mm @ mm), float(mm @ T)]
        res[str(net)] = rec
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)")
        cp.get_default_memory_pool().free_all_blocks()
    json.dump(res, open("/home/trim/wblogs/feat113d.json", "w"))

    def agg(key):
        num = den = 0.0
        ind = []
        for r in res.values():
            if key not in r:
                return None, None
            xx, xt = r[key]
            tt = r["_TT"]
            e = tt - xt * xt / max(xx, 1e-300)
            num += e
            den += tt
            ind.append(math.sqrt(max(e, 0.0) / tt))
        return math.sqrt(num / den), np.array(ind)

    log(f"\n📊 {len(res)} TEST-мереж · косинусна похибка")
    best = {}
    for mode in MODES:
        rows = []
        for kap in KAPS:
            for lam in ([0.0] if mode == "base" else LAMS):
                v, ind = agg(f"{mode}|{kap}|{lam}")
                if v is not None:
                    rows.append((v, kap, lam, ind))
        rows.sort(key=lambda r: r[0])
        best[mode] = rows[0]
        v, kap, lam, ind = rows[0]
        log(f"{mode:>9}  {100*v:8.4f}%   κ={kap} λ={lam:4.2f}   "
            f"медіана {100*np.median(ind):7.4f}%")

    E0, V0 = 0.1264e-2, 0.2404e-2
    b0 = best["base"][0]
    h0 = (E0**2)*(V0**2)/((E0**2)+(V0**2))
    log(f"\n{'режим':>9}{'×точність':>12}{'×бал ф2':>11}{'бал':>13}")
    for mode in MODES:
        k = b0 / max(best[mode][0], 1e-30)
        e = (E0/k)**2
        h = e*(V0**2)/(e+V0**2)
        log(f"{mode:>9}{k:>12.4f}{h0/h:>11.3f}{1.2250e-07*h/h0:>13.3e}")
    log(f"\n🔒 C3 чужа модель: оптимальний λ = {best['foreign'][2]:.2f} "
        f"— мусить бути 0.00 {'✅' if best['foreign'][2] == 0.0 else '🚨'}")
    log(f"🔒 C1 база = {100*b0:.4f}% (step113 дала свою — звірити)")
    log(f"🔒 C2 оракул ×{b0/max(best['orac'][0],1e-30):.4f} "
        f"(step113 inj — звірити)")
    log(f"\n🎯 частка стелі, яку бере НАВЧЕНА поправка: "
        f"{(b0/best['learn'][0]-1)/max(b0/best['orac'][0]-1,1e-9)*100:.1f}%")


if __name__ == "__main__":
    main()
