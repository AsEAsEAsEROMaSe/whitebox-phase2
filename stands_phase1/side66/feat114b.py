#!/usr/bin/env python3
"""feat114 — ПОПРАВКА ФОРМИ, НАВЧЕНА В РЕЖИМІ САБМІТУ (на ОБРІЗАНИХ вагах).

════════ ЧОМУ ПЕРЕРОБКА ════════
`feat113` навчався на ПОВНИХ вагах 1024 і обіцяв ×1.042; повний оцінювач дав
**×1.0049** (у 8.6× менше). Причина: сабміт працює на ОБРІЗАНИХ вагах (пілот
викидає ~22% нейронів як мертві), і статистики рядка `Σρ²` там інші.
📏 Закон, здобутий тим прогоном: СТЕНД І САБМІТ МУСЯТЬ РАНЖУВАТИ В ОДНІЙ
   ОБЛАСТІ. Тут область відтворена: пілот 128 пар, `keep = max(h) >= 0`.

════════ ЩО ЗБИРАЄМО ════════
ЦІЛЬ  y = (b_l − ⟨b_l,m̂⟩m̂)/s   — ЛИШЕ ортогональна компонента похибки кроку
      (масштаб `_ai` бере даром: oracP дала РІВНО нуль, oracO ×1.2165)
      b_l = крок(істинні m_l,C_l ОБРІЗАНОЇ мережі) − істинне m_{l+1}
ОЗНАКИ 16, з НАШОЇ рекурсії на w_cut, з нормуванням `sc = 1024/n_l`
      (n_l змінне по шарах ⟹ суми по рядку інтенсивні)

🔒 ПЛОМБИ
 G1 частка живих нейронів мусить лягти в 0.75…0.85 (запис корпусу: оптимум 22%)
 G2 🔒 перемішані ознаки ⟹ R² ≈ 0
 G3 ознаки з ІСТИННИХ (m,C) — верхня межа сім'ї
 G4 крос по мережах TRAIN/TEST

📉 ПЕРЕДБАЧЕННЯ: R² TEST у режимі сабміту ≥0.30 (50%) · впаде нижче 0.20 (30%)
"""
import argparse
import math
import time

import numpy as np

SQ2PI = math.sqrt(2.0 * math.pi)
NF = 16


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=32)
    ap.add_argument("--npairs", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ksig", type=int, default=8)
    ap.add_argument("--kap", type=float, default=0.9993)
    ap.add_argument("--pilot", type=int, default=128)
    ap.add_argument("--out", type=str, default="/home/trim/wblogs/feat114")
    a = ap.parse_args()

    import cupy as cp
    from cupyx.scipy.special import ndtr
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_, KS, KAP = a.width, a.depth, a.ksig, a.kap

    def phi(x):
        return cp.exp(-0.5 * x * x) / SQ2PI

    def prune(Ws, seed):
        """пілот РІВНО як у сабміті: 128 антитетичних пар, keep = max >= 0."""
        rng = cp.random.default_rng(seed)
        zp = rng.standard_normal((a.pilot, W_), dtype=cp.float32)
        h = cp.concatenate([zp, -zp], axis=0)
        keeps = []
        for i in range(D_):
            p = h @ Ws[i]
            if i < D_ - 1:
                keeps.append(cp.max(p, axis=0) >= 0.0)
            h = cp.maximum(p, 0.0)
        cuts = []
        for i in range(D_):
            w = Ws[i]
            if i > 0:
                w = w[keeps[i - 1], :]
            if i < D_ - 1:
                w = w[:, keeps[i]]
            cuts.append(w.astype(cp.float64))
        return cuts, [float(k.mean()) for k in keeps]

    def truth_cut(cuts, seed):
        """істинні (m_l, C_l) активацій ОБРІЗАНОЇ мережі + вихід."""
        N = 2 * a.npairs
        rng = cp.random.default_rng(seed)
        dims = [cuts[0].shape[0]] + [c.shape[1] for c in cuts]
        A1 = [cp.zeros(d, cp.float64) for d in dims[:-1]]
        A2 = [cp.zeros((d, d), cp.float64) for d in dims[:-1]]
        O1 = cp.zeros(dims[-1], cp.float64)
        done = 0
        while done < a.npairs:
            nb = min(a.chunk, a.npairs - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            act = cp.concatenate([z, -z], axis=0)
            for i in range(D_):
                A1[i] += act.sum(0, dtype=cp.float64)
                A2[i] += (act.T @ act).astype(cp.float64)
                act = cp.maximum(act @ cuts[i].astype(cp.float32), 0.0)
            O1 += act.sum(0, dtype=cp.float64)
            done += nb
        mt = [A1[i] / N for i in range(D_)]
        Ct = [A2[i] / N - cp.outer(mt[i], mt[i]) for i in range(D_)]
        Ct = [0.5 * (c + c.T) for c in Ct]
        return mt, Ct, O1 / N

    def step(w, C, m, i):
        cpm = w.T @ C @ w
        if i > KS:
            cpm = cpm * KAP
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-30))
        mp = m @ w
        t = mp / s
        Ph, ph = ndtr(t), phi(t)
        return mp * Ph + s * ph, mp, s, Ph, ph, cpm

    def feats(cpm, s, t, Ph, ph, l):
        n = s.shape[0]
        inv = 1.0 / s
        R = cpm * cp.outer(inv, inv)
        R = R - cp.diag(cp.diagonal(R))
        Rs = R * R
        sc = 1024.0 / n
        R2 = Rs.sum(1) * sc
        R3 = (Rs * R).sum(1) * sc
        R4 = (Rs * Rs).sum(1) * sc
        A1 = cp.abs(R).sum(1) * sc
        lPR = cp.log(cp.maximum(R2 * R2 / cp.maximum(R4, 1e-30), 1e-30))
        sn = s / cp.mean(s)
        tp = t * ph
        hp = (t * t - 1.0) * ph
        one = cp.ones_like(t)
        return cp.stack([one, t, Ph, ph, tp, hp, R2, R3, R4, A1, lPR,
                         sn, one * (l / D_), t * R2, ph * R2, hp * R2], axis=1)

    XA, XB, Y = [], [], []
    alive = []
    t0 = time.time()
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc0 = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc0)
              for _ in range(D_)]
        cuts, fr = prune(Ws, 12345)
        alive.append(float(np.mean(fr)))
        mt, Ct, T = truth_cut(cuts, 113000 + net)
        # ── ціль + ознаки B (істинні входи) ──
        for i in range(D_):
            n_in = cuts[i].shape[0]
            Cin = cp.eye(n_in, dtype=cp.float64) if i == 0 else Ct[i]
            min_ = cp.zeros(n_in, cp.float64) if i == 0 else mt[i]
            mn, mp, s, Ph, ph, cpm = step(cuts[i], Cin, min_, i)
            tgt = T if i == D_ - 1 else mt[i + 1]
            bb = mn - tgt
            g = mn / cp.maximum(cp.linalg.norm(mn), 1e-30)
            bb = bb - float(cp.dot(bb, g)) * g
            Y.append(cp.asnumpy((bb / s).astype(cp.float32)))
            XB.append(cp.asnumpy(feats(cpm, s, mp / s, Ph, ph, i).astype(cp.float32)))
        # ── ознаки A (наша рекурсія) ──
        m = cp.zeros(cuts[0].shape[0], cp.float64)
        C = cp.eye(cuts[0].shape[0], dtype=cp.float64)
        for i in range(D_):
            mn, mp, s, Ph, ph, cpm = step(cuts[i], C, m, i)
            XA.append(cp.asnumpy(feats(cpm, s, mp / s, Ph, ph, i).astype(cp.float32)))
            if i < D_ - 1:
                Co = (cpm * cp.outer(Ph, Ph)
                      + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
                d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
                C = Co + cp.diag(d - cp.diagonal(Co))
            m = mn
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с) живих {fr[0]:.3f}…{fr[-1]:.3f}")
        cp.get_default_memory_pool().free_all_blocks()

    nets = a.nets
    log(f"\n🔒 G1 частка живих нейронів: {np.mean(alive):.4f} "
        f"{'✅' if 0.70 <= np.mean(alive) <= 0.90 else '🚨'}")

    LMIN = int(__import__('os').environ.get('LMIN', '0'))
    def fit(X, name, shuffle=False):
        tr = [k for k in range(len(X))
              if (k // D_) < nets // 2 and (k % D_) >= LMIN]
        te = [k for k in range(len(X))
              if (k // D_) >= nets // 2 and (k % D_) >= LMIN]
        xt = np.concatenate([X[k] for k in tr]).astype(np.float64)
        yt = np.concatenate([Y[k] for k in tr]).astype(np.float64)
        xe = np.concatenate([X[k] for k in te]).astype(np.float64)
        ye = np.concatenate([Y[k] for k in te]).astype(np.float64)
        if shuffle:
            xt = xt[np.random.default_rng(0).permutation(len(xt))]
        sd = xt.std(0); sd[sd < 1e-12] = 1.0
        xt2, xe2 = xt / sd, xe / sd
        A = xt2.T @ xt2 + 1e-3 * len(xt2) * np.eye(NF)
        w = np.linalg.solve(A, xt2.T @ yt)
        r2 = 1 - ((ye - xe2 @ w) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
        r2t = 1 - ((yt - xt2 @ w) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        log(f"  {name:>36}  TRAIN {r2t:7.4f}   TEST {r2:7.4f}")
        return w / sd

    log(f"\n📊 feat114 · {nets} мереж (TRAIN {nets//2} / TEST {nets//2}) · "
        f"{sum(len(y) for y in Y)} прикладів · РЕЖИМ САБМІТУ")
    wB = fit(XB, "🔒G3 ознаки з ІСТИННИХ (m,C)")
    wA = fit(XA, "A: ознаки з НАШИХ (m,C) [сабміт]")
    fit(XA, "🔒G2 перемішані ознаки", shuffle=True)
    np.save(a.out + "_wA.npy", wA)
    log(f"\n💾 {a.out}_wA.npy")
    log("📉 ПЕРЕДБАЧЕННЯ: R² TEST ≥0.30 (50%) · <0.20 (30%)")


if __name__ == "__main__":
    main()
