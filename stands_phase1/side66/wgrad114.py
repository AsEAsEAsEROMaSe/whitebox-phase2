#!/usr/bin/env python3
"""wgrad114 — РУХ У ПРОСТОРІ ВАГ (ідея ЮЗЕРА: «бекпропом зсунути ваги так, щоб
один семпл дав середнє; корекції малі; шукати екстраполяції — гладка поїздка»).

════════ ВІД ІДЕЇ ДО ОБ'ЄКТА ════════
Шукаємо ΔW: F(W+ΔW, z) = m*. Лінеаризовано `J(z)·ΔW = m* − out(z)`, мінімальна
норма `ΔW = Jᵀ(JJᵀ)⁻¹r`. Ключ: **JJᵀ — це NTK**, і для ReLU без біасів
    ∂out_a/∂W_l[i,j] = x_l[i]·B_l[j,a]   ⟹   JJᵀ = Σ_l ‖x_l(z)‖²·B_lᵀB_l
де `x_l` — активації (є у форварді), `B_l` — спряжена карта (є в замиканні).
⟹ «наскільки легко зсунути ваги під цей семпл» коштує МАЙЖЕ НУЛЬ.

════════ ЩО САМЕ ПЕРЕВІРЯЄМО ════════
З мінімізації `E_z[rᵀK(z)⁻¹r]` по m виходить ЗВАЖЕНА МК:
    m̂ = Σ_i w_i·out_i / Σ_i w_i ,   w(z) = f(a(z)),  a_l(z) = ‖x_l(z)‖²
Тобто семпл важить тим менше, чим «дорожче» зсунути під нього ваги.
🔑 ВИРОК С102 («універсальної якості семпла не існує») СЮДИ НЕ ПОШИРЮЄТЬСЯ:
   там ознака була спільна для всіх мереж, тут ваги МЕРЕЖЕЗАЛЕЖНІ.
🚨 І змагаємось НЕ з голою МК, а з ВИБІЛЕНОЮ (закон С101-е).

════════ 🔒 ПЛОМБИ ════════
 W1 w ≡ const мусить відтворити базову МК ПОБІТОВО (тавтологія)
 W2 істина з НЕЗАЛЕЖНОГО сіда ⟹ шум лише ЗАНИЖУЄ ефект, не завищує
 W3 🔒 перемішані ознаки (a розсипані по семплах) ⟹ редукція = 1.00
 W4 🔒 ознаки ЧУЖОЇ мережі ⟹ редукція = 1.00
 W5 МАСШТАБУВАННЯ: N×4 ⟹ шум ÷2, редукція стоїть
 W6 оракульні ваги (фіт проти ІСТИННОЇ похибки) = СТЕЛЯ класу зважування

════════ 📉 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
   corr(похибка семпла, a) значуща, але редукція < ×1.05        45%
   редукція ×1.05…×1.3 (є що брати)                             25%
   редукція > ×1.3                                              10%
   рівно нуль, як у С102                                        20%
   🔑 оракульна стеля > ×1.5 (є об'єкт, питання лише в доступі)  35%
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
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()

    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_ = a.width, a.depth

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1.0 - c * c)))

    def truth(Ws, seed, n):
        """істина з НЕЗАЛЕЖНОГО сіда (пломба W2)."""
        rng = cp.random.default_rng(seed)
        acc = cp.zeros(W_, cp.float64)
        done = 0
        while done < n // 2:
            nb = min(a.chunk, n // 2 - done)
            z = rng.standard_normal((nb, W_), dtype=cp.float32)
            h = cp.concatenate([z, -z], axis=0)
            for l in range(D_):
                h = cp.maximum(h @ Ws[l], 0.0)
            acc += h.sum(0, dtype=cp.float64)
            done += nb
        return acc / n

    def work(Ws, seed, n):
        """робоча вибірка: антитетика + ВИБІЛЮВАННЯ (як у сабміті) + ознаки a_l."""
        rng = cp.random.default_rng(seed)
        z = rng.standard_normal((n // 2, W_), dtype=cp.float32)
        x = cp.concatenate([z, -z], axis=0)
        # вибілювання: емпірична коваріація входу ТОЧНО одинична
        cov = (x.T.astype(cp.float64) @ x.astype(cp.float64)) / x.shape[0]
        Lc = cp.linalg.cholesky(cov + 1e-9 * cp.eye(W_, dtype=cp.float64))
        x = (x.astype(cp.float64) @ cp.linalg.inv(Lc).T).astype(cp.float32)
        A = cp.zeros((x.shape[0], D_), cp.float64)      # a_l = ‖x_l‖²
        h = x
        for l in range(D_):
            A[:, l] = (h * h).sum(1, dtype=cp.float64)
            h = cp.maximum(h @ Ws[l], 0.0)
        return h.astype(cp.float64), A

    res = {}
    t0 = time.time()
    PREVA = None
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        T = truth(Ws, 990000 + net, a.ntruth)
        out, A = work(Ws, 5000 + net, a.nwork)
        N = out.shape[0]

        # ── база: рівні ваги (пломба W1) ──
        m0 = out.mean(0)
        e0 = cos_err(m0, T)

        # ── ознаки: центровані і нормовані log a_l ──
        F = cp.log(cp.maximum(A, 1e-30))
        F = F - F.mean(0, keepdims=True)
        F = F / cp.maximum(F.std(0, keepdims=True), 1e-30)

        def weighted(Fx, coef):
            w = 1.0 + Fx @ coef
            w = cp.maximum(w, 1e-3)
            return (w[:, None] * out).sum(0) / w.sum()

        # ── ОРАКУЛЬНІ ваги: фіт проти ІСТИННОЇ похибки (стеля, пломба W6) ──
        # m̂(c) ≈ m0 + (1/N)·Σ_k c_k·(F_k ⊙ (out − m0))  ⟹ лінійно по c
        G = cp.zeros((D_, W_), cp.float64)
        d = out - m0
        for k in range(D_):
            G[k] = (F[:, k][:, None] * d).sum(0) / N
        r = T - m0
        # МНК: min ‖r − Gᵀc‖
        M = G @ G.T + 1e-12 * cp.eye(D_, dtype=cp.float64)
        c_or = cp.linalg.solve(M, G @ r)
        e_or = cos_err(weighted(F, c_or), T)

        # ── 🔒 W3 перемішані ознаки ──
        perm = cp.asarray(np.random.default_rng(net).permutation(N))
        Fs = F[perm]
        Gs = cp.zeros((D_, W_), cp.float64)
        for k in range(D_):
            Gs[k] = (Fs[:, k][:, None] * d).sum(0) / N
        c_sh = cp.linalg.solve(Gs @ Gs.T + 1e-12 * cp.eye(D_, dtype=cp.float64),
                               Gs @ r)
        e_sh = cos_err(weighted(Fs, c_sh), T)

        # ── 🔒 W4 ознаки ЧУЖОЇ мережі ──
        e_fo = None
        if PREVA is not None and PREVA.shape == F.shape:
            Gf = cp.zeros((D_, W_), cp.float64)
            for k in range(D_):
                Gf[k] = (PREVA[:, k][:, None] * d).sum(0) / N
            c_fo = cp.linalg.solve(Gf @ Gf.T + 1e-12 * cp.eye(D_, dtype=cp.float64),
                                   Gf @ r)
            e_fo = cos_err(weighted(PREVA, c_fo), T)
        PREVA = F

        # ── кореляція похибки семпла з ознаками (діагностика) ──
        proj = (d @ r) / cp.maximum(cp.linalg.norm(r), 1e-30)
        cr = [float(cp.corrcoef(F[:, k], proj)[0, 1]) for k in range(D_)]

        res[str(net)] = {"e0": e0, "e_or": e_or, "e_sh": e_sh,
                         "e_fo": e_fo, "corr": cr,
                         "c_or": [float(v) for v in c_or]}
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)  база {100*e0:.4f}%  "
            f"оракул {100*e_or:.4f}%  ×{e0/max(e_or,1e-30):.4f}")
        cp.get_default_memory_pool().free_all_blocks()

    json.dump(res, open("/home/trim/wblogs/wgrad114.json", "w"))

    def agg(k):
        v = [r[k] for r in res.values() if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    e0 = agg("e0"); eo = agg("e_or"); es = agg("e_sh"); ef = agg("e_fo")
    log(f"\n📊 {len(res)} мереж · N={a.nwork} · істина {a.ntruth} (незалежний сід)")
    log(f"{'база (рівні ваги)':>32}  {100*e0:8.4f}%   ×1.0000")
    log(f"{'🔒W6 ОРАКУЛЬНІ ваги (СТЕЛЯ)':>32}  {100*eo:8.4f}%   ×{e0/eo:.4f}")
    log(f"{'🔒W3 перемішані ознаки':>32}  {100*es:8.4f}%   ×{e0/es:.4f}")
    if ef:
        log(f"{'🔒W4 ознаки чужої мережі':>32}  {100*ef:8.4f}%   ×{e0/ef:.4f}")
    C = np.array([r["corr"] for r in res.values()])
    log("\n📐 corr(похибка семпла, log‖x_l‖²) по шарах "
        f"(сер по {len(res)} мережах, шум ≈{1/np.sqrt(a.nwork):.4f}):")
    log("   " + " ".join(f"{v:+.3f}" for v in C.mean(0)))
    log("\n📉 ПЕРЕДБАЧЕННЯ: редукція <×1.05 (45%) · ×1.05–1.3 (25%) · "
        ">×1.3 (10%) · нуль (20%) · оракульна стеля >×1.5 (35%)")
    if eo:
        k = e0 / eo
        if k >= 1.5:
            log(f"🟢🟢 оракульна стеля ×{k:.3f} ⟹ ОБ'ЄКТ Є, шукати доступ")
        elif k >= 1.1:
            log(f"🟡 оракульна стеля ×{k:.3f} ⟹ малий, але не нуль")
        else:
            log(f"⛔ оракульна стеля ×{k:.3f} ⟹ зважування за нормами активацій "
                "порожнє — семпли не розрізняються цим")


if __name__ == "__main__":
    main()
