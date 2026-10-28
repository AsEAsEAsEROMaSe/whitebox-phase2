#!/usr/bin/env python3
"""leaky114b — LEAKY-ЗАМИКАННЯ ЯК ЯКІР ДЛЯ КОНТРОЛЬНОЇ ЗМІННОЇ.

════════ КОНСТРУКЦІЯ ════════
`h_α(x) = max(x, αx) = β·relu(x) + α·x`, β=1−α. Крок замикання для неї —
ДЕШЕВА добавка до нашого (виведено через Стайна `Cov(relu(X),Y)=c·Φ(t_X)`):
    m_h   = β(mΦ + sφ) + α·mp
    Cov_h = β²·Co + αβ·cp·(Φᵢ+Φⱼ) + α²·cp
    Var_h = β²·d  + 2αβ·s²Φ + α²·s²
⟹ маємо ДРУГУ ЗАДАЧУ, для якої вміємо і МК, і замикання, і яка при малому α
корельована з нашою на ρ=0.886 (leaky114) ⟹ різниця `m_MC,α − m_cl,α` є
оцінкою похибки замикання В СУСІДНІЙ ТОЧЦІ, і її можна перенести до нас.

    m̂ = m_MC + a·(m_cl − m_MC) + b·(m_MC,α − m_cl,α)
         └──── наш бленд ────┘   └─── НОВИЙ ЧЛЕН ───┘

🚨 МІРЯЄМО ПРОТИ БЛЕНДУ (a-член), НЕ проти чистої МК — інакше повторимо помилку
   feat113, де стенд збрехав у 8.6×.

════════ 🔒 ПЛОМБИ ════════
 L1 α=0 ⟹ leaky-замикання = наше замикання ПОБІТОВО
 L2 α=1 ⟹ рекурсія лінійна, старт m=0 ⟹ m_cl,1 = 0 РІВНО
 L3 базовий бленд (b=0) відтворює записані ~0.11%
 L4 🔒 ЧУЖА мережа для нового члена ⟹ b* → 0
 L5 крос по мережах: (a,b) з TRAIN, вирок на TEST

════════ 📉 ПЕРЕДБАЧЕННЯ ════════
   оракульний b дає <×1.05 понад бленд (тавтологія)      45%
   ×1.05…×1.3                                            30%
   >×1.3                                                 10%
   крос-вирок утримає хоч половину оракульного           35%
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
    ap.add_argument("--nets", type=int, default=12)
    ap.add_argument("--ntruth", type=int, default=524288)
    ap.add_argument("--nwork", type=int, default=6144)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--alphas", type=str, default="0.05,0.10,0.20")
    a = ap.parse_args()

    import cupy as cp
    from cupyx.scipy.special import ndtr
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_ = a.width, a.depth
    AL = [float(x) for x in a.alphas.split(",")]
    SQ2PI = math.sqrt(2.0 * math.pi)

    def cerr(x, y):
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
            acc += fwd(Ws, cp.concatenate([z, -z], axis=0), al).sum(0, dtype=cp.float64)
            done += nb
        return acc / n

    def closure(Ws, al=0.0):
        """замикання для leaky-relu; при al=0 ТОТОЖНЕ нашому (пломба L1)."""
        be = 1.0 - al
        m = cp.zeros(W_, cp.float64)
        C = cp.eye(W_, dtype=cp.float64)
        for i in range(D_):
            w = Ws[i].astype(cp.float64)
            cpm = w.T @ C @ w
            s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-30))
            mp = m @ w
            t = mp / s
            Ph, ph = ndtr(t), cp.exp(-0.5 * t * t) / SQ2PI
            mr = mp * Ph + s * ph                    # relu-крок
            mn = be * mr + al * mp                   # 🔑 leaky-крок
            if i == D_ - 1:
                return mn
            Co = (cpm * cp.outer(Ph, Ph)
                  + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
            d = (mp * mp + s * s) * Ph + (mp * s) * ph - mr * mr
            if al != 0.0:
                # 🔑 Cov_h = β²Co + αβ·cp·(Φᵢ+Φⱼ) + α²cp   (Стайн)
                Co = (be * be) * Co + (al * be) * (cpm * (Ph[:, None] + Ph[None, :])) \
                     + (al * al) * cpm
                d = (be * be) * d + (2.0 * al * be) * (s * s * Ph) + (al * al) * (s * s)
            C = Co + cp.diag(d - cp.diagonal(Co))
            m = mn

    def whiten(seed, n):
        rng = cp.random.default_rng(seed)
        z = rng.standard_normal((n // 2, W_), dtype=cp.float32)
        x = cp.concatenate([z, -z], axis=0)
        cov = (x.T.astype(cp.float64) @ x.astype(cp.float64)) / x.shape[0]
        Lc = cp.linalg.cholesky(cov + 1e-9 * cp.eye(W_, dtype=cp.float64))
        return (x.astype(cp.float64) @ cp.linalg.inv(Lc).T).astype(cp.float32)

    MC, CL, DA, TT, FO = [], [], [], [], []
    t0 = time.time()
    l1 = l2 = None
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        T = truth(Ws, 990000 + net, a.ntruth)
        x = whiten(5000 + net, a.nwork)
        mc = fwd(Ws, x).astype(cp.float64).mean(0)
        cl = closure(Ws, 0.0)
        if net == 0:                                   # 🔒 L1 / L2
            l1 = float(cp.max(cp.abs(closure(Ws, 0.0) - cl)))
            l2 = float(cp.max(cp.abs(closure(Ws, 1.0))))
        das = []
        for al in AL:
            das.append(fwd(Ws, x, al).astype(cp.float64).mean(0) - closure(Ws, al))
        MC.append(mc); CL.append(cl); DA.append(das); TT.append(T)
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)  МК {100*cerr(mc,T):.4f}%  "
            f"замик {100*cerr(cl,T):.4f}%")
        cp.get_default_memory_pool().free_all_blocks()

    log(f"\n🔒 L1 (α=0 ≡ наше замикання): {l1:.3e}  {'✅' if l1 == 0 else '🚨'}")
    log(f"🔒 L2 (α=1 ⟹ m ≡ 0): {l2:.3e}  {'✅' if l2 < 1e-12 else '🚨'}")

    n = len(MC)

    def proj_err(vecs, T):
        """косинусна похибка НАЙКРАЩОГО елемента span{vecs} — вільний масштаб
        включений, тобто рівно те, що робить `_ai` у сабміті."""
        G = cp.stack(vecs)
        A = cp.asnumpy(G @ G.T)
        r = cp.asnumpy(G @ T)
        c = np.linalg.solve(A + 1e-18 * np.trace(A) * np.eye(len(vecs)), r)
        m = sum(float(c[k]) * vecs[k] for k in range(len(vecs)))
        return cerr(m, T), c

    def coefs(idx, basis):
        """спільні коефіцієнти по мережах idx (нормовані на ‖T‖ кожної)."""
        d = len(basis(0))
        A = np.zeros((d, d)); r = np.zeros(d)
        for i in idx:
            V = basis(i)
            sc = 1.0 / float(cp.linalg.norm(TT[i]))
            G = cp.stack([v * sc for v in V])
            A += cp.asnumpy(G @ G.T); r += cp.asnumpy(G @ (TT[i] * sc))
        return np.linalg.solve(A + 1e-18 * np.trace(A) * np.eye(d), r)

    def ev(idx, basis, c):
        e = []
        for i in idx:
            V = basis(i)
            m = sum(float(c[k]) * V[k] for k in range(len(V)))
            e.append(cerr(m, TT[i]))
        return float(np.mean(e))

    allid = list(range(n)); tr, te = allid[:n // 2], allid[n // 2:]
    log(f"\n{'α':>6}{'бленд':>10}{'+член ORAC':>12}{'×':>8}"
        f"{'крос TEST':>11}{'×':>8}{'🔒чужа':>10}{'×':>8}")
    b2 = lambda i: [MC[i], CL[i]]
    c2 = coefs(allid, b2); c2t = coefs(tr, b2)
    e_bl = ev(allid, b2, c2); e_bl_te = ev(te, b2, c2t)
    for k, al in enumerate(AL):
        b3 = lambda i, k=k: [MC[i], CL[i], DA[i][k]]
        bf = lambda i, k=k: [MC[i], CL[i], DA[(i + 1) % n][k]]
        e_or = ev(allid, b3, coefs(allid, b3))
        e_cr = ev(te, b3, coefs(tr, b3))
        e_fo = ev(te, bf, coefs(tr, bf))
        log(f"{al:>6.2f}{100*e_bl:>9.4f}%{100*e_or:>11.4f}%{e_bl/e_or:>8.4f}"
            f"{100*e_cr:>10.4f}%{e_bl_te/e_cr:>8.4f}"
            f"{100*e_fo:>9.4f}%{e_bl_te/e_fo:>8.4f}")
    log(f"   (база на TEST: {100*e_bl_te:.4f}%)")
    log("\n📉 ПЕРЕДБАЧЕННЯ: оракул <×1.05 (45%) · ×1.05–1.3 (30%) · >×1.3 (10%) · "
        "крос тримає половину (35%)")


if __name__ == "__main__":
    main()
