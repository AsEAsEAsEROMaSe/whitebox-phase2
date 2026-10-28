#!/usr/bin/env python3
"""vsrc114 — ДЕ НАРОДЖУЄТЬСЯ ДИСПЕРСІЯ ВИХОДУ (нове питання, ніхто не ставив).

════════ ЗВІДКИ ════════
`anchor114b`: R²(out ~ x₁) = −0.038 навіть при рангу 2 ⟹ дисперсія виходу НЕ
народжується в першому шарі. А єдиний ТОЧНИЙ якір задачі (`E[relu(zW₀)]`)
лежить саме там ⟹ «точна істина там, де дисперсії нема; дисперсія там, де
істини нема». Це пояснює всі провали редукції дисперсії (КЗ ×1.0096,
сферизація ×0.74, RQMC ×1.0156, кубатура ×1.02).
⟹ ПИТАННЯ: який ВНЕСОК кожного шару у Var(out)?

════════ ПРИЛАД ════════
Беремо ДВІ незалежні траєкторії z, z'. Гібридна траєкторія H_k: іти по z до
шару k, далі підставити активацію з z' і йти далі. Тоді
    Var_k = Var( out(H_k) )  над парами
міряє, скільки дисперсії ЗАЛИШИЛОСЬ після шару k. Спад кривої по k — це і є
розподіл народження дисперсії по глибині.
🔑 Дешевша і чистіша форма: підмінюємо не активацію, а ФЛУКТУАЦІЮ —
    x_k ← E[x_k] + (x_k(z') − E[x_k])   (те саме, але видно внесок)
Тут використовуємо пряму підміну активації (H_k), бо E[x_k] невідоме.

 k = 0   уся траєкторія з z'   ⟹ Var = повна
 k = D   уся траєкторія з z    ⟹ Var = повна
 проміжні: скільки дисперсії вже «зафіксовано» першими k шарами

════════ 🔒 ПЛОМБИ ════════
 V1 k=0 і k=D мусять дати ОДНАКОВУ дисперсію (обидві — повна мережа)
 V2 корельованість: cov(out(H_k), out(z)) монотонно спадає по k
 V3 масштабування N ⟹ криві стоять

📉 ПЕРЕДБАЧЕННЯ: >60% дисперсії з останніх 4 шарів (45%) · рівномірно (30%) ·
   >60% з ПЕРШИХ 4 шарів (25%)
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
    ap.add_argument("--nsamp", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()

    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_ = a.width, a.depth

    CV, CC = [], []
    t0 = time.time()
    for net in range(a.nets):
        rng = cp.random.default_rng(7000 + net)
        sc = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
              for _ in range(D_)]
        srng = cp.random.default_rng(4000 + net)
        # накопичувачі: для кожного k — сума, сума квадратів, крос із out(z)
        S1 = cp.zeros((D_ + 1, W_), cp.float64)
        S2 = cp.zeros((D_ + 1, W_), cp.float64)
        CX = cp.zeros((D_ + 1, W_), cp.float64)
        S1r = cp.zeros(W_, cp.float64)
        S2r = cp.zeros(W_, cp.float64)
        n = 0
        done = 0
        while done < a.nsamp:
            nb = min(a.chunk, a.nsamp - done)
            z = srng.standard_normal((nb, W_), dtype=cp.float32)
            zp = srng.standard_normal((nb, W_), dtype=cp.float32)
            # прямі траєкторії, зберігаємо активації
            A = [z]
            h = z
            for l in range(D_):
                h = cp.maximum(h @ Ws[l], 0.0)
                A.append(h)
            B = [zp]
            h = zp
            for l in range(D_):
                h = cp.maximum(h @ Ws[l], 0.0)
                B.append(h)
            ref = A[D_].astype(cp.float64)
            S1r += ref.sum(0); S2r += (ref * ref).sum(0)
            for k in range(D_ + 1):
                # до шару k іти по z, далі підставити активацію з z'
                h = A[k] if k == D_ else B[k]
                if k < D_:
                    h = B[k]
                    for l in range(k, D_):
                        h = cp.maximum(h @ Ws[l], 0.0)
                o = h.astype(cp.float64)
                S1[k] += o.sum(0); S2[k] += (o * o).sum(0)
                CX[k] += (o * ref).sum(0)
            n += nb
            done += nb
        mu = S1 / n
        var = S2 / n - mu * mu
        mur = S1r / n
        varr = S2r / n - mur * mur
        cov = CX / n - mu * mur
        # нормуємо на повну дисперсію
        v = cp.asnumpy(var.sum(1) / varr.sum())
        c = cp.asnumpy(cov.sum(1) / varr.sum())
        CV.append(v); CC.append(c)
        log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)  🔒V1 "
            f"k=0 {v[0]:.4f} vs k=D {v[D_]:.4f}")
        cp.get_default_memory_pool().free_all_blocks()

    V = np.array(CV).mean(0)
    C = np.array(CC).mean(0)
    log(f"\n📊 {a.nets} мереж · N={a.nsamp}")
    log(f"🔒 V1: Var(k=0)={V[0]:.4f}  Var(k=D)={V[D_]:.4f}  "
        f"{'✅' if abs(V[0]-V[D_]) < 0.05 else '🚨'}")
    log(f"\n{'k':>4}{'Var(H_k)/Var':>14}{'corr з out(z)':>16}{'внесок шару k':>16}")
    for k in range(D_ + 1):
        d = (C[k] - C[k - 1]) if k > 0 else 0.0
        log(f"{k:>4}{V[k]:>14.4f}{C[k]:>16.4f}{d:>16.4f}")
    log("\n🔑 corr(H_k, out(z)) = частка дисперсії, ЗАФІКСОВАНА першими k шарами")
    tail4 = C[D_] - C[D_ - 4]
    log(f"📐 частка з ОСТАННІХ 4 шарів: {tail4:.4f} · з ПЕРШИХ 4: {C[4]:.4f}")
    log("\n📉 ПЕРЕДБАЧЕННЯ: >60% з останніх 4 (45%) · рівномірно (30%) · "
        ">60% з перших 4 (25%)")


if __name__ == "__main__":
    main()
