#!/usr/bin/env python3
"""feat113 — ЧИ ДІСТАЄТЬСЯ ПОХИБКА КРОКУ З ВЕЛИЧИН, ЯКІ ЗАМИКАННЯ ВЖЕ ЗНАЄ.

════════ ЗВІДКИ ПИТАННЯ ════════
Зауваження ЮЗЕРА: ранг ансамбля відображень (С109, ≥343) обмежує ПЕРЕБІР, а не
ГРАДІЄНТНИЙ ПОШУК — бекпроп навчає ЛЛМ із незрівнянно більшим числом станів, не
обходячи їх. ⟹ «висока розмірність» не є вироком; питання лише в тому, чи є
СИГНАЛ, по якому градієнт може спускатись.

С104 зміряла це у фазі 1 і форма 2 («модель → поправка НА НЕЙРОН») ВИЖИЛА:
R² 0.3459 на TEST при 🔒перемішаних 0.0029, реальний внесок ×1.011…1.024.
Тоді це коштувало ×1.002…1.005 у балі і було відкинуте як шум.
🔑 У фазі 2 та сама точність коштує в 10× більше (`_W_BLEND` 0.0798 → 0.820).

════════ ЩО МІРЯЄМО ════════
ЦІЛЬ  y_{l,j} = b_l[j] / s_l[j],  b_l = (наш крок від ІСТИННИХ (m_l,C_l)) − m_{l+1}
ОЗНАКИ — лише те, що замикання має В РОБОЧІЙ ТОЧЦІ (ціна ≈ нуль):
   t, Φ(t), φ(t), tφ, (t²−1)φ        ← локальна форма (Еджворт)
   R2=Σρ², R3=Σρ³, R4=Σρ⁴, A1=Σ|ρ|   ← ЗАЛЕЖНІСТЬ, бо С113 показала, що
   PR = R2²/R4, l/D, s/⟨s⟩              γ₄ у фазі 2 впала 0.385→0.060 ⟹
   взаємодії t·R2, φ·R2, R2·(t²−1)φ     негаусовість тут ЛИШЕ від залежності

ЕТАПИ
  A  ознаки з НАШИХ (m,C)      ← реалізовний у сабміті
  B  ознаки з ІСТИННИХ (m,C)   ← верхня межа сім'ї ознак
  C  вкидання λ·ŷ·s у рекурсію, λ з TRAIN, вирок на TEST

🔒 ПЛОМБИ
 F1 перемішані ознаки ⟹ R² ≈ 0
 F2 чужа модель (навчена на TRAIN, ознаки TEST, ціль ПЕРЕМІШАНА по нейронах)
    ⟹ оптимальний λ → 0
 F3 λ=0 на TEST мусить дати ту саму базу, що й step113
 F4 оракульне вкидання (справжня b) мусить відтворити стелю step113

📉 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ
   R² на TEST ≥ 0.30 (як у фазі 1)                      45%
   R² ≥ 0.50 (залежність читається краще, ніж у фазі 1)  20%
   R² < 0.15 (об'єкт зник разом із γ₄)                   35%
   реалізовний виграш ≥ 30% від стелі step113            30%
"""
import argparse
import json
import math
import os
import time

import numpy as np

SQ2PI = math.sqrt(2.0 * math.pi)
NPZ = "/home/trim/wblogs/feat113_data.npz"


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
    ap.add_argument("--kap", type=float, default=0.9993)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()

    import cupy as cp
    from cupyx.scipy.special import ndtr
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_, KS, KAP = a.width, a.depth, a.ksig, a.kap
    NP, CH = a.npairs, a.chunk

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

    def step(w, C, m, i):
        cpm = w.T @ C @ w
        if i > KS:
            cpm = cpm * KAP
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        Ph, ph = ndtr(t), phi(t)
        return mp * Ph + s * ph, mp, s, Ph, ph, cpm

    def nextC(cpm, mp, s, Ph, ph, mn):
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        return Co + cp.diag(d - cp.diagonal(Co))

    NF = 16   # число ознак

    def feats(cpm, s, t, Ph, ph, l):
        """ознаки рядка: локальна форма + СТАТИСТИКИ ЗАЛЕЖНОСТІ."""
        inv = 1.0 / s
        R = cpm * cp.outer(inv, inv)               # кореляційна матриця
        R = R - cp.diag(cp.diagonal(R))            # без діагоналі
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

    # ─────────────────── збір ознак і цілі ───────────────────
    if a.collect or not os.path.exists(NPZ):
        XA = np.zeros((a.nets, D_, W_, NF), np.float32)
        XB = np.zeros((a.nets, D_, W_, NF), np.float32)
        Y = np.zeros((a.nets, D_, W_), np.float32)
        SS = np.zeros((a.nets, D_, W_), np.float32)
        t0 = time.time()
        for net in range(a.nets):
            rng = cp.random.default_rng(7000 + net)
            sc = float(np.sqrt(2.0 / W_))
            Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * sc)
                  for _ in range(D_)]
            mt, Ct, T = truth(Ws, 113000 + net)
            Ws64 = [w.astype(cp.float64) for w in Ws]
            # ціль + ознаки B (істинні входи)
            for i in range(D_):
                Cin = cp.eye(W_, dtype=cp.float64) if i == 0 else Ct[i]
                min_ = cp.zeros(W_, cp.float64) if i == 0 else mt[i]
                mn, mp, s, Ph, ph, cpm = step(Ws64[i], Cin, min_, i)
                tgt = T if i == D_ - 1 else mt[i + 1]
                Y[net, i] = cp.asnumpy(((mn - tgt) / s).astype(cp.float32))
                SS[net, i] = cp.asnumpy(s.astype(cp.float32))
                XB[net, i] = cp.asnumpy(
                    feats(cpm, s, mp / s, Ph, ph, i).astype(cp.float32))
            # ознаки A (наша рекурсія)
            m = cp.zeros(W_, cp.float64)
            C = cp.eye(W_, dtype=cp.float64)
            for i in range(D_):
                mn, mp, s, Ph, ph, cpm = step(Ws64[i], C, m, i)
                XA[net, i] = cp.asnumpy(
                    feats(cpm, s, mp / s, Ph, ph, i).astype(cp.float32))
                if i < D_ - 1:
                    C = nextC(cpm, mp, s, Ph, ph, mn)
                m = mn
            log(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)")
            cp.get_default_memory_pool().free_all_blocks()
        np.savez_compressed(NPZ, XA=XA, XB=XB, Y=Y, SS=SS)
        log(f"💾 {NPZ}")

    d = np.load(NPZ)
    XA, XB, Y = d["XA"], d["XB"], d["Y"]
    nets = XA.shape[0]
    tr = np.arange(0, nets // 2)
    te = np.arange(nets // 2, nets)

    def fit_eval(X, name, shuffle=False):
        xt = X[tr].reshape(-1, NF).astype(np.float64)
        yt = Y[tr].reshape(-1).astype(np.float64)
        xe = X[te].reshape(-1, NF).astype(np.float64)
        ye = Y[te].reshape(-1).astype(np.float64)
        if shuffle:
            rs = np.random.default_rng(0)
            xt = xt[rs.permutation(len(xt))]
        sd = xt.std(0)
        sd[sd < 1e-12] = 1.0
        xt2 = xt / sd
        xe2 = xe / sd
        A = xt2.T @ xt2 + 1e-3 * len(xt2) * np.eye(NF)
        w = np.linalg.solve(A, xt2.T @ yt)
        pr = xe2 @ w
        r2 = 1.0 - ((ye - pr) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
        r2tr = 1.0 - ((yt - xt2 @ w) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        log(f"  {name:>34}  R² TRAIN {r2tr:7.4f}   TEST {r2:7.4f}")
        return w / sd, r2

    log(f"\n📊 feat113 · {nets} мереж · TRAIN {len(tr)} / TEST {len(te)} · "
        f"{Y.size} прикладів")
    wB, r2B = fit_eval(XB, "B: ознаки з ІСТИННИХ (m,C)")
    wA, r2A = fit_eval(XA, "A: ознаки з НАШИХ (m,C) [реалізовний]")
    _, r2S = fit_eval(XA, "🔒 F1 перемішані ознаки", shuffle=True)
    np.save("/home/trim/wblogs/feat113_wA.npy", wA)
    log(f"\n🔒 F1: {r2S:.4f} мусить бути ≈0 "
        f"{'✅' if abs(r2S) < 0.02 else '🚨'}")
    log("📉 ПЕРЕДБАЧЕННЯ: R²≥0.30 (45%) · ≥0.50 (20%) · <0.15 (35%)")
    json.dump({"r2A": r2A, "r2B": r2B, "r2S": r2S},
              open("/home/trim/wblogs/feat113_r2.json", "w"))


if __name__ == "__main__":
    main()
