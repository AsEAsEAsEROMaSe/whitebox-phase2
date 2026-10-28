"""feat104e — END-TO-END ТОЧНО: лінійне наближення + МНК замість Adam.

🚨 ЧОМУ ADAM ПРОВАЛИВСЯ (feat104d): він НОРМАЛІЗУЄ градієнт, тож розмір кроку
не залежить від крутизни. А мінімум тут надзвичайно гострий — `‖w‖=0.004` на
нормованих ознаках уже ПОДВОЮЄ похибку (0.4548% → 1.0%). Оптимізатор одним
кроком вискакував із ями і не повертався. Це третій випадок за сесію, коли
масштаб кроку промахується на два порядки (trans104, feat104d, і тут).

ПРАВИЛЬНИЙ ШЛЯХ — використати, що поправка МАЛА, тож вихід ЛІНІЙНИЙ по w:
    m_final(w) ≈ m0 + Σ_k w_k · J_k ,   J_k = ∂m_final/∂w_k  (чисельно, 2 оцінки)
Косинусна ціль із вільним масштабом = проєкція, ортогональна до T:
    P_i = I − T_iT_iᵀ/‖T_i‖²  ⟹  min_w Σ_i ‖P_i m0_i + P_i J_i w‖² / ‖T_i‖²
Це ЗВИЧАЙНИЙ МНК ⟹ точний оптимум у лінійному наближенні, без ітерацій,
за 29 рекурсій замість 6000.

🔒 ПЛОМБИ: (1) w=0 ⟹ база 0.4548%/0.4384% (TRAIN/TEST) — як у feat104c/d
           (2) передбачення МНК мусить збігтися з ПРЯМИМ виміром при знайденому
               w (перевірка лінійності — якщо розходиться, наближення хибне)
           (3) вирок на TEST, w з TRAIN · ридж-скан теж лише на TRAIN
           (4) 🔒 «ЧУЖА»: w застосований із переставленими мережами
           (5) стеля ×1.2436 · проксі-навчання дало ×1.011…1.024

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - лінійне наближення виявиться точним (пломба 2 зійдеться у 10%):  75%
   - TEST > ×1.05:                                                    40%
   - TEST > ×1.10:                                                    20%
   - «чужа» зламається:                                               85%
"""
import math
import os
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG, NF = 8, 14
GX = [0, 1, 4, 8, 16, 24, 31]
G3T = np.interp(np.arange(DEPTH), GX,
                [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398])
G4T = np.interp(np.arange(DEPTH), GX,
                [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220])
SRC = "/home/trim/wblogs/feat104c.npz"
if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
z = np.load(SRC)
XA, WA, TA = z["X"], z["W"], z["T"]
NN = XA.shape[0]
TR, TE = list(range(NN // 2)), list(range(NN // 2, NN))
mu = cp.asarray(XA.reshape(-1, NF).mean(0).astype(np.float64))
sd = cp.asarray(XA.reshape(-1, NF).std(0).astype(np.float64) + 1e-12)
del XA
WG = [cp.asarray(WA[k]) for k in range(NN)]
TG = [cp.asarray(TA[k]) for k in range(NN)]
del WA
print(f"feat104e · мереж {NN} · TRAIN {len(TR)} / TEST {len(TE)}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def fwd(idx, w, kap=0.9985, gam=1.048, perm=None):
    """повертає m_final (P,256) — саме вектори, а не скаляр похибки"""
    P = len(idx)
    wg = cp.asarray(w)[None, None, :]
    Wb = cp.stack([WG[k] for k in idx])
    m = cp.zeros((P, W_), dtype=cp.float64)
    C = cp.empty((P, W_, W_), dtype=cp.float32)
    C[:] = cp.eye(W_, dtype=cp.float32)
    dg = cp.arange(W_)
    for i in range(DEPTH):
        ww = Wb[:, i]
        cpm = cp.matmul(cp.matmul(ww.transpose(0, 2, 1), C), ww).astype(
            cp.float64)
        if i > KSIG:
            cpm = cpm * kap
        s2 = cp.maximum(cpm[:, dg, dg], 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, ww.astype(cp.float64))
        # 🚨 без кліпу при виродженні s→0 маємо u→∞ і (u²−1)·φ(u) = inf·0 = nan
        #    (зміряний max|t| = 5.8, тож ±40 безпечно з великим запасом)
        u = cp.clip(gam * (mp / s), -40.0, 40.0)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        if i > KSIG:
            tr = cpm[:, dg, dg].sum(1)
            er = tr * tr / cp.maximum((cpm * cpm).sum((1, 2)), 1e-30)
            rw = cp.sqrt((cpm * cpm).sum(2))
            one = cp.ones((P, W_), dtype=cp.float64)
            f = cp.stack([
                u, Ph, ph, u * u, s / cp.mean(s, 1, keepdims=True),
                one * (i / (DEPTH - 1.0)), one * cp.mean(u, 1, keepdims=True),
                one * cp.std(u, 1, keepdims=True),
                s2 / cp.mean(s2, 1, keepdims=True),
                rw / cp.maximum(cp.sqrt((cpm * cpm).sum((1, 2)))[:, None],
                                1e-30),
                (cpm * cpm).sum(2) / cp.maximum(s2 * tr[:, None], 1e-30),
                one * (er / 256.0)[:, None], one * 0.0,
                (-(G3T[i] / 6.0) * u * ph
                 + (G4T[i] / 24.0) * (u * u - 1.0) * ph)], axis=2)
            f = (f - mu) / sd
            if perm is not None:
                f = f[perm]
            # 🚨 без обмеження поправка розганяється через шари: зсув mn роздуває
            #    C, а діагональ у float32 переповнюється (mn² > 3.4e38 ⟹ inf).
            #    Зміряне b/s ≈ 0.008 ⟹ межа 0.2 дає запас 25×.
            mn = mn - cp.clip((f * wg).sum(2), -0.2, 0.2) * s
        if i == DEPTH - 1:
            break
        pv = (ph / s).astype(cp.float32)
        P32 = Ph.astype(cp.float32)
        c32 = cpm.astype(cp.float32)
        C2 = c32 * c32
        C2 *= 0.5
        C2 *= pv[:, :, None]
        C2 *= pv[:, None, :]
        c32 *= P32[:, :, None]
        c32 *= P32[:, None, :]
        c32 += C2
        del C2
        c32[:, dg, dg] = ((mp * mp + s2) * Ph + (mp * s) * ph
                          - mn * mn).astype(cp.float32)
        C, m = c32, mn
    del Wb, C, cpm
    return mn


def cerr(idx, X):
    tot = tt = 0.0
    for a, k in enumerate(idx):
        t = TG[k]
        x = X[a]
        tot += float(t @ t) - float(x @ t) ** 2 / max(float(x @ x), 1e-300)
        tt += float(t @ t)
    return math.sqrt(max(tot, 0.0) / tt)


w0 = np.zeros(NF)
m0tr, m0te = fwd(TR, w0), fwd(TE, w0)
btr, bte = cerr(TR, m0tr), cerr(TE, m0te)
print(f"\n🔒 ПЛОМБА (1) w=0 ⟹ TRAIN {100*btr:.4f}% · TEST {100*bte:.4f}%")

H = 0.0005
J = []
for k in range(NF):
    wp, wm = w0.copy(), w0.copy()
    wp[k] += H
    wm[k] -= H
    J.append(cp.asnumpy((fwd(TR, wp) - fwd(TR, wm)) / (2 * H)))
J = np.stack(J, axis=2)                       # (P, 256, NF)
M0 = cp.asnumpy(m0tr)
TT = np.stack([cp.asnumpy(TG[k]) for k in TR])

A = np.zeros((NF, NF))
b = np.zeros(NF)
for a in range(len(TR)):
    t = TT[a]
    nt = t / math.sqrt(t @ t)
    r = M0[a] - nt * (nt @ M0[a])             # P_i m0
    Ji = J[a] - np.outer(nt, nt @ J[a])       # P_i J
    sc = 1.0 / (t @ t)
    A += sc * (Ji.T @ Ji)
    b -= sc * (Ji.T @ r)

print(f"\n{'ридж':>10}{'TRAIN пред.':>13}{'TRAIN факт':>12}{'TEST':>10}"
      f"{'×TEST':>9}{'‖w‖':>9}")
best = None
for lg in (-1, 0, 1, 2, 3, 4):
    lam = 10.0 ** lg * np.trace(A) / NF * 1e-3
    w = np.linalg.solve(A + lam * np.eye(NF), b)
    pred = 0.0
    for a in range(len(TR)):
        t = TT[a]
        nt = t / math.sqrt(t @ t)
        r = M0[a] - nt * (nt @ M0[a]) + (J[a] - np.outer(nt, nt @ J[a])) @ w
        pred += (r @ r) / (t @ t)
    pred = math.sqrt(max(pred, 0.0) / len(TR))
    ftr = cerr(TR, fwd(TR, w))
    fte = cerr(TE, fwd(TE, w))
    print(f"{lg:>10}{100*pred:>12.4f}%{100*ftr:>11.4f}%{100*fte:>9.4f}%"
          f"{bte/fte:>9.4f}{np.linalg.norm(w):>9.5f}")
    if best is None or ftr < best[0]:
        best = (ftr, fte, w, pred)

ftr, fte, w, pred = best
print(f"\n🔒 ПЛОМБА (2) лінійність: МНК передбачив {100*pred:.4f}%, "
      f"прямий вимір {100*ftr:.4f}% ⟹ розбіжність "
      f"{100*abs(pred-ftr)/max(ftr,1e-12):.1f}%")
pm = cp.asarray(np.roll(np.arange(len(TE)), 1))
ce = cerr(TE, fwd(TE, w, perm=pm))
print(f"🔒 ПЛОМБА (4) «чужа»: {100*ce:.4f}%  ×{bte/ce:.4f}")
print(f"\nВИРОК: TRAIN ×{btr/ftr:.4f} · TEST ×{bte/fte:.4f}")
print(f"🔒 (5) стеля ×1.2436 · проксі-навчання ×1.011…1.024")
