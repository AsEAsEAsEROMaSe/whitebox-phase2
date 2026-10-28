"""k3diag97 — ЧИ ВХОДИТЬ κ₃ У σ (діагональ C), ЧИ ЛИШЕ В ПОЗАДІАГОНАЛЬ?

ВИВІД. Для ОДНІЄЇ змінної та сама тотожність дає Var(f) = Σ_{m,n≥1} c_mn/(m!n!) a_m a_n
з c_mn = λ_mn − λ_m0 λ_0n, де λ на ОДНІЙ змінній: λ_12 = γ₃, λ_22 = 2+γ₄, λ_10=λ_20=0.
 ⟹ Var_G = Σ_n a_n²/n!        (гаусова частина)
 ⟹ внесок κ₃:  γ₃·a_1·a_2 = γ₃·σ²·Φ(t)·φ(t)      ← ВХОДИТЬ, але через φ(t)
 ⟹ внесок γ₄:  γ₄·a_2²/4    = γ₄·σ²·φ(t)²/4      ← придушений ДВІЧІ

ПЕРЕДБАЧЕННЯ: κ₃ таки входить у σ, але з вагою ~γ₃·Φφ ≈ 0.4·8.7e-4 = 3.5e-4,
тобто ~0.035% — а допуск на σ це 1%. Тож κ₃ у 30 разів НИЖЧЕ порога, за яким
σ узагалі помітить різницю ⟹ несення κ₃ не може полагодити σ.   шанс 70%
"""
import sys, numpy as np
from math import factorial
sys.path.insert(0, ".")
from k3gate97 import K3Acc, Phi, W_PATH
from k3gen97 import phi, He, a_vec

NSUB, LAYER_SET, B = 256, [8, 16, 24, 31], 5000
Wall = np.load(W_PATH, mmap_mode="r")
print("=" * 96)
print("k3diag97 — внесок κ₃ у ДІАГОНАЛЬ (тобто в σ). N=200000")
print("=" * 96)
for net in (0, 1):
    W = np.array(Wall[net]).astype(np.float32)
    rng = np.random.default_rng(3000 + net)
    n_ = 0
    S = {l: [np.zeros(256) for _ in range(6)] for l in LAYER_SET}   # моменти h і relu
    for _ in range(200000 // B):
        a = rng.standard_normal((B, 256), dtype=np.float32)
        for l in range(max(LAYER_SET) + 1):
            h = a @ W[l]
            if l in S:
                hh = h.astype(np.float64); rr = np.maximum(hh, 0.0)
                S[l][0] += hh.sum(0); S[l][1] += (hh**2).sum(0); S[l][2] += (hh**3).sum(0)
                S[l][3] += (hh**4).sum(0)
                S[l][4] += rr.sum(0);  S[l][5] += (rr**2).sum(0)
            a = np.maximum(h, 0.0)
        n_ += B
    print("\nМЕРЕЖА %d" % net)
    print("  шар | мед|t| |  похибка діаг: гаус  +γ₃-член  +γ₃γ₄ | внесок κ₃ | медіана(живі): гаус  +γ₃ | мертв/живих")
    for l in LAYER_SET:
        m1, m2, m3, m4, r1, r2 = [x / n_ for x in S[l]]
        mu = m1; var = m2 - mu**2; s = np.sqrt(var); t = mu / s
        g3 = (m3 - 3*mu*m2 + 2*mu**3) / s**3
        g4 = (m4 - 4*mu*m3 + 6*mu**2*m2 - 3*mu**4) / s**4 - 3
        Vt = r2 - r1**2                                   # ІСТИНА: Var(ReLU)
        # мертві нейрони (ReLU завжди 0) дають Var=0 -> ділення на нуль. Маскуємо
        # і рахуємо, скільки їх: це саме по собі число про режим мережі.
        ok = Vt > 1e-10 * np.max(Vt)
        a1, a2 = a_vec(1, s, t), a_vec(2, s, t)
        VG = sum(a_vec(n, s, t)**2 / factorial(n) for n in range(1, 40))
        # 🚨 ПОДІЛ НА ПОНЕЙРОННУ Var НЕПРИДАТНИЙ: майже мертві нейрони мають
        # Var ~ 1e-17*s^2, і відносна похибка там — чистий шум МК (дає 36000%).
        # Правильний агрегат — енергетично зважений, як для позадіагоналі.
        den = np.sqrt((Vt[ok]**2).mean())
        f = lambda x: np.sqrt((x[ok]**2).mean()) / den
        d0 = f(Vt - VG)
        d1 = f(Vt - VG - g3*a1*a2)
        d2 = f(Vt - VG - g3*a1*a2 - g4*a2**2/4)
        contrib = f(g3*a1*a2)
        # додатково: медіана понейронної відносної похибки серед ЖИВИХ (t > -1)
        alive = ok & (t > -1.0)
        med0 = np.median(np.abs((Vt - VG)[alive] / Vt[alive]))
        med1 = np.median(np.abs((Vt - VG - g3*a1*a2)[alive] / Vt[alive]))
        print("  %4d | %6.3f | %9.4f%% %9.4f%% %9.4f%% | %8.4f%% | %8.4f%% %8.4f%% | %3d/%3d"
              % (l, np.median(np.abs(t)), 100*d0, 100*d1, 100*d2, 100*contrib,
                 100*med0, 100*med1, (~ok).sum(), alive.sum()))
