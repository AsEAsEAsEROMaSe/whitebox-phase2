"""k3gen97 — ЗАМКНЕНА ФОРМА ДЛЯ ПОРОДЖЕНОЇ ЧАСТИНИ κ₃ (те, що ReLU творить із ГАУСА).

k3gate97 показала: κ₃(ReLU) = ПРОПАГОВАНА + ПОРОДЖЕНА. Пропагована має ворітну
формулу Φ_jΦ_kΦ_l·κ₃(h) (похибка 2.2% і 8.1% на шарі 31) і зберігає ранг ТОЧНО.
Породжена — це те, що ranг ДОДАЄ. Тут вона виводиться в замкненій формі і
перевіряється прямим виміром.

=========================  ВИВІД  =============================================
Для ГАУСОВОЇ трійки коефіцієнти ермітового базису мають замкнену форму (Вік):
    λ_mnp = m!n!p!/(α!β!γ!) · ρ_jk^α ρ_jl^β ρ_kl^γ ,
    де  α+β = m,  α+γ = n,  β+γ = p   (0, якщо таких цілих ≥0 нема).
Підставляючи в  κ₃ = Σ_{m,n,p≥1} c_mnp/(m!n!p!)·a_m a_n a_p  (c = λ, бо під гаусом
λ_m00 = λ_0n0 = λ_00p = 0 при m,n,p ≥ 1) і переходячи до змінних (α,β,γ):

  🔑 κ₃(ReLU(g))_{jkl} = Σ_{α,β,γ ≥ 0}  ρ_jk^α ρ_jl^β ρ_kl^γ /(α! β! γ!)
                          · a_{α+β}(j) · a_{α+γ}(k) · a_{β+γ}(l)
     сума по тих (α,β,γ), де всі три суми α+β, α+γ, β+γ ≥ 1.

Це ТРИЛІНІЙНИЙ МЕЛЕР. Провідний член (α,β,γ)=(0,1,1) і його дві перестановки:
     ρ_jl·ρ_kl · a_1(j) a_1(k) a_2(l) + (2 перестановки)
⟹ несе РІВНО ОДИН множник φ(t) (через a_2), тоді як пропагована частина йде як
   a_1³ ~ Φ³ ⟹ відношення породжене/пропаговане ~ φ(t)/Φ(t) ⟹ ПАДАЄ з глибиною.
   Саме це й зміряно: ‖T_g‖/‖T_a‖ = 0.92 (шар 1) → 0.03 (шар 31).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (Д1) формула відтворить зміряний T_g з похибкою на рівні шуму МК (~2-5%)
      при обрізанні K=6.                                            шанс 70%
 (Д2) обрізання K=2 (лише провідні члени) дасть 15-40% похибки —
      тобто провідний член пояснює більшість, але не все.           шанс 60%
 (Д3) КОНТРОЛЬ: підстановка ЧУЖИХ ρ (перемішаних) зруйнує збіг.     шанс 90%
"""
import sys
import numpy as np
from math import factorial
from scipy.special import erfc

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
from k3gate97 import K3Acc, Phi, pr_and_top, rank1_residual, W_PATH, NSUB

SQ2PI = np.sqrt(2 * np.pi)
LAYER = 31


def phi(x):
    return np.exp(-np.asarray(x, float) ** 2 / 2) / SQ2PI


def He(n, x):
    x = np.asarray(x, float)
    h0 = np.ones_like(x)
    if n == 0:
        return h0
    h1 = x.copy()
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def a_vec(n, s, t):
    if n == 0:
        return s * (phi(t) + t * Phi(t))
    if n == 1:
        return s * Phi(t)
    return s * phi(t) * ((-1) ** n) * He(n - 2, t)


def k3_generated(rho, s, t, K=6):
    """Трилінійний Мелер: замкнена форма породженої частини."""
    A = [a_vec(n, s, t) for n in range(2 * K + 1)]
    w = len(s)
    T = np.zeros((w, w, w))
    rp = [np.ones_like(rho)]
    for _ in range(2 * K):
        rp.append(rp[-1] * rho)
    for al in range(K + 1):
        for be in range(K + 1):
            for ga in range(K + 1):
                m, n, p = al + be, al + ga, be + ga
                if m < 1 or n < 1 or p < 1:
                    continue
                if m > 2 * K or n > 2 * K or p > 2 * K:
                    continue
                coef = 1.0 / (factorial(al) * factorial(be) * factorial(ga))
                T += coef * (rp[al][:, :, None] * rp[be][:, None, :]
                             * rp[ga][None, :, :]
                             * A[m][:, None, None] * A[n][None, :, None]
                             * A[p][None, None, :])
    return T


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    B = 5000
    print("=" * 100)
    print("k3gen97 — замкнена форма породженої частини κ₃ (трилінійний Мелер)")
    print("=" * 100)
    Wall = np.load(W_PATH, mmap_mode="r")
    for net in range(NETS):
        W = np.array(Wall[net]).astype(np.float32)
        rs = np.sort(np.random.default_rng(11).permutation(256)[:NSUB])
        acc = K3Acc(NSUB)
        rng = np.random.default_rng(3000 + net)
        for _ in range(N // B):
            a = rng.standard_normal((B, 256), dtype=np.float32)
            for l in range(LAYER + 1):
                h = a @ W[l]
                if l == LAYER:
                    acc.add(h[:, rs])
                a = np.maximum(h, 0.0)
        muh, Ch, _ = acc.cum3()
        s = np.sqrt(np.diag(Ch))
        t = muh / s
        rho = Ch / np.outer(s, s)

        # ---- ІСТИНА для породженої частини: κ₃(ReLU(гаус із тим самим m,C)) ----
        ev, U = np.linalg.eigh(Ch)
        L = U * np.sqrt(np.clip(ev, 0, None))
        accG, accG2 = K3Acc(NSUB), K3Acc(NSUB)
        rg = np.random.default_rng(555 + net)
        for i in range(N // B):
            g = muh + rg.standard_normal((B, NSUB)) @ L.T
            r = np.maximum(g, 0.0)
            accG.add(r)
            if i < (N // B) // 2:
                accG2.add(r)
        _, _, Tg = accG.cum3()
        _, _, Tg2 = accG2.cum3()
        noise = np.linalg.norm(Tg2 - Tg) / np.linalg.norm(Tg)

        print("\n" + "-" * 100)
        print("МЕРЕЖА %d  мед|t|=%.3f   шум МК (пів-вибірка) = %.4f"
              % (net, np.median(np.abs(t)), noise))
        print("  K   ‖формула − вимір‖ / ‖вимір‖    PR(формула)   PR(вимір)")
        for K in (1, 2, 3, 4, 6):
            F = k3_generated(rho, s, t, K=K)
            err = np.linalg.norm(F - Tg) / np.linalg.norm(Tg)
            prF, _ = pr_and_top(F)
            prT, _ = pr_and_top(Tg)
            print("  %2d          %10.4f              %8.2f     %8.2f" % (K, err, prF, prT))

        # ---- КОНТРОЛЬ: чужі rho ----
        pm = np.random.default_rng(9).permutation(NSUB)
        Fsh = k3_generated(rho[np.ix_(pm, pm)], s, t, K=4)
        print("  КОНТРОЛЬ «чужі ρ» (переставлені): %.4f  <- мусить бути ГІРШЕ 1.0"
              % (np.linalg.norm(Fsh - Tg) / np.linalg.norm(Tg)))
        F0 = k3_generated(np.eye(NSUB), s, t, K=4)
        print("  КОНТРОЛЬ «ρ = I» (без кореляцій):  %.4f"
              % (np.linalg.norm(F0 - Tg) / np.linalg.norm(Tg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
