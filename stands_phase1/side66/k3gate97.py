"""k3gate97 — ЧИ ЗБЕРІГАЄ ReLU РАНГ κ₃, І ЧИ РОСТЕ РАНГ ІЗ ГЛИБИНОЮ?

===================  ВИВІД ДО ПРОГОНУ (той самий ермітів апарат)  ==============
Трилінійне продовження точної білінійної тотожності. Для f,g,q від X_j,X_k,X_l:

  E[fgq] = Σ_{m,n,p≥0} λ_mnp/(m!n!p!) · a_m a_n a_p ,  λ_mnp = E[He_m He_n He_p]

Третій кумулянт κ₃(f,g,q) = E[fgq] − E[fg]E[q] − E[fq]E[g] − E[gq]E[f] + 2E[f]E[g]E[q]
дає РІВНО ту саму форму зі сполученим коефіцієнтом

  c_mnp = λ_mnp − λ_mn0·λ_00p − λ_m0p·λ_0n0 − λ_0np·λ_m00 + 2·λ_m00·λ_0n0·λ_00p

і c_mnp ≡ 0, щойно БУДЬ-ЯКИЙ індекс = 0 (перевірено символьно) ⟹

  🔑 κ₃(ReLU)_{jkl} = Σ_{m,n,p ≥ 1} c_mnp/(m! n! p!) · a_m^(j) a_n^(k) a_p^(l)

з уже запечатаними a_1 = σΦ(t) і a_n = σφ(t)(−1)ⁿHe_{n−2}(t) при n ≥ 2.

📌 НАСЛІДОК, ЯКИЙ І ВИРІШУЄ ПИТАННЯ РАНГУ. Член (1,1,1) дає c_111 = κ̄₃(X)_{jkl},
тобто СТАНДАРТИЗОВАНИЙ третій кумулянт ВХОДУ, помножений на a_1 по кожному індексу:

  🎯 κ₃(ReLU(h))_{jkl}  ≈  Φ(t_j)·Φ(t_k)·Φ(t_l) · κ₃(h)_{jkl}          [ВОРІТНА ФОРМУЛА]

Це ПОЕЛЕМЕНТНЕ масштабування ⟹ CP-ранг ЗБЕРІГАЄТЬСЯ ТОЧНО: якщо κ₃(h) = c·a^⊗3,
то κ₃(ReLU) = c·(Φ⊙a)^⊗3, теж ранг-1. Ворота — це просто ймовірність активації.

А ВСІ інші члени мають індекс ≥2 ⟹ несуть множник φ(t) (той самий закон
придушення, що і в парному випадку) ⟹ при t≈3.5 вони придушені на φ/Φ ≈ 8.7e−4.

🚨 АЛЕ Є ДРУГЕ ДЖЕРЕЛО, І ВОНО НЕ ПРОПАГОВАНЕ. Навіть при κ₃(h) = 0 сам ReLU
ТВОРИТЬ скіс: c_112 = λ_112 = E[X_j X_k He_2(X_l)] дорівнює 2ρ_jl·ρ_kl уже під
ГАУСОМ. Ця частина має структуру «адамарів квадрат кореляцій» і рангу-1 НЕ має.
⟹ питання не «чи зберігає ReLU ранг» (зберігає, для пропагованої частини —
ТОЧНО), а «чи не затоплює породжена частина пропаговану».

===========================  ЩО МІРЯЄТЬСЯ  =====================================
Розділяю два джерела ПРЯМО, а не оцінюю:
  T_h = κ₃(h)                       третій кумулянт ПЕРЕДнейронних активацій
  T_a = κ₃(ReLU(h))                 те, що треба нести далі
  T_g = κ₃(ReLU(g)),  g ~ N(m, C)   ⟵ ПОРОДЖЕНА частина (B), точно й окремо:
                                       гаусів сурогат із ТИМИ САМИМИ (m, C)
  P   = Φ_j Φ_k Φ_l · T_h           ⟵ передбачення ВОРІТНОЇ формули
Тоді (T_a − T_g) є пропагованою частиною (A), і перевіряється ПРЯМО:
      ‖T_a − T_g − P‖ / ‖T_a − T_g‖     ← наскільки ворітна формула точна
Ранг кожного об'єкта — через сингулярні числа mode-1 розгортки.

⚠️ Сусідня сесія міряла PR на mode-1 розгортці і має розбіжність ×2.8 з С74.
   Я рахую ТУ САМУ величину на своєму зрізі — але ранг-1-ність під звуженням
   на підмножину нейронів ЗБЕРІГАЄТЬСЯ, тож висновок про ранг стійкий, навіть
   якщо саме число PR залежить від зрізу.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (протокол п.6):
 (Г1) ворітна формула опише пропаговану частину з похибкою ≤25% на шарі 31
      (поправки придушені φ(t), але t тут не всюди великий).          шанс 60%
 (Г2) породжена частина T_g має ВИЩИЙ PR, ніж T_a (вона не ранг-1),
      але її ЕНЕРГІЯ на глибині мала проти T_a.                        шанс 65%
 (Г3) PR(T_a) НЕ росте з глибиною (підтвердить вимір сусідньої сесії). шанс 75%
 (Г4) Найризикованіше: якщо T_g на глибині ПОРІВНЯННА з T_a, то ранг
      росте щошару і схема мертва. Даю цьому                           25%
"""
import sys
import numpy as np
from scipy.special import erfc

SQ2PI = np.sqrt(2 * np.pi)
W_PATH = "/mnt/win_d/llm/White-Box/arena60/fs_w.npy"
LAYERS = [1, 4, 8, 16, 24, 31]
NSUB = 64


def Phi(x):
    return 0.5 * erfc(-np.asarray(x, float) / np.sqrt(2))


class K3Acc:
    """Онлайн-накопичення 1-го, 2-го і 3-го сирих моментів для NSUB нейронів."""

    def __init__(self, w):
        self.w = w
        self.n = 0
        self.m1 = np.zeros(w)
        self.m2 = np.zeros((w, w))
        self.m3 = np.zeros((w * w, w))

    def add(self, A):
        A = np.ascontiguousarray(A, dtype=np.float64)
        b = A.shape[0]
        self.n += b
        self.m1 += A.sum(0)
        self.m2 += A.T @ A
        B = (A[:, :, None] * A[:, None, :]).reshape(b, self.w * self.w)
        self.m3 += B.T @ A

    def cum3(self):
        """Центральний третій момент = третій кумулянт."""
        w, n = self.w, self.n
        mu = self.m1 / n
        M2 = self.m2 / n
        M3 = (self.m3 / n).reshape(w, w, w)
        T = (M3
             - mu[:, None, None] * M2[None, :, :]
             - mu[None, :, None] * M2[:, None, :]
             - mu[None, None, :] * M2[:, :, None]
             + 2 * mu[:, None, None] * mu[None, :, None] * mu[None, None, :])
        return mu, M2 - np.outer(mu, mu), T


def pr_and_top(T):
    """PR і частка топ-1 за сингулярними числами mode-1 розгортки."""
    w = T.shape[0]
    s = np.linalg.svd(T.reshape(w, w * w), compute_uv=False)
    e = s ** 2
    if e.sum() <= 0:
        return np.nan, np.nan
    return float(e.sum() ** 2 / (e ** 2).sum()), float(e[0] / e.sum())


def rank1_residual(T, iters=200):
    """Найкраще симетричне ранг-1 наближення (степенева ітерація) -> залишок."""
    w = T.shape[0]
    v = np.linalg.svd(T.reshape(w, w * w), full_matrices=False)[0][:, 0]
    for _ in range(iters):
        u = np.einsum("ijk,j,k->i", T, v, v)
        nu = np.linalg.norm(u)
        if nu == 0:
            return 1.0, v
        v = u / nu
    lam = np.einsum("ijk,i,j,k->", T, v, v, v)
    R = T - lam * np.einsum("i,j,k->ijk", v, v, v)
    return float(np.linalg.norm(R) / np.linalg.norm(T)), v


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    B = 5000
    print("=" * 108)
    print("k3gate97 — ворітна формула і ранг κ₃. N=%d, %d мереж, %d нейронів у зрізі"
          % (N, NETS, NSUB))
    print("=" * 108)
    Wall = np.load(W_PATH, mmap_mode="r")

    for net in range(NETS):
        W = np.array(Wall[net]).astype(np.float32)
        rs = np.sort(np.random.default_rng(11).permutation(256)[:NSUB])
        accH = {l: K3Acc(NSUB) for l in LAYERS}      # передактивації h
        accA = {l: K3Acc(NSUB) for l in LAYERS}      # ReLU(h)
        accHalf = {l: K3Acc(NSUB) for l in LAYERS}   # половина вибірки (пломба шуму)
        rng = np.random.default_rng(3000 + net)
        for it in range(N // B):
            a = rng.standard_normal((B, 256), dtype=np.float32)
            for l in range(max(LAYERS) + 1):
                h = a @ W[l]
                if l in accH:
                    hs = h[:, rs]
                    accH[l].add(hs)
                    accA[l].add(np.maximum(hs, 0.0))
                    if it < (N // B) // 2:
                        accHalf[l].add(np.maximum(hs, 0.0))
                a = np.maximum(h, 0.0)

        print("\n" + "-" * 108)
        print("МЕРЕЖА %d" % net)
        print("  шар | мед|t| |  PR(T_a)  топ1  | PR(T_g)  топ1 | ранг1-залишок"
              "  |  ‖T_g‖/‖T_a‖ | ворітна похибка | шум")
        for l in LAYERS:
            muh, Ch, Th = accH[l].cum3()
            mua, Ca, Ta = accA[l].cum3()
            s = np.sqrt(np.diag(Ch))
            t = muh / s

            # ---- гаусів сурогат із ТИМИ САМИМИ (m, C): це рівно ПОРОДЖЕНА частина
            rg = np.random.default_rng(77000 + net * 10 + l)
            ev, U = np.linalg.eigh(Ch)
            L = U * np.sqrt(np.clip(ev, 0, None))
            accG = K3Acc(NSUB)
            for _ in range(N // B):
                g = muh + rg.standard_normal((B, NSUB)) @ L.T
                accG.add(np.maximum(g, 0.0))
            _, _, Tg = accG.cum3()

            prA, topA = pr_and_top(Ta)
            prG, topG = pr_and_top(Tg)
            res1, _ = rank1_residual(Ta)
            gate = Phi(t)
            P = Th * gate[:, None, None] * gate[None, :, None] * gate[None, None, :]
            Aprop = Ta - Tg
            gerr = np.linalg.norm(Aprop - P) / np.linalg.norm(Aprop)
            _, _, Thalf = accHalf[l].cum3()
            noise = np.linalg.norm(Thalf - Ta) / np.linalg.norm(Ta)
            print("  %4d | %6.3f | %7.2f %6.3f | %6.2f %6.3f |    %8.4f      |"
                  "   %8.4f   |    %8.4f     | %.3f"
                  % (l, np.median(np.abs(t)), prA, topA, prG, topG, res1,
                     np.linalg.norm(Tg) / np.linalg.norm(Ta), gerr, noise))

        # ---- деталі на шарі 31 ----
        l = 31
        muh, Ch, Th = accH[l].cum3()
        mua, Ca, Ta = accA[l].cum3()
        s = np.sqrt(np.diag(Ch))
        t = muh / s
        rg = np.random.default_rng(77000 + net * 10 + l)
        ev, U = np.linalg.eigh(Ch)
        L = U * np.sqrt(np.clip(ev, 0, None))
        accG = K3Acc(NSUB)
        for _ in range(N // B):
            g = muh + rg.standard_normal((B, NSUB)) @ L.T
            accG.add(np.maximum(g, 0.0))
        _, _, Tg = accG.cum3()
        prH, topH = pr_and_top(Th)
        resH, vH = rank1_residual(Th)
        resA, vA = rank1_residual(Ta)
        resP, vP = rank1_residual(Ta - Tg)
        print("\n  [ШАР 31, деталі]")
        print("    κ₃(h)      PR=%.2f  топ1=%.4f  ранг1-залишок=%.4f" % (prH, topH, resH))
        print("    κ₃(ReLU)   PR=%.2f  топ1=%.4f  ранг1-залишок=%.4f" % (prA, topA, resA))
        print("    пропагована (T_a−T_g): ранг1-залишок=%.4f" % resP)
        print("    кут між провідними напрямками κ₃(h) і κ₃(ReLU): cos=%.4f"
              % abs(float(vH @ vA)))
        print("    ворітна формула: ‖T_a−T_g−P‖/‖T_a−T_g‖ = %.4f"
              % (np.linalg.norm(Ta - Tg - Th * Phi(t)[:, None, None]
                                * Phi(t)[None, :, None] * Phi(t)[None, None, :])
                 / np.linalg.norm(Ta - Tg)))
        # контроль «чужі ворота»: перемішані Φ мусять дати ГІРШЕ
        gsh = Phi(t)[np.random.default_rng(5).permutation(NSUB)]
        Psh = Th * gsh[:, None, None] * gsh[None, :, None] * gsh[None, None, :]
        print("    КОНТРОЛЬ «чужі ворота» (переставлені Φ): %.4f  <- мусить бути ГІРШЕ"
              % (np.linalg.norm(Ta - Tg - Psh) / np.linalg.norm(Ta - Tg)))
        print("    КОНТРОЛЬ «без воріт» (P = T_h):          %.4f"
              % (np.linalg.norm(Ta - Tg - Th) / np.linalg.norm(Ta - Tg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
