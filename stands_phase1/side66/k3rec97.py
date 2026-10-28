"""k3rec97 — ЧИ ПЕРЕЖИВЕ ВОРІТНА ФОРМУЛА 31 КРОК РЕКУРСІЇ?

Усе, що я зміряла досі, — ОДНОСТУПЕНЕВЕ. ВИМІР назвав п'ять вироків, де саме
рекурсія вбивала однокрокові виграші, і розрив зміряно тричі: ×8.5, ×4.88, ×4.53.
Тут ворітна формула ставиться в ПОВНУ 31-крокову рекурсію проти тієї самої
рекурсії без κ₃ — одна таблиця, одні мережі, один жереб істини.

СХЕМА (усі ТРИ канали міняються РАЗОМ — пастка №2 від ВИМІРУ):
  m_a   = σ[φ+tΦ] − (γ₃/6)σφt                        ← κ₃ входить у СЕРЕДНЄ
  C_a   = Σ_n ρⁿ/n! a_n a_n                          ← Мелер (гаусова частина)
          + ½·κ̄₃_{jkk}·a₁(j)a₂(k) + ½·κ̄₃_{jjk}·a₂(j)a₁(k)   ← κ₃ у ДІАГОНАЛЬ і
                                                        ПОЗАДІАГОНАЛЬ одночасно
                                                        (при j=k дає γ₃a₁a₂)
  κ₃_a  = Φ_jΦ_kΦ_l ⊙ κ₃_h            ← ПРОПАГОВАНА (ворітна формула)
          + трилінійний Мелер K=1      ← ПОРОДЖЕНА (замкнена форма)
  лінійний крок: m←mW, C←WᵀCW, κ₃←κ₃×₁W×₂W×₃W    (точний, без наближень)

κ₃ несеться ПОВНИМ тензором 256³ — щоб питання рангу НЕ змішувалось із питанням
рекурсії. Якщо повний ранг не виживе, ранг-r і поготів.

ЧОТИРИ ГІЛКИ В ОДНІЙ ТАБЛИЦІ:
  G        гаусове замикання (κ₃ ≡ 0)                     ← база
  K3       повна схема
  K3_nog   КОНТРОЛЬ: ворота замінено на 1 (без Φ)         ← мусить бути ГІРШЕ
  K3_gen   КОНТРОЛЬ: лише ПОРОДЖЕНА частина, без пропагованої

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (протокол п.6):
 (Р1) K3 буде кращим за G, але НЕ в ×3.2 (це однокрокове число), а в
      ×1.3-×1.8: частина виграшу з'їдається накопиченням.          шанс 45%
 (Р2) K3 буде ГІРШИМ за G (як у п'яти попередніх вироків) —
      бо замикання є збалансованою системою похибок.               шанс 35%
 (Р3) K3 дасть ×2 і більше.                                        шанс 20%
 (Р4) K3_nog буде гіршим за K3 — інакше ворота нічого не роблять
      і весь механізм під сумнівом.                                шанс 85%
"""
import sys
import numpy as np
from math import factorial
from scipy.special import erfc

SQ2PI = np.sqrt(2 * np.pi)
W_PATH = "/mnt/win_d/llm/White-Box/arena60/fs_w.npy"
NMAX = 30
SFLOOR = 1e-12


def phi(x):
    return np.exp(-np.asarray(x, float) ** 2 / 2) / SQ2PI


def Phi(x):
    return 0.5 * erfc(-np.asarray(x, float) / np.sqrt(2))


def he_list(x, mmax):
    out = [np.ones_like(x), x.copy()]
    for k in range(1, mmax):
        out.append(x * out[-1] - k * out[-2])
    return out[:mmax + 1]


def a_table(s, t, nmax=NMAX):
    H = he_list(t, max(nmax - 2, 0))
    A = np.empty((nmax + 1, t.size))
    p, P = phi(t), Phi(t)
    A[0] = s * (p + t * P)
    A[1] = s * P
    for n in range(2, nmax + 1):
        A[n] = s * p * ((-1) ** n) * H[n - 2]
    return A


def mehler(A, rho, nmax=NMAX):
    out = np.zeros_like(rho)
    rp = rho.copy()
    lf = 0.0
    for n in range(1, nmax + 1):
        lf += np.log(n)
        out += np.exp(-lf) * rp * np.outer(A[n], A[n])
        rp = rp * rho
    return out


def mode_mult(T, W):
    """T ×₁W ×₂W ×₃W для симетричного T (256³)."""
    n = T.shape[0]
    X = W.T @ T.reshape(n, n * n)                       # мода 1
    X = X.reshape(n, n, n).transpose(1, 0, 2).reshape(n, n * n)
    X = W.T @ X                                          # мода 2
    X = X.reshape(n, n, n).transpose(2, 1, 0).reshape(n, n * n)
    X = W.T @ X                                          # мода 3
    return X.reshape(n, n, n).transpose(2, 1, 0)


def generated(A, rho):
    """Трилінійний Мелер, K=1: 4 члени (зміряно: K=1 вже дає майже все)."""
    a1, a2 = A[1], A[2]
    T = np.einsum("jk,jl,j,k,l->jkl", rho, rho, a2, a1, a1, optimize=True)
    T += np.einsum("jk,kl,j,k,l->jkl", rho, rho, a1, a2, a1, optimize=True)
    T += np.einsum("jl,kl,j,k,l->jkl", rho, rho, a1, a1, a2, optimize=True)
    T += np.einsum("jk,jl,kl,j,k,l->jkl", rho, rho, rho, a2, a2, a2, optimize=True)
    return T


def relu_step(m, C, T3, mode, lam=1.0):
    """(m,C,κ₃) переднейронних -> (m,C,κ₃) активацій. mode: G / K3 / K3_nog / K3_gen"""
    s = np.sqrt(np.maximum(np.diag(C), SFLOOR))
    t = m / s
    rho = np.clip(C / np.outer(s, s), -1.0, 1.0)
    A = a_table(s, t)
    a1, a2 = A[1], A[2]

    if T3 is None:
        g3 = np.zeros_like(s)
        kjkk = kjjk = None
    else:
        n = len(s)
        i = np.arange(n)
        g3 = T3[i, i, i] / s ** 3
        kjkk = T3[:, i, i] / (s[:, None] * s[None, :] ** 2)        # κ̄₃_{jkk}
        # ВИПРАВЛЕНО: було T3[i,i,:].T — зайве транспонування давало
        # κ̄₃[j,k] = T3[k,k,j]/(s_j² s_k) замість T3[j,j,k]/(s_j² s_k).
        # Розбіжність 54%; спіймано пломбою на безтензорні зрізи.
        kjjk = T3[i, i, :] / (s[:, None] ** 2 * s[None, :])        # κ̄₃_{jjk} = κ̄₃_{jkk}ᵀ

    m_a = A[0] + lam * (g3 / 6.0) * A[3]
    C_a = mehler(A, rho)
    if T3 is not None:
        C_a = C_a + lam * (0.5 * kjkk * np.outer(a1, a2)
                           + 0.5 * kjjk * np.outer(a2, a1))
        C_a = 0.5 * (C_a + C_a.T)

    if mode == "G":
        return m_a, C_a, None
    G = generated(A, rho)
    if mode == "K3_gen":
        return m_a, C_a, G
    if mode == "K3_prop":
        G = 0.0
    gate = Phi(t) if mode == "K3" else np.ones_like(t)
    prop = 0.0 if T3 is None else (T3 * gate[:, None, None]
                                   * gate[None, :, None] * gate[None, None, :])
    return m_a, C_a, prop + G


def run_closure(W, mode, L=32, lam=1.0):
    n = 256
    m = np.zeros(n)
    C = W[0].T @ W[0]
    T3 = None if mode == "G" else np.zeros((n, n, n))
    for l in range(1, L):
        m_a, C_a, T3_a = relu_step(m, C, T3, mode, lam)
        m = m_a @ W[l]
        C = W[l].T @ (C_a @ W[l])
        T3 = None if T3_a is None else mode_mult(T3_a, W[l])
    s = np.sqrt(np.maximum(np.diag(C), SFLOOR))
    t = m / s
    A = a_table(s, t)
    if T3 is None:
        return A[0]
    i = np.arange(n)
    g3 = T3[i, i, i] / s ** 3
    return A[0] + lam * (g3 / 6.0) * A[3]


def truth(W, N, seed, B=20000, L=32):
    rng = np.random.default_rng(seed)
    acc = np.zeros(256)
    accH = np.zeros(256)
    n_ = 0
    for i in range(N // B):
        a = rng.standard_normal((B, 256), dtype=np.float32)
        for l in range(L):
            h = a @ W[l]
            a = np.maximum(h, 0.0)
        acc += a.astype(np.float64).sum(0)
        if i < (N // B) // 2:
            accH += a.astype(np.float64).sum(0)
        n_ += B
    return acc / n_, accH / (n_ // 2)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("=" * 96)
    print("k3rec97 — ПОВНА 31-крокова рекурсія. істина: МК N=%d, %d мереж" % (N, NETS))
    print("=" * 96)
    Wall = np.load(W_PATH, mmap_mode="r")
    MODES = ("G", "K3", "K3_nog", "K3_gen", "K3_prop")
    tab = {k: [] for k in MODES}
    for net in range(NETS):
        W = np.array(Wall[net]).astype(np.float64)
        mt, mh = truth(W, N, 12345 + net)
        den = np.sqrt((mt ** 2).mean())
        noise = np.sqrt(((mh - mt) ** 2).mean()) / den
        print("\nМЕРЕЖА %d   RMS(істина)=%.4f   шум МК=%.4f%%" % (net, den, 100 * noise))
        print("   гілка   | RMS відн.похибка |  проти G")
        base = None
        for mode in MODES:
            p = run_closure(W, mode)
            e = np.sqrt(((p - mt) ** 2).mean()) / den
            tab[mode].append(e)
            if mode == "G":
                base = e
            print("   %-8s |    %8.4f%%     |   x%.3f" % (mode, 100 * e, base / e))
        if mode == "K3_prop" and net == 0:
            print("   [СКАН λ на силу поправки κ₃ у m і C — протокол п.12]")
            for lam in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
                pl = run_closure(W, "K3", lam=lam)
                el = np.sqrt(((pl - mt) ** 2).mean()) / den
                print("      λ=%.2f  %8.4f%%   x%.3f" % (lam, 100 * el, base / el))
    print("\n" + "=" * 96)
    print("ПІДСУМОК по %d мережах (RMS відносної похибки СЕРЕДНЬОГО останнього шару)" % NETS)
    for mode in MODES:
        v = np.array(tab[mode])
        print("   %-8s  сер %8.4f%%   min %8.4f%%   max %8.4f%%   множник до G x%.3f"
              % (mode, 100 * v.mean(), 100 * v.min(), 100 * v.max(),
                 np.array(tab["G"]).mean() / v.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
