"""pairtrunc97 — ГОЛОВНЕ ЧИСЛО: скільки з похибки позадіагоналі здатен ЗНЯТИ
ІДЕАЛЬНИЙ кумулянтний коректор порядку K, на СПРАВЖНІЙ мережі.

Питання ВИМІРУ було про ЗБІЖНІСТЬ. Але збіжність цікава лише як засіб: реальне
рішення — «ставити 4-кумулянтну поправку в позадіагональ чи ні». Тому тут
міряється ПРЯМО стеля класу, з ОРАКУЛЬНИМИ кумулянтами (взятими з тієї самої
вибірки, тобто НЕДОСЯЖНО добрими):

  dCov_істина[j,k] = Cov_emp(ReLU_j, ReLU_k) - Cov_Мелер(rho_jk)     <- уся негаусовість
  dCov_ord(K)      = sum_{m,n>=1, m+n<=K} (c_mn - c^G_mn)/(m!n!) a_m a_n
  залишок(K)       = RMS(dCov_істина - dCov_ord(K)) / RMS(dCov_істина)

  знято(K) = 1 - залишок(K)   <- СТЕЛЯ класу «кумулянти до порядку K»

Плюс перевірка механізму, який провалився на першому вимірі:
  ЧИ Є придушення phi(t)? RMS по ПАРАХ показав abs/phi = 122 у мережі 1, але
  RMS тягнуть пари з МАЛИМ |t| (там phi велике). Розкладаємо похибку за
  min(|t_j|,|t_k|) — якщо в кожній смузі abs ~ phi(min|t|), закон ЖИВИЙ, а
  провалився не він, а АГРЕГАТ. (Протокол п.5: спершу еталон, тоді прилад.)

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (У1) У смугах за min|t| абсолютна похибка ПАДАТИМЕ як phi(min|t|) —
      тобто «провал phi(t)» був артефактом агрегування.            шанс 70%
 (У2) знято(4) буде 10-35% ⟹ ідеальний 4-кумулянтний коректор дає
      максимум x1.1-x1.5 на позадіагоналі.                          шанс 65%
 (У3) знято(6) НЕ буде помітно кращим за знято(4) на реальній мережі
      (на Gamma ряд узагалі розбігався).                            шанс 55%
"""
import sys
import numpy as np
from scipy.special import erfc

SQ2PI = np.sqrt(2 * np.pi)
W_PATH = "/mnt/win_d/llm/White-Box/arena60/fs_w.npy"
LAYER = 31
NSUB = 48          # скільки нейронів беремо (усі пари між ними)
MMAX = 6
NMAX = 45


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
    A[0] = s * (phi(t) + t * Phi(t))
    A[1] = s * Phi(t)
    for n in range(2, nmax + 1):
        A[n] = s * phi(t) * ((-1) ** n) * H[n - 2]
    return A


def cov_gauss_pairs(A, rho, nmax=NMAX):
    out = np.zeros_like(rho)
    rp = np.array(rho)
    lf = 0.0
    for n in range(1, nmax + 1):
        lf += np.log(n)
        out += np.exp(-lf) * rp * np.outer(A[n], A[n])
        rp = rp * rho
    return out


def run_net(W, N, seed, batch=20000):
    rng = np.random.default_rng(seed)
    w = 256
    n_, h1, H2, r1, R2 = 0, np.zeros(w), np.zeros((w, w)), np.zeros(w), np.zeros((w, w))
    for _ in range(N // batch):
        a = rng.standard_normal((batch, w), dtype=np.float32)
        for l in range(LAYER + 1):
            h = a @ W[l]
            a = np.maximum(h, 0.0)
        h = h.astype(np.float64)
        r = np.maximum(h, 0.0)
        n_ += batch
        h1 += h.sum(0); H2 += h.T @ h
        r1 += r.sum(0); R2 += r.T @ r
    m = h1 / n_
    C = H2 / n_ - np.outer(m, m)
    s = np.sqrt(np.diag(C))
    rho = np.clip(C / np.outer(s, s), -0.999, 0.999)
    mr = r1 / n_
    Ce = R2 / n_ - np.outer(mr, mr)
    return m, s, rho, Ce


def lam_sub(W, N, seed, idx, m, s, batch=20000):
    """Другий прохід: lam_mn = E[He_m(X_j) He_n(X_k)] для обраних нейронів."""
    rng = np.random.default_rng(seed)      # ТОЙ САМИЙ сід => та сама вибірка
    k = len(idx)
    lam = np.zeros((MMAX + 1, MMAX + 1, k, k))
    n_ = 0
    for _ in range(N // batch):
        a = rng.standard_normal((batch, 256), dtype=np.float32)
        for l in range(LAYER + 1):
            h = a @ W[l]
            a = np.maximum(h, 0.0)
        X = (h[:, idx].astype(np.float64) - m[idx]) / s[idx]
        H = he_list(X, MMAX)
        for mm in range(MMAX + 1):
            for nn in range(MMAX + 1):
                lam[mm, nn] += H[mm].T @ H[nn]
        n_ += batch
    return lam / n_


def dcov_ord(lam, order, A, rho, idx):
    k = len(idx)
    out = np.zeros((k, k))
    f = [1.0]
    for i in range(1, MMAX + 1):
        f.append(f[-1] * i)
    for mm in range(1, MMAX + 1):
        for nn in range(1, MMAX + 1):
            if mm + nn > order:
                continue
            c = lam[mm, nn] - np.outer(lam[mm, 0].diagonal(), lam[0, nn].diagonal())
            cg = f[nn] * rho ** nn if mm == nn else 0.0
            out += (c - cg) / (f[mm] * f[nn]) * np.outer(A[mm][idx], A[nn][idx])
    return out


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print("=" * 100)
    print("pairtrunc97 — СТЕЛЯ кумулянтного класу на позадіагоналі. N=%d, %d мереж, шар %d"
          % (N, NETS, LAYER))
    print("=" * 100)
    Wall = np.load(W_PATH, mmap_mode="r")
    for net in range(NETS):
        W = np.array(Wall[net]).astype(np.float32)
        seed = 4242 + net
        m, s, rho, Ce = run_net(W, N, seed)
        t = m / s
        A = a_table(s, t)
        Cg = cov_gauss_pairs(A, rho)
        D = Ce - Cg
        off = ~np.eye(256, dtype=bool)
        print("\n" + "-" * 100)
        print("МЕРЕЖА %d   мед|t|=%.3f  RMS t=%.3f  RMS rel.поз=%.4f%%"
              % (net, np.median(np.abs(t)), np.sqrt((t ** 2).mean()),
                 100 * np.sqrt((D[off] ** 2).mean()) / np.sqrt((Ce[off] ** 2).mean())))

        # ---------- (У1) чи є придушення phi(t) всередині смуг за min|t| ----------
        print("\n  [У1] РОЗКЛАД ЗА min(|t_j|,|t_k|) — чи живий закон phi(t)?")
        tmin = np.minimum(np.abs(t)[:, None], np.abs(t)[None, :])
        Dn = D / np.outer(s, s)
        print("      смуга min|t|      N пар    RMS abs/(s s)     phi(сер.min|t|)   відношення")
        edges = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 99]
        for i in range(len(edges) - 1):
            msk = off & (tmin >= edges[i]) & (tmin < edges[i + 1])
            if msk.sum() < 50:
                continue
            ra = np.sqrt((Dn[msk] ** 2).mean())
            tc = tmin[msk].mean()
            print("      %4.1f - %4.1f  %9d      %11.3e      %11.3e     %8.2f"
                  % (edges[i], edges[i + 1], msk.sum(), ra, phi(tc), ra / phi(tc)))

        # ---------- СТЕЛЯ КЛАСУ ----------
        rs = np.random.default_rng(7).permutation(256)[:NSUB]
        rs.sort()
        lam = lam_sub(W, N, seed, rs, m, s)
        sub = np.ix_(rs, rs)
        Dsub = D[sub] / np.outer(s[rs], s[rs])
        offs = ~np.eye(NSUB, dtype=bool)
        base = np.sqrt((Dsub[offs] ** 2).mean())
        print("\n  [СТЕЛЯ] ідеальний кумулянтний коректор порядку K (кумулянти З ІСТИНИ):")
        print("      K   залишок RMS / базова   ЗНЯТО      множник на позадіагоналі")
        for K in (2, 3, 4, 5, 6):
            P = dcov_ord(lam, K, A, rho[sub], rs) / np.outer(s[rs], s[rs])
            res = np.sqrt(((Dsub - P)[offs] ** 2).mean()) / base
            print("     %2d        %8.4f          %6.1f%%        x%.2f"
                  % (K, res, 100 * (1 - res), 1 / res if res > 0 else np.inf))
        print("      (base RMS abs/(s s) = %.3e на %d парах)" % (base, offs.sum()))

        # ---------- КРОС-ФІТ: кумулянти з НЕЗАЛЕЖНОЇ вибірки ----------
        # Закон проєкту: «СІДИ ІСТИНИ Й ОРАКУЛА — РОЗДІЛЯТИ» + «не фітувати
        # похибку методу його ж виходами». Без цього стеля може бути in-sample.
        seed2 = 90000 + net
        m2, s2, rho2, Ce2 = run_net(W, N, seed2)
        lam2 = lam_sub(W, N, seed2, rs, m2, s2)
        print("\n  [КРОС-ФІТ] кумулянти з НЕЗАЛЕЖНОЇ вибірки (сід %d) -> на істину сіду %d:"
              % (seed2, seed))
        print("      K   залишок RMS / базова   ЗНЯТО      множник")
        for K in (3, 4, 6):
            P = dcov_ord(lam2, K, A, rho[sub], rs) / np.outer(s[rs], s[rs])
            res = np.sqrt(((Dsub - P)[offs] ** 2).mean()) / base
            print("     %2d        %8.4f          %6.1f%%        x%.2f"
                  % (K, res, 100 * (1 - res), 1 / res if res > 0 else np.inf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
