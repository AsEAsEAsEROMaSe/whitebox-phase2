"""pairreal97 — СКІЛЬКИ З ПОХИБКИ ПОЗАДІАГОНАЛІ (4.08%) Є НЕГАУСОВІСТЮ ПАРИ?

Це ПРЯМИЙ вимір на СПРАВЖНІЙ мережі, а не на синтетичній парі.

🚨 ЧОМУ ВІН ОБОВ'ЯЗКОВИЙ. У pairchk97 пара будувалась як Y = rho*X + sqrt(1-rho^2)*Z
з незалежними X, Z. Така конструкція ЗМУШУЄ крос-кумулянти бути пропорційними rho
(lam_21 = rho*g3, lam_31 = rho*g4) ⟹ у відношенні dCov/Cov_G множник rho
скорочується, і «незалежність від rho» виходить ЗА ПОБУДОВОЮ, а не з природи.
Справжня пара нейронів так робити не зобов'язана. Це рівно хвороба «НЕ ТОЙ
ОБ'ЄКТ» із протоколу, і єдиний спосіб її виключити — зміряти на мережі.

ОБ'ЄКТ. Для шару l беремо переднейронні активації h = a_{l-1} @ W_l і рахуємо:
  емпіричне  Cov(ReLU(h_j), ReLU(h_k))                        <- ІСТИНА
  гаусове    sum_{n>=1} rho_jk^n/n! * a_n(s_j,t_j) * a_n(s_k,t_k)   <- Мелер
             з ТОЧНИМИ (m_j, s_j, rho_jk), узятими з тієї ж вибірки
Різниця — це РІВНО похибка одного кроку замикання з точними (mu, sigma, rho),
тобто ЧИСТА негаусовість пари. У CLAUDE.md вона записана як 4.08% (позадіагональ).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (протокол п.6):
 (Р1) Відтворю записані 4.08% з точністю ~2x. Якщо вийде ~0.1% — значить
      або запис про інший об'єкт, або мій еталон інший.             шанс 55%
 (Р2) Похибка БУДЕ рости при малих |rho| як ~1/|rho|, чого синтетична пара
      НЕ показала ⟹ саме пари зі слабкою кореляцією і дають 4%.     шанс 70%
 (Р3) АБСОЛЮТНА похибка dCov/(s_j*s_k) буде придушена ~phi(t), тобто ~1e-3,
      і майже не залежатиме від rho — придушення phi(t) з pairchk97
      ПЕРЕЖИВЕ перехід на справжню мережу.                          шанс 75%
 ⟹ (Р2)+(Р3) разом означали б: 4.08% — це НЕ «велика негаусовість», а МАЛА
      абсолютна похибка, поділена на МАЛУ позадіагональ. Тоді 4-кумулянтний
      клас лікує саме те, що й так мале, і стеля виграшу мікроскопічна.

ПЛОМБИ:
 S1  шум МК: половини вибірки (незалежні) — розкид мусить бути << ефекту;
 S2  масштабування N -> N/4: шум падає вдвічі, СПРАВЖНЯ похибка стоїть;
 S3  збіжність ряду Мелера: NMAX=30 vs 45 (ловить обрізання при великих |rho|);
 S4  контроль ГАУСОМ: підставляю замість h справжній гаусів вектор із ТИМИ
     САМИМИ (m, Cov) — тоді похибка мусить впасти до рівня шуму. Якщо НЕ впаде,
     мій еталон хибний, а не мережа негаусова. Це і є «спершу еталон, тоді прилад».
"""
import sys
import numpy as np
from scipy.special import erfc

SQ2PI = np.sqrt(2 * np.pi)
W_PATH = "/mnt/win_d/llm/White-Box/arena60/fs_w.npy"
LAYERS = [1, 4, 8, 16, 24, 31]
NMAX = 45


def phi(x):
    return np.exp(-np.asarray(x, float) ** 2 / 2) / SQ2PI


def Phi(x):
    return 0.5 * erfc(-np.asarray(x, float) / np.sqrt(2))


def he_table(t, mmax):
    """He_m(t) для m = 0..mmax, t — вектор."""
    out = [np.ones_like(t), t.copy()]
    for k in range(1, mmax):
        out.append(t * out[-1] - k * out[-2])
    return out[:mmax + 1]


def a_table(s, t, nmax=NMAX):
    """a_n для кожного нейрона: (nmax+1, n_neur)."""
    H = he_table(t, max(nmax - 2, 0))
    A = np.empty((nmax + 1, t.size))
    A[0] = s * (phi(t) + t * Phi(t))
    A[1] = s * Phi(t)
    for n in range(2, nmax + 1):
        A[n] = s * phi(t) * ((-1) ** n) * H[n - 2]
    return A


def cov_gauss_pairs(A, rho, nmax=NMAX):
    """Мелер: Cov_G[j,k] = sum_{n>=1} rho^n/n! a_n[j] a_n[k]."""
    out = np.zeros_like(rho)
    rp = np.array(rho)
    lf = 0.0
    for n in range(1, nmax + 1):
        lf += np.log(n)
        out += np.exp(-lf) * rp * np.outer(A[n], A[n])
        rp = rp * rho
    return out


def stats_from(H):
    """m, s, rho з вибірки переднейронних активацій H (N, W)."""
    m = H.mean(0)
    Hc = H - m
    C = (Hc.T @ Hc) / H.shape[0]
    s = np.sqrt(np.diag(C))
    rho = C / np.outer(s, s)
    np.clip(rho, -0.999, 0.999, out=rho)
    return m, s, rho


def emp_cov_relu(H):
    A = np.maximum(H, 0.0)
    mu = A.mean(0)
    Ac = A - mu
    return (Ac.T @ Ac) / A.shape[0]


class Acc:
    """Онлайн-накопичувач других моментів (пам'ять O(W^2), не O(N*W))."""

    def __init__(self, w=256):
        self.n = 0
        self.h1 = np.zeros(w)
        self.H2 = np.zeros((w, w))
        self.r1 = np.zeros(w)
        self.R2 = np.zeros((w, w))

    def add(self, h):
        h = h.astype(np.float64)
        r = np.maximum(h, 0.0)
        self.n += h.shape[0]
        self.h1 += h.sum(0)
        self.H2 += h.T @ h
        self.r1 += r.sum(0)
        self.R2 += r.T @ r

    def __iadd__(self, o):
        self.n += o.n
        self.h1 += o.h1
        self.H2 += o.H2
        self.r1 += o.r1
        self.R2 += o.R2
        return self

    def stats(self):
        m = self.h1 / self.n
        C = self.H2 / self.n - np.outer(m, m)
        s = np.sqrt(np.diag(C))
        rho = np.clip(C / np.outer(s, s), -0.999, 0.999)
        mr = self.r1 / self.n
        Ce = self.R2 / self.n - np.outer(mr, mr)
        self.C = C
        return m, s, rho, Ce


def propagate(W, N, seed, layer_max, batch=20000, nq=4):
    """Накопичує статистики по 4 чвертях вибірки (для пломб S1/S2)."""
    rng = np.random.default_rng(seed)
    accs = {l: [Acc() for _ in range(nq)] for l in LAYERS if l <= layer_max}
    per_q = N // nq
    for q in range(nq):
        done = 0
        while done < per_q:
            b = min(batch, per_q - done)
            a = rng.standard_normal((b, 256), dtype=np.float32)
            for l in range(0, layer_max + 1):
                h = a @ W[l]
                if l in accs:
                    accs[l][q].add(h)
                a = np.maximum(h, 0.0)
            done += b
    return accs


def merge(accs, idx):
    out = Acc()
    for i in idx:
        out += accs[i]
    return out


def offdiag_metrics(Ce, Cg, s):
    """RMS-відносна похибка позадіагоналі + абсолютна, знормована на s_j s_k."""
    n = Ce.shape[0]
    off = ~np.eye(n, dtype=bool)
    d = Ce - Cg
    rel = np.sqrt((d[off] ** 2).mean()) / np.sqrt((Ce[off] ** 2).mean())
    absn = np.sqrt(((d / np.outer(s, s))[off] ** 2).mean())
    return rel, absn, d, off


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print("=" * 104)
    print("pairreal97 — негаусовість ПАРИ на справжній мережі. N=%d семплів, %d мереж"
          % (N, NETS))
    print("=" * 104)
    Wall = np.load(W_PATH, mmap_mode="r")
    print("ваги: %s %s" % (Wall.shape, Wall.dtype))

    for net in range(NETS):
        W = np.array(Wall[net]).astype(np.float32)
        print("\n" + "-" * 104)
        print("МЕРЕЖА %d" % net)
        accs = propagate(W, N, seed=1234 + net, layer_max=max(LAYERS))
        print("  шар | мед|t|  RMS t | мед|rho| |  RMS rel.поз | RMS abs/(s s)"
              " | phi(мед|t|) | abs/phi")
        for l in LAYERS:
            m, s, rho, Ce = merge(accs[l], range(4)).stats()
            t = m / s
            A = a_table(s, t)
            Cg = cov_gauss_pairs(A, rho)
            rel, absn, d, off = offdiag_metrics(Ce, Cg, s)
            print("  %4d | %6.3f %6.3f | %10.4f | %11.4f%% | %13.3e | %12.3e | %6.2f"
                  % (l, np.median(np.abs(t)), np.sqrt((t ** 2).mean()),
                     np.median(np.abs(rho[off])), 100 * rel, absn,
                     phi(np.median(np.abs(t))),
                     absn / phi(np.median(np.abs(t)))))

        # ---- деталі на останньому шарі ----
        l = 31
        m, s, rho, Ce = merge(accs[l], range(4)).stats()
        t = m / s
        A = a_table(s, t)
        Cg = cov_gauss_pairs(A, rho)
        rel, absn, d, off = offdiag_metrics(Ce, Cg, s)
        dn = (d / np.outer(s, s))[off]
        rr = np.abs(rho[off])

        print("\n  [Р2] ЧИ РОСТЕ ВІДНОСНА ПОХИБКА ПРИ МАЛИХ |rho|? (шар 31)")
        print("      |rho| смуга      N пар    RMS abs/(s s)   RMS |Cov|/(s s)   rel.похибка")
        edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
        cge = (Cg / np.outer(s, s))[off]
        for i in range(len(edges) - 1):
            msk = (rr >= edges[i]) & (rr < edges[i + 1])
            if msk.sum() < 20:
                continue
            ra = np.sqrt((dn[msk] ** 2).mean())
            rc = np.sqrt((cge[msk] ** 2).mean())
            print("      %.2f - %.2f  %9d   %13.3e   %15.3e   %10.3f%%"
                  % (edges[i], edges[i + 1], msk.sum(), ra, rc, 100 * ra / rc))

        # ---- S1: шум МК по половинах ----
        rels = []
        for idx in ((0, 1), (2, 3)):
            mh, sh, rh, Ceh = merge(accs[l], idx).stats()
            Ah = a_table(sh, mh / sh)
            r_, a_, _, _ = offdiag_metrics(Ceh, cov_gauss_pairs(Ah, rh), sh)
            rels.append((r_, a_))
        print("\n  [S1] ШУМ МК: половини вибірки  rel = %.4f%% / %.4f%%   abs = %.3e / %.3e"
              % (100 * rels[0][0], 100 * rels[1][0], rels[0][1], rels[1][1]))

        # ---- S2: масштабування N/4 ----
        mq, sq, rq, Ceq = merge(accs[l], (0,)).stats()
        Aq = a_table(sq, mq / sq)
        rq_, aq_, _, _ = offdiag_metrics(Ceq, cov_gauss_pairs(Aq, rq), sq)
        print("  [S2] МАСШТАБ N/4: rel = %.4f%% (проти %.4f%% на повній) — "
              "якщо ефект ШУМ, він виріс би вдвічі" % (100 * rq_, 100 * rel))

        # ---- S3: збіжність Мелера ----
        Cg30 = cov_gauss_pairs(A, rho, nmax=30)
        print("  [S3] МЕЛЕР NMAX 30 vs 45: max rel.різниця по позадіагоналі = %.2e"
              % np.max(np.abs((Cg30 - Cg)[off]) / (np.abs(Cg[off]) + 1e-300)))

        # ---- S4: КОНТРОЛЬ ГАУСОМ ----
        rng = np.random.default_rng(999 + net)
        acc31 = merge(accs[l], range(4))
        acc31.stats()
        Cfull = acc31.C
        jit = 1e-9 * np.trace(Cfull) / 256
        L = np.linalg.cholesky(Cfull + jit * np.eye(256))
        accg = Acc()
        done = 0
        while done < N:
            b = min(20000, N - done)
            accg.add(m + rng.standard_normal((b, 256)) @ L.T)
            done += b
        mg, sg, rg, Ceg = accg.stats()
        Ag = a_table(sg, mg / sg)
        rg_, ag_, _, _ = offdiag_metrics(Ceg, cov_gauss_pairs(Ag, rg), sg)
        print("  [S4] КОНТРОЛЬ ГАУСОМ (той самий m, Cov, але СПРАВДІ гаусів вектор):")
        print("       rel = %.4f%%  abs = %.3e   <- мусить впасти до рівня шуму [S1];"
              % (100 * rg_, ag_))
        print("       якщо НЕ впало — хибний ЕТАЛОН, а не негаусова мережа.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
