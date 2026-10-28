"""trunc97 — ЧИ ЗБІГАЄТЬСЯ КУМУЛЯНТНЕ ОБРІЗАННЯ ДЛЯ ПАРИ? (порядки 2..8)

pairchk97 показала: обрізання на 4-му порядку ловить лише 8-13% справжнього
dCov при t=3.5, а для ДОДАТНОЇ густини (Gamma) на t>=4.2 порядок 6 дає
ВІД'ЄМНЕ значення. Тут дивимось повну послідовність часткових сум.

ПЕРЕДБАЧЕННЯ (протокол п.6):
 (Т1) Еджвортова родина (скінченний ермітів розклад маргіналі) — часткові суми
      СХОДЯТЬСЯ до 1.0 монотонно з ~6-го порядку.                     шанс 75%
 (Т2) Gamma — РОЗБІГАЄТЬСЯ, і не через t, а через ХВІСТ густини: Gamma спадає
      як exp(-sqrt(k)*x), тобто ПОВІЛЬНІШЕ за exp(-x^2/4) ⟹ умова Крамера
      для збіжності ермітового ряду ПОРУШЕНА. Це властивість РОДИНИ, а не
      робочої точки.                                                   шанс 70%
 (Т3) Якщо (Т2) вірна, то «t*» — неправильно поставлене питання: межа існує,
      але вона про ХВОСТИ густини, а не про t.                         шанс 65%
"""
import numpy as np
import pairchk97 as P

ORDERS = [2, 3, 4, 5, 6, 7, 8]
TS = [2.0, 3.0, 3.5, 4.2, 5.0]
RHOS = [0.05, 0.30, 0.90]
M = 8


def lam_tensor_fast(fam, rho, M=M, npan=32):
    r = np.sqrt(1 - rho * rho)
    x, wx = P.grid1(fam.lo, fam.hi, npan)
    z, wz = P.grid1(fam.lo, fam.hi, npan)
    X, Z = np.meshgrid(x, z, indexing="ij")
    W = (wx[:, None] * wz[None, :]) * fam.pdf(X) * fam.pdf(Z)
    Y = rho * X + r * Z
    HX = [P.He(m, X) for m in range(M + 1)]
    HY = [P.He(n, Y) for n in range(M + 1)]
    return np.array([[float((W * HX[m] * HY[n]).sum()) for n in range(M + 1)]
                     for m in range(M + 1)])


def main():
    famE = P.EdgeworthFam(0.40, 0.240)
    famG = P.GammaFam(25.0)
    famN = P.GaussFam()
    print("=" * 104)
    print("trunc97 — часткові суми кумулянтного розкладу dCov, порядки 2..8")
    print("значення = (обрізане dCov) / (справжнє dCov);  1.0 = точно")
    print("=" * 104)

    # ПЛОМБА: lam для ГАУСА мусить дати lam_mn = delta_mn * n! rho^n
    print("\n[T0] ПЛОМБА тензора lam на ГАУСІ (мусить бути delta_mn * n! rho^n):")
    worst = 0.0
    for rho in RHOS:
        lam = lam_tensor_fast(famN, rho)
        f = [1.0]
        for i in range(1, M + 1):
            f.append(f[-1] * i)
        ref = np.zeros_like(lam)
        for n in range(M + 1):
            ref[n, n] = f[n] * rho ** n
        worst = max(worst, np.max(np.abs(lam - ref)))
    print("     max |lam - еталон| = %.3e   %s" % (worst, "PASS" if worst < 1e-8 else "FAIL"))

    for fam in (famE, famG):
        print("\n  Родина (%s) %s" % (fam.tag, fam.name))
        print("     t     rho  | dCov_справж |" + "".join("  ord<=%d " % o for o in ORDERS)
              + "| вирок")
        for rho in RHOS:
            lam = lam_tensor_fast(fam, rho)
            for t in TS:
                ct = P.cov_true(fam, rho, 1.0, t, 1.0, t)[0]
                cg = P.cov_true(famN, rho, 1.0, t, 1.0, t)[0]
                dt = ct - cg
                vals = [P.dcov_trunc(lam, o, 1.0, t, 1.0, t, rho) / dt for o in ORDERS]
                # діагноз: чи наближаються часткові суми до 1?
                err = [abs(v - 1) for v in vals]
                diag = "СХОДИТЬСЯ" if err[-1] < err[-3] and err[-1] < 0.35 else "РОЗБІГАЄТЬСЯ"
                print("    %5.2f %5.2f | %+.4e |" % (t, rho, dt)
                      + "".join("%9.3f" % v for v in vals) + " | " + diag)

    print("\n" + "=" * 104)
    print("[T2] ЧОМУ: перевірка умови Крамера — хвіст густини проти exp(-x^2/4)")
    print("=" * 104)
    for fam in (famE, famG):
        print("  (%s) x:" % fam.tag, end="")
        for x in (4.0, 6.0, 8.0, 10.0):
            v = float(fam.pdf(np.array([x]))[0])
            print("  %4.1f: p=%.3e p/exp(-x^2/4)=%.3e" % (x, v, v / np.exp(-x * x / 4)), end="")
        print()
    print("  ⟹ якщо p/exp(-x^2/4) РОСТЕ з x — умова Крамера порушена, ермітів ряд")
    print("     не зобов'язаний збігатись НЕЗАЛЕЖНО від t.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
