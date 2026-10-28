#!/usr/bin/env python3
"""БЛИСКАВКА: КАНАЛ ЗА ПРОВІДНІСТЮ ДО ВИХОДУ, А НЕ ЗА ТОВЩИНОЮ СТАНУ (С113).

Ідея ЮЗЕРА: «проекційна функція яка йде в умовний багатомірний напрям… схоже на
блискавку: пробій там, де вже є шлях, а решту добираємо супутніми розрядами
дешево, спираючись на припущення про конуси».

════════ ЩО ЦЕ ВИПРАВЛЯЄ ════════
`ladder113` обирав напрямки за власними значеннями `C` — «де стан гуляє» — і дав
плато аж при r≈256. Але нам байдуже, де стан гуляє; важливо, що доходить ДО ВИХОДУ.

  канал   = топ-r напрямків, які найдужче б'ють у вихід  → рахуємо ТОЧНО
  конус   = решта простору                              → точна діагональ, нуль кореляцій

════════ ТРИ БАЗИСИ ════════
  eig   топ-r власних векторів C                (як було; «де стан товстий»)
  cond  топ-r лівих сингулярних векторів C·B    («що доходить до виходу»)
  rand  випадкові ортогональні                  🔒 контроль, мусить бути плоским

`B[i,j] = q_prev[i]·W[i,j]·q_out[j]` — провідність нейрона до виходу; `q = P(p>0)`,
і замикання їх УЖЕ має (це його ж `Φ`), тобто ціна нуль.

🔒 ПЛОМБИ: r=0 однакове для всіх базисів · r=n однакове для всіх · rand плоский
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=2)
    ap.add_argument("--nsamp", type=int, default=262144)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ranks", type=str, default="0,4,8,16,32,64,128,256")
    a = ap.parse_args()
    import cupy as cp
    print(f"cupy {cp.__version__} · "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)
    n, D, N, CH = a.width, a.depth, a.nsamp, a.chunk
    ranks = [int(x) for x in a.ranks.split(",")]

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1 - c * c)))

    out = {}
    for net in range(a.nets):
        seed = 6113113 + 1000 * net
        rng = cp.random.default_rng(seed)
        s = float(np.sqrt(2.0 / n))
        Ws = [(rng.standard_normal((n, n), dtype=cp.float32) * s) for _ in range(D)]
        Wl = Ws[-1]

        def real_pre(b, g):
            h = g.standard_normal((b, n), dtype=cp.float32)
            for l in range(D - 1):
                p = h @ Ws[l]; h = cp.maximum(p, 0.0)
            return p

        g = cp.random.default_rng(seed ^ 0x1AD)
        sm = cp.zeros(n, cp.float64); sC = cp.zeros((n, n), cp.float64)
        mt = cp.zeros(n, cp.float64); qp = cp.zeros(n, cp.float64)
        qo = cp.zeros(n, cp.float64)
        smo = cp.zeros(n, cp.float64); sso = cp.zeros(n, cp.float64)
        hh = [cp.zeros(n, cp.float64), cp.zeros(n, cp.float64)]; cnt = [0, 0]
        done = k = 0
        while done < N:
            b = min(CH, N - done)
            p = real_pre(b, g)
            sm += p.sum(0, dtype=cp.float64)
            sC += p.T.astype(cp.float64) @ p.astype(cp.float64)
            qp += (p > 0).sum(0, dtype=cp.float64)
            po = cp.maximum(p, 0.0) @ Wl
            qo += (po > 0).sum(0, dtype=cp.float64)
            smo += po.sum(0, dtype=cp.float64)
            sso += (po.astype(cp.float64) ** 2).sum(0)
            o = cp.maximum(po, 0.0)
            mt += o.sum(0, dtype=cp.float64)
            hh[k % 2] += o.sum(0, dtype=cp.float64); cnt[k % 2] += b
            done += b; k += 1
        m = sm / N; C = sC / N - cp.outer(m, m); C = 0.5 * (C + C.T)
        m_true = mt / N
        floor = cos_err(hh[0] / cnt[0], hh[1] / cnt[1]) / 2.0
        mf = m.astype(cp.float32); dC = cp.diag(C)

        # ── три базиси ──
        lam, Ve = cp.linalg.eigh(C)
        Ve = Ve[:, ::-1].copy()                                     # eig
        Bm = (qp / N)[:, None].astype(cp.float32) * Wl * (qo / N)[None, :].astype(cp.float32)
        Y = C.astype(cp.float32) @ Bm                               # cond
        Uc, Sc, _ = cp.linalg.svd(Y, full_matrices=False)
        # ── sens: вага, ВИВЕДЕНА з ∂E[relu]/∂Var, а не вгадана ──
        mo = smo / N; so = cp.sqrt(cp.maximum(sso / N - mo ** 2, 1e-30))
        to = mo / so
        phi = cp.exp(-0.5 * to ** 2) / np.sqrt(2 * np.pi)
        wq = (qp / N).astype(cp.float32)
        wo = cp.sqrt(cp.maximum(phi / so, 0.0)).astype(cp.float32)
        Ys = (C.astype(cp.float32) * wq[None, :]) @ (Wl * wo[None, :])
        Us, Ss, _ = cp.linalg.svd(Ys, full_matrices=False)
        Vr_ = cp.random.default_rng(seed ^ 0xBAD).standard_normal((n, n), dtype=cp.float64)
        Vrnd, _ = cp.linalg.qr(Vr_)                                  # rand
        bases = {"eig": Ve.astype(cp.float32), "cond": Uc.astype(cp.float32),
                 "sens": Us.astype(cp.float32), "rand": Vrnd.astype(cp.float32)}
        print(f"[{net+1}/{a.nets}] seed {seed}  🔒 підлога {100*floor:.4f}%  ·  "
              f"провідність B: топ-32/усі = "
              f"{float((Sc[:32]**2).sum()/(Sc**2).sum()):.3f}"
              f"  ·  чутливість топ-32 {float((Ss[:32]**2).sum()/(Ss**2).sum()):.3f}"
              f"  ·  мед|t| виходу {float(cp.median(cp.abs(to))):.3f}", flush=True)

        srng = cp.random.default_rng(seed ^ 0xB0B)
        res = {}
        for bname, U in bases.items():
            row = {}
            for r in ranks:
                Ur = U[:, :r] if r > 0 else None
                if r > 0:
                    M = Ur.T @ C.astype(cp.float32) @ Ur          # [r,r]
                    M = 0.5 * (M + M.T) + cp.eye(r, dtype=cp.float32) * 1e-6
                    Lc = cp.linalg.cholesky(M)
                    low = ((Ur @ M) * Ur).sum(1)                  # diag(Ur M Ur^T)
                else:
                    low = cp.zeros(n, cp.float32)
                rs = cp.sqrt(cp.maximum(dC.astype(cp.float32) - low, 0.0))
                acc = cp.zeros(n, cp.float64); done = 0
                while done < N:
                    b = min(CH, N - done)
                    pp = mf + srng.standard_normal((b, n), dtype=cp.float32) * rs
                    if r > 0:
                        zr = srng.standard_normal((b, r), dtype=cp.float32)
                        pp = pp + (zr @ Lc.T) @ Ur.T
                    o = cp.maximum(cp.maximum(pp, 0.0) @ Wl, 0.0)
                    acc += o.sum(0, dtype=cp.float64); done += b
                row[r] = cos_err(acc / N, m_true)
            res[bname] = row
            print("   " + bname + "  " + "  ".join(
                f"r{r}:{100*row[r]:.4f}%" for r in ranks), flush=True)
        out[seed] = res

    print("\n════ БЛИСКАВКА: ЯКИЙ БАЗИС ДЕШЕВШИЙ ════")
    print(f"{'r':>5}" + "".join(f"{b:>12}" for b in ("eig", "cond", "sens", "rand")))
    for r in ranks:
        line = f"{r:>5}"
        for b in ("eig", "cond", "sens", "rand"):
            v = [x[b][r] for x in out.values()]
            line += f"{100*float(np.median(v)):>11.4f}%"
        print(line)


if __name__ == "__main__":
    main()
