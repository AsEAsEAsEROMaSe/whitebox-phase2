#!/usr/bin/env python3
"""ОРАКУЛЬНА СТЕЛЯ ПІДПРОСТОРУ ВИЩИХ МОМЕНТІВ (С113) — ідея ЮЗЕРА.

Питання: якщо нести не лише (m,C), а ще й вищі моменти в НЕВЕЛИКОМУ підпросторі —
скільки це дає? Бюджет це дозволяє: 4-й момент у підпросторі рангу 16 = 2730 чисел
= 1.95% підлоги, тоді як повний 3-й момент = 16000% (неможливий).

════════ ПРИЛАД (оракул, тому це СТЕЛЯ, а не метод) ════════
Стан на передостанньому шарі замінюємо гібридом:
  * у топ-`r` власних напрямках `C` беремо СПРАВЖНІ координати (усі їхні моменти
    точні — не лише 4-й);
  * ортогональне доповнення — гаусове з тими самими (m, C).
⟹ E(r) є ВЕРХНЬОЮ межею всього, що можна витягти з підпростору рангу r.

🔒 ПЛОМБИ НА ОБОХ КІНЦЯХ ОСІ (вимога С105):
  r=0     ⟹ чистий гаусів стан — мусить відтворити зміряні 0.0800%
  r=1024  ⟹ увесь стан справжній — мусить дати РІВНО нуль (тавтологія)
  + підлога шуму з двох половин вибірки
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=2)
    ap.add_argument("--nsamp", type=int, default=524288)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ranks", type=str, default="0,8,16,32,64,128,256,1024")
    a = ap.parse_args()
    import cupy as cp
    print(f"cupy {cp.__version__} · {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}",
          flush=True)
    n, D, N, CH = a.width, a.depth, a.nsamp, a.chunk
    ranks = [int(x) for x in a.ranks.split(",")]

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1 - c * c)))

    out_all = {}
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

        # ── прохід 1: моменти передостаннього шару + істина виходу ──
        g = cp.random.default_rng(seed ^ 0x1AD)
        sm = cp.zeros(n, cp.float64); sC = cp.zeros((n, n), cp.float64)
        mt = cp.zeros(n, cp.float64)
        hh = [cp.zeros(n, cp.float64), cp.zeros(n, cp.float64)]; cnt = [0, 0]
        done = k = 0
        while done < N:
            b = min(CH, N - done)
            p = real_pre(b, g)
            sm += p.sum(0, dtype=cp.float64)
            sC += p.T.astype(cp.float64) @ p.astype(cp.float64)
            o = cp.maximum(cp.maximum(p, 0.0) @ Wl, 0.0)
            mt += o.sum(0, dtype=cp.float64)
            hh[k % 2] += o.sum(0, dtype=cp.float64); cnt[k % 2] += b
            done += b; k += 1
        m = sm / N; C = sC / N - cp.outer(m, m); C = 0.5 * (C + C.T)
        m_true = mt / N
        floor = cos_err(hh[0] / cnt[0], hh[1] / cnt[1]) / 2.0
        lam, V = cp.linalg.eigh(C)
        lam = cp.maximum(lam[::-1], 0.0).copy(); V = V[:, ::-1].copy()
        Vf = V.astype(cp.float32); sq = cp.sqrt(lam).astype(cp.float32)
        mf = m.astype(cp.float32)

        print(f"[{net+1}/{a.nets}] seed {seed}  🔒 підлога шуму {100*floor:.4f}%",
              flush=True)
        res = {}
        for r in ranks:
            g = cp.random.default_rng(seed ^ 0x1AD)      # ті самі справжні семпли
            gg = cp.random.default_rng(seed ^ 0xB0B)
            acc = cp.zeros(n, cp.float64); done = 0
            Vr = Vf[:, :r] if r > 0 else None
            while done < N:
                b = min(CH, N - done)
                p = real_pre(b, g)
                if r >= n:
                    pp = p
                else:
                    u = gg.standard_normal((b, n), dtype=cp.float32)
                    gau = (u * sq) @ Vf.T                # ~N(0,C)
                    if r > 0:
                        pc = p - mf
                        coef_r = pc @ Vr                 # справжні координати
                        coef_g = gau @ Vr
                        pp = mf + gau + (coef_r - coef_g) @ Vr.T
                    else:
                        pp = mf + gau
                o = cp.maximum(cp.maximum(pp, 0.0) @ Wl, 0.0)
                acc += o.sum(0, dtype=cp.float64); done += b
            e = cos_err(acc / N, m_true)
            res[r] = e
            fv = float(lam[:r].sum() / lam.sum()) if r else 0.0
            print(f"   r={r:>5}  похибка {100*e:8.4f}%   дисперсії в підпросторі "
                  f"{100*fv:5.1f}%", flush=True)
        out_all[seed] = res

    print("\n════ СТЕЛЯ ПІДПРОСТОРУ (медіана) ════")
    print(f"{'r':>6}{'похибка':>11}{'×проти гауса':>15}")
    base = np.median([v[0] for v in out_all.values()])
    for r in ranks:
        v = [x[r] for x in out_all.values() if r in x]
        if v:
            mm = float(np.median(v))
            print(f"{r:>6}{100*mm:>10.4f}%{(base/mm if mm>0 else float('inf')):>14.2f}")


if __name__ == "__main__":
    main()
