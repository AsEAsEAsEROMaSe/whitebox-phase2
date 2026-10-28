#!/usr/bin/env python3
"""БЛОЧНИЙ ЗАКОН: КЛАСТЕР + ЗАКОН ГЕНЕРАТОРА ЗАМІСТЬ ТОЧНОЇ C (С113).

Ідея ЮЗЕРА: «блискавка бере кластер і апроксимує його по загальному, а загальне —
це рівно те, як випадковий генератор роздав ваги; беручи групами, вони гасять один
одного, і законом усього стають ваги».

Підстава, що це не фантазія: 🔒С107 — збурити ВСІ ваги на 1% ⟹ відповідь міняється
на 0.018% ⟹ відповідь є функцією АНСАМБЛЯ, не окремих чисел.

════════ СХЕМА ════════
Розбиваємо нейрони на K груп. Тримаємо лише МІЖГРУПОВІ середні `c[g,h]` (K² чисел)
плюс ТОЧНУ діагональ. Транспорт тоді не потребує жодного n³:
    C_blk = G·c·Gᵀ  (ранг ≤ K)  ⟹  Wᵀ C_blk W = (WᵀG)·c·(GᵀW)
    вартість n² + K²·n  проти  2.42·n³
    K=32:  0.02% підлоги проти 18.9%   ⟹ у ~900 разів дешевше

════════ ГРУПУВАННЯ ════════
  rnd   випадкове                          (чи працює саме усереднення)
  byt   за |t| = |m|/s                     (нейрони в однаковому режимі разом)
  bysd  за знаком головного власного вектора C

🔒 ПЛОМБИ НА ОБОХ КІНЦЯХ ОСІ
  K=1     одне число ⟹ мусить дати ≈ r=0 драбини (0.675%)
  K=n     кожен сам собі група ⟹ c = C точно ⟹ мусить дати ПОВНУ C (≈0.084%)
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
    ap.add_argument("--groups", type=str, default="1,4,8,32,128,256,1024")
    a = ap.parse_args()
    import cupy as cp
    print(f"cupy {cp.__version__} · "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)
    n, D, N, CH = a.width, a.depth, a.nsamp, a.chunk
    Ks = [int(x) for x in a.groups.split(",")]

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
        mf = m.astype(cp.float32); dC = cp.diag(C).astype(cp.float32)
        sdv = cp.sqrt(cp.maximum(cp.diag(C), 1e-30))
        tv = cp.abs(m / sdv)
        lam, Ve = cp.linalg.eigh(C)
        top = Ve[:, -1]
        print(f"[{net+1}/{a.nets}] seed {seed}  🔒 підлога {100*floor:.4f}%", flush=True)

        orders = {"rnd": cp.asarray(np.random.default_rng(seed).permutation(n)),
                  "byt": cp.argsort(tv),
                  "bysd": cp.argsort(top)}
        srng = cp.random.default_rng(seed ^ 0xB0B)
        res = {}
        for oname, order in orders.items():
            row = {}
            for K in Ks:
                gsz = n // K
                lab = cp.zeros(n, cp.int32)
                lab[order] = cp.arange(n, dtype=cp.int32) // gsz
                lab = cp.minimum(lab, K - 1)
                G = cp.zeros((n, K), cp.float32)
                G[cp.arange(n), lab] = 1.0
                cnts = G.sum(0)
                Coff = C.astype(cp.float32) - cp.diag(dC)
                cb = (G.T @ Coff @ G) / cp.maximum(cp.outer(cnts, cnts), 1.0)
                cb = 0.5 * (cb + cb.T)
                ev, Vc = cp.linalg.eigh(cb.astype(cp.float64))
                keep = ev > 1e-12
                U = (G @ (Vc[:, keep] * cp.sqrt(ev[keep])).astype(cp.float32))
                low = (U * U).sum(1)
                rs = cp.sqrt(cp.maximum(dC - low, 0.0))
                acc = cp.zeros(n, cp.float64); done = 0
                while done < N:
                    b = min(CH, N - done)
                    pp = mf + srng.standard_normal((b, n), dtype=cp.float32) * rs
                    if U.shape[1] > 0:
                        zk = srng.standard_normal((b, U.shape[1]), dtype=cp.float32)
                        pp = pp + zk @ U.T
                    o = cp.maximum(cp.maximum(pp, 0.0) @ Wl, 0.0)
                    acc += o.sum(0, dtype=cp.float64); done += b
                row[K] = cos_err(acc / N, m_true)
            res[oname] = row
            print("   " + oname + "  " + "  ".join(
                f"K{K}:{100*row[K]:.4f}%" for K in Ks), flush=True)
        out[seed] = res

    print("\n════ БЛОЧНИЙ ЗАКОН (медіана) ════")
    print(f"{'K':>6}" + "".join(f"{o:>12}" for o in ("rnd", "byt", "bysd")))
    for K in Ks:
        line = f"{K:>6}"
        for o in ("rnd", "byt", "bysd"):
            v = [x[o][K] for x in out.values()]
            line += f"{100*float(np.median(v)):>11.4f}%"
        print(line)
    print("📏 для порівняння: r=0 драбини 0.675% · повна C ≈0.084% · "
          "eig r=32 0.214%")


if __name__ == "__main__":
    main()
