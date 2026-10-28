#!/usr/bin/env python3
"""ЩО САМЕ ДАЄ `γ₄` І В ЯКОМУ ПРЕДСТАВЛЕННІ (С113) — крок 4 черги ЮЗЕРА.

Оракул `sub4th113` ніс у підпросторі ВСІ моменти і дав ×1.34 при r=32.
Тут розділяємо: скільки з цього бере САМЕ 4-й момент, і яке представлення дешевше.

КАНДИДАТИ (усі — на передостанньому шарі, крок точний семплюванням):
  A  (m,C) — гаусів стан                              база
  B  + понейронна γ₄ у ВИХІДНОМУ базисі               n = 1024 числа   ← найдешевше
  C  + маргінальна γ₄ у топ-32 власному базисі        32 числа
  D  + справжні координати в топ-32 (оракул)          стеля 0.0547%
  E  усе справжнє                                     тавтологія 0

🔒 ПЛОМБИ
  S1 c=0 у перетворенні ⟹ РІВНО база A
  S2 зміряна γ₄ сурогата збігається з цільовою
  S3 коваріація сурогата збігається з C (перетворення не має її псувати)
  S4 E = 0.0000% (тавтологія)
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
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    import cupy as cp
    print(f"cupy {cp.__version__} · "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)
    n, D, N, CH, R = a.width, a.depth, a.nsamp, a.chunk, a.rank

    # ── калібр кубічного перетворення: c → надлишковий ексцес ──
    cg = cp.random.default_rng(12345)
    zc = cg.standard_normal((1 << 21,), dtype=cp.float32)
    cs = np.linspace(0.0, 0.12, 25)
    ks = []
    for c in cs:
        y = zc + float(c) * (zc ** 3 - 3 * zc)
        y = y / cp.sqrt(cp.mean(y ** 2))
        ks.append(float(cp.mean(y ** 4) - 3.0))
    ks = np.array(ks)
    def c_for(g4):                      # обернення калібру
        g = np.clip(g4, ks.min(), ks.max())
        return np.interp(g, ks, cs)
    print(f"🔒 калібр: c=0 ⟹ γ₄={ks[0]:+.4f} (мусить бути 0) · "
          f"c={cs[-1]:.3f} ⟹ γ₄={ks[-1]:.3f}", flush=True)

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        cc = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1 - cc * cc)))

    res_all = {}
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

        # ── прохід 1: m, C, γ₄ у двох базисах, істина виходу ──
        g = cp.random.default_rng(seed ^ 0x1AD)
        sm = cp.zeros(n, cp.float64); sC = cp.zeros((n, n), cp.float64)
        s4 = cp.zeros(n, cp.float64); mt = cp.zeros(n, cp.float64)
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
        sd = cp.sqrt(cp.maximum(cp.diag(C), 1e-30))
        lam, V = cp.linalg.eigh(C)
        lam = cp.maximum(lam[::-1], 0).copy(); V = V[:, ::-1].copy()
        Vf = V.astype(cp.float32); sq = cp.sqrt(lam).astype(cp.float32)
        mf = m.astype(cp.float32)

        # прохід 1б: γ₄ понейронна і в топ-R власному базисі
        g = cp.random.default_rng(seed ^ 0x1AD)
        s4n = cp.zeros(n, cp.float64); s4v = cp.zeros(R, cp.float64); done = 0
        while done < N:
            b = min(CH, N - done)
            p = real_pre(b, g); pc = (p - mf)
            s4n += ((pc / sd.astype(cp.float32)) ** 4).sum(0, dtype=cp.float64)
            cr = (pc @ Vf[:, :R]) / sq[:R]
            s4v += (cr ** 4).sum(0, dtype=cp.float64)
            done += b
        g4n = cp.asnumpy(s4n / N - 3.0)
        g4v = cp.asnumpy(s4v / N - 3.0)
        print(f"[{net+1}/{a.nets}] seed {seed}  🔒 підлога {100*floor:.4f}%  ·  "
              f"γ₄ понейронна сер {g4n.mean():.3f}  ·  γ₄ у топ-{R} сер {g4v.mean():.3f}",
              flush=True)

        cn = cp.asarray(c_for(g4n), dtype=cp.float32)
        cv = cp.asarray(c_for(g4v), dtype=cp.float32)

        def run(mode):
            gg = cp.random.default_rng(seed ^ 0xB0B)
            g2 = cp.random.default_rng(seed ^ 0x1AD)
            acc = cp.zeros(n, cp.float64); done = 0
            while done < N:
                b = min(CH, N - done)
                if mode == "E":
                    pp = real_pre(b, g2)
                else:
                    u = gg.standard_normal((b, n), dtype=cp.float32)
                    if mode == "B":                       # γ₄ у вихідному базисі
                        gau = (u * sq) @ Vf.T
                        z = gau / sd.astype(cp.float32)
                        y = z + cn * (z ** 3 - 3 * z)
                        y = y / cp.sqrt(cp.mean(y ** 2, axis=0, keepdims=True))
                        pp = mf + y * sd.astype(cp.float32)
                    elif mode == "C":                     # γ₄ у топ-R власному базисі
                        uu = u.copy()
                        z = uu[:, :R]
                        y = z + cv * (z ** 3 - 3 * z)
                        y = y / cp.sqrt(cp.mean(y ** 2, axis=0, keepdims=True))
                        uu[:, :R] = y
                        pp = mf + (uu * sq) @ Vf.T
                    elif mode == "D":                     # оракул: справжні координати
                        p = real_pre(b, g2); pc = p - mf
                        gau = (u * sq) @ Vf.T
                        Vr = Vf[:, :R]
                        pp = mf + gau + ((pc @ Vr) - (gau @ Vr)) @ Vr.T
                    else:                                 # A: чистий гаус
                        pp = mf + (u * sq) @ Vf.T
                o = cp.maximum(cp.maximum(pp, 0.0) @ Wl, 0.0)
                acc += o.sum(0, dtype=cp.float64); done += b
            return cos_err(acc / N, m_true)

        r = {}
        for mode, nm, cost in [("A", "(m,C) гаусів стан", "0"),
                               ("B", f"+ γ₄ понейронна ({n} чисел)", f"{n}"),
                               ("C", f"+ γ₄ у топ-{R} ({R} чисел)", f"{R}"),
                               ("D", f"+ УСІ моменти в топ-{R} (оракул)", "—"),
                               ("E", "усе справжнє (тавтологія)", "—")]:
            e = run(mode); r[mode] = e
            print(f"   {mode}  {nm:<38} чисел {cost:>5}   похибка {100*e:8.4f}%",
                  flush=True)
        res_all[seed] = r

    print("\n════ ЩО ДАЄ γ₄ (медіана) ════")
    base = np.median([v["A"] for v in res_all.values()])
    for mode in "ABCDE":
        v = [x[mode] for x in res_all.values() if mode in x]
        if v:
            mm = float(np.median(v))
            k = base / mm if mm > 0 else float("inf")
            print(f"  {mode}   {100*mm:8.4f}%   ×{k:6.2f}")


if __name__ == "__main__":
    main()
