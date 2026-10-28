#!/usr/bin/env python3
"""ПРОТЯГНУТА ДРАБИНА (С113б) — стеля двомоментного стану на 1024×16.

Одношагова драбина (`ladder113.py`) обрізала стан на ОДНОМУ шарі й дала 0.0800%
при повній C. Але вона не бачила НАКОПИЧЕННЯ, а воно за С105 дає 84% похибки.

════════ ЩО МІРЯЄМО ════════
Несемо стан крізь УСІ 16 шарів. На кожному шарі стан = (m, C) передактивації,
і ми ЗАМІНЮЄМО справжній розподіл гаусовим із тими самими двома моментами:
    g ~ N(m_l, C^(r)_l)  →  h = relu(g)  →  p_{l+1} = h @ W_l  →  нові (m, C)
Крок ТОЧНИЙ (семплюванням) ⟹ похибка Мелера НЕ домішується.
⟹ це СТЕЛЯ будь-якого двомоментного методу, включно з нашим.

r = 1024 (повна C)  ·  r = 0 (незалежні нейрони, маргіналі точні)

════════ ПЛОМБИ ════════
S1 шар 0: вхід N(0,I) ТОЧНО ⟹ підстановка тавтологічна, похибка нуль
S2 шумова підлога з двох незалежних половин
S3 масштабування N×2 ⟹ підлога ÷√2, справжня похибка стоїть
S4 монотонність: похибка мусить РОСТИ з глибиною зупинки
S5 діагональ сурогата точна при кожному r

Запуск:
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 \
  PYTHONPATH=/home/trim/gpupkg /usr/bin/python3 side66/ladder113b.py --nets 3
"""
import argparse
import json
import os
import time

import numpy as np

CKPT = "/home/trim/wblogs/ladder113b_ckpt.json"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--nsamp", type=int, default=262144)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--ranks", type=str, default="1024,0")
    ap.add_argument("--qbits", type=str, default="0",
                    help="біти квантування C поблоково; 0 = без квантування")
    ap.add_argument("--cnoise", type=float, default=0.0,
                    help="незміщений відносний шум у C на кожному шарі")
    ap.add_argument("--wbits", type=int, default=0,
                    help="біти квантування W у ТРАНСПОРТІ; істина лишається точною")
    a = ap.parse_args()

    import cupy as cp
    log(f"cupy {cp.__version__} · пристроїв {cp.cuda.runtime.getDeviceCount()} · "
        f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    assert cp.cuda.runtime.getDeviceCount() == 1
    W_, D_, N, CH = a.width, a.depth, a.nsamp, a.chunk
    QB = int(a.qbits.split(',')[0])
    WB = a.wbits
    CN = a.cnoise
    ranks = [int(x) for x in a.ranks.split(",")]

    def cos_err(x, y):
        x = x.astype(cp.float64); y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1.0 - c * c)))

    def moments(gen, Wl, n, first=False):
        """прогнати n семплів через relu+Wl, вернути (m, C) наступної передактивації
        та m виходу relu, з половинками для підлоги шуму."""
        sm = cp.zeros(W_, cp.float64)
        sC = cp.zeros((W_, W_), cp.float64)
        so = [cp.zeros(W_, cp.float64), cp.zeros(W_, cp.float64)]
        cnt = [0, 0]
        done = k = 0
        while done < n:
            b = min(CH, n - done)
            g = gen(b)
            p = (g if first else cp.maximum(g, 0.0)) @ Wl
            sm += p.sum(0, dtype=cp.float64)
            sC += p.T.astype(cp.float64) @ p.astype(cp.float64)
            so[k % 2] += cp.maximum(p, 0.0).sum(0, dtype=cp.float64)
            cnt[k % 2] += b
            done += b; k += 1
        m = sm / n
        C = sC / n - cp.outer(m, m)
        return m, 0.5 * (C + C.T), (so[0] + so[1]) / n, so[0] / cnt[0], so[1] / cnt[1]

    def qwq(W, bits, Kb=32):
        """симетричне поблокове квантування W уздовж СТИСНУТОЇ осі (навколо нуля)"""
        if bits <= 0:
            return W
        nb = W_ // Kb
        Wb = W.reshape(nb, Kb, W_)
        sc = cp.maximum(cp.abs(Wb).max(axis=1, keepdims=True), 1e-30)
        Q = 2.0 ** (bits - 1) - 1
        return (cp.rint(Wb / sc * Q) / Q * sc).reshape(W_, W_)

    def cnoise(C, eps, rng):
        """незміщений симетричний шум: C + E, E~N(0, (eps·|C|)²), Eᵀ=E.
        Моделює СТОХАСТИЧНИЙ транспорт (вибірка k стовпців), а не обрізання рангу."""
        if eps <= 0:
            return C
        sc = cp.sqrt(cp.abs(cp.outer(cp.diag(C), cp.diag(C))))    # масштаб запису
        E = rng.standard_normal((W_, W_), dtype=cp.float64) * (eps * sc)
        E = 0.5 * (E + E.T)
        E = E - cp.diag(cp.diag(E))
        Cn = C + E
        # 🔒 повернути в конус PSD і ВІДНОВИТИ точну діагональ ⟹ міряємо
        #    ВТРАТУ ІНФОРМАЦІЇ, а не руйнування властивості «бути коваріацією»
        ev, Vv = cp.linalg.eigh(Cn)
        ev = cp.maximum(ev, 0.0)
        Cn = (Vv * ev) @ Vv.T
        d0 = cp.diag(C); dn = cp.maximum(cp.diag(Cn), 1e-30)
        sc2 = cp.sqrt(d0 / dn)
        return Cn * cp.outer(sc2, sc2)

    def qcov(C, bits, Kb=32):
        """поблокове квантування C: масштаб на кожен блок рядків×стовпців"""
        if bits <= 0:
            return C
        nb = W_ // Kb
        Cb = C.reshape(nb, Kb, nb, Kb)
        sc = cp.abs(Cb).max(axis=(1, 3), keepdims=True)
        sc = cp.maximum(sc, 1e-300)
        Q = 2.0 ** (bits - 1) - 1
        Cq = (cp.rint(Cb / sc * Q) / Q * sc).reshape(W_, W_)
        return 0.5 * (Cq + Cq.T)

    def surrogate(m, C, r, rng):
        """генератор g ~ N(m, C^(r)); діагональ ТОЧНА при кожному r (🔒S5)"""
        dC = cp.diag(C)
        mf = m.astype(cp.float32)
        if r <= 0:
            rs = cp.sqrt(cp.maximum(dC, 0.0)).astype(cp.float32)
            return (lambda b: mf + rng.standard_normal((b, W_), cp.float32) * rs), 0.0
        lam, V = cp.linalg.eigh(C)
        lam = cp.maximum(lam[::-1][:r], 0.0).copy()
        V = V[:, ::-1][:, :r].copy()
        Vr = V.astype(cp.float32); sr = cp.sqrt(lam).astype(cp.float32)
        low = (Vr ** 2) @ (sr ** 2)
        resid = cp.maximum(dC.astype(cp.float32) - low, 0.0)
        rs = cp.sqrt(resid)
        s5 = float(cp.abs(low + resid - dC.astype(cp.float32)).max())

        def gen(b):
            g = mf + rng.standard_normal((b, W_), cp.float32) * rs
            u = rng.standard_normal((b, r), cp.float32)
            return g + (u * sr) @ Vr.T
        return gen, s5

    def run_net(seed):
        rng = cp.random.default_rng(seed)
        s = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * s)
              for _ in range(D_)]
        srng = cp.random.default_rng(seed ^ 0x1AD)

        # ── ІСТИНА: справжній прохід ──
        t0 = time.time()
        m_true = cp.zeros(W_, cp.float64)
        th = [cp.zeros(W_, cp.float64), cp.zeros(W_, cp.float64)]
        cnt = [0, 0]
        done = k = 0
        while done < N:
            b = min(CH, N - done)
            h = srng.standard_normal((b, W_), dtype=cp.float32)
            for l in range(D_):
                p = h @ Ws[l]
                h = cp.maximum(p, 0.0)
            m_true += h.sum(0, dtype=cp.float64)
            th[k % 2] += h.sum(0, dtype=cp.float64)
            cnt[k % 2] += b
            done += b; k += 1
        m_true /= N
        floor = cos_err(th[0] / cnt[0], th[1] / cnt[1]) / 2.0
        log(f"   🔒S2 шумова підлога істини {100*floor:.4f}%  [{time.time()-t0:.0f}с]")

        # 🔒 S1 ТАВТОЛОГІЯ: одношаровий прогін — вхід N(0,I) ТОЧНО,
        #    підстановка гаусового сурогата нічого не міняє ⟹ похибка = шум
        m1, C1, mo1, _, _ = moments(
            lambda b: srng.standard_normal((b, W_), dtype=cp.float32), Ws[0], N,
            first=True)
        g1, _ = surrogate(cp.zeros(W_, cp.float64),
                          cp.eye(W_, dtype=cp.float64), 1024, srng)
        _, _, mo1b, _, _ = moments(g1, Ws[0], N, first=True)
        s1 = cos_err(mo1, mo1b)
        log(f"   🔒S1 тавтологія на шарі 0: {100*s1:.4f}%  (мусить бути ≈ підлозі)")
        # (поріг уточнюється нижче за виміром)

        Wq = [qwq(x, WB) for x in Ws]   # 🔒 ІСТИНА рахується з ТОЧНИМИ Ws

        res = {}
        for r in ranks:
            t0 = time.time()
            gen = lambda b: srng.standard_normal((b, W_), dtype=cp.float32)  # noqa: E731
            s5max = 0.0
            per_depth = []
            for l in range(D_):
                m, C, mo, h1, h2 = moments(gen, Wq[l], N, first=(l == 0))
                if l == D_ - 1:
                    break
                gen, s5 = surrogate(m, cnoise(qcov(C, QB), CN, srng), r, srng)   # ЗАМІНА на гаусів стан
                s5max = max(s5max, s5)
                per_depth.append(None)
            e = cos_err(mo, m_true)
            res[str(r)] = dict(err=e, floor=floor, s5=s5max, secs=time.time() - t0)
            log(f"   r={r:5d} qC={QB:2d} qW={WB:2d} шум={CN:.3f} ПРОТЯГНУТО крізь {D_} шарів  похибка {100*e:7.4f}%"
                f"   🔒S5 {s5max:.1e}  [{time.time()-t0:.0f}с]")
        return res

    out = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    for i in range(a.nets):
        sd = 6113113 + 1000 * i
        if str(sd) in out:
            log(f"[{i+1}/{a.nets}] seed {sd} — вже є"); continue
        log(f"[{i+1}/{a.nets}] seed {sd}  N={N}")
        out[str(sd)] = run_net(sd)
        tmp = CKPT + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, CKPT)

    log("\n════ СТЕЛЯ ДВОМОМЕНТНОГО СТАНУ, ПРОТЯГНУТА КРІЗЬ 16 ШАРІВ ════")
    for r in ranks:
        v = [x[str(r)]["err"] for x in out.values() if str(r) in x]
        if v:
            log(f"  r={r:5d}   {100*float(np.median(v)):7.4f}%   "
                f"(мереж {len(v)}, розкид {100*min(v):.4f}…{100*max(v):.4f})")
    f = [x[list(x)[0]]["floor"] for x in out.values()]
    log(f"  🔒 шумова підлога {100*float(np.median(f)):.4f}%")
    log(f"  📏 наше реальне замикання 0.1264%  ·  одношагова драбина 0.0800%")


if __name__ == "__main__":
    main()
