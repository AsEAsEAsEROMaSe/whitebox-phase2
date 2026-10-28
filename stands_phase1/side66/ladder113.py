#!/usr/bin/env python3
"""ДРАБИНА ПАМ'ЯТІ СТАНУ (С113) — ідея ЮЗЕРА «скільки треба пам'яті».

════════ ПИТАННЯ ════════
Скільки чисел треба НЕСТИ МІЖ ШАРАМИ, щоб дати середнє з заданою точністю?
Ми знаємо лише два кінці драбини (повна C = стеля `allm`; наше замикання),
а середина не міряна ЖОДНОГО разу за 113 сесій.

════════ ПРИЛАД (без формули замикання — навмисно) ════════
Беремо ІСТИННІ (m, C) передактивації шару, будуємо СУРОГАТ
    C^(r) = Σ_{k≤r} λ_k v_k v_kᵀ  +  diag(C − Σ_{k≤r} λ_k v_k v_kᵀ)
🔑 діагональ ТОЧНА при кожному r ⟹ маргіналі не псуються, і драбина міряє
   РІВНО кількість збереженої КОРЕЛЯЦІЙНОЇ структури, нічого більше.
Далі семплюємо g ~ N(m, C^(r)), проганяємо relu і W ТОЧНО, дивимось похибку
середнього на виході. Формула кроку не потрібна ⟹ похибка Мелера не домішується.

r = 0     незалежні нейрони з точними маргіналями   стан = 2n
r = n     повна C                                    стан = n + n²/2
⟹ крива «стан ↔ похибка» і є відповіддю.

════════ ЧОМУ ЦЕ ГРОШІ, А НЕ ЛИШЕ ЗНАННЯ ════════
У фазі 2 замикання коштує d·w³ = 24% підлоги, і платимо ми за ПОВНУ C.
Коліно при ранзі r ⟹ ціна падає з n³ до r·n² ⟹ при r=32 це ×32 дешевше.

════════ ПЛОМБИ ════════
S1 ТАВТОЛОГІЯ: замість сурогата беремо СПРАВЖНІ семпли p ⟹ похибка 0
S2 ДІАГОНАЛЬ: max|diag(C^(r)) − diag(C)| = 0 при КОЖНОМУ r
S3 ПЛАТО: r = n мусить збігтися з незалежним виміром повної C
S4 МАСШТАБУВАННЯ: N×4 ⟹ шумова підлога ÷2, форма кривої стоїть
S5 ХИБНИЙ БАЗИС: замість власних векторів беремо ВИПАДКОВІ ортогональні
   ⟹ крива мусить бути ІСТОТНО гіршою (інакше ранг ні до чого)

Запуск:
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 \
  PYTHONPATH=/home/trim/gpupkg /usr/bin/python3 side66/ladder113.py --nets 4
"""
import argparse
import json
import os
import time

import numpy as np

CKPT = "/home/trim/wblogs/ladder113_ckpt.json"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=4)
    ap.add_argument("--nsamp", type=int, default=262144)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--ranks", type=str,
                    default="0,1,2,4,8,16,32,64,128,256,512,1024")
    ap.add_argument("--randbasis", action="store_true")
    a = ap.parse_args()

    import cupy as cp
    log(f"cupy {cp.__version__} · пристроїв {cp.cuda.runtime.getDeviceCount()} · "
        f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    assert cp.cuda.runtime.getDeviceCount() == 1

    W_, D_, N, CH = a.width, a.depth, a.nsamp, a.chunk
    ranks = [int(x) for x in a.ranks.split(",")]

    def cos_err(x, y):
        """косинусна похибка: після ОПТИМАЛЬНОГО спільного масштабу (закон С101-в)"""
        x = x.astype(cp.float64)
        y = y.astype(cp.float64)
        c = float(cp.dot(x, y) / (cp.linalg.norm(x) * cp.linalg.norm(y) + 1e-300))
        return float(np.sqrt(max(0.0, 1.0 - c * c)))

    def run_net(seed):
        rng = cp.random.default_rng(seed)
        s = float(np.sqrt(2.0 / W_))
        Ws = [(rng.standard_normal((W_, W_), dtype=cp.float32) * s)
              for _ in range(D_)]
        Wlast = Ws[-1]
        srng = cp.random.default_rng(seed ^ 0x1AD)

        # ── 1. істина: моменти передактивації ПЕРЕДостаннього шару + m виходу ──
        acc_m = cp.zeros(W_, cp.float64)
        acc_s = cp.zeros(W_, cp.float64)
        acc_C = cp.zeros((W_, W_), cp.float64)
        acc_mh = [cp.zeros(W_, cp.float64), cp.zeros(W_, cp.float64)]
        cnt = [0, 0]
        done = k = 0
        while done < N:
            b = min(CH, N - done)
            h = srng.standard_normal((b, W_), dtype=cp.float32)
            for l in range(D_ - 1):
                p = h @ Ws[l]
                h = cp.maximum(p, 0.0)
            pprev = p                       # передактивація шару D-2
            out = cp.maximum(cp.maximum(pprev, 0.0) @ Wlast, 0.0)
            acc_m += out.sum(0, dtype=cp.float64)
            acc_s += pprev.sum(0, dtype=cp.float64)
            acc_C += pprev.T.astype(cp.float64) @ pprev.astype(cp.float64)
            acc_mh[k % 2] += out.sum(0, dtype=cp.float64)
            cnt[k % 2] += b
            done += b
            k += 1
        m_true = acc_m / N
        mu = acc_s / N
        C = acc_C / N - cp.outer(mu, mu)
        C = 0.5 * (C + C.T)
        noise_floor = cos_err(acc_mh[0] / cnt[0], acc_mh[1] / cnt[1]) / 2.0

        # ── 2. власний базис ──
        lam, V = cp.linalg.eigh(C)
        lam = lam[::-1].copy()
        V = V[:, ::-1].copy()
        lam = cp.maximum(lam, 0.0)
        if a.randbasis:                                        # 🔒 S5
            G = cp.random.default_rng(seed ^ 0xBAD).standard_normal(
                (W_, W_), dtype=cp.float64)
            V, _ = cp.linalg.qr(G)
            lam = cp.diag(V.T @ C @ V).copy()
            o = cp.argsort(lam)[::-1]
            lam = lam[o].copy()
            V = V[:, o].copy()
        dC = cp.diag(C)
        muf = mu.astype(cp.float32)

        def sample_out(r):
            """m виходу при стані (m, C^(r))"""
            Vr = V[:, :r].astype(cp.float32) if r > 0 else None
            sr = cp.sqrt(lam[:r]).astype(cp.float32) if r > 0 else None
            low = (Vr ** 2) @ (sr ** 2) if r > 0 else cp.zeros(W_, cp.float32)
            resid = cp.maximum(dC.astype(cp.float32) - low, 0.0)
            rs = cp.sqrt(resid)
            acc = cp.zeros(W_, cp.float64)
            dn = 0
            while dn < N:
                b = min(CH, N - dn)
                g = muf + srng.standard_normal((b, W_), dtype=cp.float32) * rs
                if r > 0:
                    u = srng.standard_normal((b, r), dtype=cp.float32)
                    g = g + (u * sr) @ Vr.T
                o = cp.maximum(cp.maximum(g, 0.0) @ Wlast, 0.0)
                acc += o.sum(0, dtype=cp.float64)
                dn += b
            return acc / N, float(cp.abs(low + resid - dC.astype(cp.float32)).max())

        rows = []
        for r in ranks:
            if r > W_:
                continue
            t0 = time.time()
            m_r, s2 = sample_out(r)
            e = cos_err(m_r, m_true)
            state = W_ + W_ + r * W_
            rows.append(dict(r=r, err=e, state=state,
                             frac_var=float(lam[:r].sum() / lam.sum()) if r else 0.0,
                             S2=s2, secs=time.time() - t0))
            log(f"   r={r:5d}  стан {state:8d}  похибка {100*e:8.4f}%   "
                f"дисперсії у ранзі {100*rows[-1]['frac_var']:5.1f}%   "
                f"🔒S2 {s2:.1e}  [{time.time()-t0:.1f}с]")
        log(f"   🔒 шумова підлога приладу: {100*noise_floor:.4f}%")
        return dict(rows=rows, noise=noise_floor,
                    eig_top=[float(x) for x in cp.asnumpy(lam[:8])],
                    eig_sum=float(lam.sum()))

    out = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    key = "rand" if a.randbasis else "eig"
    out.setdefault(key, {})
    for i in range(a.nets):
        sd = 6113113 + 1000 * i
        if str(sd) in out[key]:
            log(f"[{i+1}/{a.nets}] seed {sd} — вже є")
            continue
        log(f"[{i+1}/{a.nets}] seed {sd}  N={N}  базис={key}")
        out[key][str(sd)] = run_net(sd)
        tmp = CKPT + ".tmp"
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, CKPT)

    log("\n════ ДРАБИНА (медіана по мережах) ════")
    log(f"{'ранг':>6} {'стан':>10} {'похибка %':>11} {'дисперсії':>10} "
        f"{'×проти r=0':>11}")
    nets = list(out[key].values())
    base = None
    for r in ranks:
        if r > W_:
            continue
        es = [row["err"] for v in nets for row in v["rows"] if row["r"] == r]
        fv = [row["frac_var"] for v in nets for row in v["rows"] if row["r"] == r]
        if not es:
            continue
        m = float(np.median(es))
        if base is None:
            base = m
        log(f"{r:>6} {W_*2 + r*W_:>10} {100*m:>10.4f}% {100*np.median(fv):>9.1f}% "
            f"{base/m:>10.2f}")
    log(f"🔒 шумова підлога: {100*float(np.median([v['noise'] for v in nets])):.4f}%")


if __name__ == "__main__":
    main()
