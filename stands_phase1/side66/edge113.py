#!/usr/bin/env python3
"""ФЕРОМОН НА РЕБРАХ (С113) — ідея ЮЗЕРА після мурашиної аналогії.

════════ ОБ'ЄКТ ════════
Тотожність (точна, не наближення):
    p_{L+1}[j] = Σ_i x[i]·W[i,j],  x = h_L
    m_{L+1}[j] = E[relu(p_{L+1}[j])] = Σ_i W[i,j]·A[i,j]
    A[i,j] = E[ x[i] · 1{p_{L+1}[j] > 0} ]        ← ФЕРОМОН РЕБРА

⟹ ідеальне `A` ⟺ ідеальне `m` ⟹ ОРАКУЛЬНА СТЕЛЯ ВЖЕ ЗМІРЯНА (`allm`, ×30.9).
Питання НЕ про стелю, а про ПРЕДСТАВЛЕННЯ: чи залишок
    R = A_справжнє − A_гаусове
має структуру, якої немає у вузловому представленні (низький ранг, залежність
від кількох спостережуваних). Якщо так — з'являється ДЕТЕРМІНОВАНА поправка.

════════ ГАУСІВ ДВІЙНИК ════════
`A_гаусове` — не формула з голови, а операціоналізація припущення замикання:
синтетичні передактивації `p_L ~ N(m_L, C_L)` з ІСТИННИМИ (m_L, C_L), далі
relu і W точно. ⟹ різниця ізолює РІВНО негаусовість `p_L`, і нічого більше.

════════ ПЛОМБИ ════════
S1  тотожність:  Σ_i W[i,j]·A_real[i,j] = m_true[j]           (мусить збігтись)
S2  тотожність на двійнику: Σ_i W[i,j]·A_gauss[i,j] = m_gauss[j]
S3  ⟹ Δm_ребер = Σ_i W·R  МУСИТЬ дорівнювати (m_true − m_gauss)
S4  НУЛЬОВИЙ КОНТРОЛЬ: стадія L=-1 (вхід z гаусів ТОЧНО за побудовою)
    ⟹ R там мусить бути ЧИСТИМ ШУМОМ, ratio ≈ 1.0
S5  підлога шуму — з двох НЕЗАЛЕЖНИХ половин вибірки
S6  масштабування: N×4 ⟹ шум ÷2, справжній сигнал стоїть

════════ ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (записано, щоб не тлумачити заднім числом) ══
  ratio (сигнал/шум) на глибоких шарах > 3     : 70%
  ефективний ранг R < 16                        : 20%
  corr(R, W) істотна (|corr| > 0.1)             : 25%
  з цього вийде детермінована поправка           : 10%

Запуск (GPU за UUID, як вимагає CLAUDE.md):
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 \
  PYTHONPATH=/home/trim/gpupkg /usr/bin/python3 side66/edge113.py --nets 6
"""
import argparse
import json
import os
import sys
import time

import numpy as np

CKPT = "/home/trim/wblogs/edge113_ckpt.json"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--nets", type=int, default=6)
    ap.add_argument("--nsamp", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--stages", type=str, default="-1,0,3,7,11,14")
    ap.add_argument("--scaletest", action="store_true")
    a = ap.parse_args()

    import cupy as cp
    log(f"cupy {cp.__version__} · пристроїв видно {cp.cuda.runtime.getDeviceCount()}"
        f" · {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    assert cp.cuda.runtime.getDeviceCount() == 1, "пінувати ЛИШЕ по UUID"

    W_, D_, N, CH = a.width, a.depth, a.nsamp, a.chunk
    stages = [int(s) for s in a.stages.split(",")]

    def make_net(seed):
        rng = cp.random.default_rng(seed)
        s = float(np.sqrt(2.0 / W_))
        return [(rng.standard_normal((W_, W_), dtype=cp.float32) * s)
                for _ in range(D_)]

    def pre_at(Ws, x, L):
        """передактивація шару L (L=-1 ⟹ сам вхід z)."""
        if L < 0:
            return x
        h = x
        for l in range(L + 1):
            p = h @ Ws[l]
            h = cp.maximum(p, 0.0)
        return p

    def collect(Ws, L, gen, n, halves=True):
        """Накопичити A, m, і моменти x та p_L. gen(k) віддає передактивацію p_L."""
        Wn = Ws[L + 1]
        A = [cp.zeros((W_, W_), cp.float32) for _ in range(2 if halves else 1)]
        cnt = [0, 0]
        m_out = cp.zeros(W_, cp.float64)
        qsum = cp.zeros(W_, cp.float64)
        xs = cp.zeros(W_, cp.float64)
        xs2 = cp.zeros(W_, cp.float64)
        sx = cp.zeros(W_, cp.float64)
        Cx = cp.zeros((W_, W_), cp.float64)
        done = 0
        k = 0
        while done < n:
            b = min(CH, n - done)
            pL = gen(b)                       # [b, W] передактивація шару L
            x = cp.maximum(pL, 0.0) if L >= 0 else pL
            pn = x @ Wn
            g = (pn > 0).astype(cp.float32)
            half = k % 2 if halves else 0
            A[half] += x.T @ g
            cnt[half] += b
            m_out += cp.maximum(pn, 0.0).sum(0, dtype=cp.float64)
            qsum += g.sum(0, dtype=cp.float64)
            xs += x.sum(0, dtype=cp.float64)
            xs2 += (x.astype(cp.float64) ** 2).sum(0)
            sx += pL.sum(0, dtype=cp.float64)
            Cx += (pL.T.astype(cp.float64) @ pL.astype(cp.float64))
            done += b
            k += 1
        Ah = [A[i] / max(1, cnt[i]) for i in range(len(A))]
        Afull = (A[0] + A[1]) / n if halves else A[0] / n
        return dict(A=Afull, Ah=Ah, m=m_out / n, mx=sx / n, Cx=Cx / n,
                    qg=qsum / n, ex=xs / n, sx=cp.sqrt(cp.maximum(
                        xs2 / n - (xs / n) ** 2, 0.0)))

    def run_net(seed, out):
        Ws = make_net(seed)
        rng = cp.random.default_rng(seed ^ 0xE06E)
        res = {}
        for L in stages:
            t0 = time.time()

            def gen_real(b, _L=L, _Ws=Ws, _rng=rng):
                z = _rng.standard_normal((b, W_), dtype=cp.float32)
                return pre_at(_Ws, z, _L)

            R1 = collect(Ws, L, gen_real, N)
            mx = R1["mx"]
            C = R1["Cx"] - cp.outer(mx, mx)
            C = 0.5 * (C + C.T)
            C += cp.eye(W_, dtype=cp.float64) * (1e-6 * float(cp.trace(C)) / W_)
            Lc = cp.linalg.cholesky(C).astype(cp.float32)
            mxf = mx.astype(cp.float32)

            def gen_gauss(b, _Lc=Lc, _mx=mxf, _rng=rng):
                u = _rng.standard_normal((b, W_), dtype=cp.float32)
                return _mx + u @ _Lc.T

            R2 = collect(Ws, L, gen_gauss, N)

            Wn = Ws[L + 1]
            m_id = (Wn * R1["A"]).sum(0)
            m_id2 = (Wn * R2["A"]).sum(0)
            s1 = float(cp.abs(m_id - R1["m"]).max() / (cp.abs(R1["m"]).mean() + 1e-30))
            s2 = float(cp.abs(m_id2 - R2["m"]).max() / (cp.abs(R2["m"]).mean() + 1e-30))

            Rm = R1["A"] - R2["A"]
            dm_edge = (Wn * Rm).sum(0)
            dm_true = R1["m"] - R2["m"]
            s3 = float(cp.abs(dm_edge - dm_true).max()
                       / (cp.abs(dm_true).mean() + 1e-30))

            nA = float(cp.linalg.norm(R1["A"]))
            nR = float(cp.linalg.norm(Rm))
            noise = float(cp.linalg.norm(R1["Ah"][0] - R1["Ah"][1])) / 2.0
            ratio = nR / max(noise, 1e-30)

            U, sv_g, Vt = cp.linalg.svd(Rm.astype(cp.float32), full_matrices=False)
            sv = cp.asnumpy(sv_g).astype(np.float64)
            e = sv ** 2
            pr = float(e.sum() ** 2 / (e ** 2).sum())
            top16 = float(e[:16].sum() / e.sum())
            # ── відділити те, що замикання ВЖЕ знає на рівні вузлів ──
            # ранг-1 R ≈ Δ(вхідний профіль) ⊗ (частота воріт) — це вузлова похибка
            Rr1 = cp.outer(U[:, 0] * sv_g[0], Vt[0])
            Rp = Rm - Rr1
            dm_r1 = (Wn * Rr1).sum(0)
            dm_rp = (Wn * Rp).sum(0)
            nd = float(cp.linalg.norm(dm_true)) + 1e-30
            f_r1 = float(cp.linalg.norm(dm_r1)) / nd
            f_rp = float(cp.linalg.norm(dm_rp)) / nd
            ratio_perp = float(cp.linalg.norm(Rp)) / max(noise, 1e-30)
            e_perp = e[1:]
            pr_perp = float(e_perp.sum() ** 2 / (e_perp ** 2).sum())

            # ── ЩО САМЕ СИДИТЬ У ДВОХ НАПРЯМКАХ? ──
            # кандидати з боку ВХОДУ (i) і з боку ВИХОДУ (j) — усе, що
            # замикання або вже знає, або могло б знати
            q_r = R1["qg"]; q_g = R2["qg"]
            ex_r = R1["ex"]; ex_g = R2["ex"]
            cand_i = {"E[x] справж": ex_r, "Δ E[x]": ex_r - ex_g,
                      "sd(x)": R1["sx"]}
            cand_j = {"ворота q": q_r, "Δ ворота": q_r - q_g,
                      "m виходу": R1["m"]}
            def cr(v, w):
                v = cp.asnumpy(v).astype(np.float64)
                w = cp.asnumpy(w).astype(np.float64)
                if v.std() < 1e-30 or w.std() < 1e-30:
                    return 0.0
                return float(np.corrcoef(v, w)[0, 1])
            vecs = {}
            for r in range(3):
                for nm, c in cand_i.items():
                    vecs[f"u{r+1}·{nm}"] = cr(U[:, r], c)
                for nm, c in cand_j.items():
                    vecs[f"v{r+1}·{nm}"] = cr(Vt[r], c)
            row_vecs = vecs
            wf = cp.asnumpy(Wn).ravel()
            rf = cp.asnumpy(Rm).ravel()
            af = cp.asnumpy(R2["A"]).ravel()
            cw = float(np.corrcoef(wf, rf)[0, 1])
            ca = float(np.corrcoef(af, rf)[0, 1])

            rel_err_m = float(cp.linalg.norm(dm_true) / cp.linalg.norm(R1["m"]))

            row = dict(L=L, relR=nR / nA, noise_rel=noise / nA, ratio=ratio,
                       PR=pr, top16=top16, corr_W=cw, corr_A=ca,
                       f_r1=f_r1, f_rp=f_rp, ratio_perp=ratio_perp,
                       e12=float(e[:2].sum() / e.sum()),
                       e3p=float(e[2:].sum() / e.sum()),
                       PR_perp=pr_perp, N=N,
                       step_err_m=rel_err_m, S1=s1, S2=s2, S3=s3,
                       sv_head=[float(x) for x in sv[:6]], vecs=row_vecs,
                       secs=time.time() - t0)
            res[str(L)] = row
            log(f"  L={L:3d} ratio {ratio:6.2f} ⊥{ratio_perp:6.2f}  PR {pr:6.1f} "
                f"⊥{pr_perp:6.1f}  ранг1 дає {100*f_r1:5.1f}% Δm, решта "
                f"{100*f_rp:5.1f}%  крок m {100*rel_err_m:.4f}%  "
                f"🔒S {s1:.0e}/{s2:.0e}/{s3:.0e} [{row['secs']:.0f}с]")
            out[str(seed)] = res
            tmp = CKPT + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, CKPT)
        return res

    out = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    seeds = [6113113 + 1000 * k for k in range(a.nets)]
    fails = 0
    for si, sd in enumerate(seeds):
        if str(sd) in out and len(out[str(sd)]) == len(stages):
            log(f"[{si+1}/{len(seeds)}] seed {sd} — вже пораховано, пропуск")
            continue
        log(f"[{si+1}/{len(seeds)}] seed {sd}  N={N}")
        try:
            run_net(sd, out)
            fails = 0
        except Exception as ex:                                   # noqa: BLE001
            fails += 1
            import traceback
            log(f"  🚨 збій: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            if fails >= 3:
                log("три збої поспіль — чистий вихід зі збереженим станом")
                break

    log("\n════ ЗВЕДЕННЯ (медіана по мережах) ════")
    log(f"{'L':>4} {'ratio':>7} {'ratio⊥':>8} {'PR':>7} {'PR⊥':>8} "
        f"{'ранг1 Δm':>9} {'решта Δm':>9} {'крок m %':>9}")
    for L in stages:
        rows = [v[str(L)] for v in out.values() if str(L) in v]
        if not rows:
            continue
        med = lambda k: float(np.median([r[k] for r in rows]))    # noqa: E731
        log(f"{L:>4} {med('ratio'):>7.2f} {med('ratio_perp'):>8.2f} "
            f"{med('PR'):>7.1f} {med('PR_perp'):>8.1f} "
            f"{100*med('f_r1'):>8.1f}% {100*med('f_rp'):>8.1f}% "
            f"{100*med('step_err_m'):>8.4f}%")
    log(f"\nмереж у файлі: {len(out)} · {CKPT}")


if __name__ == "__main__":
    main()
