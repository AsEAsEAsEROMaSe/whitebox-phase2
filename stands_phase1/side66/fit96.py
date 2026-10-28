"""fit96 — РОЗШИРИТИ λ-РОДИНИ З 8 ШАРІВ НА ВСІ 31 І ПЕРЕФІТУВАТИ НА ІСТИНІ

════════════════ ЧОМУ САМЕ ЦЕ ════════════════
У сабміті `_KSIG = 8`: чотири родини контрольних варіатів («вибірка проти
замикання») стоять ЛИШЕ на шарах 1..8, а дають ×1.283 — це ЄДИНИЙ канал
зменшення `V`, який реально працює. `[ЦИТАТА С88]` «ГРОШІ ЖИВУТЬ У ГЛИБИНІ:
шари 9..32 дають ×1.333» ⟹ дві третини глибини НЕ ВИКОРИСТАНІ.

Ціна розширення зміряна в `sub103`: статистики на всіх шарах ≈ **0.11% B**.
Ваги досі фітувались на 60 мережах × 6 жеребів; тут — на 1000 мережах з
ТОЧНОЮ істиною `final_means` (1e9 семплів) з arrow.

════════════════ КЛЮЧОВЕ СПРОЩЕННЯ: ВСЕ ЛІНІЙНЕ ЗА λ ════════════════
```
   v ← (v @ W_i)·gate_i  +  Σ_f λ_f[i]·feat_f[i]        (рекурсія сабміту)
   ours = mc_raw − v
⟹ ours = mc_raw − transport(v₀) − Σ_{i,f} λ_f[i]·B_f[i]
   де B_f[i] = feat_f[i], перенесене валом-із-вибіркових-воріт до виходу
```
⟹ ОДИН лінійний МНК на 124 невідомі замість градієнтного спуску.
🔒 ПЛОМБА: підставивши записані `_LAM_*` для шарів 1..8 і нулі далі, мусимо
   відтворити `ours` діючого сабміту.

════════════════ ЧОТИРИ РОДИНИ (як у сабміті) ════════════════
```
 S: φ(m̂/ŝ)·(ŝ − s_замик)      розбіжність σ
 A: (x̄ − m_замик)              розбіжність середнього активації
 Q: (x² − q_замик) − dq        розбіжність другого моменту (з Еджвортом dq)
 G: (gate − Φ_замик)           розбіжність САМОЇ маски
```

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) розширення 8 → 31 шар дасть ×1.15…×1.5 у `V`                шанс 45%
 (б) ваги глибоких шарів вийдуть МЕНШИМИ за ваги шарів 1..8      шанс 70%
 (в) LOO по мережах втратить <20% виграшу (124 параметри на
     сотні мереж × розіграшів — перенавчання малоймовірне)       шанс 65%
 (г) родина G (маска) виявиться найсильнішою в глибині           шанс 40%
 (д) 🔑 сумарно бал зрушиться нижче 1.80e-07                     шанс 35%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
REP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
KSIG_NEW = min(int(sys.argv[5]) if len(sys.argv) > 5 else 30, 30)
GATE_N = 1024
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
_C3, _C4 = 1.0 / 3.0, 1.0 / 12.0


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks()
                        .flatten()).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i]
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def closure_parts(W, ksig):
    """Точний порт `_closure_parts` сабміту: Мелер K=2, діагональ точна."""
    dev = W[0].device
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = torch.eye(256, dtype=torch.float64, device=dev)
    S, M, Q, P, A, B = [], [], [], [], [], []
    for i in range(32):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), Phi(t)
        if 1 <= i <= ksig:
            S.append(s)
            P.append(Ph)
            s2ph = (s * s) * ph
            A.append(s2ph)
            B.append(s2ph * t)
        m = mp * Ph + s * ph
        if i == 31:
            break
        C = (cp * torch.outer(Ph, Ph)
             + (0.5 * (cp * cp)) * torch.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if 1 <= i <= ksig:
            M.append(m)
            Q.append(C.diagonal() + m * m)
    return S, M, Q, P, A, B, m


def basis(W, seed, npair, ksig, dev):
    """mc_raw, v₀-перенесене, і 4×ksig базисних векторів B_f[i]."""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, 256, generator=g, device=dev)
    cov = (z.T @ z).double() / npair
    wh = torch.linalg.inv(torch.linalg.cholesky(cov)).T.float()
    w0 = wh @ W[0]
    y = z @ w0
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    denom = float(2 * npair)
    sg0 = (W[0] * W[0]).sum(0).clamp_min(1e-30).sqrt().double()
    v0 = x.double().sum(0) / denom - sg0 * S2PI
    S, M, Q, P, A, B, _ = closure_parts(W, ksig)

    feats = []                       # (шар, родина, вектор) у моменті появи
    gates = []
    for i in range(1, 32):
        p = x @ W[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        gates.append(gate)
        if i <= ksig:
            pd = p.double()
            p2 = pd * pd
            mh = pd.sum(0) / denom
            e2 = p2.sum(0) / denom
            sh2 = (e2 - mh * mh).clamp_min(1e-30)
            sh = sh2.sqrt()
            e3 = (p2 * pd).sum(0) / denom
            e4 = (p2 * p2).sum(0) / denom
            mh2 = mh * mh
            k3 = e3 - 3.0 * mh * e2 + 2.0 * (mh2 * mh)
            k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                  - 3.0 * (mh2 * mh2) - 3.0 * (sh2 * sh2))
            dq = ((k3 / (sh2 * sh)) * (_C3 * A[i - 1])
                  - (k4 / (sh2 * sh2)) * (_C4 * B[i - 1]))
            xr = torch.relu(p).double()
            xm = xr.sum(0) / denom
            xq = (xr * xr).sum(0) / denom
            feats.append((i, [phi(mh / sh) * (sh - S[i - 1]),
                              xm - M[i - 1],
                              (xq - Q[i - 1]) - dq,
                              gate - P[i - 1]]))
        x = torch.relu(p)
    mc_raw = x.double().sum(0) / denom

    def transport(vec, frm):
        for j in range(frm + 1, 32):
            vec = (vec @ W[j].double()) * gates[j - 1]
        return vec

    v0t = transport(v0, 0)
    Bmat = []
    for i, fl in feats:
        for fv in fl:
            Bmat.append(transport(fv, i))
    return mc_raw, v0t, torch.stack(Bmat) if Bmat else None


print(f"fit96 · {DEV} · {NNET} мереж × {REP} жеребів · ksig={KSIG_NEW}",
      flush=True)
t0 = time.time()
ROWS = []
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        for r in range(REP):
            mc, v0t, Bm = basis(W, 730000 + 613 * q + 17 * r, NPAIR,
                                KSIG_NEW, DEV)
            ROWS.append((q, mc - v0t - tru, Bm, tru))
        if q % 10 == 9 or q == 0:
            print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)

nb = ROWS[0][2].shape[0]
print(f"\n  базисних векторів: {nb} (= 4 родини × {KSIG_NEW} шарів)")


def fit(idx, lam):
    G = torch.zeros(nb, nb, dtype=torch.float64, device=DEV)
    h = torch.zeros(nb, dtype=torch.float64, device=DEV)
    for k in idx:
        _, r0, Bm, _ = ROWS[k]
        G += Bm @ Bm.T
        h += Bm @ r0
    G += lam * torch.eye(nb, dtype=torch.float64, device=DEV) * G.diagonal().mean()
    return torch.linalg.solve(G, h)


def score(idx, w):
    tot = tn = 0.0
    for k in idx:
        _, r0, Bm, tru = ROWS[k]
        res = r0 - (w @ Bm if w is not None else 0.0)
        tot += float((res * res).sum())
        tn += float((tru * tru).sum())
    return tot / tn


nets = sorted(set(r[0] for r in ROWS))
half = len(nets) // 2
tr = [k for k, r in enumerate(ROWS) if r[0] in nets[:half]]
va = [k for k, r in enumerate(ROWS) if r[0] in nets[half:]]
base = score(va, None)
print(f"🔒 БАЗА (λ=0, чиста МК з вибілюванням): val MSE={base:.4e} "
      f"({base**0.5*100:.4f}%)")
def fit_sub(idx, lam, ncols):
    G = torch.zeros(ncols, ncols, dtype=torch.float64, device=DEV)
    h = torch.zeros(ncols, dtype=torch.float64, device=DEV)
    for k in idx:
        _, r0, Bm, _ = ROWS[k]
        Bs = Bm[:ncols]
        G += Bs @ Bs.T
        h += Bs @ r0
    G += lam * torch.eye(ncols, dtype=torch.float64,
                         device=DEV) * G.diagonal().mean()
    return torch.linalg.solve(G, h)


def score_sub(idx, w, ncols):
    tot = tn = 0.0
    for k in idx:
        _, r0, Bm, tru = ROWS[k]
        res = r0 - w @ Bm[:ncols]
        tot += float((res * res).sum())
        tn += float((tru * tru).sum())
    return tot / tn


print(f"{'шарів':>6s}{'гребінь':>10s}{'train':>12s}{'VAL':>12s}{'V ×':>9s}")
BEST = {}
for nl in (8, 16, 30):
    nc = 4 * nl
    for lam in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
        w = fit_sub(tr, lam, nc)
        s_tr, s_va = score_sub(tr, w, nc), score_sub(va, w, nc)
        g = base / s_va
        if BEST.get(nl, (0,))[0] < g:
            BEST[nl] = (g, lam, s_va, w)
        print(f"{nl:>6d}{lam:>10.0e}{s_tr**0.5*100:>11.4f}%"
              f"{s_va**0.5*100:>11.4f}%{g:>9.3f}")
print()
for nl in (8, 16, 30):
    g, lam, s_va, _ = BEST[nl]
    print(f"🔑 {nl:2d} шарів: найкраще V ×{g:.3f} (гребінь {lam:.0e}) ⟹ "
          f"бал ≈ {1.86762e-07 / g:.4e}")
w = BEST[30][3]

w16 = BEST[16][3].reshape(16, 4)
print("\n═══ ВАГИ ДЛЯ 16 ШАРІВ (у порядку S, A, Q, G) ═══")
for fam, k in (("_LAM_S", 0), ("_LAM_A", 1), ("_LAM_Q", 2), ("_LAM_G", 3)):
    print(f"{fam} = (" + ", ".join(f"{float(w16[i,k]):.7f}" for i in range(16))
          + ")")
wm = w.reshape(-1, 4)
print("\nваги по шарах (S=σ, A=середнє, Q=2-й момент, G=маска):")
for i in range(0, KSIG_NEW, 3):
    print(f"  шар {i+1:2d}: " + " ".join(f"{float(x):+.5f}" for x in wm[i]))

print("🔑 ЧИТАТИ: якщо ваги шарів 9..31 близькі до нуля — глибина не несе")
print("   інформації, і `_KSIG=8` уже оптимальний.")
