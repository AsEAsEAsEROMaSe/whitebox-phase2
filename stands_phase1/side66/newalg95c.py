"""newalg95c — ТОЙ САМИЙ ТЕСТ, АЛЕ НА ДРУГОМУ МОМЕНТІ (де й живе проблема)

════════════════ ЗВІДКИ ════════════════
`[ЗМІРЯНО newalg95b, 4 мережі × 2¹⁸]` негаусовість ОДНОВИМІРНА:
   шар 30: γ₃(û)=0.4268 · γ₃(ПК⊥)=−0.0490 · γ₄(û)=0.2945 · γ₄(ПК⊥)=−0.1257
   АЛЕ виграш на E[ReLU(p)] = ×1.02 (гаусова 0.00139 → rank-1 0.00136) ⟹ НУЛЬ
🔑 МЕХАНІЗМ ПРОВАЛУ: при t≈3.5 ReLU майже лінійний, `E[ReLU]` — гладкий
   функціонал; внесок γ₃ іде як σ·γ₃·φ(3.5)/6 ≈ 6e-5·σ проти m=3.5σ ⟹
   **ПЕРШИЙ МОМЕНТ НЕ БАЧИТЬ ФОРМИ РОЗПОДІЛУ.**
`[ЦИТАТА С88]` `C` соло ×0.53 · `σ` соло ×0.54 ⟹ проблема в ДРУГОМУ моменті.
`[ЦИТАТА С91]` `C` існує РІВНО щоб видати `σ` (L8≡L6); `σ` терпить <1%.
`[ЦИТАТА С87]` позадіагональ 1.69% проти діагоналі 0.23%.

════════════════ ЩО ПЕРЕВІРЯЄМО ════════════════
Cov(h) через УМОВНЕ представлення (точна одновимірна густина R):
    Cov(h) = E_R[ Cov(h|R) ] + Cov_R( m(R) )
             ├─ Мелер з умовними (μ+R·c, σ_g, ρ_g)   ├─ ТОЧНИЙ rank-1 член
Проти безумовного Мелера з (μ, σ, ρ) — це і є звичайне гаусове замикання.

════════════════ ЩО МІРЯЄМО (шари 7,15,23,30) ════════════════
 M1  ‖Cov_оцінка − Cov_істина‖_F / ‖Cov_істина‖_F : гаусова проти умовної
 M2  окремо ДІАГОНАЛЬ (це і є `σ`, яка терпить <1%) і ПОЗАДІАГОНАЛЬ
 M3  для довідки — та сама похибка на E[h] (мусить лишитись ×1.0)
 ⛔ КОНТРОЛЬ «ЧУЖИЙ R»: сітка R з ІНШОЇ мережі ⟹ виграш мусить зникнути
 🔒 ПЛОМБА: K_сітки = 1 (одна точка в E[R]) мусить ТОЧНО відтворити гаусову

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) похибка ДІАГОНАЛІ гаусової на шарі 30 буде 0.2…1.5%          шанс 65%
 (б) 🔑 умовна схема покращить ДІАГОНАЛЬ у >1.5×                  шанс 45%
 (в) позадіагональ буде помітно гіршою за діагональ (як у С87)    шанс 60%
 (г) умовна схема покращить ПОЗАДІАГОНАЛЬ сильніше за діагональ   шанс 40%
 (д) чужий R дасть ×1.0                                          шанс 85%
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

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
KGRID = int(sys.argv[4]) if len(sys.argv) > 4 else 32
SPHERE = int(sys.argv[5]) if len(sys.argv) > 5 else 0
NSAMP = 1 << LOGN
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
PROBE = [7, 15, 23, 30]
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def mehler(mu, s, rho):
    """(E[relu], Cov(relu)) для гаусового вектора з (mu, s, rho). Ряд Мелера."""
    t = mu / s
    P, f = Phi(t), phi(t)
    ma = s * (f + t * P)
    a1 = s * P
    cov = torch.outer(a1, a1) * rho
    Rn = rho.clone()
    Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
    fact, sf = 2.0, s * f
    for nn in range(2, MK + 1):
        Rn = Rn * rho
        an = sf * Hm / fact
        cov = cov + torch.outer(an, an) * Rn * fact
        Hpre, Hm = Hm, (-t) * Hm - float(nn - 2) * Hpre
        fact *= (nn + 1)
    return ma, cov


def analyse(h, Wn, n, Rext=None, kgrid=KGRID):
    dev = h.device
    z64 = dict(dtype=torch.float64, device=dev)
    D = h.shape[1]
    # ── істина: m, Cov від p та від relu(p)
    sp = torch.zeros(D, **z64)
    spp = torch.zeros(D, D, **z64)
    sh = torch.zeros(D, **z64)
    shh = torch.zeros(D, D, **z64)
    smh = torch.zeros(D, **z64)
    for o in range(0, n, CH):
        hb = h[o:o + CH].double()
        pb = hb @ Wn
        rb = pb.clamp_min(0.0)
        sp += pb.sum(0); spp += pb.T @ pb
        sh += rb.sum(0); shh += rb.T @ rb
        smh += hb.sum(0)
    mu = sp / n
    Cp = spp / n - torch.outer(mu, mu)
    m_t = sh / n
    C_t = shh / n - torch.outer(m_t, m_t)
    mh = smh / n
    s = Cp.diagonal().clamp_min(1e-300).sqrt()
    rho = (Cp / torch.outer(s, s)).clamp(-1.0, 1.0)

    # ── (i) безумовний Мелер = звичайне гаусове замикання
    m_g, C_g = mehler(mu, s, rho)

    # ── канал R
    u = mh / mh.norm().clamp_min(1e-300)
    c1 = u @ Wn
    mu_u = float(mh @ u)
    R = torch.empty(n, **z64)
    for o in range(0, n, CH):
        R[o:o + CH] = h[o:o + CH].double() @ u - mu_u
    Rsrc = R if Rext is None else (Rext[:n] - Rext[:n].mean())
    # умовна коваріація залишку: Cov(g) = Cp − Var(R)·c1c1ᵀ
    vR = float((Rsrc * Rsrc).mean())
    Cg = Cp - vR * torch.outer(c1, c1)
    sg = Cg.diagonal().clamp_min(1e-12).sqrt()
    rg = (Cg / torch.outer(sg, sg)).clamp(-1.0, 1.0)
    # сітка: стратифіковані середні квантилів емпіричного R
    Rs, _ = torch.sort(Rsrc)
    edges = torch.linspace(0, n, kgrid + 1, device=dev).long()
    grid = torch.stack([Rs[edges[k]:edges[k + 1]].mean()
                        for k in range(kgrid)])
    w = 1.0 / kgrid
    # ── (ii) умовна схема
    m_c = torch.zeros(D, **z64)
    C_in = torch.zeros(D, D, **z64)
    MS = []
    for k in range(kgrid):
        mk, ck = mehler(mu + grid[k] * c1, sg, rg)
        MS.append(mk)
        m_c += w * mk
        C_in += w * ck
    MS = torch.stack(MS)
    dm = MS - m_c
    C_c = C_in + (dm.T * w) @ dm

    def frel(A, B):
        return float((A - B).norm() / B.norm())
    off = ~torch.eye(D, dtype=torch.bool, device=dev)
    return dict(
        diag_g=frel(C_g.diagonal(), C_t.diagonal()),
        diag_c=frel(C_c.diagonal(), C_t.diagonal()),
        off_g=frel(C_g[off], C_t[off]),
        off_c=frel(C_c[off], C_t[off]),
        full_g=frel(C_g, C_t), full_c=frel(C_c, C_t),
        m_g=frel(m_g, m_t), m_c=frel(m_c, m_t),
        sig_g=frel(C_g.diagonal().clamp_min(1e-300).sqrt(),
                   C_t.diagonal().clamp_min(1e-300).sqrt()),
        sig_c=frel(C_c.diagonal().clamp_min(1e-300).sqrt(),
                   C_t.diagonal().clamp_min(1e-300).sqrt()),
    ), R


print(f"newalg95c · {DEV} · {NNET} мереж · N=2^{LOGN} · K={KGRID} · SPHERE={SPHERE}",
      flush=True)
t0 = time.time()
RES, RES1, RESX = [], [], []
prevR = {}
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        g = torch.Generator(device=DEV).manual_seed(952000 + 149 * q)
        X = torch.randn(NSAMP, 256, generator=g, device=DEV)
        if SPHERE:
            X = X * (16.0 / X.norm(dim=1, keepdim=True))
        row, row1, rowx = {}, {}, {}
        for L in range(32):
            X = torch.relu(X @ W[L])
            if L in PROBE and L + 1 < 32:
                Wn = W[L + 1].double()
                a, R = analyse(X, Wn, NSAMP)
                row[L] = a
                # 🔒 пломба: K=1 мусить відтворити гаусову
                a1, _ = analyse(X, Wn, NSAMP, kgrid=1)
                row1[L] = a1
                # ⛔ чужий R
                if L in prevR and prevR[L].numel() >= NSAMP:
                    ax, _ = analyse(X, Wn, NSAMP, Rext=prevR[L])
                    rowx[L] = ax
                prevR[L] = R
                torch.cuda.empty_cache()
        RES.append(row); RES1.append(row1); RESX.append(rowx)
        print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)


def avg(store, L, key):
    v = [r[L][key] for r in store if L in r]
    return float(np.mean(v)) if v else float('nan')


print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 84)
print("M1/M2 — ПОХИБКА Cov(relu(p)): ГАУСОВЕ ЗАМИКАННЯ проти УМОВНОГО ПО R")
print("=" * 84)
print(f"{'шар':>4s}{'σ гаус':>10s}{'σ умовн':>10s}{'×σ':>7s}"
      f"{'діаг г':>9s}{'діаг у':>9s}{'×діаг':>7s}"
      f"{'поза г':>9s}{'поза у':>9s}{'×поза':>7s}")
for L in PROBE:
    sg, sc = avg(RES, L, 'sig_g'), avg(RES, L, 'sig_c')
    dg, dc = avg(RES, L, 'diag_g'), avg(RES, L, 'diag_c')
    og, oc = avg(RES, L, 'off_g'), avg(RES, L, 'off_c')
    print(f"{L:>4d}{sg:>10.5f}{sc:>10.5f}{sg/max(sc,1e-12):>7.2f}"
          f"{dg:>9.5f}{dc:>9.5f}{dg/max(dc,1e-12):>7.2f}"
          f"{og:>9.5f}{oc:>9.5f}{og/max(oc,1e-12):>7.2f}")

print("\nM3 — для довідки, перший момент E[relu(p)] (з newalg95b очікуємо ×1.0)")
for L in PROBE:
    mg, mc = avg(RES, L, 'm_g'), avg(RES, L, 'm_c')
    print(f"  шар {L:>2d}: гаусова {mg:.5f} · умовна {mc:.5f} · "
          f"× {mg/max(mc,1e-12):.3f}")

print("\n🔒 ПЛОМБА K=1 (мусить збігтись із гаусовою до ~1e-3 відносно)")
for L in PROBE:
    print(f"  шар {L:>2d}: σ гаус {avg(RES,L,'sig_g'):.6f} · "
          f"σ умовна@K=1 {avg(RES1,L,'sig_c'):.6f}")

print("\n⛔ КОНТРОЛЬ «ЧУЖИЙ R» (виграш мусить зникнути)")
for L in PROBE:
    sx = avg(RESX, L, 'sig_c')
    sg = avg(RES, L, 'sig_g')
    print(f"  шар {L:>2d}: ×σ своя {sg/max(avg(RES,L,'sig_c'),1e-12):.3f} · "
          f"×σ чужа {sg/max(sx,1e-12):.3f}")
