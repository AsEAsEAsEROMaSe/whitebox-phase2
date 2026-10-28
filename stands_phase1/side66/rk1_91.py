"""rk1_91 — ПОНЕЙРОННИЙ СТАН `(m, d, v)` ЗАМІСТЬ МАТРИЦІ `C`.

════════════════ ЗМІНА КОНЦЕПЦІЇ, А НЕ ПОПРАВКА ════════════════
Уся ціна замикання сидить в `einsum("ij,ia,jb->ab", C, w, w)` — 50.3e6 флопів
на шар, 32 шари = 1.65e9 = 0.61% B. Але зі всієї матриці `C` споживається
ЛИШЕ понейронне: `s=√diag(cp)`, `m`, `q=diag(C)+m²`, `P=Φ(t)`, `s²φ`, `s²tφ`.
Матриця потрібна ВИКЛЮЧНО щоб порахувати наступну діагональ.

Заміна: `C ≈ diag(d) + v vᵀ` ⟹ стан = ТРИ вектори по 256:
    mh_a  = Σ_i m_i w_ia        ← редукція 1
    var_a = Σ_i d_i w_ia²       ← редукція 2
    u_a   = Σ_i v_i w_ia        ← редукція 3      Var(h_a) = var_a + u_a²
🏆 ТРИ РЕДУКЦІЇ НА НЕЙРОН НА ШАР — це РІВНО профіль лідера, зміряний у С88
   («F = три редукції/нейрон/шар, понейронні вектори, жодної матриці»).
Ціна ~7n² замість 50.3e6 ⟹ **×111 дешевше**.

Рекурсія рангу-1 (головний член Мелера для позадіагоналі):
    Ca_ab ≈ cp_ab·Φ_a·Φ_b,  cp_ab = u_a u_b  ⟹  v' = u⊙Φ   (знову ранг 1)
    d' = Var(a) − v'²        (діагональ лишається ТОЧНОЮ формулою)

════════════════ ЧОМУ Є ШАНС ════════════════
`chk90`: топ-1 власне число `C` = 55.82% сліду на шарі 32 — домінантна мода
не абстракція, це РАДІАЛЬНИЙ масштаб (`f(z)=|z|f(u)`, конуси С90).
Захист: замикання строго 1-однорідне ⟹ систематична помилка МАСШТАБУ від
рангового обрізання поглинається `_ALPHA` ЗА ПОБУДОВОЮ. Псується лише ФОРМА.

════════════════ ПЛОМБИ ════════════════
P1 FULL мусить відтворити сабмітний рівень замикання: відносна похибка
   `m_deep` після α-калібровки ≈ 0.49…0.60% (пам'ять: α-каліброване 0.4894–0.6013%).
P2 РОЗДІЛИТИ ПРЕДСТАВЛЕННЯ І НАКОПИЧЕННЯ. Одношаговий діагностик: узяти
   ІСТИННУ `C` з FULL, стиснути в `diag+ранг1` ОРАКУЛЬНО (топ-власпара), зробити
   ОДИН крок і звірити `diag(cp)` з FULL. Якщо вже тут провал — пропагувати
   нема сенсу, і це вирок ПРЕДСТАВЛЕННЯ, а не приладу.
P3 КОНТРОЛЬ «ЧУЖИЙ ВЕКТОР»: ранг-1 з ВИПАДКОВИМ `v` тієї ж норми. Мусить бути
   не кращим за DIAG. Якщо чужий дає стільки ж — виграє норма, а не напрямок.
P4 α-калібровка LOO по мережах (наш закон: LOO на зрізі з N рахується по N−1).

════════════════ ПЕРЕДБАЧЕННЯ (записано ДО запуску) ════════════════
 (а) DIAG провалиться, похибка ≥2× від FULL            — шанс 80%
 (б) RK1 (пропагований) у межах 1.5× від FULL          — шанс 45%
 (в) ОРАКУЛЬНИЙ ранг-1 за ОДИН крок дасть <5% похибки  — шанс 55%
 (г) чужий вектор буде не кращим за DIAG               — шанс 85%
 🎯 сумарний шанс, що `(m,d,v)` придатне як заміна хвоста: **40%**
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NMC = 5432
NTRUTH = 1 << 19
DT = torch.float32
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · ширина {n} · глибина {L} · істина {NTRUTH}\n",
      flush=True)


def _gauss(mp, s):
    """(Φ, φ, m_act, Var_act) з передактиваційних (mp, s)."""
    t = mp / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mp * p + s * f
    va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
    return p, f, ma, va


def full_closure(ws, keep_C=False):
    """РІВНО як `_closure_parts` сабміта (Мелер K=2, діагональ точна)."""
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = torch.eye(n, dtype=torch.float64, device=DEV)
    diags, Cs = [], []
    for i in range(L):
        w = ws[i].double()
        cp = w.T @ (C @ w)
        s = torch.diagonal(cp).clamp(min=1e-20).sqrt()
        mp = m @ w
        p, f, ma, va = _gauss(mp, s)
        diags.append(s.clone())
        if keep_C:
            Cs.append((C.clone(), w))
        m = ma
        if i == L - 1:
            break
        C = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        C = C + torch.diag(va - torch.diagonal(C))
    return m, diags, Cs


def dv_closure(ws, rank1=True, foreign=False, seed=0):
    """ПОНЕЙРОННИЙ стан. rank1=False ⟹ чиста діагональ (без `v`)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    d = torch.ones(n, dtype=torch.float64, device=DEV)
    v = torch.zeros(n, dtype=torch.float64, device=DEV)
    diags = []
    for i in range(L):
        w = ws[i].double()
        var = d @ (w * w)                 # редукція 2
        u = v @ w                          # редукція 3
        cpd = (var + u * u).clamp(min=1e-20)
        s = cpd.sqrt()
        mp = m @ w                         # редукція 1
        p, f, ma, va = _gauss(mp, s)
        diags.append(s.clone())
        m = ma
        if i == L - 1:
            break
        if rank1:
            vn = u * p
            if foreign:
                # ⛔ P3: той самий МАСШТАБ, випадковий НАПРЯМОК
                r = torch.randn(n, generator=g, device=DEV, dtype=torch.float64)
                vn = r * (vn.norm() / r.norm().clamp(min=1e-30))
            d = (va - vn * vn).clamp(min=0.0)
            v = vn
        else:
            d = va
    return m, diags


def onestep_rank(Cs, kmax=4):
    """⛔ P2: ПРЕДСТАВЛЕННЯ окремо від НАКОПИЧЕННЯ.

    Для кожного шару беремо ІСТИННУ `C`, стискаємо ОРАКУЛЬНО в `diag + ранг-k`
    і робимо ОДИН крок. Порівнюємо `diag(wᵀCw)` з істинним.
    """
    err = {k: [] for k in range(kmax + 1)}
    for C, w in Cs:
        true_d = torch.diagonal(w.T @ (C @ w))
        ev, U = torch.linalg.eigh(C)
        ev = ev.flip(0)
        U = U.flip(1)
        for k in range(kmax + 1):
            if k == 0:
                approx = torch.diagonal(C) @ (w * w)
            else:
                Vk = U[:, :k] * ev[:k].clamp(min=0).sqrt()
                dk = (torch.diagonal(C) - (Vk * Vk).sum(1)).clamp(min=0.0)
                uk = Vk.T @ w
                approx = dk @ (w * w) + (uk * uk).sum(0)
            err[k].append(float(((approx - true_d).abs() / true_d.abs().clamp(min=1e-30)).mean()))
    return {k: float(np.mean(v)) for k, v in err.items()}


def fwd(ws, z):
    x = z
    for i in range(L):
        h = x @ ws[i]
        x = torch.relu(h) if i < L - 1 else h
    return torch.relu(h)


def mc(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NMC // 2, n, generator=g, device=DEV, dtype=DT)
    z = torch.cat([z, -z], 0)
    cov = (z.T @ z).double() / z.shape[0]
    wh = torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT)
    return fwd(ws, z @ wh).mean(0).double()


def truth(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < NTRUTH:
        b = min(1 << 13, NTRUTH - done)
        acc += fwd(ws, torch.randn(b, n, generator=g, device=DEV,
                                   dtype=DT)).sum(0).double()
        done += b
    return acc / NTRUTH


VAR = ["FULL", "DIAG", "RK1", "RK1_чужий"]
CL = {k: [] for k in VAR}
SD = {k: [] for k in VAR}
MCv, TR, ONE = [], [], []
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    mf, sf, Cs = full_closure(ws, keep_C=(q < 8))
    CL["FULL"].append(mf.cpu().numpy()); SD["FULL"].append(
        torch.stack(sf).cpu().numpy())
    for name, kw in (("DIAG", dict(rank1=False)), ("RK1", dict(rank1=True)),
                     ("RK1_чужий", dict(rank1=True, foreign=True, seed=q))):
        md, sd = dv_closure(ws, **kw)
        CL[name].append(md.cpu().numpy()); SD[name].append(
            torch.stack(sd).cpu().numpy())
    if q < 8:
        ONE.append(onestep_rank(Cs))
    MCv.append(mc(ws, 6000 + q).cpu().numpy())
    TR.append(truth(ws, 3000 + 17 * q).cpu().numpy())
    if q % 8 == 0:
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
CL = {k: np.array(v) for k, v in CL.items()}
SD = {k: np.array(v) for k, v in SD.items()}
MCv, TR = np.array(MCv), np.array(TR)
print(f"\n  час {time.time()-t0:.0f} с\n", flush=True)


def alpha_loo(X):
    """α LOO по мережах: (1+α)·X ≈ TR, спільний скаляр."""
    num = (X * TR).sum(1)
    den = (X * X).sum(1)
    e = 0.0
    for q in range(NNET):
        o = np.arange(NNET) != q
        a = num[o].sum() / max(den[o].sum(), 1e-300)
        r = a * X[q] - TR[q]
        e += (r * r).sum() / (TR[q] ** 2).sum()
    return e / NNET, float(num.sum() / den.sum())


def blend_loo(X):
    """pred = MC + w·(a·X − MC); (w, a) фітяться LOO по мережах."""
    e = 0.0
    for q in range(NNET):
        o = np.arange(NNET) != q
        a = (X[o] * TR[o]).sum() / max((X[o] * X[o]).sum(), 1e-300)
        D = a * X[o] - MCv[o]
        R = TR[o] - MCv[o]
        wgt = (D * R).sum() / max((D * D).sum(), 1e-300)
        p = MCv[q] + wgt * (a * X[q] - MCv[q])
        e += ((p - TR[q]) ** 2).sum() / (TR[q] ** 2).sum()
    return e / NNET


emc = float(np.mean([((MCv[q] - TR[q]) ** 2).sum() / (TR[q] ** 2).sum()
                     for q in range(NNET)]))
print("=" * 96)
print(f"ПОНЕЙРОННИЙ СТАН (m,d,v) ПРОТИ ПОВНОЇ МАТРИЦІ · {NNET} мереж · LOO")
print("=" * 96)
print(f"{'варіант':>12s} {'похибка m_deep':>16s} {'α':>9s} "
      f"{'× до FULL':>11s} {'бленд+МК':>11s} {'бал ×':>9s}")
base = None
for k in VAR:
    ec, a = alpha_loo(CL[k])
    eb = blend_loo(CL[k])
    if base is None:
        base = eb
    print(f"{k:>12s} {100*ec**0.5:>15.4f}% {a-1:>+9.5f} "
          f"{(ec/alpha_loo(CL['FULL'])[0])**0.5:>11.4f} "
          f"{100*eb**0.5:>10.4f}% {base/eb:>9.4f}")
print(f"{'чиста МК':>12s} {'—':>16s} {'—':>9s} {'—':>11s} "
      f"{100*emc**0.5:>10.4f}% {base/emc:>9.4f}")

eF = alpha_loo(CL["FULL"])[0]
print(f"\n⛔ P1 FULL похибка {100*eF**0.5:.4f}% (пам'ять: 0.4894–0.6013%)  "
      f"{'✅' if 0.004 < eF**0.5 < 0.008 else '⚠️ ПОЗА ОЧІКУВАННЯМ'}")

if ONE:
    print(f"\n⛔ P2 ОДИН КРОК З ОРАКУЛЬНИМ СТИСКОМ (представлення без накопичення)")
    print(f"   {'ранг':>6s} {'сер.віднос.похибка diag(wᵀCw)':>32s}")
    for k in sorted(ONE[0]):
        print(f"   {k:>6d} {100*float(np.mean([o[k] for o in ONE])):>31.4f}%")

sf = SD["FULL"]
for k in VAR[1:]:
    r = np.abs(SD[k] - sf) / np.maximum(np.abs(sf), 1e-30)
    print(f"\n   {k:>10s}: похибка `s` шар1 {100*r[:, 0].mean():>7.3f}% · "
          f"шар8 {100*r[:, 7].mean():>7.3f}% · шар32 {100*r[:, -1].mean():>7.3f}%")

print("\n📣 ВИРОК")
eR = alpha_loo(CL["RK1"])[0]
eD = alpha_loo(CL["DIAG"])[0]
eX = alpha_loo(CL["RK1_чужий"])[0]
print(f"   RK1 / FULL = ×{(eR/eF)**0.5:.4f} по похибці   "
      f"(передбачення: <1.5 з шансом 45%)")
print(f"   RK1 проти DIAG = ×{(eD/eR)**0.5:.4f}   ⛔ЧУЖИЙ проти DIAG = "
      f"×{(eD/eX)**0.5:.4f}")
print(f"   {'✅ НАПРЯМОК НЕСЕ ІНФОРМАЦІЮ' if (eD/eR)**0.5 > (eD/eX)**0.5 * 1.05 else '⛔ ВИГРАЄ НОРМА, А НЕ НАПРЯМОК'}")
gain = base / blend_loo(CL["RK1"])
free = 23 * 50.3e6 / (0.09011 * 2.72e11)
print(f"\n   бленд на RK1 зберігає ×{gain:.4f} з бленду на FULL")
print(f"   звільнені флопи (хвіст 9..32): {23*50.3e6:.3e} = "
      f"{100*23*50.3e6/2.72e11:.3f}% B ⟹ +{100*free:.2f}% семплів МК")
print(f"   ⟹ очікуваний бал {1.86762e-07/(gain*(1+free)):.4e} "
      f"(рекорд 1.86762e-07)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/rk1_91.npz",
         TR=TR, MC=MCv, **{f"CL_{k}": CL[k] for k in VAR},
         **{f"SD_{k}": SD[k] for k in VAR})
