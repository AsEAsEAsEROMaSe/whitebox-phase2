"""lamt90 — λ, ЗАЛЕЖНА ВІД НЕЙРОНА: λ(t_j) замість одного скаляра.

════════════════ ІДЕЯ ЮЗЕРА (С90) ════════════════
«ваги можуть підказати досить точно яка буде форма, а замикання і МК дають
форму, і з цього бульйону можна витягнути суть».

Наш бленд зараз: `pred = w·closure + (1−w)·MC` з ОДНИМ `w=0.0883` на всю
мережу. Але замикання ВЖЕ видає `t_j = m_j/s_j` понейронно — тобто саме каже,
де воно надійне (великий `t` ⟹ нейрон майже завжди активний ⟹ гаусова
формула точна), а де ні. Ми цю підказку викидаємо.

Це ЄДИНИЙ пункт, позначений у пам'яті як «НЕ ПРОБУВАНО».

════════════════ СХЕМА ════════════════
Бінуємо нейрони по `t_j` (з замикання), у кожному біні фітимо свою `λ_b`:
    pred_j = λ_b·closure_j + (1−λ_b)·MC_j
🚨 LOO ПО МЕРЕЖАХ — інакше фітимо шум (наш закон: LOO-α на зрізі з N мереж
рахується по N−1, і мала вибірка перевертає знак).
🚨 КРОС-ЦІЛЬ: істина з банку ШУМНА, тому дві незалежні реалізації МК як
ціль/оцінка не годяться — беру істину з ВЕЛИКОГО N і окремим сідом.

════════════════ ПЛОМБИ ════════════════
P1 ОДИН БІН мусить відтворити глобальний бленд (це той самий об'єкт).
P2 λ мусить лежати в [0,1] у кожному біні; вихід за межі = ознака фітингу шуму.
P3 КОНТРОЛЬ «ЧУЖА ОЗНАКА»: бінувати по ВИПАДКОВОМУ числу замість `t_j`.
   Виграш там мусить бути ×1.00. Якщо чужа дає стільки ж — весь ефект є
   ефектом зайвих ступенів свободи, а не інформації.

════════════════ ПЕРЕДБАЧЕННЯ ════════════════
 (а) λ справді залежатиме від `t`: при великому `t` замикання точніше ⟹
     λ мусить РОСТИ з `t` (шанс 70%);
 (б) виграш LOO ×1.03…1.15 (шанс 45%);
 (в) чужа ознака дасть ≤×1.01 (шанс 75%);
 🎯 шанс на >×1.10 із чистим контролем: **40%**.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NMC = 5432
NTRUTH = 1 << 20
NBIN = int(sys.argv[3]) if len(sys.argv) > 3 else 6
DT = torch.float32
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · бінів {NBIN} · МК {NMC} · істина {NTRUTH}\n")


def closure(ws):
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = torch.eye(n, dtype=torch.float64, device=DEV)
    for i in range(L):
        w = ws[i].double()
        mh = m @ w
        Ch = w.T @ (C @ w)
        d = torch.diagonal(Ch).clamp(min=1e-30)
        s = d.sqrt()
        t = mh / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        ma = s * (t * p + f)
        if i == L - 1:
            return ma, t
        va = ((mh * mh + d) * p + mh * s * f - ma * ma).clamp(min=1e-30)
        Ca = Ch * torch.outer(p, p)
        C = Ca + torch.diag(va - torch.diagonal(Ca))
        m = ma


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
    return fwd(ws, z @ torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT)).mean(0).double()


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


CL, MC, TR, TT, RND = [], [], [], [], []
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    c, t = closure(ws)
    CL.append(c.cpu().numpy()); TT.append(t.cpu().numpy())
    MC.append(mc(ws, 6000 + q).cpu().numpy())
    TR.append(truth(ws, 3000 + 17 * q).cpu().numpy())
    RND.append(np.random.default_rng(q).standard_normal(n))
    if q % 12 == 0:
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
CL, MC, TR, TT, RND = map(np.array, (CL, MC, TR, TT, RND))
print(f"\n  час {time.time()-t0:.0f} с\n")


def fit_lambda(idx_nets, bins, nb):
    """λ_b = argmin Σ (λ·c + (1−λ)·m − y)²  ⟹  λ = Σd(y−m)/Σd², d = c−m."""
    lam = np.zeros(nb)
    for b in range(nb):
        msk = bins[idx_nets] == b
        d = (CL[idx_nets] - MC[idx_nets])[msk]
        r = (TR[idx_nets] - MC[idx_nets])[msk]
        lam[b] = float((d * r).sum() / max((d * d).sum(), 1e-30))
    return lam


def evaluate(bins, nb):
    """LOO по мережах."""
    err = 0.0
    for q in range(NNET):
        oth = np.array([i for i in range(NNET) if i != q])
        lam = fit_lambda(oth, bins, nb)
        lj = lam[bins[q]]
        p = lj * CL[q] + (1 - lj) * MC[q]
        err += float(((p - TR[q]) ** 2).sum() / (TR[q] ** 2).sum())
    return err / NNET


def qbins(x, nb):
    e = np.quantile(x.reshape(-1), np.linspace(0, 1, nb + 1)[1:-1])
    return np.clip(np.searchsorted(e, x), 0, nb - 1)


one = np.zeros_like(TT, dtype=int)
e1 = evaluate(one, 1)
bt = qbins(TT, NBIN)
et = evaluate(bt, NBIN)
br = qbins(RND, NBIN)
er = evaluate(br, NBIN)
emc = float(np.mean([((MC[q] - TR[q]) ** 2).sum() / (TR[q] ** 2).sum()
                     for q in range(NNET)]))

print("=" * 88)
print(f"λ, ЗАЛЕЖНА ВІД НЕЙРОНА · {NNET} мереж · LOO по мережах")
print("=" * 88)
print(f"{'схема':>26s} {'віднос.похибка':>16s} {'× до глобальної λ':>19s}")
print(f"{'чиста МК (без бленду)':>26s} {100*emc**0.5:>15.4f}% {e1/emc:>19.4f}")
print(f"{'глобальна λ (сабміт)':>26s} {100*e1**0.5:>15.4f}% {'1.0000':>19s}")
print(f"{'λ(t_j), ' + str(NBIN) + ' бінів':>26s} {100*et**0.5:>15.4f}% {e1/et:>19.4f}")
print(f"{'⛔ ЧУЖА (випадкові біни)':>26s} {100*er**0.5:>15.4f}% {e1/er:>19.4f}")

lam_all = fit_lambda(np.arange(NNET), bt, NBIN)
lam1 = fit_lambda(np.arange(NNET), one, 1)
print(f"\n⛔ P1 один бін: λ = {lam1[0]:.4f}  (сабміт має 0.0883)")
print(f"⛔ P2 λ по бінах t: " + " · ".join(f"{v:.4f}" for v in lam_all))
tm = [float(TT.reshape(-1)[bt.reshape(-1) == b].mean()) for b in range(NBIN)]
print(f"   середнє t у біні:  " + " · ".join(f"{v:+.3f}" for v in tm))
print(f"   {'✅ усі в [0,1]' if lam_all.min() > -0.05 and lam_all.max() < 1.05 else '⚠️ ВИХІД ЗА [0,1] — фітинг шуму'}")
print(f"\n📣 ВИРОК: λ(t) ×{e1/et:.4f}  проти  ЧУЖОЇ ×{e1/er:.4f}")
print(f"   {'✅ ІНФОРМАЦІЯ, не калібровка' if e1/et > (e1/er) * 1.03 else '⛔ ВЕСЬ ЕФЕКТ — ЗАЙВІ СТУПЕНІ СВОБОДИ'}")
print(f"   ⟹ бал {1.86762e-07 / max(e1/et, 1e-9):.4e}  (рекорд 1.86762e-07)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/lamt90.npz",
         CL=CL, MC=MC, TR=TR, TT=TT, lam=lam_all, e1=e1, et=et, er=er, emc=emc)
