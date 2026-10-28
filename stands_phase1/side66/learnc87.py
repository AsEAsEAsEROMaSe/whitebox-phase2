"""ВИВЧЕНА ПОПРАВКА ДО ЗАМИКАННЯ `δC_jk = F(t_j,t_k,ρ)`. (С87)

## ЗВІДКИ — ЛАНЦЮГ ВІДКРИТТІВ ЗА СЬОГОДНІ
1. Похибка кроку сидить у ПОЗАДІАГОНАЛІ (1.69% проти 0.23%) ⟹ вона є
   властивістю ПАРИ, а не нейрона (питання ЮЗЕРА про кореляцію з вагами).
2. `pairscan87`: понейронні ознаки дають R²≈0, ПАРНІ — 0.238 лінійно.
3. 🔑НЕПАРАМЕТРИЧНА СТЕЛЯ від `(t_j,t_k,ρ)` = **0.666** (сітка 14³, LOO).
4. С57 мікромережею мала 0.82, але з `κ₃κ₄` З ІСТИНИ. ⟹ 66 з 82 пунктів
   доступні БЕЗКОШТОВНО: `t` і `ρ` ланцюг уже рахує на кожному кроці.

## 📐 ЩО РОБИТЬСЯ
Замикання: `C_a = Φ_jΦ_k C_jk + ½C²φφ/(ss)`. Додаємо вивчену поправку:
```
C_a ← C_a + s_j·s_k · F(t_j, t_k, ρ_jk)
```
`F` — таблиця на сітці квантилів (симетризована по `j↔k`), навчена ОФЛАЙН на
банку, LOO ПО МЕРЕЖАХ. У сабміт пішла б як `.npz` (0 флопів на завантаження).
💰ЦІНА: інтерполяція для `256²` пар × 32 шари ≈ 4e7 флопів = **0.015%** бюджету.

## ⛔ПЛОМБИ
 P1 `F ≡ 0` мусить дати РІВНО базу.
 P2 навчання ЛОО ПО МЕРЕЖАХ — модель ніколи не бачить свою мережу.
 P3 ⛔ПЕРЕМІШАНА `F` (таблиця з перемішаними комірками) мусить ПОГІРШИТИ.
 P4 ⛔`F` З ЧУЖОГО ШАРУ — мусить бути гіршою за свою.
 P5 α-калібровка LOO на кожній сходинці.

📣 ПЕРЕДБАЧЕННЯ:
 (а) поправка зменшить похибку кроку (свіжу) щонайменше в 1.4×. Шанс 65% —
     це прямий наслідок R²=0.666 (√(1−0.666)=0.578).
 (б) У ЛАНЦЮГУ (з накопиченням) виграш буде МЕНШИЙ, бо поправка сама шумна.
     Очікую ×1.3–2.0 у MSE після α. Шанс 40%.
 (в) виграш ≥×2. Шанс 25%.
 ⚠️РИЗИК: у ланцюгу `t,ρ` беруться зі СВОЇХ (неточних) `m,C`, а навчалась `F`
   на ІСТИННИХ — це зсув розподілу входів, класична пастка.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NB = 12
NS = 1 << LOG2N; BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · сітка {NB}^3 · {DEV}\n', flush=True)
ED_T = torch.linspace(-4.0, 4.0, NB + 1, dtype=torch.float64, device=DEV)
ED_R = torch.linspace(-0.6, 1.0, NB + 1, dtype=torch.float64, device=DEV)


def bin_idx(t, r):
    """Індекси комірки для всіх пар; симетризовано по j<->k."""
    a = torch.clamp(torch.bucketize(t, ED_T) - 1, 0, NB - 1)
    c = torch.clamp(torch.bucketize(r, ED_R) - 1, 0, NB - 1)
    A = a[:, None].expand(W_, W_); B = a[None, :].expand(W_, W_)
    lo = torch.minimum(A, B); hi = torch.maximum(A, B)
    return (lo * NB + hi) * NB + c


def collect_fresh(W, TRH, TRA, acc_sum, acc_cnt):
    """🚨 ЦІЛЬ — СВІЖА похибка замикання: на вхід кроку подаються ІСТИННІ
    `m_h, C_h` (з МК), і міряється `істинна C_a − мелерова C_a`.
    Раніше ціль бралася в ЛАНЦЮГУ, тобто містила НАКОПИЧЕНУ похибку, і
    поправка намагалась виправити все накопичення за крок ⟹ ×0.009."""
    for i in range(L_ - 1):
        mh, Ch = TRH[i]
        s_ = torch.sqrt(torch.clamp(torch.diagonal(Ch), min=1e-30))
        t = mh / s_; P, f = Phi(t), phi(t)
        m_a = mh * P + s_ * f
        Ca = Ch * torch.outer(P, P) + (0.5 * (Ch * Ch)) * torch.outer(f / s_, f / s_)
        q2 = (mh * mh + s_ * s_) * P + (mh * s_) * f
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30) - torch.diagonal(Ca))
        y = (TRA[i] - Ca) / torch.outer(s_, s_)
        idx = bin_idx(t, Ch / torch.outer(s_, s_))
        off = ~torch.eye(W_, dtype=torch.bool, device=Ch.device)
        acc_sum.scatter_add_(0, idx[off], y[off])
        acc_cnt.scatter_add_(0, idx[off], torch.ones_like(y[off]))


def chain(W, F=None, wgt=1.0, collect=None, TRU=None):
    """F — таблиця поправки (NB^3,) або None. collect — акумулятори навчання."""
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(1, L_ + 1):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        t = m / s; P, f = Phi(t), phi(t)
        m_a = m * P + s * f
        if i == L_:
            return m_a
        rho = C / torch.outer(s, s)
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        q2 = (m * m + s * s) * P + (m * s) * f
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30) - torch.diagonal(Ca))
        if F is not None:
            idx = bin_idx(t, rho)
            corr = wgt * F[idx] * torch.outer(s, s)
            corr = corr - torch.diag(torch.diagonal(corr))   # діагональ ТОЧНА
            Ca = Ca + corr
        m = m_a @ W[i]; C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


TRUEC, TRUEH, WLIST, TRUE_ = [], [], [], np.zeros((len(FS), W_))
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        SA = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        GA = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        SHh = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        GHh = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(8800 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]; hd = h.double()
                SHh[q] += hd.sum(0); GHh[q] += hd.T @ hd
                a = torch.relu(h); ad = a.double()
                SA[q] += ad.sum(0); GA[q] += ad.T @ ad
            acc += a.sum(0, dtype=torch.float64); d += b
        TRUE_[n] = (acc / NS).cpu().numpy()
        ma = SA / NS
        TRUEC.append([(GA[q] / NS - torch.outer(ma[q], ma[q])).cpu() for q in range(L_)])
        mhh = SHh / NS
        TRUEH.append([(mhh[q].cpu(),
                       (GHh[q] / NS - torch.outer(mhh[q], mhh[q])).cpu())
                      for q in range(L_)])
    WLIST.append(sd0)
    print(f'  МК net{n}', flush=True)

SUM = torch.zeros(len(FS), NB ** 3, dtype=torch.float64, device=DEV)
CNT = torch.zeros(len(FS), NB ** 3, dtype=torch.float64, device=DEV)
for n in range(len(FS)):
    W = [torch.tensor(np.asarray(WZ[have[WLIST[n]]][q], np.float64), device=DEV) for q in range(L_)]
    TRA = [x.to(DEV) for x in TRUEC[n]]
    TRH = [(a.to(DEV), b.to(DEV)) for a, b in TRUEH[n]]
    with torch.no_grad():
        collect_fresh(W, TRH, TRA, SUM[n], CNT[n])
print('таблиці зібрано', flush=True)

WGTS = (0.25, 0.5, 1.0)
TAGS = (['база'] + [f'F w={w:g}' for w in WGTS]
        + [f'⊥F w={w:g}' for w in WGTS]
        + ['⛔F перемішана w=0.5', '⛔F знак −F w=0.5'])
P = {t: np.zeros((len(FS), W_)) for t in TAGS}
rng = np.random.default_rng(11)
perm = torch.tensor(rng.permutation(NB ** 3), device=DEV)
for n in range(len(FS)):
    W = [torch.tensor(np.asarray(WZ[have[WLIST[n]]][q], np.float64), device=DEV) for q in range(L_)]
    msk = torch.ones(len(FS), dtype=torch.bool, device=DEV); msk[n] = False
    sm = SUM[msk].sum(0); cn = CNT[msk].sum(0)
    F = torch.where(cn > 50, sm / torch.clamp(cn, min=1.0), torch.zeros_like(sm))
    with torch.no_grad():
        base = chain(W)
        P['база'][n] = base.cpu().numpy()
        for w in WGTS:
            P[f'F w={w:g}'][n] = chain(W, F=F, wgt=w).cpu().numpy()
        # 🔑 ОРТОГОНАЛІЗАЦІЯ ДО БАЗИ: поправка покращує СИРУ похибку (×1.22),
        #    але конкурує з `_ALPHA` за масштаб і після α програє. Лишаємо
        #    ЛИШЕ ортогональну складову — те, чого один скаляр дати не може
        #    ([[finding-corrections-must-be-orthogonalized-against-alpha]]).
        den = float((base * base).sum())
        for w in WGTS:
            cor = chain(W, F=F, wgt=w) - base
            if den > 1e-30:
                cor = cor - base * (float((cor * base).sum()) / den)
            P[f'⊥F w={w:g}'][n] = (base + cor).cpu().numpy()
        P['⛔F перемішана w=0.5'][n] = chain(W, F=F[perm], wgt=0.5).cpu().numpy()
        P['⛔F знак −F w=0.5'][n] = chain(W, F=-F, wgt=0.5).cpu().numpy()
    print(f'  ланцюг net{n}', flush=True)


def am(t):
    pt = (P[t]*TRUE_).sum(1); pp = (P[t]*P[t]).sum(1); tt = (TRUE_*TRUE_).sum(1)
    a_n = pt/np.maximum(pp,1e-30); out=np.zeros(len(FS))
    for i in range(len(FS)):
        m_ = np.arange(len(FS))!=i
        a = a_n[m_].mean() if len(FS)>1 else a_n[i]
        out[i] = (a*a*pp[i]-2*a*pt[i]+tt[i])/W_
    return out.mean()


nrm = (TRUE_**2).mean()
b = am('база')
print(f'\n{"="*64}\nВИВЧЕНА ПОПРАВКА ДО ЗАМИКАННЯ, ПІСЛЯ α')
print(f'{"варіант":20s} {"віднос.":>9s} {"ПІСЛЯ α":>11s} {"vs база":>9s}')
for t in TAGS:
    e = math.sqrt(((P[t]-TRUE_)**2).mean()/nrm)
    print(f'{t:20s} {e:8.4%} {am(t):11.3e} {"x%.3f" % (b/am(t)):>9s}')
g = max(b/am(f'⊥F w={w:g}') for w in WGTS)
gp = b/am('⛔F перемішана w=0.5'); gs = b/am('⛔F знак −F w=0.5')
print(f'\nВИРОК: вивчена F дає x{g:.3f} · ⛔перемішана x{gp:.3f} · ⛔знак x{gs:.3f}')
print('   контроль пройдено' if g > 1.1 and g > 1.5*max(gp, gs) else '   контроль НЕ пройдено')
