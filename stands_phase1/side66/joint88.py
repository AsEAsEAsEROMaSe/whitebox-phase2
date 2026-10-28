"""ТРИ КАНАЛИ ЗАМИКАННЯ ПРАЦЮЮТЬ ЛИШЕ РАЗОМ. ЩО З ЦЬОГО МОЖНА ВИВЧИТИ? (С88)

## 🚨 ЩО ПОКАЗАВ `dag88` (40 мереж, ОРАКУЛЬНІ поправки, точні за побудовою)
```
база                        0.4809% після α          x1.000
оракул ТІЛЬКИ m_a           0.5036%                  x0.912   ← ГІРШЕ
оракул ТІЛЬКИ діаг C_a      0.9763%                  x0.243   ← НАБАГАТО ГІРШЕ
оракул ТІЛЬКИ позадіаг C_a  0.6040%                  x0.634   ← ГІРШЕ
оракул C_a повний           0.5650%                  x0.725   ← ГІРШЕ
оракул m_a + C_a (СТАН)     0.1137%                 x17.898   ← 🎯
P1 оракул УСЕ + вихід       0.0240%                x400.850   ✅пломба
```
⟹ **ПОХИБКИ ТРЬОХ КАНАЛІВ ВЗАЄМНО СКОРОЧУЮТЬСЯ В РЕКУРСІЇ.** Виправлення
ОДНОГО каналу — навіть ТОЧНЕ, з `R²=1` — ЛАМАЄ скорочення і робить гірше.
Це закриває гілку «вивчена `F(t_j,t_k,ρ)` до позадіагоналі» ЗА ПОБУДОВОЮ:
її стеля = ×0.634, тобто від'ємна. Жодне навчання цього не зрушить.

## 📐 НОВА ПОСТАНОВКА
Мішень — не позадіагональ, а **ТРІЙКА разом**:
```
δm_j  = (E[relu(h_j)]     − m_j Φ_j − s_j φ_j) / s_j          ознака: t_j
δd_j  = (Var(relu(h_j))   − гаусова)          / s_j²          ознака: t_j
δC_jk = (Cov(relu h_j,h_k)− гаусова)          / (s_j s_k)     ознаки: t_j,t_k,ρ
```
Оракульна трійка дає **0.1137%**, а поріг `sub97` = 0.1456%, лідера = 0.0760%
⟹ навіть повний оракул стану НЕ б'є лідера, але б'є наш рекорд.

## ❓ ПИТАННЯ ЦЬОГО СТЕНДУ
1. Який `R²` мають `δm(t)` і `δd(t)` — понейронні канали? (історія каже ≈0)
2. Наскільки ТОЧНИМИ мусять бути канали: оракул × (1+шум) при різному шумі.
3. Чи виживає виграш, якщо вивчити ВСІ ТРИ канали таблицями?
4. Чи допомагає гібрид «оракул одного каналу + вивчений інший» — де вузько?

📣 ПЕРЕДБАЧЕННЯ:
 (а) `R²` понейронних каналів від `t` буде <0.35 (вони залежать від κ₃,κ₄
     конкретного нейрона, а не від `t`). Шанс 70%.
 (б) вимога до точності буде ЖОРСТКОЮ: 10% шуму на всіх каналах з'їсть
     більшу частину ×17.9. Шанс 75% (закон «важіль на дві точності = добуток»).
 (в) вивчена трійка дасть <×1.5. Шанс 65%.
"""
import glob, math, sys, os, time
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NLB = int(sys.argv[4]) if len(sys.argv) > 4 else 4
assert DEV not in ('cuda:0', 'cuda:4')
SP = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad'
CACHE = f'{SP}/mc88cache'
L_, W_ = 32, 256
KAPPA = 200.0
NB1 = 48               # дрібніша сітка для ПОНЕЙРОННИХ каналів (дешеві, 1-D)
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
print(__doc__.split('📣')[1], flush=True)

WZ = np.load(f'{SP}/fs_w.npy', mmap_mode='r')
files = sorted(glob.glob(f'{CACHE}/n*.npz'))[:NETS]
N = len(files)
IDXW = [int(f.rsplit('/', 1)[1][1:4]) for f in files]
print(f'мереж: {N} · сітка {NB}^3 · 1-D {NB1} · шар-бакетів {NLB} · {DEV}\n', flush=True)
TRUE_ = np.zeros((N, W_))
LBL = [min((q * NLB) // (L_ - 1), NLB - 1) for q in range(L_ - 1)]
NBIN = NLB * NB ** 3
NBIN1 = NLB * NB1
EYE = torch.eye(W_, dtype=torch.bool, device=DEV)
OFF = ~EYE


def gauss_step(m, C):
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s; P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
    q2 = (m * m + s * s) * P + (m * s) * f
    Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30) - torch.diagonal(Ca))
    return m_a, Ca, s, t, C / torch.outer(s, s)


def i1(t, q):
    return torch.clamp(torch.bucketize(t, ED_T1), 0, NB1 - 1) + LBL[q] * NB1


def i3(t, rho, q):
    a = torch.clamp(torch.bucketize(t, ED_T), 0, NB - 1)
    c = torch.clamp(torch.bucketize(rho, ED_R), 0, NB - 1)
    A = a[:, None].expand(W_, W_); B = a[None, :].expand(W_, W_)
    return ((torch.minimum(A, B) * NB + torch.maximum(A, B)) * NB + c
            + LBL[q] * NB ** 3)


def chain(W, ORA=None, ora=(0, 0, 0, 0), noise=0.0, gen=None,
          FM=None, FD=None, FY=None, wm=1.0, wd=1.0, wy=1.0,
          acc=None, targ=None):
    """ora=(m_a, діаг, позадіаг, вихід) — оракул. FM/FD/FY — вивчені таблиці."""
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(1, L_ + 1):
        m_a, Ca, s, t, rho = gauss_step(m, C)
        q = i - 1
        if i == L_:
            if ORA is not None and ora[3]:
                m_a = m_a + ORA[2][q] * s
            elif FM is not None and ora[3]:
                m_a = m_a + wm * FM[i1(t, L_ - 2)] * s
            return m_a
        oo = torch.outer(s, s)
        nz = (lambda x: x * (1.0 + noise * torch.randn(x.shape, dtype=x.dtype,
                                                       device=x.device, generator=gen))
              ) if noise > 0 else (lambda x: x)
        if ORA is not None:
            if ora[0]:
                m_a = m_a + nz(ORA[2][q]) * s
            if ora[1] or ora[2]:
                cr = nz(ORA[0][q]) * oo
                if not ora[2]:
                    cr = torch.diag(torch.diagonal(cr))
                elif not ora[1]:
                    cr = cr - torch.diag(torch.diagonal(cr))
                Ca = Ca + cr
        if FM is not None or FD is not None or acc is not None:
            j1 = i1(t, q)
        if FM is not None:
            m_a = m_a + wm * FM[j1] * s
        if FY is not None or acc is not None:
            j3 = i3(t, rho, q)
        if acc is not None:
            acc[0].scatter_add_(0, j3[OFF], targ[0][q][OFF])
            acc[1].scatter_add_(0, j3[OFF], torch.ones_like(targ[0][q][OFF]))
            acc[2].scatter_add_(0, j1, targ[1][q])
            acc[3].scatter_add_(0, j1, torch.ones_like(targ[1][q]))
            acc[4].scatter_add_(0, j1, targ[2][q])
        if FY is not None:
            cr = wy * FY[j3] * oo
            Ca = Ca + (cr - torch.diag(torch.diagonal(cr)))
        if FD is not None:
            Ca = Ca + torch.diag(wd * FD[j1] * s * s)
        m = m_a @ W[i]; C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


NETW = []
tpool, rpool = [], []
for n, f in enumerate(files):
    d = np.load(f)
    TRUE_[n] = d['out']
    W = [torch.tensor(np.asarray(WZ[IDXW[n]][q], np.float64), device=DEV) for q in range(L_)]
    mh = torch.tensor(d['mh'], dtype=torch.float64, device=DEV)
    ma = torch.tensor(d['ma'], dtype=torch.float64, device=DEV)
    Ch = torch.tensor(d['Ch'], dtype=torch.float64, device=DEV)
    Ca = torch.tensor(d['Ca'], dtype=torch.float64, device=DEV)
    dY, dD, dM, TT = [], [], [], []
    for q in range(L_):
        m_a, Cg, s, t, rho = gauss_step(mh[q], Ch[q])
        dM.append((ma[q] - m_a) / s)
        if q < L_ - 1:
            Yf = (Ca[q] - Cg) / torch.outer(s, s)
            dY.append(Yf); dD.append(torch.diagonal(Yf).clone()); TT.append((t, rho))
            if n < 8 and q % 3 == 0:
                tpool.append(t.cpu().numpy()); rpool.append(rho[OFF].cpu().numpy())
    NETW.append((W, dY, dD, dM, TT))
    del mh, ma, Ch, Ca
tp = np.concatenate(tpool); rp = np.concatenate(rpool)
ED_T = torch.tensor(np.quantile(tp, np.linspace(0, 1, NB + 1))[1:-1], dtype=torch.float64, device=DEV)
ED_R = torch.tensor(np.quantile(rp, np.linspace(0, 1, NB + 1))[1:-1], dtype=torch.float64, device=DEV)
ED_T1 = torch.tensor(np.quantile(tp, np.linspace(0, 1, NB1 + 1))[1:-1], dtype=torch.float64, device=DEV)
del tpool, rpool, tp, rp
print(f'завантажено · GPU {torch.cuda.memory_allocated(DEV)/2**30:.2f} GiB', flush=True)

# ---------- off-policy акумулятори всіх трьох каналів ----------
AY = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
CY = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
AD = torch.zeros(N, NBIN1, dtype=torch.float64, device=DEV)
CD = torch.zeros(N, NBIN1, dtype=torch.float64, device=DEV)
AM = torch.zeros(N, NBIN1, dtype=torch.float64, device=DEV)
with torch.no_grad():
    for n in range(N):
        _, dY, dD, dM, TT = NETW[n]
        for q in range(L_ - 1):
            j3 = i3(TT[q][0], TT[q][1], q); j1 = i1(TT[q][0], q)
            AY[n].scatter_add_(0, j3[OFF], dY[q][OFF])
            CY[n].scatter_add_(0, j3[OFF], torch.ones_like(dY[q][OFF]))
            AD[n].scatter_add_(0, j1, dD[q])
            CD[n].scatter_add_(0, j1, torch.ones_like(dD[q]))
            AM[n].scatter_add_(0, j1, dM[q])
print('таблиці зібрані', flush=True)

NRM = (TRUE_ ** 2).mean()
RES = {}


def tb(S, C_, n, kap=KAPPA):
    msk = torch.ones(S.shape[0], dtype=torch.bool, device=DEV)
    if n is not None:
        msk[n] = False
    return S[msk].sum(0) / (C_[msk].sum(0) + kap)


def am(Pm):
    pt = (Pm * TRUE_).sum(1); pp = (Pm * Pm).sum(1); tt = (TRUE_ * TRUE_).sum(1)
    a_n = pt / np.maximum(pp, 1e-30); out = np.zeros(N)
    for i in range(N):
        a = a_n[np.arange(N) != i].mean()
        out[i] = (a * a * pp[i] - 2 * a * pt[i] + tt[i]) / W_
    return out.mean()


def run(tag, tabs=None, **kw):
    Pm = np.zeros((N, W_))
    with torch.no_grad():
        for n in range(N):
            kk = dict(kw)
            if 'ORA' in kk:
                kk['ORA'] = (NETW[n][1], NETW[n][2], NETW[n][3])
            if tabs:
                for k, fn in tabs.items():
                    kk[k] = fn(n)
            Pm[n] = chain(NETW[n][0], **kk).cpu().numpy()
    RES[tag] = (math.sqrt(((Pm - TRUE_) ** 2).mean() / NRM), am(Pm))


# ---------- 1) R² ПОНЕЙРОННИХ КАНАЛІВ (LOO по мережах, бінінг по t) ----------
print(f'\n{"="*76}\n1) R² ПОНЕЙРОННИХ КАНАЛІВ ВІД `t` (LOO по мережах, {NB1} бінів × {NLB} бакетів)')
for nm, A, C_ in (('δd (діаг C_a)', AD, CD), ('δm (перший момент)', AM, CD)):
    yy, pp = [], []
    for n in range(N):
        F = tb(A, C_, n, kap=0.0)
        _, dY, dD, dM, TT = NETW[n]
        for q in range(L_ - 1):
            j1 = i1(TT[q][0], q)
            tgt = dD[q] if 'δd' in nm else dM[q]
            yy.append(tgt.cpu().numpy()); pp.append(F[j1].cpu().numpy())
    yy = np.concatenate(yy); pp = np.concatenate(pp)
    r2 = 1.0 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    print(f'  {nm:22s} RMS {yy.std():.4e} · R² {r2:+.4f}')
# R² позадіагоналі — для порівняння в ТІЙ САМІЙ таблиці
yy, pp = [], []
for n in range(N):
    F = tb(AY, CY, n, kap=0.0)
    _, dY, dD, dM, TT = NETW[n]
    for q in range(0, L_ - 1, 3):
        j3 = i3(TT[q][0], TT[q][1], q)
        yy.append(dY[q][OFF].cpu().numpy()); pp.append(F[j3[OFF]].cpu().numpy())
yy = np.concatenate(yy); pp = np.concatenate(pp)
print(f'  {"δC (позадіагональ)":22s} RMS {yy.std():.4e} · '
      f'R² {1.0-((yy-pp)**2).sum()/((yy-yy.mean())**2).sum():+.4f}')

# ---------- 2) ЧУТЛИВІСТЬ ОРАКУЛА ДО ШУМУ ----------
t0 = time.time()
run('база')
run('оракул СТАН (m+C)', ORA=1, ora=(1, 1, 1, 0))
g = torch.Generator(device=DEV); g.manual_seed(4242)
for nz in (0.05, 0.1, 0.2, 0.4, 0.8):
    run(f'оракул СТАН + шум {nz:.0%}', ORA=1, ora=(1, 1, 1, 0), noise=nz, gen=g)
print(f'чутливість {time.time()-t0:.0f}s', flush=True)

# ---------- 3) ВИВЧЕНА ТРІЙКА ----------
TM = {'FM': lambda n: tb(AM, CD, n), 'FD': lambda n: tb(AD, CD, n),
      'FY': lambda n: tb(AY, CY, n)}
for w in (0.25, 0.5, 0.75, 1.0):
    run(f'вивчена ТРІЙКА w={w:g}', tabs=TM, wm=w, wd=w, wy=w)
run('вивчена ТРІЙКА + вихід w=1', tabs=TM, wm=1.0, wd=1.0, wy=1.0, ora=(0, 0, 0, 1))
# гібриди: де вузьке місце
run('оракул m + вивч. d,C', ORA=1, ora=(1, 0, 0, 0),
    tabs={'FD': TM['FD'], 'FY': TM['FY']})
run('оракул C(діаг+поза) + вивч. m', ORA=1, ora=(0, 1, 1, 0), tabs={'FM': TM['FM']})
run('оракул m,діаг + вивч. поза', ORA=1, ora=(1, 1, 0, 0), tabs={'FY': TM['FY']})
run('оракул m,поза + вивч. діаг', ORA=1, ora=(1, 0, 1, 0), tabs={'FD': TM['FD']})
# контролі
rng = np.random.default_rng(11)
p3 = torch.tensor(rng.permutation(NBIN), device=DEV)
p1 = torch.tensor(rng.permutation(NBIN1), device=DEV)
run('⛔перемішана ТРІЙКА w=0.5', tabs={'FM': lambda n: tb(AM, CD, n)[p1],
                                      'FD': lambda n: tb(AD, CD, n)[p1],
                                      'FY': lambda n: tb(AY, CY, n)[p3]},
    wm=0.5, wd=0.5, wy=0.5)
run('⛔знак −ТРІЙКА w=0.5', tabs={'FM': lambda n: -tb(AM, CD, n),
                                 'FD': lambda n: -tb(AD, CD, n),
                                 'FY': lambda n: -tb(AY, CY, n)},
    wm=0.5, wd=0.5, wy=0.5)
run('⛔чужа ТРІЙКА (1 мережа) w=0.5',
    tabs={'FM': lambda n: AM[(n + 1) % N] / (CD[(n + 1) % N] + KAPPA),
          'FD': lambda n: AD[(n + 1) % N] / (CD[(n + 1) % N] + KAPPA),
          'FY': lambda n: AY[(n + 1) % N] / (CY[(n + 1) % N] + KAPPA)},
    wm=0.5, wd=0.5, wy=0.5)
print(f'вивчена трійка {time.time()-t0:.0f}s', flush=True)

b = RES['база'][1]
print(f'\n{"="*84}\nРЕЗУЛЬТАТ · {N} мереж · сітка {NB}^3 · 1-D {NB1} · бакетів {NLB}')
print(f'{"варіант":34s} {"сира":>10s} {"після α":>10s} {"MSE після α":>13s} {"vs база":>9s}')
for k, (e, a) in RES.items():
    print(f'{k:34s} {e:9.4%} {math.sqrt(a/NRM):9.4%} {a:13.4e} {"x%.3f" % (b/a):>9s}')
print(f'\n🔧 RMS(істина)² = {NRM:.5f} · поріг sub97 0.1456%, лідер 0.0760%')
