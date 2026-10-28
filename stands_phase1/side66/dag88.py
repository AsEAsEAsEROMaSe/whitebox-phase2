"""УЗГОДЖЕННЯ РЕЖИМІВ НАВЧАННЯ І ЗАСТОСУВАННЯ (DAgger) ДЛЯ `F(t_j,t_k,ρ)`. (С88)

## ЗВІДКИ
`learnc87`: `F` навчена на СВІЖІЙ похибці (вхід кроку = ІСТИННІ m,C), а
застосована В ЛАНЦЮГУ (вхід = власні спотворені m,C) ⟹ зсув розподілу ознак.
Сира похибка падає ×1.22, після `_ALPHA` виграш зникає.
`r2chk88` (40 мереж × 31 шар) ПІДТВЕРДИВ стелю: 0.695 (14³) / 0.727 (з бакетами
глибини) — тобто сигнал є і він трохи БІЛЬШИЙ за заявлений 0.666.

## 📐 ЩО ТУТ НОВОГО
1. **ON-POLICY БІНІНГ**: ціль лишається СВІЖОЮ (інакше ×0.009 з С87), але
   КООРДИНАТИ, за якими вона біниться, беруться з ЛАНЦЮГА. Ми вчимо
   `E[свіжа похибка | координати, якими БУДЕМО ЗАПИТУВАТИ]`.
2. **DAgger** (ітерація + агрегація датасету) і **scheduled sampling**
   (`p·істинні + (1−p)·ланцюгові` — це РІВНО опукла суміш акумуляторів).
3. **ПОШАРОВА `F`** (бакети глибини) + **КВАНТИЛЬНІ краї**.
4. **ДРАБИНА ОРАКУЛІВ** — де живе похибка ланцюга: `m_a` / діаг `C_a` /
   позадіаг `C_a` / вихідне відображення останнього шару.

## 🔑 ЧОМУ ПОЗИТИВНА ОДНОРІДНІСТЬ ОБМЕЖУЄ ВИГРАШ ON-POLICY
Якщо стан ланцюга відрізняється від істини СПІЛЬНИМ масштабом `c`
(`m→cm, C→c²C`), то `t` і `ρ` НЕ МІНЯЮТЬСЯ, а поправка `s_j s_k F` масштабується
рівно як `C_a` ⟹ чисто масштабний дрейф для `F` НЕВИДИМИЙ. Реальний зсув —
лише ФОРМА.

## ⛔ПЛОМБИ
 P0 `F≡0` ⟹ РІВНО база.
 P1 ТОЧНА свіжа поправка усіх каналів (разом із виходом) ⟹ ланцюг мусить
    відтворити ІСТИНУ з точністю МК-шуму (індукція по кроках).
 P2 LOO по мережах скрізь · P3 ⛔перемішана `F` · P4 ⛔`−F` ·
 P5 ⛔чужа `F` (з ОДНІЄЇ іншої мережі) · P6 ⚠️витік (F бачила свою мережу) —
    верхня межа якості таблиці.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) P1 (усі оракули) дасть похибку ≤2% від бази. Шанс 85%.
 (б) on-policy дасть у ЛАНЦЮГУ помітно більше за off-policy (≥×1.15 по MSE
     після α проти off-policy). Шанс 35% — однорідність каже, що зсув малий.
 (в) головний внесок у похибку ланцюга дає `m_a`, а не позадіагональ `C_a`
     (бо `G_m→вихід`=0.993). Шанс 45%.
 (г) сукупно гілка дасть ≥×1.5 по MSE замикання після α. Шанс 25%.
 (д) сира похибка бази вийде 0.90–0.98% (пам'ять: 0.9416%). Шанс 70%.
"""
import glob, math, sys, os, time
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 12
NLB = int(sys.argv[4]) if len(sys.argv) > 4 else 1
assert DEV not in ('cuda:0', 'cuda:4')
SP = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad'
CACHE = f'{SP}/mc88cache'
L_, W_ = 32, 256
KAPPA = 200.0          # усадка таблиці за наповненістю комірки
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
print(__doc__.split('📣')[1], flush=True)

WZ = np.load(f'{SP}/fs_w.npy', mmap_mode='r')
files = sorted(glob.glob(f'{CACHE}/n*.npz'))[:NETS]
N = len(files)
print(f'мереж: {N} · сітка {NB}^3 · шар-бакетів {NLB} · κ={KAPPA} · {DEV}\n', flush=True)
IDXW = [int(f.rsplit('/', 1)[1][1:4]) for f in files]
TRUE_ = np.zeros((N, W_))
LBL = [min((q * NLB) // (L_ - 1), NLB - 1) for q in range(L_ - 1)]
NBIN = NLB * NB ** 3
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


def bin_idx(t, rho, q):
    a = torch.clamp(torch.bucketize(t, ED_T), 0, NB - 1)
    c = torch.clamp(torch.bucketize(rho, ED_R), 0, NB - 1)
    A = a[:, None].expand(W_, W_); B = a[None, :].expand(W_, W_)
    return ((torch.minimum(A, B) * NB + torch.maximum(A, B)) * NB + c
            + LBL[q] * NB ** 3)


def chain(W, F=None, wgt=1.0, ORA=None, ora=(0, 0, 0, 0), acc=None, targ=None):
    """ora = (m_a проміжні, діаг C_a, позадіаг C_a, m_a ОСТАННЬОГО кроку).
    acc — акумулятори (SUM,CNT) для ON-POLICY бінінгу свіжої цілі `targ`."""
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(1, L_ + 1):
        m_a, Ca, s, t, rho = gauss_step(m, C)
        q = i - 1
        if i == L_:
            if ORA is not None and ora[3]:
                m_a = m_a + ORA[2][q] * s
            return m_a
        oo = torch.outer(s, s)
        if ORA is not None:
            if ora[0]:
                m_a = m_a + ORA[2][q] * s
            if ora[1] or ora[2]:
                cr = ORA[0][q] * oo
                if not ora[2]:
                    cr = torch.diag(torch.diagonal(cr))
                elif not ora[1]:
                    cr = cr - torch.diag(torch.diagonal(cr))
                Ca = Ca + cr
        if F is not None or acc is not None:
            idx = bin_idx(t, rho, q)
        if acc is not None:
            acc[0].scatter_add_(0, idx[OFF], targ[q][OFF])
            acc[1].scatter_add_(0, idx[OFF], torch.ones_like(targ[q][OFF]))
        if F is not None:
            cr = wgt * F[idx] * oo
            Ca = Ca + (cr - torch.diag(torch.diagonal(cr)))
        m = m_a @ W[i]; C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


# ---------- завантаження + свіжі цілі ----------
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
        dM.append((ma[q] - m_a) / s)          # свіжа похибка ПЕРШОГО моменту
        if q < L_ - 1:
            Yf = (Ca[q] - Cg) / torch.outer(s, s)
            dY.append(Yf); dD.append(torch.diagonal(Yf).clone()); TT.append((t, rho))
            if n < 8 and q % 3 == 0:
                tpool.append(t.cpu().numpy()); rpool.append(rho[OFF].cpu().numpy())
    NETW.append((W, dY, dD, dM, TT))
    del mh, ma, Ch, Ca
print(f'завантажено · GPU {torch.cuda.memory_allocated(DEV)/2**30:.2f} GiB', flush=True)

tp = np.concatenate(tpool); rp = np.concatenate(rpool)
ED_T = torch.tensor(np.quantile(tp, np.linspace(0, 1, NB + 1))[1:-1], dtype=torch.float64, device=DEV)
ED_R = torch.tensor(np.quantile(rp, np.linspace(0, 1, NB + 1))[1:-1], dtype=torch.float64, device=DEV)
del tpool, rpool, tp, rp

# ---------- off-policy акумулятори (координати з ІСТИНИ) ----------
S0 = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
C0 = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
with torch.no_grad():
    for n in range(N):
        _, dY, _, _, TT = NETW[n]
        for q in range(L_ - 1):
            idx = bin_idx(TT[q][0], TT[q][1], q)
            S0[n].scatter_add_(0, idx[OFF], dY[q][OFF])
            C0[n].scatter_add_(0, idx[OFF], torch.ones_like(dY[q][OFF]))
print('off-policy таблиця зібрана', flush=True)

NRM = (TRUE_ ** 2).mean()
RES = {}


def tab(S, C_, n):
    msk = torch.ones(S.shape[0], dtype=torch.bool, device=DEV)
    if n is not None:
        msk[n] = False
    return S[msk].sum(0) / (C_[msk].sum(0) + KAPPA)


def am(Pm):
    pt = (Pm * TRUE_).sum(1); pp = (Pm * Pm).sum(1); tt = (TRUE_ * TRUE_).sum(1)
    a_n = pt / np.maximum(pp, 1e-30); out = np.zeros(N)
    for i in range(N):
        a = a_n[np.arange(N) != i].mean()
        out[i] = (a * a * pp[i] - 2 * a * pt[i] + tt[i]) / W_
    return out.mean()


def run(tag, Ff=None, **kw):
    Pm = np.zeros((N, W_))
    with torch.no_grad():
        for n in range(N):
            kk = dict(kw)
            if Ff is not None:
                kk['F'] = Ff(n)
            if 'ORA' in kk:
                kk['ORA'] = (NETW[n][1], NETW[n][2], NETW[n][3])
            Pm[n] = chain(NETW[n][0], **kk).cpu().numpy()
    RES[tag] = (math.sqrt(((Pm - TRUE_) ** 2).mean() / NRM), am(Pm))
    return RES[tag]


t0 = time.time()
run('база')
for nm, o in (('оракул m_a (крок)', (1, 0, 0, 0)),
              ('оракул діаг C_a', (0, 1, 0, 0)),
              ('оракул позадіаг C_a', (0, 0, 1, 0)),
              ('оракул C_a повний', (0, 1, 1, 0)),
              ('оракул m_a+C_a (стан)', (1, 1, 1, 0)),
              ('оракул лише ВИХІД', (0, 0, 0, 1)),
              ('P1 оракул УСЕ', (1, 1, 1, 1))):
    run(nm, ORA=1, ora=o)
print(f'драбина оракулів {time.time()-t0:.0f}s', flush=True)

WGTS = (0.25, 0.5, 0.75, 1.0)
rng = np.random.default_rng(11)
perm = torch.tensor(rng.permutation(NBIN), device=DEV)
for w in WGTS:
    run(f'off-policy w={w:g}', Ff=lambda n: tab(S0, C0, n), wgt=w)
run('⛔перемішана w=0.5', Ff=lambda n: tab(S0, C0, n)[perm], wgt=0.5)
run('⛔знак −F w=0.5', Ff=lambda n: -tab(S0, C0, n), wgt=0.5)
run('⛔чужа F (1 мережа) w=0.5',
    Ff=lambda n: S0[(n + 1) % N] / (C0[(n + 1) % N] + KAPPA), wgt=0.5)
run('⚠️ВИТІК (бачила свою) w=0.5', Ff=lambda n: tab(S0, C0, None), wgt=0.5)
print(f'off-policy готово {time.time()-t0:.0f}s', flush=True)

# ---------- DAgger: on-policy бінінг ----------
W_DAG = float(os.environ.get('WDAG', '0.5'))
CUR_S, CUR_C = S0, C0
AGG_S, AGG_C = S0.clone(), C0.clone()
for it in range(1, 4):
    S1 = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
    C1 = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
    with torch.no_grad():
        for n in range(N):
            chain(NETW[n][0], F=tab(CUR_S, CUR_C, n), wgt=W_DAG,
                  acc=(S1[n], C1[n]), targ=NETW[n][1])
    CUR_S, CUR_C = S1, C1
    AGG_S = AGG_S + S1; AGG_C = AGG_C + C1
    for w in WGTS:
        run(f'DAgger it{it} w={w:g}', Ff=lambda n: tab(CUR_S, CUR_C, n), wgt=w)
    run(f'DAgger it{it} АГРЕГ w={W_DAG:g}', Ff=lambda n: tab(AGG_S, AGG_C, n), wgt=W_DAG)
    for p in (0.25, 0.5):
        run(f'sched.samp p={p:g} it{it} w={W_DAG:g}',
            Ff=lambda n, p=p: tab(p * S0 + (1 - p) * CUR_S, p * C0 + (1 - p) * CUR_C, n),
            wgt=W_DAG)
    run(f'⛔DAgger it{it} перемішана', Ff=lambda n: tab(CUR_S, CUR_C, n)[perm], wgt=W_DAG)
    run(f'⛔DAgger it{it} знак −F', Ff=lambda n: -tab(CUR_S, CUR_C, n), wgt=W_DAG)
    print(f'DAgger it{it} {time.time()-t0:.0f}s', flush=True)

b = RES['база'][1]
print(f'\n{"="*84}\nРЕЗУЛЬТАТ · {N} мереж · сітка {NB}^3 · шар-бакетів {NLB} · κ={KAPPA} · w_dag={W_DAG}')
print(f'{"варіант":32s} {"сира":>10s} {"після α":>10s} {"MSE після α":>13s} {"vs база":>9s}')
for k, (e, a) in RES.items():
    print(f'{k:32s} {e:9.4%} {math.sqrt(a/NRM):9.4%} {a:13.4e} {"x%.3f" % (b/a):>9s}')
print(f'\n🔧 прилад: RMS(істина)² = {NRM:.5f} (пам\'ять: 0.88404)')
