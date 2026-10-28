"""ЧОМУ `R²=0.70` НА ПАРАХ ДАЄ ×1.27 У ЛАНЦЮГУ. ПРОЄКЦІЙНИЙ `R²`. (С88)

## 📐 МЕХАНІЗМ, ЯКИЙ ПЕРЕВІРЯЄМО
Ланцюг НЕ споживає `δC_jk` поелементно. Він споживає ЛИШЕ квадратичну форму
```
δσ_i² = Σ_jk W_ji W_ki δC_jk       (похибка дисперсії наступного шару)
δm_i  = Σ_j  W_ji δm_j             (похибка середнього наступного шару)
```
Ваги `W` знакозмінні й нульового середнього ⟹ ПОЗАДІАГОНАЛЬНА сума майже
скорочується, а ДІАГОНАЛЬНА (`Σ_j W_ji² δC_jj`) — ні, там усі ваги додатні.
📌 ГІПОТЕЗА: поелементний `R²` міряє НЕ ТУ величину. Треба `R²` у проєкції.
Це пояснило б одразу все: `оракул m,поза + вивч. діаг` = ×7.34 (діагональ
конвертується), `оракул m,діаг + вивч. поза` = ×1.27 (позадіагональ — ні).

## ⛔ПЛОМБИ
 P1 тавтологія: підставити ІСТИННИЙ `δC` замість таблиці ⟹ R²_проєкц = 1.0000.
 P2 ⛔перемішана таблиця ⟹ R²_проєкц ≈ 0.
 P3 LOO по мережах.
 P4 БУТСТРЕП по мережах на головних рядках (60 мереж) — «малі прогони не
    розсуджують»: ×1.05 без інтервалу нічого не варте.

📣 ПЕРЕДБАЧЕННЯ:
 (а) R²_проєкц позадіагоналі буде НАБАГАТО нижчим за 0.70 (<0.25). Шанс 70%.
 (б) R²_проєкц діагоналі буде близьким до поелементного 0.33. Шанс 60%.
 (в) бутстреп на 60 мережах покаже, що ×1.05 не відрізняється від 1.0
     на рівні 90%. Шанс 55%.
"""
import glob, math, sys, time
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NB, NB1, NLB = 12, 48, 4
KAPPA = 200.0
SP = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad'
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4')
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
print(__doc__.split('📣')[1], flush=True)
WZ = np.load(f'{SP}/fs_w.npy', mmap_mode='r')
files = sorted(glob.glob(f'{SP}/mc88cache/n*.npz'))[:NETS]
N = len(files); IDXW = [int(f.rsplit('/', 1)[1][1:4]) for f in files]
LBL = [min((q * NLB) // (L_ - 1), NLB - 1) for q in range(L_ - 1)]
NBIN, NBIN1 = NLB * NB ** 3, NLB * NB1
EYE = torch.eye(W_, dtype=torch.bool, device=DEV); OFF = ~EYE
print(f'мереж: {N} · {DEV}\n', flush=True)


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
    return ((torch.minimum(A, B) * NB + torch.maximum(A, B)) * NB + c + LBL[q] * NB ** 3)


def chain(W, ORA=None, ora=(0, 0, 0, 0), FM=None, FD=None, FY=None, w=1.0):
    C = W[0].T @ W[0]; m = torch.zeros(W_, dtype=torch.float64, device=DEV)
    for i in range(1, L_ + 1):
        m_a, Ca, s, t, rho = gauss_step(m, C); q = i - 1
        if i == L_:
            return m_a
        oo = torch.outer(s, s)
        if ORA is not None:
            if ora[0]:
                m_a = m_a + ORA[2][q] * s
            if ora[1] or ora[2]:
                cr = ORA[0][q] * oo
                cr = torch.diag(torch.diagonal(cr)) if not ora[2] else (
                    cr if ora[1] else cr - torch.diag(torch.diagonal(cr)))
                Ca = Ca + cr
        if FM is not None or FD is not None:
            j1 = i1(t, q)
        if FM is not None:
            m_a = m_a + w * FM[j1] * s
        if FY is not None:
            cr = w * FY[i3(t, rho, q)] * oo
            Ca = Ca + (cr - torch.diag(torch.diagonal(cr)))
        if FD is not None:
            Ca = Ca + torch.diag(w * FD[j1] * s * s)
        m = m_a @ W[i]; C = W[i].T @ (Ca @ W[i])


NETW = []; TRUE_ = np.zeros((N, W_)); tpool, rpool = [], []
for n, f in enumerate(files):
    d = np.load(f); TRUE_[n] = d['out']
    W = [torch.tensor(np.asarray(WZ[IDXW[n]][q], np.float64), device=DEV) for q in range(L_)]
    mh = torch.tensor(d['mh'], dtype=torch.float64, device=DEV)
    ma = torch.tensor(d['ma'], dtype=torch.float64, device=DEV)
    Ch = torch.tensor(d['Ch'], dtype=torch.float64, device=DEV)
    Ca = torch.tensor(d['Ca'], dtype=torch.float64, device=DEV)
    dY, dD, dM, TT, SS = [], [], [], [], []
    for q in range(L_):
        m_a, Cg, s, t, rho = gauss_step(mh[q], Ch[q])
        dM.append((ma[q] - m_a) / s)
        if q < L_ - 1:
            Yf = (Ca[q] - Cg) / torch.outer(s, s)
            dY.append(Yf); dD.append(torch.diagonal(Yf).clone())
            TT.append((t, rho)); SS.append(s)
            if n < 8 and q % 3 == 0:
                tpool.append(t.cpu().numpy()); rpool.append(rho[OFF].cpu().numpy())
    NETW.append((W, dY, dD, dM, TT, SS)); del mh, ma, Ch, Ca
tp = np.concatenate(tpool); rp = np.concatenate(rpool)
ED_T = torch.tensor(np.quantile(tp, np.linspace(0, 1, NB + 1))[1:-1], dtype=torch.float64, device=DEV)
ED_R = torch.tensor(np.quantile(rp, np.linspace(0, 1, NB + 1))[1:-1], dtype=torch.float64, device=DEV)
ED_T1 = torch.tensor(np.quantile(tp, np.linspace(0, 1, NB1 + 1))[1:-1], dtype=torch.float64, device=DEV)
del tpool, rpool, tp, rp
print(f'завантажено · GPU {torch.cuda.memory_allocated(DEV)/2**30:.2f} GiB', flush=True)

AY = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
CY = torch.zeros(N, NBIN, dtype=torch.float64, device=DEV)
AD = torch.zeros(N, NBIN1, dtype=torch.float64, device=DEV)
CD = torch.zeros(N, NBIN1, dtype=torch.float64, device=DEV)
AM = torch.zeros(N, NBIN1, dtype=torch.float64, device=DEV)
with torch.no_grad():
    for n in range(N):
        _, dY, dD, dM, TT, _ = NETW[n]
        for q in range(L_ - 1):
            j3 = i3(TT[q][0], TT[q][1], q); j1 = i1(TT[q][0], q)
            AY[n].scatter_add_(0, j3[OFF], dY[q][OFF])
            CY[n].scatter_add_(0, j3[OFF], torch.ones_like(dY[q][OFF]))
            AD[n].scatter_add_(0, j1, dD[q]); CD[n].scatter_add_(0, j1, torch.ones_like(dD[q]))
            AM[n].scatter_add_(0, j1, dM[q])
print('таблиці зібрані', flush=True)


def tb(S, C_, n):
    msk = torch.ones(S.shape[0], dtype=torch.bool, device=DEV)
    if n is not None:
        msk[n] = False
    return S[msk].sum(0) / (C_[msk].sum(0) + KAPPA)


# ================= 1) ПРОЄКЦІЙНИЙ R² =================
rng = np.random.default_rng(11)
p3 = torch.tensor(rng.permutation(NBIN), device=DEV)
acc = {k: ([], []) for k in ('поза-ел', 'поза-ПРОЄКЦ', 'діаг-ел', 'діаг-ПРОЄКЦ',
                             'm-ел', 'm-ПРОЄКЦ', '⛔поза-ПРОЄКЦ переміш',
                             'P1 поза-ПРОЄКЦ тавтол')}
with torch.no_grad():
    for n in range(N):
        W, dY, dD, dM, TT, SS = NETW[n]
        FY = tb(AY, CY, n); FD = tb(AD, CD, n); FM = tb(AM, CD, n); FYp = FY[p3]
        for q in range(0, L_ - 1, 2):
            s = SS[q]; oo = torch.outer(s, s); Wn = W[q + 1]
            j3 = i3(TT[q][0], TT[q][1], q); j1 = i1(TT[q][0], q)
            Dt = dY[q] * oo
            Dt_off = Dt - torch.diag(torch.diagonal(Dt))
            Dp = FY[j3] * oo; Dp_off = Dp - torch.diag(torch.diagonal(Dp))
            Dq = FYp[j3] * oo; Dq_off = Dq - torch.diag(torch.diagonal(Dq))
            pr = lambda M: torch.diagonal(Wn.T @ (M @ Wn)).cpu()
            sub = torch.randint(0, W_ * W_, (4000,), device=DEV)
            ad = lambda k, u, v: (acc[k][0].append(u.cpu() if u.is_cuda else u),
                                  acc[k][1].append(v.cpu() if v.is_cuda else v))
            ad('поза-ел', dY[q].reshape(-1)[sub], FY[j3].reshape(-1)[sub])
            ad('поза-ПРОЄКЦ', pr(Dt_off), pr(Dp_off))
            ad('⛔поза-ПРОЄКЦ переміш', pr(Dt_off), pr(Dq_off))
            ad('P1 поза-ПРОЄКЦ тавтол', pr(Dt_off), pr(Dt_off))
            dd_t = torch.diagonal(Dt); dd_p = FD[j1] * s * s
            ad('діаг-ел', dD[q], FD[j1])
            ad('діаг-ПРОЄКЦ', pr(torch.diag(dd_t)), pr(torch.diag(dd_p)))
            ad('m-ел', dM[q], FM[j1])
            ad('m-ПРОЄКЦ', (dM[q] * s) @ Wn, (FM[j1] * s) @ Wn)
print(f'\n{"="*76}\n1) R²: ПОЕЛЕМЕНТНИЙ проти ПРОЄКЦІЙНОГО (LOO по мережах, 60 мереж)')
print(f'{"канал":26s} {"RMS цілі":>12s} {"R²":>9s}')
for k, (a, b_) in acc.items():
    yy = torch.cat(a).numpy(); pp = torch.cat(b_).numpy()
    r2 = 1.0 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    print(f'  {k:24s} {yy.std():12.4e} {r2:+9.4f}')

# ================= 2) ГОЛОВНА ТАБЛИЦЯ НА 60 МЕРЕЖАХ + БУТСТРЕП =================
NRM = (TRUE_ ** 2).mean()
PRED = {}


def run(tag, **kw):
    Pm = np.zeros((N, W_))
    with torch.no_grad():
        for n in range(N):
            kk = dict(kw)
            if 'ORA' in kk:
                kk['ORA'] = (NETW[n][1], NETW[n][2], NETW[n][3])
            for key, src in (('FY', (AY, CY)), ('FD', (AD, CD)), ('FM', (AM, CD))):
                if kk.get(key) == 1:
                    kk[key] = tb(src[0], src[1], n)
                elif kk.get(key) == 2:
                    kk[key] = tb(src[0], src[1], n)[p3 if key == 'FY' else slice(None)]
            Pm[n] = chain(NETW[n][0], **kk).cpu().numpy()
    PRED[tag] = Pm


t0 = time.time()
run('база')
run('оракул СТАН (m+C)', ORA=1, ora=(1, 1, 1, 0))
run('оракул m,поза + вивч.діаг', ORA=1, ora=(1, 0, 1, 0), FD=1, w=1.0)
for w in (0.5, 1.0):
    run(f'вивчена ТРІЙКА w={w:g}', FM=1, FD=1, FY=1, w=w)
run('вивч. лише поза w=0.25', FY=1, w=0.25)
run('⛔перемішана ТРІЙКА w=0.5', FM=2, FD=2, FY=2, w=0.5)
print(f'ланцюги {time.time()-t0:.0f}s', flush=True)


def mse_pernet(Pm, idx):
    pt = (Pm * TRUE_).sum(1); pp = (Pm * Pm).sum(1); tt = (TRUE_ * TRUE_).sum(1)
    a_n = pt / np.maximum(pp, 1e-30)
    out = np.zeros(len(idx))
    for k, i in enumerate(idx):
        a = a_n[[j for j in idx if j != i]].mean()
        out[k] = (a * a * pp[i] - 2 * a * pt[i] + tt[i]) / W_
    return out


full = np.arange(N)
base_pn = mse_pernet(PRED['база'], full)
print(f'\n{"="*84}\n2) ГОЛОВНА ТАБЛИЦЯ · {N} МЕРЕЖ · бутстреп 4000 по мережах')
print(f'{"варіант":30s} {"сира":>9s} {"після α":>9s} {"MSE":>12s} {"vs база":>9s} {"95% CI":>18s}')
bidx = [np.random.default_rng(1000 + k).choice(N, N, replace=True) for k in range(4000)]
for k, Pm in PRED.items():
    pn = mse_pernet(Pm, full)
    e = math.sqrt(((Pm - TRUE_) ** 2).mean() / NRM)
    g = base_pn.mean() / pn.mean()
    bs = np.array([base_pn[b].mean() / pn[b].mean() for b in bidx])
    print(f'{k:30s} {e:8.4%} {math.sqrt(pn.mean()/NRM):8.4%} {pn.mean():12.4e} '
          f'{"x%.3f" % g:>9s} {"[%.3f, %.3f]" % (np.quantile(bs,0.025), np.quantile(bs,0.975)):>18s}')
print(f'\n🔧 RMS(істина)² = {NRM:.5f} на {N} мережах')
