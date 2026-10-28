"""КАЛІБРОВАНЕ ЗАМИКАННЯ: чи систематична похибка рекурсії (С78).

Позиція юзера: рахувати треба з ВАГ, а не ітеруватись по точках — семплінг це
те саме, що перелічувати виходи генератора замість знати його закон.

Числа, які це прайсують:
   наша аналітика (Мелер K=2 + перекіс)  0.636%
   вибірка (те, що подаємо)              0.177%
   стеля аналітики з точними m,s,κ₃,κ₄   0.0496%
⟹ аналітика МОЖЕ обігнати семплінг у 3.6×, але зараз відстає у 3.6×. Уся
різниця — накопичення похибки рекурсії, не теорія.

🔑 ВАЖІЛЬ: `attractor.py` довів, що форма НЕ залежить від входу ⟹ похибка
   замикання СИСТЕМАТИЧНА, а не випадкова ⟹ її можна зміряти раз і зашити.
   Перевіряємо: фітимо пошарові множники на s і на m, з LOO ПО МЕРЕЖАХ.

🚨 LOO по мережах обов'язковий: [[finding-scale-noise-is-predictable-but-only-within-a-network]]
   — «OOS по жеребах ≠ OOS по мережах», на цьому вже горіли.
"""
import sys
import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
KMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 16
W_, D, CH = 256, 32, 1 << 13
S2PI = float(1.0/np.sqrt(2*np.pi)); SQ2 = float(np.sqrt(2.0))
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('🚨')[1], flush=True)

def _truth(W, seed, npairs):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(3, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]; pd = p.double()
            Z[L][0] += pd.sum(0); Z[L][1] += (pd*pd).sum(0)
            x = torch.relu(p); Z[L][2] += x.double().sum(0)
        done += b
    N = float(2*npairs); out = []
    for L in range(D):
        a1, a2 = Z[L][0]/N, Z[L][1]/N
        out.append(dict(s=(a2-a1*a1).clamp_min(1e-30).sqrt(), relu=Z[L][2]/N))
    return out

def _chain(W, us=None, um=None):
    """Мелер K=2; us,um — пошарові калібрувальні множники (None ⟹ без них)."""
    m = torch.zeros(W_, dtype=torch.float64, device=DEV)
    C = torch.eye(W_, dtype=torch.float64, device=DEV)
    ss, mm = [], []
    for i in range(KMAX+1):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        if us is not None and i < len(us):
            s = s * us[i]
            cp = cp * (us[i].unsqueeze(0) * us[i].unsqueeze(1))
        mp = m @ w; t = mp/s
        ph = torch.exp(-0.5*t*t)*S2PI; Ph = 0.5*(1.0+torch.erf(t/SQ2))
        m = mp*Ph + s*ph
        if um is not None and i < len(um):
            m = m * um[i]
        C = cp*torch.outer(Ph, Ph) + (0.5*(cp*cp))*torch.outer(ph/s, ph/s)
        d = (mp*mp+s*s)*Ph + (mp*s)*ph - m*m
        C = C + torch.diag(d - C.diagonal())
        if i >= 1:
            ss.append(s.clone()); mm.append(m.clone())
    return ss, mm

T, WW = [], []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        T.append(_truth(W, 91_000+13*n, 1 << LOGN))
    WW.append(W)
    print(f'  … істина {n+1}/{NETS}', flush=True)

# скалярні поправки на шар: відношення істина/замикання, медіана по нейронах
US = np.zeros((NETS, KMAX)); UM = np.zeros((NETS, KMAX))
for n in range(NETS):
    with torch.no_grad():
        ss, mm = _chain(WW[n])
    for L in range(KMAX):
        US[n, L] = float((T[n][L+1]['s']/ss[L]).median())
        UM[n, L] = float((T[n][L+1]['relu']/mm[L]).median())

print('\nкалібрувальні множники (медіана по мережах, розкид по мережах):')
print('шар    u_s      розкид      u_m      розкид')
for L in (0, 1, 3, 7, 11, 15):
    if L < KMAX:
        print(f'{L:3d} {np.median(US[:,L]):8.4f} {US[:,L].std()/abs(US[:,L].mean()):9.2%} '
              f'{np.median(UM[:,L]):8.4f} {UM[:,L].std()/abs(UM[:,L].mean()):9.2%}')

res = np.zeros(3)
for n in range(NETS):
    oth = [k for k in range(NETS) if k != n]
    us = torch.tensor(np.median(US[oth], 0), device=DEV, dtype=torch.float64)
    um = torch.tensor(np.median(UM[oth], 0), device=DEV, dtype=torch.float64)
    us = us.unsqueeze(1).expand(KMAX, W_).contiguous()
    with torch.no_grad():
        _, m0 = _chain(WW[n])
        _, m1 = _chain(WW[n], us=us)
        _, m2 = _chain(WW[n], us=us, um=um)
    for L in range(KMAX):
        tr = T[n][L+1]['relu']; nr = float((tr*tr).sum())
        for k, mv in enumerate((m0, m1, m2)):
            res[k] += float((mv[L]-tr).pow(2).sum())/nr
res /= NETS*KMAX
print(f'\n{NETS} мереж, LOO ПО МЕРЕЖАХ, шари 1..{KMAX}\n')
print(f'{"замикання":>26s} {"відн. похибка m":>17s} {"проти базового":>15s}')
print('-'*62)
for nm, v in (('базове Мелер K=2', res[0]), ('+ калібрування s', res[1]),
              ('+ калібрування s і m', res[2])):
    print(f'{nm:>26s} {np.sqrt(v):17.4e} {np.sqrt(res[0]/max(v,1e-30)):14.3f}×')
best = np.sqrt(res.min())
print(f'\n🚨 ВИБІРКА = 1.77e-03. 🔑 НАЙКРАЩА АНАЛІТИКА: {best:.3e} ⟹ '
      f'{"🟢 ОБІГНАЛА СЕМПЛІНГ" if best < 1.77e-3 else f"⛔ все ще гірше у {best/1.77e-3:.1f}×"}')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
