"""ДВА НЕЗІБРАНІ ВАЖЕЛІ `sub93`: МАСШТАБ ПОПРАВКИ І МАСШТАБ `λ_Q` (С84)

## ЗВІДКИ
`estimator_sub93.py` дав ×1.0134 проти sub87 (sub92 — ×1.0046), але два
параметри в ньому НЕ підбирались:
1. коефіцієнти поправки стоять ТЕОРЕТИЧНІ (`γ₁/3`, `γ₂/12`) — виведені для
   ідеального Еджворта, а не для нашої ознаки;
2. `_LAM_Q` лишились ті, що підганялись під БРУДНУ ознаку, хоч `whatfeeds.py`
   зміряв: після чистки оптимум `λ` зсувається ×1.0 → **×1.5**.
Стеля чистки з ОРАКУЛОМ — ×1.0896, узято ×1.0088 ⟹ лежить ще ~90%.

## 🔑 ЧОМУ НЕ ПОТРІБНА СІТКА: ПРОГНОЗ ЛІНІЙНИЙ ПО ОБОХ
`pr = (1−w)·ours + w(1+α)·m_deep`, `ours = last − v`, а `v` входить лінійно:
    `v ← v + λ_Q·(δ_Q − c_e·Δq)`  ⟹  `pr(A,B) = pr₀ + A·U + B·V`
де `pr₀` — прогноз при `λ_Q=0`, `U` — внесок родини `Q` з нинішніми `λ_Q`,
`V` — внесок поправки. Гейти й вибірка ТІ САМІ ⟹ лінійність ТОЧНА, не наближена.
⟹ три прогони на жереб дають ПОВНУ поверхню, оптимум — 2×2 система:
```
MSE(A,B) = ⟨e₁e₂⟩ + A⟨U·(e₁+e₂)⟩ + B⟨V·(e₁+e₂)⟩ + A²⟨U²⟩ + B²⟨V²⟩ + 2AB⟨UV⟩
```
Переклад назад: `A = c_λ`, `B = −c_λ·c_e` ⟹ `c_e = −B/A`.

⛔ ПЛОМБИ:
 (а) ЛІНІЙНІСТЬ: `pr(1,−1)` з формули мусить збігтися з ПРЯМИМ прогоном sub93
     побітово — якщо ні, розклад хибний і вся поверхня недійсна;
 (б) `(A,B) = (1,0)` мусить відтворити sub92, `(1,−1)` — sub93;
 (в) LOO ПО МЕРЕЖАХ: оптимум фітується без поточної мережі;
 (г) ⛔контроль: та сама оптимізація на ЧИСТОМУ ШУМІ замість `V`;
 (д) істина — БАНК, крос-ціль.

📣 ПЕРЕДБАЧЕНО: `c_λ` вийде **>1** (бо чистка зменшила зміщення ⟹ можна
   довіряти ознаці більше) і `c_e` близько 1..2. Чекаю сумарно ×1.01–1.03 до
   sub93, тобто помітно менше за стелю ×1.0896: два скаляри не можуть
   замінити ПОШАРОВУ оцінку зміщення. Якщо LOO дасть <1.005 — важелі порожні
   і треба одразу йти в апарат С70 (фіт зміщення `q` з ознак нейрона).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qtune.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS,
                        ALPHA, W_BLEND)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, NDRAW = 32, 8
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів, {DEV}\n', flush=True)


def run(wc, npair, dev, seed, cl, aux, lamq_on, edge_on):
    """Прогін сабміту. `lamq_on` — вмикає родину Q, `edge_on` — чистку."""
    depth = len(wc)
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    den = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / den - sg0 * INV_SQRT_2PI
    s_cl, m_cl, q_cl, P_cl, m_deep = cl
    AA, BB = aux                       # s²φ, s²tφ із замикання
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        pd = p.double()
        p2 = pd * pd
        mh = pd.sum(0) / den
        e2 = p2.sum(0) / den
        sh2 = (e2 - mh * mh).clamp_min(1e-30)
        sh = sh2.sqrt()
        x = torch.relu(p)
        xm = x.double().sum(0) / den
        xq = x.double().pow(2).sum(0) / den
        if i <= KSIG:
            dq = 0.0
            if edge_on:
                e3 = (p2 * pd).sum(0) / den
                e4 = (p2 * p2).sum(0) / den
                mh2 = mh * mh
                k3 = e3 - 3.0 * mh * e2 + 2.0 * mh2 * mh
                k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                      - 3.0 * mh2 * mh2 - 3.0 * sh2 * sh2)
                dq = ((k3 / (sh2 * sh)) * (AA[i - 1] / 3.0)
                      - (k4 / (sh2 * sh2)) * (BB[i - 1] / 12.0))
            v = (v + (SP.LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + SP.LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + SP.LAM_G[i - 1] * (gate - P_cl[i - 1]))
            if lamq_on:
                v = v + SP.LAM_Q[i - 1] * ((xq - q_cl[i - 1]) - dq)
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


def aux_parts(wc, dev):
    """`s²φ(t)` і `s²tφ(t)` із замикання на шарах 1..KSIG (як у sub93)."""
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    AA, BB = [], []
    for i in range(len(wc)):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s2ph = (s * s) * ph
            AA.append(s2ph); BB.append(s2ph * t)
        m = mp * Ph + s * ph
        if i < len(wc) - 1:
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
    return AA, BB


# скаляри на мережу: [e1e2, U(e1+e2), V(e1+e2), U², V², UV] + контроль із шумом
NS_ = 6
S = np.zeros((len(FS), NS_))
SC = np.zeros((len(FS), NS_))          # ⛔контроль: V ← чистий шум тієї ж норми
SEAL_LIN = 0.0

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        gen = torch.Generator(device=DEV); gen.manual_seed(31 + n)
        for d in range(NDRAW):
            sd = 97_000_000 + 733 * n + d
            p0 = run(wc, npair, DEV, sd, cl, aux, False, False)   # λ_Q = 0
            pu = run(wc, npair, DEV, sd, cl, aux, True, False)    # sub92
            pv = run(wc, npair, DEV, sd, cl, aux, True, True)     # sub93
            U = pu - p0
            Vv = U - (pv - p0)          # внесок ПОПРАВКИ (з плюсом)
            if d == 0:                  # ⛔ПЛОМБА (а): лінійність
                SEAL_LIN = max(SEAL_LIN,
                               float((p0 + U - Vv - pv).abs().max()))
            e1, e2 = p0 - T1, p0 - T2
            es = e1 + e2
            rnd = torch.randn(Vv.shape, dtype=torch.float64, device=DEV,
                              generator=gen)
            Vc = rnd * (Vv.norm() / rnd.norm())
            for arr, Vx in ((S, Vv), (SC, Vc)):
                arr[n, 0] += float((e1 * e2).sum()) / NDRAW
                arr[n, 1] += float((U * es).sum()) / NDRAW
                arr[n, 2] += float((Vx * es).sum()) / NDRAW
                arr[n, 3] += float((U * U).sum()) / NDRAW
                arr[n, 4] += float((Vx * Vx).sum()) / NDRAW
                arr[n, 5] += float((U * Vx).sum()) / NDRAW
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/qtune.npz', S=S, SC=SC)


def mse(arr, idx, A, B):
    z = arr[idx]
    return (z[:, 0] + A*z[:, 1] + B*z[:, 2]
            + A*A*z[:, 3] + B*B*z[:, 4] + 2*A*B*z[:, 5]).mean()


def opt(arr, idx):
    """Аналітичний оптимум (A,B) на підмножині мереж `idx`."""
    z = arr[idx].sum(0)
    M = np.array([[2*z[3], 2*z[5]], [2*z[5], 2*z[4]]])
    r = -np.array([z[1], z[2]])
    try:
        return np.linalg.solve(M, r)
    except np.linalg.LinAlgError:
        return np.array([1.0, -1.0])


print(f'\n⛔ПЛОМБА (а) лінійність pr₀+U−V ≡ sub93: max|Δ| = {SEAL_LIN:.2e} '
      f'{"✅" if SEAL_LIN < 1e-15 else "🚨 РОЗКЛАД ХИБНИЙ, ПОВЕРХНЯ НЕДІЙСНА"}')
ALL = np.arange(len(FS))
b92 = mse(S, ALL, 1.0, 0.0)
b93 = mse(S, ALL, 1.0, -1.0)
print(f'⛔ПЛОМБА (б) sub92 = {b92:.6e} · sub93 = {b93:.6e} · '
      f'sub93/sub92 = ×{b92/b93:.4f}\n')

Ao, Bo = opt(S, ALL)
print(f'🔑 ОПТИМУМ НА ВСІХ (оптимістично): A = {Ao:.4f}  B = {Bo:.4f}  '
      f'⟹ c_λ = {Ao:.3f}, c_e = {-Bo/Ao:.3f}')
print(f'   MSE = {mse(S, ALL, Ao, Bo):.6e} ⟹ ×{b93/mse(S, ALL, Ao, Bo):.4f} до sub93')

print(f'\n{"варіант":>34s} {"MSE":>13s} {"× до sub93":>11s} {"× до sub92":>11s}')
print('─' * 74)
rows = [('⛔ sub92 (A=1, B=0)', 1.0, 0.0),
        ('⛔ sub93 (A=1, B=−1)', 1.0, -1.0),
        ('🔵 лише λ_Q ×1.5, без чистки', 1.5, 0.0),
        ('🔵 λ_Q ×1.5 + чистка', 1.5, -1.5),
        ('🎯 оптимум на всіх', Ao, Bo)]
for nm, A, B in rows:
    v = mse(S, ALL, A, B)
    print(f'{nm:>34s} {v:13.6e} {b93/v:11.4f} {b92/v:11.4f}')

# ── ⛔(в) ЧЕСНИЙ LOO ПО МЕРЕЖАХ + ⛔(г) контроль на шумі
for nm, arr in (('🎯 справжня поправка', S), ('⛔ КОНТРОЛЬ: шум замість V', SC)):
    acc = 0.0
    As, Bs = [], []
    for n in range(len(FS)):
        o = np.array([k for k in range(len(FS)) if k != n])
        A, B = opt(arr, o)
        As.append(A); Bs.append(B)
        acc += mse(arr, np.array([n]), A, B)
    acc /= len(FS)
    base = mse(arr, ALL, 1.0, 0.0)
    print(f'\n{nm}: LOO MSE = {acc:.6e} ⟹ ×{base/acc:.4f} до sub92 · '
          f'×{mse(arr, ALL, 1.0, -1.0)/acc:.4f} до sub93')
    print(f'   A: сер {np.mean(As):.3f} ст.в {np.std(As):.4f} · '
          f'B: сер {np.mean(Bs):.3f} ст.в {np.std(Bs):.4f}')

print('\n🎯 ВИРОК')
loo = 0.0
for n in range(len(FS)):
    o = np.array([k for k in range(len(FS)) if k != n])
    A, B = opt(S, o)
    loo += mse(S, np.array([n]), A, B)
loo /= len(FS)
g = b93 / loo
print(f'   чесний LOO до sub93: ×{g:.4f} ⟹ бал 1.928e-07 → {1.928e-07/max(g,1e-9):.4e}')
print(f'   📌 стеля чистки `q` з ОРАКУЛОМ: ×1.0896 (взято було ×1.0088)')
if g < 1.005:
    print('   ⟹ ⛔ ВАЖЕЛІ ПОРОЖНІ — два скаляри не замінюють пошарову оцінку')
    print('      ⟹ йти в апарат С70: фіт зміщення `q` з ознак нейрона')
else:
    print('   ⟹ 🟢 є що брати; далі — чи переживе `ab_files` на 48 мережах')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
