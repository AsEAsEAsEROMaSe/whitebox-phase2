"""СКАН ПО ПАРАХ: З ЧИМ КОРЕЛЮЄ ПОХИБКА `C_jk`? (С87, питання ЮЗЕРА)

## ЗВІДКИ — ПИТАННЯ ЮЗЕРА
«Чи міряли ми, як кожна вага впливає, тобто з ким корелює в вагах конкретних
σ або γ?»
ВІДПОВІДЬ: НІ. І це виявило дірку в усіх попередніх сканах.

## 🚨 ДІРКА
Усі скани були ПОНЕЙРОННІ: `sigscan87` — 107520 точок (нейрони), 15 ознак виду
`ΣW²`, `Σ|W|³`, `ΣW⁴`; `stepcov87` — R² на `γ₃,γ₄,t`. Усі дали ≈0.
А сьогоднішній розклад каже: похибка кроку в ПОЗАДІАГОНАЛІ 1.69%, у діагоналі
0.23% ⟹ **похибка є властивістю ПАРИ `(j,k)`, а не нейрона**.
📌 Понейронна ознака не може її пояснити ЗА ПОБУДОВОЮ — це та сама пастка
парності, що вбила знаки для `σ`, лише з іншого боку.

## 📐 ЯК ВАГИ ПЕРЕНОСЯТЬ ПОХИБКУ (виведено)
```
σ_i² = Σ_jk W_ji W_ki C_jk        ⟹  ∂σ_i²/∂W_ji = 2·(C·W)_ji
δσ_i² = Σ_jk W_ji W_ki · δC_jk
```
Нейрон `i` успадковує похибку РІВНО тих `j`, з якими має великі ваги.
⟹ Питання не «яка вага корелює з σ», а «як `δC_jk` залежить від ПАРИ».

## 🎯 ЩО МІРЯЄТЬСЯ
Ціль: `y_jk = (C_a^істина − C_a^Мелер)_jk / (s_j·s_k)` — нормована похибка
замикання для пари, з ІСТИННИМИ `m,C` на вході (тобто СВІЖА, без накопичення).
Ознаки пари: `ρ_jk`, `t_j`, `t_k`, симетричні комбінації, `Φ`, `φ`.
🔑 Мелер — ряд по `ρ`; ми перевіряли його ВИЩІ ПОРЯДКИ (нуль), але ЖОДНОГО
разу не питали, з чим корелює ЗАЛИШОК як функція `ρ`.

## ⛔ПЛОМБИ
 P1 ТАВТОЛОГІЯ: ознака = сама ціль ⟹ R²=1.0000.
 P2 ⛔ПІДЛОГА: випадкова ознака ⟹ R²≈0.
 P3 ⛔ПЕРЕМІШАНІ пари ⟹ R²≈0.
 P4 LOO ПО МЕРЕЖАХ у кожній регресії.

📣 ПЕРЕДБАЧЕННЯ:
 (а) `ρ` буде НАЙСИЛЬНІШОЮ одиничною ознакою (|corr|>0.3). Шанс 60% —
     похибка Мелера мусить залежати від того, наскільки обрізаний ряд.
 (б) сукупний R² парних ознак ≥0.5. Шанс **35%**.
 (в) R² ≥0.9 (що дало б реальний важіль). Шанс 10%.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N; BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256; PROBE = (7, 15, 23, 30); NPAIR = 4000
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
print(f'{len(FS)} мереж · 2^{LOG2N} · {NPAIR} пар/шар · {DEV}\n', flush=True)
FEAT = ('rho', '|rho|', 'rho^2', 'rho^3', 't_j+t_k', '|t_j-t_k|', 't_j*t_k',
        'Phi_j*Phi_k', 'phi_j*phi_k', 'rho*tt', 'rho*|dt|', 'rho^2*tt')
rows_y, rows_X, rows_g = [], [], []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        NP = len(PROBE)
        SH = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GH = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GA = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(3900 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]; a = torch.relu(h)
                if q in PROBE:
                    j = PROBE.index(q); hd, ad = h.double(), a.double()
                    SH[j] += hd.sum(0); GH[j] += hd.T @ hd
                    SA[j] += ad.sum(0); GA[j] += ad.T @ ad
            d += b
        rg = torch.Generator(device=DEV); rg.manual_seed(100 + n)
        for j, q in enumerate(PROBE):
            mh = SH[j] / NS; Ch = GH[j] / NS - torch.outer(mh, mh); Ch = 0.5*(Ch+Ch.T)
            ma = SA[j] / NS; Ca = GA[j] / NS - torch.outer(ma, ma); Ca = 0.5*(Ca+Ca.T)
            s = torch.sqrt(torch.clamp(torch.diagonal(Ch), min=1e-30))
            t = mh / s; P_, f_ = Phi(t), phi(t)
            Ccl = Ch * torch.outer(P_, P_) + (0.5*(Ch*Ch)) * torch.outer(f_/s, f_/s)
            rho = Ch / torch.outer(s, s)
            # 🚨 НОРМУВАТИ НА `s` ПРЕАКТИВАЦІЙ, а не на `sqrt(diag(Ca))`:
            #    для майже мертвих нейронів (t≈−3, а їх ТРЕТИНА) `Var(a)→0`,
            #    і ділення на нього дало RMS цілі 6e+21 (спіймано одразу).
            Y = (Ca - Ccl) / torch.outer(s, s)
            iu = torch.randint(0, W_, (NPAIR,), device=DEV, generator=rg)
            ju = torch.randint(0, W_, (NPAIR,), device=DEV, generator=rg)
            msk = iu != ju; iu, ju = iu[msk], ju[msk]
            r = rho[iu, ju]; ti, tj = t[iu], t[ju]
            X = torch.stack([r, r.abs(), r*r, r**3, ti+tj, (ti-tj).abs(), ti*tj,
                             P_[iu]*P_[ju], f_[iu]*f_[ju], r*(ti*tj),
                             r*(ti-tj).abs(), r*r*(ti*tj)], 1)
            rows_y.append(Y[iu, ju].cpu().numpy()); rows_X.append(X.cpu().numpy())
            rows_g.append(np.full(len(iu), n))
    print(f'  net{n}', flush=True)
y = np.concatenate(rows_y); X = np.concatenate(rows_X); grp = np.concatenate(rows_g)
print(f'\n{"="*66}\nточок: {len(y)} · RMS цілі: {y.std():.4e}')


def r2(Xs, yy, gg):
    Xs = np.column_stack([Xs, np.ones(len(Xs))])
    out = np.zeros(len(yy))
    for gi in np.unique(gg):
        tr, te = gg != gi, gg == gi
        A = Xs[tr].T @ Xs[tr] + 1e-8*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T @ yy[tr]); out[te] = Xs[te] @ w
    return 1.0 - ((yy-out)**2).sum()/max(((yy-yy.mean())**2).sum(), 1e-30)


print(f'\nОДИНИЧНІ КОРЕЛЯЦІЇ З ПОХИБКОЮ ПАРИ')
cs = []
for k, nm in enumerate(FEAT):
    c = np.corrcoef(X[:, k], y)[0, 1]
    cs.append((abs(c), nm, c))
for a, nm, c in sorted(cs, reverse=True):
    print(f'  {nm:14s} {c:+.4f}')
print(f'\nСУКУПНИЙ R² (LOO по мережах)')
print(f'  усі 12 парних ознак           {r2(X, y, grp):.4f}')
print(f'  лише rho-родина (4)           {r2(X[:, :4], y, grp):.4f}')
print(f'  лише t-родина (5..9)          {r2(X[:, 4:9], y, grp):.4f}')
rng = np.random.default_rng(7)
print(f'  ⛔P2 підлога (випадкова)       {r2(rng.standard_normal((len(y),3)), y, grp):+.5f}')
print(f'  ⛔P3 перемішані пари           {r2(X[rng.permutation(len(y))], y, grp):+.5f}')
print(f'  ⛔P1 тавтологія                {r2(y[:, None], y, grp):.4f}')

# 🔑 СТЕЛЯ БУДЬ-ЯКОЇ МОДЕЛІ ВІД (t_j, t_k, ρ): непараметричний бінінг.
#    Лінійна регресія дає 0.238; С57 мікромережею дістала 0.82 (але з κ з
#    ІСТИНИ). Питання: скільки дає САМА НЕЛІНІЙНІСТЬ від трьох координат,
#    без жодних кумулянтів. Бінінг — верхня межа для будь-якої такої моделі.
tj_ = 0.5 * (X[:, 4] + np.sqrt(np.maximum(X[:, 4] ** 2 - 4 * X[:, 6], 0)))
tk_ = X[:, 4] - tj_
rho_ = X[:, 0]
print(f'\n{"="*66}\n🔑 СТЕЛЯ МОДЕЛІ ВІД (t_j, t_k, ρ) — НЕПАРАМЕТРИЧНИЙ БІНІНГ')
print(f'{"сітка":>12s} {"комірок":>9s} {"R² (LOO по мережах)":>21s}')
for nb in (6, 10, 14, 20):
    ed_t = np.quantile(np.concatenate([tj_, tk_]), np.linspace(0, 1, nb + 1))
    ed_r = np.quantile(rho_, np.linspace(0, 1, nb + 1))
    a_ = np.clip(np.searchsorted(ed_t, tj_) - 1, 0, nb - 1)
    b_ = np.clip(np.searchsorted(ed_t, tk_) - 1, 0, nb - 1)
    c_ = np.clip(np.searchsorted(ed_r, rho_) - 1, 0, nb - 1)
    lo, hi = np.minimum(a_, b_), np.maximum(a_, b_)     # симетризація пари
    idx = (lo * nb + hi) * nb + c_
    out = np.zeros(len(y))
    for gi in np.unique(grp):
        tr, te = grp != gi, grp == gi
        sm = np.bincount(idx[tr], weights=y[tr], minlength=nb ** 3)
        cn = np.bincount(idx[tr], minlength=nb ** 3)
        mu = np.where(cn > 5, sm / np.maximum(cn, 1), y[tr].mean())
        out[te] = mu[idx[te]]
    r = 1.0 - ((y - out) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'{nb:>6d}^3 {len(np.unique(idx)):9d} {r:21.4f}')
print('\n📌 якщо стеля ≥0.8 — С57 відтворено БЕЗ κ, і гілка жива')
print('   якщо ~0.3 — 82% у С57 трималися на κ з ІСТИНИ, і це блокер')
