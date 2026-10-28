"""ЧЕСНА СИМУЛЯЦІЯ КАНАЛУ `n_act`: λ оцінюється З ТИХ САМИХ СЕМПЛІВ. (С86)

## ЗВІДКИ
`actbeta.py` пройшов гейт `β`: виграш ×1.3128 при ідеальній опорі, ×1.3055 з
реальним `β` замикання ⟹ у балі ×1.2392. 🚨АЛЕ там `Cov`,`Var_g` бралися
ОРАКУЛЬНО. У сабміті їх треба оцінювати з ТИХ САМИХ N семплів, і це:
* вносить похибку оцінки `λ` (класична втрата ~(N−k−1)/(N−1));
* робить оцінку ЗМІЩЕНОЮ (самофіт), чого теорія контрольних змінних не ловить.

## 🎯 ЩО РОБИМО — ПОВНИЙ ЦИКЛ, ЯК У САБМІТІ
Для кожного жереба `r`: беремо `N` СВІЖИХ зразків, рахуємо
    `â = mean_z a_L(z)` · `ĝ_ℓ = mean_z n_act_ℓ(z)` · опору `Σ_i Φ(t_i)` із ЗАМИКАННЯ
    `λ̂ = (Ĝ + N·β̂β̂ᵀ)⁻¹ Ĉov`   — УСЕ з тих самих N зразків
    `â_cv = â − λ̂ᵀ(ĝ − опора)`
і міряємо MSE проти ІСТИНИ (незалежна велика вибірка). Багато жеребів ⟹ чесний MSE.

⛔ПЛОМБИ:
 (1) `λ̂ = 0` мусить дати РІВНО базовий MSE (пломба обліку);
 (2) 🔑ОРАКУЛЬНЕ `λ` (з великої вибірки) мусить відтворити ×1.3055 з actbeta —
     інакше баг тут, а не втрата від самофіту;
 (3) ⛔контроль: опора від ЧУЖОЇ мережі ⟹ виграш мусить ЗНИКНУТИ або стати <1.

📣 ПЕРЕДБАЧЕНО: самофіт з'їсть частину. Класична оцінка (N−k−1)/(N−1) при
   N=5000, k=32 дає лише −0.6%, але вона для НЕЗМІЩЕНОГО випадку; тут `β̂`
   теж оцінюється, тож втрата може бути більшою.
   Шанс, що чесний виграш ≥×1.20 у балі: **60%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/actreal.py [cuda:N] [NETS] [DRAWS]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
N_WORK = 5000
TRUTH_LOG2 = 21
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
SQ2PI = math.sqrt(2.0 * math.pi)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж × {DRAWS} жеребів × N={N_WORK} · {DEV}\n', flush=True)
_AB = np.load('/mnt/win_d/llm/White-Box/arena62/logs/actbeta.npz')
BBAR2 = torch.tensor((_AB['beta'] ** 2).mean(0), dtype=torch.float64, device=DEV)
print(f'β̄² з банку (actbeta, {_AB["beta"].shape[0]} мереж): '
      f'шар1 {float(BBAR2[0]):.2e} · шар16 {float(BBAR2[15]):.2e} · '
      f'шар32 {float(BBAR2[31]):.2e}\n', flush=True)


def closure_gates(W):
    m = torch.zeros(W_, dtype=torch.float64, device=DEV)
    C = torch.eye(W_, dtype=torch.float64, device=DEV)
    out = []
    for i in range(L_):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(cp.diagonal(), min=1e-20))
        mp = m @ w
        t = mp / s
        ph = torch.exp(-0.5 * t * t) / SQ2PI
        Ph = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
        out.append(Ph.sum())
        m = mp * Ph + s * ph
        if i == L_ - 1:
            break
        C = cp * torch.outer(Ph, Ph) + (0.5 * (cp * cp)) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + torch.diag(d - C.diagonal())
    return torch.stack(out)


def run(W, ns, seed, batch=1 << 12):
    """`mean a_L` і `mean n_act` (+ сирі для коваріацій) на `ns` зразках."""
    sa = torch.zeros(W_, dtype=torch.float64, device=DEV)
    sg = torch.zeros(L_, dtype=torch.float64, device=DEV)
    gg = torch.zeros(L_, L_, dtype=torch.float64, device=DEV)
    ga = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    d = 0
    while d < ns:
        b = min(batch, ns - d)
        a = torch.randn(b, W_, device=DEV, generator=g)
        na = torch.empty(b, L_, dtype=torch.float64, device=DEV)
        for q in range(L_):
            h = a @ W[q]
            na[:, q] = (h > 0).double().sum(1)
            a = torch.relu(h)
        ad = a.double()
        sa += ad.sum(0)
        sg += na.sum(0)
        gg += na.T @ na
        ga += na.T @ ad
        d += b
    return sa / ns, sg / ns, gg / ns, ga / ns


ERR = {k: np.zeros((len(FS), DRAWS)) for k in ('base', 'cv', 'cvb', 'cvk', 'cv16', 'cv8', 'early8', 'early4', 'early12', 'orac', 'ctl')}
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    Wx = [torch.tensor(np.asarray(WZ[have[int(np.load(FS[(n + 1) % len(FS)])['seed'])]][q],
                                  np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        anchor = closure_gates(W)
        anchor_x = closure_gates(Wx)
        # істина + ОРАКУЛЬНІ Cov/Var (велика незалежна вибірка)
        ta, tg, tgg, tga = run(W, 1 << TRUTH_LOG2, 91000 + n, 1 << 13)
        truth = ta
        Gor = tgg - torch.outer(tg, tg)
        Cor = tga - torch.outer(tg, ta)
        bet = anchor - tg
        eye = torch.eye(L_, dtype=torch.float64, device=DEV)
        # оракул МУСИТЬ включати істинний β — без нього λ завеликий і зміщення
        # `λᵀβ` губить усе (перевірено: ×0.75). Це пломба(2) на ×1.3055.
        Lor = torch.linalg.solve(Gor + N_WORK * torch.outer(bet, bet)
                                 + eye * 1e-8 * Gor.diagonal().mean(), Cor)
        for r in range(DRAWS):
            a1, g1, gg1, ga1 = run(W, N_WORK, 5000 + 977 * n + r)
            G1 = gg1 - torch.outer(g1, g1)
            C1 = ga1 - torch.outer(g1, a1)
            # 🔑 КЛЮЧОВЕ (С86): `β̂ = опора − ĝ` з ТИХ САМИХ семплів — це майже
            #    ЧИСТИЙ ШУМ (std = σ_g/√N ≈ 0.07 проти справжнього β 0.01..0.12).
            #    Член `N·β̂β̂ᵀ ≈ σ_g²` роздуває знаменник і обнуляє λ ⟹ ×1.002.
            #    Оскільки СПРАВЖНІЙ β майже не кусає (actbeta: ×1.3055 vs ×1.3128),
            #    правильно його ПРОСТО НЕ ВКЛЮЧАТИ. 'cvb' лишено як контраст.
            reg = eye * 1e-8 * G1.diagonal().mean()
            lam_plain = torch.linalg.solve(G1 + reg, C1)
            # 🔑 РІШЕННЯ: `β` не оцінювати з семплів (там шум), а взяти ТИПОВИЙ
            #    квадрат із БАНКУ як константу ⟹ `diag(β̄²)` замість `β̂β̂ᵀ`.
            lam_bank = torch.linalg.solve(
                G1 + N_WORK * torch.diag(BBAR2) + reg, C1)
            # 🔑 менше регресорів = менший самофіт. Беремо ГЛИБОКІ шари, де
            #    actcv зміряв три чверті сигналу (1..8 давали лише R²=0.062).
            # 🔑 actbeta: `β` НАЙМЕНШИЙ на шарах 1..8 (|β|/поріг 0.09..0.78), а на
            #    глибині сягає 3.02 ⟹ пробуємо саме РАННІ шари, де опора точна.
            #    Шар 1 узагалі ТОЧНИЙ: m=0 ⟹ t=0 ⟹ Σ Φ(0) = 128 без похибки.
            SUB = {'cv16': slice(16, 32), 'cv8': slice(24, 32),
                   'early8': slice(0, 8), 'early4': slice(0, 4),
                   'early12': slice(0, 12)}
            lam_sub = {}
            for kk, sl_ in SUB.items():
                idx = torch.arange(L_, device=DEV)[sl_]
                Gs = (G1[idx][:, idx] + N_WORK * torch.diag(BBAR2[idx])
                      + torch.eye(len(idx), dtype=G1.dtype, device=DEV)
                      * 1e-8 * G1.diagonal().mean())
                lam_sub[kk] = (idx, torch.linalg.solve(Gs, C1[idx]))
            b1 = anchor - g1
            lam_beta = torch.linalg.solve(
                G1 + N_WORK * torch.outer(b1, b1) + reg, C1)
            for key, anc, lam in (('cv', anchor, lam_plain),
                                  ('cvb', anchor, lam_beta),
                                  ('cvk', anchor, lam_bank),
                                  ('orac', anchor, Lor),
                                  ('ctl', anchor_x, lam_plain)):
                est = a1 - lam.T @ (g1 - anc)
                ERR[key][n, r] = float(((est - truth) ** 2).mean())
            for kk, (idx, lm) in lam_sub.items():
                est = a1 - lm.T @ (g1[idx] - anchor[idx])
                ERR[kk][n, r] = float(((est - truth) ** 2).mean())
            ERR['base'][n, r] = float(((a1 - truth) ** 2).mean())
    print(f'  … {n+1}/{len(FS)} seed {sd0} · база {ERR["base"][n].mean():.4e} → '
          f'CV {ERR["cv"][n].mean():.4e} (×{ERR["base"][n].mean()/ERR["cv"][n].mean():.3f}) · '
          f'оракул ×{ERR["base"][n].mean()/ERR["orac"][n].mean():.3f}', flush=True)
    del W, Wx
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/actreal.npz', **ERR)
b = ERR['base'].mean()
print('\n' + '=' * 74)
print(f'📊 MSE ОЦІНКИ `E[a_L]` ({len(FS)} мереж × {DRAWS} жеребів, N={N_WORK})')
for k, nm in (('base', 'БАЗА (як зараз)'), ('cv', '🔑КАНАЛ, λ̂ з тих самих семплів'),
              ('cvb', '⛔з шумним β̂ (самофіт)'),
              ('cvk', '🔑🔑β̄² з банку, УСІ 32 шари'),
              ('cv16', '🔑🔑β̄² з банку, шари 17..32'),
              ('cv8', '🔑🔑β̄² з банку, шари 25..32'),
              ('early12', '🔑🔑β̄² з банку, шари 1..12'),
              ('early8', '🔑🔑β̄² з банку, шари 1..8'),
              ('early4', '🔑🔑β̄² з банку, шари 1..4'),
              ('orac', 'ОРАКУЛЬНЕ λ (пломба 2)'), ('ctl', '⛔контроль: чужа опора')):
    v = ERR[k].mean()
    print(f'   {nm:<32s} {v:.5e}   ×{b/v:.4f}')
print('─' * 74)
print(f'⛔ПЛОМБА(2): оракул мусить дати ≈×1.3055 (actbeta) ⟹ тут ×{b/ERR["orac"].mean():.4f}')
print(f'⛔ПЛОМБА(3): контроль мусить бути ≲1.0 ⟹ тут ×{b/ERR["ctl"].mean():.4f}')
_CAND = ('cvk', 'cv16', 'cv8', 'early12', 'early8', 'early4')
gv = max(b / ERR[k].mean() for k in _CAND)
bestk = max(_CAND, key=lambda k: b / ERR[k].mean())
print(f'\n🏆 НАЙКРАЩИЙ РЕАЛІЗОВНИЙ ВАРІАНТ: {bestk} ⟹ ×{gv:.4f} дисперсії')
print(f'\n📈 У БАЛІ (дисперсія 82.5% MSE, зміщення 17.5%):')
print(f'   ×{1/(0.825/gv + 0.175):.4f}  ⟹  1.873e-07 → {1.873e-07/(1/(0.825/gv+0.175)):.4e}')
print(f'   (лідер 5.102e-08)')
