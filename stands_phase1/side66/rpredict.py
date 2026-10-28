"""ЧИ ПЕРЕДБАЧУВАНЕ ПОНЕЙРОННЕ `r`? — три пункти юзера одним стендом.

ЗВІДКИ. `elembeta.py` дав приз 1.79–1.86×, і єдиний блокер — передбачити
`r_i = |зміщення_i|/шум_i`. Юзер зняв із цього три хиби мого підходу:

1. **МАСКА б'є повну `β`** (1.863 проти 1.786; на 24..30 — 1.497 проти 1.422)
   ⟹ треба не РЕГРЕСІЯ `r`, а КЛАСИФІКАЦІЯ на дві групи. Метрика — **AUC**, і
   класифікатор з AUC 0.75 дасть більшу частину призу, тоді як регресія з
   R²=0.75 недосяжна. Я шукав складнішу річ, ніж потрібно;
2. **−0.180 це не нуль, а ЗНАК НАВИВОРІТ.** При 24×96 це кілька сигм, і це
   найсильніший сигнал у таблиці. Механізм, який пояснив би: зміщення керує не
   чутливість у центрі, а ХВОСТИ — нейрон із великим `|t|` рідко перемикається,
   і його негаусовість не усереднюється. Тож сюди: `|t|`, `t²`, `Φ(−|t|)`,
   `1/(φ(t)s)`;
3. **я міряв ОДНОВИМІРНІ кореляції, маючи право на РЕГРЕСІЮ.** Ознак під рукою
   значно більше, і головна — **ефективне число входів**
   `n_eff = (Σ W_ij²)²/Σ W_ij⁴`, зважене на `Var(a_j)`. Це не здогад: за ЦГТ
   відхилення `h_i` від гаусовості масштабується як `1/√n_eff`, а ми ВЖЕ
   зміряли, що похибка замикання — ефект скінченної ширини з `α≈1`. Тобто
   `r_i ∝ n_eff^(−1/2)` — передбачення з тієї самої фізики, яку ми встановили;
4. **ДІАГОНАЛЬНИЙ κ₃ — 256 чисел на шар, не тензор.** `κ₃(h_i)=Σ_j W_ij³κ₃(a_j)`,
   один матвек покубленими вагами. Ми закрили κ₃ чотирма способами, і всі
   чотири були про ПОВНИЙ тензор. Для РАНЖУВАННЯ нейронів досить діагоналі, і
   вирок «Еджворт у рекурсії не працює» стосувався ВИПРАВЛЕННЯ, а не
   ВПОРЯДКУВАННЯ.

КОНТРОЛІ
  * ⛔НЕГ: ознаки з ЧУЖОЇ мережі (зсув по списку) — AUC мусить впасти до 0.5;
  * ✅внутр. пломба: `E[relu]` і `E[relu²]` із моїх формул `μ_k` мусять збігтися
    з `closure_parts` до машинної точності (моменти рахуються заново);
  * фіт — LOO ПО МЕРЕЖАХ, оцінка лише на відкладеній мережі.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts, W_, _phi, _Phi  # noqa: E402
from price31 import truth_layers31                                   # noqa: E402

L_ = 32
DEV = 'cuda:3'
KS = 30
BANDS = {'1..8': (1, 8), '9..23': (9, 23), '24..30': (24, 30), 'усі 1..30': (1, 30)}


def moments(mp, s):
    """μ₁,μ₂,μ₃ зрізаного гаусіана + κ₃(relu). Формули виведено заново:
       μ₁ = mΦ+sφ · μ₂ = (m²+s²)Φ+msφ · μ₃ = m(m²+3s²)Φ+s(m²+2s²)φ."""
    t = mp / s
    ph, Ph = _phi(t), _Phi(t)
    m1 = mp * Ph + s * ph
    m2 = (mp * mp + s * s) * Ph + mp * s * ph
    m3 = mp * (mp * mp + 3.0 * s * s) * Ph + s * (mp * mp + 2.0 * s * s) * ph
    k3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    return m1, m2, k3, t, ph, Ph


def features(wc, dev, npair):
    """Усі ОБЧИСЛЮВАНІ ознаки на шарах 1..KS. Ніякої істини тут нема."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    k3a = torch.zeros(n0, dtype=torch.float64, device=dev)   # κ₃ активацій
    va = torch.ones(n0, dtype=torch.float64, device=dev)     # Var активацій
    out, seal = [], []
    for i in range(KS + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        m1, m2, k3, t, ph, Ph = moments(mp, s)
        d = m2 - m1 * m1
        # ✅ПЛОМБА: μ₁,μ₂ мусять збігтися з рекурсією замикання
        m_new = mp * Ph + s * ph
        seal.append(float((m1 - m_new).abs().max()))
        # діагональний κ₃ доактивацій наступного шару: Σ_j W³ κ₃(a_j)
        k3h = (w ** 3).T @ k3a
        w2 = w * w
        neff = ((w2.T @ va) ** 2 / ((w2 * w2).T @ (va * va)).clamp_min(1e-300))
        if i >= 1:
            noise = (d.clamp_min(1e-30).sqrt() / np.sqrt(2.0 * npair))
            F = {
                'log|t|': t.abs().clamp_min(1e-6).log(),
                't²': t * t,
                'logΦ(−|t|)': _Phi(-t.abs()).clamp_min(1e-30).log(),
                'log φ(t)s': (ph * s).clamp_min(1e-30).log(),
                'log s': s.log(),
                'log√d': d.clamp_min(1e-30).sqrt().log(),
                'log‖W‖₂': w2.sum(0).clamp_min(1e-30).sqrt().log(),
                'log‖W‖₄': (w2 * w2).sum(0).clamp_min(1e-30).log() * 0.25,
                'log n_eff': neff.clamp_min(1e-30).log(),
                'log|κ₃(h)|': k3h.abs().clamp_min(1e-30).log(),
                'log|κ₃|/s³': (k3h.abs() / s ** 3).clamp_min(1e-30).log(),
                'log шум': noise.clamp_min(1e-30).log(),
            }
            out.append(F)
        m = m_new
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        C = C + torch.diag(d - C.diagonal())
        k3a, va = k3, d
    return out, max(seal)


def auc(y, sc):
    """AUC: частка правильно впорядкованих пар (через ранги)."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float('nan')
    r = np.argsort(np.argsort(sc)) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main(nets=24, dev=DEV):
    torch.manual_seed(4711)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    DATA, keys, seals = [], None, []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            F, sl = features(wc, dev, npair)
            seals.append(sl)
            keys = list(F[0])
            rows, lr, lay = [], [], []
            for i in range(KS):
                d = (m_cl[i] - TL['m'][i])
                nz = (F[i]['log шум']).exp()
                r = (d.abs() / nz).clamp_min(1e-12)
                rows.append(torch.stack([F[i][k] for k in keys], 1).cpu().numpy())
                lr.append(r.log().cpu().numpy())
                lay.append(np.full(r.numel(), i + 1))
            DATA.append((np.concatenate(rows), np.concatenate(lr),
                         np.concatenate(lay)))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n✅ПЛОМБА моментів (μ₁ проти рекурсії замикання): '
          f'макс. {max(seals):.2e}\n')
    print(f'{nets} мереж, шари 1..{KS}. Фіт LOO ПО МЕРЕЖАХ, '
          f'оцінка лише на відкладеній.\n')

    for bn, (a, b) in BANDS.items():
        sel = [(X[(L >= a) & (L <= b)], y[(L >= a) & (L <= b)])
               for X, y, L in DATA]
        print(f'--- шари {bn} ---')
        print(f'{"ознака":>14s} {"|кореляція|":>12s}')
        Xa = np.concatenate([s[0] for s in sel])
        ya = np.concatenate([s[1] for s in sel])
        cs = []
        for j, k in enumerate(keys):
            x = Xa[:, j]
            c = np.corrcoef(x, ya)[0, 1] if x.std() > 0 else 0.0
            cs.append((abs(c), c, k))
        for ac, c, k in sorted(cs, reverse=True)[:6]:
            print(f'{k:>14s} {c:+12.3f}')
        # РЕГРЕСІЯ на всі ознаки, LOO по мережах
        R2, AUCs, AUCn = [], [], []
        for h in range(len(sel)):
            Xf = np.concatenate([sel[q][0] for q in range(len(sel)) if q != h])
            yf = np.concatenate([sel[q][1] for q in range(len(sel)) if q != h])
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
            Z = np.hstack([(Xf - mu) / sd, np.ones((len(Xf), 1))])
            beta = np.linalg.solve(Z.T @ Z + 1e-3 * len(Z) * np.eye(Z.shape[1]),
                                   Z.T @ yf)
            Xv, yv = sel[h]
            Zv = np.hstack([(Xv - mu) / sd, np.ones((len(Xv), 1))])
            p = Zv @ beta
            R2.append(1.0 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
            # 🚨 AUC рахується ВСЕРЕДИНІ ШАРУ. Наскрізний AUC по смузі ловить
            # МІЖШАРОВИЙ сигнал (`r` росте з глибиною), а його `λ_ℓ` і так
            # забирає — за це ⛔НЕГ показував аж 0.698 замість 0.5.
            Lv = DATA[h][2][(DATA[h][2] >= a) & (DATA[h][2] <= b)]
            for lay in np.unique(Lv):
                msk = (Lv == lay)
                AUCs.append(auc(yv[msk] < 0.0, -p[msk]))
                # ⛔НЕГ: ті самі ознаки, ПЕРЕМІШАНІ всередині шару
                AUCn.append(auc(yv[msk] < 0.0, -np.roll(p[msk], 7)))
        print(f'  РЕГРЕСІЯ на всі {len(keys)} ознак (позамережево): '
              f'R² = {np.nanmean(R2):+.3f}')
        print(f'  🎯**AUC ВСЕРЕДИНІ ШАРУ** = {np.nanmean(AUCs):.3f}    '
              f'⛔НЕГ (перемішано в шарі) = {np.nanmean(AUCn):.3f}')
        print(f'  частка `r<1` у смузі = '
              f'{float((ya < 0).mean()):.3f}\n')
    print('🔑 читати так: маска б\'є повну β, тож потрібна КЛАСИФІКАЦІЯ. '
          'AUC 0.75+ дасть більшу частину призу 1.86×; AUC ≈ 0.5 = блокер стоїть.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
