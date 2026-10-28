"""РЕКУРСИВНЕ ЗМІЩЕННЯ + БЕЗРОЗМІРНІ ОЗНАКИ — пункт 3 юзера, і виведення джерела.

ЗВІДКИ (дві поправки юзера).
1. **`r` безрозмірне ⟹ ознаки теж мусять бути.** Сирий `κ₃` має розмірність `s³`,
   тож кореляція −0.421 могла ловити `1/σ`, а не негаусовість. Правильна
   величина — скошеність `γ₁ = κ₃/s³`.
2. **Предиктор мусить бути РЕКУРСИВНИМ**: на шарі 1 `r≈0.09`, тобто зміщення не
   успадковується від входу — воно НАРОДЖУЄТЬСЯ на кожному шарі й ПЕРЕНОСИТЬСЯ.
   Форма юзера: `b̂_ℓ = Φ(t_ℓ)⊙(W_ℓᵀ b̂_{ℓ−1}) + γ_ℓ·джерело_ℓ`.

🔑 ДЖЕРЕЛО НЕ ТРЕБА ВГАДУВАТИ — ВОНО ВИВОДИТЬСЯ. Замикання бере точні моменти
зрізаного ГАУСІАНА, тож локальна похибка = поправка Еджворта за 3-м кумулянтом:
    `E[f(h)] ≈ E[f(Z)] + (κ₃/6)·E[f(Z)·He₃(z)]/s³`,  `z=(Z−m)/s`.
Для `f=relu`: `relu(m+su) = s·(u+t)₊`, тож потрібен
    `I(t) = ∫_{−t}^∞ (u+t)(u³−3u)φ(u)du`.
Через `∫u^kφ`: `I(t) = 3Φ−(t³+3t)φ + t(t²+2)φ − 3(Φ−tφ) − 3tφ = **−t·φ(t)**`.
⟹ **зміщення замикання (замикання мінус істина):**
    `b_i = + κ₃(h_i)·m_i·φ(t_i) / (6·s_i³)  =  γ₁(h_i)·t_i·φ(t_i)·s_i/6`.
Це ПІДПИСАНА величина і вона ✅безрозмірно-узгоджена: `γ₁` безрозмірне, решта
має одиниці активації.

🎓 І ВОНА ПОЯСНЮЄ ЗНАК НАВИВОРІТ. Мій кандидат С66 був `φ(t)·s` — тобто ця сама
формула БЕЗ множників `γ₁·t`. `κ₃` знакозмінний ⟹ пропущений множник міняє знак
по нейронах, і одновимірна кореляція з `|r|` виходить від'ємна. −0.180 був не
«форма навиворіт», а «формула недописана».

ЩО МІРЯЄМО
  * ✅ПЛОМБА: `I(t) = −tφ(t)` проти прямої квадратури;
  * 🎯 ПІДПИСАНА кореляція `b̂` з ІСТИННИМ `b` — те, чого я не міряв жодного разу
    (усі 5 кандидатів були МОДУЛІ проти `log r`);
  * джерело окремо проти джерело+перенос — чи додає рекурсія;
  * повна таблиця кореляцій: СИРІ ознаки проти ЦЕНТРОВАНИХ У ШАРІ (маска працює
    ВСЕРЕДИНІ шару, `λ_ℓ` і так забирає рівень шару);
  * регресія + AUC усередині шару, ⛔НЕГ — перемішування в шарі.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts, W_, _phi, _Phi  # noqa: E402
from price31 import truth_layers31                                   # noqa: E402
from rpredict import moments, auc                                    # noqa: E402

L_ = 32
DEV = 'cuda:2'
KS = 30
BANDS = {'1..8': (1, 8), '9..23': (9, 23), '24..30': (24, 30),
         'усі 1..30': (1, 30)}


def seal_I():
    """✅ПЛОМБА виведення: `I(t) = ∫_{−t}^∞ (u+t)(u³−3u)φ(u)du` = −t·φ(t)."""
    x, w = np.polynomial.legendre.leggauss(400)
    out = []
    for t in (-2.0, -0.5, 0.0, 0.7, 1.9, 3.3):
        lo, hi = -t, -t + 14.0
        h = 0.5 * (hi - lo)
        u = h * x + 0.5 * (hi + lo)
        f = (u + t) * (u ** 3 - 3.0 * u) * np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
        num = float((f * w).sum() * h)
        ana = -t * np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
        out.append(abs(num - ana))
    return max(out)


def features2(wc, dev, npair):
    """Ознаки + ПІДПИСАНЕ передбачення зміщення (джерело та рекурсія)."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    k3a = torch.zeros(n0, dtype=torch.float64, device=dev)
    va = torch.ones(n0, dtype=torch.float64, device=dev)
    bh = torch.zeros(n0, dtype=torch.float64, device=dev)   # b̂ активацій (рекурс)
    out = []
    for i in range(KS + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        m1, m2, k3, t, ph, Ph = moments(mp, s)
        d = m2 - m1 * m1
        k3h = (w ** 3).T @ k3a                       # діагональний κ₃ доактивацій
        g1 = k3h / s ** 3                            # 🔑 СКОШЕНІСТЬ (безрозмірна)
        src = g1 * t * ph * s / 6.0                  # 🔑 виведене ДЖЕРЕЛО (підписане)
        dmp = bh @ w                                 # успадкований зсув середнього
        bh = Ph * dmp + src                          # рекурсія юзера, γ_ℓ ≡ 1
        w2 = w * w
        neff = ((w2.T @ va) ** 2 / ((w2 * w2).T @ (va * va)).clamp_min(1e-300))
        if i >= 1:
            nz = (d.clamp_min(1e-30).sqrt() / np.sqrt(2.0 * npair))
            lg = lambda z: z.abs().clamp_min(1e-30).log()          # noqa: E731
            F = {
                # --- безрозмірні ---
                'log|t|': lg(t), 't²': t * t,
                'logΦ(−|t|)': _Phi(-t.abs()).clamp_min(1e-30).log(),
                'log|γ₁|': lg(g1),
                'log n_eff': neff.clamp_min(1e-30).log(),
                'log(s/шум)': (s / nz).log(),
                'log(√d/s)': (d.clamp_min(1e-30).sqrt() / s).log(),
                'log(‖W‖₄/‖W‖₂)': ((w2 * w2).sum(0).clamp_min(1e-30) ** 0.25
                                   / w2.sum(0).clamp_min(1e-30).sqrt()).log(),
                # --- 🔑 виведені предиктори зміщення, у одиницях шуму ---
                'log|джерело|/шум': lg(src / nz),
                'log|рекурс|/шум': lg(bh / nz),
                'log|перенос|/шум': lg(Ph * dmp / nz),
                # --- сирі (розмірні) — лишені для аудиту юзера ---
                'log s': s.log(), 'log√d': d.clamp_min(1e-30).sqrt().log(),
                'log φ(t)s': (ph * s).clamp_min(1e-30).log(),
                'log|κ₃(h)|': lg(k3h),
            }
            out.append((F, src, bh.clone(), Ph * dmp, nz))
        m = m1
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        C = C + torch.diag(d - C.diagonal())
        k3a, va = k3, d
    return out


def main(nets=24, dev=DEV):
    torch.manual_seed(1618)
    print(f'✅ПЛОМБА виведення I(t)=−tφ(t): макс. нев\'язка {seal_I():.2e}\n')
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    DATA, keys = [], None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            FF = features2(wc, dev, npair)
            keys = list(FF[0][0])
            X, y, lay, bt, bs, br, bp = [], [], [], [], [], [], []
            for i in range(KS):
                F, src, rec, tr, nz = FF[i]
                b = (m_cl[i] - TL['m'][i])                 # 🎯 ПІДПИСАНЕ зміщення
                X.append(torch.stack([F[k] for k in keys], 1).cpu().numpy())
                y.append((b.abs() / nz).clamp_min(1e-12).log().cpu().numpy())
                lay.append(np.full(b.numel(), i + 1))
                bt.append((b / nz).cpu().numpy())
                bs.append((src / nz).cpu().numpy())
                br.append((rec / nz).cpu().numpy())
                bp.append((tr / nz).cpu().numpy())
            DATA.append((np.concatenate(X), np.concatenate(y),
                         np.concatenate(lay), np.concatenate(bt),
                         np.concatenate(bs), np.concatenate(br),
                         np.concatenate(bp)))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    def cin(v, L, N):
        """центрування у клітинці (МЕРЕЖА, ШАР) — саме там працює маска, і саме
        цей рівень забирає `λ_ℓ`. Центрування лише по шару лишало б МІЖМЕРЕЖНУ
        варіацію, якої маска теж не бачить."""
        o = np.empty_like(v)
        key = N.astype(np.int64) * 1000 + L.astype(np.int64)
        for q in np.unique(key):
            k = key == q
            o[k] = v[k] - v[k].mean()
        return o

    print(f'\n{nets} мереж, шари 1..{KS}. Фіт LOO ПО МЕРЕЖАХ.\n')
    for bn, (a, b) in BANDS.items():
        sel = [tuple(z[(D[2] >= a) & (D[2] <= b)] if z.ndim == 1
                     else z[(D[2] >= a) & (D[2] <= b)] for z in D)
               for D in DATA]
        Xa = np.concatenate([s[0] for s in sel])
        ya = np.concatenate([s[1] for s in sel])
        La = np.concatenate([s[2] for s in sel])
        Na = np.concatenate([np.full(len(s[1]), q) for q, s in enumerate(sel)])
        print(f'--- шари {bn} ---')
        # 🎯 ПІДПИСАНА кореляція (те, чого не міряв жодного разу)
        tsig = cin(np.concatenate([s[3] for s in sel]), La, Na)
        for nm, j in (('джерело (Еджворт)', 4), ('рекурс = джерело+перенос', 5),
                      ('лише перенос', 6)):
            p_ = cin(np.concatenate([s[j] for s in sel]), La, Na)
            cw = np.corrcoef(p_, tsig)[0, 1]
            print(f'  🎯 ПІДПИСАНА corr({nm}, b_іст) у клітинці: {cw:+.3f}')
        pn = cin(np.roll(np.concatenate([s[5] for s in sel]), 7), La, Na)
        print(f'  ⛔НЕГ (рекурс, перемішаний зсувом): '
              f'{np.corrcoef(pn, tsig)[0, 1]:+.3f}')
        # 🎯 predictor БЕЗ ЖОДНОГО ФІТУ: r̂ = |b̂_рекурс|/шум
        z0, zn = [], []
        for q, s in enumerate(sel):
            for lq in np.unique(s[2]):
                k = s[2] == lq
                z0.append(auc(s[1][k] < 0.0, -np.abs(s[5][k])))
                zn.append(auc(s[1][k] < 0.0, -np.roll(np.abs(s[5][k]), 7)))
        print(f'  🎯AUC З ГОЛОГО `|b̂|` (нуль фітованих параметрів) = '
              f'{np.nanmean(z0):.3f}   ⛔НЕГ = {np.nanmean(zn):.3f}')
        print(f'{"ознака":>18s} {"сира":>9s} {"центр. у клітинці":>18s}')
        yc = cin(ya, La, Na)
        for j, k in enumerate(keys):
            x = Xa[:, j]
            c0 = np.corrcoef(x, ya)[0, 1] if x.std() > 0 else 0.0
            xc = cin(x, La, Na)
            c1 = np.corrcoef(xc, yc)[0, 1] if xc.std() > 0 else 0.0
            print(f'{k:>18s} {c0:+9.3f} {c1:+18.3f}')
        R2, A1, AN = [], [], []
        for h in range(len(sel)):
            Xf = np.concatenate([sel[q][0] for q in range(len(sel)) if q != h])
            yf = np.concatenate([sel[q][1] for q in range(len(sel)) if q != h])
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
            Z = np.hstack([(Xf - mu) / sd, np.ones((len(Xf), 1))])
            be = np.linalg.solve(Z.T @ Z + 1e-3 * len(Z) * np.eye(Z.shape[1]),
                                 Z.T @ yf)
            Xv, yv, Lv = sel[h][0], sel[h][1], sel[h][2]
            p = np.hstack([(Xv - mu) / sd, np.ones((len(Xv), 1))]) @ be
            R2.append(1.0 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
            for q in np.unique(Lv):
                k = Lv == q
                A1.append(auc(yv[k] < 0.0, -p[k]))
                AN.append(auc(yv[k] < 0.0, -np.roll(p[k], 7)))
        print(f'  РЕГРЕСІЯ на {len(keys)} ознак: R² = {np.nanmean(R2):+.3f}')
        print(f'  🎯AUC ВСЕРЕДИНІ ШАРУ = {np.nanmean(A1):.3f}   '
              f'⛔НЕГ = {np.nanmean(AN):.3f}\n')
    print('🔑 читати так: якщо ПІДПИСАНА кореляція джерела помітно НЕ нуль — '
          'механізм зміщення знайдено, і далі це вже інженерія. Якщо «рекурс» '
          'б\'є «джерело» — форма юзера підтверджена. Стовпчик «центр. у шарі» '
          'проти «сира» показує, скільки в ознаці було МАСШТАБУ, а не форми.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
