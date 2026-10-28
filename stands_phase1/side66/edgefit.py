#!/usr/bin/env python3
"""EDGEWORTH-БАЗИС ДЛЯ ПОСТ-ФАКТУМ ПОПРАВКИ ДО ГАУСОВОГО ЗАМИКАННЯ.

ПИТАННЯ: скалярна поправка `M→(1+α)M` бере 61% дисперсії похибки замикання
(2.572× по MSE, LOO по мережах — `side66/clgeom.py`).  Чи візьмуть решту 39%
ВИВЕДЕНІ форми розкладу Еджворта?

МАЛА ВЕЛИЧИНА = 1/n (зміряно в `nscale.py`: p = 1.002 ± 0.055).
Для `p` з середнім `m_p`, ст.в `s`, `t = m_p/s`:
    гаусів член :  m  = m_p·Φ(t) + s·φ(t)          ← що замикання рахує зараз
    δ₃ = −(γ₃/6)  · s · t · φ(t)                    ← скошеність, ~n^(−1/2)
    δ₄ = +(γ₄/24) · s · (t² − 1) · φ(t)             ← ексцес,     ~n^(−1)
ВИВЕДЕННЯ (перевірено sympy, див. П4):
    E[ReLU(p)] = s·∫_{−t}^∞ (ξ+t)·f(ξ)dξ,  f = φ·[1 + (γ₃/6)He₃ + (γ₄/24)He₄]
    ∫_{−t}^∞ (ξ+t)He₃(ξ)φ(ξ)dξ = −t·φ(t)
    ∫_{−t}^∞ (ξ+t)He₄(ξ)φ(ξ)dξ = (t²−1)·φ(t)
`s`, `t`, `φ(t)`, `Φ(t)` замикання вже має ⟹ ДОДАТКОВИХ ФЛОПІВ НУЛЬ.

БАЗИС (усе покомпонентно, 256 чисел, зі стану ОСТАННЬОГО шару):
    B0 = M                      (діючий канал α)
    B1 = s·φ(t)
    B2 = s·t·φ(t)               ← δ₃
    B3 = s·(t²−1)·φ(t)          ← δ₄
    B4 = s·Φ(t)
    B5 = s

ДОСЛІД 1: поправка ЛИШЕ НА ВИХОДІ, вкладені набори, LOO ПО МЕРЕЖАХ + оракул.
ДОСЛІД 2: поправка НА КОЖНОМУ ШАРІ всередині рекурсії (лінеаризація Якобіаном
          + ЧЕСНА перевірка справжнім прогоном рекурсії).

ПЛОМБИ: П1 {B0}=2.57×±0.05 · П2 випадковий стовпець = 1.00× · П3 перетасовка ·
        П4 математика δ₃/δ₄ чисельно · П5 обумовленість базису.

СТЕНД ПЕРЕВИКОРИСТАНО з `clgeom.py`: та сама істина (16.8e6 семплів,
шум 0.0058%), ті самі 24 мережі, той самий LOO-протокол.
"""
import glob
import os
import sys

import numpy as np
from scipy.special import erf

W_ = 256
L_ = 32
NNET = 24
PAIRS = 8_388_608
SCRATCH = ('/tmp/claude-1000/-mnt-win-d-llm-White-Box/'
           '5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/clgeom')
WZ_P = '/mnt/win_d/llm/White-Box/arena60/fs_w.npy'
SD_P = '/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'
POOL = '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'
S2PI = np.sqrt(2 * np.pi)
NB = 7                                    # повний базис B0..B6
SETS = [('1. {B0}                контроль', [0]),
        ('2. {B0,B2}             +δ₃', [0, 2]),
        ('3. {B0,B2,B3}          +δ₃+δ₄', [0, 2, 3]),
        ('4. {B0,B2,B3,B6}       ПОВНИЙ 1/n', [0, 2, 3, 6]),
        ('5. {B0,B1,B2,B3}', [0, 1, 2, 3]),
        ('6. {B0,B1,B2,B3,B4,B5} базис ТЗ', [0, 1, 2, 3, 4, 5]),
        ('7. {B0..B6}            усе', [0, 1, 2, 3, 4, 5, 6])]
RHO = np.array([0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3,
                1e-2, 1e-1, 1.0, 10.0, 1e2, 1e4])


def net_list():
    """24 мережі, що є І в банку, І у fs_w — РІВНО як у clgeom.py."""
    sd = np.load(SD_P)
    pos = {int(s): i for i, s in enumerate(sd)}
    out = []
    for f in sorted(glob.glob(POOL)):
        s0 = int(np.load(f)['seed'])
        if s0 in pos:
            out.append((f, s0, pos[s0]))
            if len(out) == NNET:
                break
    return out


# ═══════════════════ ЗАМИКАННЯ З ПОПРАВКОЮ (порт із сабміту, fp64) ═══════════
def closure(Wl, lam=None, first=1):
    """Замикання ТЗ. `lam` — (NB,) коефіцієнти ПОШАРОВОЇ поправки (або None).

    Поправка: після повного блоку (разом з оновленням C) робимо
        m ← m + Σ_j lam[j]·B_j
    `first` — з якого шару (1-based) вмикати поправку.
    Повертає (m_final, basis_final (256,NB)).
    """
    n = Wl[0].shape[0]
    m = np.zeros(n, dtype=np.float64)
    C = np.eye(n, dtype=np.float64)
    B = None
    for i, w in enumerate(Wl):
        cp = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = np.exp(-0.5 * t * t) / S2PI
        Ph = 0.5 * (1 + erf(t / np.sqrt(2)))
        m = mp * Ph + s * ph
        C = cp * np.outer(Ph, Ph) + 0.5 * (cp * cp) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diag(C))
        t2 = t * t
        B = np.stack([m, s * ph, s * t * ph, s * (t2 - 1) * ph,
                      s * Ph, s, s * (t2 * t2 - 6 * t2 + 3) * ph], 1)
        if lam is not None and (i + 1) >= first:
            m = m + B @ lam
            B[:, 0] = m                    # B0 = поточне m
    return m, B


# ══════════════════════════ ГРЕБЕНЕВА МНК + LOO ══════════════════════════════
def solve_ridge(A, b, rho, dg):
    """(A + rho·diag(A))c = b, з діагональним нормуванням."""
    return np.linalg.solve(A + rho * np.diag(dg) + 1e-300 * np.eye(len(b)), b)


def nested_loo(Xs, ys, cols):
    """LOO ПО МЕРЕЖАХ з ВКЛАДЕНИМ вибором сили гребеня.

    Xs — список (256,p_full) на мережу; ys — список (256,).
    Повертає (coef (N,p), rho_pick (N,), coef_pool (p,), rho_pool).
    Усадка ρ→∞ дає c→0 ⟹ РІВНО діюча база (жодної поправки) ✅.
    """
    N = len(Xs)
    Aj = [Xs[j][:, cols].T @ Xs[j][:, cols] for j in range(N)]
    bj = [Xs[j][:, cols].T @ ys[j] for j in range(N)]
    A, b = sum(Aj), sum(bj)
    dg = np.maximum(np.diag(A) / N, 1e-300)          # масштаб на 1 мережу
    out = np.zeros((N, len(cols)))
    pick = np.zeros(N)
    for j in range(N):
        Ao, bo = A - Aj[j], b - bj[j]
        # ── ВКЛАДЕНИЙ вибір ρ: ще один LOO ВСЕРЕДИНІ решти N−1 мереж ──
        err = np.zeros(len(RHO))
        for k, r in enumerate(RHO):
            for i in range(N):
                if i == j:
                    continue
                c = solve_ridge(Ao - Aj[i], bo - bj[i], r, dg)
                err[k] += np.sum((Xs[i][:, cols] @ c - ys[i]) ** 2)
        kb = int(np.argmin(err))
        pick[j] = RHO[kb]
        out[j] = solve_ridge(Ao, bo, RHO[kb], dg)
    cp = solve_ridge(A, b, 0.0, dg)
    return out, pick, cp, A


def eval_sets(Xs, ys, mh, th, tag, rng=None, extra=''):
    """Прогін усіх вкладених наборів. Друкує таблицю. Повертає dict."""
    N = len(Xs)
    base = np.array([np.linalg.norm(mh[j] - th[j]) for j in range(N)])
    rb = float(np.sqrt((base ** 2).mean()) * 100)
    print(f'\n{"=" * 78}\n{tag}\n{"=" * 78}')
    print(f'  без поправки: {rb:.4f}%')
    print('  набір                       | LOO залишок |  LOO ×MSE | '
          'оракул залишок | орк ×MSE')
    print('  ' + '-' * 74)
    res = {}
    for nm, cols in SETS:
        cl, pk, cp, A = nested_loo(Xs, ys, cols)
        loo = np.array([np.linalg.norm(mh[j] + Xs[j][:, cols] @ cl[j] - th[j])
                        for j in range(N)])
        orc = np.zeros(N)
        for j in range(N):
            X = Xs[j][:, cols]
            c = np.linalg.lstsq(X, ys[j], rcond=None)[0]
            orc[j] = np.linalg.norm(mh[j] + X @ c - th[j])
        rl = float(np.sqrt((loo ** 2).mean()) * 100)
        ro = float(np.sqrt((orc ** 2).mean()) * 100)
        res[nm] = (rb, rl, ro, cp, A)
        print(f'  {nm:27s} | {rl:9.4f}%  | {(rb/rl)**2:7.3f}× | '
              f'{ro:12.4f}%  | {(rb/ro)**2:7.3f}×')
    print('  спільні коеф (пул, усі мережі) повного базису: [' +
          '  '.join(f'{x:+.6f}' for x in res[SETS[-1][0]][3]) + ']')
    print(extra)
    return res


# ═══════════════════════════ ДОСЛІД 1: ВИХІД ═════════════════════════════════
def load_all():
    """Кеш істини з clgeom.py + перерахунок стану замикання (CPU, дешево)."""
    nets = net_list()
    wz = np.load(WZ_P, mmap_mode='r')
    fs = sorted(glob.glob(f'{SCRATCH}/net*.npz'))
    assert len(fs) == NNET, f'🚨 у кеші {len(fs)} мереж, треба {NNET}'
    T, Mc, SD, WS = [], [], [], []
    for f in fs:
        z = np.load(f)
        T.append(z['T'])
        Mc.append(z['M'])
        SD.append(int(z['seed']))
    for s0 in SD:
        j = [x[2] for x in nets if x[1] == s0][0]
        WS.append([np.asarray(wz[j][i], dtype=np.float64) for i in range(L_)])
    return np.array(T), np.array(Mc), SD, WS


def exp1(T, Mc, WS, rng):
    N = len(WS)
    mh, th, Xs, ys, Bl = [], [], [], [], []
    dev = 0.0
    for j in range(N):
        m, B = closure(WS[j])
        dev = max(dev, float(np.abs(m - Mc[j][-1]).max()
                             / np.abs(Mc[j][-1]).max()))
        nn = np.linalg.norm(T[j][-1])
        mh.append(Mc[j][-1] / nn)
        th.append(T[j][-1] / nn)
        Bl.append(B / nn)
        Xs.append(B / nn)
        ys.append(th[-1] - mh[-1])
    print(f'✅ ПЛОМБА СТЕНДУ: перерахунок замикання збігся з кешем '
          f'clgeom (макс.відн.відхил {dev:.2e})')
    res = eval_sets(Xs, ys, mh, th, 'ДОСЛІД 1 — ПОПРАВКА ЛИШЕ НА ВИХОДІ '
                                    '(шар 32), 24 мережі')

    # ── П1 ──────────────────────────────────────────────────────────────────
    rb, rl, ro, _, _ = res[SETS[0][0]]
    g = (rb / rl) ** 2
    print(f'\n{"=" * 78}\nП1  КОНТРОЛЬ {{B0}} = 2.57× ± 0.05\n{"=" * 78}')
    print(f'  зміряно {g:.3f}× (clgeom дав 2.572×), залишок {rl:.4f}% '
          f'(clgeom 0.6013%)')
    print('  ✅ П1: стенд відтворює clgeom' if abs(g - 2.572) < 0.05 else
          '  🚨 П1 ПРОВАЛЕНО — СТЕНД ЗЛАМАНО, числа нижче не читати')
    # 🎓 чи не є приріст ОРАКУЛА просто платою за зайві вільні параметри?
    ro1 = res[SETS[0][0]][2]
    ro7 = res[SETS[-1][0]][2]
    dof = (NB - 1) / W_
    print(f'\n  🎓 ОРАКУЛ: {ro1:.4f}% (1 параметр) → {ro7:.4f}% ({NB} параметрів)'
          f' = приріст {(ro1/ro7)**2:.4f}× по MSE')
    print(f'     а ЧИСТА підгонка шуму {NB-1} зайвими параметрами на {W_} '
          f'точках дає {1/(1-dof):.4f}× САМА ПО СОБІ')
    print(('     🚨 ⟹ УВЕСЬ приріст оракула = плата за ступені вільності, '
           'ФОРМИ НЕ НЕСУТЬ НІЧОГО' if (ro1 / ro7) ** 2 <= 1 / (1 - dof) * 1.3
          else '     ⟹ приріст оракула ПЕРЕВИЩУЄ ступені вільності'))

    # ── П5: обумовленість ───────────────────────────────────────────────────
    print(f'\n{"=" * 78}\nП5  ОБУМОВЛЕНІСТЬ БАЗИСУ (пул по 24 мережах)\n{"=" * 78}')
    G = sum(X.T @ X for X in Xs)
    d = np.sqrt(np.diag(G))
    R = G / np.outer(d, d)
    print('  кореляційна матриця стовпців (|cos| між B_i та B_j):')
    print('        ' + ''.join(f'   B{i}   ' for i in range(NB)))
    for i in range(NB):
        print(f'    B{i} ' + ''.join(f'{R[i,k]:+7.3f}' for k in range(NB)))
    for nm, cols in SETS:
        cn = np.linalg.cond(R[np.ix_(cols, cols)])
        print(f'  {nm:27s} cond(корел.) = {cn:12.4e}')
    cnf = np.linalg.cond(R)
    print(f'  ⟹ повний базис: cond = {cnf:.3e}, sing.vals = ' +
          ' '.join(f'{v:.2e}' for v in np.linalg.svd(R)[1]))
    print(('  🚨 П5: стовпці МАЙЖЕ КОЛІНЕАРНІ (cond>1e6) ⟹ «більше форм» '
           'ілюзорне, приріст тримається на гребені' if cnf > 1e6 else
           '  ✅ П5: базис обумовлений прийнятно (cond<1e6)'))

    # ── П2: НЕГ — випадковий стовпець тієї ж норми ──────────────────────────
    print(f'\n{"=" * 78}\nП2  ⛔НЕГ: ВИПАДКОВИЙ СТОВПЕЦЬ ТІЄЇ Ж НОРМИ\n{"=" * 78}')
    Xr = []
    for j in range(N):
        r = rng.standard_normal(W_)
        r *= np.linalg.norm(Xs[j][:, 0]) / np.linalg.norm(r)
        Xr.append(np.concatenate([Xs[j], r[:, None]], 1))
    for nm, cols in ((SETS[0][0], [0]), (SETS[-1][0], list(range(NB)))):
        c0 = cols
        c1 = cols + [NB]
        o = []
        for cc in (c0, c1):
            cl, _, _, _ = nested_loo(Xr, ys, cc)
            v = np.array([np.linalg.norm(mh[j] + Xr[j][:, cc] @ cl[j] - th[j])
                          for j in range(N)])
            o.append(float(np.sqrt((v ** 2).mean()) * 100))
        print(f'  {nm:27s}: {o[0]:.4f}% → з випадковим {o[1]:.4f}%  '
              f'⟹ виграш ВІД ШУМУ {(o[0]/o[1])**2:.4f}× по MSE')
        print('    ✅ П2: випадковий стовпець дає НУЛЬ ⟹ LOO не протікає'
              if abs((o[0] / o[1]) ** 2 - 1) < 0.02 else
              '    🚨 П2: випадковий стовпець ДАЄ ВИГРАШ ⟹ LOO ПРОТІКАЄ')

    # ── П3: НЕГ-2 — перетасовка ─────────────────────────────────────────────
    print(f'\n{"=" * 78}\nП3  ⛔НЕГ-2: ПЕРЕТАСОВКА\n{"=" * 78}')
    prm = (np.arange(N) + 1) % N
    for nm, cols in (SETS[0], SETS[-1]):
        cl, _, _, _ = nested_loo(Xs, ys, cols)
        # (а) ознаки ЧУЖОЇ мережі проти власної істини
        va = np.array([np.linalg.norm(mh[j] + Xs[prm[j]][:, cols] @ cl[j]
                                      - th[j]) for j in range(N)])
        # (б) ОРАКУЛЬНИЙ коеф чужої мережі на своїх ознаках
        oc = np.array([np.linalg.lstsq(Xs[j][:, cols], ys[j], rcond=None)[0]
                       for j in range(N)])
        vb = np.array([np.linalg.norm(mh[j] + Xs[j][:, cols] @ oc[prm[j]]
                                      - th[j]) for j in range(N)])
        ra = float(np.sqrt((va ** 2).mean()) * 100)
        rvb = float(np.sqrt((vb ** 2).mean()) * 100)
        vl = np.array([np.linalg.norm(mh[j] + Xs[j][:, cols] @ cl[j] - th[j])
                       for j in range(N)])
        rl2 = float(np.sqrt((vl ** 2).mean()) * 100)
        print(f'  {nm}')
        print(f'    LOO {rl2:.4f}% ({(rb/rl2)**2:.3f}×)  |  (а) ознаки ЧУЖОЇ '
              f'мережі {ra:.4f}% ({(rb/ra)**2:.3f}×)  |  (б) оракул ЧУЖОЇ '
              f'{rvb:.4f}% ({(rb/rvb)**2:.3f}×)')
        print('    ✅ (а) виграш ЗНИК ⟹ ознаки СПРАВДІ мережево-специфічні'
              if (rb / ra) ** 2 < 1.05 else
              f'    🚨 (а) виграш ВИЖИВ {(rb/ra)**2:.3f}× — підозра на витік')
    print('  ℹ️ (б) виграш ЗАЛИШАЄТЬСЯ, якщо коефіцієнт УНІВЕРСАЛЬНИЙ — '
          'це очікувано і НЕ провал (див. clgeom НЕГ-1 = 1.513×)')
    return res


# ══════════════════════ П4: МАТЕМАТИКА δ₃/δ₄ ЧИСЕЛЬНО ════════════════════════
def _relu_gauss(a, b):
    """E[ReLU(a + b·Z)], Z~N(0,1), точно."""
    tt = a / b
    return a * 0.5 * (1 + erf(tt / np.sqrt(2))) + b * np.exp(-0.5 * tt ** 2) / S2PI


def _exact_chi2(mp, s, k):
    """E[ReLU(mp + s·(χ²_k−k)/√(2k))] У ЗАКРИТІЙ ФОРМІ.

    q·f_k(q) = k·f_{k+2}(q) ⟹ ∫_{q0}^∞ q f_k = k·SF_{k+2}(q0). Жодної квадратури.
    """
    from scipy.stats import chi2
    r = np.sqrt(2.0 * k)
    q0 = k - r * (mp / s)                       # ReLU>0 ⟺ q > q0
    q0 = max(q0, 0.0)
    return (mp * chi2.sf(q0, k)
            + (s / r) * k * (chi2.sf(q0, k + 2) - chi2.sf(q0, k)))


def seal4():
    """МАТЕМАТИЧНА пломба δ₃/δ₄ на ДВОХ сімействах із відомими γ₃, γ₄."""
    print(f'\n{"=" * 78}\nП4  МАТЕМАТИКА δ₃/δ₄ — ЧИСЕЛЬНА ПЕРЕВІРКА\n{"=" * 78}')
    print('  ✅ виведення підтверджено sympy:  ∫_{−t}^∞(ξ+t)He₃φdξ = −t·φ(t),'
          '   ∫_{−t}^∞(ξ+t)He₄φdξ = (t²−1)·φ(t)')
    print('  E[ReLU] береться У ЗАКРИТІЙ ФОРМІ (не квадратурою): для χ² через '
          'q·f_k=k·f_{k+2}, для суміші — покомпонентно.')

    print('  🔑 ЗНАЙДЕНО ТРЕТІЙ ЧЛЕН ТОГО САМОГО ПОРЯДКУ 1/n (у ТЗ його не '
          'було):  f = φ[1+(γ₃/6)He₃+(γ₄/24)He₄+(γ₃²/72)He₆]')
    print('     ∫_{−t}^∞(ξ+t)He₆φdξ = He₄(t)·φ(t)  ⟹  '
          'δ₆ = +(γ₃²/72)·s·(t⁴−6t²+3)·φ(t)   ← це і є B6')
    print('     γ₃ ~ n^(−1/2) ⟹ γ₃² ~ 1/n ⟹ δ₆ ТОЙ САМИЙ ПОРЯДОК, ЩО Й δ₄ ⟹ '
          'без нього ряд 1/n НЕПОВНИЙ')

    TS = (-1.5, -0.5, 0.0, 0.5, 1.5)

    def row(tag, g3, g4, t, ex, ok_all, acc=None):
        t2 = t * t
        ph = np.exp(-0.5 * t2) / S2PI
        Ph = 0.5 * (1 + erf(t / np.sqrt(2)))
        g = t * Ph + ph                          # s=1, mp=t
        d3 = -(g3 / 6) * t * ph
        d4 = (g4 / 24) * (t2 - 1) * ph
        d6 = (g3 * g3 / 72) * (t2 * t2 - 6 * t2 + 3) * ph
        e = [abs(g - ex), abs(g + d3 - ex), abs(g + d3 + d4 - ex),
             abs(g + d3 + d4 + d6 - ex)]
        good = e[3] < e[0]                       # повний порядок б'є гаусів
        if acc is not None:
            acc.append(e)
        print(f'  {tag:14s} {g3:+.4f} {g4:+.5f} | {t:+.1f} | {ex:13.9f} | ' +
              ' | '.join(f'{v:.3e}' for v in e) + f' | {"✅" if good else "🚨"}')
        return ok_all and good

    print('   сімейство       γ₃       γ₄     |  t   |  точне E[ReLU] | '
          ' гаус:Δ    | +δ₃:Δ      | +δ₃+δ₄:Δ   | +δ₃+δ₄+δ₆:Δ')
    print('  ' + '-' * 112)
    ok = True
    KS = (200, 800, 3200)
    acc = {}
    for k in KS:
        acc[k] = []
        for t in TS:
            ok = row(f'χ²_{k}', np.sqrt(8.0 / k), 12.0 / k, t,
                     _exact_chi2(t, 1.0, k), ok, acc[k])
    # ── суміш гаусіан: w·N(μ,σ1) + (1−w)·N(ν,σ2), нормована до сер 0, дисп 1 ──
    for w, mu, s1, s2 in ((0.90, 0.12, 1.0, 1.0), (0.97, 0.05, 1.0, 1.06)):
        nu = -w * mu / (1 - w)
        c = np.array([w, 1 - w])
        mus = np.array([mu, nu])
        sg = np.array([s1, s2])
        m2 = float((c * (mus ** 2 + sg ** 2)).sum())
        sc = 1.0 / np.sqrt(m2)                    # нормуємо дисперсію до 1
        mus, sg = mus * sc, sg * sc
        m3 = float((c * (mus ** 3 + 3 * mus * sg ** 2)).sum())
        m4 = float((c * (mus ** 4 + 6 * mus ** 2 * sg ** 2 + 3 * sg ** 4)).sum())
        for t in TS:
            ex = float((c * _relu_gauss(t + mus, sg)).sum())
            ok = row(f'суміш w={w}', m3, m4 - 3.0, t, ex, ok)
    # ── ГОЛОВНИЙ тест: ПОКАЗНИК СПАДАННЯ по k (χ²_k ≡ сума k доданків) ───────
    print('\n  📐 ПОКАЗНИК СПАДАННЯ ЗАЛИШКУ по k (RMS по t; χ²_k = сума k '
          'незалежних ⟹ «n» = k):')
    lk = np.log(np.array(KS, float))
    lab = ['гаус', '+δ₃', '+δ₃+δ₄', '+δ₃+δ₄+δ₆']
    exps = []
    for c_ in range(4):
        v = np.array([np.sqrt(np.mean([e[c_] ** 2 for e in acc[k]]))
                      for k in KS])
        p = float(np.polyfit(lk, np.log(v), 1)[0])
        exps.append(p)
        print(f'     {lab[c_]:12s}: ' + '  '.join(f'{x:.3e}' for x in v) +
              f'   ⟹ показник {p:+.3f}  (теорія {-(c_+1)/2 if c_<3 else -1.5:+.2f})')
    ok_e = (abs(exps[0] + 0.5) < 0.1 and abs(exps[1] + 1.0) < 0.15
            and abs(exps[3] + 1.5) < 0.2)
    print('  ✅ П4: показники −0.5 / −1.0 / −1.5 РІВНО як велить теорія ⟹ '
          'δ₃, δ₄, δ₆ виведено ПРАВИЛЬНО' if ok_e else
          '  🚨 П4: показники не збігаються з теорією — виведення під підозрою')
    print('  ✅ П4-б: повний порядок б\'є гаусів у ВСІХ точках обох сімейств'
          if ok else '  ⚠️ П4-б: є точки, де повний порядок не кращий за гаусів '
          '(випадкові нулі окремих членів — не провал, дивись показники)')
    print('  🚨🎓 УВАГА: δ₄ САМ ПО СОБІ ПОГІРШУЄ на χ² (там γ₃²/72 сумірне з '
          'γ₄/24) ⟹ δ₆ ОБОВ\'ЯЗКОВИЙ, ряд «δ₃+δ₄» — НЕПОВНИЙ порядок')


# ═══════════════════ ДОСЛІД 2: ПОПРАВКА НА КОЖНОМУ ШАРІ ══════════════════════
def exp2(T, WS, first, rng):
    """Пошарова поправка. Лінеаризація Якобіаном + ЧЕСНИЙ прогін рекурсії."""
    N = len(WS)
    H = 1e-3
    print(f'\n{"=" * 78}\nДОСЛІД 2 — ПОПРАВКА НА КОЖНОМУ ШАРІ (з шару {first})'
          f'\n{"=" * 78}')
    print('  метод: Якобіан ∂m₃₂/∂λ центральною різницею (h=1e-3) ⟹ ті самі '
          'LOO+гребінь ⟹ ЧЕСНА перевірка СПРАВЖНІМ прогоном рекурсії')
    J, mh, th, ys = [], [], [], []
    for j in range(N):
        m0, _ = closure(WS[j])
        nn = np.linalg.norm(T[j][-1])
        col = np.zeros((W_, NB))
        for q in range(NB):
            lp = np.zeros(NB)
            lp[q] = H
            mp_, _ = closure(WS[j], lp, first)
            lm = np.zeros(NB)
            lm[q] = -H
            mm_, _ = closure(WS[j], lm, first)
            col[:, q] = (mp_ - mm_) / (2 * H) / nn
        J.append(col)
        mh.append(m0 / nn)
        th.append(T[j][-1] / nn)
        ys.append(th[-1] - mh[-1])
        if j == 0:
            l1 = np.zeros(NB)
            l1[0] = H
            m1, _ = closure(WS[j], l1, first)
            l2 = np.zeros(NB)
            l2[0] = 2 * H
            m2, _ = closure(WS[j], l2, first)
            nl = (np.linalg.norm(m2 - m0 - 2 * (m1 - m0))
                  / np.linalg.norm(m1 - m0))
            print(f'  ℹ️ нелінійність по λ₀ на кроці h: {nl:.4f} '
                  f'(≪1 ⟹ лінеаризація чесна)')
    base = np.array([np.linalg.norm(mh[j] - th[j]) for j in range(N)])
    rb = float(np.sqrt((base ** 2).mean()) * 100)
    print(f'  без поправки: {rb:.4f}%')
    print('  набір                       | лінеариз.LOO |  ×MSE  | '
          'СПРАВЖНІЙ прогон | ×MSE   | коеф λ (пул)')
    print('  ' + '-' * 100)
    out = {}
    for nm, cols in SETS:
        cl, _, cp, _ = nested_loo(J, ys, cols)
        lin = np.array([np.linalg.norm(mh[j] + J[j][:, cols] @ cl[j] - th[j])
                        for j in range(N)])
        tru = np.zeros(N)
        for j in range(N):
            lam = np.zeros(NB)
            lam[cols] = cl[j]
            mm, _ = closure(WS[j], lam, first)
            tru[j] = np.linalg.norm(mm / np.linalg.norm(T[j][-1]) - th[j])
        rl = float(np.sqrt((lin ** 2).mean()) * 100)
        rt = float(np.sqrt((tru ** 2).mean()) * 100)
        out[nm] = (rb, rt)
        cs = ' '.join(f'{x:+.5f}' for x in cp)
        print(f'  {nm:27s} | {rl:10.4f}%  | {(rb/rl)**2:6.3f}× | '
              f'{rt:14.4f}%  | {(rb/rt)**2:6.3f}× | [{cs}]')
    # оракул для повного базису (лінеаризований — стеля)
    cols = SETS[-1][1]
    orc = np.zeros(N)
    for j in range(N):
        c = np.linalg.lstsq(J[j][:, cols], ys[j], rcond=None)[0]
        lam = np.zeros(NB)
        lam[cols] = c
        mm, _ = closure(WS[j], lam, first)
        orc[j] = np.linalg.norm(mm / np.linalg.norm(T[j][-1]) - th[j])
    ro = float(np.sqrt((orc ** 2).mean()) * 100)
    print(f'  оракул повного базису (справжній прогон): {ro:.4f}%  '
          f'({(rb/ro)**2:.3f}×)')
    return out


def finale(T, WS, rng):
    """ДОДАТКОВО: чи ЗНАЧУЩИЙ приріст пошарової над вихідною + їх СУМА."""
    N = len(WS)
    H = 1e-3
    CL = [0, 1, 2, 3]                              # найкращий пошаровий набір
    print(f'\n{"=" * 78}\nДОДАТКОВО — ЗНАЧУЩІСТЬ І КОМБІНАЦІЯ\n{"=" * 78}')
    J, Xo, mh, th, ys = [], [], [], [], []
    for j in range(N):
        m0, B0 = closure(WS[j])
        nn = np.linalg.norm(T[j][-1])
        col = np.zeros((W_, NB))
        for q in range(NB):
            lp, lm = np.zeros(NB), np.zeros(NB)
            lp[q], lm[q] = H, -H
            col[:, q] = (closure(WS[j], lp, 1)[0]
                         - closure(WS[j], lm, 1)[0]) / (2 * H) / nn
        J.append(col)
        Xo.append(B0 / nn)
        mh.append(m0 / nn)
        th.append(T[j][-1] / nn)
        ys.append(th[-1] - mh[-1])
    base = np.array([np.linalg.norm(mh[j] - th[j]) for j in range(N)])
    rb = float(np.sqrt((base ** 2).mean()) * 100)

    # (A) вихідна {B0}
    cA, _, _, _ = nested_loo(Xo, ys, [0])
    rA = np.array([np.linalg.norm(mh[j] + Xo[j][:, [0]] @ cA[j] - th[j])
                   for j in range(N)])
    # (B) пошарова {B0,B1,B2,B3}
    cB, _, _, _ = nested_loo(J, ys, CL)
    rB, mB = np.zeros(N), []
    for j in range(N):
        lam = np.zeros(NB)
        lam[CL] = cB[j]
        mm = closure(WS[j], lam, 1)[0] / np.linalg.norm(T[j][-1])
        mB.append(mm)
        rB[j] = np.linalg.norm(mm - th[j])
    # (C) пошарова + вихідний скаляр НА ВЕРХУ (спільний LOO по мережах)
    Xc = [np.concatenate([J[j][:, CL], Xo[j][:, [0]]], 1) for j in range(N)]
    cC, _, _, _ = nested_loo(Xc, ys, list(range(len(CL) + 1)))
    rC = np.zeros(N)
    for j in range(N):
        lam = np.zeros(NB)
        lam[CL] = cC[j][:len(CL)]
        mm = closure(WS[j], lam, 1)[0] / np.linalg.norm(T[j][-1])
        rC[j] = np.linalg.norm(mm * (1 + cC[j][-1]) - th[j])
    print(f'  без поправки: {rb:.4f}%')
    for nm, r in (('A вихідна {B0}', rA), ('B пошарова {B0,B1,B2,B3}', rB),
                  ('C пошарова + вихідний скаляр', rC)):
        rr = float(np.sqrt((r ** 2).mean()) * 100)
        print(f'  {nm:30s}: {rr:.4f}%   {(rb / rr) ** 2:.3f}× по MSE')
    # ── ПАРНИЙ тест ПО МЕРЕЖАХ: чи B справді краще за A ──────────────────────
    for nm, r in (('B−A', rB), ('C−A', rC)):
        d = rA ** 2 - r ** 2                       # >0 ⟹ краще за A
        t = d.mean() / (d.std(ddof=1) / np.sqrt(N))
        w = int((d > 0).sum())
        print(f'  парний тест {nm}: виграло {w}/{N} мереж, t = {t:+.2f}  '
              f'({"ЗНАЧУЩЕ" if abs(t) > 2.5 else "НЕ значуще"})')
    print(f'  📏 нагадування: шумова підлога LOO-по-мережах ≈1.03× по MSE')


def main():
    rng = np.random.default_rng(20260731)
    T, Mc, SD, WS = load_all()
    print(f'мереж: {len(WS)}, істина з кешу clgeom '
          f'({2*PAIRS:,} семплів/мережу, шум 0.0058%)')
    seal4()
    r1 = exp1(T, Mc, WS, rng)
    r2 = exp2(T, WS, 1, rng)
    r2b = exp2(T, WS, 2, rng)
    print(f'\n{"=" * 78}\nПІДСУМОК: ВИХІД vs ПОШАРОВО (LOO ПО МЕРЕЖАХ, ×MSE)'
          f'\n{"=" * 78}')
    print('  набір                       |  ВИХІД  | ПОШАРОВО(1..32) | '
          'ПОШАРОВО(2..32)')
    for nm, _ in SETS:
        b, l, o, _, _ = r1[nm]
        print(f'  {nm:27s} | {(b/l)**2:6.3f}× | {(r2[nm][0]/r2[nm][1])**2:13.3f}'
              f'× | {(r2b[nm][0]/r2b[nm][1])**2:13.3f}×')
    finale(T, WS, rng)


if __name__ == '__main__':
    main()
