"""ПІДСТАНОВКА: що дає поправка `κ₃` в РЕКУРСІЇ на 30 шарах? (С74)

ЗВІДКИ. `k3diag.py`: низькоранговий тензор `κ₃` (ранг 16, `k³` чисел) пояснює
**0.355** похибки `C` при оракульних 0.430. Але правило: **R² → множник лише
ПІДСТАНОВКОЮ** (прайс-листи завищують: `tfit` обіцяв 1.70×, дав 1.056×).

📐 ЩО ПЕРЕВОДИМО. Драбина `price_closure` (K=30): `k=1.0`→1.002×, `k=0.5`→1.152×,
`k=0.25`→1.612×, `k=0`→10.463×, де `k` — множник похибки `m`. Наївно
`R²=0.355 ⟹ k=√(1−0.355)=0.803 ⟹ ≈1.06×`. 🚨АЛЕ драбина масштабує похибку **`m`**,
а `κ₃` виправляє **`C`** — переклад між ними НЕВІДОМИЙ. Цей стенд його міряє.

ЯК. Рекурсія замикання, на КОЖНОМУ шарі:
    `C ← C_гаус + (sₐ·s_b)·Σⱼ βⱼ·ознакаⱼ(K, t, ρ)`,
де `K` — низькоранговий `κ₃` (оракульний), `β` — коефіцієнти форми поправки.
Далі звичайний крок. Міряємо похибку `m₃₀` проти НЕЗАЛЕЖНОЇ істини ⟹ це і є
реальне `k`, яке підставляється в драбину.

ДВА РЕЖИМИ `β` (різниця показує, скільки коштує НЕ знати `β` заздалегідь):
  ⚠️**ОРАКУЛ**: `β` з ТІЄЇ САМОЇ мережі — СТЕЛЯ (і завідомо завищує);
  ✅**LOO**: `β` з ІНШИХ мереж — те, що реально доступне (як `λ` у сабміті).

ПЛОМБИ
  ✅ `β=0` мусить відтворити базову похибку `m₃₀` ДО ОСТАННЬОЇ ЦИФРИ;
  ⛔ ознаки з ШУМУ тієї ж норми ⟹ похибка НЕ мусить упасти;
  ⚠️ верхня межа — рекурсія з ІСТИННИМИ `C` на кожному кроці (скільки взагалі
     можна взяти, виправляючи лише `C`).
🚨 `U` і `G` тут ОРАКУЛЬНІ (з вибірки) ⟹ це стеля СХЕМИ; реальне джерело `G`
   мало б іти із замикання або з НЕЗАЛЕЖНОЇ половини вибірки.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_, _phi, _Phi                # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NP_, CH = 65_536, 8192
KR = 16                      # ранг підпростору для тензора κ₃
KS = 30                      # глибина рекурсії


def collect(wc, dev, seed):
    """Оракульні `U`,`G`,`κ₃`-статистики + ІСТИННІ `m`,`C` активацій по шарах."""
    n = [wc[i].shape[1] for i in range(KS + 1)]
    S = [torch.zeros(q, dtype=torch.float64, device=dev) for q in n]
    Q = [torch.zeros(q, q, dtype=torch.float64, device=dev) for q in n]
    # моменти ДОАКТИВАЦІЙ `p` — джерело для Еджворта
    P1 = [torch.zeros(q, dtype=torch.float64, device=dev) for q in n]
    P2 = [torch.zeros(q, dtype=torch.float64, device=dev) for q in n]
    P3 = [torch.zeros(q, dtype=torch.float64, device=dev) for q in n]
    P4 = [torch.zeros(q, dtype=torch.float64, device=dev) for q in n]
    rng = np.random.default_rng(seed)
    done = 0
    while done < NP_:
        c = min(CH, NP_ - done)
        zz = torch.tensor(rng.standard_normal((c, W_)).astype(np.float32),
                          device=dev)
        a = torch.cat([zz, -zz], 0)
        for i in range(KS + 1):
            pp = a @ wc[i]
            pdd = pp.double()
            P1[i] += pdd.sum(0)
            P2[i] += (pdd * pdd).sum(0)
            P3[i] += (pdd ** 3).sum(0)
            P4[i] += (pdd ** 4).sum(0)
            a = torch.relu(pp)
            ad = a.double()
            S[i] += ad.sum(0)
            Q[i] += ad.T @ ad
        done += c
    N = float(2 * NP_)
    M = [S[i] / N for i in range(KS + 1)]
    C = [Q[i] / N - torch.outer(M[i], M[i]) for i in range(KS + 1)]
    GAM = []
    for i in range(KS + 1):
        m1 = P1[i] / N
        v = (P2[i] / N - m1 * m1).clamp_min(1e-30)
        sd = v.sqrt()
        mu3 = P3[i] / N - 3.0 * m1 * (P2[i] / N) + 2.0 * m1 ** 3
        mu4 = (P4[i] / N - 4.0 * m1 * (P3[i] / N)
               + 6.0 * m1 * m1 * (P2[i] / N) - 3.0 * m1 ** 4)
        GAM.append((mu3 / sd ** 3, mu4 / sd ** 4 - 3.0))
    # базис `U` — головні напрямки активацій кожного шару
    U = []
    for i in range(KS + 1):
        kk = min(KR, C[i].shape[0])
        ev, V = torch.linalg.eigh(C[i])
        U.append(V[:, -kk:].contiguous())
    # тензор `G` у підпросторі — другий прохід
    G = [torch.zeros(U[i].shape[1], U[i].shape[1], U[i].shape[1],
                     dtype=torch.float64, device=dev) for i in range(KS + 1)]
    rng2 = np.random.default_rng(seed)
    done = 0
    while done < NP_:
        c = min(CH, NP_ - done)
        zz = torch.tensor(rng2.standard_normal((c, W_)).astype(np.float32),
                          device=dev)
        a = torch.cat([zz, -zz], 0)
        for i in range(KS + 1):
            a = torch.relu(a @ wc[i])
            u = (a.double() - M[i]) @ U[i]
            G[i] += torch.einsum('ma,mb,mc->abc', u, u, u)
        done += c
    for i in range(KS + 1):
        G[i] /= N
    return M, C, U, G, GAM


def feats(K, t, ph, Ph, s, cp):
    """Ознаки поправки до `C`, нормовані на `sₐ·s_b`. Порядок як у `k3diag`."""
    sc = torch.outer(s, s)
    nz = (s * s).unsqueeze(1) * s.unsqueeze(0)
    r = cp / sc
    Q = K / nz
    Qt = Q.T
    Ti = t.unsqueeze(1).expand_as(Q)
    Tj = t.unsqueeze(0).expand_as(Q)
    Pa = ph.unsqueeze(1).expand_as(Q)
    Pb = ph.unsqueeze(0).expand_as(Q)
    return torch.stack([r, r * r, torch.ones_like(r),
                        Q, Qt, Q * Ti, Qt * Tj, Q * Pa * Pb,
                        (Q + Qt) * Ti * Tj], 0), sc


def step(w, m, C, K=None, beta=None, noise=None, gam=None):
    """Один крок замикання; за наявності `beta` — з поправкою до `C`."""
    cp = w.T @ (C @ w)
    s = cp.diagonal().clamp_min(1e-20).sqrt()
    mp = m @ w
    t = mp / s
    ph, Ph = _phi(t), _Phi(t)
    m2 = mp * Ph + s * ph
    if gam is not None:
        # 📐 Еджворт (M46 §B): `−(γ₁/6)·s·t·φ + (γ₂/24)·s·(t²−1)·φ`
        g1, g2 = gam
        m2 = (m2 - (g1 / 6.0) * s * t * ph
              + (g2 / 24.0) * s * (t * t - 1.0) * ph)
    Cg = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
    d = (mp * mp + s * s) * Ph + mp * s * ph - m2 * m2
    Cg = Cg + torch.diag(d - Cg.diagonal())
    if beta is not None:
        F, sc = feats(K if noise is None else noise, t, ph, Ph, s, cp)
        corr = torch.einsum('j,jab->ab', beta, F) * sc
        corr = 0.5 * (corr + corr.T)              # `C` симетрична
        corr = corr - torch.diag(corr.diagonal())  # діагональ ТОЧНА
        # 🚨 ЗАПОБІЖНИК: поправка не має права вивести `C` за межі додатної
        #    визначеності — інакше `s²<0` і рекурсія йде в nan
        nrm = corr.norm() / Cg.norm().clamp_min(1e-30)
        if nrm > 0.30:
            corr = corr * (0.30 / nrm)
        Cg = Cg + corr
    return m2, Cg, (t, ph, Ph, s, cp)


def mfeats(t, ph, Ph, s, g1, g2, mode='all'):
    """Ознаки для локальної поправки до `m`. Перші дві — члени Еджворта."""
    one = torch.ones_like(t)
    if mode == 'free':
        # 🔑 ЛИШЕ те, що є в рекурсії БЕЗ жодного семпла
        return torch.stack([s * t * ph, s * (t * t - 1.0) * ph, s * ph,
                            s * Ph, s * t * t * ph, s * one], 0)
    if mode == 'rand':
        g = torch.Generator(device=t.device).manual_seed(int(len(t)))
        return torch.stack([torch.randn(t.shape, dtype=t.dtype,
                                        device=t.device, generator=g)
                            for _ in range(6)], 0)
    return torch.stack([
        s * t * ph,                 # −γ₁/6 член (без коефіцієнта)
        s * (t * t - 1.0) * ph,     # +γ₂/24 член
        g1 * s * t * ph,            # те саме, зважене РЕАЛЬНИМ скосом
        g2 * s * (t * t - 1.0) * ph,
        s * ph, s * Ph, s * t * t * ph, s * one, g1 * s * ph, g2 * s * ph,
    ], 0)


def lift(K3, U, w):
    """`κ₃(pₐ,pₐ,p_b) = Σ_αβγ G_αβγ V_αa V_βa V_γb`, `V = UᵀW`."""
    V = U.T @ w
    H = torch.einsum('abg,am,bm->gm', K3, V, V)
    return H.T @ V


def main(nets=6, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NETS = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            M, C, U, G, GAM = collect(wc, dev, 990_000 + 7919 * n)
            # нормальні рівняння для `β` по шарах (на ІСТИННІЙ похибці `C`)
            n0 = wc[0].shape[0]
            m = torch.zeros(n0, dtype=torch.float64, device=dev)
            Cc = torch.eye(n0, dtype=torch.float64, device=dev)
            AA, AB = [], []
            MA = {k: [] for k in ('all', 'free', 'rand')}
            MB = {k: [] for k in ('all', 'free', 'rand')}
            for i in range(KS + 1):
                w = wc[i].double()
                m2, Cg, (t, ph, Ph, s, cp) = step(w, m, Cc)
                Kl = lift(G[i - 1], U[i - 1], w) if i >= 1 else None
                if Kl is not None:
                    F, sc = feats(Kl, t, ph, Ph, s, cp)
                    off = ~torch.eye(len(s), dtype=torch.bool, device=dev)
                    X = F[:, off].T
                    yv = ((C[i] - Cg) / sc)[off]
                    AA.append(X.T @ X)
                    AB.append(X.T @ yv)
                else:
                    AA.append(None)
                    AB.append(None)
                # ── локальна поправка: ціль `δᵢ` при ІСТИННИХ входах ──
                cin_ = (torch.eye(n0, dtype=torch.float64, device=dev)
                        if i == 0 else C[i - 1])
                min0 = (torch.zeros(n0, dtype=torch.float64, device=dev)
                        if i == 0 else M[i - 1])
                mtrue, _, (tt, pph, PPh, ss, _) = step(w, min0, cin_)
                dtrue = M[i] - mtrue
                g1t, g2t = GAM[i]
                for md in ('all', 'free', 'rand'):
                    Xm = mfeats(tt, pph, PPh, ss, g1t, g2t, md).T
                    MA[md].append(Xm.T @ Xm)
                    MB[md].append(Xm.T @ dtrue)
                m, Cc = m2, Cg
            NETS.append(dict(wc=wc, M=M, C=C, U=U, G=G, GAM=GAM, AA=AA, AB=AB,
                             MA=MA, MB=MB, W=W))
        print(f'  … збір {n+1}/{nets}', flush=True)

    def run(net, beta_list, noisy=False, scale=1.0, edge=False,
            delta=None, truec=False, mbeta=None):
        wc = net['wc']
        n0 = wc[0].shape[0]
        m = torch.zeros(n0, dtype=torch.float64, device=DEV)
        Cc = torch.eye(n0, dtype=torch.float64, device=DEV)
        gg = torch.Generator(device=DEV).manual_seed(5)
        for i in range(KS + 1):
            w = wc[i].double()
            Kl = lift(net['G'][i - 1], net['U'][i - 1], w) if i >= 1 else None
            nz = None
            if noisy and Kl is not None:
                nz = torch.randn(Kl.shape, dtype=Kl.dtype, device=DEV,
                                 generator=gg) * Kl.std()
            b = (beta_list[i] * scale
                 if (beta_list is not None and i >= 1) else None)
            gm = net['GAM'][i] if edge else None
            cin = net['C'][i - 1] if (truec and i >= 1) else Cc
            m, Cc, aux = step(w, m, cin, K=Kl, beta=b, noise=nz, gam=gm)
            if mbeta is not None:
                tt, pph, PPh, ss, _ = aux
                g1t, g2t = net['GAM'][i]
                m = m + mfeats(tt, pph, PPh, ss, g1t, g2t,
                               mbeta[0]).T @ mbeta[1][i]
            if delta is not None:
                m = m + delta[i]
        return m

    def err(got, tru):
        return 100.0 * float(((got - tru) ** 2).mean().sqrt()
                             / (tru ** 2).mean().sqrt())

    RES = {k: [] for k in ('база', '⚠️ОРАКУЛ β', '✅LOO β', '⛔ШУМ β',
                           '⚠️істинна C', '⚠️істинне m', '⚠️істинні m і C',
                           '✅ЕДЖВОРТ m (оракул γ)', '⚠️ідеальна локальна m',
                           '⚠️лок. m + істинна C',
                           '⚠️ЛОКАЛЬНА (з оракул. γ)',
                           '✅ЛОКАЛЬНА (лише замикання)',
                           '⛔ЛОКАЛЬНА (випадкові ознаки)')}
    R2L = {}
    for j, net in enumerate(NETS):
        T = net['M'][KS]
        RES['база'].append(err(run(net, None), T))
        RES['✅ЕДЖВОРТ m (оракул γ)'].append(err(run(net, None, edge=True), T))
        # ⚠️ІДЕАЛЬНА ЛОКАЛЬНА поправка: на кожному кроці додаємо ІСТИННУ похибку
        #    кроку, пораховану при ІСТИННИХ входах. Накопичення ЛИШАЄТЬСЯ ⟹
        #    різниця з «істинним m» = внесок САМОГО накопичення.
        wc0 = net['wc']
        I00 = torch.eye(wc0[0].shape[0], dtype=torch.float64, device=DEV)
        dl = []
        for i in range(KS + 1):
            w_ = wc0[i].double()
            cin = I00 if i == 0 else net['C'][i - 1]
            min0 = (torch.zeros(wc0[0].shape[0], dtype=torch.float64,
                                device=DEV) if i == 0 else net['M'][i - 1])
            mm_, _, _ = step(w_, min0, cin)
            dl.append(net['M'][i] - mm_)
        RES['⚠️ідеальна локальна m'].append(err(run(net, None, delta=dl), T))
        # ✅РЕАЛЬНА локальна поправка: `β` для `δ` фітяться на ІНШИХ мережах
        oth = [q for q in range(len(NETS)) if q != j]
        NAME = {'all': '⚠️ЛОКАЛЬНА (з оракул. γ)',
                'free': '✅ЛОКАЛЬНА (лише замикання)',
                'rand': '⛔ЛОКАЛЬНА (випадкові ознаки)'}
        for md in ('all', 'free', 'rand'):
            mb, r2s = [], []
            for i in range(KS + 1):
                A = sum(NETS[q]['MA'][md][i] for q in oth)
                Bv = sum(NETS[q]['MB'][md][i] for q in oth)
                rg = 1e-8 * float(A.diagonal().mean())
                co = torch.linalg.solve(
                    A + rg * torch.eye(A.shape[0], dtype=A.dtype,
                                       device=A.device), Bv)
                mb.append(co)
                An, Bn = net['MA'][md][i], net['MB'][md][i]
                sr = (float(dl[i] @ dl[i]) - 2 * float(co @ Bn)
                      + float(co @ (An @ co)))
                r2s.append(1.0 - sr / max(float(dl[i] @ dl[i]), 1e-30))
            R2L.setdefault(md, []).append(float(np.median(r2s[1:])))
            RES[NAME[md]].append(err(run(net, None, mbeta=(md, mb)), T))
        # 🔑 РОЗДІЛЬНИЙ ВИМІР: залишок 0.58% — це нелінійність переносу
        #    чи вплив накопиченої `C` через `s`?
        RES['⚠️лок. m + істинна C'].append(
            err(run(net, None, delta=dl, truec=True), T))
        # β з ТІЄЇ САМОЇ мережі (стеля) і з ІНШИХ (LOO)
        for nm, idx in (('⚠️ОРАКУЛ β', [j]),
                        ('✅LOO β', [q for q in range(len(NETS)) if q != j])):
            bl = [None] * (KS + 1)
            for i in range(1, KS + 1):
                A = sum(NETS[q]['AA'][i] for q in idx)
                B = sum(NETS[q]['AB'][i] for q in idx)
                # 🚨 ridge ВІДНОСНО масштабу `A` (абсолютний 1e-8 не тримає:
                #    нормальні рівняння тут погано обумовлені і β вибухають)
                rg = 1e-6 * float(A.diagonal().mean())
                bl[i] = torch.linalg.solve(
                    A + rg * torch.eye(A.shape[0], dtype=A.dtype,
                                       device=A.device), B)
            RES[nm].append(err(run(net, bl), T))
            if nm == '✅LOO β':
                RES['⛔ШУМ β'].append(err(run(net, bl, noisy=True), T))
                # 🚨 ЧИ НЕ ЗАВЕЛИКИЙ МАСШТАБ? якщо оптимум при 0 — поправка
                #    шкідлива ПО СУТІ, а не через силу
                for sc_ in (0.1, 0.25, 0.5):
                    RES.setdefault(f'✅LOO β×{sc_}', []).append(
                        err(run(net, bl, scale=sc_), T))
        # ⚠️ РОЗКЛАД: що дає підстановка ІСТИННИХ `m` / `C` / обох.
        #    🔑 саме тут видно, чи їхні похибки ГАСЯТЬ одна одну.
        wc = net['wc']
        I0 = torch.eye(wc[0].shape[0], dtype=torch.float64, device=DEV)
        for nm, tm, tc in (('⚠️істинна C', False, True),
                           ('⚠️істинне m', True, False),
                           ('⚠️істинні m і C', True, True)):
            m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=DEV)
            Cc = I0.clone()
            for i in range(KS + 1):
                w = wc[i].double()
                min_ = net['M'][i - 1] if (tm and i >= 1) else m
                cin = net['C'][i - 1] if (tc and i >= 1) else Cc
                m, Cc, _ = step(w, min_, cin)
            RES[nm].append(err(m, T))
        print(f'  … прогін {j+1}/{len(NETS)}', flush=True)

    md = np.median
    b0 = md(RES['база'])
    print(f'\n{len(NETS)} мереж, істина {2*NP_} семплів, ранг {KR}, '
          f'рекурсія {KS} шарів\n')
    print(f'{"варіант":>16s} {"похибка m₃₀":>12s} {"k = від бази":>13s} '
          f'{"драбина":>10s}')
    print('-' * 56)
    LAD = [(1.0, 1.002), (0.5, 1.152), (0.25, 1.612), (0.1, 3.605), (0.0, 10.463)]

    def ladder(k):
        if k >= 1.0:
            return LAD[0][1]
        for (k1, g1), (k2, g2) in zip(LAD, LAD[1:]):
            if k2 <= k <= k1:
                f = (k1 - k) / max(k1 - k2, 1e-9)
                return g1 + f * (g2 - g1)
        return LAD[-1][1]

    for nm in ('база', '✅LOO β', '⚠️істинна C', '✅ЕДЖВОРТ m (оракул γ)',
               '⛔ЛОКАЛЬНА (випадкові ознаки)', '✅ЛОКАЛЬНА (лише замикання)',
               '⚠️ЛОКАЛЬНА (з оракул. γ)', '⚠️ідеальна локальна m',
               '⚠️істинне m', '⚠️істинні m і C'):
        if nm not in RES or not RES[nm]:
            continue
        v = md(RES[nm])
        k = v / b0
        print(f'{nm:>16s} {v:11.4f}% {k:12.4f} {ladder(k):9.3f}×')
    print('-' * 56)
    print('\n📐 R² локальної поправки (β з ІНШИХ мереж, медіана по шарах):')
    for md in ('all', 'free', 'rand'):
        print(f'      {md:>5s}: {np.median(R2L[md]):.4f}')
    print('   порівняти з записаною «стелею локальних ознак 0.31».\n')
    print('🎯 ЧИТАТИ: колонка `k` — у скільки разів похибка МЕНША за базу;\n'
          '   «драбина» — виграш за `price_closure` (K=30) для цього `k`.\n'
          '✅ПЛОМБИ: база = 1.0000 тотожно · ⛔ШУМ мусить лишитись ≈1.0 ·\n'
          '   ⚠️ОРАКУЛ ≥ ✅LOO (інакше фіт нестабільний).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
