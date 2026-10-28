"""СИМЕТРИЧНИЙ КРОС-ФІТИНГ + ЗГЛАДЖУВАННЯ `b̂` ПО ГЛИБИНІ — пункти 2,3,4 юзера.

ЗВІДКИ (три поправки юзера, кожна знімає окрему мою помилку).
2. Моя схема «`r̂` з половини, якір із усіх» — ЧАСТКОВЕ ОТРУЄННЯ: `r̂` з половини
   A корелює з A-компонентою якоря. Правильна схема СИМЕТРИЧНА:
   `r̂(A) → якір(B)`, `r̂(B) → якір(A)`, потім усереднити.
3. У моїй таблиці підписаних кореляцій: **повний рекурс +0.246, лише перенос
   +0.242** ⟹ зміщення практично ПОВНІСТЮ переноситься, локальне джерело майже
   нічого не додає. Отже `b` на 30 шарах — не 30 незалежних величин, а ОДНА
   траєкторія вздовж ВІДОМОГО лінійного оператора, спостережена 30 разів із
   НЕЗАЛЕЖНИМ шумом ⟹ канонічне згладжування (Калман по глибині).
4. Три джерела `r̂` треба ПОЄДНУВАТИ, а не вибирати. Апріорі — формула Еджворта
   (`b_i = γ₁·t·φ(t)·s/6`, ✅пломба 1.24e-14, corr 0.25, НУЛЬ параметрів);
   спостереження — крос-фіт; апостеріорна — зважена за оберненими дисперсіями.
   🔑 У фільтрі Калмана це ОДНЕ І ТЕ САМЕ: джерело Еджворта — детермінований
   член процесу, `Q = q·джерело²` — його невизначеність.

СХЕМА (ціна РІВНО НУЛЬ: ті самі `npair` пар, лише поділені навпіл)
    `b^pred_ℓ = Φ(t_ℓ)⊙(W_ℓᵀ b^post_{ℓ−1}) + джерело_ℓ`
    `P^pred_ℓ = Φ²⊙((W²)ᵀ P^post_{ℓ−1}) + q·джерело²`
    спостереження `y_ℓ = m_замик − x̄_ℓ(половина)`, `R_ℓ = d_ℓ/(2·npair/2)` ТОЧНО
    `b^post = (b^pred/P^pred + y/R)/(1/P^pred + 1/R)`,  `P^post = (1/P^pred+1/R)⁻¹`

🚨 ЦІНУ ПОДІЛУ НЕ ХОВАЄМО: база для ВСІХ відношень — MSE ПОВНОЇ вибірки, тож
рядок `рівно (половини)` показує штраф за поділ НАПРЯМУ.

КОНТРОЛІ
  * ✅`рівно (половини)` — скільки коштує сам поділ;
  * ⚠️оракул `β` і `β·маска` у ТІЙ САМІЙ структурі — стеля тут же;
  * ⛔НЕГ: калманівська `β̂`, перемішана всередині шару;
  * 📊 три зрізи; `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ варіанті.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, W_, _phi, _Phi)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402
from vrank import state_and_v                                     # noqa: E402
from rpredict import moments                                      # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
KS = 30
QGRID = [0.25, 1.0, 4.0, 1e12]        # 🚨 останнє — ✅ПЛОМБА: q→∞ ≡ сирий крос-фіт
BANDS = {'1..8': list(range(0, 8)), '24..30': list(range(23, 30)),
         '1..30': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def closure_chain(wc, dev):
    """Ph, s, d, джерело Еджворта — по шарах 1..KS, плюс самі ваги."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    k3a = torch.zeros(n0, dtype=torch.float64, device=dev)
    out = []
    for i in range(KS + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        m1, m2, k3, t, ph, Ph = moments(mp, s)
        d = m2 - m1 * m1
        k3h = (w ** 3).T @ k3a
        src = (k3h / s ** 3) * t * ph * s / 6.0        # 🔑 Еджворт, ✅пломба
        if i >= 1:
            out.append({'Ph': Ph, 'd': d, 'src': src, 'w': w})
        m = m1
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        C = C + torch.diag(d - C.diagonal())
        k3a = k3
    return out


def kalman(CH, obs, nph, q):
    """Прохід уперед по глибині. `obs[i]` оцінює `b_i`, шум `nph[i]²`."""
    bp = None
    bpost, Ppost = [], []
    for i in range(KS):
        c = CH[i]
        if bp is None:
            bpred = c['src']
            Ppred = q * c['src'] ** 2 + 1e-30
        else:
            bpred = c['Ph'] * (bp @ c['w']) + c['src']
            Ppred = c['Ph'] ** 2 * ((c['w'] ** 2).T @ Pp) + q * c['src'] ** 2
            Ppred = Ppred.clamp_min(1e-30)
        R = (nph[i] ** 2).clamp_min(1e-30)
        iP, iR = 1.0 / Ppred, 1.0 / R
        b = (bpred * iP + obs[i] * iR) / (iP + iR)
        P = 1.0 / (iP + iR)
        bpost.append(b); Ppost.append(P)
        bp, Pp = b, P
    return bpost, Ppost


def main(nets=24, dev=DEV):
    torch.manual_seed(577)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = (['рівно (половини) — ЦІНА ПОДІЛУ', '🟡крос-фіт сирий',
              '🟡крос-фіт дебіас']
             + [f'🟢КАЛМАН q={q}' for q in QGRID]
             + ['🟢КАЛМАН q=1 + послаблення', '⚠️оракул β',
                '⚠️оракул β·маска q=.3', '⛔НЕГ (Калман перемішано)'])
    base_mse, G, B, C, DIAG = [], None, None, [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            vs, nz = state_and_v(wc, dev, npair)
            CH = closure_chain(wc, dev)
            nh = npair // 2
            # шум ПОЛОВИНИ (для спостереження) і ПРОДАКШН (для `r`)
            nph = [(CH[i]['d'].clamp_min(1e-30).sqrt()
                    / np.sqrt(2.0 * nh)) for i in range(KS)]
            npr = [nz[i + 1] for i in range(KS)]     # 🚨 зсув: m_cl[i] ↔ шар i+1
            ora, oram = [], []
            for i in range(KS):
                ro = ((m_cl[i] - TL['m'][i]).abs() / npr[i]).clamp_min(1e-12)
                bo = 1.0 / (1.0 + ro * ro)
                co = (vs[i + 1] * npr[i]) ** 2 * bo
                ora.append(bo)
                oram.append(bo * (co >= torch.quantile(co, 0.7)).double())

            nv = len(names)
            if G is None:
                G = [[] for _ in range(nv)]
                B = [[] for _ in range(nv)]
            gg = torch.zeros(nv, KS, KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, KS, dtype=torch.float64, device=dev)
            cfull, chalf = 0.0, 0.0

            for d in range(NDRAW):
                sd = 33_000_000 + 733 * n + d
                # ✅ БАЗА — ПОВНА вибірка (проти неї рахуються ВСІ відношення)
                prf, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=LAM31)
                cfull += float(((prf - T) ** 2).sum())
                g = torch.Generator(device=dev).manual_seed(int(sd))
                z = torch.randn(npair, W_, device=dev, generator=g)
                zh = (z[:nh], z[nh:2 * nh])
                pr, MM = [], []
                for h in range(2):
                    p, M = run_side1(wc, nh, s_cl, m_cl, dev, 0, lam_s=LAM31,
                                     collect_means=True, z_ext=zh[h])
                    pr.append(p)
                    MM.append([m_cl[i] - M[i][0] for i in range(KS)])
                e = 0.5 * (pr[0] + pr[1]) - T
                chalf += float((e * e).sum())
                if d < 4:
                    # 🚨 ЧИ НЕЗАЛЕЖНИЙ ШУМ СПОСТЕРЕЖЕНЬ ПО ГЛИБИНІ? Калман це
                    #    ПРИПУСКАЄ. `ε_ℓ = m_іст − x̄_ℓ` несеться ТИМ САМИМ
                    #    оператором, що й `b` ⟹ якщо corr висока, згладжування
                    #    не має чого відокремлювати.
                    eps = [TL['m'][i] - (m_cl[i] - MM[0][i]) for i in range(KS)]
                    cs = []
                    for i in range(KS - 1):
                        tr = CH[i + 1]['Ph'] * (eps[i] @ CH[i + 1]['w'])
                        a_, b_ = tr - tr.mean(), eps[i + 1] - eps[i + 1].mean()
                        cs.append(float((a_ * b_).sum()
                                        / (a_.norm() * b_.norm()).clamp_min(1e-30)))
                    DIAG.append(float(np.mean(cs)))

                # ваги, ОЦІНЕНІ НА ПРОТИЛЕЖНІЙ половині (симетричний крос-фіт)
                AWh = [[], []]
                for h in range(2):
                    o = MM[1 - h]                     # 🔑 спостереження з ІНШОЇ
                    kb, kP = {}, {}
                    for q in QGRID:
                        kb[q], kP[q] = kalman(CH, o, nph, q)
                    for i in range(KS):
                        raw = (o[i] / npr[i]) ** 2
                        deb = (raw - float(npair) / nh).clamp_min(0.0)
                        rows = [torch.ones_like(raw),
                                1.0 / (1.0 + raw), 1.0 / (1.0 + deb)]
                        for q in QGRID:
                            rows.append(1.0 / (1.0 + (kb[q][i] / npr[i]) ** 2))
                        rows.append(1.0 / (1.0 + (kb[1.0][i] / npr[i]) ** 2
                                           + kP[1.0][i] / npr[i] ** 2))
                        rows.append(ora[i])
                        rows.append(oram[i])
                        rows.append(torch.roll(
                            1.0 / (1.0 + (kb[1.0][i] / npr[i]) ** 2), 7))
                        AWh[h].append(torch.stack(rows))
                Am = None
                for h in range(2):
                    _, A = run_side1(wc, nh, s_cl, m_cl, dev, 0, lam_s=LAM31,
                                     collect_anchors=True, anchor_w=AWh[h],
                                     z_ext=zh[h])
                    Av = A.view(nv, KS, -1)
                    Am = Av if Am is None else Am + Av
                Am = 0.5 * Am
                gg += Am @ Am.transpose(1, 2)
                bb += Am @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(chalf); base_mse.append(cfull)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair}, nh={nh})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. База — MSE ПОВНОЇ вибірки, тож '
          f'штраф за поділ ВХОДИТЬ у кожен рядок.')
    print(f'🚨 ДІАГНОЗ ДЛЯ КАЛМАНА: corr(перенесений шум шару ℓ, шум шару ℓ+1) '
          f'= **{np.mean(DIAG):+.3f}**   (фільтр ПРИПУСКАЄ 0)\n')
    for bn, idx in BANDS.items():
        print(f'--- шари {bn} ---')
        print(f'{"варіант":>30s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = '  ⛔' if nm.startswith('⛔') else ''
            print(f'{nm:>30s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    print('🔑 читати так: перший рядок — ЦІНА поділу навпіл (мусить бути <1, і '
          'це відрахування з усіх інших). Калман мусить бити «крос-фіт сирий», '
          'інакше згладжування по глибині не додає. Порівнювати з ⚠️оракулом '
          'У ЦІЙ ЖЕ таблиці — він теж платить за поділ.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
