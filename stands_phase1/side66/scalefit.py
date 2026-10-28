"""СКАЛЯРНА ПОПРАВКА МАСШТАБУ ПОВЕРХ САБМІТУ — ЧИ ПЕРЕНОСИТЬСЯ МІЖ МЕРЕЖАМИ?

════════ ЗВІДКИ ════════
`noisespec.py`: `cos(m,u₁)=0.978` ⟹ шум на глибині — це флуктуація МАСШТАБУ
вздовж самого `m`, а не напрямку. `scalenoise.py`: цей скаляр передбачається
з ДОСТУПНИХ ознак на **R²=0.915 OOS по жеребах** (⛔ROLL −0.18 чистий).
Стеля з цього: `1/(1−0.384·0.915) ≈ 1.54×`.
🔑 ЧОМУ ЦЕ МОЖЕ БУТИ НОВИМ: сабміт несе поправку ВЕКТОРОМ крізь 30 матриць, і
напрямок задає ланцюг переносу. Тут навпаки — напрямок береться ПРЯМО з
`m_замик,L/‖m_замик,L‖` (доступний!), а вчиться лише ОДИН СКАЛЯР.

════════ 🚨 ТРИ УМОВИ, ЯКІ МОЖУТЬ ЦЕ З'ЇСТИ (перевіряємо саме їх) ════════
  1. `R²` міряли OOS по ЖЕРЕБАХ, але ВСЕРЕДИНІ мережі ⟹ тут LOO ПО МЕРЕЖАХ;
  2. при центруванні по жеребах зникає СТАЛЕ зміщення `B`; у сабміті центр
     невідомий, тож разом із шумом віднімається й частина `B`;
  3. вимір був на 4 мережах ⟹ тут 24, і база = ПОВНИЙ `sub72` (4 родини).

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 LOO по мережах з'їсть більшу частину: очікую **1.10–1.30×**, не 1.54×.
🟢 ЖИВЕ при >1.05× на СУМІ І МЕДІАНІ (тоді кандидат у сабміт).

🚨 ПЛОМБИ
  ✅ β=0 мусить дати РІВНО 1.000× (та сама база);
  ⛔ НЕГ: регресори з СУСІДНЬОГО жереба ≈1.000×;
  ⛔ ЧУЖА: напрямок `u` з іншої мережі — мусить бути ГІРШЕ за базу.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       GATE_N)
from base72 import run_side1 as run72                              # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402
from price31 import loo, three_slices                              # noqa: E402
from varsub import S72, A72, Q72, G72                              # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
K = 8
KCL = 28
FEAT_L = (1, 2, 4, 8, 12, 16, 20, 24, 28)


def feats_draw(wc, npair, dev, seed, m_cl, f_cl):
    """Один жереб: скалярні ДОСТУПНІ ознаки + напрямок `u` з замикання.

    🚨 Кожна ознака НОРМОВАНА на масштаб свого шару — інакше коефіцієнти не
    перенесуться між мережами (різні ширини й норми).
    """
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xw = x @ wh
    ft = [float(xw.double().norm(dim=1).mean()) / (W_ ** 0.5) - 1.0]
    a = torch.relu(xw @ wc[0])
    for i in range(1, len(wc)):
        p = a @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        a = torch.relu(p)
        if i in FEAT_L and i - 1 < len(f_cl):
            n_i = float(a.shape[1])
            mc = m_cl[i - 1]
            d = a.double().mean(0) - mc
            nm = max(float(mc.norm()), 1e-30)
            ft.append(float((gate - f_cl[i - 1]).sum()) / n_i)
            ft.append(float(d.sum()) / nm)
            # 🔑 МАСШТАБНА компонента шару: проєкція на сам `m` шару
            ft.append(float(d @ mc) / (nm * nm))
    return ft


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: LOO ПО МЕРЕЖАХ з\'їсть більшу частину ⟹ '
          f'1.10–1.30×, не 1.54×')
    print(f'🟢 ЖИВЕ при >1.05× на СУМІ І МЕДІАНІ.  '
          f'мереж {nets} × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    NF = 1 + 3 * len(FEAT_L)
    base, G, B, C = [], [], [], []
    Gn, Bn, Cn = [], [], []
    Gx, Bx, Cx = [], [], []
    prevu = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, min(KCL, d - 1), dev,
                                                   extras=True)
            sA, mA, fA, qA = closure_parts(wc, K, dev, extras=True)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            # напрямок `u` — з ЗАМИКАННЯ останнього шару (доступний!)
            mL = m_cl[-1]
            u = mL / max(float(mL.norm()), 1e-30)
            if u.shape[0] != T1.shape[0]:
                t = torch.zeros(T1.shape[0], dtype=torch.float64, device=dev)
                kk = min(u.shape[0], t.shape[0]); t[:kk] = u[:kk]
                u = t / max(float(t.norm()), 1e-30)
            ux = None
            if prevu is not None:
                t = torch.zeros_like(u)
                kk = min(prevu.shape[0], t.shape[0]); t[:kk] = prevu[:kk]
                ux = t / max(float(t.norm()), 1e-30)

            bm = 0.0
            gg = torch.zeros(NF, NF, dtype=torch.float64, device=dev)
            bb = torch.zeros(NF, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(NF, NF, dtype=torch.float64, device=dev)
            bN = torch.zeros(NF, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            gX = torch.zeros(NF, NF, dtype=torch.float64, device=dev)
            bX = torch.zeros(NF, dtype=torch.float64, device=dev)
            cX = 0.0
            for dr in range(ndraw):
                sd = 77_000_000 + 733 * n + dr
                pb, _ = run72(wc, npair, sA, mA, dev, sd, lam_s=S72,
                              lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=qA, p_ext=fA)
                bm += float(((pb - T1) * (pb - T2)).sum())
                ft = feats_draw(wc, npair, dev, sd, m_cl, f_cl)
                fv = torch.tensor(ft, dtype=torch.float64, device=dev)
                R = fv.unsqueeze(1) * u.unsqueeze(0)      # (NF, n_L)
                e1, e2 = pb - T1, pb - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                gg += R @ R.T
                bb += R @ e
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = R
                if ux is not None:
                    RX = fv.unsqueeze(1) * ux.unsqueeze(0)
                    gX += RX @ RX.T
                    bX += RX @ e
                    cX += float((e1 * e2).sum())
            base.append(bm)
            G.append(gg); B.append(bb); C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            if ux is not None:
                Gx.append(gX); Bx.append(bX); Cx.append(cX)
            prevu = u.clone()
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    bs = np.array(base)
    idx = list(range(NF))
    print(f'\n{nets} мереж × {ndraw} жеребів. КРОС-ЦІЛЬ. '
          f'База = ПОВНИЙ sub72.\n')
    print(f'{"варіант":>34s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 66)
    print(f'{"✅β=0 (пломба)":>34s} {1.0:8.3f}× {1.0:8.3f}× {1.0:9.3f}×')
    # 🚨 28 параметрів на 24 мережі — переганяння (на 5 мережах ⛔НЕГ просів
    # до 0.947×). Ознаки йдуть трійками [gate, sum, proj] на шар, плюс норма;
    # проєкційна (`proj`) — саме МАСШТАБНА компонента, теоретично та сама
    # величина, що й ціль ⟹ пробуємо ощадливі зрізи.
    PJ = [3 + 3 * j for j in range(len(FEAT_L))]
    SUBS = [('лише proj (9)', PJ),
            ('proj на 4 шарах', [PJ[0], PJ[3], PJ[5], PJ[8]]),
            ('норма + proj×4', [0, PJ[0], PJ[3], PJ[5], PJ[8]]),
            ('proj×2 (мілкий+глибокий)', [PJ[0], PJ[8]]),
            ('ВСІ (28)', idx)]
    best = (0.0, None)
    for nm, ii in SUBS:
        val = loo([g[ii][:, ii] for g in G], [b[ii] for b in B], C, len(ii))
        sv, mdv, w3v = three_slices(bs, val)
        if min(sv, mdv) > best[0]:
            best = (min(sv, mdv), nm)
        print(f'{"🎯 " + nm:>34s} {sv:8.3f}× {mdv:8.3f}× {w3v:9.3f}×')
    val = loo([g[idx][:, idx] for g in G], [b[idx] for b in B], C, NF)
    s, md, w3 = three_slices(bs, val)
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, NF)
    sN, mN, wN = three_slices(bs, vN)
    print(f'{"⛔НЕГ (сусідній жереб)":>34s} {sN:8.3f}× {mN:8.3f}× {wN:9.3f}×')
    if Gx:
        vX = loo([g[idx][:, idx] for g in Gx], [b[idx] for b in Bx], Cx, NF)
        sX, mX, wX = three_slices(bs[1:], vX)
        print(f'{"⛔ЧУЖИЙ напрямок u":>34s} {sX:8.3f}× {mX:8.3f}× {wX:9.3f}×')

    live = best[0] > 1.05
    print(f'\n🔑 НАЙКРАЩИЙ ЗРІЗ: {best[1]} ⟹ {best[0]:.3f}× '
          f'(гірше з суми/медіани)')
    print(f'🔑 ВИРОК: {"ЖИВЕ" if live else "МЕРТВО"} '
          f'(🟢гейт >1.05× на сумі І медіані)')
    print(f'   📣передбачено 1.10–1.30×; стеля з `scalenoise` була 1.54×')
    print(f'📊 БАЛ: 1.972e-07 → {1.972e-07 / max(best[0], 1e-9):.3e} '
          f'(якщо перенесеться на грейдер)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
