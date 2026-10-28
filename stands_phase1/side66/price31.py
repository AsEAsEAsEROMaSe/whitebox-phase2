"""ПРАЙСИНГ ЗАМИКАННЯ НА ВСІ 31 ШАРИ — ГЕЙТ ДЛЯ П'ЯТИ ІДЕЙ ЮЗЕРА ОДРАЗУ.

ЗВІДКИ. Юзер приніс п'ять апаратів: ієрархічні матриці/FMM (Хакбуш, Грінгард-
Рохлін), зворотний аналіз похибки й модифіковані рівняння (Хайрер), розщеплення
Тротера й композиції Йосіди, формалізм Вігнера-Вейля-Мойяля, підпростори Крилова.
🔑 УСІ П'ЯТЬ виробляють ОДНЕ І ТЕ САМЕ: кращу рекурсію `(m, s)` замість гаусового
замикання. Ієрархія здешевлює 3-й порядок Плефки; модифіковані рівняння дають
похибку замикання формулою; Тротер підвищує порядок композиції кроку; Мойяль
дає ряд поправок до гаусової еволюції; Крилов будує базис оператора переносу.
⟹ **одна крива віддачі гейтить усі п'ять**.

🚨 ЧОМУ НЕ ВИСТАЧАЄ `price_closure.py`. Той стенд міряв те саме, але **лише на
шарах 1..8** (`KSIG=8`, бо саме там стоїть σ-поправка сабміту) і дав 1.2535× за
ІДЕАЛЬНЕ замикання. Це я й записав як «мур A: замикання не допоможе». Але С67
основної сесії міряла оракульне `s` на ВСІХ 31 шарі і отримала **2.83×** — тобто
мур A був спрайсений на восьмушці ланцюга. Тут — на всьому, на ЗВІРЕНІЙ базі.

ЩО РАХУЄМО. `s_k = s_іст + k·(s_замик − s_іст)`: `k=1` — наше замикання, `k=0` —
ідеальне. Свіп по `k` × глибині `K` дає ПОВНУ карту віддачі.

🔑 ЯК ФІТУЄТЬСЯ λ (без цього свіп не міряє нічого — урок С67).
Ланцюг переносу ЛІНІЙНИЙ за `v`, тож внесок σ-поправки шару `i` можна перенести
до виходу окремо: `pred(λ) = pred₀ − λᵀS`, де `S[i−1]` = `φ(m̂/ŝ)(ŝ−s_k)` шару `i`,
перенесене воротами. Тоді 31 коефіцієнт λ — це нормальні рівняння, а зрізи
`λ[:K]` дають глибинний свіп ЗАДАРМА (підматриці). Фіт — LOO ПО МЕРЕЖАХ.

КОНТРОЛІ
  * ✅ база = справжній сабміт (`KSIG=8`, фіксовані `LAM_S`, замикання) — усі
    виграші рахуються ПРОТИ НЕЇ;
  * ✅ `k=1, K=8` з фітованим λ мусить дати ≈1.0 (LAM_S уже підігнані);
  * ⛔ НЕГ: `S` з жеребу `d`, помилка `e` з жеребу `d+1` — рве зв'язок одного
    жеребу, мусить дати ≈1.0 (виграє поправка, а не підгонка);
  * ✅ `k=0, K=31` мусить вийти в район 2.83× (перехресна звірка з С67 основної);
  * 📊 ТРИ ЗРІЗИ на кожен рядок: сума · МЕДІАНА · сума без топ-3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, W_)
from tilt_wide import truth_last                                  # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 32
K31 = 31
KGRID = [1.0, 0.5, 0.25, 0.0]
DEPTHS = [8, 16, 24, 31]
RIDGE = 1e-8


def truth_layers31(wc, dev, seed, npair=524_288, chunk=65536):
    """Справжні `s` доактивацій і `m` активацій шарів 1..31."""
    ns = [w.shape[1] for w in wc]
    s1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    s2 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    a1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = x @ w
            pd = p.double()
            s1[i] += pd.sum(0)
            s2[i] += (pd * pd).sum(0)
            x = torch.relu(p)
            a1[i] += x.double().sum(0)
        done += k
    n = 2.0 * done
    s = [(s2[i] / n - (s1[i] / n) ** 2).clamp_min(1e-30).sqrt()
         for i in range(1, K31 + 1)]
    m = [a1[i] / n for i in range(1, K31 + 1)]
    return {'s': s, 'm': m}


def loo(G, b, c, depth, ridge=RIDGE):
    """LOO ПО МЕРЕЖАХ на зрізі перших `depth` шарів. Повертає (MSE по мережах)."""
    nn = len(G)
    Gs = [g[:depth, :depth] for g in G]
    bs = [v[:depth] for v in b]
    Gt = sum(Gs)
    bt = sum(bs)
    eye = torch.eye(depth, dtype=torch.float64, device=Gt.device)
    out = []
    for n in range(nn):
        A = Gt - Gs[n]
        # 🚨 СТАНДАРТИЗАЦІЯ ОБОВ'ЯЗКОВА: ридж, узятий від СЕРЕДНЬОЇ діагоналі,
        # убиває регресори тих шарів, де масштаб малий (спіймано: та сама
        # інформація дала 5.72× замість 10.46× лише через перемасштабування
        # рядків). У кореляційній формі фіт інваріантний до масштабу рядка.
        dg = torch.diagonal(A).clamp_min(1e-300).sqrt()
        An = A / torch.outer(dg, dg)
        lam = torch.linalg.solve(An + ridge * eye, (bt - bs[n]) / dg) / dg
        out.append(float(c[n] - 2.0 * lam @ bs[n] + lam @ (Gs[n] @ lam)))
    return np.array(out)


def three_slices(base, var):
    """сума · МЕДІАНА · сума без топ-3 (правило, що коштувало скасованого сабміту)."""
    base, var = np.asarray(base), np.asarray(var)
    s = base.sum() / max(var.sum(), 1e-300)
    md = float(np.median(base / np.maximum(var, 1e-300)))
    keep = np.argsort(-base)[3:]
    w3 = base[keep].sum() / max(var[keep].sum(), 1e-300)
    return s, md, w3


def main(nets=20, dev=DEV):
    torch.manual_seed(31415)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    base_mse = []
    G = {k: [] for k in KGRID}
    B = {k: [] for k in KGRID}
    C = {k: [] for k in KGRID}
    Gn, Bn, Cn = [], [], []                       # ⛔ НЕГ: зсунутий жереб

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, K31, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            nl = wc[-1].shape[1]

            bm = 0.0
            gg = {k: torch.zeros(K31, K31, dtype=torch.float64, device=dev)
                  for k in KGRID}
            bb = {k: torch.zeros(K31, dtype=torch.float64, device=dev)
                  for k in KGRID}
            cc = {k: 0.0 for k in KGRID}
            gN = torch.zeros(K31, K31, dtype=torch.float64, device=dev)
            bN = torch.zeros(K31, dtype=torch.float64, device=dev)
            cN = 0.0
            prevS, prevE = None, None

            for d in range(NDRAW):
                sd = 55_000_000 + 733 * n + d
                # ✅ БАЗА = справжній сабміт: KSIG=8, фіксовані LAM_S
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T) ** 2).sum())
                for k in KGRID:
                    sv = [TL['s'][i] + k * (s_cl[i] - TL['s'][i])
                          for i in range(K31)]
                    p0, S = run_side1(wc, npair, sv, m_cl, dev, sd,
                                      collect_sigma=True)
                    e = p0 - T
                    gg[k] += S @ S.T
                    bb[k] += S @ e
                    cc[k] += float((e * e).sum())
                    if k == 1.0:
                        if prevS is not None:
                            gN += prevS @ prevS.T      # S жеребу d−1
                            bN += prevS @ e            # e жеребу d
                            cN += float((e * e).sum())
                        prevS, prevE = S, e
        del W, wc
        torch.cuda.empty_cache()
        base_mse.append(bm)
        for k in KGRID:
            G[k].append(gg[k]); B[k].append(bb[k]); C[k].append(cc[k])
        Gn.append(gN); Bn.append(bN); Cn.append(cN)
        print(f'  … {n+1}/{nets}  (nl={nl}, npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів, ПОВНИЙ ланцюг сабміту '
          f'(вибілювання + антитетика + якір-0), λ фітується LOO ПО МЕРЕЖАХ\n')
    print(f'{"k":>6s} {"глибина K":>10s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 50)
    for k in KGRID:
        for dep in DEPTHS:
            v = loo(G[k], B[k], C[k], dep)
            s, md, w3 = three_slices(base, v)
            tag = ''
            if k == 1.0 and dep == 8:
                tag = '  ✅КОНТРОЛЬ: має бути ≈1.0'
            if k == 0.0 and dep == 31:
                tag = '  ← ІДЕАЛЬНЕ замикання на ВСІХ шарах = СТЕЛЯ ГІЛКИ'
            print(f'{k:6.2f} {dep:10d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    vN = loo(Gn, Bn, Cn, K31)
    s, md, w3 = three_slices(base, vN)
    print(f'{"⛔НЕГ":>6s} {"31 (зсув)":>10s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×'
          f'   ⛔ мусить бути ≈1.0')
    print('\n🔑 читати так: рядок `k=0, K=31` — це стеля ВСЬОГО класу '
          '«краще замикання», тобто ієрархічних матриць, модифікованих рівнянь, '
          'Тротера-Йосіди, Мойяля і Крилова РАЗОМ. Якщо він низький — жодну з '
          'п\'яти відкривати не варто, хоч би скільки коштувала доставка.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
