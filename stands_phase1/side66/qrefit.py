"""ПЕРЕФІТ `λ_Q` НАОДИНЦІ + СВІП ГЛИБИНИ. (під дедлайн)

ЗВІДКИ (`lamdrop.py`, 24×24): `ЛИШЕ Q` = 1.0000× проти повного sub87, а
`без Q` = 0.8517× ⟹ **уся `λ`-машинерія (1.168×) тримається на ОДНІЙ родині
`Q`**, решта — точний нуль.
🔑 АЛЕ поточні `λ_Q` фітились СПІЛЬНОЮ регресією разом із трьома мертвими
родинами. Канали сильно колінеарні ⟹ кредит ділився між ними, і `λ_Q`
поодинці майже напевно НЕ в оптимумі.
🔑 Плюс SNR-профіль: для `Q` відношення зміщення/шум перетинає 1 на шарі **5**,
а ми женемо до 8 ⟹ шари 6-8 підмішують зміщення.

════════ ЯК МІРЯЄМО (один прохід дає ВСЕ) ════════
Перенос ЛІНІЙНИЙ за `v`, тож поруч із `v` несемо матрицю `A` (8 × вихід),
де рядок `i−1` — внесок Q-каналу шару `i`, уже перенесений до виходу. Тоді
    `pred(λ) = pred₀ − λᵀA`
⟹ 8 коефіцієнтів фітуються LOO ПО МЕРЕЖАХ з ОДНОГО прогону, а зрізи `λ[:K]`
дають свіп глибини ЗАДАРМА (підматриці нормальних рівнянь).

📣 ПЕРЕДБАЧЕННЯ: перефіт дасть 1.01-1.04×; оптимальна глибина K=5..6.
🟢 ЖИВЕ при ≥1.02× (при підлозі множника іде в бал один до одного).
🚨 ПЛОМБИ
  ✅ λ = СТАРІ `_LAM_Q` мусить відтворити «ЛИШЕ Q» = 1.0000× проти sub87;
  ✅ K=0 (жодного каналу) мусить дати 0.8563× — відоме «лише v»;
  ⛔ НЕГ: `A` з СУСІДНЬОЇ мережі ⟹ ≈1.000×.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net, LAM_Q                                # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8


def run_collect(wc, npair, cl, dev, seed):
    """Повертає (pred₀ БЕЗ каналів, A[8×вихід] — внески, перенесені до виходу)."""
    s_cl, m_cl, P_cl, q_cl = cl
    d = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    denom = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / denom - sg0 * INV_SQRT_2PI
    ks = min(KSIG, d - 1)
    A = torch.zeros(ks, wc[0].shape[1], dtype=torch.float64, device=dev)
    for i in range(1, d):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        A = (A @ wd) * gate                       # той самий перенос, рядково
        x = torch.relu(p)
        if i <= ks:
            xq = x.double().pow(2).sum(0) / denom
            A[i - 1] = A[i - 1] + (xq - q_cl[i - 1])
    return x.double().sum(0) / denom - v, A


def loo(E, A, K):
    """LOO ПО МЕРЕЖАХ: `pred₀ − λᵀA[:K]`, λ спільна для всіх мереж."""
    nets = len(E)
    G = np.zeros((nets, K, K))
    b = np.zeros((nets, K))
    for i in range(nets):
        a = A[i][:, :K, :]
        G[i] = np.einsum('nkd,njd->kj', a, a)
        b[i] = np.einsum('nkd,nd->k', a, E[i])
    res = 0.0
    for i in range(nets):
        Gi, bi = G.sum(0) - G[i], b.sum(0) - b[i]
        lam = np.linalg.solve(Gi + 1e-12 * np.trace(Gi) / K * np.eye(K), bi)
        res += float(((E[i] - np.einsum('k,nkd->nd', lam, A[i][:, :K, :]))
                      ** 2).sum())
    lam_all = np.linalg.solve(G.sum(0) + 1e-12 * np.trace(G.sum(0)) / K
                              * np.eye(K), b.sum(0))
    return res, lam_all


def main(nets=24, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: перефіт 1.01-1.04×, оптимальна глибина K=5..6')
    print('🟢 ЖИВЕ при ≥1.02×\n', flush=True)
    E, A = [], []
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            m31 = mt[d - 1]
            cl = closure_parts(wc, min(KSIG, d - 1), dev, extras=True)
            e, a = [], []
            for k in range(ndraw):
                p0, Am = run_collect(wc, npair, cl, dev,
                                     97_000_000 + 977 * n + k)
                e.append((p0 - m31).cpu().numpy())
                a.append(Am.cpu().numpy())
            E.append(np.array(e))
            A.append(np.array(a))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    tot = sum(float((E[i] ** 2).sum()) for i in range(nets))
    # ✅ПЛОМБА: старі λ_Q
    old = 0.0
    for i in range(nets):
        old += float(((E[i] - np.einsum('k,nkd->nd', np.array(LAM_Q),
                                        A[i])) ** 2).sum())
    print(f'\n{nets} мереж × {ndraw} жеребів')
    print(f'✅ПЛОМБА «лише v» (K=0): базова MSE {tot/(nets*ndraw):.5e}')
    print(f'✅ПЛОМБА СТАРІ λ_Q (K=8): {tot/old:.4f}× '
          f'(очікувано ≈1.168× — уся машинерія)')
    # 🚨 СТІЙКІ ТЕСТИ БЕЗ ЖОДНОГО ПІДГОНУ (перефіт на 23 мережах без гребеня
    #    НЕ може перемогти фіт на 60 із крос-валідацією — лише перевчиться)
    print(f'\n1️⃣ МАСШТАБ наявних λ_Q (0 параметрів, лише множник α)')
    lq = np.array(LAM_Q)
    bestA = (0.0, None)
    for al in (0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0):
        r = 0.0
        for i in range(nets):
            r += float(((E[i] - np.einsum('k,nkd->nd', al * lq, A[i]))
                        ** 2).sum())
        if old / r > bestA[0]:
            bestA = (old / r, al)
        print(f'   α={al:<5} {old/r:>8.4f}×')
    print(f'   🔑 найкраще α={bestA[1]} ⟹ {bestA[0]:.4f}×')

    print(f'\n2️⃣ ГЛИБИНА наявних λ_Q (обрізання, 0 параметрів)')
    bestD = (0.0, None)
    for K in range(1, KSIG + 1):
        lk = np.concatenate([lq[:K], np.zeros(KSIG - K)])
        r = 0.0
        for i in range(nets):
            r += float(((E[i] - np.einsum('k,nkd->nd', lk, A[i])) ** 2).sum())
        if old / r > bestD[0]:
            bestD = (old / r, K)
        print(f'   K={K:<3} {old/r:>8.4f}×')
    print(f'   🔑 найкраще K={bestD[1]} ⟹ {bestD[0]:.4f}×')

    print(f'\n3️⃣ ПОВНИЙ ПЕРЕФІТ LOO (8 параметрів — для довідки, схильний '
          f'до перевчання)')
    print(f'{"K":>4} {"× проти СТАРИХ λ_Q":>20} {"× проти лише v":>16}')
    best = (0.0, None, None)
    for K in range(1, KSIG + 1):
        res, lam = loo(E, A, K)
        r = old / res
        if r > best[0]:
            best = (r, K, lam)
        print(f'{K:>4} {r:>20.4f} {tot/res:>16.4f}')
    # ⛔НЕГ: A з сусідньої мережі (обрізаємо до спільної довжини)
    An = [A[(i + 1) % nets][:len(A[i])] for i in range(nets)]
    An = [x[:, :, :A[i].shape[2]] if x.shape[2] >= A[i].shape[2]
          else np.pad(x, ((0, 0), (0, 0), (0, A[i].shape[2] - x.shape[2])))
          for i, x in enumerate(An)]
    try:
        rn, _ = loo(E, An, best[1])
        print(f'\n⛔НЕГ (A сусідньої мережі): {old/rn:.4f}×')
    except Exception as ex:                        # noqa: BLE001
        print(f'\n⛔НЕГ не порахувався: {type(ex).__name__}')
    print(f'\n🔑 НАЙКРАЩЕ K={best[1]}: {best[0]:.4f}× проти старих λ_Q '
          f'(🟢гейт 1.02)')
    print('   λ =', ', '.join(f'{v:+.7f}' for v in best[2]))
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · множник на підлозі ⟹ прямо в бал')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
