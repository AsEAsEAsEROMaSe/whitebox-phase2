"""ЧИ МОЖНА ЗМЕНШИТИ `b₃₀` — ЄДИНЕ, ЩО ЛИШИЛОСЬ.

ЗВІДКИ (`mdepth.py`): увесь `m`-канал — це ОДИН шар 30 (13.435× з 14.893×).
(`ladder31.py`): зміщення `m` менше вдвічі ⟹ 1.176×, вчетверо ⟹ 1.694×.
⟹ ТЗ: зменшити `b₃₀ = m_замик,₃₀ − m_іст,₃₀` (256 чисел на мережу).

ЩО ВЖЕ ВІДОМО ([[finding-closure-bias-is-a-universal-function-of-8-neuron-features]]):
FEAT(8) тримає 0.642 енергії `b` на шарі 30; спільні по мережах коефіцієнти
дають `R²_oos = 0.568` — АЛЕ це міряно на ВСІХ шарах разом, і гілку вбивав
механізм «залишок СКОРЕЛЬОВАНИЙ МІЖ ШАРАМИ». Тут канал ОДИН ⟹ механізм не діє.

ОЧІКУВАННЯ ДО ПРОГОНУ: `R²≈0.6` ⟹ залишок `k≈0.63` ⟹ за драбиною **1.08–1.10×**.
ПОРІГ: <1.15× — гілка не варта сабміта (ціновий гейт дає 1.054× безризиково).

ЯК. Прохід 1: для кожної мережі `b₃₀` та ознаки (істина потрібна ЛИШЕ тут).
Фіт `b/s` на ознаках, LOO ПО МЕРЕЖАХ, вага `v²` не потрібна — ціль уже
безрозмірна. Прохід 2: підстановка `m₃₀ ← m_замик,₃₀ − b̂₃₀` у ЯКІР шару 30,
виграш КРОС-ЦІЛЛЮ.

ПЕЧАТКИ: `b̂=0` → 1.007× · `b̂=b` (ідеальне) → 13.435× · ⛔коеф. з ІНШОЇ мережі ·
⛔шум тієї самої норми · ⛔НЕГ.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, _phi, _Phi)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KS = 31
SIG31 = list(range(0, 31))
MU30 = list(range(KS, KS + 30))
IDX = SIG31 + MU30
RIDGE = 1e-6


def feats(m, s):
    """Ознаки нейрона шару 30, усі ДОСТУПНІ ЗАДАРМА (лише із замикання)."""
    t = m / s
    ph, Ph = _phi(t), _Phi(t)
    return torch.stack([torch.ones_like(t), t, Ph, ph, t * ph, t * t,
                        t.abs(), s / s.mean()], 1)


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    # ── ПРОХІД 1: `b₃₀` та ознаки (істина потрібна ЛИШЕ тут) ──────────────
    X, Y, Sc, Bt, MM = [], [], [], [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            b30 = m_cl[29] - TA['m'][29]
            X.append(feats(m_cl[29], s_cl[29]))
            Y.append(b30 / s_cl[29])
            Sc.append(s_cl[29])
            Bt.append(b30)
            # 🔑 ПЕРЕНОС шар30 → вихід, доступний ІЗ ЗАМИКАННЯ ЗАДАРМА:
            # `∂E[вих_j]/∂m₃₀ᵢ = W₃₁[i,j]·Φ(t₃₁ⱼ)`. Балу важить `‖Mᵀb‖²`,
            # НЕ `‖b‖²` — тому фіт у метриці `MMᵀ`, а не в евклідовій.
            w31 = wc[31].double()
            t31 = (m_cl[29] @ w31) / s_cl[30]
            M = w31 * _Phi(t31).unsqueeze(0)
            MM.append(M @ M.T)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  прохід1 … {n+1}/{nets}', flush=True)

    nf = X[0].shape[1]
    GG = [x.T @ x for x in X]
    BB = [x.T @ y for x, y in zip(X, Y)]
    eye = torch.eye(nf, dtype=torch.float64, device=dev)
    bhat, r2, COEF = [], [], []
    for n in range(nets):
        A = sum(GG) - GG[n]
        v = sum(BB) - BB[n]
        co = torch.linalg.solve(A + RIDGE * torch.diagonal(A).mean() * eye, v)
        COEF.append(co)
        pred = X[n] @ co
        bhat.append(pred * Sc[n])
        ss = float(((Y[n] - pred) ** 2).sum())
        st = float(((Y[n] - Y[n].mean()) ** 2).sum())
        r2.append(1.0 - ss / max(st, 1e-300))
    print(f'\n`R²_oos` (LOO по мережах) для `b₃₀/s`: медіана '
          f'{np.median(r2):.4f}, середнє {np.mean(r2):.4f}')
    en = [float((bh ** 2).sum() / (bt ** 2).sum()) for bh, bt in zip(bhat, Bt)]
    print(f'частка енергії `‖b̂‖²/‖b‖²`: медіана {np.median(en):.4f}')
    kk = [float(((bt - bh) ** 2).sum() / (bt ** 2).sum())
          for bh, bt in zip(bhat, Bt)]
    print(f'ЗАЛИШОК `‖b−b̂‖²/‖b‖²`: медіана {np.median(kk):.4f} '
          f'⟹ ефективне `k` = {np.sqrt(np.median(kk)):.3f}')
    # ⛔контроль: КОЕФІЦІЄНТИ з іншої мережі, застосовані до ВЛАСНИХ ознак
    # (вектор `b̂` переносити НЕ можна — після відсіву мережі мають різну
    # кількість живих нейронів, спіймано падінням 193 проти 155)
    bhat_oth = [X[n] @ COEF[(n + 1) % nets] * Sc[n] for n in range(nets)]
    # 🎯 ФІТ У ПРАВИЛЬНІЙ МЕТРИЦІ: min ‖Mᵀ(b − diag(s)Xc)‖²
    Z = [Sc[n].unsqueeze(1) * X[n] for n in range(nets)]
    GW = [Z[n].T @ (MM[n] @ Z[n]) for n in range(nets)]
    BW = [Z[n].T @ (MM[n] @ Bt[n]) for n in range(nets)]
    bhat_w = []
    for n in range(nets):
        A = sum(GW) - GW[n]
        v = sum(BW) - BW[n]
        co = torch.linalg.solve(A + RIDGE * torch.diagonal(A).mean() * eye, v)
        bhat_w.append(Z[n] @ co)
    kw = [float(((bt - bh) ** 2).sum() / (bt ** 2).sum())
          for bh, bt in zip(bhat_w, Bt)]
    # залишок У МЕТРИЦІ БАЛУ — саме він мусить корелювати з виграшом
    kmw = [float((((bt - bh) @ mm) @ (bt - bh)) / ((bt @ mm) @ bt))
           for bh, bt, mm in zip(bhat_w, Bt, MM)]
    km0 = [float((((bt - bh) @ mm) @ (bt - bh)) / ((bt @ mm) @ bt))
           for bh, bt, mm in zip(bhat, Bt, MM)]
    print(f'🎯 фіт у метриці `MMᵀ`: залишок ‖·‖² {np.median(kw):.4f}, '
          f'а В МЕТРИЦІ БАЛУ {np.median(kmw):.4f} ⟹ `k` = '
          f'{np.sqrt(np.median(kmw)):.3f}')
    print(f'   (для порівняння: євклідів фіт у метриці балу '
          f'{np.median(km0):.4f} ⟹ `k` = {np.sqrt(np.median(km0)):.3f})\n')
    ko = [float(((bt - bh) ** 2).sum() / (bt ** 2).sum())
          for bh, bt in zip(bhat_oth, Bt)]
    print(f'⛔ те саме з ЧУЖИМИ коеф.: залишок {np.median(ko):.4f} '
          f'⟹ `k` = {np.sqrt(np.median(ko)):.3f}\n')

    # ── ПРОХІД 2: підстановка в ЯКІР шару 30 ─────────────────────────────
    NAMES = ['⛔b̂=0 (пломба 1.007×)', '🎯b̂ = фіт FEAT(8), LOO по мережах',
             '🎯🆕b̂ = фіт у МЕТРИЦІ `MMᵀ`', '⛔шум тієї самої норми',
             '✅b̂ = b (пломба 13.435×)']
    nv = len(NAMES)
    base_mse, C = [], []
    G = [[] for _ in range(nv)]
    B = [[] for _ in range(nv)]
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            g = torch.Generator(device=dev); g.manual_seed(4242 + n)
            nz = torch.randn(bhat[n].shape, generator=g, device=dev,
                             dtype=torch.float64)
            nz = nz * (bhat[n].norm() / nz.norm().clamp_min(1e-30))
            corr = [torch.zeros_like(bhat[n]), bhat[n],
                    bhat_w[n], nz, Bt[n]]
            mvars = [[m_cl[i] if i != 29 else m_cl[29] - c
                      for i in range(KS)] for c in corr]

            bm = 0.0
            nreg = 2 * KS
            gg = torch.zeros(nv, nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     collect_sigma=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                for v in range(nv):
                    _, A = run_side1(wc, npair, s_cl, mvars[v], dev, sd,
                                     lam_s=None, collect_anchors=True)
                    R = torch.cat([S_cl, A], 0)
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                    if v == 1:
                        RI = R
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = RI
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  прохід2 … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. База = САБМІТ.\n')
    print(f'{"варіант":>38s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 70)
    for v, nm in enumerate(NAMES):
        val = loo([g[IDX][:, IDX] for g in G[v]],
                  [b[IDX] for b in B[v]], C, len(IDX))
        s, md, w3 = three_slices(base, val)
        print(f'{nm:>38s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    vN = loo([g[IDX][:, IDX] for g in Gn], [b[IDX] for b in Bn], Cn, len(IDX))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ (зсув жеребу)":>38s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    print('\n🎯 ПОРІГ: <1.15× ⟹ гілка не варта сабміта (ціновий гейт 1.054× '
          'безризиковий).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
