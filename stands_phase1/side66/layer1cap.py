"""СКІЛЬКИ НЕВЗЯТОГО В ШАРІ 1 — ЄДИНОМУ, ДЕ `E[g]` ВІДОМЕ ТОЧНО. (С73)

════════ ЧОМУ САМЕ ШАР 1 ════════
Контрольна змінна працює як `g − E[g]` і вимагає `E[g]` ВІДОМЕ НЕЗАЛЕЖНО від
вибірки. Сьогодні зміряно, що це і є ЄДИНЕ вузьке місце всієї задачі:
активації шару 30 корелюють із виходом на 0.996 (`ρ²=0.9915`, тобто 118×), але
`E[a₃₀]` відоме лише із замикання з похибкою 0.873% — у 485× грубішою за шум,
який нею міряють.
🔑 А на ШАРІ 1 `E[a₁]` відоме ТОЧНО: ZCA робить емпіричну коваріацію входу
РІВНО одиничною ⟹ `s₁ⱼ = ‖W₁[:,j]‖` точно, і `E[relu(p)] = s·φ(0)`.
📊 corr(a₁, вихід) = 0.527 ⟹ `ρ²=0.278` ⟹ стеля **1.385×**, тоді як УСЯ наша
КЗ дає 1.125×, і шар 1 у ній — 9.8% з 11.1%.

════════ ПИТАННЯ ════════
Чинний сабміт несе нев'язку шару 1 вперед ОДНИМ вектором `v`, переносячи її
`v ← (v@W)·gate` з вагою 1. Це наближення. Питання: скільки з 1.385× ми
реально беремо, і чи є запас у КРАЩИХ ВАГАХ (діагональних / матричних).

ВАРІАНТИ (усі — на ТІЙ САМІЙ нев'язці `r₁ = x̄₁ − E[a₁]`, ціна однакова):
  A як у сабміті: перенос `v` затвором, вага 1;
  B скаляр λ: `pred = x̄ − λ·перенесене`;
  C діагональ: своя вага на КОЖНУ компоненту виходу;
  D ⚠️ОРАКУЛ: повна матриця `Λ = Cov(x̄₃₁, r₁)·Var(r₁)⁻¹` (стеля класу).

🚨 ПЛОМБИ
  ✅ `E[a₁]` перевіряється прямо: `‖x̄₁ − s·φ(0)‖` мусить бути на рівні ШУМУ
     вибірки, а не зміщення;
  🔑 `Λ` фітиться на ОДНОМУ наборі жеребів, міряється на ІНШОМУ (інакше
     оракул ділить жереб із ціллю і завищує);
  ⛔ КОНТРОЛЬ: `r₁` з ЧУЖОЇ мережі ⟹ мусить дати ≈1.00×.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_                      # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NPAIR = 3100          # робочий розмір жеребу
NDRAW = 120           # жеребів на мережу (60 фіт + 60 вимір)
INV_SQRT_2PI = 0.3989422804014327
GATE_N = 1024


def whiten_of(x):
    """ZCA: робить емпіричну коваріацію РІВНО одиничною (як у сабміті)."""
    cov = x.T @ x / float(x.shape[0])
    ev, U = torch.linalg.eigh(cov)
    return U @ torch.diag(1.0 / ev.clamp_min(1e-8).sqrt()) @ U.T


def draw(wc, dev, seed):
    """Один жереб: повертає `x̄₃₁`, нев'язку шару 1 `r₁`, перенесене `v`."""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(NPAIR, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    x = x @ whiten_of(x)
    w0 = wc[0]
    p1 = x @ w0
    a1 = torch.relu(p1)
    n = float(a1.shape[0])
    s1 = (w0 * w0).sum(0).clamp_min(1e-30).sqrt()
    r1 = a1.double().sum(0) / n - s1.double() * INV_SQRT_2PI   # 🔑 E[a₁] ТОЧНЕ
    v = r1.clone()
    a = a1
    for i in range(1, len(wc)):
        p = a @ wc[i]
        gate = (p[:GATE_N] > 0).double().mean(0)
        v = (v @ wc[i].double()) * gate
        a = torch.relu(p)
    return a.double().sum(0) / n, r1, v


def main(nets=6, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = ['A як у сабміті (вага 1)', 'B скаляр λ', 'C діагональ',
             '⚠️D матриця Λ (оракул класу)', '⛔НЕГ (v чужої мережі)']
    res = {k: [] for k in names}
    prev = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            D = [draw(wc, dev, 60_000 + 7919 * n + d) for d in range(NDRAW)]
            M = torch.stack([d[0] for d in D])          # NDRAW × out
            V = torch.stack([d[2] for d in D])
            if n == 0:
                r0 = torch.stack([d[1] for d in D])
                print(f'  ✅ПЛОМБА `E[a₁]`: ‖сер. нев\'язки‖/‖розкид‖ = '
                      f'{float(r0.mean(0).norm() / r0.std(0).norm()):.4f} '
                      f'(мусить бути ≲1/√{NDRAW}={1/np.sqrt(NDRAW):.3f} '
                      f'⟹ зміщення НЕМА)')
            tgt = M.mean(0)                              # ~істина
            fit, tst = slice(0, NDRAW // 2), slice(NDRAW // 2, NDRAW)
            Mf, Vf, Mt, Vt = M[fit], V[fit], M[tst], V[tst]
            base = float(((Mt - tgt) ** 2).mean())

            def sc(P):
                return base / float(((Mt - P - tgt) ** 2).mean())

            res[names[0]].append(sc(Vt))
            # B: скаляр з фіт-половини
            e = Mf - tgt
            lam = float((e * Vf).sum() / (Vf * Vf).sum().clamp_min(1e-30))
            res[names[1]].append(sc(lam * Vt))
            # C: діагональ
            ld = ((e * Vf).sum(0) / (Vf * Vf).sum(0).clamp_min(1e-30))
            res[names[2]].append(sc(ld * Vt))
            # D: повна матриця (оракул класу)
            G = Vf.T @ Vf / Vf.shape[0]
            B_ = Vf.T @ e / Vf.shape[0]
            Lam = torch.linalg.solve(G + 1e-10 * torch.eye(
                G.shape[0], dtype=torch.float64, device=dev), B_)
            res[names[3]].append(sc(Vt @ Lam))
            if prev is not None and prev.shape == Vt.shape:
                res[names[4]].append(sc(prev))
            prev = Vt.clone()
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів (половина фіт / половина вимір)\n')
    print(f'{"варіант":32s} {"виграш":>9s} {"мереж >1":>10s}')
    print('-' * 54)
    for k in names:
        if res[k]:
            print(f'{k:32s} {np.median(res[k]):8.4f}× '
                  f'{sum(1 for q in res[k] if q > 1.0):6d}/{len(res[k])}')
    print('-' * 54)
    print('🎯 стеля шару 1 з corr=0.527 ⟹ 1.385×. Скільки з неї бере рядок A?')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
