"""ДВА КАНДИДАТИ ЗІ СКРИНІНГУ: `mean(a³)` і НЕДІАГОНАЛЬ `Ĉ` НА ВИХОДІ.

════════ ЗВІДКИ ════════
`statsweep.py` (24×32, опора = середнє по жеребах ⟹ СТЕЛЯ, усі ⛔НЕГ=0.0000):
понад уже працюючий `q=mean(a²)` (сирий `R²`=0.3974, звірка з паралельною
лінією 0.347 ✅) додають лише двоє:
    `mean(a³)`        ⊥q = **0.0777** ⟹ стеля **1.084×**, ціна ≈0
    недіагональ `Ĉ`   ⊥q = **0.1109** ⟹ стеля **1.125×**, ціна ~0.27% бюджету
⛔мертві: `gate` 0.0022 · пара `a⁺⊙a⁻` 0.0048 · розкид 0.0000 · `a⁴` 0.0392.
⛔радіус `a·‖z‖` виглядає найкраще (0.267), але це РАДІАЛЬНИЙ RB, уже
зміряний як 0.995× — вибілювання пришпилює `Σ‖z‖²`, ступінь свободи знятий.

════════ ЩО ТУТ МІРЯЄТЬСЯ ════════
Стеля була з ІДЕАЛЬНОЮ опорою. Тут — з РЕАЛЬНОЮ (замикання), і одразу двома
приладами:
  1️⃣ `r = ‖зміщення опори‖/‖шум‖` — БЕЗ істини (середнє по жеребах = істина).
     `r>1` ⟹ ознака вливає зміщення, і жоден `λ` її не врятує.
  2️⃣ реальний виграш, LOO ПО МЕРЕЖАХ, крос-ціль, база = ПОВНИЙ сабміт.

📐 ОПОРА ДЛЯ `mean(a³)` виводиться точно з тих самих `(m,s)` замикання:
    I₀=Φ(t) · I₁=mΦ+sφ · **I_k = m·I_{k−1} + (k−1)s²·I_{k−2}**,  t=m/s
    (✅звірка: I₂ = (m²+s²)Φ + msφ — рівно формула `d+m²` у коді сабміту)
📐 ОПОРА ДЛЯ НЕДІАГОНАЛІ — другий момент замикання `C + m mᵀ`, рядкові суми
    мінус діагональ. 🚨 РИЗИК НАЗВАНО: С74 казала, що саме НЕДІАГОНАЛЬ несе
    похибку замикання на глибині ⟹ `r` може бути великим. Це і перевіряємо.

════════ ЦІНА (порахована ДО прогону) ════════
`mean(a³)`: одна редукція на виході ⟹ ~0.00% бюджету, гейт **1.000×**.
недіагональ: `âᵀâ` = 2n·256² ≈ один зайвий шар = 0.27% ⟹ гейт **1.027×**.
🚨 обидва потребують ГЛИБОКОГО замикання (до шару 31) для опори: 23 зайві
шари × 0.0247% = **0.57%** ⟹ спільний гейт зростає до **1.057×**.
(Паралельна лінія має той самий блокер і шукає `C` рангу ~32.)

════════ 📣 ПЕРЕДБАЧЕННЯ ════════
📣 `r` для недіагоналі буде ГІРШИМ, ніж для `a³` (недіагональ = слабке місце
   замикання), і саме `r` вирішить, а не стеля.
📣 реальний виграш очікую **1.01–1.06×** — тобто спільний гейт 1.057× цілком
   може не взятись, і тоді хід переїжджає в «здешевити глибоке замикання».
🟢 ЖИВЕ, якщо виграш > свого гейта на СУМІ І МЕДІАНІ.

🚨 ПЛОМБИ
  ✅ β=0 = РІВНО 1.000× · ✅база = MSE 2.300e-06 (звірка з `biassplit`)
  ⛔ НЕГ: ознаки з СУСІДНЬОГО жеребу (🚨тут це НЕ нуль — див. `gdeep`:
     опора несе МЕРЕЖЕВУ складову ⟹ шумова підлога приладу ≈1.03×)
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, closure_parts, W_, GATE_N,        # noqa: E402
                    _phi, _Phi)
from varsub import S72, A72, Q72, G72                                 # noqa: E402
from price31 import loo, three_slices                                 # noqa: E402
from biassplit import truth_full, run_collect                         # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8
LAM = (S72, A72, Q72, G72)
GATE_A3, GATE_OFF, GATE_BOTH = 1.057, 1.057, 1.057   # усі несуть глибоке замикання


def closure_out(wc, dev):
    """Замикання до ОСТАННЬОГО шару. Повертає (m, s, Φ, C) виходу."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    mp = s = Ph = None
    for i in range(len(wc)):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
    return mp, s, Ph, C, m


def moments_ref(mp, s, Ph, k=3):
    """E[relu(p)^k] через I_k = m·I_{k−1} + (k−1)s²·I_{k−2}."""
    ph = _phi(mp / s)
    I = [Ph, mp * Ph + s * ph]
    for j in range(2, k + 1):
        I.append(mp * I[j - 1] + (j - 1) * s * s * I[j - 2])
    return I[k]


def feats(wc, npair, dev, seed):
    """Один жереб: `mean(a³)` і рядкові суми недіагоналі 2-го моменту виходу."""
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (x.T @ x) / denom
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    a = torch.relu(x @ (wh @ wc[0]))
    for i in range(1, depth):
        a = torch.relu(a @ wc[i])
    ad = a.double()
    M2 = (ad.T @ ad) / denom
    return ad.pow(3).mean(0), M2.sum(1) - M2.diagonal()


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print('📣 ПЕРЕДБАЧЕНО: `r` недіагоналі ГІРШИЙ за `a³`; виграш 1.01–1.06× '
          f'при СПІЛЬНОМУ гейті {GATE_BOTH:.3f}× (глибоке замикання 0.57%)')
    print(f'🟢 ЖИВЕ, якщо виграш > гейта на сумі І медіані.  '
          f'{nets} мереж × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    NB = 2
    base, G, B, C = [], [], [], []
    Gn, Bn, Cn = [], [], []
    R1 = [0.0, 0.0]      # [Σ‖b‖², Σшум²] для a³
    R2 = [0.0, 0.0]      # те саме для недіагоналі
    mse_chk, nout = 0.0, 0
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            cl8 = closure_parts(wc, KSIG, dev, extras=True)
            mp, s, Ph, Ccl, mcl = closure_out(wc, dev)
            ref3 = moments_ref(mp, s, Ph, 3)
            M2cl = Ccl + torch.outer(mcl, mcl)
            refo = M2cl.sum(1) - M2cl.diagonal()
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            nout += T1.shape[0]
            lam = torch.tensor([LAM[q][i] for q in range(4)
                                for i in range(KSIG)],
                               dtype=torch.float64, device=dev)
            bm, cc = 0.0, 0.0
            gg = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bb = torch.zeros(NB, dtype=torch.float64, device=dev)
            gN = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bN = torch.zeros(NB, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            s3, so = [], []
            for dr in range(ndraw):
                sd = 88_000_000 + 733 * n + dr
                p0, F = run_collect(wc, npair, cl8, dev, sd)
                pb = p0 - lam @ F
                f3, fo = feats(wc, npair, dev, sd)
                s3.append(f3); so.append(fo)
                R = torch.stack([f3 - ref3, fo - refo], 0)
                e1, e2 = pb - T1, pb - T2
                e = 0.5 * (e1 + e2)
                bm += float((e1 * e2).sum()); cc += float((e1 * e2).sum())
                gg += R @ R.T; bb += R @ e
                if prevR is not None:
                    gN += prevR @ prevR.T; bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = R
            # r = ‖зміщення опори‖/‖шум‖ (істина = середнє по жеребах)
            for lst, ref, acc in ((s3, ref3, R1), (so, refo, R2)):
                A = torch.stack(lst, 0)
                mu = A.mean(0)
                b2 = float(((mu - ref) ** 2).sum()) - \
                    float(((A - mu) ** 2).sum()) / (ndraw * (ndraw - 1.0))
                acc[0] += max(b2, 0.0)
                acc[1] += float(((A - mu) ** 2).sum()) / ndraw
            mse_chk += bm / ndraw
            base.append(bm); G.append(gg); B.append(bb); C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    bs = np.array(base)
    mse = mse_chk / nout
    print(f'\n✅ПЛОМБА БАЗИ: MSE {mse:.3e} (biassplit 2.300e-06, '
          f'розбіжність {abs(mse-2.300e-6)/2.300e-6*100:.1f}%)')
    r_a3 = (R1[0] / max(R1[1], 1e-300)) ** 0.5
    r_of = (R2[0] / max(R2[1], 1e-300)) ** 0.5
    print(f'\n1️⃣ `r` = ‖зміщення опори‖/‖шум‖  (r<1 ознака чиста):')
    print(f'      mean(a³)          r = {r_a3:.3f}')
    print(f'      недіагональ Ĉ     r = {r_of:.3f}')
    print(f'   📣передбачено: у недіагоналі ГІРШЕ ⟹ '
          f'{"ПІДТВЕРДИЛОСЬ" if r_of > r_a3 else "🚨 НЕ підтвердилось"}')

    print(f'\n2️⃣ реальний виграш, LOO ПО МЕРЕЖАХ, база = ПОВНИЙ сабміт:\n')
    print(f'{"канал":>26s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s} '
          f'{"гейт":>8s} {"вирок":>7s}')
    print('-' * 76)
    print(f'{"✅β=0 (пломба)":>26s} {1.0:8.3f}× {1.0:8.3f}× {1.0:9.3f}× '
          f'{"—":>8s} {"—":>7s}')
    best = (0.0, None)
    for nm, ii, gt in (('mean(a³)', [0], GATE_A3),
                       ('недіагональ Ĉ', [1], GATE_OFF),
                       ('обидва', [0, 1], GATE_BOTH)):
        val = loo([g[ii][:, ii] for g in G], [b[ii] for b in B], C, len(ii))
        sv, mdv, w3v = three_slices(bs, val)
        ok = min(sv, mdv) > gt
        if min(sv, mdv) / gt > best[0]:
            best = (min(sv, mdv) / gt, nm)
        print(f'{"🎯 " + nm:>26s} {sv:8.3f}× {mdv:8.3f}× {w3v:9.3f}× '
              f'{gt:7.3f}× {"ЖИВЕ" if ok else "мертво":>7s}')
    idx = [0, 1]
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, NB)
    sN, mN, wN = three_slices(bs, vN)
    print(f'{"⛔НЕГ (сусідній жереб)":>26s} {sN:8.3f}× {mN:8.3f}× {wN:9.3f}× '
          f'{"":>8s} {"":>7s}')
    print(f'   🚨підлога приладу ≈1.03× (⛔НЕГ несе МЕРЕЖЕВУ складову опори)')

    print(f'\n🔑 НАЙКРАЩЕ ДО СВОГО ГЕЙТА: {best[1]} ⟹ {best[0]:.3f}× (>1 = виграш)')
    print(f'🔑 ВИРОК: {"ЖИВЕ" if best[0] > 1.0 else "МЕРТВО"}')
    print(f'   📣стелі зі скринінгу були 1.084× (a³) і 1.125× (недіаг)')
    print(f'📊 БАЛ: 1.954e-07 → '
          f'{1.954e-07 / max(best[0], 1e-9) if best[0] > 1 else 1.954e-07:.3e} '
          f'· лідер 8.0e-09 (розрив 24.4×)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
