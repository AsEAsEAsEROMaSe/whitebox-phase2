"""ЧИ ПАДАЄ РАНГ АКТИВАЦІЙ ІЗ ГЛИБИНОЮ — І ЧИ МОЖНА ЦЕ ВЗЯТИ? (С74, ідея юзера)

ЗВІДКИ. Закриті були ДВА інші об'єкти:
  ⛔ранг ВАГ — 127.7/256, це рівно спектр iid-гауса, викидати нема чого;
  ⛔малорангова `C` у ЗАМИКАННІ — ранг 192, економія 2.67×, але замикання йде
    9 кроків із 31 ⟹ 1.012×.
Ранг АКТИВАЦІЙ у робочому циклі не мірявся НІКОЛИ. Механізм на боці ідеї: без
зсувів на критичній ініціалізації кореляція між двома входами росте з глибиною
⟹ хмара активацій пласкішає.

💰 ЩО ЦЕ ДАЛО Б. Якщо `X_ℓ ≈ 1·mᵀ + Y·Uᵀ` з `k` компонентами, то шар коштує
`2N·n·k` (стиснути → множити → розгорнути) замість `N·n²` ⟹ економія `n/2k`.
Базис береться з ПІЛОТА (128 пар уже рахуються) — шукати його дорого не треба.
При `k=30` це 4× бюджету ⟹ MSE вчетверо менша.

🚨 ПАСТКА, ВІДОМА НАПЕРЕД. Ефективний ранг міряє УЧАСТЬ, а зсув `m` чутливий до
ХВОСТА спектра — саме на цьому померла малорангова `C`. ⟹ міряємо НЕ ранг, а
ПОХИБКУ ВИХОДУ від обрізання.

📐 ГЕЙТ (число ДО прогону): `k`, при якому похибка `E[relu(x₃₁)]` < 0.1%.
    `k ≤ 60`  ⟹ економія >2× — КАНАЛ ВІДКРИТИЙ і великий;
    `k ≤ 100` ⟹ економія 1.3× — гранично;
    `k > 128` ⟹ економія <1 — МЕРТВО (стиснення дорожче за сам матмул).
Очікування: еф.ранг 128 → 40–80, але `k(99.9%)` лишиться >150 ⟹ гранично.

ЩО РАХУЄМО
  1. спектр центрованих активацій по шарах: PR-ранг, `k` для 90/99/99.9/99.99%;
  2. ЦІНОВИЙ вимір: прогін із обрізанням рангу на шарах ≥`L0`, похибка виходу
     проти ПОВНОГО прогону на ТИХ САМИХ семплах (⟹ шум вибірки скорочується
     ТОТОЖНО, лишається чиста похибка обрізання);
  3. базис — ДВА джерела: ⚠️оракул (та сама вибірка, СТЕЛЯ) і ✅пілот 128 пар
     (те, що реально доступне за ціною).

ПЛОМБИ
  ✅ `k = n` мусить дати похибку РІВНО 0 (обрізання нічого не викидає);
  ⛔ ВИПАДКОВИЙ базис того самого рангу — мусить бути помітно ГІРШИЙ за головні
     компоненти, інакше стенд міряє не структуру;
  ✅ шар 1 (вхід гаусів, вибілений) мусить показати ПОВНИЙ ранг — якщо там теж
     «низький», стенд міряє артефакт вибірки, а не мережу.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_                            # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NPAIR = 4096
PILOT = 128
FRACS = (0.90, 0.99, 0.999, 0.9999)
KGRID = (16, 32, 64, 128, 192)
L0 = 8                       # з якого шару обрізаємо ранг


def acts(wc, dev, seed, npair):
    """Антитетична вибірка; повертає СПИСОК активацій усіх шарів."""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, W_, device=dev, generator=g)
    a = torch.cat([z, -z], 0)
    out = []
    for i in range(len(wc)):
        a = torch.relu(a @ wc[i])
        out.append(a)
    return out


def spectrum(X):
    """Власні числа коваріації центрованих активацій (спадно)."""
    Xd = X.double()
    mu = Xd.mean(0)
    Xc = Xd - mu
    C = (Xc.T @ Xc) / float(Xc.shape[0])
    ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0.0)
    return ev, mu


def kfrac(ev, f):
    c = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-300)
    return int((c < f).sum().item()) + 1


def basis(X, k, mode, dev, seed=0):
    """Базис `k` головних напрямків центрованих активацій."""
    Xd = X.double()
    mu = Xd.mean(0)
    if mode == 'rand':
        g = torch.Generator(device=dev).manual_seed(seed)
        U = torch.randn(Xd.shape[1], k, dtype=torch.float64, device=dev,
                        generator=g)
        U, _ = torch.linalg.qr(U)
        return U, mu
    Xc = Xd - mu
    C = (Xc.T @ Xc) / float(Xc.shape[0])
    ev, V = torch.linalg.eigh(C)
    return V[:, -k:].contiguous(), mu


def run_trunc(wc, X0, bases, l0):
    """Прогін від шару `l0` з обрізанням рангу; `bases[i] = (U, mu)` або None."""
    a = X0
    for i in range(l0, len(wc)):
        b = bases.get(i)
        if b is not None:
            U, mu = b
            ad = a.double()
            a = (mu + (ad - mu) @ U @ U.T).to(a.dtype)
        a = torch.relu(a @ wc[i])
    return a.double().mean(0)


def main(nets=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    SP = {}
    ERR = {}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            A = acts(wc, dev, 910_000 + 7919 * n, NPAIR)
            AP = acts(wc, dev, 310_000 + 131 * n, PILOT)      # пілотна вибірка
            for i in range(L_):
                ev, _ = spectrum(A[i])
                pr = float((ev.sum() ** 2) / (ev * ev).sum().clamp_min(1e-300))
                row = SP.setdefault(i, {'n': [], 'pr': [],
                                        **{f'k{q}': [] for q in FRACS}})
                row['n'].append(int(ev.numel()))
                row['pr'].append(pr)
                for q in FRACS:
                    row[f'k{q}'].append(kfrac(ev, q))

            # ── ЦІНОВИЙ ВИМІР: похибка виходу від обрізання, шари ≥ L0 ──
            ref = A[-1].double().mean(0)
            nrm = float((ref ** 2).mean().sqrt())
            X0 = A[L0 - 1]                       # вхід у шар L0
            for mode in ('pca', 'pilot', 'rand'):
                for k in KGRID:
                    bs = {}
                    ok = True
                    for i in range(L0, L_):
                        nl = wc[i].shape[0]
                        if k >= nl:
                            ok = False
                            break
                        src = A[i - 1] if mode != 'pilot' else AP[i - 1]
                        bs[i] = basis(src, k, 'rand' if mode == 'rand' else 'pca',
                                      dev, seed=k + i)
                    if not ok:
                        continue
                    got = run_trunc(wc, X0, bs, L0)
                    e = 100.0 * float(((got - ref) ** 2).mean().sqrt()) / nrm
                    ERR.setdefault((mode, k), []).append(e)
            # ── ДЕ НАРОДЖУЄТЬСЯ ПОХИБКА: обрізати РІВНО ОДИН шар ──
            for i in range(L0, L_):
                for k in KGRID:
                    if k >= wc[i].shape[0]:
                        continue
                    bs = {i: basis(A[i - 1], k, 'pca', dev)}
                    got = run_trunc(wc, X0, bs, L0)
                    ERR.setdefault(('solo', i, k), []).append(
                        100.0 * float(((got - ref) ** 2).mean().sqrt()) / nrm)
            # ── ХВІСТ: обрізати лише шари ≥ L1 (усі одразу) ──
            for l1 in (16, 20, 24, 28):
                for k in KGRID:
                    bs, ok = {}, True
                    for i in range(l1, L_):
                        if k >= wc[i].shape[0]:
                            ok = False
                            break
                        bs[i] = basis(A[i - 1], k, 'pca', dev)
                    if not ok:
                        continue
                    got = run_trunc(wc, X0, bs, L0)
                    ERR.setdefault(('tail', l1, k), []).append(
                        100.0 * float(((got - ref) ** 2).mean().sqrt()) / nrm)
            # ✅ПЛОМБА: k = повний ранг ⟹ рівно 0
            full = {i: basis(A[i - 1], wc[i].shape[0], 'pca', dev)
                    for i in range(L0, L_)}
            g0 = run_trunc(wc, X0, full, L0)
            ERR.setdefault(('✅k=n', 0), []).append(
                100.0 * float(((g0 - ref) ** 2).mean().sqrt()) / nrm)
        del W, wc, A, AP
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    m = np.median
    print(f'\n{nets} мереж × {2*NPAIR} семплів (антитетично)\n')
    print(f'{"шар":>4s} {"n":>5s} {"PR-ранг":>8s} '
          + ' '.join(f'{"k" + format(100*q, "g") + "%":>8s}' for q in FRACS))
    print('-' * 56)
    for i in range(L_):
        r = SP[i]
        print(f'{i+1:4d} {int(m(r["n"])):5d} {m(r["pr"]):8.1f} '
              + ' '.join(f'{int(m(r[f"k{q}"])):8d}' for q in FRACS))
    print('-' * 56)

    print(f'\n💰 ЦІНОВИЙ ВИМІР: обрізання рангу на шарах {L0+1}..32, '
          f'похибка `E[relu(x₃₁)]` на ТИХ САМИХ семплах\n')
    print(f'{"k":>5s} {"економія":>9s} {"⚠️оракул PCA":>14s} '
          f'{"✅пілот 128":>13s} {"⛔випадковий":>14s}')
    print('-' * 60)
    for k in KGRID:
        if ('pca', k) not in ERR:
            continue
        eco = 256.0 / (2.0 * k)
        cells = []
        for mode in ('pca', 'pilot', 'rand'):
            v = ERR.get((mode, k))
            cells.append(f'{m(v):11.4f}%' if v else f'{"—":>12s}')
        print(f'{k:5d} {eco:8.2f}× {cells[0]:>14s} {cells[1]:>13s} '
              f'{cells[2]:>14s}')
    print('-' * 60)
    print(f'\n🔬 ДЕ НАРОДЖУЄТЬСЯ ПОХИБКА: обрізано РІВНО ОДИН шар\n')
    print(f'{"шар":>4s} ' + ' '.join(f'{"k=" + str(k):>9s}' for k in KGRID))
    print('-' * 54)
    for i in range(L0, L_):
        cells = []
        for k in KGRID:
            v = ERR.get(('solo', i, k))
            cells.append(f'{m(v):8.4f}%' if v else f'{"—":>9s}')
        print(f'{i+1:4d} ' + ' '.join(f'{c:>9s}' for c in cells))
    print('-' * 54)

    print(f'\n🎯 ТІЛЬКИ ХВІСТ: обрізано шари ≥ L1 (де ранг найменший)\n')
    print(f'{"L1":>4s} {"економія":>9s} '
          + ' '.join(f'{"k=" + str(k):>9s}' for k in KGRID))
    print('-' * 64)
    for l1 in (16, 20, 24, 28):
        for k in KGRID:
            pass
        nt = L_ - l1
        cells = []
        for k in KGRID:
            v = ERR.get(('tail', l1, k))
            cells.append(f'{m(v):8.4f}%' if v else f'{"—":>9s}')
        # економія: `nt` шарів дешевшають у n/2k, решта — ні
        eco = [L_ / ((L_ - nt) + nt * (2.0 * k) / 256.0) for k in KGRID]
        print(f'{l1+1:4d} {eco[0]:8.2f}× ' + ' '.join(f'{c:>9s}' for c in cells))
    print('-' * 64)
    print('   (економія в колонці — для k=16; для більших k вона менша)')
    pl = ERR.get(('✅k=n', 0))
    print(f'✅ПЛОМБА (k=n, нічого не викидається): {m(pl):.3e}%  '
          f'{"OK" if m(pl) < 1e-9 else "🛑"}')
    print('\n🎯 ГЕЙТ: `k` із похибкою <0.1% при економії >2× (тобто k≤60) — '
          'КАНАЛ ВІДКРИТИЙ.\n   Якщо потрібне k>128 — стиснення дорожче за '
          'сам матмул ⟹ МЕРТВО.\n'
          '🚨 «пілот» — те, що реально доступне; «оракул» — стеля цієї схеми.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
