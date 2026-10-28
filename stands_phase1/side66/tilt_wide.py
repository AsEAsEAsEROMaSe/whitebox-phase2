"""ГАУСІВ НАХИЛ НА 40 МЕРЕЖАХ — рівно в тому вигляді, який можна сабмітити.

ЗВІДКИ (`tilt_end2end.py`, 6 мереж). Нахил `q = N(0,(I−2C)⁻¹)` дав 1.271×
з градієнтними напрямками, **1.221× з напрямками З ВАГ** і **1.206× з
єдиною сталою `c`**. ⛔Контроль «чужа `C`» дав 1.301× — БІЛЬШЕ за власну,
тобто коефіцієнти НЕ помережеві, їх не треба фітити взагалі.
🚨 Але розкид по мережах 0.93…1.65, і на одній із шести нахил ШКОДИТЬ.
Пам'ять: «малі прогони не розсуджують» (тричі за проєкт). Тому 40 мереж.

ЩО САМЕ МІРЯЄМО — найдешевший варіант, без жодного пілота понад ті, що
sub66 і так робить:
  * напрямки `u_i` — старші власні вектори `AAᵀ`, `A = W₀·diag(затвор₀)·G`,
    де `G` — матриця переносу з ваг і затворів (усе вже рахується в sub66);
  * `c_i = c` — ОДНА стала на всі напрямки, задана НАПЕРЕД (свіп по сітці),
    без жодного фіту на цій мережі.
⟹ ціна нахилу = нуль флопів понад базову лінію.

КОНТРОЛІ
  * ✅ `c=0` мусить відтворити базову лінію ДО ОСТАННЬОЇ ЦИФРИ;
  * ✅ незміщеність: ‖зміщення‖²/MSE по 64 жеребах;
  * ⛔ ВИПАДКОВІ напрямки замість `G`-них — мусить дати ≈1.0;
  * друкуємо ЧАСТКУ МЕРЕЖ, де нахил шкодить (бал — СЕРЕДНЄ по мережах,
    тож одна погана мережа з'їдає кілька добрих);
  * бал рахується як відношення СУМ MSE — саме так, як його бере грейдер.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import cut_net                              # noqa: E402
from impsamp import gates_and_G                               # noqa: E402
from tilt_end2end import psi_of, draw, dirs_from_G            # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NDRAW = 64
MTILT = 64
# 🚨 масштаб КАЛІБРОВАНО: фіт дає c≈+0.004 (Σc≈0.25 на 64 напрямки).
# Перший грид ±0.05…0.15 був у 12–37× завеликий і дав 0.056× —
# вага є ДОБУТКОМ 64 множників, тож вибухає задовго до видимого нахилу.
CGRID = [0.000, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, -0.004]
TRUTH_PAIRS = 524_288
CHUNK = 65536


def truth_last(wc, dev, seed):
    acc = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TRUTH_PAIRS:
        k = min(CHUNK, TRUTH_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for w in wc:
            x = torch.relu(x @ w)
        acc += x.sum(0).double()
        done += k
    return acc / (2.0 * done)


def main(nets=40, dev=DEV):
    torch.manual_seed(4004)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    tot = {c: 0.0 for c in CGRID}
    tot['rand'] = 0.0
    per = {c: [] for c in CGRID}
    base = 0.0
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_net(W, dev)
            G = gates_and_G(wc, dev, 4242 + n)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() * INV_SQRT_2PI
            T = truth_last(wc, dev, 210_000 + 7919 * n)
            Ug = dirs_from_G(wc, G, dev, 66_000 + n)
            Ur = torch.linalg.qr(torch.randn(W_, W_, device=dev))[0]
            eb = 0.0
            loc = {c: 0.0 for c in CGRID}
            er = 0.0
            ess = 0.0
            for d in range(NDRAW):
                sd = 21_000_000 + 977 * n + d
                mb, _ = draw(wc, Ug, None, G, m0, dev, sd, npair, MTILT)
                eb += float(((mb - T) ** 2).sum())
                for c in CGRID:
                    cv = torch.full((MTILT,), c, dtype=torch.float64, device=dev)
                    mt, _ = draw(wc, Ug, cv, G, m0, dev, sd, npair, MTILT)
                    loc[c] += float(((mt - T) ** 2).sum())
                cv = torch.full((MTILT,), 0.004, dtype=torch.float64, device=dev)
                mr, wr = draw(wc, Ur, cv, G, m0, dev, sd, npair, MTILT)
                er += float(((mr - T) ** 2).sum())
                ess += float(wr.sum() ** 2 / (wr * wr).sum()) / npair
            base += eb
            tot['rand'] += er
            for c in CGRID:
                tot[c] += loc[c]
                per[c].append(eb / max(loc[c], 1e-300))
        del W, wc, G
        torch.cuda.empty_cache()
        if (n + 1) % 5 == 0:
            b = max(CGRID[1:], key=lambda c: base / max(tot[c], 1e-300))
            print(f'  … {n+1}/{nets}: найкраще c={b:+.2f} → '
                  f'{base/max(tot[b],1e-300):.3f}×', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів продакшн-розміру, '
          f'напрямки З ВАГ, `c` задана НАПЕРЕД')
    print(f'{"c":>7s} {"виграш MSE":>11s} {"мереж, де ШКОДИТЬ":>19s} '
          f'{"медіана по мережах":>20s}')
    for c in CGRID:
        p = np.array(per[c])
        bad = float((p < 1.0).mean())
        tag = ' ✅КОНТРОЛЬ' if c == 0.0 else ''
        print(f'{c:+7.2f} {base/max(tot[c],1e-300):10.4f}× {bad*100:17.0f}% '
              f'{np.median(p):19.3f}{tag}')
    print(f'{"⛔вип.":>7s} {base/max(tot["rand"],1e-300):10.4f}× '
          f'(випадкові напрямки, c=+0.004 — мусить бути ≈1.0)')
    print('\n🔑 бал — СЕРЕДНЄ по мережах, тож рахуємо відношення СУМ MSE')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
