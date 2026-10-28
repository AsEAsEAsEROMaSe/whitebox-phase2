"""ТЕНЗОРНИЙ СКЕТЧ (Count-Sketch / TensorSketch, Пейгам-Фам): чи дає він κ_iij?

ЗВІДКИ (нитка юзера). «Скетч атакує рівно цифру 114%: якщо дія тензора коштує
`n log n`, а не `n³`, то 114% стають частками відсотка, і виміряні 5.87× стають
реальними без жодної нової фізики». Логіка бездоганна — вирок С64 справді був
не «сигналу нема», а «доставка коштує 114% бюджету».

🔑 ЩО САМЕ МІРЯЄМО І ЧОГО НЕ МІРЯЄМО. Вартість ПОБУДОВИ скетчу `O(n log n)` на
семпл — це теорема, її перевіряти нема чого. Під питанням ТОЧНІСТЬ: скетч —
випадкова незміщена оцінка з дисперсією `‖T‖_F²/D`, і з нього треба ВИТЯГТИ
131072 конкретні входи (зрізи `κ_iij`, `κ_ijj`), а не одне скалярне добуток.
Тому тут скетч будується ЕКВІВАЛЕНТНИМ прямим шляхом (розсіювання з точного
тензора — це ТОЙ САМИЙ лінійний відбиток, що дає FFT-конструкція), і міряється
рівно те, що під питанням: відносна похибка `κ_iij` як функція розміру `D`.

📐 ЯК ПЕРЕВОДИТИ ПОХИБКУ В БАЛ — крива вже зміряна в С64 (`k3_carry_gate.py`):
    відн. похибка κ_iij  63.3% → 1.77×   48.8% → 2.04×   33.2% → 2.63%   0% → 5.84×
тож стенду досить видати ВІДНОСНУ ПОХИБКУ, а цінник береться з готової кривої.

КОНТРОЛІ
  * ✅ ПОЗ: РОЗРІДЖЕНИЙ тензор (K ненулів) — скетч мусить брати його добре при
    малому `D` (саме там він і виграє). Не бере ⟹ моя реалізація зламана;
  * ✅ ЗВІРКА З ТЕОРІЄЮ: передбачена похибка на вхід = `‖T‖_F/√(D·R)`. Якщо
    виміряне збігається — можна екстраполювати на будь-яке `D` без прогону;
  * ⛔ ЧЕСНИЙ СУПЕРНИК при ТІЙ САМІЙ пам'яті: лишити `D·R` НАЙБІЛЬШИХ входів
    тензора. Якщо він не гірший — скетч не додає нічого понад «тензор плаский».
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                                  # noqa: E402
from hmat_k3 import k3_tensor                                     # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:2'
LAYERS = [4, 16]
DGRID = [4096, 16384, 65536, 262144]
RGRID = [1, 4]
NSAMP = 49_152
# крива «відн. похибка κ_iij → × зменшення похибки кроку» з С64
CURVE = [(0.633, 1.77), (0.488, 2.04), (0.332, 2.63), (0.0, 5.84)]


def value_of(err):
    """Лінійна інтерполяція зміряної кривої С64 (за межами — крайні значення)."""
    xs = [c[0] for c in CURVE][::-1]
    ys = [c[1] for c in CURVE][::-1]
    if err >= xs[-1]:
        return 1.60                       # рівень «без κ» (той самий стенд)
    return float(np.interp(err, xs, ys))


def sketch_and_recover(T, D, R, gen, ii, jj):
    """Скетч TensorSketch розміру D, R повторів; повертає оцінку зрізу κ_iij."""
    n = T.shape[0]
    est = torch.zeros(len(ii), dtype=torch.float64, device=T.device)
    for _ in range(R):
        h = [torch.randint(0, D, (n,), device=T.device, generator=gen)
             for _ in range(3)]
        s = [(torch.randint(0, 2, (n,), device=T.device, generator=gen)
              .double() * 2.0 - 1.0) for _ in range(3)]
        Y = torch.zeros(D, dtype=torch.float64, device=T.device)
        for a in range(n):                       # частинами за 1-м індексом
            idx = (h[0][a] + h[1].view(-1, 1) + h[2].view(1, -1)) % D
            val = (s[0][a] * s[1].view(-1, 1) * s[2].view(1, -1)
                   * T[a].double())
            Y.scatter_add_(0, idx.reshape(-1), val.reshape(-1))
        k = (h[0][ii] + h[1][ii] + h[2][jj]) % D
        est += s[0][ii] * s[1][ii] * s[2][jj] * Y[k]
    return est / R


def top_entries(T, budget, ii, jj):
    """⛔ЧЕСНИЙ СУПЕРНИК: лишити `budget` найбільших входів, решта = 0."""
    flat = T.reshape(-1)
    b = min(budget, flat.numel())
    v, idx = torch.topk(flat.abs().double(), b)
    keep = torch.zeros_like(flat, dtype=torch.float64)
    keep[idx] = flat[idx].double()
    n = T.shape[0]
    return keep.view(n, n, n)[ii, ii, jj]


def study(name, T, dev, seed):
    n = T.shape[0]
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    g = torch.meshgrid(torch.arange(n, device=dev), torch.arange(n, device=dev),
                       indexing='ij')
    ii, jj = g[0].reshape(-1), g[1].reshape(-1)
    exact = T[ii, ii, jj].double()
    nrm = float(exact.norm().clamp_min(1e-30))
    fro = float(T.double().norm())
    print(f'\n{name}   n={n}, ‖T‖_F={fro:.4g}, ‖κ_iij‖={nrm:.4g}, '
          f'n³={n**3:,}')
    print(f'{"D":>9s} {"R":>3s} {"пам\'ять/n³":>11s} {"відн.похибка":>13s} '
          f'{"теорія":>9s} {"⛔топ-входи":>12s} {"→ × кроку":>10s}')
    for D in DGRID:
        for R in RGRID:
            est = sketch_and_recover(T, D, R, gen, ii, jj)
            err = float((est - exact).norm()) / nrm
            th = fro * np.sqrt(len(ii) / (D * R)) / nrm
            te = top_entries(T, D * R, ii, jj)
            terr = float((te - exact).norm()) / nrm
            print(f'{D:9d} {R:3d} {D * R / n ** 3:11.4f} {err:13.4f} '
                  f'{th:9.4f} {terr:12.4f} {value_of(err):9.2f}×')


def main(nets=1, dev=DEV):
    torch.manual_seed(2024)
    n = 96
    gen = torch.Generator(device=dev).manual_seed(5)
    Ts = torch.zeros(n, n, n, device=dev)
    K = 2000
    pos = torch.randint(0, n ** 3, (K,), device=dev, generator=gen)
    Ts.view(-1)[pos] = torch.randn(K, device=dev, generator=gen)
    print('=' * 78)
    print(f'✅ ПОЗ.КОНТРОЛЬ — РОЗРІДЖЕНИЙ тензор ({K} ненулів із {n**3:,}).')
    print('   Скетч мусить брати його ДОБРЕ при малому D. Не бере ⟹ реалізація зламана.')
    print('=' * 78)
    study('розріджений', Ts, dev, 11)
    del Ts
    torch.cuda.empty_cache()

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    for q, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][r], np.float32), device=dev)
             for r in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            for lay in LAYERS:
                T, _, _ = k3_tensor(wc, lay, dev, 7700 + 31 * q + lay,
                                    nsamp=NSAMP)
                print('\n' + '=' * 78)
                study(f'κ₃ шару {lay}, мережа {q + 1}', T, dev, 900 + lay)
                del T
                torch.cuda.empty_cache()
        del W, wc
        torch.cuda.empty_cache()

    print('\n🔑 читати так: «пам\'ять/n³» — це ціна скетчу відносно повного тензора '
          '(114% бюджету). Якщо для потрібної точності треба «пам\'ять/n³» ≳ 1, '
          'скетч ДОРОЖЧИЙ за сам тензор, і 114% нікуди не діваються.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
