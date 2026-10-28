"""ІЄРАРХІЧНЕ СТИСНЕННЯ κ₃: чи є в тензора «далеке поле» з низьким рангом?

ЗВІДКИ (ідея юзера, точна й доречна). Наш вирок звучав «повний тензор 256³ =
114% бюджету», і всі ТРИ спроби обійти його були ГЛОБАЛЬНИМИ: повний тензор ·
глобальний Такер (`k3_carry_gate.py`: r=32 → 10% ефекту, треба ~256) · зрізи
(`k3_slice_update.py`: нуль). Жодна не питала, чи тензор **ієрархічно**
стисливий. Хакбуш/Грінгард-Рохлін: об'єкт може бути ГЛОБАЛЬНО повноранговим і
при цьому мати низький ранг У БЛОКАХ між далекими групами індексів — і це
найпоширеніша ситуація, яка ЗЗОВНІ виглядає рівно як «низький ранг не працює».

ЩО РАХУЄМО. Розбиваємо 256 нейронів на `G` груп. Для трійки РІЗНИХ груп
`(a,b,c)` блок `T[I_a, I_b, I_c]` — це «далеке поле». Його стисливість =
чисельний ранг розгортки за модою 1 (розмір `n/G × (n/G)²`) при відносному
порозі ε. Якщо ранги далеких блоків спадають зі зростанням `G` швидше за розмір
блока — ієрархічне представлення існує, і його ціна на порядки менша за n³.

🚨 АПРІОРНЕ ЗАПЕРЕЧЕННЯ, ЯКЕ САМЕ ЦЕЙ СТЕНД І ПЕРЕВІРЯЄ. У FMM низький ранг
далекого поля береться з ГЛАДКОСТІ ядра `1/|x−y|` за координатами точок. Наші
індекси — нейрони з густими випадковими вагами: ні метрики, ні порядку, ні
гладкості. Якщо метрики нема, будь-яке розбиття не краще за випадкове — і тоді
ранг блока дорівнює його розміру. Це і є ⛔контроль.

КОНТРОЛІ
  * ✅ ПОЗ: синтетичний тензор `T_ijk = f(|i−j|)f(|j−k|)f(|i−k|)`, `f(d)=1/(1+d)`
    — ГЛАДКЕ ядро на 1-вимірній сітці, ієрархічно стисливе за побудовою. У
    ПРИРОДНОМУ порядку ранги далеких блоків мусять бути МАЛІ, у ВИПАДКОВІЙ
    перестановці — великі. Якщо стенд цієї різниці не бачить — стенд зламаний;
  * ⛔ НЕГ: те саме κ₃, але групи набрані ВИПАДКОВО. Якщо «розумне» розбиття не
    відрізняється від випадкового — ієрархії нема;
  * ✅ ПІДЛОГА ШУМУ: гаусів сурогат (та сама коваріація, κ₃ ТОТОЖНО нуль) —
    його «ранги» це чистий шум вибірки, і справжній κ₃ має бути НИЖЧЕ.

РОЗУМНЕ РОЗБИТТЯ. Алгебраїчне, як в H-матрицях без геометрії: спектральна
кластеризація за грамом рядків розгортки (`Gm = M Mᵀ`, нормований) — саме та
матриця «хто з ким взаємодіє», яку H-конструкція й використовує, коли координат
нема.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                                  # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:2'
NSAMP = 98_304
CHUNK = 8192
LAYERS = [4, 8, 16, 24]
GRID = [4, 8, 16, 32]      # G≥3, інакше трійок РІЗНИХ груп не існує
EPS = 0.05


def k3_tensor(wc, layer, dev, seed, nsamp=NSAMP):
    """Третій ЦЕНТРАЛЬНИЙ момент активацій шару `layer` (для 3-го порядку = κ₃)."""
    n = wc[layer].shape[1]
    s1 = torch.zeros(n, dtype=torch.float64, device=dev)
    s2 = torch.zeros(n, n, dtype=torch.float64, device=dev)
    rng = np.random.default_rng(int(seed))
    xs = []
    done = 0
    while done < nsamp:
        k = min(CHUNK, nsamp - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc[:layer + 1]):
            x = torch.relu(x @ w)
        xs.append(x)
        s1 += x.double().sum(0)
        done += k
    X = torch.cat(xs, 0)
    del xs
    m = (s1 / X.shape[0]).float()
    B = X - m                                    # центровані активації
    del X
    s2 = (B.T @ B).double() / B.shape[0]         # коваріація (для гаусового сурогату)
    # κ₃ по частинах за першим індексом, щоб не тримати проміжних тензорів
    T = torch.zeros(n, n, n, dtype=torch.float32, device=dev)
    for a in range(n):
        T[a] = (B * B[:, a:a + 1]).T @ B / B.shape[0]
    return T, s2, B.shape[0]


def gauss_surrogate(cov, n, ns, dev, seed):
    """✅ПІДЛОГА ШУМУ: та сама коваріація, κ₃ ТОТОЖНО нуль. Ранги = шум вибірки."""
    ev, V = torch.linalg.eigh(cov)
    A = (V * ev.clamp_min(0).sqrt()).float()
    g = torch.Generator(device=dev).manual_seed(int(seed))
    T = torch.zeros(n, n, n, dtype=torch.float32, device=dev)
    B = torch.randn(ns, n, device=dev, generator=g) @ A.T
    B = B - B.mean(0, keepdim=True)
    for a in range(n):
        T[a] = (B * B[:, a:a + 1]).T @ B / ns
    return T


def synth_hier(n, dev):
    """✅ПОЗ.КОНТРОЛЬ: гладке ядро на 1-вим. сітці — ієрархічно стисливе."""
    i = torch.arange(n, device=dev, dtype=torch.float32)
    f = lambda d: 1.0 / (1.0 + d)                                  # noqa: E731
    a = f((i[:, None] - i[None, :]).abs())
    return a[:, :, None] * a[None, :, :] * a[:, None, :]


def spectral_order(T):
    """Алгебраїчна кластеризація: спектральний порядок за грамом розгортки."""
    n = T.shape[0]
    M = T.reshape(n, -1)
    Gm = (M @ M.T).double()
    d = Gm.diagonal().clamp_min(1e-30).rsqrt()
    Gm = Gm * torch.outer(d, d)
    ev, V = torch.linalg.eigh(Gm)
    return torch.argsort(V[:, -2]).tolist()      # Фідлерів вектор нормованого грама


def block_ranks(T, order, G, eps=EPS, maxtrip=200):
    """Середній чисельний ранг ДАЛЕКИХ блоків (усі три групи різні), у частках
    від повного рангу блока."""
    n = T.shape[0]
    idx = torch.tensor(order, device=T.device)
    sz = n // G
    grp = [idx[g * sz:(g + 1) * sz] for g in range(G)]
    rel, seen = [], 0
    for a in range(G):
        for b in range(G):
            for c in range(G):
                if a == b or b == c or a == c:
                    continue
                if seen >= maxtrip:
                    break
                seen += 1
                blk = T[grp[a]][:, grp[b]][:, :, grp[c]].reshape(sz, -1)
                gm = (blk @ blk.T).double()
                ev = torch.linalg.eigvalsh(gm).clamp_min(0)
                sv = ev.flip(0).sqrt()
                r = int((sv > eps * sv[0].clamp_min(1e-30)).sum())
                rel.append(r / sz)
    if not rel:
        return float('nan'), 0
    return float(np.mean(rel)), seen


def report(name, T, dev):
    n = T.shape[0]
    orders = {'спектр': spectral_order(T),
              '⛔вип': torch.randperm(n, device=dev).tolist()}
    print(f'\n{name}   (n={n}, ε={EPS})')
    print(f'{"G":>4s} {"розмір блока":>13s} ' +
          ' '.join(f'{k:>10s}' for k in orders))
    for G in GRID:
        if n // G < 4:
            continue
        row = []
        for k, o in orders.items():
            r, _ = block_ranks(T, o, G)
            row.append(f'{r:10.3f}')
        print(f'{G:4d} {n // G:13d} ' + ' '.join(row))


def main(nets=2, dev=DEV):
    torch.manual_seed(777)
    n = 128
    print('=' * 64)
    print('✅ ПОЗ.КОНТРОЛЬ — гладке ядро на 1-вим. сітці (ІЄРАРХІЧНЕ за побудовою)')
    print('   у стовпці «спектр» ранги мусять бути МАЛІ й спадати з G,')
    print('   у «⛔вип» — великі. Нема різниці ⟹ стенд зламаний.')
    print('=' * 64)
    report('синтетика 1/(1+d)', synth_hier(n, dev), dev)

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
                T, cov, ns = k3_tensor(wc, lay, dev, 6100 + 31 * q + lay)
                print('\n' + '=' * 64)
                print(f'МЕРЕЖА {q+1}, ШАР {lay}')
                print('=' * 64)
                report(f'κ₃ шару {lay}', T, dev)
                Tg = gauss_surrogate(cov, T.shape[0], ns, dev, 9200 + lay)
                report(f'✅ПІДЛОГА ШУМУ: гаусів сурогат шару {lay} (κ₃≡0)', Tg, dev)
                del T, Tg, cov
                torch.cuda.empty_cache()
        del W, wc
        torch.cuda.empty_cache()

    print('\n🔑 читати так: «ранг блока / розмір блока» ≈ 1.0 означає, що далекого '
          'поля НЕМА — блок повнорангий, і ієрархічне представлення коштує стільки '
          'само, скільки повний тензор. Значення, що СПАДАЄ з G у стовпці «спектр» '
          'і НЕ спадає у «⛔вип», означало б, що ієрархія є.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 2)
