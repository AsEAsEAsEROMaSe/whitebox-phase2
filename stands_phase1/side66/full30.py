"""ПОВНА РЕКУРСІЯ 30 КРОКІВ: K=2 проти K=3 — ТАМ, ДЕ СИГНАЛ. (С73)

🛑 ВИПРАВЛЕННЯ ПОМИЛКИ ВИМІРУ. `mehler3.py` міряв ефект третього члена Прайса
на ОДНОМУ кроці й дістав 1.00× — але шум істини 0.2% (доведено `gaussgate.py`:
на шарі 0 `p` ІДЕАЛЬНО гаусів, скіс 0.0000, а розбіжність двох вибірок усе
одно 0.2146%), тоді як внесок одного кроку ~0.03%. **Вимір був поставлений
рівно там, де прилад сліпий.**

✅ ПРАВИЛЬНЕ МІСЦЕ: НАКОПИЧЕНА похибка на шарі 30 = **0.873%**, тобто вчетверо
вище шуму. Якщо третій член дає хоч щось, за 30 кроків із переносом 93.9% це
там видно. Розклад по кроках для рішення НЕ ПОТРІБЕН.

ЩО РАХУЄМО: рекурсію від штатного старту (`m=0`, `C=I`) до кінця, БЕЗ
перезавантажень, трьома варіантами — і порівнюємо `m_ℓ` з істиною.

🚨 ПЛОМБИ
  ✅ K=2 мусить відтворити відому криву 0.276% (шар 1) → 0.873% (шар 30);
  ⛔ КОНТРОЛЬ ЗНАКУ: `k3=−1` мусить дати ГІРШЕ, ніж `k3=+1`, інакше стенд
     не розсуджує і будь-яке «покращення» — випадковість;
  ⛔ ⚠️ЗАВИЩЕНА ВАГА `k3=+3`: якщо вона краща за `k3=+1`, то ми не член
     додаємо, а випадково підганяємо масштаб.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi          # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
TR_PAIRS, CHUNK = 131_072, 16384
SHOW = (1, 2, 4, 8, 12, 16, 20, 24, 28, 30)
VAR = [('K=2 (сабміт)', 0.0), ('K=3 (+1)', 1.0),
       ('⛔K=3 знак −', -1.0), ('⛔K=3 вага ×3', 3.0)]


def truth_m(wc, dev, seed):
    d = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(d):
            a = torch.relu(a @ wc[i])
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * TR_PAIRS) for t in acc]


def closure_full(wc, k3=0.0):
    """ПОВНА рекурсія від штатного старту. `k3` — вага третього члена Прайса."""
    d = len(wc)
    dev = wc[0].device
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    out = []
    for i in range(d):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        if k3 != 0.0:
            u3 = t * ph / (s * s)                    # t·φ(t)/s²
            C = C + (k3 / 6.0) * (cp * cp * cp) * torch.outer(u3, u3)
        dg = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dg - C.diagonal())
        out.append(m.clone())
    return out


def main(nets=4, dev=DEV):
    torch.manual_seed(313)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    acc = {nm: {l: [] for l in SHOW} for nm, _ in VAR}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            mt = truth_m(wc, dev, 970_000 + 7919 * n)
            for nm, k3 in VAR:
                mc = closure_full(wc, k3=k3)
                for l in SHOW:
                    if l < len(wc):
                        e = 100.0 * float(((mc[l] - mt[l]) ** 2).mean().sqrt()
                                          / (mt[l] ** 2).mean().sqrt())
                        acc[nm][l].append(e)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж, ПОВНА рекурсія від штатного старту, істина '
          f'{2*TR_PAIRS} семплів\nпохибка `m` у % (медіана по мережах)\n')
    hdr = f'{"шар":>4s}' + ''.join(f'{nm:>16s}' for nm, _ in VAR) + f'{"K2/K3":>9s}'
    print(hdr)
    print('-' * len(hdr))
    for l in SHOW:
        if not acc[VAR[0][0]][l]:
            continue
        v = [np.median(acc[nm][l]) for nm, _ in VAR]
        print(f'{l:4d}' + ''.join(f'{q:15.4f}%' for q in v)
              + f'{v[0]/max(v[1],1e-12):8.3f}×')
    print('-' * len(hdr))
    print('✅ПЛОМБА: K=2 на шарі 1 ≈0.276%, на шарі 30 ≈0.873% (відома крива).')
    print('⛔якщо «знак −» ≈ «+1» ⟹ член НЕ впливає, стенд не розсуджує.')
    print('⛔якщо «вага ×3» краща за «+1» ⟹ це підгін масштабу, а не член ряду.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
