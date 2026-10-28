"""СТЕЛЯ **НЕЛОКАЛЬНИХ** ОЗНАК ДЛЯ `b₃₀` — гейт для всього класу «нейронка». (С73)

ЗВІДКИ. `bceil.py` зміряв НЕПАРАМЕТРИЧНО стелю ЛОКАЛЬНИХ ознак (властивостей
одного нейрона: `t`, `s`): **0.31**, і лінійний фіт `b30.py` дав 0.316 — тобто
ми ВЖЕ НА СТЕЛІ, і справа не в моделі, а в ознаках. ⟹ нейронка на тих самих
ознаках не додасть НІЧОГО за визначенням.

🎯 ПИТАННЯ ЦЬОГО СТЕНДА: чи є інформація в ознаках, що описують нейрон
ВІДНОСНО РЕШТИ ШАРУ (спектр `C`, корелятори, норми) — тобто там, куди
локальний опис не дивиться.

ЩО РАХУЄМО — та сама методика, щоб числа були ПОРІВНЯННІ з 0.31:
умовне середнє по комірках, LOO ПО МЕРЕЖАХ (інакше комірка підглядає у власну
відповідь). Це стеля БУДЬ-ЯКОЇ моделі на цих ознаках (умовне сподівання —
найкращий передбачувач у L²), нейронки зокрема.

🚨 ОЗНАКИ БЕРУТЬСЯ ОРАКУЛЬНО ДЕШЕВІ, АЛЕ НЕ БЕЗКОШТОВНІ: усі вони — з `C₃₀`
замикання, тобто вимагають рекурсії до шару 30 (у сабміті `KSIG=8`). Це
СВІДОМО: спершу гейт «чи є інформація ВЗАГАЛІ», і лише якщо є — питання ціни.

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: якщо ваги iid-гаусові (а вони такі за побудовою),
то нейрон не має «індивідуальності» поза своїми `m`,`s` ⟹ ставлю, що приріст
понад 0.31 буде **<0.05**.
ПОРІГ: приріст <0.10 ⟹ клас «нейронка» закрито остаточно; >0.30 ⟹ з'являється
ціль, для якої нейронка — правильний інструмент.

КОНТРОЛІ
  ⛔одна комірка (=константа) — мусить дати ≈0;
  ⛔ЗСУВ мереж (ознаки однієї, ціль іншої) — мусить впасти до ≈0;
  ✅відтворення `лише t` ≈0.31 — пломба, що методика та сама.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi           # noqa: E402

L_, KS = 32, 30
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
TR_PAIRS, CHUNK = 65_536, 16384


def truth_m(wc, dev, seed, upto):
    acc = torch.zeros(wc[upto].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(upto + 1):
            a = torch.relu(a @ wc[i])
        acc += a.double().sum(0)
        done += k
    return acc / float(2 * TR_PAIRS)


def closure_to(wc, upto):
    """Рекурсія замикання; повертає `m`,`s`,`C` шару `upto` (0-based)."""
    dev = wc[0].device
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    for i in range(upto + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        dg = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dg - C.diagonal())
    return m, C.diagonal().clamp_min(1e-20).sqrt(), C, t


def cell_r2(key, y, net, nets, nb):
    """`R²` умовного середнього по комірках, LOO ПО МЕРЕЖАХ (як у `bceil.py`)."""
    tot = torch.zeros(nb, dtype=torch.float64, device=y.device)
    cnt = torch.zeros(nb, dtype=torch.float64, device=y.device)
    tot.index_add_(0, key, y)
    cnt.index_add_(0, key, torch.ones_like(y))
    ss, st = 0.0, 0.0
    for n in range(nets):
        msk = net == n
        to = torch.zeros(nb, dtype=torch.float64, device=y.device)
        cn = torch.zeros(nb, dtype=torch.float64, device=y.device)
        to.index_add_(0, key[msk], y[msk])
        cn.index_add_(0, key[msk], torch.ones_like(y[msk]))
        mu = (tot - to) / (cnt - cn).clamp_min(1.0)
        ss += float(((y[msk] - mu[key[msk]]) ** 2).sum())
        st += float(((y[msk] - y.mean()) ** 2).sum())
    return 1.0 - ss / max(st, 1e-300)


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    F = {k: [] for k in ('t', 's', 'crow', 'csq', 'proj', 'wn', 'skew3')}
    Y_, N_ = [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            k = min(KS - 1, len(wc) - 2)
            m_cl, s_cl, C, t = closure_to(wc, k)
            mt = truth_m(wc, dev, 880_000 + 7919 * n, k)
            b = m_cl - mt
            R = C / torch.outer(s_cl, s_cl)          # кореляційна матриця
            R = R - torch.diag(R.diagonal())
            ev, U = torch.linalg.eigh(R)
            F['t'].append(t if t.numel() == b.numel() else m_cl / s_cl)
            F['s'].append(s_cl / s_cl.mean())
            F['crow'].append(R.abs().mean(1))                      # нелокальна
            F['csq'].append((R * R).mean(1))                       # нелокальна
            F['proj'].append(U[:, -1].abs())                       # нелокальна
            F['wn'].append(wc[k].double().norm(dim=0)
                           / wc[k].double().norm(dim=0).mean())    # нелокальна
            F['skew3'].append((R ** 3).mean(1))                    # нелокальна
            Y_.append(b / s_cl)
            N_.append(torch.full_like(b, float(n)))
        del W, wc
        torch.cuda.empty_cache()
        if (n + 1) % 6 == 0:
            print(f'  … {n+1}/{nets}', flush=True)

    y = torch.cat(Y_)
    net = torch.cat(N_).long()
    ft = {k: torch.cat(v) for k, v in F.items()}
    print(f'\nнейронів {y.numel()}, мереж {nets}; ціль `b₃₀/s₃₀`\n')

    def rb(x, nb):
        r = torch.argsort(torch.argsort(x))
        return (r.double() * nb / x.numel()).long().clamp(0, nb - 1)

    print(f'{"ознаки":>40s} {"R² (стеля)":>12s} {"приріст":>9s}')
    print('-' * 65)
    base0 = cell_r2(torch.zeros_like(net), y, net, nets, 1)
    print(f'{"⛔одна комірка (=константа)":>40s} {base0:12.4f}')
    r_t = cell_r2(rb(ft['t'], 32), y, net, nets, 32)
    print(f'{"✅лише t, 32 комірки (пломба ≈0.31)":>40s} {r_t:12.4f}')
    r_ts = cell_r2(rb(ft['t'], 16) * 16 + rb(ft['s'], 16), y, net, nets, 256)
    print(f'{"локальні t × s, 16×16":>40s} {r_ts:12.4f} '
          f'{r_ts - r_t:+9.4f}')
    print()
    best = r_t
    for k in ('crow', 'csq', 'proj', 'wn', 'skew3'):
        r = cell_r2(rb(ft['t'], 16) * 16 + rb(ft[k], 16), y, net, nets, 256)
        best = max(best, r)
        print(f'{f"t × {k} (НЕЛОКАЛЬНА), 16×16":>40s} {r:12.4f} '
              f'{r - r_t:+9.4f}')
    # найкраща трійка
    r3 = cell_r2(rb(ft['t'], 8) * 64 + rb(ft['crow'], 8) * 8 + rb(ft['proj'], 8),
                 y, net, nets, 512)
    print(f'{"t × crow × proj, 8×8×8":>40s} {r3:12.4f} {r3 - r_t:+9.4f}')
    best = max(best, r3)
    ysh = torch.cat([Y_[(n + 1) % nets] for n in range(nets)])
    if ysh.numel() == y.numel():
        rs = cell_r2(rb(ft['t'], 16) * 16 + rb(ft['crow'], 16), ysh, net,
                     nets, 256)
        print(f'\n{"⛔ЗСУВ мереж (мусить впасти до ≈0)":>40s} {rs:12.4f}')
    print('-' * 65)
    print(f'🎯 НАЙКРАЩА СТЕЛЯ {best:.4f}, приріст понад локальні {best-r_t:+.4f}')
    print('   <0.10 ⟹ клас «нейронка» ЗАКРИТО: інформації поза локальними '
          'ознаками нема.\n   >0.30 ⟹ з\'явилась ціль, для якої нейронка — '
          'правильний інструмент.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
