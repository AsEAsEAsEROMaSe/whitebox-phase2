"""ЕФЕКТИВНА ВИМІРНІСТЬ ІНТЕГРАНДА І СТЕЛЯ ЗАДАЧІ. (ідея юзера: IBC)

ЗАСТЕРЕЖЕННЯ ПРО IBC. Класичні нижні межі складності за інформацією
(Бахвалов, Трауб-Возняковський) стосуються задач, де функція відома ЛИШЕ через
n запитів. У нас ваги на руках ⟹ інформаційного бар'єра НЕМА, бар'єр
обчислювальний, і нижніх меж для нього ніхто доводити не вміє. Але число, яке
вирішує «чи лишився порядок», IBC називає правильно: ЕФЕКТИВНА ВИМІРНІСТЬ.

ЩО МІРЯЄМО. `F(z)` — вихід мережі, `z ~ N(0,I₂₅₆)`. Якщо `F` фактично залежить
від кількох лінійних комбінацій `z`, то поліном у цьому підпросторі має ТОЧНО
ВІДОМЕ гаусове середнє і працює як контрольна змінна, а квадратура/QMC стають
можливими. Ключ — АКТИВНИЙ ПІДПРОСТІР:
    C = E_z[ ∇(u·F)(z) · ∇(u·F)(z)ᵀ ]        (256×256)
Градієнт для кусково-лінійної мережі береться ТОЧНО: `g ← (g ⊙ mask_l) Wₗᵀ`.

🚨 ГОЛОВНЕ ПИТАННЯ. С62 виміряла, що якобіан у ТОЧЦІ має ранг ~4/256, але
«структура обертається разом із семплом». Тому вирішує не ранг у точці, а
ранг УСЕРЕДНЕНОГО `C`. Малий ⟹ у задачі лишається порядок. Заповнений ⟹
QMC/квадратура мертві за побудовою і 56× лідера мають іншу природу.

СТЕЛЯ. Будуємо КЗ з точно нульовим середнім у координатах `y = zU` (U
ортонормований ⟹ `y ~ N(0,I_k)`, отже `E[y_i]=0`, `E[y_iy_j−δ_ij]=0`):
    лінійна    {y_i}
    квадратична {y_i} ∪ {y_iy_j − δ_ij}
β підганяється на ПОЛОВИНІ семплів, дисперсія міряється на ДРУГІЙ.
Виграш = `Var(F) / Var(F − КЗ)` — це і є стеля «vs sampling» цього класу
методів (ми зараз 3.4×, лідер 56×).

КОНТРОЛІ
  * ⛔ ВИПАДКОВИЙ ортонормований підпростір тієї ж вимірності;
  * held-out ⟹ підгін не може роздути виграш;
  * ✅ середнє КЗ на семплах друкуємо — має бути ~0 без жодного підгону;
  * градієнт перевіряється скінченною різницею на кількох точках.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NS = 32768
CH = 8192
KS = (2, 4, 8, 16, 32)


def fwd_grad(wc, z, u):
    """F(z) і ТОЧНИЙ ∇(u·F)(z) для кусково-лінійної мережі."""
    xs, ms = [], []
    x = z
    for i in range(L_):
        p = x @ wc[i]
        m = (p > 0).to(p.dtype)
        x = p * m
        ms.append(m)
        xs.append(x)
    g = u.unsqueeze(0).expand(z.shape[0], -1)
    for i in range(L_ - 1, -1, -1):
        g = (g * ms[i]) @ wc[i].T
    return xs[-1], g


def main(nets=6, dev=DEV):
    torch.manual_seed(77)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    pr, gain = [], {k: {'акт.лін': [], 'акт.квадр': [], '⛔вип.квадр': []}
                    for k in KS}
    chk = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            nw = wc[-1].shape[1]
            u = torch.ones(nw, device=dev) / math.sqrt(nw)
            Z, Y, G = [], [], []
            for _ in range(NS // CH):
                z = torch.randn(CH, W_, device=dev)
                x, g = fwd_grad(wc, z, u)
                Z.append(z.double()); Y.append((x @ u).double()); G.append(g.double())
            Z = torch.cat(Z); Y = torch.cat(Y); G = torch.cat(G)

            # ✅ перевірка градієнта скінченною різницею
            eps = 1e-3
            d = torch.randn(4, W_, device=dev, dtype=torch.float64)
            d = d / d.norm(dim=1, keepdim=True)
            z0 = Z[:4]
            f0 = (fwd_grad(wc, z0.float(), u)[0] @ u).double()
            f1 = (fwd_grad(wc, (z0 + eps * d).float(), u)[0] @ u).double()
            chk.append(float(((f1 - f0) / eps - (G[:4] * d).sum(1)).abs().mean()
                             / (G[:4] * d).sum(1).abs().mean()))

            C = (G.T @ G) / G.shape[0]
            ev, U = torch.linalg.eigh(C)
            ev, U = ev.flip(0).clamp_min(0), U.flip(1)
            pr.append(float(ev.sum() ** 2 / (ev * ev).sum()))

            h = Z.shape[0] // 2
            v0 = float(Y[h:].var())
            for k in KS:
                for tag, B in (('акт', U[:, :k]),
                               ('вип', torch.linalg.qr(torch.randn(
                                   W_, k, dtype=torch.float64, device=dev))[0])):
                    y = Z @ B
                    lin = y
                    idx = torch.triu_indices(k, k, device=dev)
                    quad = (y[:, idx[0]] * y[:, idx[1]]
                            - (idx[0] == idx[1]).double())
                    for nm, Ф in (('лін', lin), ('квадр', torch.cat([lin, quad], 1))):
                        if tag == 'вип' and nm == 'лін':
                            continue
                        A = Ф[:h]
                        b = torch.linalg.lstsq(
                            A.T @ A + 1e-8 * torch.eye(A.shape[1], device=dev,
                                                       dtype=torch.float64),
                            A.T @ (Y[:h] - Y[:h].mean())).solution
                        r = Y[h:] - Ф[h:] @ b
                        key = ('акт.' if tag == 'акт' else '⛔вип.') + nm
                        gain[k][key].append(v0 / float(r.var()))
        print(f'  мережа {n+1}/{len(fs)}  PR(активного C)={pr[-1]:.1f}  '
              f'k=8 квадр {gain[8]["акт.квадр"][-1]:.2f}×  '
              f'✅градієнт {chk[-1]:.1e}', flush=True)

    print(f'\n{len(fs)} мереж × {NS} семплів, ✅відн. похибка градієнта '
          f'{np.mean(chk):.1e}')
    print(f'\nЕФЕКТИВНА ВИМІРНІСТЬ: PR усередненого `E[∇F∇Fᵀ]` = '
          f'**{np.mean(pr):.1f}** ± {np.std(pr):.1f} із 256')
    print(f'\n{"k":>4s} {"акт.лін":>10s} {"акт.квадр":>11s} {"⛔вип.квадр":>12s}')
    for k in KS:
        print(f'{k:4d} {np.mean(gain[k]["акт.лін"]):10.2f} '
              f'{np.mean(gain[k]["акт.квадр"]):11.2f} '
              f'{np.mean(gain[k]["⛔вип.квадр"]):12.2f}')
    print('\n(для орієнтиру: наш сабміт = 3.4× vs sampling, лідер = 56×)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
