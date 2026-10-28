"""ЧИ ПЕРЕНОСИТЬСЯ `κ₃` ДЕШЕВО: діагональне наближення проти істини (С78).

`field.py` показав, що маргінальні кумулянти — жива гілка: гаус 1.054%, +κ₃
0.138%, +κ₄ 0.039%. Але там κ брались з ОРАКУЛА. Щоб це стало методом, κ₃ треба
нести крізь шари самим ланцюгом.

Точно:  κ₃(p_j) = Σ_{ikl} W_ij W_kj W_lj · κ₃(x_i, x_k, x_l)
        — СПІЛЬНИЙ кумулянт, тензор 256³ ⟹ 256⁴ = 4.3e9 флопів на шар,
          × 32 шари = 1.4e11 при бюджеті 2.7e10 ⟹ у 5× понад ⛔

Дешева гіпотеза (ДІАГОНАЛЬ): лишити тільки i=k=l ⟹
        κ₃(p_j) ≈ Σ_i W_ij³ · κ₃(x_i)
        — один поелементний матмул на шар, ~0.13% бюджету ✅

🔑 ПИТАННЯ ЦЬОГО СТЕНДА: наскільки діагональ відтворює істину при ТОЧНИХ
   κ₃(x_i) попереднього шару. Це вимір ОДНОГО кроку — якщо він промахує вже
   тут, рекурсія на 32 шари безнадійна, і міряти далі нема сенсу.

📣 ПЕРЕДБАЧЕНО: активації `x_i` СИЛЬНО корельовані (усі народжені з того самого
   `p`), тож перехресні члени `i≠k` мають домінувати над діагоналлю. Ставлю на
   те, що діагональ дасть кореляцію з істиною ≲0.5 і промах у рази. Якщо так —
   гілка «нести κ₃ самим ланцюгом» мертва, і живою лишається тільки та, де κ₃
   береться з ВИБІРКИ (вона вже оплачена: одна редукція на шар).
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
W_ = 256
D = 32
CH = 1 << 14

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _mom(W, seed, npairs):
    """κ₃ активацій `x_ℓ` і κ₃ передактивацій `p_ℓ` — обидва з MC."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Sx = [torch.zeros(3, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    Sp = [torch.zeros(3, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            pd = p.double()
            Sp[L][0] += pd.sum(0); Sp[L][1] += (pd * pd).sum(0)
            Sp[L][2] += (pd * pd * pd).sum(0)
            x = torch.relu(p)
            xd = x.double()
            Sx[L][0] += xd.sum(0); Sx[L][1] += (xd * xd).sum(0)
            Sx[L][2] += (xd * xd * xd).sum(0)
        done += b
    N = float(2 * npairs)

    def k3(S):
        a1, a2, a3 = S[0] / N, S[1] / N, S[2] / N
        return a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3      # μ₃ = κ₃
    return [k3(s) for s in Sx], [k3(s) for s in Sp]


rows = []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        kx, kp = _mom(W, 77_000 + 19 * n, 1 << LOGN)
        r = []
        for L in range(1, D):
            # ДІАГОНАЛЬ: κ₃(p_j) ≈ Σ_i W_ij³ κ₃(x_i)
            w3 = (W[L].double()) ** 3
            approx = kx[L - 1] @ w3
            tr = kp[L]
            cs = float((approx * tr).sum() /
                       (approx.norm() * tr.norm() + 1e-300))
            rat = float(approx.norm() / (tr.norm() + 1e-300))
            r.append((cs, rat))
        rows.append(r)
    del W
    torch.cuda.empty_cache()
    a = np.array(r)
    print(f'  … мережа {n+1}/{NETS}  cos={a[:,0].mean():+.3f}  '
          f'‖набл‖/‖іст‖={a[:,1].mean():.3f}', flush=True)

R = np.array(rows).mean(0)
print(f'\n{NETS} мереж × {2*(1<<LOGN)} семплів\n')
print(f'{"шар":>4s} {"cos(діаг, істина)":>19s} {"‖діаг‖/‖іст‖":>14s}')
print('-' * 40)
for i, L in enumerate(range(1, D)):
    star = ' <<<' if R[i, 0] > 0.9 else ''
    print(f'{L:4d} {R[i,0]:19.4f} {R[i,1]:14.4f}{star}')
cs = R[:, 0].mean()
print(f'\n{"":=^62s}')
print(f'СЕРЕДНЄ cos(діагональ, істина) = **{cs:+.4f}**   '
      f'масштаб {R[:,1].mean():.3f}')
ok = cs > 0.9
print(f'🔑 ВИРОК: діагональ {"🟢 ВІДТВОРЮЄ" if ok else "⛔ НЕ ВІДТВОРЮЄ"} κ₃ '
      f'(гейт cos>0.9) ⟹ нести κ₃ ланцюгом '
      f'{"МОЖНА" if ok else "НЕ МОЖНА — лишається κ₃ З ВИБІРКИ"}')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
