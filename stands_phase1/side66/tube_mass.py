"""ФОРМУЛА ТРУБКИ / КАЦ-РАЙС: наскільки близько гаусова маса до зламів?

ЗВІДКИ (ідея юзера). Гіпотеза «неберучі 10–31% — це ЗЛАМИ ReLU» має вже два
підтвердження: softplus β=4 обвалює хвіст у 3–7 разів (причинне втручання) і
пошаровий спектр показує, що хвіст росте з КОЖНИМ ReLU. Третій, незалежний
шлях — інтегральна геометрія: скільки гаусової маси лежить у тонкому шарі
біля поверхонь перемикання.

🎯 АЛЕ МІРЯЮ НЕ МАСУ, А МАСШТАБ ДОВЖИНИ — бо саме він вирішує наперед:
  * якщо типова відстань до найближчого зламу ВЕЛИКА (порядку 1), то злами
    рідкі, кожна комірка велика, і метод «інтегрувати по комірках» має шанс;
  * якщо вона МІЗЕРНА, комірки крихітні, їх астрономічно багато, і клас
    методів «працювати з особливостями» мертвий ЩЕ ДО реалізації.
Заразом це звіряється з softplus: β=4 згладжує на ширині ~1/β=0.25, і якщо
на цьому масштабі маси багато — обидва виміри кажуть одне.

ЩО РАХУЄМО. Для кожного нейрона (ℓ,j) відстань точки `z` до його поверхні
перемикання в МЕТРИЦІ ВХОДУ:
    `d_{ℓj}(z) = |p_{ℓj}(z)| / ‖∇_z p_{ℓj}(z)‖`
Градієнти беремо точно, ведучи якобіан `J_ℓ = ∂x_ℓ/∂z` крізь шари:
    `P = J_{ℓ−1} W_ℓ` (стовпцеві норми = ‖∇p_ℓ‖), далі `J_ℓ = P ⊙ маска`.
Потім `d(z) = min_{ℓ,j} d_{ℓj}(z)` — відстань до НАЙБЛИЖЧОГО зламу.

КОНТРОЛІ
  * ✅ ШАР 0 має ТОЧНУ відповідь: `∇p_{0j} = w_{0j}` ⟹ `d_{0j} = |w_{0j}·z|/‖w_{0j}‖`
    розподілена рівно як `|N(0,1)|`. Друкуємо виміряні квантилі поруч із
    теоретичними — розбіжність означає зламаний якобіан;
  * ✅ Кац-Райс на прямій: очікуване число перетинів уздовж гаусової прямої
    вже зміряно НЕЗАЛЕЖНО в `newman_1d.py` (K=1384…1945). Тут рахуємо
    середню щільність зламів і звіряємо порядок;
  * ⚠️ друкуємо частку маси в трубці для сітки ε, і окремо — по шарах.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NSAMP = 192
EPS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0]


def dists(wc, z, dev):
    """Відстані до всіх поверхонь перемикання для ОДНІЄЇ точки `z`."""
    J = torch.eye(W_, dtype=torch.float32, device=dev)
    x = z
    per_layer = []
    for i, w in enumerate(wc):
        P = J @ w                      # (256, n_ℓ) = ∇_z p_ℓ
        p = x @ w
        gn = P.norm(dim=0).clamp_min(1e-30)
        per_layer.append((p.abs() / gn).double())
        msk = (p > 0).to(P.dtype)
        J = P * msk.unsqueeze(0)
        x = p * msk
    return per_layer


def main(nets=4, dev=DEV):
    torch.manual_seed(1234)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    allmin, l0all, bylayer = [], [], {}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            g = torch.Generator(device=dev).manual_seed(9100 + n)
            for k in range(NSAMP):
                z = torch.randn(W_, device=dev, generator=g)
                dl = dists(wc, z, dev)
                allmin.append(float(torch.cat(dl).min()))
                l0all.append(dl[0].cpu().numpy())
                for i, d in enumerate(dl):
                    bylayer.setdefault(i, []).append(float(d.min()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … мережа {n+1}/{nets}', flush=True)

    l0 = np.concatenate(l0all)
    print(f'\n✅ КОНТРОЛЬ ШАРУ 0 — `d` мусить бути рівно |N(0,1)|')
    print(f'{"квантиль":>10s} {"вимір":>9s} {"теорія":>9s}')
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        from math import sqrt
        import scipy.stats as st
        print(f'{q:10.2f} {np.quantile(l0, q):9.4f} '
              f'{st.norm.ppf(0.5 + q / 2):9.4f}')

    a = np.array(allmin)
    print(f'\nВІДСТАНЬ ДО НАЙБЛИЖЧОГО ЗЛАМУ, {len(a)} точок '
          f'({nets} мереж × {NSAMP})')
    print(f'  медіана {np.median(a):.5f}   1% {np.quantile(a,0.01):.5f}   '
          f'99% {np.quantile(a,0.99):.5f}')
    print(f'\nЧАСТКА МАСИ В ТРУБЦІ ШИРИНИ ε (тобто P(d < ε)):')
    for e in EPS:
        print(f'   ε={e:<6.3f}  {float((a < e).mean())*100:6.1f}%'
              + ('   ← ширина згладжування softplus β=4' if e == 0.25 else ''))

    print(f'\nМЕДІАНА ВІДСТАНІ ПО ШАРАХ (мінімум усередині шару):')
    for i in sorted(bylayer)[:12] + sorted(bylayer)[-4:]:
        v = np.array(bylayer[i])
        print(f'   шар {i:2d}: {np.median(v):.5f}')
    print('\n🔑 читати так: якщо майже вся маса сидить у ДУЖЕ тонкій трубці — '
          'комірок астрономічно багато, і клас «інтегрувати по комірках / '
          'працювати з особливостями» мертвий ще до реалізації')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
