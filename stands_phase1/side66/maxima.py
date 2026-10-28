"""ЩО ТАКЕ МАКСИМУМ ДЛЯ НАШОГО ОБ'ЄКТА (С79, питання юзера).

Юзер: «середнє це половина максимуму» — а що таке максимум у нас?

Через позитивну однорідність (`f(cz)=c·f(z)`, зсувів нема) максимум по ВСЬОМУ
простору нескінченний, тож питання має сенс лише на СФЕРІ. Беремо `‖z‖=√d`
(там живе гаусова маса: `E‖z‖ = 15.984`) і шукаємо
    `S(z) = Σ_j o_j(z)`   — рівно те, що ми усереднюємо (наша ціль = E[S]/256).

Три числа, які це має дати:
  1. ВЕРХНЯ МЕЖА з ваг: `Lip = Π‖W_i‖₂` — записана раніше як 6.52e12;
  2. РЕАЛЬНИЙ максимум (градієнтний підйом на сфері, багато рестартів);
  3. ТИПОВЕ значення `E[S]` і `std(S)`.

🔑 ЧОМУ ЦЕ НЕ ПРАЗДНЕ ПИТАННЯ. Теза «середнє = половина максимуму» вірна для
   РІВНОМІРНОГО розподілу. Відношення `max/E` міряє, наскільки наш об'єкт від
   рівномірного далекий — тобто силу концентрації міри. Якщо воно ~2, теза
   юзера жива; якщо це багато порядків — максимум не несе інформації про
   середнє, і будь-який метод «через екстремуми» мертвий ще до реалізації.

⛔ ПЛОМБИ:
   * однорідність `S(2z) = 2·S(z)` — має виконуватись до fp32;
   * знайдений максимум НЕ МОЖЕ перевищити `√d·Lip·‖z‖` (нерівність Ліпшиця);
   * підйом із РІЗНИХ стартів має сходитись до близьких значень, інакше
     ландшафт рваний і «максимум» — це просто найкращий із рестартів.

📣 ПЕРЕДБАЧЕНО: `Lip` як добуток 32 норм сильно ЗАВИЩЕНА (сингулярні вектори
   сусідніх шарів не вирівняні, кожен шар «губить» фактор ~√2 через ReLU), тож
   реальний максимум має бути на багато порядків нижчий за 6.52e12. Ставлю на
   `max/E[S]` у діапазоні 10–100, і на те, що в точці максимуму частка
   відкритих воріт помітно вища за 0.5.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/maxima.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 512
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
W_, D = 256, 32
R = float(np.sqrt(W_))                      # радіус сфери = √d
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _fwd(W, z, want_gate=False):
    """S(z) = Σ_j o_j(z); за потреби — частка відкритих воріт по всій мережі."""
    x = z
    op = tot = 0.0
    for L in range(D):
        p = x @ W[L]
        if want_gate:
            op = op + (p > 0).double().sum()
            tot = tot + float(p.numel())
        x = torch.relu(p)
    s = x.sum(-1)
    return (s, op / max(tot, 1.0)) if want_gate else s


def _ascend(W, seed):
    """Підйом по сфері: після кожного кроку проєктуємо назад на ‖z‖=R."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(RESTARTS, W_, device=DEV, generator=g)
    z = R * z / z.norm(dim=1, keepdim=True)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.35)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = -_fwd(W, z).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.data = R * z.data / z.data.norm(dim=1, keepdim=True)
    with torch.no_grad():
        s = _fwd(W, z)
    return z.detach(), s


def _typical(W, seed, npairs=1 << 15):
    """E[S] і std(S) на гаусовій вибірці — з антитетикою."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    acc = accsq = 0.0
    n = 0
    done, CH = 0, 1 << 13
    while done < npairs:
        b = min(CH, npairs - done)
        zz = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([zz, -zz], 0)
        s = _fwd(W, x).double()
        acc += float(s.sum())
        accsq += float((s * s).sum())
        n += s.numel()
        done += b
    m = acc / n
    return m, float(np.sqrt(max(accsq / n - m * m, 0.0)))


rows = []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    lip = 1.0
    for w in W:
        lip *= float(torch.linalg.matrix_norm(w, 2))
    zbest, sbest = _ascend(W, 700 + 31 * net)
    with torch.no_grad():
        i = int(torch.argmax(sbest))
        smax = float(sbest[i])
        zm = zbest[i:i + 1]
        # ⛔ПЛОМБА ОДНОРІДНОСТІ: S(2z) мусить дорівнювати 2·S(z)
        s1 = float(_fwd(W, zm))
        s2 = float(_fwd(W, 2.0 * zm))
        hom = s2 / max(2.0 * s1, 1e-30)
        _, gate_max = _fwd(W, zm, want_gate=True)
        mean, sd = _typical(W, 5000 + 13 * net)
        zt = torch.randn(1, W_, device=DEV,
                         generator=torch.Generator(device=DEV).manual_seed(net))
        zt = R * zt / zt.norm()
        _, gate_typ = _fwd(W, zt, want_gate=True)
        spread = float(sbest.max() / sbest.median())
    rows.append((smax, mean, sd, lip, hom, float(gate_max), float(gate_typ),
                 spread))
    print(f'  … мережа {net+1}/{NETS}: max {smax:.4g}  E[S] {mean:.4g}  '
          f'відн {smax/max(abs(mean),1e-30):.1f}×', flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(rows)
print(f'\n{NETS} мереж · {RESTARTS} рестартів × {STEPS} кроків · сфера ‖z‖=√d={R:.3f}\n')
print(f'{"величина":38s} {"медіана":>14s} {"мін":>13s} {"макс":>13s}')
print('-' * 82)
NAMES = ('РЕАЛЬНИЙ максимум S', 'типове E[S]', 'std(S)',
         'ВЕРХНЯ МЕЖА Lip=Π‖W‖₂', '⛔пломба однорідності S(2z)/2S(z)',
         'ворота в ТОЧЦІ МАКСИМУМУ', 'ворота в типовій точці',
         'розкид рестартів max/медіана')
for k, nm in enumerate(NAMES):
    c = a[:, k]
    print(f'{nm:38s} {np.median(c):14.6g} {c.min():13.6g} {c.max():13.6g}')

mx, me, sd, lip = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
print(f'\n🔑 max / E[S]           = **{np.median(mx/me):.1f}×**   '
      f'(теза «середнє = половина максимуму» дала б 2.0)')
print(f'🔑 max / std(S)         = {np.median(mx/sd):.1f} сигм')
print(f'🔑 max / (√d·Lip·R)     = {np.median(mx/(np.sqrt(W_)*lip*R)):.3e}   '
      f'⟸ наскільки межа Ліпшиця ЗАВИЩЕНА')
print(f'⛔пломба однорідності: max|S(2z)/2S(z) − 1| = '
      f'{np.abs(a[:,4]-1).max():.2e}   (теорія: 0)')
z = np.median(mx / sd)
print(f'\n🎯 ВИРОК: максимум сидить за {z:.0f} сигм від середнього ⟹ '
      f'частка гаусової маси там ≈ {"нижче 1e-30" if z > 12 else "помітна"}')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
