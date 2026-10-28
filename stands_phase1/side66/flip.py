"""ПЕРЕВЕРНУТА МЕРЕЖА З ТИХ САМИХ ВАГ — ЧИ ВОНА ЩОСЬ КАЖЕ ПРО НАШУ? (С80)

Юзер: «а якщо ми перевернем оці ж самі ваги моделі?»
Перевернута = `R(x) = relu(…relu(x·W₃₁ᵀ)…·W₀ᵀ)` — інша функція з ТИХ САМИХ ваг.

## ⛔ ЩО ВІДПАДАЄ ОДРАЗУ (не міряємо)
* **відновити семпл** — `side66/invcycle.py` щойно довів конструктивно, що
  інформації нема; транспонування її не створює;
* **вільна КЗ** — істина перевернутої мережі нам теж невідома;
* **подвоїти мережі для фіту `λ`** — `λ` і так фітуються на 2454 мережах при
  ~35 параметрах, це не дефіцит даних.

## 🎯 ЩО ЛИШАЄТЬСЯ І ЧОМУ ЦЕ ЖИВЕ
`side66/hitmiss2.py` (сьогодні): **40.9% енергії промаху — це МАСШТАБ**.
Важіль по масштабу закрито [[finding-error-is-one-scalar-plus-covariance]] §3
з причини «аналітика знає масштаб УТРИЧІ гірше за жереб» (4.17e-3 проти
1.34e-3). 🔑 Але то була ОДНА аналітика — наше замикання. Перевернута мережа
дає ДРУГЕ, незалежне джерело з тих самих ваг.
🔑 Питання: **чи корелює похибка масштабу замикання ВПЕРЕД із похибкою НАЗАД?**
Якщо так, перевернута мережа — новий передвісник для ручки «на мережу»,
оракул якої **1.089×** і якого ЖОДЕН передвісник із ваг не досяг
([[finding-per-network-scale-knob-is-worth-under-1.09x]], `R²=−0.81`).

## ЩО МІРЯЄМО (24 мережі)
Для ВПЕРЕД і НАЗАД окремо:
  * істинний масштаб `‖E[f]‖` великим МК;
  * масштаб із замикання `‖m_замик‖` останнього шару;
  * відносна похибка масштабу `e = ‖m_замик‖/‖E[f]‖ − 1`.
Далі `corr(e_вперед, e_назад)` по мережах.

⛔ ПЛОМБИ:
 * ⛔НЕГ: `e_назад` від ЧУЖОЇ мережі ⟹ кореляція мусить зникнути;
 * ✅перевернута мережа мусить мати ТІ САМІ сингулярні числа добутку
   (математична тотожність `(W₀…W₃₁)ᵀ = W₃₁ᵀ…W₀ᵀ`) — друкуємо як перевірку
   того, що я справді перевернув, а не помилився з порядком.

📣 ПЕРЕДБАЧЕНО: кореляція БУДЕ помітною (обидві похибки породжує та сама
   негаусовість тих самих ваг), але НЕ вистачить: ручка коштує ≤1.089×, а
   передвісник ніколи не буває ідеальним. Ставлю на `|corr| = 0.3…0.6` ⟹
   реальна вартість <1.03×. Якщо `|corr| > 0.9` — це нова жива гілка.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/flip.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import closure_parts                              # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W_, D, KS = 256, 32, 31
TRUTH, CH = 1 << 19, 8192
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ЩО МІРЯЄМО')[0], flush=True)


def _truth_scale(W, gen, npair=TRUTH):
    acc = torch.zeros(W[-1].shape[1], dtype=torch.float64, device=DEV)
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.randn(k, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for w in W:
            x = torch.relu(x @ w)
        acc += x.double().sum(0)
        done += k
    return acc / (2.0 * done)


def _spec(W):
    """log-сингулярні числа добутку (QR-розгортка) — для ✅пломби."""
    Q = torch.eye(W_, dtype=torch.float64, device=DEV)
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    for M in W:
        Q, R = torch.linalg.qr(M.transpose(0, 1).double() @ Q)
        acc = acc + R.diagonal().abs().clamp_min(1e-300).log()
    return torch.sort(acc).values


EF, ER, SEAL = [], [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV)
         for q in range(D)]
    Wr = [W[D - 1 - q].transpose(0, 1).contiguous() for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(8800 + 41 * net)
        SEAL.append(float((_spec(W) - _spec(Wr)).abs().max()))
        for Ws, box in ((W, EF), (Wr, ER)):
            T = _truth_scale(Ws, g)
            _, m_cl, _, _ = closure_parts(Ws, KS, DEV, extras=True)
            box.append(float(m_cl[KS - 1].norm() / T.norm()) - 1.0)
    del W, Wr
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

EF, ER = np.array(EF), np.array(ER)
neg = np.roll(ER, 1)                                   # ⛔НЕГ: чужа мережа
print(f'\n{NETS} мереж, істина {2*TRUTH} семплів\n')
print(f'✅ПЛОМБА (спектри вперед і назад однакові): '
      f'max|Δ| = {max(SEAL):.2e}  (0 = ок)\n')
print(f'{"величина":38s} {"середнє":>10s} {"ст.в.":>9s}')
print('-' * 60)
print(f'{"похибка масштабу ВПЕРЕД":38s} {EF.mean():10.4f} {EF.std(ddof=1):9.4f}')
print(f'{"похибка масштабу НАЗАД":38s} {ER.mean():10.4f} {ER.std(ddof=1):9.4f}')
c = float(np.corrcoef(EF, ER)[0, 1])
cn = float(np.corrcoef(EF, neg)[0, 1])
print(f'\n🔑 corr(вперед, назад)      = {c:+.4f}')
print(f'⛔НЕГ corr(вперед, ЧУЖА назад) = {cn:+.4f}   (мусить бути ≈0)')
print(f'🎯 частка дисперсії, яку пояснює перевернута: R² = {c*c:.4f}')
print(f'📐 стеля ручки «на мережу» (С68): 1.089× ⟹ при R²={c*c:.3f} '
      f'реально ≈ {1 + 0.089*c*c:.4f}×')
print('📊 БАЛ 1.954e-07')
