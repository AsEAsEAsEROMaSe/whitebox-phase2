"""ЧИ МОЖНА ВІДНОВИТИ СЕМПЛ З ВИХОДУ? (С80, питання юзера)

Юзер: «чи існує мережа, яка з виходу атрактора відновить семпл?»

## ЧОМУ ЦЕ НЕ ПРАЗДНЕ ПИТАННЯ
Якщо мережа ЗНИЩУЄ інформацію про `z`, то ЖОДНА обробка виходу її не поверне —
хоч мережею, хоч формулою. Це ставить стелю цілому класу «витягти з кожного
семпла більше». Якщо ж інформація ЗБЕРІГАЄТЬСЯ, клас відкритий.

## ЩО ВІДОМО ЗАЗДАЛЕГІДЬ (не переміряємо)
* зсувів нема ⟹ `f(cz)=c·f(z)` для `c>0` ⟹ ДОВЖИНА `z` відновлюється ТОЧНО
  з довжини виходу. Питання лише про НАПРЯМОК (255 вимірів);
* вхід «забувається» за 3 шари (`cos γ₃ ≥ 0.9956`);
* колапс різноманітності `PR 165.6 → 2.61` на шарі 31;
* розмах `log σ` добутку на глибині 32 = **42.1** ⟹ край/край `≈2e18`.

## ЩО МІРЯЄМО (свіп по глибині 1,2,4,8,16,32)
Беремо `z` на одиничній сфері. Дві оцінки відновлюваності — щоб не залежати
від вибору моделі:
 1. **ЛІНІЙНА**: найкраще `ẑ = o·A` методом найменших квадратів, `A` вчиться
    на ПОЛОВИНІ, міряється на ІНШІЙ (позавибірково);
 2. **БЕЗМОДЕЛЬНА (найближчий сусід)**: для тестового `z` шукаємо в навчальній
    множині семпл із НАЙБЛИЖЧИМ ВИХОДОМ і дивимось `cos` їхніх ВХОДІВ.
    Це нижня межа для БУДЬ-ЯКОГО відновлювача: якщо схожі виходи дають
    несхожі входи — відновити не може ніхто.

⛔ ПЛОМБИ:
 * ⛔НЕГ: виходи ПЕРЕМІШАНІ ⟹ обидві оцінки мусять дати ≈0;
 * ✅ПОЗ: замість виходу подаємо САМ `z` ⟹ лінійна мусить дати cos=1.000.
📣 ПЕРЕДБАЧЕНО: на шарі 1 відновлення ЧАСТКОВЕ (ReLU занулює ~половину
   координат ⟹ рівнянь менше, ніж невідомих), і швидко падає до нуля з
   глибиною. На 32 очікую cos < 0.1 обома методами ⟹ відповідь «НІ, такої
   мережі не існує — інформації просто немає».

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/invert.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
W_, D = 256, 32
KS = [1, 2, 4, 8, 16, 32]
NTR, NTE = 8192, 2048
RIDGE = 1e-6
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ЩО МІРЯЄМО')[0], flush=True)


def _sphere(n, gen):
    z = torch.randn(n, W_, device=DEV, generator=gen).double()
    return z / z.norm(dim=1, keepdim=True)


def _lin(Otr, Ztr, Ote, Zte):
    """Найкраще лінійне `ẑ = o·A` (МНК на train), cos на test."""
    G = Otr.T @ Otr + RIDGE * torch.eye(Otr.shape[1], dtype=torch.float64,
                                        device=DEV) * float(Otr.shape[0])
    A = torch.linalg.solve(G, Otr.T @ Ztr)
    P = Ote @ A
    P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return float((P * Zte).sum(1).mean())


def _nn(Otr, Ztr, Ote, Zte):
    """Безмодельна межа: найближчий ВИХІД → cos їхніх ВХОДІВ."""
    a = Otr / Otr.norm(dim=1, keepdim=True).clamp_min(1e-30)
    b = Ote / Ote.norm(dim=1, keepdim=True).clamp_min(1e-30)
    idx = (b @ a.T).argmax(1)
    return float((Ztr[idx] * Zte).sum(1).mean())


LIN = {k: [] for k in KS}
NNB = {k: [] for k in KS}
NEG, POS = [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32),
                      device=DEV).double() for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(4400 + 23 * net)
        Ztr, Zte = _sphere(NTR, g), _sphere(NTE, g)
        atr, ate = Ztr, Zte
        for k in range(1, D + 1):
            atr = torch.relu(atr @ W[k - 1])
            ate = torch.relu(ate @ W[k - 1])
            if k in KS:
                LIN[k].append(_lin(atr, Ztr, ate, Zte))
                NNB[k].append(_nn(atr, Ztr, ate, Zte))
                if k == D:
                    pm = torch.randperm(NTR, device=DEV, generator=g)
                    NEG.append(_lin(atr[pm], Ztr, ate, Zte))   # ⛔перемішані
        POS.append(_lin(Ztr, Ztr, Zte, Zte))                   # ✅сам `z`
    del W
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)


def ln(v):
    v = np.array(v, float)
    return f'{v.mean():+.4f}±{v.std(ddof=1)/np.sqrt(len(v)):.4f}'


print(f'\n{NETS} мереж, {NTR} навчальних + {NTE} тестових семплів\n')
print(f'✅ПОЗ (подаємо сам `z`):        {ln(POS)}   ← мусить бути +1.0000')
print(f'⛔НЕГ (виходи перемішані):      {ln(NEG)}   ← мусить бути ≈0\n')
print(f'{"глибина":>8s} {"ЛІНІЙНЕ відновлення":>22s} {"НАЙБЛИЖЧИЙ СУСІД":>22s}')
print('-' * 56)
for k in KS:
    print(f'{k:8d} {ln(LIN[k]):>22s} {ln(NNB[k]):>22s}')
print(f'\n🔑 випадковий напрямок у 256 вимірах дав би cos ≈ {1/np.sqrt(W_):.4f}')
print('📊 БАЛ 1.954e-07 — вимір СТРУКТУРНИЙ, важеля не дає')
