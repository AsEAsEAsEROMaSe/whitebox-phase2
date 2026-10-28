"""ВЛУЧАЄ ЧИ ЙДЕ МИМО: геометрія ходи біжучого середнього в ПОВНИХ 256 вимірах (С80).

Юзер: «тобто рухається умовна крива до вірної середньої і кудись потрапляє або
іде мимо». Картинка правильна для ОДНОГО виміру. Тут перевіряємо, що з нею
робить розмірність 256.

⚠️ `walk.npz` (С80) зберігав лише ДВОВИМІРНУ проєкцію для малюнка. Тут беремо
повний вектор похибки `m_N − істина` і міряємо його НАПРЯМОК, а не лише довжину.

## ЩО МІРЯЄМО (на сітці `N = 1,2,4,…,16384`)
 1. **чи монотонно** — скільки подвоєнь `N` РОБЛЯТЬ ГІРШЕ (‖err‖ зросла);
 2. **чи стабільний напрямок промаху** — `cos(err(N), err(K))`.
    📣 ТЕОРІЯ для вкладених незалежних семплів: `cos = √(N/K)`.
    Якщо збігається — напрямок весь час ПЕРЕВЕРТАЄТЬСЯ, «одного боку» нема;
 3. **чи промах радіальний** — `cos(err, істина)`. Якщо ≈1, промах = один
    скаляр (масштаб) і його можна виправити; якщо ≈`1/√256`=0.0625 — промах
    у випадковому напрямку й скалярної ручки нема;
 4. **скільки координат «влучили»** — змінили знак похибки між `N` і `2N`
    (у скількох із 256 одночасно — ключове число);
 5. ⛔ **чи є хоч один `N`, де влучили ТОЧНО** (усі 256 координат разом).

⛔ ПЛОМБА: `err(K)` мусить дорівнювати `mean(усі K) − істина` з точністю fp;
✅ КОНТРОЛЬ: `cos` НЕЗАЛЕЖНИХ похибок мусить дати ≈0 (немає спільного зсуву).

🚨 ВИПРАВЛЕНО ПІСЛЯ ПЕРШОГО ПРОГОНУ. Перша версія брала ОДНУ пару промахів на
   мережу й медіану по 24 — це дало −0.1029 і фальшиву тривогу. Розподіл `cos`
   виявився широким (промах майже одновимірний), тож одна пара на мережу —
   надто слабкий прилад. Тепер `RC` незалежних промахів на мережу, СЕРЕДНЄ по
   всіх парах, і поруч друкується власна похибка контролю ⟹ вирок «0 чи не 0»
   стає можливим. 🎓 УРОК: контроль сам потребує оцінки своєї точності.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/hitmiss.py [cuda:N] [NETS] [K]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
RC = 32                              # незалежних хід у контролі
TRUTH = 1 << 21                      # 2М пар на істину (антитетика)
W_, D = 256, 32
CH = 8192
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ЩО')[0], flush=True)


def _fwd(W, z):
    x = z
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x


def _truth(W, gen, npair):
    """Істина: антитетичні пари (±z), fp64-акумулятор."""
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.randn(k, W_, device=DEV, generator=gen)
        acc += _fwd(W, torch.cat([z, -z], 0)).double().sum(0)
        done += k
    return acc / (2.0 * done)


NS = [1 << j for j in range(int(np.log2(K)) + 1)]
mono, cosK, cosT, flips, exact, seal, ctrl, rel = ([] for _ in range(8))

for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV)
         for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(4200 + 31 * net)
        T = _truth(W, g, TRUTH)
        nrm = float(T.norm())

        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        errs, done = [], 0
        while done < K:                       # хода семпл за семплом
            k = min(CH, K - done)
            o = _fwd(W, torch.randn(k, W_, device=DEV, generator=g)).double()
            cs = acc + torch.cumsum(o, 0)
            for n in NS:
                if done < n <= done + k:
                    errs.append(cs[n - 1 - done] / n - T)
            acc = cs[-1]
            done += k
        E = torch.stack(errs)                 # (len(NS), 256)
        L = E.norm(dim=1) / nrm

        seal.append(float((acc / K - T - E[-1]).norm() / (nrm + 1e-30)))
        mono.append(float((L[1:] > L[:-1]).double().mean()))
        eK = E[-1] / E[-1].norm()
        cosK.append((E @ eK / E.norm(dim=1)).cpu().numpy())
        cosT.append((E @ (T / nrm) / E.norm(dim=1)).cpu().numpy())
        sg = torch.sign(E)
        flips.append((sg[1:] != sg[:-1]).double().mean(1).cpu().numpy())
        exact.append(int((E.abs() < 1e-12).all(1).sum()))

        Z = []                          # ✅контроль: RC НЕЗАЛЕЖНИХ хід, усі пари
        for _ in range(RC):
            acc2, done = torch.zeros(W_, dtype=torch.float64, device=DEV), 0
            while done < K:
                k = min(CH, K - done)
                acc2 += _fwd(W, torch.randn(k, W_, device=DEV,
                                            generator=g)).double().sum(0)
                done += k
            Z.append(acc2 / K - T)
        Z = torch.stack(Z)
        Z = Z / Z.norm(dim=1, keepdim=True)
        iu = torch.triu_indices(RC, RC, 1)
        ctrl.append(float((Z @ Z.T)[iu[0], iu[1]].mean()))
        rel.append(float(L[-1]))
    del W
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

cosK = np.stack(cosK); cosT = np.stack(cosT)
seK = cosK.std(0, ddof=1) / np.sqrt(cosK.shape[0])   # власна похибка стовпця
seT = cosT.std(0, ddof=1) / np.sqrt(cosT.shape[0])
cosK = cosK.mean(0); cosT = cosT.mean(0)
flips = np.median(np.stack(flips), 0)

print(f'\n{NETS} мереж, хода до K={K}, істина {2*TRUTH} семплів\n')
print('⛔ПЛОМБА (хода vs пряме середнє)  ', f'{np.max(seal):.2e}  (0 = ок)')
cv = np.array(ctrl)
se = cv.std(ddof=1) / np.sqrt(len(cv))
print(f'✅КОНТРОЛЬ cos(незалежні промахи)   {cv.mean():+.4f} ± {se:.4f} '
      f'(0 = ок; {RC} хід/мережу, усі пари)')
print(f'\n{"N":>7s} {"cos(err_N, err_K)":>20s} {"√(N/K)":>8s} '
      f'{"cos(err, істина)":>20s} {"знак змінили %":>15s}')
print('-' * 86)
for i, n in enumerate(NS):
    f = f'{100*flips[i-1]:14.1f}' if i else f'{"—":>14s}'
    print(f'{n:7d} {cosK[i]:+13.4f}±{seK[i]:.4f} {np.sqrt(n/K):8.4f} '
          f'{cosT[i]:+13.4f}±{seT[i]:.4f} {f}')

print(f'\n🔑 подвоєнь, що зробили ГІРШЕ: {100*np.median(mono):.1f}%')
print(f'🔑 випадковий напрямок у 256 вимірах дав би |cos| = {1/np.sqrt(W_):.4f}')
print(f'⛔ разів, коли влучили ТОЧНО (усі 256 координат): {sum(exact)}')
print(f'📊 відносна похибка на K={K}: медіана {100*np.median(rel):.3f}%')
