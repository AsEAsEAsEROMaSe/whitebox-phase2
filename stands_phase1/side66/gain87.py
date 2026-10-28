"""КОЕФІЦІЄНТ ПІДСИЛЕННЯ: ЧИ МОЖНА ВЗАГАЛІ СТАБІЛІЗУВАТИ РЕКУРСІЮ. (С87)

## ЗВІДКИ
`stepcov87.py` показав: свіжа похибка ОДНОГО кроку замикання = 1–2%, а
накопичена по ланцюгу = 14.8%. ⟹ формулу лікувати марно, питання в ДИНАМІЦІ.
`selfsig87.py` (С87) уже натякав: чутливість ланцюга до поправки `σ` ≈ **−18**,
через що ітеративне самоузгодження розлетілось 0.87 → 3.39 → 0.10 → 5e7.

## 🎯 ЩО МІРЯЄТЬСЯ — ЧИСТА ДИНАМІКА, БЕЗ ЖОДНОЇ МОДЕЛІ
На шарі `l₀` множимо `σ` на `(1+ε)` і дивимось, ЩО з цього виросте:
```
G_σ(l₀) = [Δσ₃₁/σ₃₁] / ε        підсилення в самій σ
G_out(l₀) = [Δвихід/вихід] / ε   підсилення у величині, з якої рахується бал
```
🔑 ВИРОК ЧИТАЄТЬСЯ ОДНОЗНАЧНО:
* `G < 1` ⟹ збурення ЗАБУВАЄТЬСЯ. Тоді 14.8% береться з ПОСТІЙНОГО джерела на
  кожному кроці, і локальна поправка кроку (0.2–0.4%) ВИЛІКУВАЛА Б усе.
* `G > 1` ⟹ рекурсія ПІДСИЛЮЄ. Тоді жодна локальна поправка не допоможе:
  залишок множиться, і треба або точний якір, або інші змінні.
* `G ≈ 1` ⟹ нейтральний перенос, похибки просто додаються.

## 📐 ЛІНІЙНІСТЬ — ПЛОМБА, А НЕ ПРИПУЩЕННЯ
Міряємо на `ε = ±0.1%, ±1%`: якщо `G` однакове ⟹ режим лінійний і число має
сенс; якщо різне ⟹ нелінійність, і тоді `G` треба брати при робочому `ε`.

## ⛔ПЛОМБИ
 P1 `ε = 0` мусить дати `G` невизначене/нуль і РІВНО ту саму траєкторію.
 P2 СИМЕТРІЯ: `G(+ε)` і `G(−ε)` мусять збігатися в лінійному режимі.
 P3 збурення на ОСТАННЬОМУ шарі мусить дати `G_out` ≈ відомий аналітичний
    коефіцієнт `∂E[ReLU]/∂σ · σ/E[ReLU] = φ(t)·σ/(mΦ+σφ)`, при `t≈3` це ~0.4%.
    Якщо ні — стенд міряє не те.
 P4 контроль напрямку: збурення `m` замість `σ` (інша величина, інший `G`).

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) `G_σ > 1` для мілких `l₀` (збурення підсилюється). Шанс **70%** — на це
     вказує чутливість −18 із `selfsig87`.
 (б) `G_out` спадатиме з `l₀` (пізніші збурення менше встигають вирости).
     Шанс 60%.
 (в) режим ЛІНІЙНИЙ при ε≤1% (G однакове на 0.1% і 1%). Шанс 65%.
 (г) якщо `G_σ` виявиться <1 — це НАЙКРАЩИЙ можливий результат сесії, бо тоді
     поправка 0.3% на крок закриває 13.5%. Мій шанс на це — лише 25%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/gain87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
L0S = (1, 4, 8, 12, 16, 20, 24, 28, 30)      # де вносимо збурення
EPS = (0.001, -0.001, 0.01, -0.01)
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def chain(W, kick_at=None, eps=0.0, kick_m=False):
    """Наше замикання; на шарі `kick_at` множимо `σ` (або `m`) на `(1+eps)`.

    🔑 Збурення вноситься УЗГОДЖЕНО: масштабується вся `C` на `(1+eps)²`, щоб
    `diag(C) = σ²` лишалось істинним — інакше Мелер бачить неузгоджену пару і
    ми міряли б артефакт неузгодженості, а не підсилення."""
    C = W[0].T @ W[0]
    m = torch.zeros(W[0].shape[1], dtype=torch.float64, device=W[0].device)
    s31 = None
    for i in range(1, len(W) + 1):
        if kick_at == i and eps != 0.0:
            if kick_m:
                m = m * (1.0 + eps)
            else:
                C = C * ((1.0 + eps) ** 2)
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        if i == len(W):
            s31 = s
        t = m / s
        P, f = Phi(t), phi(t)
        m_a = m * P + s * f
        if i == len(W):
            return m_a, s31
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        q2 = (m * m + s * s) * P + (m * s) * f
        d = torch.clamp(q2 - m_a * m_a, min=1e-30)
        Ca = Ca + torch.diag(d - torch.diagonal(Ca))
        m = m_a @ W[i]
        C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · {DEV} · збурення на шарах {L0S}\n', flush=True)

GS = np.zeros((len(FS), len(L0S), len(EPS)))
GO = np.zeros((len(FS), len(L0S), len(EPS)))
GM = np.zeros((len(FS), len(L0S)))
P1 = np.zeros(len(FS))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float64), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        out0, s0 = chain(W)
        # ⛔P1: eps=0 мусить дати ту саму траєкторію
        outz, _ = chain(W, kick_at=8, eps=0.0)
        P1[n] = float(torch.abs(outz - out0).max())
        for a, l0 in enumerate(L0S):
            # 🚨 НОРМУВАННЯ ПО НОРМІ, А НЕ ПОЕЛЕМЕНТНО: `out0` для частини
            #    нейронів близьке до нуля ⟹ поелементне `(o−out0)/out0`
            #    вибухає і дає фальшиві G≈10 замість аналітичних 0.0015
            #    (спіймано пломбою P4). Відношення норм від цього вільне.
            def rel(x, x0):
                return float(torch.sqrt(((x - x0) ** 2).sum()
                                        / (x0 * x0).sum()))

            for b, e in enumerate(EPS):
                o, s = chain(W, kick_at=l0, eps=e)
                GS[n, a, b] = rel(s, s0) / abs(e)
                GO[n, a, b] = rel(o, out0) / abs(e)
            om, _ = chain(W, kick_at=l0, eps=0.01, kick_m=True)
            GM[n, a] = rel(om, out0) / 0.01
    print(f'  net{n} · G_out(l=1) {GO[n, 0, 2]:+.3f} · G_out(l=16) '
          f'{GO[n, 4, 2]:+.3f} · G_σ(l=16) {GS[n, 4, 2]:+.3f}', flush=True)

print(f'\n{"═" * 74}\n⛔ПЛОМБИ')
print(f' P1 eps=0 ⟹ траєкторія незмінна: max|Δ| = {P1.max():.3e}  '
      f'{"✅" if P1.max() < 1e-15 else "⛔"}')
sym = np.abs(GO[:, :, 0].mean(0) - GO[:, :, 1].mean(0)).max() / max(GO[:, :, 0].mean(), 1e-12)
print(f' P2 симетрія G(+ε) vs G(−ε) при 0.1%: max розб = {sym:.4f}  '
      f'{"✅" if sym < 0.05 else "⛔ нелінійно"}')
lin = np.abs(GO[:, :, 0].mean(0) - GO[:, :, 2].mean(0)).max() / max(GO[:, :, 0].mean(), 1e-12)
print(f' P3 лінійність 0.1% vs 1%: max розб = {lin:.4f}  '
      f'{"✅" if lin < 0.05 else "⛔ нелінійно"}')
print(f' P4 останній шар G_out = {GO[:, -1, 2].mean():.4f}  '
      f'(аналітично φ(t)σ/(mΦ+σφ) ≈ 0.004 при t≈3)')

print(f'\n{"═" * 74}')
print('🎯 ПІДСИЛЕННЯ ЗБУРЕННЯ `σ`, ВНЕСЕНОГО НА ШАРІ l₀ (ε=1%)')
print('l₀        : ' + ' '.join(f'{l:8d}' for l in L0S))
print('G_σ→σ₃₁   : ' + ' '.join(f'{GS[:, a, 2].mean():+8.3f}'
                                for a in range(len(L0S))))
print('G_σ→вихід : ' + ' '.join(f'{GO[:, a, 2].mean():+8.3f}'
                                for a in range(len(L0S))))
print('G_m→вихід : ' + ' '.join(f'{GM[:, a].mean():+8.3f}'
                                for a in range(len(L0S))))
print('розкид G_σ: ' + ' '.join(f'{GS[:, a, 2].std():8.3f}'
                                for a in range(len(L0S))))

g1 = GS[:, 0, 2].mean()
print(f'\n📣 ВИРОК ПО ДИНАМІЦІ:')
if abs(g1) < 1.0:
    print(f'   |G_σ| = {abs(g1):.3f} < 1 ⟹ збурення ЗАБУВАЄТЬСЯ. Тоді 14.8% —')
    print('   це ПОСТІЙНЕ джерело на кожному кроці, і поправка 0.3%/крок')
    print('   закрила б усе. 🟢 НАЙКРАЩИЙ можливий результат.')
else:
    print(f'   |G_σ| = {abs(g1):.3f} ≥ 1 ⟹ рекурсія ПІДСИЛЮЄ збурення ⟹')
    print('   локальна поправка кроку множиться разом із сигналом і НЕ рятує.')
    print('   Потрібен ЯКІР (точне значення ззовні), а не краща формула.')
