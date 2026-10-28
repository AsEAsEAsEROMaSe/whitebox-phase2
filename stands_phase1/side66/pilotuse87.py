"""ПІЛОТНІ СЕМПЛИ: ОПЛАЧЕНІ Й ВИКИНУТІ. ПЕРЕМІР ПРОТИ ПОВНОЇ БАЗИ. (С87)

## ЗВІДКИ
`estimator_sub97.py` (рядки 271-279) жене 256 антитетичних семплів крізь
**ПОВНУ** мережу, бере з них ЛИШЕ маску «хто мертвий» (`max(h) ≥ 0`), а вихід
останнього шару ВИКИДАЄ. Флопи вже оплачені (3.9% бюджету).
🔑 А це НЕЗМІЩЕНА оцінка рівно тієї величини, яку міряє грейдер — причому
ПОВНОЇ мережі, тоді як основна гілка йде крізь ОБРІЗАНУ і несе зміщення
обрізання.

## 🚨 ЧОМУ ПЕРЕМІР
С76 зміряла **1.0080×** (LOO по мережах) — але проти бази ТІЄЇ епохи (`sub87`).
Відтоді в сабміт увійшли Еджвортова чистка `δ_Q` і бленд. Закон С86:
**ВИГРАШ — ВЛАСТИВІСТЬ ПАРИ (ОЗНАКА, БАЗА)** ⟹ вимір проти застарілої бази
не рахується. Тут відтворюємо ПОВНИЙ конвеєр: пілот → маска → обрізані ваги →
основна МК крізь обрізану → бленд із замиканням.

## 📐 ЧОМУ ВИГРАШ МАЄ БУТИ МЕНШИМ ЗА АРИФМЕТИЧНИЙ
Формула `w = MSE₁/(MSE₁+MSE₂)` вимагає НЕЗАЛЕЖНОСТІ, а **обрізання САМЕ
визначається пілотом**: більші пілотні активації ⟹ менше нейронів здаються
мертвими ⟹ менше зміщення ⟹ обидві оцінки їдуть в один бік. Пілот
використовується ДВІЧІ, і це з'їдає ⅔ виграшу (С76: 1.0161 обіцяно → 1.0080).

## ⛔ПЛОМБИ
 P1 `w = 0` мусить дати РІВНО базу (тотожність).
 P2 частка мертвих нейронів мусить бути ~20-30% (відомо з С69: 24% нейронів
    відкриваються рідше ніж у 0.2% семплів).
 P3 ⛔ПІЛОТ ІЗ ЧУЖОЇ МЕРЕЖІ — мусить ПОГІРШИТИ. Якщо ні, працює не пілот.
 P4 вага `w` підбирається LOO ПО МЕРЕЖАХ, не глобально.
 P5 α-калібровка на кожній сходинці (закон С87).

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) виграш проти повної бази буде МЕНШИЙ за 1.0080 (бо база тепер сильніша).
     Шанс 70%.
 (б) виграш лишиться додатним (≥1.002). Шанс **40%**.
 (в) оптимальна `w` вийде 0.005–0.02 (як у С76: 0.0117). Шанс 65%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/pilotuse87.py [cuda:N] [NETS] [NSAMP] [LOG2TRUE]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
NTRUE = 1 << (int(sys.argv[4]) if len(sys.argv) > 4 else 20)
PILOT_PAIRS = 128
BATCH = 1 << 13
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)
SUB97 = 1.873e-06


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def closure_chain(W):
    """Наше замикання від входу — те, що йде в бленд як КЗ."""
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(len(W)):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        t = m / s
        P, f = Phi(t), phi(t)
        m_a = m * P + s * f
        if i == len(W) - 1:
            return m_a
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        q2 = (m * m + s * s) * P + (m * s) * f
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                             - torch.diagonal(Ca))
        m = m_a @ W[i + 1]
        C = W[i + 1].T @ (Ca @ W[i + 1])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · N={NSAMP} · пілот {2 * PILOT_PAIRS} · {DEV}\n',
      flush=True)

MAIN = np.zeros((len(FS), W_))    # основна МК крізь ОБРІЗАНУ мережу
PILOT = np.zeros((len(FS), W_))   # оцінка з ПІЛОТНИХ семплів (повна мережа)
PILOTX = np.zeros((len(FS), W_))  # ⛔пілот із ЧУЖОЇ мережі
CLOS = np.zeros((len(FS), W_))    # замикання (для бленду)
TRUE_ = np.zeros((len(FS), W_))
DEAD = np.zeros(len(FS))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(770000 + n)
        d = 0
        while d < NTRUE:
            b = min(BATCH, NTRUE - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                a = torch.relu(a @ W32[q])
            acc += a.sum(0, dtype=torch.float64)
            d += b
        TRUE_[n] = (acc / NTRUE).cpu().numpy()

        # ── 1) ПІЛОТ: 128 антитетичних пар крізь ПОВНУ мережу ────────────
        gp = torch.Generator(device=DEV)
        gp.manual_seed(880000 + n)
        zp = torch.randn(PILOT_PAIRS, W_, device=DEV, generator=gp)
        xp = torch.cat([zp, -zp], 0)
        alive = []
        a = xp
        for i in range(L_):
            h = a @ W32[i]
            if i < L_ - 1:
                alive.append(h.max(0).values >= 0.0)
            a = torch.relu(h)
        PILOT[n] = a.double().mean(0).cpu().numpy()   # ← ЦЕ САБМІТ ВИКИДАЄ
        DEAD[n] = float(1.0 - torch.stack(alive).double().mean())

        # ── 2) ОБРІЗАНІ ваги (як у сабміті) ──────────────────────────────
        Wc = []
        for i in range(L_):
            wi = W32[i]
            if i > 0:
                wi = wi[alive[i - 1], :]
            if i < L_ - 1:
                wi = wi[:, alive[i]]
            Wc.append(wi)

        # ── 3) основна МК крізь ОБРІЗАНУ мережу ──────────────────────────
        gw = torch.Generator(device=DEV)
        gw.manual_seed(660000 + n)
        z = torch.randn(NSAMP // 2, W_, device=DEV, generator=gw)
        a = torch.cat([z, -z], 0)
        for i in range(L_):
            # 🚨 БЕЗ повторного обрізання: `Wc[i]` уже має форму
            #    (alive[i-1].sum(), alive[i].sum()), тож ширина `a` збігається
            #    автоматично. Додаткова маска давала IndexError.
            a = torch.relu(a @ Wc[i])
        MAIN[n] = a.double().mean(0).cpu().numpy()
        CLOS[n] = closure_chain(W).cpu().numpy()
    print(f'  net{n} · мертвих {DEAD[n]:.1%}', flush=True)

# ⛔P3: пілот із ЧУЖОЇ мережі (зсув на 1)
for n in range(len(FS)):
    PILOTX[n] = PILOT[(n + 1) % len(FS)]

NRM2 = (TRUE_ ** 2).mean(1)


def mse_of(P):
    return ((P - TRUE_) ** 2).mean(1)


def blend_loo(A, B):
    """Оптимальна вага `w` для `w·A + (1−w)·B`, підібрана LOO ПО МЕРЕЖАХ."""
    dA, dB = A - TRUE_, B - TRUE_
    num = (dB * (dB - dA)).sum(1)
    den = ((dB - dA) ** 2).sum(1)
    w_n = num / np.maximum(den, 1e-30)
    out = np.zeros(len(FS))
    ws = np.zeros(len(FS))
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        w = float(np.clip(w_n[msk].mean(), -0.5, 1.5)) if len(FS) > 1 else float(w_n[i])
        ws[i] = w
        r = w * dA[i] + (1.0 - w) * dB[i]
        out[i] = float((r * r).mean())
    return out.mean(), ws.mean()


print(f'\n{"═" * 70}\n⛔ПЛОМБИ')
print(f' P2 мертвих нейронів: {DEAD.mean():.1%}  (очік. 20–30% з С69)  '
      f'{"✅" if 0.15 < DEAD.mean() < 0.40 else "⛔"}')
b_only = mse_of(MAIN).mean()
print(f' P1 w=0 ≡ база: {b_only:.4e}')

# ПОВНА БАЗА = МК крізь обрізану + бленд із замиканням (як у сабміті)
base_mse, w_cl = blend_loo(CLOS, MAIN)
print(f'\n{"═" * 70}')
print('🎯 ПІЛОТ ПРОТИ ПОВНОЇ БАЗИ (МК-обрізана + бленд із замиканням)')
print(f'{"варіант":34s} {"MSE":>11s} {"vs база":>9s} {"w":>8s}')
print(f'{"МК крізь обрізану (сама)":34s} {b_only:11.4e} '
      f'{"×%.4f" % (base_mse / b_only):>9s}')
print(f'{"ПОВНА БАЗА (+бленд замикання)":34s} {base_mse:11.4e} '
      f'{"×1.0000":>9s} {w_cl:8.4f}')

# пілот ПОВЕРХ повної бази
# 🚨 БАЗА ДЛЯ ПОРІВНЯННЯ мусить бути ТА САМА, що йде в бленд: `base_mse`
#    рахується з ІНДИВІДУАЛЬНОЮ LOO-вагою, а `base_pred` — із СЕРЕДНЬОЮ.
#    Через цю неузгодженість ⛔чужий пілот показував фальшивий ×1.019 при w≈0.
base_pred = w_cl * CLOS + (1.0 - w_cl) * MAIN
base2 = mse_of(base_pred).mean()
p_mse, w_p = blend_loo(PILOT, base_pred)
px_mse, w_px = blend_loo(PILOTX, base_pred)
print(f'{"🎯 +ПІЛОТ поверх повної бази":34s} {p_mse:11.4e} '
      f'{"×%.4f" % (base2 / p_mse):>9s} {w_p:8.4f}')
print(f'{"⛔ +пілот ЧУЖОЇ мережі":34s} {px_mse:11.4e} '
      f'{"×%.4f" % (base2 / px_mse):>9s} {w_px:8.4f}')

print(f'\n📣 ВИРОК: пілот проти ПОВНОЇ бази = ×{base2 / p_mse:.4f}   '
      f'(база для обох: {base2:.4e})')
print(f'   (С76 давала ×1.0080 проти бази sub87-епохи)')
print(f'   ⛔чужий пілот ×{base2 / px_mse:.4f} — мусить бути помітно гіршим')
if base2 / p_mse >= 1.002 and (base2 / p_mse - 1) > 2.0 * (base2 / px_mse - 1):
    print(f'   🟢 ЖИВИЙ · очікуваний бал {SUB97 * 0.1 / (base2 / p_mse):.4e}')
else:
    print('   ⛔ проти повної бази виграш зник ⟹ база вже забрала цю інформацію')
