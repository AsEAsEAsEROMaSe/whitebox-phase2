"""АТРАКТОР ЯК ОПОРА НА ПРОМІЖНИХ ШАРАХ: чи він кращий еталон за замикання (С80).

Ідея юзера: «чи можна використати атракторні внутрішні проміжні стани шарів
як опору?»

## ЩО В КОДІ НАЗИВАЄТЬСЯ ОПОРОЮ (`base_real.run_side1`, рядок ~246)
    `aw = mean(a_ℓ) − m_cl[ℓ-1]`
тобто ВІДХИЛЕННЯ виміряного середнього шару від ЕТАЛОНА `m_cl` (із замикання),
перенесене до виходу; далі `λ` фітується LOO. Пропозиція юзера = поставити на
місце `m_cl` активації атрактора `a*_ℓ`.

🔑 ЧОМУ ЦЕ НЕ ТЕ, ЩО ВЖЕ ЗАКРИТО. Закрито ⛔КЗ з атрактора (усі його тотожності
лінійні, вибілювання їх уже спожило) і ⛔усадку виходу вздовж `x*` (1.0007×
при потрібних 2.00×). ✅ Тут інше: атрактор НЕ як контрольна змінна й НЕ на
виході, а як ЕТАЛОН СЕРЕДНЬОГО на ПРОМІЖНИХ шарах.

🔑 ЧОМУ ЦЕ МОЖЕ ПРАЦЮВАТИ. Зсувів нема ⟹ додатна однорідність ⟹ масштаб
`a*_ℓ` вільний, тож еталон = НАПРЯМОК + один скаляр. Якщо `E[a_ℓ]` майже
паралельне `a*_ℓ`, то 256-вимірне середнє зводиться до ОДНОГО числа.
🎓 Це та сама вісь, що дала «2.9 ефективних напрямки промаху» в `hitmiss2.py`.

## ВОРОТА ВИМІРУ (правило «СТЕЛЯ КАНАЛУ = ШАР ЙОГО ТОЧНОСТІ»)
Порівнюємо ВІДНОСНУ похибку трьох еталонів проти істини `E[a_ℓ]` пошарово:
  1. **замикання** `m_cl[ℓ]`      — що стоїть у сабміті зараз;
  2. **атрактор** `α·a*_ℓ`        — 🟢 пропозиція, `α` = найкращий скаляр (МНК);
  3. ⛔НЕГ **один випадковий семпл** `α·a_ℓ(z)` — має бути помітно гірше.
Плюс `σ_ℓ(N)` — шум вибіркового середнього на нашому `npair` ⟹ видно, чи
еталон узагалі проходить умову придатності `b_ℓ < σ_ℓ(N)`.

⛔ ПЛОМБА: `α·a*_ℓ` при `a*_ℓ = E[a_ℓ]` мусить дати рівно 0.
📣 ПЕРЕДБАЧЕНО: атрактор програє замиканню на ВСІХ шарах, бо на виході він
   збоку на 11.7–15.1%, а замикання — на 0.93%. Тобто ~16× гірше. Якщо ж на
   МІЛКИХ шарах атрактор виявиться точнішим за замикання — це нова гілка.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/attref.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts                # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, W_, KS = 32, 256, 31
TRUTH, CH = 1 << 20, 8192
print(__doc__.split('Запуск')[0].split('## ЩО')[0], flush=True)


def _fwd_all(wc, x):
    """Список активацій a_1..a_{L-1} (a_i = relu(a_{i-1} @ w_i))."""
    acts = []
    a = x
    for i in range(len(wc)):
        a = torch.relu(a @ wc[i])
        acts.append(a)
    return acts


def _attractor(wc, gen, it=1200):
    x = torch.randn(1, W_, device=DEV, generator=gen)
    x = x / x.norm()
    for _ in range(it):
        y = _fwd_all(wc, x)[-1]
        x = y / y.norm().clamp_min(1e-30)
    return x


def _relerr(ref, tru):
    """Відносна похибка ПІСЛЯ найкращого скалярного масштабу (МНК)."""
    a = float((ref * tru).sum() / (ref * ref).sum().clamp_min(1e-300))
    return float((a * ref - tru).norm() / tru.norm().clamp_min(1e-300)), a


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
fs = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

ECL, EAT, ENG, SIG, COS, SEAL = ([] for _ in range(6))
for n, f in enumerate(fs):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        _, m_cl, _, _ = closure_parts(wc, KS, DEV, extras=True)
        g = torch.Generator(device=DEV).manual_seed(9100 + 37 * n)

        # 🚨 ширини РІЗНІ, і `m_cl[j]` відповідає `acts[j+1]`, не `acts[j]`
        ws = [wc[i + 1].shape[1] for i in range(KS)]
        acc = [torch.zeros(w, dtype=torch.float64, device=DEV) for w in ws]
        ac2 = [torch.zeros(w, dtype=torch.float64, device=DEV) for w in ws]
        done = 0
        while done < TRUTH:                       # істина: антитетичні пари
            k = min(CH, TRUTH - done)
            z = torch.randn(k, W_, device=DEV, generator=g)
            A = _fwd_all(wc, torch.cat([z, -z], 0))
            for i in range(KS):
                d = A[i + 1].double()
                acc[i] += d.sum(0)
                ac2[i] += d.pow(2).sum(0)
            done += k
        nn = 2.0 * done
        T = [acc[i] / nn for i in range(KS)]
        # шум вибіркового середнього на НАШОМУ бюджеті (2·npair семплів)
        sig = [float((((ac2[i] / nn - T[i] ** 2).clamp_min(0)
                       / (2.0 * npair)).sum()).sqrt() / T[i].norm())
               for i in range(KS)]

        xs = _attractor(wc, g)
        AST = _fwd_all(wc, xs)
        zr = torch.randn(1, W_, device=DEV, generator=g)
        ARD = _fwd_all(wc, zr)                    # ⛔НЕГ: один випадковий семпл

        ecl, eat, eng, cs, sl = [], [], [], [], []
        for i in range(KS):
            ecl.append(_relerr(m_cl[i], T[i])[0])
            e, _ = _relerr(AST[i + 1].double().reshape(-1), T[i])
            eat.append(e)
            eng.append(_relerr(ARD[i + 1].double().reshape(-1), T[i])[0])
            u = AST[i + 1].double().reshape(-1)
            cs.append(float((u @ T[i]) / (u.norm() * T[i].norm())))
            sl.append(_relerr(T[i], T[i])[0])     # ⛔пломба: мусить бути 0
        ECL.append(ecl); EAT.append(eat); ENG.append(eng)
        SIG.append(sig); COS.append(cs); SEAL.append(max(sl))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

ECL, EAT, ENG = map(lambda a: np.median(np.array(a), 0), (ECL, EAT, ENG))
SIG, COS = np.median(np.array(SIG), 0), np.median(np.array(COS), 0)

print(f'\n{NETS} мереж, істина {2*TRUTH} семплів. ВІДНОСНА ПОХИБКА ЕТАЛОНА '
      f'(медіана по мережах)\n')
print(f'⛔ПЛОМБА (еталон = істина): {max(SEAL):.2e}  (0 = ок)\n')
print(f'{"шар":>4s} {"замикання":>11s} {"🟢атрактор":>12s} {"⛔1 семпл":>11s} '
      f'{"шум вибірки":>12s} {"cos(a*,E[a])":>13s} {"атр/замик":>10s}')
print('-' * 82)
for i in range(KS):
    print(f'{i+1:4d} {100*ECL[i]:10.4f}% {100*EAT[i]:11.4f}% '
          f'{100*ENG[i]:10.4f}% {100*SIG[i]:11.4f}% {COS[i]:13.6f} '
          f'{EAT[i]/max(ECL[i],1e-30):9.2f}×')
b = int(np.argmin(EAT / np.maximum(ECL, 1e-30)))
print(f'\n🔑 найкращий для атрактора шар {b+1}: він у '
      f'{EAT[b]/max(ECL[b],1e-30):.2f}× гірший за замикання')
print(f'🔑 шарів, де атрактор ТОЧНІШИЙ за замикання: '
      f'{int((EAT < ECL).sum())}/{KS}')
print(f'🔑 шарів, де атрактор проходить умову придатності `b<σ`: '
      f'{int((EAT < SIG).sum())}/{KS}   (замикання: {int((ECL < SIG).sum())}/{KS})')
print('📊 БАЛ 1.954e-07 — це ВОРОТА, приросту сам вимір не дає')
