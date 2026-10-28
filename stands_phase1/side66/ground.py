"""МЕТОД З НУЛЯ: ДЕ САМЕ НАРОДЖУЄТЬСЯ ПОХИБКА ЗАМИКАННЯ (С83)

## ЗВІДКИ ПОСТАНОВКА (вказівка ЮЗЕРА «виводимо метод з 0, спуститись на низ»)
Обидва наші класи мають зміряні стелі, і жоден не дістає лідера:
  * кращий МК — ×7 ([[finding-s2-degree-spectrum-bounds-the-whole-exact-family]]), ми на ×3.4;
  * ІДЕАЛЬНЕ замикання — ×12.1 ([[finding-s2-perfect-closure-ceiling-is-12x-and-that-caps-everything]]);
  * лідер — ×24.4.
⟹ лідер робить ТРЕТЄ. Отже треба не полірувати класи, а знайти єдиний хибний
крок у точній математиці.

## 🔑 ТОЧНА РЕКУРСІЯ МАЄ РІВНО ДВА КРОКИ НА ШАР
    h_ℓ = Wᵀ·a_{ℓ−1}   ⟹  E[h] = Wᵀm,  Cov(h) = WᵀCW     ← ТОЧНО, БЕЗ ПРИПУЩЕНЬ
    a_ℓ = ReLU(h_ℓ)    ⟹  E[a] = ?                        ← ЄДИНА АПРОКСИМАЦІЯ
Для СЕРЕДНЬОГО потрібен лише ОДНОВИМІРНИЙ маргінал кожного `h_i`. Тобто
256-вимірний інтеграл зводиться до 256 одновимірних, і все питання — наскільки
негаусовий маргінал.

## 🚨 ЩО САМЕ МІРЯЄМО (чого досі не робили)
Еджворт провалювався ТРИЧІ, але завжди як поправка до НАКОПИЧЕНОГО замикання.
Ніхто не розділив дві похибки. Тут вони рахуються окремо, на кожному шарі:

  A ПОХИБКА ФОРМИ      `m_гаус(ТОЧНІ m_h, s_h) − E[ReLU(h)]`
      — чиста негаусовість маргіналу, при ІДЕАЛЬНОМУ вході;
  B ПОХИБКА НАКОПИЧЕННЯ `m_замик(ℓ) − E[a_ℓ]`
      — те, що дає нинішній ланцюг Мелера;
  C ЕДЖВОРТ НА ФОРМІ    чи `−(γ₁/6)·mp·φ` з ІСТИННИМ `γ₁` зменшує A.

🎯 ТРИ МОЖЛИВІ ВИРОКИ, І ВОНИ ВЕДУТЬ У РІЗНІ БОКИ:
  1. A ≪ B  ⟹ форма майже точна, уся біда — НАКОПИЧЕННЯ ⟹ лікувати рекурсію
             (перезапуск ланцюга, гібрид «вибірка кожні k шарів»);
  2. A ≈ B  ⟹ похибка народжується щошару локально ⟹ лікувати ФОРМУЛУ;
  3. A > B  ⟹ 🚨 похибки ГАСЯТЬ одна одну — це кількісне підтвердження
             «точки балансу», і тоді ясно, ЧОМУ Еджворт тричі шкодив.

⛔ ПЛОМБИ: (а) `Cov(h) = WᵀCW` мусить збігтися з вибірковою до шуму МК;
   (б) на шарі 1 `h` ТОЧНО гаусів ⟹ A мусить бути ≈0 (чистий тест приладу);
   (в) шум істини оцінюється половинками вибірки.

📣 ПЕРЕДБАЧЕНО: на шарі 1 A = 0 (пломба). Далі A росте й виходить на полицю
   ~0.1–0.3% (це «народження»), а B росте монотонно до 0.5% (накопичення).
   Якщо так — вирок 1, і лікувати треба рекурсію, а не формулу. Якщо ж A
   виявиться БІЛЬШОЮ за B — підтверджено баланс, і тоді єдиний чесний шлях —
   виправляти форму й аргумент РАЗОМ, а не поодинці.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/ground.py [cuda:N] [NETS] [log2(семплів)]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import closure91, _phi, _Phi                     # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS, BS, L_, W_ = 1 << LG, 1 << 14, 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
print(f'{NETS} мереж × {NS} семплів (fp64 акумулятори), {DEV}\n', flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

# накопичувачі по шарах: відносні похибки A, B і ефект Еджворта
RA = np.zeros((NETS, L_)); RB = np.zeros((NETS, L_)); RC = np.zeros((NETS, L_))
RS = np.zeros((NETS, L_))          # шум істини (половинки)
G1 = np.zeros((NETS, L_))          # середній |γ₁| маргіналу
G2 = np.zeros((NETS, L_))          # середній |γ₂| (ексцес)
SEAL_COV = np.zeros(NETS)
SEAL_L1 = np.zeros(NETS)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        # ── ТОЧНІ моменти великою вибіркою, по батчах, fp64 акумулятори
        gen = torch.Generator(device=DEV); gen.manual_seed(31337 + n)
        S1 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S2 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S3 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S4 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        AM = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        AH = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        CH = None
        done = 0
        while done < NS:
            b = min(BS, NS - done)
            z = torch.randn(b, W_, device=DEV, generator=gen)
            a = z
            for i in range(L_):
                h = a @ W[i]
                hd = h.double()
                S1[i] += hd.sum(0); S2[i] += (hd * hd).sum(0)
                S3[i] += (hd ** 3).sum(0); S4[i] += (hd ** 4).sum(0)
                a = torch.relu(h)
                ad = a.double()
                AM[i] += ad.sum(0)
                if done < NS // 2:
                    AH[i] += ad.sum(0)            # перша половина — для шуму
                if i == 0 and CH is None and done == 0:
                    CH = (hd.T @ hd) / b          # ⛔пломба коваріації шару 1
            done += b
        mh = [s / NS for s in S1]
        vh = [(S2[i] / NS - mh[i] ** 2).clamp_min(1e-30) for i in range(L_)]
        k3 = [S3[i] / NS - 3 * mh[i] * (S2[i] / NS) + 2 * mh[i] ** 3
              for i in range(L_)]
        k4 = [S4[i] / NS - 4 * mh[i] * (S3[i] / NS)
              + 6 * mh[i] ** 2 * (S2[i] / NS) - 3 * mh[i] ** 4
              - 3 * vh[i] ** 2 for i in range(L_)]
        ma = [AM[i] / NS for i in range(L_)]
        mah = [AH[i] / (NS // 2) for i in range(L_)]

        # ⛔ПЛОМБА (а): Cov(h₁) = W₁ᵀW₁ (бо a₀ = z має Cov = I)
        exact = W[0].double().T @ W[0].double()
        SEAL_COV[n] = float((CH - exact).abs().max() / exact.abs().max())

        # ── ЗАМИКАННЯ Мелера (те, що в сабміті), усі 32 шари
        s_cl, m_cl, q_cl, P_cl, m_deep = closure91(W, L_ - 1, DEV, kdeep=None)

        for i in range(L_):
            s = vh[i].sqrt(); t = mh[i] / s
            ph, Ph = _phi(t), _Phi(t)
            # A: ФОРМА — гаусова формула при ТОЧНИХ m_h, s_h
            m_g = mh[i] * Ph + s * ph
            RA[n, i] = float(((m_g - ma[i]).norm() / ma[i].norm()))
            # C: Еджворт-3 на формі, з ІСТИННИМ γ₁
            g1 = k3[i] / s ** 3
            m_e = m_g - (g1 / 6.0) * mh[i] * ph
            RC[n, i] = float(((m_e - ma[i]).norm() / ma[i].norm()))
            # B: НАКОПИЧЕННЯ — ланцюг Мелера.
            # 🚨 ІНДЕКСАЦІЯ: `closure91` накопичує `m_o` лише при 1<=i<=ksig,
            #    тож `m_cl[k]` — це шар k+2 у нумерації «ma[0] = після W[0]».
            #    Перша версія звіряла зі зсувом на шар і дала B≈1.1 (110%!)
            #    замість відомих 0.5% ⟹ 🎓 АБСУРДНЕ ЧИСЛО = ІНДЕКС, А НЕ ФІЗИКА.
            if 1 <= i <= len(m_cl):
                RB[n, i] = float(((m_cl[i - 1] - ma[i]).norm() / ma[i].norm()))
            # шум істини: половина проти повної
            RS[n, i] = float(((mah[i] - ma[i]).norm() / ma[i].norm()))
            G1[n, i] = float(g1.abs().mean())
            G2[n, i] = float((k4[i] / vh[i] ** 2).abs().mean())
        SEAL_L1[n] = RA[n, 0]
    del W
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/ground.npz',
         RA=RA, RB=RB, RC=RC, RS=RS, G1=G1, G2=G2)

print(f'\n⛔ПЛОМБА (а) Cov(h₁)=W₁ᵀW₁: max відн. {SEAL_COV.max():.2e} '
      f'{"✅" if SEAL_COV.max() < 5e-3 else "🚨"}')
print(f'⛔ПЛОМБА (б) шар 1 гаусів ⟹ A≈шум: A={SEAL_L1.mean():.2e} проти '
      f'шуму {RS[:,0].mean():.2e} '
      f'{"✅" if SEAL_L1.mean() < 4*RS[:,0].mean()+1e-9 else "🚨 форма НЕ нульова там, де мусить"}')
print(f'⛔ПЛОМБА (в) шум істини: медіана по шарах {np.median(RS):.2e}\n')

print(f'{"шар":>4s} {"A ФОРМА":>10s} {"B НАКОПИЧ":>11s} {"A/B":>7s} '
      f'{"C ЕДЖВОРТ":>11s} {"C/A":>7s} {"|γ₁|":>8s} {"|γ₂|":>8s} {"шум":>9s}')
print('─' * 84)
for i in list(range(0, 8)) + list(range(8, L_, 3)) + [L_ - 1]:
    a, b, c = RA[:, i].mean(), RB[:, i].mean(), RC[:, i].mean()
    print(f'{i+1:4d} {a:10.3e} {b:11.3e} {a/max(b,1e-30):7.3f} '
          f'{c:11.3e} {c/max(a,1e-30):7.3f} {G1[:,i].mean():8.4f} '
          f'{G2[:,i].mean():8.4f} {RS[:,i].mean():9.2e}')
print('\n🎯 ВИРОК')
mA, mB = RA[:, 1:].mean(), RB[:, 1:len(m_cl)+1].mean()
if mA < 0.5 * mB:
    print('   1️⃣ A ≪ B ⟹ ФОРМА майже точна, біда — НАКОПИЧЕННЯ.')
    print('      ⟹ лікувати РЕКУРСІЮ (перезапуск ланцюга з вибірки кожні k шарів)')
elif mA < 1.5 * mB:
    print('   2️⃣ A ≈ B ⟹ похибка народжується щошару ЛОКАЛЬНО ⟹ лікувати ФОРМУЛУ')
else:
    print('   3️⃣ 🚨 A > B ⟹ похибки ГАСЯТЬ одна одну — кількісне підтвердження')
    print('      «точки балансу»; виправляти форму й аргумент можна лише РАЗОМ')
print(f'   середні по шарах 2..32:  A = {mA:.3e}   B = {mB:.3e}   '
      f'A/B = {mA/max(mB,1e-30):.3f}')
print(f'   Еджворт на ФОРМІ: C/A = {RC[:,1:].mean()/max(mA,1e-30):.3f} '
      f'{"🟢 зменшує" if RC[:,1:].mean() < mA else "⛔ НЕ зменшує навіть з істинним γ₁"}')
print('\n📊 БАЛ 1.954e-07 · вимір ДІАГНОСТИЧНИЙ, у сабміт нічого не йде')
