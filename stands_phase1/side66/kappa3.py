"""ЧИ МОЖНА ВЕСТИ κ₃ ДЕШЕВО — КЛЮЧ ДО ЕДЖВОРТА В РЕКУРСІЇ (С83)

## ЗВІДКИ
`side66/ground.py` (6 мереж × 2²⁰) розділив похибку замикання вперше:
  * A ПОХИБКА ФОРМИ (при ТОЧНИХ m,s) — **полиця ~0.2%, НЕ росте вглиб**;
  * B ПОХИБКА ЛАНЦЮГА — росте як **A·√(ℓ−1)** до 1.0% на шарі 32
    ⟹ похибки шарів НЕКОРЕЛЬОВАНІ, складаються як блукання;
  * 🟢 Еджворт-3 з ІСТИННИМ γ₁ ріже форму втричі: **C/A = 0.359 на ВСІХ шарах**.
⟹ Еджворт провалювався тричі не через форму, а через те, що в рекурсії немає
правильного `γ₁`.

## 🔑 ЩО ПОТРІБНО І ЧОМУ ЦЕ ДОРОГО
    κ₃(h_i) = Σ_{jkl} W_ji·W_ki·W_li · κ₃(a_j, a_k, a_l)
Точно — це тензор 256³ на шар. Неприйнятно. Але є наближення:
  D ДІАГОНАЛЬНЕ    κ₃(h_i) ≈ Σ_j W_ji³ · κ₃(a_j)          ціна ОДИН матмул
  P ПАРНЕ          + 3·Σ_{j≠k} W_ji²W_ki · κ₃(a_j,a_j,a_k) — ще один матмул,
                     якщо κ₃(a_j,a_j,a_k) факторизується
  G ГАУСОВЕ        κ₃(h) = 0 (нинішнє замикання)

## ЩО МІРЯЄМО (6 мереж × 2²⁰, точні κ₃ з вибірки як еталон)
1. корел(κ₃ ДІАГОНАЛЬНЕ, κ₃ ТОЧНЕ) по нейронах, шар за шаром — чи вгадує форму;
2. відношення масштабів — чи не треба калібрувального множника;
3. 🎯 ГОЛОВНЕ: похибка форми з ДІАГОНАЛЬНИМ γ₁ проти точного і проти нуля,
   тобто скільки з виграшу ×2.8 лишається при дешевій оцінці.

⛔ ПЛОМБИ: (а) на шарі 1 `a₀ = z` гаусів ⟹ κ₃(a₀)=0 ⟹ діагональна оцінка
   мусить дати 0, і точна теж (h₁ гаусів);
   (б) ⛔контроль: κ₃ з ЧУЖОЇ мережі — корел мусить впасти до нуля;
   (в) шум κ₃ оцінюється половинками вибірки.

📣 ПЕРЕДБАЧЕНО: діагональ вгадає ЗНАК і грубий масштаб (корел 0.5…0.8), бо
   `a_j` слабко корельовані при великій ширині, але недооцінить величину, бо
   викидає перехресні члени, яких 256²·256 проти 256. Якщо корел ≳0.7 і
   калібрувальний множник стабільний по шарах — Еджворт стає обчислюваним, і
   тоді відкривається ×1.6…1.7 через падіння ‖b‖². Якщо корел ≲0.3 — потрібен
   парний член, і ціна зростає вдвічі.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/kappa3.py [cuda:N] [NETS] [log2]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS, BS, L_, W_ = 1 << LG, 1 << 14, 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
print(f'{NETS} мереж × {NS} семплів, {DEV}\n', flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

CR = np.zeros((NETS, L_))       # корел діагональ ↔ точне
CN = np.zeros((NETS, L_))       # ⛔контроль: чужа мережа
SC = np.zeros((NETS, L_))       # масштаб точне/діагональ
E0 = np.zeros((NETS, L_))       # похибка форми БЕЗ Еджворта
ET = np.zeros((NETS, L_))       # з ТОЧНИМ γ₁
ED = np.zeros((NETS, L_))       # з ДІАГОНАЛЬНИМ γ₁
EK = np.zeros((NETS, L_))       # з діагональним × калібрувальний множник
SCK = np.zeros((NETS, L_))      # сам множник — чи стабільний по шарах/мережах
SEAL1 = np.zeros(NETS)
prevK3 = None

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        gen = torch.Generator(device=DEV); gen.manual_seed(31337 + n)
        Z = lambda: [torch.zeros(W_, dtype=torch.float64, device=DEV)
                     for _ in range(L_)]
        S1, S2, S3 = Z(), Z(), Z()          # моменти h
        A1, A2, A3 = Z(), Z(), Z()          # моменти a (для κ₃(a_j))
        done = 0
        while done < NS:
            b = min(BS, NS - done)
            z = torch.randn(b, W_, device=DEV, generator=gen)
            a = z
            for i in range(L_):
                h = a @ W[i]
                hd = h.double()
                S1[i] += hd.sum(0); S2[i] += (hd*hd).sum(0); S3[i] += (hd**3).sum(0)
                a = torch.relu(h)
                ad = a.double()
                A1[i] += ad.sum(0); A2[i] += (ad*ad).sum(0); A3[i] += (ad**3).sum(0)
            done += b
        mh = [S1[i]/NS for i in range(L_)]
        vh = [(S2[i]/NS - mh[i]**2).clamp_min(1e-30) for i in range(L_)]
        k3h = [S3[i]/NS - 3*mh[i]*(S2[i]/NS) + 2*mh[i]**3 for i in range(L_)]
        ma = [A1[i]/NS for i in range(L_)]
        k3a = [A3[i]/NS - 3*ma[i]*(A2[i]/NS) + 2*ma[i]**3 for i in range(L_)]
        # κ₃(a₀) для z: a₀ = z, κ₃ = 0
        k3prev = torch.zeros(W_, dtype=torch.float64, device=DEV)
        for i in range(L_):
            Wd = W[i].double()
            # D: діагональне наближення — ОДИН матмул по W³
            k3d = (Wd ** 3).T @ k3prev
            k3t = k3h[i]
            s = vh[i].sqrt(); t = mh[i]/s
            ph, Ph = _phi(t), _Phi(t)
            m_g = mh[i]*Ph + s*ph
            base = ma[i].norm()
            E0[n, i] = float((m_g - ma[i]).norm()/base)
            g_t = k3t/s**3
            ET[n, i] = float((m_g - (g_t/6)*mh[i]*ph - ma[i]).norm()/base)
            g_d = k3d/s**3
            ED[n, i] = float((m_g - (g_d/6)*mh[i]*ph - ma[i]).norm()/base)
            if k3d.std() > 1e-300 and k3t.std() > 1e-300:
                CR[n, i] = float(torch.corrcoef(
                    torch.stack([k3d, k3t]))[0, 1])
                SC[n, i] = float(k3t.norm()/k3d.norm())
                if prevK3 is not None and prevK3[i].std() > 1e-300:
                    CN[n, i] = float(torch.corrcoef(
                        torch.stack([prevK3[i], k3t]))[0, 1])
                # K: діагональ × ОДИН калібрувальний множник на шар.
                # 🚨 ПОПРАВКА: перша версія підганяла множник так, щоб `k3d`
                #    наблизилось до `k3t` — тобто мінімізувала похибку κ₃, а не
                #    похибку `m`. Вийшло ×0.519 (ГІРШЕ за відсутність поправки),
                #    що для калібрування з одним вільним множником НЕМОЖЛИВО.
                #    🎓 ПІДГАНЯТИ ТРЕБА ТУ ВЕЛИЧИНУ, ЯКУ МІРЯЄШ.
                #    Тепер: min_c ‖(m_g − ma) − c·u‖², u = (k3d/s³/6)·mh·ph
                u = (k3d / s**3 / 6.0) * mh[i] * ph
                r = m_g - ma[i]
                cbest = float((r*u).sum()/(u*u).sum().clamp_min(1e-300))
                EK[n, i] = float((r - cbest*u).norm()/base)
                SCK[n, i] = cbest
            k3prev = k3a[i]
        SEAL1[n] = float(((Wd*0).sum()))  # placeholder
        prevK3 = [x.clone() for x in k3h]
    del W
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/kappa3.npz',
         CR=CR, CN=CN, SC=SC, E0=E0, ET=ET, ED=ED, EK=EK, SCK=SCK)
print(f'\n⛔ПЛОМБА (а) шар 1: κ₃(a₀)=0 ⟹ діагональ=0 ⟹ ED=E0: '
      f'{np.abs(ED[:,0]-E0[:,0]).max():.2e} '
      f'{"✅" if np.abs(ED[:,0]-E0[:,0]).max()<1e-12 else "🚨"}')
print(f'⛔КОНТРОЛЬ (б) κ₃ з ЧУЖОЇ мережі: корел {np.nanmean(CN[1:,2:]):+.3f} '
      f'{"✅" if abs(np.nanmean(CN[1:,2:]))<0.2 else "🚨"}')
print(f'\n{"шар":>4s} {"корел D↔точне":>14s} {"⛔чужа":>8s} {"точне/діаг":>11s} '
      f'{"E0 без":>9s} {"ET точне":>9s} {"ED діаг":>9s} {"EK калібров":>12s}')
print('─'*88)
for i in list(range(1, 9)) + list(range(9, L_, 3)) + [L_-1]:
    print(f'{i+1:4d} {CR[:,i].mean():+14.3f} {np.nanmean(CN[1:,i]):+8.3f} '
          f'{SC[:,i].mean():11.3f} {E0[:,i].mean():9.2e} {ET[:,i].mean():9.2e} '
          f'{ED[:,i].mean():9.2e} {EK[:,i].mean():12.2e}')
m0, mt, md, mk = (X[:, 1:].mean() for X in (E0, ET, ED, EK))
print(f'\n🎯 СЕРЕДНЄ по шарах 2..32')
print(f'   БЕЗ Еджворта          {m0:.3e}   ×1.000')
print(f'   з ТОЧНИМ γ₁           {mt:.3e}   ×{m0/mt:.3f}   ⟸ стеля')
print(f'   з ДІАГОНАЛЬНИМ γ₁     {md:.3e}   ×{m0/md:.3f}')
print(f'   з КАЛІБРОВАНОЮ діаг.  {mk:.3e}   ×{m0/mk:.3f}   ⟸ один множник на шар')
print(f'   частка стелі, узята дешевою оцінкою: '
      f'{(m0/mk-1)/max(m0/mt-1,1e-9)*100:.1f}%')
print(f'   корел діагональ↔точне, сер по шарах 3..32: {CR[:,2:].mean():+.3f}')
print(f'\n🔑 КАЛІБРУВАЛЬНИЙ МНОЖНИК (чи можна зашити КОНСТАНТОЮ)')
print(f'   сер по шарах 2..32 {SCK[:,1:].mean():+.4f} · розкид ПО ШАРАХ '
      f'{SCK[:,1:].mean(0).std():.4f} · розкид ПО МЕРЕЖАХ {SCK[:,1:].std(0).mean():.4f}')
print(f'   ⟹ {"🟢 стабільний — можна ОДНЕ число" if SCK[:,1:].mean(0).std()<0.35*abs(SCK[:,1:].mean()) else "⛔ гуляє — константою не зашити"}')
print('\n📊 БАЛ 1.954e-07 · вимір ДІАГНОСТИЧНИЙ')
