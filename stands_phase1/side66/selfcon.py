"""КУДИ ДІВАЄТЬСЯ ×20: ПОПРАВКА ЛІКУЄ КРОК, АЛЕ ВБИВАЄ ЛАНЦЮГ (С84)

## 🚨 ПОСТАНОВКА — СУПЕРЕЧНІСТЬ У ВЛАСНИХ ЧИСЛАХ
Три записані виміри не сходяться:
  * при ТОЧНИХ `m,s` поправка `κ₃+κ₄` зменшує похибку кроку **×12.96**;
  * похибки шарів НЕКОРЕЛЬОВАНІ (зміряно: `B_ланцюг = A_крок·√(ℓ−1)`);
  * ⟹ ланцюг із поправкою мав би дати ≈×12.96, а дає **×0.65**.
Розрив **×20**. Пояснення «аргументи зміщені» — слова, не число.
🎯 Поки ця діра відкрита, будувати рекурсію для `κ₄` немає сенсу: вона лікує
джерело `γ`, а розрив ×20 — не про джерело `γ` (з вибірки ×0.64 ≈ точні ×0.65).

## 🔑 ГІПОТЕЗА, ЯКУ МІРЯЄМО
Ланцюг подає у крок НЕ істинні `m,C`, а свої, з похибкою `ε`. Крок працює як
підсилювач: `похибка_виходу ≈ A_власна + G·ε`. Нинішня формула має якесь `G`;
поправка Еджворта міняє **і `A`, і `G`**. Якщо вона ділить `A` на 13, але
множить `G` на щось >1 — при накопиченні по 32 шарах виграє `G`, і саме це
дасть ×0.65 при ×12.96 на кроці.
⟹ 🎓 **ПОПРАВКА, ЩО ЛІКУЄ ЛОКАЛЬНУ ФОРМУ, МОЖЕ ПОГІРШУВАТИ СТІЙКІСТЬ.**

## ЩО САМЕ МІРЯЄМО (пошарово, на кожному шарі окремо)
Істина — велика вибірка: `m_a[i] = E[a_i]`, `C_a[i] = Cov(a_i)`, плюс `κ₃,κ₄`
маргіналів `h`. Далі на КОЖНОМУ шарі роблять ОДИН крок із двох різних входів:
  T ТОЧНИЙ вхід   (`m_a[i-1]`, `C_a[i-1]`)      → похибка = чиста ФОРМА `A`
  X ЗБУРЕНИЙ вхід (`m_ch[i-1]`, `C_ch[i-1]`)    ← реальний стан ланцюга
і для кожного — у двох варіантах формули: без поправки і з `κ₃+κ₄`.
🔑 ГОЛОВНЕ ЧИСЛО: `G = ‖out(X) − out(T)‖ / ‖вхідне збурення‖` — коефіцієнт
передачі збурення, окремо для формули без поправки і з поправкою.
Вхідне збурення міряємо в тій же одиниці, що й вихід: `‖mp_X − mp_T‖`, бо в
крок `m` і `C` входять лише через `mp = mᵀW` та `s = √diag(WᵀCW)`.

⛔ ПЛОМБИ:
 (а) крок(γ=0) ≡ крок нинішній — РІВНО 0;
 (б) ланцюг стенду ≡ `closure91` (той самий об'єкт, не «за мотивами»);
 (в) на шарі 1 вхід ТОЧНИЙ за побудовою (`C_a[0]=I`, `m_a[0]=0`) ⟹ `A≈шум`;
 (г) шум істини — половинками вибірки;
 (д) A через `C_a` мусить збігтися з A через прямі вибіркові `mh,s`
     (два незалежні шляхи до тієї самої величини).

📣 ПЕРЕДБАЧЕНО: `G_без ≈ 1` (крок майже не підсилює — інакше ланцюг би вибухав),
   а `G_Едж > 1`, бо поправка `−(γ₁/6)·mp·φ` має похідну по `mp`, що додається
   до Φ(t). Якщо так — розрив ×20 пояснено, і напрям «самоузгоджена рекурсія»
   мертвий у нинішньому вигляді: треба не кращі `γ`, а ФОРМУ, стійку до `ε`.
   Якщо ж `G_Едж ≈ G_без` — гіпотеза хибна, розрив десь інде, і тоді наступний
   підозрюваний — кореляція похибок шарів при увімкненій поправці.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/selfcon.py [cuda:N] [NETS] [log2N]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import closure91, _phi, _Phi                     # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LG = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS, BS, L_, W_ = 1 << LG, 1 << 14, 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
print(f'{NETS} мереж × {NS} семплів, {DEV}\n', flush=True)


def step(m, C, w, g1=None, g2=None):
    """ОДИН крок замикання. g1,g2 = None ⟹ рівно те, що в сабміті.

    Повертає (m_out, C_out, mp, s) — mp,s потрібні для виміру збурення входу.
    """
    cp = w.T @ (C @ w)
    s = cp.diagonal().clamp_min(1e-20).sqrt()
    mp = m @ w
    t = mp / s
    ph, Ph = _phi(t), _Phi(t)
    m_o = mp * Ph + s * ph
    d2 = (mp * mp + s * s) * Ph + mp * s * ph          # E[a²] без поправки
    if g1 is not None:
        m_o = m_o - (g1 / 6.0) * mp * ph
        d2 = d2 + (g1 / 3.0) * s * s * ph
    if g2 is not None:
        m_o = m_o + (g2 / 24.0) * s * (t * t - 1.0) * ph
        d2 = d2 - (g2 / 12.0) * s * s * t * ph
    C_o = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s,
                                                                  ph / s)
    dg = d2 - m_o * m_o
    C_o = C_o + torch.diag(dg - C_o.diagonal())
    return m_o, C_o, mp, s


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

# накопичувачі: [мережа, шар]
AF = np.zeros((NETS, L_))     # A: похибка форми, ТОЧНИЙ вхід, без поправки
AE = np.zeros((NETS, L_))     # A: те саме з поправкою κ₃+κ₄
GF = np.zeros((NETS, L_))     # G: передача збурення, без поправки
GE = np.zeros((NETS, L_))     # G: передача збурення, з поправкою
EIN = np.zeros((NETS, L_))    # відносне збурення входу (стан ланцюга)
DF = np.zeros((NETS, L_))     # похибка виходу зі ЗБУРЕНОГО входу, без поправки
DE = np.zeros((NETS, L_))     # те саме з поправкою
RS = np.zeros((NETS, L_))     # шум істини (половинки)
CB = np.zeros((NETS, L_))     # ЛАНЦЮГ нинішній, похибка проти істини пошарово
CE = [np.zeros((NETS, L_)) for _ in range(3)]   # ланцюг із κ₃ / κ₄ / обома
SEAL_G0 = 0.0                 # ⛔(а)
SEAL_CH = np.zeros(NETS)      # ⛔(б)
SEAL_AA = np.zeros(NETS)      # ⛔(д)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        gen = torch.Generator(device=DEV); gen.manual_seed(31337 + n)
        AS = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        AH = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        CA = [torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S1 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S2 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S3 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
        S4 = [torch.zeros(W_, dtype=torch.float64, device=DEV) for _ in range(L_)]
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
                AS[i] += ad.sum(0)
                CA[i] += (a.T @ a).double()     # fp32 матмул, fp64 акумулятор
                if done < NS // 2:
                    AH[i] += ad.sum(0)
            done += b
        m_a = [AS[i] / NS for i in range(L_)]
        C_a = [CA[i] / NS - torch.outer(m_a[i], m_a[i]) for i in range(L_)]
        mah = [AH[i] / (NS // 2) for i in range(L_)]
        mh = [S1[i] / NS for i in range(L_)]
        vh = [(S2[i] / NS - mh[i] ** 2).clamp_min(1e-30) for i in range(L_)]
        k3 = [S3[i] / NS - 3 * mh[i] * (S2[i] / NS) + 2 * mh[i] ** 3
              for i in range(L_)]
        k4 = [S4[i] / NS - 4 * mh[i] * (S3[i] / NS)
              + 6 * mh[i] ** 2 * (S2[i] / NS) - 3 * mh[i] ** 4
              - 3 * vh[i] ** 2 for i in range(L_)]
        G1 = [k3[i] / vh[i] ** 1.5 for i in range(L_)]
        G2 = [k4[i] / vh[i] ** 2 for i in range(L_)]

        # ── ЛАНЦЮГ СТЕНДУ (нинішня формула), зі збереженням стану пошарово
        m_ch = [None] * L_; C_ch = [None] * L_
        m, C = (torch.zeros(W_, dtype=torch.float64, device=DEV),
                torch.eye(W_, dtype=torch.float64, device=DEV))
        for i in range(L_):
            m, C, _, _ = step(m, C, W[i].double())
            m_ch[i], C_ch[i] = m, C
        # ⛔ПЛОМБА (б): ланцюг стенду ≡ closure91
        _, _, _, _, m_deep = closure91(W, L_ - 1, DEV, kdeep=None)
        SEAL_CH[n] = float((m_ch[L_ - 1] - m_deep).norm() / m_deep.norm())

        # ── 🎯 ТРЕТІЙ ІНГРЕДІЄНТ: ПОВНИЙ ланцюг із поправкою на КОЖНОМУ шарі,
        #    γ — ІСТИННІ (з вибірки). Саме те, що дало ×0.65 в edgesamp.py.
        for v, (u1, u2) in enumerate(((G1, None), (None, G2), (G1, G2))):
            m, C = (torch.zeros(W_, dtype=torch.float64, device=DEV),
                    torch.eye(W_, dtype=torch.float64, device=DEV))
            for i in range(L_):
                m, C, _, _ = step(m, C, W[i].double(),
                                  None if u1 is None else u1[i],
                                  None if u2 is None else u2[i])
                CE[v][n, i] = float((m - m_a[i]).norm() / m_a[i].norm())
        for i in range(L_):
            CB[n, i] = float((m_ch[i] - m_a[i]).norm() / m_a[i].norm())

        for i in range(L_):
            w = W[i].double()
            # вхід ТОЧНИЙ (для i=0 він точний за побудовою)
            mT = torch.zeros(W_, dtype=torch.float64, device=DEV) if i == 0 else m_a[i - 1]
            CT = torch.eye(W_, dtype=torch.float64, device=DEV) if i == 0 else C_a[i - 1]
            mX = mT if i == 0 else m_ch[i - 1]
            CX = CT if i == 0 else C_ch[i - 1]
            g1, g2 = G1[i], G2[i]
            oT, _, mpT, sT = step(mT, CT, w)
            oTe, _, _, _ = step(mT, CT, w, g1, g2)
            oX, _, mpX, sX = step(mX, CX, w)
            oXe, _, _, _ = step(mX, CX, w, g1, g2)
            if n == 0 and i == 5:            # ⛔ПЛОМБА (а): γ=0 ≡ без поправки
                z0 = torch.zeros(W_, dtype=torch.float64, device=DEV)
                SEAL_G0 = float((step(mT, CT, w, z0, z0)[0] - oT).abs().max())
            ref = m_a[i]; rn = ref.norm()
            AF[n, i] = float((oT - ref).norm() / rn)
            AE[n, i] = float((oTe - ref).norm() / rn)
            DF[n, i] = float((oX - ref).norm() / rn)
            DE[n, i] = float((oXe - ref).norm() / rn)
            # збурення входу в одиницях кроку: (mp, s) — рівно те, що бачить формула
            din = float(((mpX - mpT).norm() ** 2 + (sX - sT).norm() ** 2).sqrt())
            EIN[n, i] = din / float((mpT.norm() ** 2 + sT.norm() ** 2).sqrt())
            GF[n, i] = float((oX - oT).norm()) / max(din, 1e-300)
            GE[n, i] = float((oXe - oTe).norm()) / max(din, 1e-300)
            RS[n, i] = float((mah[i] - ref).norm() / rn)
            # ⛔ПЛОМБА (д): A через C_a проти A через прямі вибіркові mh,s
            if i > 0:
                sd_ = vh[i].sqrt(); t_ = mh[i] / sd_
                m_dir = mh[i] * _Phi(t_) + sd_ * _phi(t_)
                SEAL_AA[n] = max(SEAL_AA[n],
                                 float((m_dir - oT).norm() / rn) / max(AF[n, i], 1e-30))
    del W, CA, C_a, C_ch
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/selfcon.npz',
         AF=AF, AE=AE, GF=GF, GE=GE, EIN=EIN, DF=DF, DE=DE, RS=RS)

print(f'\n⛔ПЛОМБА (а) крок(γ=0) ≡ крок: max|Δ| = {SEAL_G0:.2e} '
      f'{"✅" if SEAL_G0 == 0 else "🚨"}')
print(f'⛔ПЛОМБА (б) ланцюг стенду ≡ closure91: {SEAL_CH.max():.2e} '
      f'{"✅" if SEAL_CH.max() < 1e-12 else "🚨 стенд веде ІНШИЙ ланцюг"}')
print(f'⛔ПЛОМБА (д) A через C_a проти A через mh,s: {SEAL_AA.max():.3f} '
      f'(відн. до A) {"✅" if SEAL_AA.max() < 0.05 else "🚨 два шляхи розійшлись"}')
print(f'⛔ПЛОМБА (в) шар 1: A = {AF[:,0].mean():.2e} проти шуму '
      f'{RS[:,0].mean():.2e} {"✅" if AF[:,0].mean() < 4*RS[:,0].mean()+1e-12 else "🚨"}')
print(f'⛔ПЛОМБА (г) шум істини, медіана по шарах: {np.median(RS):.2e}\n')

print(f'{"шар":>4s} {"A форма":>10s} {"A+Едж":>10s} {"A/A+Е":>7s} '
      f'{"ε вхід":>9s} {"G без":>8s} {"G Едж":>8s} {"GЕ/Gб":>7s} '
      f'{"D без":>10s} {"D Едж":>10s}')
print('─' * 96)
for i in list(range(0, 8)) + list(range(8, L_, 3)) + [L_ - 1]:
    print(f'{i+1:4d} {AF[:,i].mean():10.3e} {AE[:,i].mean():10.3e} '
          f'{AF[:,i].mean()/max(AE[:,i].mean(),1e-30):7.2f} '
          f'{EIN[:,i].mean():9.2e} {GF[:,i].mean():8.4f} {GE[:,i].mean():8.4f} '
          f'{GE[:,i].mean()/max(GF[:,i].mean(),1e-30):7.4f} '
          f'{DF[:,i].mean():10.3e} {DE[:,i].mean():10.3e}')

print(f'\n🎯 ТРЕТІЙ ІНГРЕДІЄНТ: ПОВНИЙ ЛАНЦЮГ, γ ІСТИННІ, пошарово')
print(f'{"шар":>4s} {"ланцюг ⛔":>11s} {"+κ₃":>11s} {"+κ₄":>11s} '
      f'{"+обидва":>11s} {"×κ₃":>7s} {"×κ₄":>7s} {"×обидва":>8s} {"A форма":>10s}')
print('─' * 92)
for i in list(range(0, 8)) + list(range(8, L_, 3)) + [L_ - 1]:
    b = CB[:, i].mean()
    e = [CE[v][:, i].mean() for v in range(3)]
    print(f'{i+1:4d} {b:11.3e} {e[0]:11.3e} {e[1]:11.3e} {e[2]:11.3e} '
          f'{b/max(e[0],1e-30):7.3f} {b/max(e[1],1e-30):7.3f} '
          f'{b/max(e[2],1e-30):8.3f} {AF[:,i].mean():10.3e}')
bl = CB[:, L_-1].mean()
el = [CE[v][:, L_-1].mean() for v in range(3)]
print(f'\n   НА ВИХОДІ (шар 32): ⛔{bl:.4e} · κ₃ ×{bl/el[0]:.3f} · '
      f'κ₄ ×{bl/el[1]:.3f} · ОБИДВА ×{bl/el[2]:.3f}')

sl = slice(1, L_)
gf, ge = GF[:, sl].mean(), GE[:, sl].mean()
af, ae = AF[:, sl].mean(), AE[:, sl].mean()
print('\n🎯 ВИРОК')
print(f'   ФОРМА на точному вході:  A_без = {af:.3e}  A_Едж = {ae:.3e}  '
      f'⟹ ×{af/max(ae,1e-30):.2f}')
print(f'   ПЕРЕДАЧА збурення:       G_без = {gf:.4f}   G_Едж = {ge:.4f}   '
      f'⟹ ×{ge/max(gf,1e-30):.4f} за крок')
print(f'   за 31 шар це множник {(ge/max(gf,1e-30))**31:.3f} — '
      f'{"🟢 стійкість НЕ гіршає" if ge <= gf*1.002 else "⛔ ПОПРАВКА ПІДСИЛЮЄ ЗБУРЕННЯ"}')
if ge > gf * 1.002 and (ge / gf) ** 31 > 3.0:
    print('   ⟹ 1️⃣ РОЗРИВ ×20 ПОЯСНЕНО: поправка ділить локальну похибку, але')
    print('      множить передачу збурення; за 32 шари друге перемагає.')
    print('      🎓 ЛІКУВАТИ ТРЕБА СТІЙКІСТЬ ФОРМИ ДО ε, А НЕ ДЖЕРЕЛО γ')
    print('      ⟹ рекурсія для κ₄ (самоузгоджена схема) НЕ ЛІКУЄ ЦЬОГО')
else:
    print('   ⟹ 2️⃣ гіпотеза підсилення ХИБНА — розрив ×20 не тут.')
    print('      наступний підозрюваний: КОРЕЛЯЦІЯ похибок шарів при поправці')
    print('      (зміряно, що без поправки вони некорельовані: B = A·√(ℓ−1))')
print(f'\n📊 БАЛ 1.954e-07 · вимір ДІАГНОСТИЧНИЙ, у сабміт нічого не йде')
