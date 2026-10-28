"""ЧИ МОЖНА НЕСТИ НЕГАУСОВІСТЬ ТРЬОМА РЕДУКЦІЯМИ, ЯК ЛІДЕР? (С88, БЕЗ МК)

## ЗВІДКИ — `price88` ЗАКРИВ ГАУСОВУ ГІЛКУ ОДНИМ ЧИСЛОМ
Ряд Прайса до `K=20` дає **×1.001** (24 мережі) ⟹ обрізання відповідає за
**0.1%** похибки, негаусовість — за **99.9%**. Гаусове замикання ми вже
робимо практично точно. Лишається рівно одне: НЕСТИ негаусовість.

## 📐 ВИВІД ПОПРАВКИ (не вгадування)
`h = m + σu`, щільність Еджворта `p(u) = φ(u)[1 + Σ_k c_k He_k(u)]`.
```
E[ReLU(h)] = σ∫_{−t}^∞ (t+u)φ(u)[1 + Σ c_k He_k(u)] du
I_k(t) := ∫_{−t}^∞ (t+u)He_k(u)φ(u) du
```
Через `He_k φ = (−1)^k φ^(k)` та інтегрування частинами (крайовий член зникає,
бо `(t+u)|_{u=−t} = 0`):
```
🔑 I_k(t) = (−1)^k · He_{k−2}(t) · φ(t)      (k ≥ 2)
```
Перевірка: `I₂ = φ`, `I₃ = −tφ`, `I₄ = (t²−1)φ`. З `c₃ = γ₃/6`, `c₄ = γ₄/24`:
```
🔑 E[ReLU] = σ(tΦ + φ) + σφ(t)·[ −γ₃·t/6 + γ₄·(t²−1)/24 ]
```
Абсолютна поправка несе множник `φ(t)`, тож при `|t|→3` вона САМА гасне —
на відміну від Еджворта для щільності, який там розходиться. 📌 Це знімає
заперечення «Еджворт розходиться при t=3» ДЛЯ ЦЬОГО функціонала.

## 📐 РЕКУРСІЯ КУМУЛЯНТІВ = ТРИ РЕДУКЦІЇ (гіпотеза лідера)
Якщо знехтувати залежністю `a_j`, то
```
κ_r(h_i) = Σ_j W_ji^r · κ_r(a_j)      r = 2,3,4
```
— рівно ТРИ редукції довжини 256 на нейрон на шар, тобто відбиток `F` лідера.
`κ_r(a_j)` беремо з замикання: моменти обрізаного нормального
`M₁=mΦ+σφ` · `M₂=(m²+σ²)Φ+mσφ` · `M₃=(m³+3mσ²)Φ+(m²+2σ²)σφ` ·
`M₄=(m⁴+6m²σ²+3σ⁴)Φ+(m³+5mσ²)σφ`, далі `κ₃=M₃−3M₁M₂+2M₁³`,
`κ₄=M₄−4M₁M₃−3M₂²+12M₁²M₂−6M₁⁴`.

## 🎯 ЩО МІРЯЄТЬСЯ
 (1) ЯКІСТЬ наближення: істинний `κ₃(h)` проти `Σ_j W_ji³ κ₃(a_j)` — корел
     і відносна похибка по шарах. Це ключ: якщо наближення нікчемне, гілка
     мертва незалежно від того, як добре працює поправка.
 (2) ЛАНЦЮГ: база · діагональні `κ` · ОРАКУЛЬНІ `κ` (стеля).
 (3) 🚨 РІЗНИЦЯ (2)↔(3) = скільки коштує саме наближення незалежності.

## ⛔ ПЛОМБИ
 P1 база = наш ланцюг **0.9416%** (`price88`, 24 мережі, незалежно).
 P2 🔑АНАЛІТИЧНА: на шарі 1 вхід ТОЧНО гаусів ⟹ істинний `κ₃(h⁽¹⁾) = 0`.
    Міряємо і друкуємо — мусить бути на рівні шуму МК.
 P3 🔑РЕЖИМ ОБ'ЄКТА: якщо подати ДІАГОНАЛЬНУ `C` (незалежні `a_j`), формула
    `κ_r(h)=Σ W^r κ_r(a)` стає ТОЧНОЮ ⟹ відносна похибка мусить впасти до
    рівня шуму. Без цієї пломби «погане наближення» не відрізнити від бага.
 P4 оракульні `κ₃,κ₄` з поправкою мусять дати ≈ `L5` = 0.0162% (запис С87).
    ⚠️ Якщо дадуть суттєво гірше — або `L5` міряв інший об'єкт, або поправка
    неповна (потрібні вищі порядки).
 P5 шум МК: дві незалежні половини.

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ
 A. Наближення незалежності буде ПОГАНИМ: віднос. похибка `κ₃(h)` &gt; 50%.
    Шанс **70%**. Підстава: `a_j` сильно залежні (еф. ранг падає до 3.1).
 B. Оракульні `κ` дадуть 0.02–0.15%, тобто ×6…×45. Шанс **60%**.
 C. Діагональні `κ` дадуть ×1.0–1.3, тобто майже нічого. Шанс **65%**.
    ⟹ якщо A і C справдяться, гіпотеза «лідер = три редукції з кумулянтами»
    ПОМИРАЄ, і його `F` пояснюється чимось іншим.
 D. 🎯 Якщо C дасть ×3 і більше — це прорив і пряма дорога до сабміта.
    Шанс **15%**.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = (0, 4, 8, 12, 16, 20, 24, 28, 31)
assert DEV not in ('cuda:0', 'cuda:4'), 'cuda:0 ТАБУ, cuda:4 — монітор юзера'
print(__doc__.split('📣')[1], flush=True)

I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV}\n', flush=True)


def relu_cums(m, s):
    """`κ₂,κ₃,κ₄` активації `ReLU(N(m,s²))` — замкнені формули."""
    t = m / s
    P, f = Phi(t), phi(t)
    s2 = s * s
    M1 = m * P + s * f
    M2 = (m * m + s2) * P + m * s * f
    M3 = (m ** 3 + 3 * m * s2) * P + (m * m + 2 * s2) * s * f
    M4 = (m ** 4 + 6 * m * m * s2 + 3 * s2 * s2) * P + (m ** 3 + 5 * m * s2) * s * f
    k2 = M2 - M1 * M1
    k3 = M3 - 3 * M1 * M2 + 2 * M1 ** 3
    k4 = M4 - 4 * M1 * M3 - 3 * M2 * M2 + 12 * M1 * M1 * M2 - 6 * M1 ** 4
    return M1, k2, k3, k4


def step(m, C, k3=None, k4=None):
    """Замикання + (опційно) поправка від кумулянтів. Діагональ — точна формула."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    if k3 is not None:
        g3 = k3 / (s ** 3)
        g4 = k4 / (s ** 4)
        m_a = m_a + s * f * (-(g3 * t) / 6.0 + (g4 * (t * t - 1.0)) / 24.0)
    Ca = C * torch.outer(P, P) + 0.5 * (C * C) * torch.outer(f / s, f / s)
    d = (m * m + s * s) * P + (m * s) * f - (m * P + s * f) ** 2
    Ca = Ca + torch.diag(d - torch.diagonal(Ca))
    return m_a, Ca, s, t


MODES = ('база', 'діаг κ', 'оракул κ')
NM = len(MODES)
ERR = np.zeros((len(FS), NM))
ERRH = np.zeros((len(FS), NM, 2))
NP = len(PROBE)
K3REL = np.zeros((len(FS), NP))     # віднос. похибка діаг. наближення κ₃(h)
K3COR = np.zeros((len(FS), NP))     # кореляція
K3TRU = np.zeros((len(FS), NP))     # RMS істинного γ₃ = κ₃/s³
P3REL = np.zeros((len(FS), NP))     # ⛔P3: те саме, але з ДІАГОНАЛЬНОЮ C
P3SCL = np.zeros((len(FS), NP))     # ⛔P3b: відношення похибок при N і 4N (~2 = шум)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    with torch.no_grad():
        # ---------- ІСТИНА: моменти преактивацій і активацій по шарах ----------
        S1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        S3 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        S4 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        A1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        A2 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        A3 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        A4 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(2, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(6600 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            hlf = 0 if d < NS // 2 else 1
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ W32[q].float()).double()
                h2 = h * h
                S1[q] += h.sum(0); S2[q] += h2.sum(0)
                S3[q] += (h2 * h).sum(0); S4[q] += (h2 * h2).sum(0)
                ad = torch.clamp(h, min=0.0); a2 = ad * ad
                A1[q] += ad.sum(0); A2[q] += a2.sum(0)
                A3[q] += (a2 * ad).sum(0); A4[q] += (a2 * a2).sum(0)
                a = torch.relu(h).float()
            SA[hlf] += a.double().sum(0)
            d += b

        def cums(s1, s2, s3, s4):
            m1 = s1 / NS; m2 = s2 / NS; m3 = s3 / NS; m4 = s4 / NS
            c2 = m2 - m1 * m1
            c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
            c4 = m4 - 4 * m1 * m3 - 3 * m2 * m2 + 12 * m1 * m1 * m2 - 6 * m1 ** 4
            return m1, c2, c3, c4

        a_true = SA.sum(0) / NS
        a_h = SA * (2.0 / NS)
        MH, KH2, KH3, KH4 = cums(S1, S2, S3, S4)      # преактивації (істина)
        _, _, KA3, KA4 = cums(A1, A2, A3, A4)         # активації   (істина)
        for j, q in enumerate(PROBE):
            if q == 0:
                K3TRU[n, j] = float(torch.sqrt(((KH3[0] / KH2[0] ** 1.5) ** 2).mean()))
                K3REL[n, j] = float('nan'); K3COR[n, j] = float('nan')
                continue
            w3 = W32[q] ** 3
            appr = KA3[q - 1] @ w3
            tru = KH3[q]
            K3REL[n, j] = float(torch.sqrt(((appr - tru) ** 2).sum()
                                           / (tru * tru).sum()))
            K3COR[n, j] = float(torch.corrcoef(torch.stack([appr, tru]))[0, 1])
            K3TRU[n, j] = float(torch.sqrt(((tru / KH2[q] ** 1.5) ** 2).mean()))

        # ---------- ⛔P3 РЕЖИМ ОБ'ЄКТА: незалежні `a_j` ----------
        for j, q in enumerate(PROBE):
            if q == 0:
                P3REL[n, j] = float('nan'); continue
            # 🚨 ПЛОМБА МАСШТАБУВАННЯ ВСЕРЕДИНІ ПЛОМБИ: рахуємо при N і 4N.
            #    Якщо розбіжність падає вдвічі — це ШУМ виміру `κ₃`, а не
            #    похибка формули (закон: пломба мусить ЖИТИ В РЕЖИМІ ОБ'ЄКТА).
            sd = torch.sqrt(torch.clamp(KH2[q - 1], min=1e-30))
            res = []
            for NI in (1 << 16, 1 << 18):
                gg = torch.Generator(device=DEV); gg.manual_seed(77000 + n * 64 + q)
                acc = torch.zeros(4, W_, dtype=torch.float64, device=DEV)
                accA = torch.zeros(3, W_, dtype=torch.float64, device=DEV)
                dn2 = 0
                while dn2 < NI:
                    bb = min(1 << 14, NI - dn2)
                    zz = torch.randn(bb, W_, device=DEV, generator=gg,
                                     dtype=torch.float32)
                    aa = torch.clamp(zz.double() * sd + MH[q - 1], min=0.0)
                    hh = aa @ W32[q]
                    acc[0] += hh.sum(0); acc[1] += (hh * hh).sum(0)
                    acc[2] += (hh ** 3).sum(0); acc[3] += (hh ** 4).sum(0)
                    accA[0] += aa.sum(0); accA[1] += (aa * aa).sum(0)
                    accA[2] += (aa ** 3).sum(0)
                    dn2 += bb
                    del zz, aa, hh
                p1, p2_, p3_ = accA[0] / NI, accA[1] / NI, accA[2] / NI
                c3a = p3_ - 3 * p1 * p2_ + 2 * p1 ** 3
                ap = c3a @ (W32[q] ** 3)
                q1, q2, q3 = acc[0] / NI, acc[1] / NI, acc[2] / NI
                c3h = q3 - 3 * q1 * q2 + 2 * q1 ** 3
                res.append(float(torch.sqrt(((ap - c3h) ** 2).sum()
                                            / (c3h * c3h).sum())))
            P3REL[n, j] = res[1]
            P3SCL[n, j] = res[0] / max(res[1], 1e-12)   # ~2.0 ⟹ чистий шум

        # ---------- ЛАНЦЮГИ ----------
        for mi, mode in enumerate(MODES):
            m = torch.zeros(W_, dtype=torch.float64, device=DEV)
            C = torch.eye(W_, dtype=torch.float64, device=DEV)
            c3 = torch.zeros(W_, dtype=torch.float64, device=DEV)
            c4 = torch.zeros(W_, dtype=torch.float64, device=DEV)
            for q in range(L_):
                w = W32[q]
                cp = w.T @ (C @ w); cp = 0.5 * (cp + cp.T)
                mp = m @ w
                if mi == 0:
                    m, C, s, t = step(mp, cp)
                elif mi == 1:
                    m, C, s, t = step(mp, cp, c3, c4)
                else:
                    m, C, s, t = step(mp, cp, KH3[q], KH4[q])
                if mi == 1 and q < L_ - 1:
                    _, ka2, ka3, ka4 = relu_cums(mp, torch.sqrt(
                        torch.clamp(torch.diagonal(cp), min=1e-30)))
                    w3 = W32[q + 1] ** 3; w4 = W32[q + 1] ** 4
                    c3 = ka3 @ w3; c4 = ka4 @ w4
            ERR[n, mi] = float(torch.sqrt(((m - a_true) ** 2).sum()
                                          / (a_true ** 2).sum()))
            for hh_ in range(2):
                ERRH[n, mi, hh_] = float(torch.sqrt(((m - a_h[hh_]) ** 2).sum()
                                                    / (a_h[hh_] ** 2).sum()))
    print(f'  net{n}  база {ERR[n,0]:.4%} · діаг κ {ERR[n,1]:.4%} · '
          f'оракул κ {ERR[n,2]:.4%}', flush=True)

print(f'\n{"=" * 76}\n⛔ ПЛОМБИ')
print(f' P1  база                  = {ERR[:,0].mean():.4%}  (price88 дав 0.9416%)')
print(f' P2  γ₃ істинний на шарі 1 = {np.nanmean(K3TRU[:,0]):.5f}  '
      f'(мусить бути ~0: вхід ТОЧНО гаусів)')
print(f' P3  режим об\'єкта (незалежні a) = {np.nanmean(P3REL[:,1:]):.4%} · масштаб N→4N дає ×{np.nanmean(P3SCL[:,1:]):.2f} '
      f'(мусить впасти до шуму)')
print(f' P4  оракул κ              = {ERR[:,2].mean():.4%}  (L5 з С87: 0.0162%)')
print(f' P5  розбіжність половин   = {np.abs(ERRH[:,:,0]-ERRH[:,:,1]).mean():.4%}')

print(f'\n{"=" * 76}\n📈 ЯКІСТЬ ДІАГОНАЛЬНОГО НАБЛИЖЕННЯ `κ₃(h)=Σ W³κ₃(a)`')
print(f'{"шар":>5s} {"віднос.похиб":>13s} {"корел":>8s} {"RMS γ₃ істини":>14s} '
      f'{"⛔P3 незалежні":>15s}')
for j, q in enumerate(PROBE):
    if q == 0:
        print(f'{q+1:5d} {"—":>13s} {"—":>8s} {np.nanmean(K3TRU[:,j]):14.5f} '
              f'{"—":>15s}')
        continue
    print(f'{q+1:5d} {K3REL[:,j].mean():13.2%} {K3COR[:,j].mean():8.4f} '
          f'{K3TRU[:,j].mean():14.5f} {P3REL[:,j].mean():15.2%}')

print(f'\n{"=" * 76}\n📈 ЛАНЦЮГ ({len(FS)} мереж)')
b = ERR[:, 0].mean()
for mi, mode in enumerate(MODES):
    print(f'{mode:>12s} {ERR[:,mi].mean():10.4%}  ×{b/max(ERR[:,mi].mean(),1e-12):.3f}')
print(f'\n📣 ВИРОК')
print(f' C. діагональні κ: ×{b/max(ERR[:,1].mean(),1e-12):.3f} '
      f'(передбачав ×1.0–1.3, прорив якщо ≥×3)')
print(f' B. оракульні κ:   ×{b/max(ERR[:,2].mean(),1e-12):.3f}')
print(f' 🎯 пороги: рекорд 0.1456% · лідер 0.0760%')
