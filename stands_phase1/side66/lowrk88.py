"""ЧИ МОЖНА ВЕСТИ `C = D + низький ранг` І НЕ ВТРАТИТИ ТОЧНІСТЬ? (С88, БЕЗ МК)

## ЗВІДКИ — ТРИ НЕЗАЛЕЖНІ ВИМІРИ ЗІЙШЛИСЯ В ОДНУ ТОЧКУ
1. 🆕`tprof88` (32 мережі, 2¹⁸): **ефективний ранг `tr(C)²/tr(C²)` падає
   127.6 → 3.1** по 32 шарах. Тобто на глибині дисперсія живе у 3 напрямках.
2. 🏆ВІДБИТОК ЛІДЕРА: `F` = «ТРИ редукції довжини 256 на нейрон на шар»
   (пришпилено 1.6–3.4%), жодного матмула 256³.
   📐 `3 × 256 × 256 × 2 × 32 шари = 1.26e7` проти його **1.30e7** — збіг 3%.
3. 📜НАШ ЕСТИМАТОР уже вписує ДІАГОНАЛЬ точною формулою (`_closure_parts`),
   і саме вона тримається: діаг 1.11% проти позадіаг 4.45% (шар 32).

⟹ ГІПОТЕЗА: лідер веде `C = D + VVᵀ`, `V` — 256×3. Тоді перенос дешевий:
   `Cov(h) = WᵀDW + (WᵀV)(WᵀV)ᵀ` — одна редукція на діагональ (через `W²`)
   плюс ТРИ матвеки. Це рівно його `F`.

## 🚨 ЧОМУ ПОПЕРЕДНІЙ ВИМІР НИЗЬКОГО РАНГУ НЕ РАХУЄТЬСЯ
С88 (агент): `rank-1 → 17.7%`, `rank-4 → 52.7%`, `rank-16 → 185%` — «тим
гірший, чим вищий ранг». Але в його ж застереженні: «різав ЛИШЕ позадіагональ,
**PSD ламається**». Це не вимір низького рангу — це вимір ЗЛАМАНОЇ матриці
(третя хвороба контролю: НЕ ТОЙ ОБ'ЄКТ). Монотонність «гірше з ростом рангу»
і є підписом зламаного PSD: чим більше рангу, тим більше від'ємних напрямків.

## 🔬 ПРАВИЛЬНИЙ ІНСТРУМЕНТ (зберігає І діагональ, І PSD)
```
R = D^{-1/2} C D^{-1/2}          # кореляційна, diag(R) ≡ 1
R_r = Σ_{i≤r} λ_i v_i v_iᵀ       # r головних компонент
R̂  = R_r + diag(1 − diag(R_r))   # ДІАГОНАЛЬ ПОВЕРТАЄМО В 1
C̃  = D^{1/2} R̂ D^{1/2}          # діагональ ТОЧНА за побудовою
```
Це і є «точна діагональ + позадіагональ рангу `r`».

## ⛔ ПЛОМБИ НА ОБОХ КРАЯХ (інструмент перевіряє СЕБЕ до висновків)
 P1 `r = 256` ⟹ `R_r = R` ⟹ поправка діагоналі тотожно нуль ⟹ `C̃ = C`
    РІВНО ⟹ похибка мусить збігтися з ПОВНИМ ланцюгом до 1e-6.
 P2 `r = 0`   ⟹ `R̂ = I` ⟹ `C̃ = D` = чисте ДІАГОНАЛЬНЕ поширення.
    Незалежний вимір С88 дав для нього **4.02%** — мусимо відтворити.
 P3 діагональ `C̃` збігається з `C` до 1e-12 при БУДЬ-ЯКОМУ `r` (за побудовою).
 P4 мінімальне власне значення `R̂` ≥ 0 — рахуємо і друкуємо частку зривів.
    Якщо PSD ламається — це видно ЧИСЛОМ, а не здогадкою.
 P5 шум МК-істини: половинки вибірки дають ту саму похибку.

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (щоб не тлумачити заднім числом)
 A. P1 і P2 пройдуть ⟹ інструмент чесний. Шанс **85%**.
 B. `r = 3` дасть **1.5–4%**, тобто ГІРШЕ за нашу повну коваріацію 0.974%.
    Шанс **80%**. Підстава: наш власний закон — «негаусовість живе у ХВОСТІ
    спектру, домінантні компоненти за ЦГТ майже гаусові». Ранг 3 зберігає
    дисперсію і викидає рівно те, що потрібне.
 C. Крива по `r` буде МОНОТОННОЮ (більший ранг — краще). Шанс **75%**.
    ⚠️ Якщо НЕ монотонна — інструмент усе ще зламаний, і це видно по P4.
 D. 🎯 ЯКЩО `r=3` дасть &lt; 0.974% — гіпотеза про лідера підтверджена, і це
    міняє все. Шанс **20%**, але ціна перевірки — один прогін.
 📌 У БУДЬ-ЯКОМУ РАЗІ дізнаємось, скільки рангу треба, щоб не втратити
    точність — а це прямий гейт ціни для будь-якої векторної архітектури.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
RANKS = (0, 1, 2, 3, 4, 6, 8, 16, 32, 64, 256)
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
print(f'{len(FS)} мереж · 2^{LOG2N} · {DEV} · ранги {RANKS}\n', flush=True)


def closure(m, C):
    """Крок замикання — ЗВІРЕНО з `estimator_sub97.py:_closure_parts` рядок у рядок:
    позадіагональ — ряд Мелера K=2, ДІАГОНАЛЬ вписується ТОЧНОЮ формулою."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    Ca = C * torch.outer(P, P) + 0.5 * (C * C) * torch.outer(f / s, f / s)
    d = (m * m + s * s) * P + (m * s) * f - m_a * m_a
    Ca = Ca + torch.diag(d - torch.diagonal(Ca))
    return m_a, Ca


def truncate(C, r):
    """`C → D^{1/2}(R_r + diag(1−diag R_r))D^{1/2}`. Діагональ ТОЧНА за побудовою.
    Повертає (C̃, мін.власне значення R̂) — друге для пломби P4."""
    if r >= W_:
        return C, 1.0
    d = torch.clamp(torch.diagonal(C), min=1e-30)
    inv = 1.0 / torch.sqrt(d)
    R = C * torch.outer(inv, inv)
    if r == 0:
        return torch.diag(d), 1.0
    ev, U = torch.linalg.eigh(R)                    # зростаюче
    Ur = U[:, -r:]; ev_r = torch.clamp(ev[-r:], min=0.0)
    Rr = (Ur * ev_r) @ Ur.T
    Rh = Rr + torch.diag(1.0 - torch.diagonal(Rr))
    mn = float(torch.linalg.eigvalsh(Rh)[0])
    sq = torch.sqrt(d)
    return Rh * torch.outer(sq, sq), mn


NR = len(RANKS)
ERR = np.zeros((len(FS), NR))        # віднос. похибка E[a] ОСТАННЬОГО шару
ERRH = np.zeros((len(FS), NR, 2))    # P5: по двох половинах МК-істини
DIAG = np.zeros((len(FS), NR))       # P3: розбіжність діагоналі
PSD = np.zeros((len(FS), NR))        # P4: мін. власне значення R̂ по шарах
EFR = np.zeros((len(FS), L_))        # еф.ранг ІСТИННОЇ C_a (звірка з tprof88)

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    with torch.no_grad():
        # ---------- ІСТИНА: МК на льоту, дві незалежні половини ----------
        SA = torch.zeros(2, W_, dtype=torch.float64, device=DEV)
        SAA = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        SA_L = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(4400 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            hlf = 0 if d < NS // 2 else 1
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                a = torch.relu(a @ W32[q].float())
                ad = a.double()
                SA_L[q] += ad.sum(0)
                SAA[q] += (a.T @ a).double()
                if q == L_ - 1:
                    SA[hlf] += ad.sum(0)
            d += b
        a_true = SA_L[L_ - 1] / NS
        a_h = SA * (2.0 / NS)
        for q in range(L_):
            mq = SA_L[q] / NS
            Cq = SAA[q] / NS - torch.outer(mq, mq)
            t1 = float(torch.diagonal(Cq).sum()); t2 = float((Cq * Cq).sum())
            EFR[n, q] = t1 * t1 / max(t2, 1e-30)
        del SAA
        # ---------- ЛАНЦЮГИ З УСІЧЕННЯМ ----------
        for k, r in enumerate(RANKS):
            m = torch.zeros(W_, dtype=torch.float64, device=DEV)
            C = torch.eye(W_, dtype=torch.float64, device=DEV)
            mn_all = 1.0; dmax = 0.0
            for q in range(L_):
                w = W32[q]
                cp = w.T @ (C @ w)
                cp = 0.5 * (cp + cp.T)
                mp = m @ w
                m, C = closure(mp, cp)
                if q < L_ - 1:
                    d0 = torch.diagonal(C).clone()
                    C, mn = truncate(C, r)
                    mn_all = min(mn_all, mn)
                    dmax = max(dmax, float((torch.diagonal(C) - d0).abs().max()))
            ERR[n, k] = float(torch.sqrt(((m - a_true) ** 2).sum()
                                         / (a_true ** 2).sum()))
            for h in range(2):
                ERRH[n, k, h] = float(torch.sqrt(((m - a_h[h]) ** 2).sum()
                                                 / (a_h[h] ** 2).sum()))
            PSD[n, k] = mn_all; DIAG[n, k] = dmax
    print(f'  net{n}  r=0 {ERR[n,0]:.4%} · r=3 {ERR[n,3]:.4%} · '
          f'r=256 {ERR[n,-1]:.4%}', flush=True)

print(f'\n{"=" * 78}\n⛔ ПЛОМБИ')
print(f' P1  r=256 (повна C)        = {ERR[:,-1].mean():.4%}   '
      f'(мусить = наш повний ланцюг ≈0.97%)')
print(f' P2  r=0 (лише діагональ)   = {ERR[:,0].mean():.4%}   '
      f'(незалежний вимір С88: 4.02%)')
print(f' P3  max |Δdiag| по всіх r  = {DIAG.max():.3e}   (мусить бути ~1e-12)')
print(f' P4  min λ(R̂) по всіх r,шарах= {PSD.min():+.4f}   (мусить бути ≥0)')
sh = np.abs(ERRH[:, :, 0] - ERRH[:, :, 1]).mean()
print(f' P5  розбіжність половин МК = {sh:.4%}   (шум істини)')
print(f' P6  еф.ранг: шар1 {EFR[:,0].mean():.1f} → шар32 {EFR[:,-1].mean():.1f}   '
      f'(tprof88 дав 127.6 → 3.1)')

print(f'\n{"=" * 78}\n📈 ПОХИБКА `E[a_last]` ЗАЛЕЖНО ВІД РАНГУ ПОЗАДІАГОНАЛІ '
      f'({len(FS)} мереж)')
print(f'{"ранг":>6s} {"похибка":>10s} {"×повної":>9s} {"min λ(R̂)":>10s} '
      f'{"вартість F":>12s} {"C/B":>8s}')
base = ERR[:, -1].mean()
for k, r in enumerate(RANKS):
    # гейт ціни: діагональ через W² (1 редукція) + r матвеків, ×32 шари, ×2 флопи
    fl = (1 + max(r, 1)) * W_ * W_ * 2 * L_ if r < W_ else 2 * W_ ** 3 * L_
    print(f'{r:6d} {ERR[:,k].mean():10.4%} {base/max(ERR[:,k].mean(),1e-12):9.3f} '
          f'{PSD[:,k].min():10.4f} {fl:12.3e} {fl/2.72e11:8.5f}')

print(f'\n{"=" * 78}\n📣 ВИРОК')
i3 = RANKS.index(3)
print(f' B. r=3 дає {ERR[:,i3].mean():.4%} проти повної {base:.4%} '
      f'⟹ ×{base/max(ERR[:,i3].mean(),1e-12):.3f}')
mono = all(ERR[:, k].mean() >= ERR[:, k + 1].mean() - 1e-9 for k in range(NR - 1))
print(f' C. крива монотонна: {mono}')
good = [r for k, r in enumerate(RANKS) if ERR[:, k].mean() <= base * 1.05]
print(f' 🎯 мінімальний ранг у межах +5% від повної: '
      f'{min(good) if good else "НЕ ДОСЯГНУТО"}')
