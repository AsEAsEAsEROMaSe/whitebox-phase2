"""ПРОФІЛЬ `|t| = |m|/σ` ПО ГЛИБИНІ + ЕФЕКТИВНИЙ РАНГ + СВІЖА ПОХИБКА ПО ШАРАХ (С88)

## ЗВІДКИ — ТРИ ФАКТИ, ЯКІ ДОСІ НЕ ЗВОДИЛИ РАЗОМ
1. 📐 ІДЕЯ ЮЗЕРА/КОМЕНТАТОРА (аналітично): на 2-му шарі `μ/σ ≈ 0.54`.
   `E[h⁽¹⁾] = 1/√π = 0.5642` (бо `Var(z⁽¹⁾)=2` при He) ⟹ `μ⁽²⁾ = (1/√π)·W⁽²⁾·1`
   має std `√2/√π = 0.798`; `E[(z⁽²⁾)²] ≈ 2` ⟹ `Var = 2 − 0.64 = 1.36`, `σ=1.17`.
2. 📜 НАШ ВИМІР (`stepcov87`, ГЛИБОКІ шари): `|m/σ| = 3.00`, розподіл `t`
   ДВОМОДАЛЬНИЙ ~±3, `Var(a)/Var(p) = 0.48`.
   ⟹ 🔑 `|t|` РОСТЕ ВІД ~0.55 ДО ~3 — У 5×. ПРОФІЛЮ ПО ШАРАХ МИ НЕ МІРЯЛИ.
3. 🆕 ВИМІР С88 (`calib88w`): `γ` на шарах 1..8 знижує СИРУ похибку на 40%
   (0.854%→0.511%), але після `α` дає **РІВНО ×1.000**; шари 9..32 — **×1.333**.
   ⟹ «ГРОШІ ЖИВУТЬ У ГЛИБИНІ», але МЕХАНІЗМУ ЦЬОМУ В НАС НЕМА.

## 🎯 ГІПОТЕЗА, ЯКУ ПЕРЕВІРЯЄМО
На мілких шарах `|t| ≈ 0.5` ⟹ гаусове замикання там майже точне (виправляти
нічого, і будь-яка «поправка» вироджується в масштаб, який забирає `α`).
Похибка вмикається там, де `|t|` виходить у робочу зону.
⟹ ПРОФІЛЬ `|t|(l)` МУСИТЬ КОРЕЛЮВАТИ З ПРОФІЛЕМ СВІЖОЇ ПОХИБКИ `err(l)`.

## 🔬 ДРУГЕ ПИТАННЯ (коментатор): ЦГТ ПРАЦЮЄ ПО ЕФЕКТИВНОМУ РАНГУ, НЕ ПО ШИРИНІ
`z_i = Σ_j W_ij h_j` — сума 256 доданків, але НЕ 256 незалежних (усі `h_j` —
функції того самого входу). Кількість незалежних внесків ≈ `tr(C)²/tr(C²)`.
Якщо ефективний ранг падає з глибиною — негаусовість НЕ зникає, і заперечення
«ЦГТ усе згладить» знімається. Міряємо прямо.

## ⛔ ПЛОМБИ (перевіряються ДО висновків)
 P1 ШАР 1: `m ≡ 0` тотожно (вхід центрований) ⟹ `RMS|t| < 0.02` при 2^19.
    І `E[a⁽¹⁾] = σ/√(2π) = 0.5642` при `σ²=2` — ТОЧНЕ число, не підгонка.
 P2 ШАР 2: `RMS(t) ≈ 0.55` (аналітичне передбачення вище, незалежне від коду).
 P3 ЕФЕКТИВНИЙ РАНГ шару 1 = **128**, а НЕ 256.
    🚨 МОЯ ПОМИЛКА В ПЕРШІЙ ВЕРСІЇ: я написав «256, бо C⁽¹⁾=2·I». Це НЕПРАВДА.
    `Cov(z⁽¹⁾) = W⁽¹⁾ᵀW⁽¹⁾` — це ВІШАРТ, а не одинична матриця. Для квадратної
    (`λ = n/m = 1`) спектр Марченка–Пастура дає `tr(C)²/tr(C²) = n/(1+λ) = 128`.
    Димовий тест дав **127.06** ⟹ пломба ПРОЙДЕНА, помилка була в передбаченні,
    а не в коді. 📌 Записано як приклад: пломба зловила МЕНЕ, а не програму.
 P4 ШУМ: половинки вибірки дають ту саму криву; при N×4 шум падає вдвічі,
    справжній профіль стоїть.
 P5 СВІЖА похибка: на вхід кроку подаються ІСТИННІ `m,C` цього шару
    (НЕ накопичені) — інакше міряємо накопичення, а не крок (пастка С87, ×0.009).

## 📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (щоб не тлумачити заднім числом)
 A. `RMS|t|` монотонно росте з глибиною і виходить на плато ~2.5–3.5
    після ~10-го шару. Шанс **70%**.
 B. Профіль свіжої похибки НЕ плоский: на шарах 1..5 вона в рази менша,
    ніж на 20..32. Шанс **65%**. ⟹ це і буде механізм «грошей у глибині».
 C. Ефективний ранг падає з 256 до десятків. Шанс **55%**.
    ⚠️ Якщо він лишається великим (>100) — заперечення ЦГТ ЗМІЦНЮЄТЬСЯ, і це
    аргумент ПРОТИ CF-гілки, а не за неї. Готовий прийняти обидва результати.
 D. 🎯 ГОЛОВНЕ: кореляція між `|t|(l)` і `err(l)` по шарах буде > 0.8.
    Шанс **50%**. Якщо ні — «гроші в глибині» мають іншу причину.
"""
import glob, math, os, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = tuple(range(0, 32, 4)) + (31,)          # 9 шарів із ПОВНОЮ коваріацією
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
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV} · probe={PROBE}\n', flush=True)


def closure(m, C):
    """Гаусове замикання кроку: з `(m,C)` преактивацій дати `E[a]` і `Cov(a)`.

    🚨 ЗВІРЕНО З `estimator_sub97.py:_closure_parts` РЯДОК У РЯДОК.
    Перша версія цього стенду рахувала ДІАГОНАЛЬ рядом Мелера — а естиматор
    вписує її ТОЧНОЮ формулою (`ρ=1` — саме там ряд збігається найгірше).
    Це був контроль «НЕ ТОЙ ОБ'ЄКТ»: `err діаг` виміряла похибку формули, якої
    в нашому ланцюгу НЕМАЄ (давала 3.3% на шарі 1, де вхід ТОЧНО гаусів).
    📌 Профіль `|t|` та ефективний ранг цим НЕ зачеплені — вони міряють ІСТИНУ.
    """
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    Ca = C * torch.outer(P, P) + 0.5 * (C * C) * torch.outer(f / s, f / s)
    # ← точна діагональ, як в естиматорі
    d = (m * m + s * s) * P + (m * s) * f - m_a * m_a
    Ca = Ca + torch.diag(d - torch.diagonal(Ca))
    return m_a, Ca


NPB = len(PROBE)
TRMS = np.zeros((len(FS), L_))       # RMS(t) по всіх 32 шарах
TABS = np.zeros((len(FS), L_))       # mean|t|
TPOS = np.zeros((len(FS), L_))       # частка t>0
EFR = np.zeros((len(FS), NPB))       # ефективний ранг tr(C)²/tr(C²)
ERRM = np.zeros((len(FS), NPB))      # СВІЖА відносна похибка E[a]
ERRC = np.zeros((len(FS), NPB))      # СВІЖА відносна похибка Cov(a), позадіаг.
ERRD = np.zeros((len(FS), NPB))      # СВІЖА відносна похибка діагоналі Cov(a)
HALF = np.zeros((len(FS), L_, 2))    # P4: RMS(t) по двох половинах вибірки
SEAL = np.zeros((len(FS), 3))        # P1 RMS|t| шару1 · P1b E[a1] · P3 ранг шару1

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    with torch.no_grad():
        SH = torch.zeros(2, L_, W_, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(2, L_, W_, dtype=torch.float64, device=DEV)
        SHH = torch.zeros(NPB, W_, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(NPB, W_, dtype=torch.float64, device=DEV)
        SAA = torch.zeros(NPB, W_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(3300 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            hlf = 0 if d < NS // 2 else 1
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                hd = h.double()
                SH[hlf, q] += hd.sum(0)
                S2[hlf, q] += (hd * hd).sum(0)
                a = torch.relu(h)
                if q in PROBE:
                    j = PROBE.index(q)
                    # 🚨 матмул у fp32 (P104 у fp64 повільніша в 32×), накопичення у fp64.
                    # Похибка fp32-редукції по 8192 елементах ~1e-6 ≪ вимірюваних ~1%.
                    SHH[j] += (h.T @ h).double()
                    SA[j] += a.double().sum(0)
                    SAA[j] += (a.T @ a).double()
            d += b
        for q in range(L_):
            for k in range(2):
                nn = NS // 2
                m = SH[k, q] / nn
                v = torch.clamp(S2[k, q] / nn - m * m, min=1e-30)
                HALF[n, q, k] = float(torch.sqrt(((m / torch.sqrt(v)) ** 2).mean()))
            m = SH[:, q].sum(0) / NS
            v = torch.clamp(S2[:, q].sum(0) / NS - m * m, min=1e-30)
            t = m / torch.sqrt(v)
            TRMS[n, q] = float(torch.sqrt((t * t).mean()))
            TABS[n, q] = float(t.abs().mean())
            TPOS[n, q] = float((t > 0).double().mean())
        for j, q in enumerate(PROBE):
            m_h = SH[:, q].sum(0) / NS
            C_h = SHH[j] / NS - torch.outer(m_h, m_h)
            tr1 = float(torch.diagonal(C_h).sum()); tr2 = float((C_h * C_h).sum())
            EFR[n, j] = tr1 * tr1 / max(tr2, 1e-30)
            a_true = SA[j] / NS
            Ca_true = SAA[j] / NS - torch.outer(a_true, a_true)
            m_a, Ca = closure(m_h, C_h)          # 🚨 P5: вхід — ІСТИННІ m,C цього шару
            ERRM[n, j] = float(torch.sqrt(((m_a - a_true) ** 2).sum()
                                          / (a_true ** 2).sum()))
            dg = torch.diagonal(Ca) - torch.diagonal(Ca_true)
            ERRD[n, j] = float(torch.sqrt((dg * dg).sum()
                                          / (torch.diagonal(Ca_true) ** 2).sum()))
            off = (Ca - Ca_true) - torch.diag(dg)
            offT = Ca_true - torch.diag(torch.diagonal(Ca_true))
            ERRC[n, j] = float(torch.sqrt((off * off).sum() / (offT * offT).sum()))
        m1 = SH[:, 0].sum(0) / NS
        v1 = torch.clamp(S2[:, 0].sum(0) / NS - m1 * m1, min=1e-30)
        SEAL[n, 0] = float(torch.sqrt(((m1 / torch.sqrt(v1)) ** 2).mean()))
        SEAL[n, 1] = float((SA[0] / NS).mean())
        SEAL[n, 2] = EFR[n, 0]
    print(f'  net{n}  RMS|t|: шар1 {TRMS[n,0]:.4f} · шар2 {TRMS[n,1]:.3f} · '
          f'шар32 {TRMS[n,31]:.3f}', flush=True)

print(f'\n{"="*74}\n⛔ ПЛОМБИ')
print(f' P1  RMS|t| шару 1        = {SEAL[:,0].mean():.5f}   (мусить бути ~0, <0.02)')
print(f' P1b E[a⁽¹⁾]             = {SEAL[:,1].mean():.5f}   (ТОЧНО 0.56419)')
print(f' P2  RMS(t) шару 2        = {TRMS[:,1].mean():.4f}    (передбачення 0.55)')
print(f' P3  еф.ранг шару 1       = {SEAL[:,2].mean():.2f}    (мусить бути 256.00)')
d_half = np.abs(HALF[:, :, 0] - HALF[:, :, 1]).mean()
print(f' P4  розбіжність половин  = {d_half:.5f}    (шум; профіль стоїть, якщо ≪ сигнал)')

print(f'\n{"="*74}\n📈 ПРОФІЛЬ ПО ГЛИБИНІ ({len(FS)} мереж)')
print(f'{"шар":>4s} {"RMS(t)":>8s} {"mean|t|":>8s} {"частка t>0":>11s}'
      f' {"еф.ранг":>9s} {"err E[a]":>9s} {"err діаг":>9s} {"err поза":>9s}')
for q in range(L_):
    row = f'{q+1:4d} {TRMS[:,q].mean():8.3f} {TABS[:,q].mean():8.3f} {TPOS[:,q].mean():11.3f}'
    if q in PROBE:
        j = PROBE.index(q)
        row += (f' {EFR[:,j].mean():9.1f} {ERRM[:,j].mean():9.4%}'
                f' {ERRD[:,j].mean():9.4%} {ERRC[:,j].mean():9.4%}')
    print(row)

pq = np.array(PROBE)
tt = TRMS[:, pq].mean(0); ee = ERRM[:, :].mean(0)
cc = np.corrcoef(tt, ee)[0, 1]
cc2 = np.corrcoef(tt, ERRC[:, :].mean(0))[0, 1]
print(f'\n{"="*74}\n📣 ВИРОК')
print(f' A. RMS|t|: шар1 {TRMS[:,0].mean():.3f} → шар2 {TRMS[:,1].mean():.3f}'
      f' → шар8 {TRMS[:,7].mean():.3f} → шар16 {TRMS[:,15].mean():.3f}'
      f' → шар32 {TRMS[:,31].mean():.3f}')
print(f' B. свіжа err E[a]: шар1 {ERRM[:,0].mean():.4%} → шар32 {ERRM[:,-1].mean():.4%}'
      f'   (відношення ×{ERRM[:,-1].mean()/max(ERRM[:,0].mean(),1e-12):.2f})')
print(f' C. еф.ранг: шар1 {EFR[:,0].mean():.1f} → шар32 {EFR[:,-1].mean():.1f}')
print(f' D. 🎯 corr(|t|, err E[a]) по шарах = {cc:+.4f}   (передбачення >0.8)')
print(f'    corr(|t|, err позадіаг)        = {cc2:+.4f}')
