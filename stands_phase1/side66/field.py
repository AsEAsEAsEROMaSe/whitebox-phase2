"""СТЕЛЯ КЛАСУ «НЕСТИ ПОЛЕ»: скільки коштує сама гіпотеза гаусовості (С78).

Ідея юзера: 256 вимірів — це по суті велика сума, і на вхід приходить не набір
точок, а ЩІЛЬНЕ ПОЛЕ, як потік води. Тоді задача — не кидати семпли, а
проштовхнути щільність крізь 32 шари. Саме це робить наше замикання: воно несе
параметри поля `(m, C)`, а не точки.

Поле у 256 вимірах точно нести НЕМОЖЛИВО: після першої складки ReLU воно вже не
гаусове, а повний опис потребує нескінченно багато чисел. Ми зрізаємо до (m, C).
🔑 ПИТАННЯ, ЯКЕ ЦЕ РОЗДІЛЯЄ: скільки з похибки замикання — від НАКОПИЧЕННЯ
   (неточні m, s їдуть з попередніх шарів), а скільки — від САМОГО ЗРІЗАННЯ?

Для `E[ReLU(p_j)]` потрібен лише ОДНОВИМІРНИЙ маргінал `p_j`. Дамо формулі
    E[ReLU(p)] = m·Φ(t) + s·φ(t),  t = m/s
ТОЧНІ `m_j` і `s_j` з великого прогону. Усе, що лишиться, — ціна гіпотези
«маргінал гаусів». Це СТЕЛЯ всього класу «нести поле параметрично»: жоден
скільки завгодно точний перенос `C` нижче неї не опуститься.

Далі похибка кожного шару несеться до виходу тим самим точним оператором ⊙g
(машинерія з `gcsplit.py`), і сума порівнюється з MSE нашого сабміта.

📣 ПЕРЕДБАЧЕНО: `p_j` — сума 256 доданків ⟹ CLT дає негаусовість ~1/√256 ≈ 6%
   у третьому кумулянті, але вплив на САМЕ СЕРЕДНЄ ReLU має бути значно менший,
   бо ReLU гладка майже скрізь. Якщо стеля вийде ≫45× — клас живий і бити треба
   в перенос `C`; якщо ≲2× — «нести поле» мертве незалежно від якості переносу.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18   # істина: 2^LOGN ПАР
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
W_ = 256
D = 32
CH = 1 << 14
S2PI = float(1.0 / np.sqrt(2 * np.pi))

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _stats(W, seed, npairs, whiten=False):
    """Суми p, p², relu(p), gate по шарах ⟹ ТОЧНІ m, s, g, c кожного шару."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(8, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]                 # 🚨 fp32: fp64 на P104 у 32× повільніше
            pd = p.double()
            Z[L][0] += pd.sum(0)
            Z[L][1] += (pd * pd).sum(0)
            Z[L][2] += (p > 0).double().sum(0)
            p3 = pd * pd * pd
            Z[L][4] += p3.sum(0)
            p4 = p3 * pd
            Z[L][5] += p4.sum(0)
            p5 = p4 * pd
            Z[L][6] += p5.sum(0)
            Z[L][7] += (p5 * pd).sum(0)
            x = torch.relu(p)
            Z[L][3] += x.double().sum(0)
        done += b
    N = float(2 * npairs)
    mp = [Z[L][0] / N for L in range(D)]
    s = [(Z[L][1] / N - mp[L] * mp[L]).clamp_min(1e-30).sqrt() for L in range(D)]
    g = [Z[L][2] / N for L in range(D)]
    m = [Z[L][3] / N for L in range(D)]
    c = [m[L] - mp[L] * g[L] for L in range(D)]
    # центральні з сирих; κ₃=μ₃, κ₄=μ₄−3σ⁴, κ₅=μ₅−10σ²μ₃,
    # κ₆ = μ₆ − 15σ²μ₄ − 10μ₃² + 30σ⁶
    k3, k4, k5, k6 = [], [], [], []
    for L in range(D):
        a1, a2 = mp[L], Z[L][1] / N
        a3, a4, a5, a6 = Z[L][4] / N, Z[L][5] / N, Z[L][6] / N, Z[L][7] / N
        mu3 = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
        mu4 = a4 - 4.0 * a1 * a3 + 6.0 * a1 * a1 * a2 - 3.0 * a1 ** 4
        mu5 = (a5 - 5.0 * a1 * a4 + 10.0 * a1 * a1 * a3
               - 10.0 * a1 ** 3 * a2 + 4.0 * a1 ** 5)
        mu6 = (a6 - 6.0 * a1 * a5 + 15.0 * a1 * a1 * a4
               - 20.0 * a1 ** 3 * a3 + 15.0 * a1 ** 4 * a2 - 5.0 * a1 ** 6)
        s2 = s[L] * s[L]
        k3.append(mu3)
        k4.append(mu4 - 3.0 * s2 * s2)
        k5.append(mu5 - 10.0 * s2 * mu3)
        k6.append(mu6 - 15.0 * s2 * mu4 - 10.0 * mu3 * mu3 + 30.0 * s2 ** 3)
    return mp, s, g, c, m, k3, k4, k5, k6


SQ2 = float(np.sqrt(2.0))


def _delta(W, Wd, seed):
    """δ_ℓ гаусової формули на власних (m,s), і те саме з поправкою Еджворта.

    🔑 ПОПРАВКА ВИВОДИТЬСЯ ТОЧНО. Нехай p = m + s·u, t = m/s, γ = κ₃/s³.
    Еджворт: f_u ≈ φ(x)[1 + γ/6·He₃(x)], He₃ = x³−3x. Тоді
        ∫_{−t}^∞ (t+x)φ(x)He₃(x)dx = t·φ(t)(t²−1) − φ(t)t³ = −t·φ(t)
    (через ∫_a^∞ φHe_n = φ(a)He_{n−1}(a) і x·He₃ = He₄ + 3He₂), отже
        E[ReLU(p)] ≈ m·Φ(t) + s·φ(t) − κ₃·t·φ(t)/(6s²)
    Ціна на шар — 256 чисел, а не тензор 3-го рангу.
    """
    mp, s, g, _, m, k3, k4, k5, k6 = _stats(W, seed, 1 << LOGN)
    dl, de, df, dg, dh = [], [], [], [], []
    for L in range(D):
        t = mp[L] / s[L]
        Ph = 0.5 * (1.0 + torch.erf(t / SQ2))
        ph = torch.exp(-0.5 * t * t) * S2PI
        gauss = mp[L] * Ph + s[L] * ph
        s2 = s[L] * s[L]
        s3, s4 = s2 * s[L], s2 * s2
        t2 = t * t
        t3 = t2 * t
        # 🔑 ЗАГАЛЬНА ФОРМУЛА: ∫_{−t}^∞ (t+x)φ(x)He_n(x)dx = (−1)ⁿ·φ(t)·He_{n−2}(t)
        #    (перевірено проти виведених вручну n=3,4,6). Далі — просто He.
        He3 = t3 - 3.0 * t
        He5 = t3 * t2 - 10.0 * t3 + 15.0 * t
        He7 = t3 * t2 * t2 - 21.0 * t3 * t2 + 105.0 * t3 - 105.0 * t
        c3 = -k3[L] * t * ph / (6.0 * s2)                        # γ₃/6·He₃
        c4 = k4[L] * ph * (t2 - 1.0) / (24.0 * s3)               # γ₄/24·He₄
        c33 = (k3[L] ** 2) * ph * (t2 * t2 - 6.0 * t2 + 3.0) / (72.0 * s4 * s[L])
        # ТРЕТІЙ ПОРЯДОК: γ₅/120·He₅ + γ₃γ₄/144·He₇ + γ₃³/1296·He₉
        c5 = -k5[L] * ph * He3 / (120.0 * s4)
        c34 = -k3[L] * k4[L] * ph * He5 / (144.0 * s4 * s2)
        c333 = -(k3[L] ** 3) * ph * He7 / (1296.0 * s4 * s4)
        dl.append(gauss - m[L])
        de.append(gauss + c3 - m[L])
        df.append(gauss + c3 + c4 + c33 - m[L])
        # ЧЕТВЕРТИЙ ПОРЯДОК: γ₆/720·He₆ + γ₃γ₅/720·He₈ + γ₄²/1152·He₈
        #                   + γ₃²γ₄/1728·He₁₀ + γ₃⁴/31104·He₁₂
        # (усі парні n ⟹ знак +, і He_n → He_{n−2})
        t4 = t2 * t2; t6 = t4 * t2; t8 = t4 * t4; t10 = t8 * t2
        He4 = t4 - 6.0 * t2 + 3.0
        He6 = t6 - 15.0 * t4 + 45.0 * t2 - 15.0
        He8 = t8 - 28.0 * t6 + 210.0 * t4 - 420.0 * t2 + 105.0
        He10 = (t10 - 45.0 * t8 + 630.0 * t6 - 3150.0 * t4
                + 4725.0 * t2 - 945.0)
        s5, s7 = s4 * s[L], s4 * s2 * s[L]
        c6 = k6[L] * ph * He4 / (720.0 * s5)
        c35 = k3[L] * k5[L] * ph * He6 / (720.0 * s7)
        c44 = k4[L] * k4[L] * ph * He6 / (1152.0 * s7)
        c334 = k3[L] ** 2 * k4[L] * ph * He8 / (1728.0 * s7 * s2)
        c3333 = k3[L] ** 4 * ph * He10 / (31104.0 * s7 * s4)
        dg.append(gauss + c3 + c4 + c33 + c5 + c34 + c333 - m[L])
        dh.append(gauss + c3 + c4 + c33 + c5 + c34 + c333
                  + c6 + c35 + c44 + c334 + c3333 - m[L])

    def carry(src):
        out = []
        for L in range(D):
            v = src[L]
            for j in range(L + 1, D):
                v = (v @ Wd[j]) * g[j]
            out.append(v)
        return out

    return dl, carry(dl), carry(de), carry(df), carry(dg), carry(dh), m[-1]


rows, seals = [], []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    Wd = [w.double() for w in W]
    with torch.no_grad():
        # 🚨 ДВІ НЕЗАЛЕЖНІ ІСТИНИ. `δ` містить шум МК того ж порядку, що й сам
        #    ефект (зміряно: пломба δ₀ = 1.1e-03 там, де формула ТОЧНА). Квадрат
        #    однієї вибірки міряв би шум. Крос-добуток двох незалежних — ні.
        dA, cA, eA, fA, gA, hA, tA = _delta(W, Wd, 91_000 + 13 * n)
        dB, cB, eB, fB, gB, hB, tB = _delta(W, Wd, 40_000 + 29 * n)

        # ✅ПЛОМБА: на шарі 0 вхід рівно гаусів ⟹ ⟨δ₀ᴬ, δ₀ᴮ⟩ мусить бути ~0
        seals.append(float((dA[0] * dB[0]).sum()) /
                     float(dA[0].norm() * dB[0].norm() + 1e-300))

        ceil2 = float((torch.stack(cA).sum(0) * torch.stack(cB).sum(0)).sum())
        edge2 = float((torch.stack(eA).sum(0) * torch.stack(eB).sum(0)).sum())
        edg4 = float((torch.stack(fA).sum(0) * torch.stack(fB).sum(0)).sum())
        edg5 = float((torch.stack(gA).sum(0) * torch.stack(gB).sum(0)).sum())
        edg6 = float((torch.stack(hA).sum(0) * torch.stack(hB).sum(0)).sum())
        prof = [float((cA[L] * cB[L]).sum()) for L in range(D)]

        # наш робочий MSE — теж на КРОС-ЦІЛІ
        ours = 0.0
        for d in range(DRAWS):
            mh = _stats(W, 55_000 + 7 * n + 1013 * d, NPAIR, whiten=True)[4]
            ours += float(((mh[-1] - tA) * (mh[-1] - tB)).sum())
        ours /= DRAWS

        rows.append([ceil2, ours, float((tA*tB).sum()), edge2, edg4, edg5, edg6] + prof)
    del W, Wd
    torch.cuda.empty_cache()
    print(f"  … мережа {n+1}/{NETS}  стеля={ceil2:.3e}  наш MSE={ours:.3e}  "
          f"⟹ запас {ours/max(ceil2,1e-30):.1f}×  cos(δ₀ᴬ,δ₀ᴮ) {seals[-1]:+.3f}", flush=True)

R = np.array(rows)
CEIL, OURS, EDGE = R[:, 0].sum(), R[:, 1].sum(), R[:, 3].sum()
EDG4, EDG5, EDG6 = R[:, 4].sum(), R[:, 5].sum(), R[:, 6].sum()
P = R[:, 7:].sum(0)
print(f'\n{NETS} мереж (відношення СУМ)\n')
print(f'✅ПЛОМБА cos(δ₀ᴬ,δ₀ᴮ) — на ш.0 формула ТОЧНА ⟹ мусить бути ~0: {np.median(seals):+.4f}\n')
print(f'{"шар":>4s} {"‖перенесена δ_ℓ‖":>18s} {"частка":>9s}')
print('-' * 34)
for L in range(D):
    print(f'{L:4d} {P[L]:18.4e} {P[L]/P.sum():8.1%}')
print(f'\n{"":=^70s}')
print(f'СТЕЛЯ КЛАСУ «нести поле» (гаусова гіпотеза на ТОЧНИХ m,s): {CEIL/NETS:.4e}')
print(f'НАШ робочий MSE (вибірка {2*NPAIR} семплів):               {OURS/NETS:.4e}')
print(f'🔑 ЗАПАС ДО СТЕЛІ: **{OURS/max(CEIL,1e-30):.1f}×**   '
      f'(лідеру треба 24.4×, повний приз ~45×)')
print(f'   відн. похибка стелі: {np.sqrt(max(CEIL,0)/R[:,2].sum()):.3e}  '
      f'(наша робоча 0.164%, лідерова 0.02%)')
print(f'\n🔑 З ПОПРАВКОЮ ЕДЖВОРТА (третій кумулянт, 256 чисел на шар):')
print(f'   стеля {EDGE/NETS:.4e}  ⟹ негаусовість знято на '
      f'{100*(1-EDGE/max(CEIL,1e-30)):.1f}%,  нова відн. похибка '
      f'{np.sqrt(max(EDGE,0)/R[:,2].sum()):.3e}')
print(f'🔑 ДРУГИЙ ПОРЯДОК (κ₄ і κ₃², ще 512 чисел на шар):')
print(f'   стеля {EDG4/NETS:.4e}  ⟹ знято {100*(1-EDG4/max(CEIL,1e-30)):.2f}%,  '
      f'відн. похибка {np.sqrt(max(EDG4,0)/R[:,2].sum()):.3e}')
print(f'   ⟹ запас до НАШОГО MSE: {OURS/max(EDG4,1e-30):.1f}×  '
      f'(лідеру треба 24.4×)')
print(f'🔑 ТРЕТІЙ ПОРЯДОК (κ₅, κ₃κ₄, κ₃³):')
print(f'   стеля {EDG5/NETS:.4e}  ⟹ знято {100*(1-EDG5/max(CEIL,1e-30)):.3f}%,  '
      f'відн. похибка {np.sqrt(max(EDG5,0)/R[:,2].sum()):.3e}')
print(f'   ⟹ запас до НАШОГО MSE: {OURS/max(EDG5,1e-30):.1f}×   '
      f'{"🟢 РЯД ЗБІГАЄТЬСЯ" if EDG5 < EDG4 else "🚨 РЯД РОЗХОДИТЬСЯ — стоп на 2-му"}')
print(f'🔑 ЧЕТВЕРТИЙ ПОРЯДОК (κ₆, κ₃κ₅, κ₄², κ₃²κ₄, κ₃⁴):')
print(f'   стеля {EDG6/NETS:.4e}  ⟹ знято {100*(1-EDG6/max(CEIL,1e-30)):.4f}%,  '
      f'відн. похибка {np.sqrt(max(EDG6,0)/R[:,2].sum()):.3e}')
print(f'   ⟹ запас до НАШОГО MSE: {OURS/max(EDG6,1e-30):.1f}×   '
      f'{"🟢 ЩЕ ЗБІГАЄТЬСЯ" if EDG6 < EDG5 else "🚨 РЯД РОЗІЙШОВСЯ — стоп на κ₅"}')
print('📊 БАЛ: 1.954e-07 — стеля, не виграш; прайсується картою (K,ε)')
