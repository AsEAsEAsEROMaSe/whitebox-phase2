"""ПОХИБКА ОДНОГО КРОКУ ЗАМИКАННЯ — І ЧИ ПОЯСНЮЄ ЇЇ НЕГАУСОВІСТЬ МАРГІНАЛІВ.

## ЗВІДКИ
`lowrk87.py` (етап 1) виявив, що я міряв НЕ ТУ ВЕЛИЧИНУ: низький ранг `C`
ЗДЕШЕВЛЮЄ, а не УТОЧНЮЄ — при `k=256` він дає рівно ту саму `σ` з тією самою
похибкою 13.5%. Джерело похибки — САМЕ ЗАМИКАННЯ, а не обрізання.
⟹ треба міряти похибку ЗАМИКАННЯ, ізольовану від накопичення по ланцюгу.

## 🎯 ЩО САМЕ МІРЯЄТЬСЯ (ізоляція, якої досі не було)
На шарі `l` беремо ІСТИННІ `m_h, C_h` (з МК) ⟹ подаємо в замикання ⟹
`C_a^мелер` ⟹ `σ'_мелер = sqrt(diag(Wᵀ C_a^мелер W))`.
Порівнюємо з `σ'_точна` від ІСТИННОЇ `C_a` (теж з МК).
🔑 Це «СВІЖА» похибка одного кроку — без накопичення. Досі ми бачили лише
накопичену (0.049% на шарі 1 → 14.8% на шарі 31, атрактор ~15%), і не знали,
скільки з неї народжується НА КРОЦІ, а скільки успадковано.

## 📐 ДІРА, ЯКУ ПЕРЕВІРЯЄМО
Ряд Мелера — це розклад по степенях КОРЕЛЯЦІЇ при ГАУСОВИХ маргіналах.
Ми закрили `K=3..6` (рівно нуль, бо всі похідні ReLU від 2-ї несуть `φ(t)`,
`t→3.00`). Але це закрило залежність від `ρ`, а НЕ негаусовість самих
маргіналів `p_j` (їхні `γ₃` ростуть 0→0.5, `γ₄` теж ненульова).
⟹ тут перевіряємо, чи залишок кроку пояснюється МАРГІНАЛЬНИМИ `γ₃,γ₄`.
🔑 Практична вага питання: маргінальні `γ` — це ВЕКТОР (n чисел, можна вести
матвеком); змішані кумулянти — ТЕНЗОР `n³` (10.2 ГБ, закрито в С86).

## ⛔ПЛОМБИ
 P1 ТАВТОЛОГІЯ: подати в замикання істинну `C_a` замість мелерової ⟹ похибка
    мусить бути РІВНО 0 (перевірка, що `qform` і порівняння зібрані вірно).
 P2 ШУМ ІСТИНИ (піврозмір МК) окремою колонкою; мусить бути ≪ ефекту.
 P3 ⛔КОНТРОЛЬ ЧУЖОЇ ФОРМИ: регресія залишку на `γ` ІНШОГО шару. Якщо чужі
    `γ` пояснюють не гірше — сигнал не в `γ`.
 P4 ⛔ПЕРЕМІШАНІ по нейронах `γ` того самого шару: рве зв'язок «нейрон↔його
    форма», лишаючи розподіл. Якщо не падає — працює не понейронна форма.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) свіжа похибка кроку буде 1–4% (проти накопичених 13.5%). Шанс 55%.
     Якщо вона ~13% — накопичення НЕМАЄ, і задача суто локальна (краще для нас).
     Якщо вона <0.5% — уся біда в накопиченні, локальні поправки не врятують.
 (б) залишок кроку КОРЕЛЮЄ з маргінальними `γ₃γ₃` сильніше за 0.3. Шанс 40%.
 (в) ⛔перемішані `γ` дадуть помітно гірше. Шанс 70%.
 (г) знак: замикання ЗАНИЖУЄ `σ` на кроці (як і накопичено). Шанс 75%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/stepcov87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = (3, 7, 11, 15, 19, 23, 27, 30)
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def qform(W, C):
    return torch.clamp((C @ W * W).sum(0), min=1e-30)


def closure(m, C):
    """РІВНО те замикання, що стоїть у сабміті (Мелер K=2 + точна діагональ)."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
    m_a = m * P + s * f
    q2 = (m * m + s * s) * P + (m * s) * f
    d = torch.clamp(q2 - m_a * m_a, min=1e-30)
    return Ca + torch.diag(d - torch.diagonal(Ca)), m_a


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV} · шари {PROBE}\n', flush=True)

NP = len(PROBE)
E_STEP = np.zeros((len(FS), NP))     # свіжа похибка кроку в σ
E_TAUT = np.zeros((len(FS), NP))     # ⛔P1 тавтологія
E_NOISE = np.zeros((len(FS), NP))    # ⛔P2 шум істини
BIAS = np.zeros((len(FS), NP))       # знак: <1 = замикання ЗАНИЖУЄ
E_OFF = np.zeros((len(FS), NP))      # похибка від ПОЗАДІАГОНАЛІ (діаг точна)
E_DIA = np.zeros((len(FS), NP))      # похибка від ДІАГОНАЛІ (позадіаг точна)
DGAP = np.zeros((len(FS), NP))       # Var(a)/Var(p) — межа каже <=1
G3 = np.zeros((len(FS), NP, W_))
G4 = np.zeros((len(FS), NP, W_))
RESID = np.zeros((len(FS), NP, W_))  # понейронний залишок log(σ_точна/σ_мелер)
TVEC = np.zeros((len(FS), NP, W_))
OFFR = np.zeros((len(FS), NP, W_))   # частка позадіагоналі

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        SH = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GH = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SA = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        GA = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        GAH = torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
        SAH = torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
        M34 = torch.zeros(NP, 2, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(7300 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                a = torch.relu(h)
                if q in PROBE:
                    j = PROBE.index(q)
                    hd, ad = h.double(), a.double()
                    SH[j] += hd.sum(0)
                    GH[j] += hd.T @ hd
                    SA[j] += ad.sum(0)
                    GA[j] += ad.T @ ad
                    M34[j, 0] += (hd ** 3).sum(0)
                    M34[j, 1] += (hd ** 4).sum(0)
                    if d < NS // 2:
                        SAH[j] += ad.sum(0)
                        GAH[j] += ad.T @ ad
            d += b

        for j, q in enumerate(PROBE):
            mh = SH[j] / NS
            Ch = GH[j] / NS - torch.outer(mh, mh)
            Ch = 0.5 * (Ch + Ch.T)
            ma = SA[j] / NS
            Ca = GA[j] / NS - torch.outer(ma, ma)
            Ca = 0.5 * (Ca + Ca.T)
            Wn = W[q + 1]

            s_true = torch.sqrt(qform(Wn, Ca))
            Ca_cl, _ = closure(mh, Ch)              # ІСТИННІ m,C на вході
            s_cl = torch.sqrt(qform(Wn, Ca_cl))
            # 🔑 РОЗЩЕПЛЕННЯ ДІАГОНАЛЬ / ПОЗАДІАГОНАЛЬ.
            #    Строга нерівність `Var(ReLU(p)) <= Var(p)` обмежує похибку
            #    ДІАГОНАЛІ на глибині величиною 0.25%. Якщо крок помиляється
            #    на 1.6%, похибка мусить сидіти в ПОЗАДІАГОНАЛІ — перевіряємо.
            dcl = torch.diagonal(Ca_cl)
            dtr = torch.diagonal(Ca)
            mixD = Ca_cl + torch.diag(dtr - dcl)   # позадіаг. МЕЛЕР, діаг ТОЧНА
            mixO = Ca + torch.diag(dcl - dtr)      # позадіаг. ТОЧНА, діаг МЕЛЕР
            E_OFF[n, j] = float(torch.sqrt(
                (((torch.sqrt(qform(Wn, mixD)) - s_true) ** 2).mean()
                 / (s_true * s_true).mean())))
            E_DIA[n, j] = float(torch.sqrt(
                (((torch.sqrt(qform(Wn, mixO)) - s_true) ** 2).mean()
                 / (s_true * s_true).mean())))
            DGAP[n, j] = float((dtr / torch.diagonal(Ch)).mean())
            E_STEP[n, j] = float(torch.sqrt(
                (((s_cl - s_true) ** 2).mean() / (s_true * s_true).mean())))
            BIAS[n, j] = float((s_cl * s_true).sum() / (s_true * s_true).sum())
            RESID[n, j] = (torch.log(s_true / s_cl)).cpu().numpy()

            # ⛔P1 ТАВТОЛОГІЯ: істинна Ca замість мелерової ⟹ рівно 0
            E_TAUT[n, j] = float(torch.sqrt(
                (((torch.sqrt(qform(Wn, Ca)) - s_true) ** 2).mean()
                 / (s_true * s_true).mean())))

            # ⛔P2 шум істини
            mah = SAH[j] / (NS // 2)
            Cah = GAH[j] / (NS // 2) - torch.outer(mah, mah)
            E_NOISE[n, j] = float(torch.sqrt(
                (((torch.sqrt(qform(Wn, 0.5 * (Cah + Cah.T))) - s_true) ** 2).mean()
                 / (s_true * s_true).mean())))

            # маргінальні γ₃,γ₄ ПРЕАКТИВАЦІЙ шару (те, що можна вести вектором)
            m1 = mh
            m2 = torch.diagonal(GH[j]) / NS
            m3 = M34[j, 0] / NS
            m4 = M34[j, 1] / NS
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4 - 3.0 * c2 * c2
            sg = torch.sqrt(c2)
            G3[n, j] = (c3 / sg ** 3).cpu().numpy()
            G4[n, j] = (c4 / sg ** 4).cpu().numpy()
            TVEC[n, j] = (m1 / sg).cpu().numpy()
            dg1 = (Wn * Wn * torch.diagonal(Ca)[:, None]).sum(0)
            OFFR[n, j] = (1.0 - dg1 / qform(Wn, Ca)).cpu().numpy()
    print(f'  net{n} · крок {E_STEP[n].mean():.3%} · знак {BIAS[n].mean():.4f} '
          f'· шум {E_NOISE[n].mean():.4%}', flush=True)

print(f'\n{"═" * 76}\n⛔ПЛОМБИ')
print(f' P1 тавтологія (істинна Ca) = {E_TAUT.mean():.3e}   '
      f'{"✅" if E_TAUT.mean() < 1e-12 else "⛔"}')
print(f' P2 шум істини = {E_NOISE.mean():.4%}  (ефект нижче мусить бути більшим)')

print(f'\n{"═" * 76}')
print('🎯 СВІЖА ПОХИБКА ОДНОГО КРОКУ (істинні m,C на вході) — ПО ШАРАХ')
print('шар       : ' + ' '.join(f'{q + 1:8d}' for q in PROBE))
print('похибка σ : ' + ' '.join(f'{E_STEP[:, j].mean():8.3%}' for j in range(NP)))
print('масштаб   : ' + ' '.join(f'{BIAS[:, j].mean():8.4f}' for j in range(NP)))
print('шум       : ' + ' '.join(f'{E_NOISE[:, j].mean():8.4%}' for j in range(NP)))
print('t=m/σ     : ' + ' '.join(f'{TVEC[:, j].mean():8.3f}' for j in range(NP)))
print('γ₃ сер    : ' + ' '.join(f'{G3[:, j].mean():8.3f}' for j in range(NP)))
print('γ₄ сер    : ' + ' '.join(f'{G4[:, j].mean():8.3f}' for j in range(NP)))
print('\n🔑 РОЗЩЕПЛЕННЯ ДЖЕРЕЛА (сума мусить приблизно дати повну похибку):')
print('лише позадіаг: ' + ' '.join(f'{E_OFF[:, j].mean():8.3%}' for j in range(NP)))
print('лише діагон. : ' + ' '.join(f'{E_DIA[:, j].mean():8.3%}' for j in range(NP)))
print('Var(a)/Var(p): ' + ' '.join(f'{DGAP[:, j].mean():8.4f}' for j in range(NP)))
print('   🎯 строга межа: Var(a)/Var(p) <= 1 ЗАВЖДИ; при t=3 розрив = 0.2496%')
print('\n🔑 порівняння: НАКОПИЧЕНА похибка σ у ланцюгу = 14.8% на шарі 31')
print('   якщо свіжа ≈ накопиченій ⟹ задача ЛОКАЛЬНА (лікується на кроці)')
print('   якщо свіжа ≪ накопиченої ⟹ біда в НАКОПИЧЕННІ, крок лікувати марно')


def r2_loo(X, y, groups):
    """R² з LOO ПО МЕРЕЖАХ (гребінь). Уся регресія — на відкладених мережах."""
    X = np.column_stack([X, np.ones(len(X))])
    out = np.zeros(len(y))
    for gi in np.unique(groups):
        tr, te = groups != gi, groups == gi
        A = X[tr].T @ X[tr] + 1e-8 * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ y[tr])
        out[te] = X[te] @ w
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - out) ** 2).sum() / max(ss, 1e-30)


print(f'\n{"═" * 76}')
print('ЧИ ПОЯСНЮЄ ЗАЛИШОК КРОКУ МАРГІНАЛЬНА ФОРМА `γ₃,γ₄`?')
print('  (ціль: понейронний log(σ_точна/σ_мелер); LOO по мережах)')
print(f'{"шар":>5s} {"RMS цілі":>10s} {"γ₃,γ₄,t":>9s} {"+off":>9s} '
      f'{"⛔перемішані":>13s} {"⛔чужий шар":>12s}')
rng = np.random.default_rng(20260805)
for j, q in enumerate(PROBE):
    y = RESID[:, j].ravel()
    grp = np.repeat(np.arange(len(FS)), W_)
    g3, g4, tv, of = (G3[:, j].ravel(), G4[:, j].ravel(),
                      TVEC[:, j].ravel(), OFFR[:, j].ravel())
    base = np.column_stack([g3, g4, tv, g3 * g3, g3 * tv, g4 * tv])
    r_base = r2_loo(base, y, grp)
    r_off = r2_loo(np.column_stack([base, of, of * of]), y, grp)
    # ⛔P4 перемішані ПО НЕЙРОНАХ усередині кожної мережі
    sh = base.reshape(len(FS), W_, -1).copy()
    for i in range(len(FS)):
        sh[i] = sh[i][rng.permutation(W_)]
    r_sh = r2_loo(sh.reshape(len(y), -1), y, grp)
    # ⛔P3 форма ЧУЖОГО шару
    jo = (j + 3) % NP
    alt = np.column_stack([G3[:, jo].ravel(), G4[:, jo].ravel(),
                           TVEC[:, jo].ravel()])
    r_alt = r2_loo(alt, y, grp)
    print(f'{q + 1:5d} {y.std():9.3%} {r_base:9.4f} {r_off:9.4f} '
          f'{r_sh:13.4f} {r_alt:12.4f}')
print('\n🎯 R² треба ≥0.9995, щоб довести σ з 8.72% до 0.465% '
      '(закон: ЦІННІСТЬ РЕГРЕСОРА = min(СИГНАЛ, ТОЧНІСТЬ ОПОРИ))')
