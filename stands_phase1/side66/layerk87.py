"""СТЕЛЯ ІДЕЇ ЮЗЕРА «МІКРОНЕЙРОМЕРЕЖА НА КОЖЕН НЕЙРОН» (С87)

## ЗВІДКИ ПИТАННЯ
ЮЗЕР: «знаєш теорію універсальної апроксимації? може для кожного нейрона треба
виводити мікронейромережу?»

## 🔑 ПЕРЕФОРМУЛЮВАННЯ ДО ГРАНИЧНОЇ ФОРМИ
Теорема каже про ПОТУЖНІСТЬ апроксиматора. Наше вузьке місце — ІНФОРМАЦІЯ на
вході. Тому міряємо не мережу, а ЇЇ СТЕЛЮ: замість «навчити модель» даємо
ОРАКУЛЬНУ шарову константу і дивимось, куди сяде ланцюг. Жоден апроксиматор
цього не перевищить.

📐 ЧОМУ САМЕ ШАРОВА КОНСТАНТА, А НЕ ПОНЕЙРОННА МОДЕЛЬ: `ladder87.py` показав
`масшт r` 1.0000 → 0.8724 по ланцюгу ⟹ 12.8% похибки `σ` СПІЛЬНІ для всіх 256
нейронів шару. Мікромережа на нейрон витрачала б потужність на понейронну
частину; гранична форма ідеї — 2 скаляри на шар (`k_m`, `k_σ`) = 64 числа на
всю мережу. Ціна інференсу ≈ 0 ⟹ ГЕЙТ ЦІНИ ПРОЙДЕНО, лишається лише точність.

## 🚨 ЩО ЗМІНИЛОСЬ ПРОТИ ПОПЕРЕДНЬОЇ ОЦІНКИ (виправлення ДО запуску)
1. «понейронний розкид 1.18%» — ХИБНО. `TR_R[:,q].std()` у `ladder87.py` — це
   розкид константи МІЖ МЕРЕЖАМИ. Понейронний залишок = `TR_RS` = **8.72%**.
2. `L6` (оракул ЛИШЕ `σ`) = **1.2436%** ГІРШЕ за базу 0.9416% ⟹ виправлення
   `σ` без `m` ШКОДИТЬ: похибки `σ` і `m` частково скорочуються.
   ⟹ шарова константа на самій `σ` рухає в ТОЙ САМИЙ бік, що й `L6`.

## 🔧 ПРИЛАД — ПРЯМА MSE, БЕЗ ЖОДНОЇ КАЛІБРУВАЛЬНОЇ КОНСТАНТИ
`MSE = mean((pred − true)²)` — рівно грейдерна метрика. Ніяких переводів через
`0.941%/1.698e-5` (та калібровка брехала в 4.61×, С87).
Бал детермінованого методу = `0.1 · MSE` (множник на підлозі).
Пороги: `sub97` = 1.873e-07 ⟹ MSE 1.873e-06 · лідер = 5.102e-08 ⟹ MSE 5.10e-07.

⛔ПЛОМБИ (ставляться ПЕРЕД висновками):
 P1 `RMS(істина)²` мусить дати **0.88404** (незалежно зміряно `nrmcheck87.py`).
    Якщо ні — прилад міряє не той об'єкт, далі не читати.
 P2 База `B0` мусить дати **0.9416%** і MSE **7.84e-05** (двічі зміряно).
 P3 `L5` (оракул `m,σ,γ`) мусить бути НАЙКРАЩОЮ сходинкою — інакше баг.
 P4 Шум істини віднімається явно (піврозмір МК), і мусить бути ≪ сигналу.
 P5 ⛔КОНТРОЛЬ ЗВОРОТНОЇ КОНСТАНТИ: `k → 1/k` мусить ПОГІРШИТИ. Якщо і пряма,
    і зворотна покращують — виграш не від напрямку, а від самого масштабування.
 P6 ⛔КОНТРОЛЬ ТАВТОЛОГІЇ: `KMS` при повному понейронному оракулі ≡ `L4`.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (записую, щоб не тлумачити заднім числом):
 (а) `KC` (константа лише на `σ`, узгоджено через `C`) ПОГІРШИТЬ базу або дасть
     ≤×1.1. Шанс **70%**. Механізм: це той самий напрямок, що `L6`=1.2436%.
 (б) `KMC` (обидві константи) сяде в 0.3–0.6%, тобто ×2…×4 у MSE до бази, але
     НЕ дійде до `sub97` (0.1456%). Шанс **45%**.
 (в) `KMC` дійде до ≤0.1456% (рівень `sub97`). Шанс **12%**.
 (г) розклад `m` покаже, що похибка `m` переважно МАСШТАБНА (>60% дисперсії).
     Шанс 50%. Якщо НІ — ідея мертва навіть в оракульній формі.
 (д) ТЗ: `KMC` при ε=1% втратить <30% виграшу, при ε=3% — більшу частину.
     Шанс 60%. Це визначає, яку точність мусить дати мікромережа.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/layerk87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 21
NS = 1 << LOG2N
BATCH = 1 << 13
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)
SUB97_MSE = 1.873e-06
LEADER_MSE = 5.102e-07


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def edge_relu(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    G0 = t * P + f
    G3 = -t * f
    G4 = (t * t - 1.0) * f
    G6 = (t ** 4 - 6.0 * t * t + 3.0) * f
    return s * (G0 + (g3 / 6.0) * G3 + (g4 / 24.0) * G4 + (g3 * g3 / 72.0) * G6)


def edge_relu2(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    s2 = s * s
    return ((m * m + s2) * P + m * s * f
            + (g3 / 3.0) * s2 * f - (g4 / 12.0) * s2 * t * f
            + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)


def proj(a, b):
    """Оптимальний скаляр `k`: `min_k ||k·a − b||²` ⟹ `k = <a,b>/<a,a>`.

    🔑 Саме ЦЕ вгадувала б мікромережа: одне число на шар. Беремо ОРАКУЛЬНЕ
    (з істинного `b`) ⟹ отримуємо СТЕЛЮ, якої модель не перевищить."""
    den = float((a * a).sum())
    if den < 1e-25:
        return 1.0
    return float((a * b).sum()) / den


def chain(W, orc_true, orc_g=None, orc_ms=None, only=None, coherent=False,
          ks='', keps=0.0, kseed=0, kinv=False, trace=None,
          kfix_m=None, kfix_s=None, kout=None):
    """Ланцюг замикання з ШАРОВИМИ константами.

    `ks` містить 's' → константа на `σ` УЗГОДЖЕНО (масштабується вся `C` на k²)
    `ks` містить 'd' → константа лише на ВЕКТОР `σ`, матриця `C` не чіпається
    `ks` містить 'm' → константа на `m`
    `keps` → константа відома з відносною похибкою (ТЗ на мікромережу)
    `kinv` → ⛔КОНТРОЛЬ: константа перевернута (`k → 1/k`), мусить погіршити
    `kfix_m/kfix_s` → 🔑 РЕАЛЬНИЙ РЕЖИМ: константа взята ЗЗОВНІ (з банку, LOO),
        а не з оракула цієї мережі. Оракул дає СТЕЛЮ, банк — те, що можна мати.
    `orc_true` — істинні `(m,σ)` по шарах, ЛИШЕ джерело константи (і трасування)
    """
    dev = W[0].device
    n = W[0].shape[1]
    zero = torch.zeros(n, dtype=torch.float64, device=dev)
    C = W[0].T @ W[0]
    m = zero.clone()
    gg = None
    if keps > 0.0:
        gg = torch.Generator(device='cpu')
        gg.manual_seed(kseed)
    for i in range(1, len(W) + 1):
        mo, so = orc_true[i - 1]
        # ── ШАРОВІ КОНСТАНТИ (гранична форма «мікромережі») ──────────────────
        if 'm' in ks:
            km = kfix_m[i - 1] if kfix_m is not None else proj(m, mo)
            if kinv and km > 1e-12:
                km = 1.0 / km
            if keps > 0.0:
                km *= 1.0 + keps * float(torch.randn(1, generator=gg))
            m = m * km
            if kout is not None:
                kout['m'].append(km)
        s_own = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        if 's' in ks or 'd' in ks:
            kk = kfix_s[i - 1] if kfix_s is not None else proj(s_own, so)
            if kinv and kk > 1e-12:
                kk = 1.0 / kk
            if keps > 0.0:
                kk *= 1.0 + keps * float(torch.randn(1, generator=gg))
            if 's' in ks:
                # 🔑 УЗГОДЖЕНО: масштабуємо ВСЮ `C` на k², а не лише діагональ —
                #    інакше Мелер бачить `diag(C) ≠ σ²` і бере чужий масштаб.
                C = C * (kk * kk)
            s_own = s_own * kk
            if kout is not None:
                kout['s'].append(kk)
        # ── оракульні підміни (опорні сходинки драбини) ──────────────────────
        if orc_ms is not None:
            m_o, s_o = orc_ms[i - 1]
            m_u = m if only == 's' else m_o
            s = s_own if only == 'm' else s_o
        else:
            s, m_u = s_own, m
        g3, g4 = orc_g[i - 1] if orc_g is not None else (zero, zero)
        if trace is not None:
            # 🔑 РОЗКЛАД ОБОХ величин на МАСШТАБ + ЗАЛИШОК. Масштаб — це рівно
            #    те, що може дати шарова константа; залишок — те, чого вона
            #    дати НЕ може за жодної потужності апроксиматора.
            for key, cur, tru_ in (('m', m_u, mo), ('s', s, so)):
                k = proj(cur, tru_)
                trace[key + '_err'].append(
                    float(torch.sqrt(((cur - tru_) ** 2).mean()
                                     / (tru_ * tru_).mean())))
                trace[key + '_k'].append(k)
                trace[key + '_res'].append(
                    float(torch.sqrt(((k * cur - tru_) ** 2).mean()
                                     / (tru_ * tru_).mean())))
        m_a = edge_relu(m_u, s, g3, g4)
        if i == len(W):
            return m_a
        t = m_u / s
        P, f = Phi(t), phi(t)
        C_a = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        q2 = (edge_relu2(m_u, s, g3, g4) if coherent
              else (m_u * m_u + s * s) * P + (m_u * s) * f)
        d = q2 - m_a * m_a
        C_a = C_a + torch.diag(torch.clamp(d, min=1e-30) - torch.diagonal(C_a))
        m = m_a @ W[i]
        C = W[i].T @ (C_a @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV}\n', flush=True)

TAGS = ('B0 база (наше замик.)',
        'KS σ×k (лише діаг)',
        'KC σ×k УЗГОДЖЕНО (C·k²)',
        'KM m×k',
        'KM ε=0.3%',
        'KM ε=1%',
        'KM ε=3%',
        'KMC m×k і C·k²',
        'KMC+γ оракул форма',
        'KMC ε=1%',
        'KMC ε=3%',
        '🔑KM БАНК (LOO)',
        '🔑KMC БАНК (LOO)',
        '⛔KM БАНК перемішаний',
        '⛔KC ЗВОРОТНА k→1/k',
        '⛔KMC ЗВОРОТНА k→1/k',
        'L6 оракул ЛИШЕ σ',
        'L4 оракул m,σ (гаус)',
        'L5 оракул m,σ,γ')
E = {t: np.zeros(len(FS)) for t in TAGS}
MS = {t: np.zeros(len(FS)) for t in TAGS}
# 🚨 ЗАКОН «ВИГРАШ — ВЛАСТИВІСТЬ ПАРИ (ОЗНАКА, БАЗА)»: у сабміті вже стоїть
#    глобальний множник `_ALPHA`. Отже чесна база — це база ПІСЛЯ α, і кожну
#    сходинку теж треба міряти після α. Інакше вимір покаже те, що α вже вміє.
PT = {t: np.zeros(len(FS)) for t in TAGS}
PP = {t: np.zeros(len(FS)) for t in TAGS}
TT = np.zeros(len(FS))
NVAR = np.zeros(len(FS))
NRM2 = np.zeros(len(FS))
TR = {k: np.zeros((len(FS), L_)) for k in
      ('m_err', 'm_k', 'm_res', 's_err', 's_k', 's_res')}

KM_USED = np.zeros((len(FS), L_))
KS_USED = np.zeros((len(FS), L_))
STASH = []


def load_w(idx):
    sd = int(np.load(FS[idx])['seed'])
    w32 = [torch.tensor(np.asarray(WZ[have[sd]][q], np.float32), device=DEV)
           for q in range(L_)]
    return w32, [w.double() for w in w32]


# ══ ПРОХІД 1: дорогий (МК) — істина, оракульні моменти, ФАКТИЧНО ВЖИТІ k ══
for n, f in enumerate(FS):
    W32, W = load_w(n)
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        acch = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(7700 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                hd = h.double()
                p = hd
                for kk in range(4):
                    MOM[q, kk] += p.sum(0)
                    p = p * hd
                a = torch.relu(h)
            r = a.sum(0, dtype=torch.float64)
            acc += r
            if d < NS // 2:
                acch += r
            d += b
        tru = acc / NS
        NRM2[n] = float((tru ** 2).mean())
        # ⛔P4: дисперсія шуму істини. `half` має N/2 семплів ⟹
        #      Var(half − full) = Var(half) − Var(full) = Var(full).
        NVAR[n] = float((((acch / (NS // 2)) - tru) ** 2).mean())

        MOM /= NS
        ORC_G, ORC_MS = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4 - 3.0 * c2 * c2
            s = torch.sqrt(c2)
            ORC_G.append((c3 / s ** 3, c4 / s ** 4))
            ORC_MS.append((m1.clone(), s))

        tr = {k: [] for k in TR}
        chain(W, ORC_MS, trace=tr)
        for k in TR:
            TR[k][n] = np.array(tr[k])
        # 🚨 КОНСТАНТИ ЗБИРАЮТЬСЯ У ВЖЕ ВИПРАВЛЕНОМУ ЛАНЦЮГУ. Саме на цьому
        #    розлетівся `selfsig87.py`: там `r̄` міряли на НЕвиправленому
        #    ланцюгу і застосовували до виправленого ⟹ додатний зворотний
        #    зв'язок, масштаб пішов 0.87 → 3.39 → 0.10 → 5e7.
        ko = {'m': [], 's': []}
        chain(W, ORC_MS, ks='ms', kout=ko)
        KM_USED[n] = np.array(ko['m'])
        KS_USED[n] = np.array(ko['s'])
        STASH.append((torch.stack([a for a, _ in ORC_MS]).cpu(),
                      torch.stack([b for _, b in ORC_MS]).cpu(),
                      torch.stack([a for a, _ in ORC_G]).cpu(),
                      torch.stack([b for _, b in ORC_G]).cpu(),
                      tru.cpu()))
    print(f'  МК net{n} готова', flush=True)

print(f'\n══ ПРОХІД 2: сходинки (МК не повторюється) ══', flush=True)
for n in range(len(FS)):
    W32, W = load_w(n)
    with torch.no_grad():
        a1, a2, a3, a4, trc = STASH[n]
        ORC_MS = [(a1[q].to(DEV), a2[q].to(DEV)) for q in range(L_)]
        ORC_G = [(a3[q].to(DEV), a4[q].to(DEV)) for q in range(L_)]
        tru = trc.to(DEV)
        msk = np.arange(len(FS)) != n
        bank_m = (KM_USED[msk].mean(0) if len(FS) > 1 else KM_USED[n])
        bank_s = (KS_USED[msk].mean(0) if len(FS) > 1 else KS_USED[n])
        runs = {
            'B0 база (наше замик.)': chain(W, ORC_MS),
            # KS: виправляємо ЛИШЕ вектор σ, матрицю C лишаємо — неузгоджено,
            # але саме так виглядала б «наївна» вставка моделі.
            'KS σ×k (лише діаг)': chain(W, ORC_MS, ks='d'),
            'KC σ×k УЗГОДЖЕНО (C·k²)': chain(W, ORC_MS, ks='s'),
            'KM m×k': chain(W, ORC_MS, ks='m'),
            # 🎯 ТЗ на модель: з якою точністю мусить бути відома константа.
            #    ε підбирається так, щоб накрити РОЗКИД `k_m` між мережами —
            #    якщо розкид 0.3%, то ε=0.3% і є режимом «константа з банку».
            'KM ε=0.3%': chain(W, ORC_MS, ks='m', keps=0.003, kseed=31 + n),
            'KM ε=1%': chain(W, ORC_MS, ks='m', keps=0.01, kseed=61 + n),
            'KM ε=3%': chain(W, ORC_MS, ks='m', keps=0.03, kseed=91 + n),
            'KMC m×k і C·k²': chain(W, ORC_MS, ks='ms'),
            'KMC+γ оракул форма': chain(W, ORC_MS, ks='ms', orc_g=ORC_G,
                                        coherent=True),
            'KMC ε=1%': chain(W, ORC_MS, ks='ms', keps=0.01, kseed=101 + n),
            'KMC ε=3%': chain(W, ORC_MS, ks='ms', keps=0.03, kseed=303 + n),
            # 🔑 РЕАЛЬНИЙ РЕЖИМ: константа — середнє по ІНШИХ мережах банку.
            #    Оракульні рядки вище дають стелю; ці — те, що можна мати
            #    БЕЗ жодної моделі, просто зашивши 32 числа в сабміт.
            '🔑KM БАНК (LOO)': chain(W, ORC_MS, ks='m', kfix_m=bank_m),
            '🔑KMC БАНК (LOO)': chain(W, ORC_MS, ks='ms', kfix_m=bank_m,
                                      kfix_s=bank_s),
            # ⛔КОНТРОЛЬ: ті самі числа, але розкладені по шарах НАВПАКИ.
            #    Якщо виграє й він — справа не в профілі, а в самому факті
            #    множення на щось трохи менше за 1.
            '⛔KM БАНК перемішаний': chain(W, ORC_MS, ks='m',
                                          kfix_m=bank_m[::-1].copy()),
            '⛔KC ЗВОРОТНА k→1/k': chain(W, ORC_MS, ks='s', kinv=True),
            '⛔KMC ЗВОРОТНА k→1/k': chain(W, ORC_MS, ks='ms', kinv=True),
            'L6 оракул ЛИШЕ σ': chain(W, ORC_MS, orc_ms=ORC_MS, only='s'),
            'L4 оракул m,σ (гаус)': chain(W, ORC_MS, orc_ms=ORC_MS),
            'L5 оракул m,σ,γ': chain(W, ORC_MS, orc_ms=ORC_MS, orc_g=ORC_G,
                                     coherent=True),
        }
        TT[n] = float((tru * tru).sum())
        for t, pr in runs.items():
            MS[t][n] = float(((pr - tru) ** 2).mean())
            E[t][n] = math.sqrt(MS[t][n] / NRM2[n])
            PT[t][n] = float((pr * tru).sum())
            PP[t][n] = float((pr * pr).sum())
    print(f'  net{n} · ' + ' · '.join(f'{t.split()[0]} {E[t][n]:.4%}'
                                      for t in TAGS[:6]), flush=True)

print(f'\n{"═" * 74}\n⛔ПЛОМБИ')
print(f' P1 RMS(істина)² = {NRM2.mean():.5f}   (мусить бути 0.88404, '
      f'`nrmcheck87.py`)  {"✅" if abs(NRM2.mean() - 0.88404) < 0.02 else "⛔"}')
b0 = MS['B0 база (наше замик.)'].mean() - NVAR.mean()
print(f' P2 база: {E["B0 база (наше замик.)"].mean():.4%} · MSE {b0:.3e}   '
      f'(мусить бути 0.9416% / 7.84e-05)  '
      f'{"✅" if abs(E["B0 база (наше замик.)"].mean() - 0.009416) < 0.0006 else "⛔"}')
print(f' P4 шум істини: MSE {NVAR.mean():.3e} = '
      f'{NVAR.mean() / b0:.2%} бази')


def alpha_mse(t):
    """MSE після ГЛОБАЛЬНОГО множника, підібраного LOO по мережах.

    Це рівно те, що `_ALPHA` робить у сабміті. Без цієї колонки будь-яка
    масштабна поправка виглядала б виграшем, хоч α вже її вміє."""
    out = np.zeros(len(FS))
    a_n = PT[t] / np.maximum(PP[t], 1e-30)
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        a = a_n[msk].mean() if len(FS) > 1 else a_n[i]
        out[i] = (a * a * PP[t][i] - 2.0 * a * PT[t][i] + TT[i]) / W_
    return out.mean(), a_n


print(f'\n{"═" * 74}')
print('СТЕЛЯ ШАРОВИХ КОНСТАНТ (оракульні k ⟹ жодна модель не перевищить)')
b0a = alpha_mse('B0 база (наше замик.)')[0] - NVAR.mean()
print(f'{"сходинка":26s} {"віднос.":>8s} {"MSE−шум":>10s} {"ПІСЛЯ α":>10s} '
      f'{"vs sub97":>9s} {"vs базаα":>9s}')
for t in TAGS:
    mse = MS[t].mean() - NVAR.mean()
    msa, a_n = alpha_mse(t)
    msa -= NVAR.mean()
    lo = NVAR.mean() * 0.3
    fl = '<шум' if msa < lo else ''
    msa_c = max(msa, lo)
    print(f'{t:26s} {E[t].mean():7.4%} {max(mse, lo):10.3e} {msa_c:10.3e}{fl:<4s}'
          f'{"×%.3f" % (SUB97_MSE / msa_c):>9s} {"×%.3f" % (b0a / msa_c):>9s}')
print(f'{"🏆 sub97 (наш рекорд)":26s} {0.001456:7.4%} {SUB97_MSE:10.3e} '
      f'{1.873e-07:10.3e} {"×1.000":>9s}')
print(f'{"🥇 ЛІДЕР 322059":26s} {0.000760:7.4%} {LEADER_MSE:10.3e} '
      f'{5.102e-08:10.3e} {"×3.671":>9s}')

print(f'\n{"═" * 74}')
print('🔑 ЧИ Є ПОХИБКА МАСШТАБНОЮ? (масштаб лікується константою, залишок — ні)')
print('шар      : ' + ' '.join(f'{q + 1:7d}' for q in range(0, L_, 4)))
for lbl, key in (('m повна ', 'm_err'), ('m після k', 'm_res'),
                 ('σ повна ', 's_err'), ('σ після k', 's_res')):
    print(f'{lbl:9s}: ' + ' '.join(f'{TR[key][:, q].mean():7.3%}'
                                   for q in range(0, L_, 4)))
print('k_m      : ' + ' '.join(f'{TR["m_k"][:, q].mean():7.4f}'
                               for q in range(0, L_, 4)))
print('k_σ      : ' + ' '.join(f'{TR["s_k"][:, q].mean():7.4f}'
                               for q in range(0, L_, 4)))
print(f'\n🔑 РОЗКИД КОНСТАНТИ МІЖ МЕРЕЖАМИ — чи взагалі потрібна МОДЕЛЬ?')
print('   Якщо розкид ≈ 0, константа СПІЛЬНА ⟹ зашивається в сабміт числом,')
print('   і жодної мікромережі не треба. Якщо великий — модель має її вгадати.')
print('шар          : ' + ' '.join(f'{q + 1:7d}' for q in range(0, L_, 4)))
print('k_m std      : ' + ' '.join(f'{TR["m_k"][:, q].std():7.4f}'
                                   for q in range(0, L_, 4)))
print('k_σ std      : ' + ' '.join(f'{TR["s_k"][:, q].std():7.4f}'
                                   for q in range(0, L_, 4)))
print('k_m відхил.% : ' + ' '.join(f'{TR["m_k"][:, q].std() / TR["m_k"][:, q].mean():6.2%} '
                                   for q in range(0, L_, 4)))
fm = 1.0 - (TR['m_res'][:, L_ - 1].mean() / TR['m_err'][:, L_ - 1].mean()) ** 2
fs = 1.0 - (TR['s_res'][:, L_ - 1].mean() / TR['s_err'][:, L_ - 1].mean()) ** 2
print(f'\n📣(г) частка МАСШТАБНОЇ дисперсії на шарі 31: m {fm:.1%} · σ {fs:.1%}'
      f'   (передбачав >60% для m)')
