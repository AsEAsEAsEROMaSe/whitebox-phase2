"""ДРУГА КАЛІБРОВКА САБМІТА — ВАГА БЛЕНДУ `_W_BLEND`. ЧОМУ ВОНА СКАЛЯР? (С88)

## 🔑 ЗВІДКИ ПИТАННЯ
`estimator_sub97.py:372`  `out = ours + w·((1+α)·m_deep − ours)`, `w = 0.0883`.
Тобто в сабміті ДВІ калібрувальні константи: `α` (масштаб замикання) і `w`
(скільки замикання підмішувати). Обидві — ОДИН скаляр на всі 256 нейронів.

## 📐 ЧОМУ СКАЛЯР ТУТ ЗАВІДОМО НЕОПТИМАЛЬНИЙ
`MSE_j(w) = (1−w)²·v_m,j + w²·v_c,j` ⟹ `w*_j = v_m,j/(v_m,j+v_c,j)`.
Оптимум ПОНЕЙРОННИЙ, бо обидві дисперсії понейронні і — головне —
**ПРОТИЛЕЖНО НАПРАВЛЕНІ**:
* `v_c,j` (зміщення замикання) сидить у зоні `|t|<2`: 31.6% нейронів = 96.3%
  похибки, а при `t>2` замикання точне до 0.017%
  ([[finding-sigma-error-is-a-geometric-sum-of-offdiagonal-step-error]]);
* `v_m,j` (шум МК) ∝ `Var(a_j)`, і він НАЙБІЛЬШИЙ саме там, де `t` великий
  (нейрон завжди відкритий), тобто ТАМ, ДЕ ЗАМИКАННЯ ТОЧНЕ.
⟹ там, де МК шумить, замикання майже точне, і навпаки. Один спільний `w` —
компроміс між двома режимами, яких можна не змішувати.

🔑 І `t_j`, і `Var(a_j)` замикання рахує САМО, безкоштовно (`VC`).

## СХЕМИ
 C0  w=0                    чиста МК
 C1  один глобальний w      ЯК У САБМІТІ (LOO по мережах)
 C2  ⛔ОРАКУЛ w_j = v_m,j/(v_m,j+b_j²), `b_j` — справжнє зміщення (СТЕЛЯ)
 C3  w_j з ПРОФІЛЮ ĝ(t)     реалізовно: v̂_c,j = ĝ(t_j)·s_j², v̂_m,j = VC_j/N
 C4  дві корзини |t|<2/≥2   найдешевша реалізація
 ⛔C3x  профіль на ПЕРЕМІШАНОМУ t (контроль: чи це інформація, а не згладжування)
 ⛔C3n  w_j = 1 − w*_j       (перевернутий контроль)

## ⛔ПЛОМБИ
 P1 C1 з w≈0.09 мусить дати покращення ≈6% проти C0 (`blendyn.py`: без бленду
    0.9404×). Якщо ні — проксі `ours` не той об'єкт.
 P2 C3 ≤ C2 завжди (реалізовне не може бити оракула).
 P3 ⛔C3x мусить бути гіршим за C3.
 🚨 ЗАСТЕРЕЖЕННЯ: `ours` тут = ЧИСТА МК, а в сабміті `ours` має канали і в ~3×
    точніший ⟹ рівень `v_m` інший. Тому таблиця рахується на ДВОХ рівнях шуму
    (1 жереб і середнє з 3), і читати треба ВІДНОШЕННЯ всередині рівня.
"""
import math, sys
import numpy as np

F = sys.argv[1] if len(sys.argv) > 1 else '/mnt/win_d/llm/White-Box/side66/calib88.npz'
D = np.load(F)
YA, YB, B = D['YA'], D['YB'], D['base']
MCP, VC, TL, SL = D['MCP'], D['VC'], D['TL'], D['SL']
N, W_ = YA.shape
NREP = MCP.shape[1]
NMC = int(D['NMC'])
YM = 0.5 * (YA + YB); TTc = (YA * YB).sum(1)
NRM = TTc.mean() / W_

# ── α як у сабміті: один глобальний масштаб замикання, LOO по мережах ──
pt = (B * YM).sum(1); pp = (B * B).sum(1)
ALP = np.zeros(N)
for i in range(N):
    m = np.arange(N) != i
    ALP[i] = pt[m].sum() / pp[m].sum()
CL = ALP[:, None] * B
# ⛔незміщений квадрат зміщення замикання (крос-ціль)
BI2 = np.maximum((CL - YA) * (CL - YB), 0.0)
print(f'{N} мереж · {NREP} жеребів МК по {NMC} · α−1 = {ALP.mean()-1:+.5f}')
print(f'⟨yA,yB⟩/W = {NRM:.5f}\n')


def mse(out):
    """крос-цільова MSE, усереднена по мережах і жеребах."""
    return float(((out - YA[:, None, :]) * (out - YB[:, None, :])).mean())


def make_ours(k):
    """`ours` із шумом, зменшеним у k разів по дисперсії (середнє k жеребів)."""
    r = NREP // k
    return MCP[:, :r * k, :].reshape(N, r, k, W_).mean(2)


# ── профіль ĝ(t) = E[(зміщення/s)²] у корзинах по t, LOO по мережах ──
ED = np.array([-9, -2.5, -1.5, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 9.0])
NB = len(ED) - 1
IDX = np.clip(np.digitize(TL, ED) - 1, 0, NB - 1)
Z = BI2 / np.maximum(SL * SL, 1e-30)
SUM = np.zeros((N, NB)); CNT = np.zeros((N, NB))
for n in range(N):
    np.add.at(SUM[n], IDX[n], Z[n]); np.add.at(CNT[n], IDX[n], 1.0)
rng = np.random.default_rng(7)
IDXs = np.array([rng.permutation(IDX[n]) for n in range(N)])   # ⛔контроль


def profile_w(idx, k):
    """w_j із профілю, навченого LOO по мережах."""
    Wt = np.zeros((N, W_))
    for n in range(N):
        m = np.arange(N) != n
        g = np.where(CNT[m].sum(0) > 20, SUM[m].sum(0) / np.maximum(CNT[m].sum(0), 1), 0.0)
        vc = g[idx[n]] * SL[n] ** 2
        vm = VC[n] / (NMC * k)
        Wt[n] = vm / np.maximum(vm + vc, 1e-30)
    return Wt


print('═' * 74)
print('ВАГА БЛЕНДУ: СКАЛЯР ПРОТИ ПОНЕЙРОННОЇ')
print(f'{"рівень шуму":12s} {"схема":22s} {"MSE":>11s} {"віднос.":>9s} {"×C1":>8s}')
for k in (1, 2, 4):
    OU = make_ours(k)
    # C1: один глобальний w, LOO по мережах
    num = ((OU - CL[:, None, :]) * (0.5 * (YA + YB)[:, None, :] - CL[:, None, :])).sum((1, 2))
    den = ((OU - CL[:, None, :]) ** 2).sum((1, 2))
    W1 = np.zeros(N)
    for i in range(N):
        m = np.arange(N) != i
        W1[i] = 1.0 - num[m].sum() / den[m].sum()
    rows = [('C0 чиста МК', OU),
            ('C1 скаляр w (сабміт)', (1 - W1[:, None, None]) * OU + W1[:, None, None] * CL[:, None, :])]
    vm_or = MCP.var(1, ddof=1)[:, None, :] * (NREP / max(NREP // k, 1)) / max(NREP // k, 1)
    vm_or = MCP.var(1, ddof=1)[:, None, :] / k
    wor = vm_or / np.maximum(vm_or + BI2[:, None, :], 1e-30)
    rows.append(('⛔C2 ОРАКУЛ w_j', (1 - wor) * OU + wor * CL[:, None, :]))
    for nm, idx in (('C3 профіль ĝ(t)', IDX), ('⛔C3x перемішаний t', IDXs)):
        wp = profile_w(idx, k)[:, None, :]
        rows.append((nm, (1 - wp) * OU + wp * CL[:, None, :]))
    wp = profile_w(IDX, k)[:, None, :]
    rows.append(('⛔C3n знак 1−w', wp * OU + (1 - wp) * CL[:, None, :]))
    base_c1 = mse(rows[1][1])
    for nm, o in rows:
        v = mse(o)
        print(f'{"×1/"+str(k)+" дисп.":12s} {nm:22s} {v:11.4e} '
              f'{math.sqrt(max(v,0)/NRM):9.4%} {"x%.4f" % (base_c1/v):>8s}')
    print(f'{"":12s} {"(w скаляр = %.4f)" % W1.mean():22s}')
    print()
