"""ЧИ ДОДАЄ 2-Й МОМЕНТ ПОНАД μ-ЯКІР, ЧИ ДУБЛЮЄ ЙОГО? (С82)

`infoscan_x.py` знайшов: якір на **2-й момент активацій шару 24** знімає
**10.07%** крос-MSE поодинці — БІЛЬШЕ за штатний μ-якір того ж шару (7.39%).
Пломби зійшлись (ланцюг переносу corr=1.000000, пломба потужності
9.32·9.32·9.32 ⟹ прилад зрячий).

## 🚨 АЛЕ ПООДИНЦІ — НЕ ВІДПОВІДЬ
`E[a²]` і `E[a]` для ReLU сильно корельовані (обидва ростуть із масштабом
нейрона). Тож 10.07% може бути ТИМ САМИМ каналом, лише в кращих одиницях —
тоді додавання ознаки не дасть нічого, а ми вже маємо вирок
[[finding-lambda-absorbs-any-mechanism-you-fix]].
Питання одне: **скільки дають ДВІ разом понад кожну окремо.**

## ЩО МІРЯЄМО (LOO ПО МЕРЕЖАХ, крос-ціль, дані з `infoscan_x.npz`)
  1 · сам μ-якір ℓ           2 · сам 2-й момент ℓ
  3 · ОБИДВА разом           4 · + σ-ознака ℓ (три)
  ⛔контроль: μ-якір + ЙОГО Ж копія зі зсувом (має дати рівно стільки ж, як 1)
  📐корel(μ-якір, 2-й момент) — прямо показує, чи це один канал

📣 ПЕРЕДБАЧЕНО: кореляція буде >0.9, і «обидва разом» дадуть ≤+1% понад
   найкращий поодинці ⟹ це той самий канал у кращих одиницях. Якщо ж
   приріст ≥+3% — це НОВА ознака, і вона просто додається в λ-регресію.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/infoscan_join.py
"""
import numpy as np

D = np.load('/mnt/win_d/llm/White-Box/arena62/logs/infoscan_x.npz')
X, e1, e2, net = D['X'].astype(np.float64), D['e1'], D['e2'], D['net']
NETS = int(net.max()) + 1
LAYERS = (0, 1, 3, 7, 15, 23, 30)
NL = len(LAYERS)
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

# розкладка колонок `infoscan_x`: [a]*7, [ind]*7, [a2]*7, [p3]*7, [p4]*7,
# потім μ-якір24(еталон), σ-ознака24(еталон), 2 пломбові
IA, IIND, IA2, IP3, IP4 = 0, NL, 2 * NL, 3 * NL, 4 * NL


def fit(cols, a1, a2):
    """LOO по мережах, БАГАТОВИМІРНА лінійна регресія на `cols`."""
    M = np.stack([X[:, c] for c in cols] + [np.ones(len(a1))], 1)
    ee = 0.5 * (a1 + a2)
    base = float(np.mean(a1 * a2))
    fitv = np.zeros(len(a1))
    for n in range(NETS):
        te, tr = net == n, net != n
        A, b = M[tr], ee[tr]
        G = A.T @ A + 1e-9 * np.eye(M.shape[1]) * float(np.trace(A.T @ A))
        w = np.linalg.solve(G, A.T @ b)
        fitv[te] = M[te] @ w
    return 100.0 * (1.0 - float(np.mean((a1 - fitv) * (a2 - fitv))) / base)


print(f'{"шар":>5s} {"corr":>7s} │{"μ-якір":>9s}{"2-й мом.":>10s}{"ОБИДВА":>9s}'
      f'{"+σ":>8s} │{"приріст":>9s}')
print('─' * 62)
for k, l in enumerate(LAYERS):
    ja, jq, js = IA + k, IA2 + k, None
    u, v = X[:, ja], X[:, jq]
    c = float((u - u.mean()) @ (v - v.mean())
              / (np.linalg.norm(u - u.mean()) * np.linalg.norm(v - v.mean())))
    f1, f2 = fit([ja], e1, e2), fit([jq], e1, e2)
    f3 = fit([ja, jq], e1, e2)
    f4 = fit([ja, jq, IIND + k], e1, e2)
    print(f'{l+1:5d} {c:7.3f} │{f1:8.2f}%{f2:9.2f}%{f3:8.2f}%{f4:7.2f}% │'
          f'{f3 - max(f1, f2):+8.2f}%')

# ⛔ КОНТРОЛЬ: та сама ознака двічі (з мікрозсувом) — приріст мусить бути ≈0
j = IA + 5
X = np.concatenate([X, (X[:, j] * 1.0000001 + 1e-12)[:, None]], 1)
c0, c1 = fit([j], e1, e2), fit([j, X.shape[1] - 1], e1, e2)
print(f'\n⛔ КОНТРОЛЬ ДУБЛЯ (та сама ознака двічі): {c0:.2f}% → {c1:.2f}%  '
      f'{"✅" if abs(c1 - c0) < 0.3 else "🚨 РЕГРЕСІЯ ТЕЧЕ"}')

ALL_A = [IA + k for k in range(NL)]
ALL_Q = [IA2 + k for k in range(NL)]
print(f'\nУСІ 7 ШАРІВ РАЗОМ: лише μ-якорі {fit(ALL_A, e1, e2):.2f}%  ·  '
      f'лише 2-й момент {fit(ALL_Q, e1, e2):.2f}%  ·  '
      f'РАЗОМ {fit(ALL_A + ALL_Q, e1, e2):.2f}%')
print(f'📊 БАЛ 1.954e-07')
