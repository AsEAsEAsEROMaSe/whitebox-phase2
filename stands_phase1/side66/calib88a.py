"""ОФЛАЙН-АЛГЕБРА КАЛІБРОВОК на векторах із `calib88np.py`. (С88)

## 📐 ТОТОЖНІСТЬ, ЩО ПОЯСНЮЄ ВСІ ЧОТИРИ ПРОВАЛИ
```
c_n = κ_n·b_n + c⊥_n ,   κ_n = ⟨c_n,b_n⟩/⟨b_n,b_n⟩
α·(b_n + c_n) = α(1+κ_n)·b_n + α·c⊥_n
```
⟹ при вазі поправки, ЖОРСТКО ПРИБИТІЙ до 1, ефективний масштаб мережі стає
`α(1+κ_n)` і ГУЛЯЄ ПО МЕРЕЖАХ. Глобальний скаляр `α` цього не відстежує ⟹
поправка ДОДАЄ розкид масштабу, і він з'їдає її ж виграш у формі.
🔑 Ортогоналізація = примусове `κ_n≡0`. Це ПОНЕЙМЕРЕЖНА операція, тобто те,
чого глобальний скаляр не вміє В ПРИНЦИПІ. Ось чому вона взагалі щось дає.
🔑 АЛЕ вона ж і ВИКИДАЄ інформацію: якщо `κ_n` корелює зі справжнім
понейромережним масштабом `ν_n`, то `κ` — це ПРЕДИКТОР, а не сміття. Тоді
правильна схема не «викинути κ», а «дати κ власний коефіцієнт».

## СХЕМИ (усі — глобальні коефіцієнти, LOO по мережах)
 A0   p = α·b                    база
 A0m  α = mean_n ν_n             рецепт наших стендів (чи він гірший за МНК)
 A1   p = α·(b+c)                ПОТОЧНА ПРАКТИКА (вага поправки прибита до 1)
 A2   p = α·(b+c⊥)               ортогоналізована (gcut87)
 A3   p = α·b + β·c⊥             🆕 спільний МНК; нормальні рівняння БЛОЧНІ
 A4   p = α·b + β·c              🆕 спільний МНК без ортогоналізації
 A5   p = (α+λκ)·b               🆕 ЛИШЕ масштабна персоналізація з κ
 A6   p = (α+λκ)·b + β·c⊥        🆕 повна: κ у масштаб, форма окремо
 ORC  p = ν_n·b                  ⛔оракульний понейромережний масштаб (стеля)

## 🚨 КРОС-ЦІЛЬ
`MSE(θ) = ⟨Xθ−yA, Xθ−yB⟩/W` — незміщена при будь-якому скінченному МК.
"""
import math, sys
import numpy as np

F = sys.argv[1] if len(sys.argv) > 1 else '/mnt/win_d/llm/White-Box/side66/calib88.npz'
D = np.load(F)
YA, YB = D['YA'], D['YB']
N, W_ = YA.shape
YM = 0.5 * (YA + YB)
TTc = (YA * YB).sum(1)                    # крос-оцінка ⟨y,y⟩ (незміщена)
NRM = TTc.mean() / W_
TAGS = [k for k in D.files if k not in ('YA', 'YB', 'seeds', 'MCP', 'VA',
                                        'TL', 'SL', 'VC', 'NMC')]
print(f'{N} мереж · ⟨yA,yB⟩/W = {NRM:.5f}  '
      f'(nrmcheck87 на 12 мережах давав 0.88404; величина дуже розкидана '
      f'по мережах, тож розбіжність — це вибірка мереж, не прилад)\n')


def loo_fit(cols, ret_th=False):
    k = len(cols)
    G = np.zeros((N, k, k)); H = np.zeros((N, k))
    for a in range(k):
        H[:, a] = (cols[a] * YM).sum(1)
        for b_ in range(k):
            G[:, a, b_] = (cols[a] * cols[b_]).sum(1)
    out = np.zeros(N); TH = np.zeros((N, k))
    for i in range(N):
        m = np.arange(N) != i
        A = G[m].sum(0)
        th = np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(k), H[m].sum(0))
        TH[i] = th
        out[i] = (th @ G[i] @ th - 2.0 * th @ H[i] + TTc[i]) / W_
    return (out.mean(), TH) if ret_th else out.mean()


B = D['base']
bb = (B * B).sum(1)
nu = (B * YM).sum(1) / bb


def loo_mean_alpha():
    pp = bb; pt = (B * YM).sum(1); out = np.zeros(N)
    a_n = pt / pp
    for i in range(N):
        m = np.arange(N) != i
        a = a_n[m].mean()
        out[i] = (a * a * pp[i] - 2 * a * pt[i] + TTc[i]) / W_
    return out.mean()


base_mse, TH0 = loo_fit([B], ret_th=True)
a0 = TH0[:, 0].mean()
# розклад: MSE(α) = ‖залишок‖² + (α−ν)²‖b‖²
shape = (TTc - nu * nu * bb) / W_
scale = ((a0 - nu) ** 2 * bb) / W_
print('═' * 76)
print('ЧАСТИНА 1 · З ЧОГО СКЛАДЕНА ПОХИБКА ЗАМИКАННЯ')
raw = ((B - YA) * (B - YB)).mean()
print(f'  сира (без α)        MSE {raw:.4e}   віднос. {math.sqrt(raw/NRM):.4%}')
print(f'  A0 після α (МНК)    MSE {base_mse:.4e}   віднос. '
      f'{math.sqrt(base_mse/NRM):.4%}   α−1 = {a0-1:+.5f}')
am = loo_mean_alpha()
print(f'  A0m рецепт стендів  MSE {am:.4e}   ×{am/base_mse:.4f} проти МНК')
print(f'  ⤷ РОЗКЛАД A0: ФОРМА {shape.mean():.4e} '
      f'({shape.mean()/(shape.mean()+scale.mean()):.1%}) · '
      f'РОЗКИД МАСШТАБУ {scale.mean():.4e} '
      f'({scale.mean()/(shape.mean()+scale.mean()):.1%})')
print(f'  ⤷ ⛔ОРАКУЛ ν_n (понейромережний масштаб): MSE {shape.mean():.4e} '
      f'⟹ СТЕЛЯ персоналізації масштабу ×{base_mse/shape.mean():.3f}')
print(f'  ⤷ ν: {nu.mean():.5f} ± {nu.std():.5f}  ({nu.std()/nu.mean():.2%})')

print('\n' + '═' * 76)
print('ЧАСТИНА 2 · СХЕМИ КАЛІБРОВКИ (LOO по мережах, усе ПІСЛЯ α)')
print(f'{"поправка":9s} {"сира%":>8s} {"A1 вага1":>9s} {"A2 ⊥":>9s} '
      f'{"A3 α,β⊥":>9s} {"A4 α,β":>9s} {"A5 κ":>9s} {"A6 повна":>9s}')
res = {}
for t in TAGS:
    if t == 'base':
        continue
    Pv = D[t]; C = Pv - B
    kap = (C * B).sum(1) / bb
    CO = C - kap[:, None] * B
    r = {'A1': loo_fit([B + C]), 'A2': loo_fit([B + CO]),
         'A4': loo_fit([B, C]), 'A5': loo_fit([B, kap[:, None] * B])}
    m3, TH3 = loo_fit([B, CO], ret_th=True); r['A3'] = m3
    r['A6'] = loo_fit([B, kap[:, None] * B, CO])
    r['_a3'] = TH3[:, 0].mean(); r['_b3'] = TH3[:, 1].mean()
    r['_kap'] = (kap.mean(), kap.std())
    res[t] = r
    rr = ((Pv - YA) * (Pv - YB)).mean()
    print(f'{t:9s} {math.sqrt(max(rr,0)/NRM):8.4%} ' + ' '.join(
        f'x{base_mse/r[k]:8.3f}' for k in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6')))

print('\n⛔ПЛОМБА БЛОЧНОЇ ДІАГОНАЛЬНОСТІ: α(A3) мусить = α(A0) (бо ⟨b,c⊥⟩≡0)')
for t in TAGS:
    if t == 'base':
        continue
    print(f'  {t:9s} α(A3)−α(A0) = {res[t]["_a3"]-a0:+.3e} · β={res[t]["_b3"]:+.4f}'
          f' · κ = {res[t]["_kap"][0]:+.5f} ± {res[t]["_kap"][1]:.5f}')

print('\n' + '═' * 76)
print('ЧАСТИНА 3 · ЧИ Є В `κ` ІНФОРМАЦІЯ ПРО ПОНЕЙМЕРЕЖНИЙ МАСШТАБ')
for t in TAGS:
    if t == 'base':
        continue
    C = D[t] - B
    kap = (C * B).sum(1) / bb
    c = np.corrcoef(kap, nu)[0, 1]
    print(f'  {t:9s} corr(κ_n, ν_n) = {c:+.4f}')
