"""ВАГА БЛЕНДУ ЗІ СПОСТЕРЕЖУВАНОГО ШУМУ — ГІЛКА С77, ЯКА НЕ ЗАПУСКАЛАСЬ (С83)

## ЗВІДКИ
[[finding-half-split-knows-level-not-deviation]] закінчується «📣 НЕ
ЗАПУСКАЛОСЬ». Там виведено, що оптимальна вага бленду спостережувана БЕЗ
істини. Алгебра точна:
    ours = T + e   (e — шум вибірки, E[e]=0, E‖e‖²=σ²)
    m'   = T + b   (b — зміщення замикання, ДЕТЕРМІНОВАНЕ, шуму не має)
    Δ = m' − ours = b − e
    ⟨a,Δ⟩ = ⟨e,b−e⟩ ⟹ E = −σ² ;  E‖Δ‖² = ‖b‖² + σ²
    ⟹ w* = σ²/(‖b‖²+σ²) = **σ²/E‖Δ‖²**  ← знаменник спостережуваний ПРЯМО
Отже вагу не вгадують — її лишається порахувати, знаючи власне σ².

## 🚨 ЧОМУ ρ=0.14 У `wpred.py` НЕ ЗАКРИВАЄ ЦЕ
Кандидат 1 там — рівно ця формула, але σ² бралась як `vout` із порту:
`Σ_j Var(a_j)/(2·npair)`, де Var рахується по ОБ'ЄДНАНИХ антитетичних
половинах `[relu(y); relu(−y)]`. Це дисперсія СУМІШІ, а не середнього по
ПАРАХ: для гаусового `y` перше дає 0.341·s², друге 0.182·s² ⟹ завищення
**1.87×**. Гірше: `vout` — дисперсія сирого `last`, а прогноз є
`ours = last − v`, і `v` теж шумна й корельована з `last`.
⟹ НЕ ЗМІРЯНО, чи 0.14 — властивість ФОРМУЛИ, чи властивість поганої σ̂².

## 🔑 ЯК МІРЯЄМО σ² ЧЕСНО І ЗА НУЛЬ ФЛОПІВ
Два прогони по `npair/2` замість одного по `npair` коштують ті самі флопи
(матмул лінійний по рядках). Тоді
    σ̂² = ‖ours_A − ours_B‖²/4
бо Var(A−B) = 4σ² при Var((A+B)/2) = σ². С77 виміряла точність такої оцінки
як 8%. 🔑 Вона знає РІВЕНЬ, не ВІДХИЛЕННЯ — а для ваги ПО МЕРЕЖІ треба саме
рівень, тож обмеження С77 тут не заважає.

## КАНАЛИ (48 мереж × 8 жеребів, банкова істина, крос-ціль)
  ⛔1 vout/‖Δ‖²            — ПЛОМБА: мусить відтворити ≈0.14 з `wpred.py`
  🎯2 півподіл/‖Δ‖²        — гілка С77
  🎯3 півподіл/‖Δ‖² усер.  — те саме, обидві частини усереднені по жеребах
  ⛔4 σ²_орак/‖Δ‖²         — σ² із РОЗКИДУ `ours` по жеребах ⟹ СТЕЛЯ ФОРМУЛИ
  ⛔5 НЕГ (зсунутий по мережах) · ⛔6 ЧИСТИЙ ШУМ — підлога

## 🚨 ДІАГНОСТИКА, ЯКОЇ БРАКУВАЛО ДЛЯ ЧИТАННЯ ЧИСЛА 0.28
`w_орак` сам оцінений на скінченних даних. Рахую його НАДІЙНІСТЬ: жереби 0-3
проти 4-7, корел + поправка Спірмена-Брауна `r_повн = 2r/(1+r)`.
🔑 Стеля будь-якої кореляції з `w_орак` = √надійність. Якщо надійність 0.3,
то стеля 0.55 — і тоді 0.28 близьке до МАКСИМУМУ, а не «сигналу нема».
🎓 Це правило «спершу ЕТАЛОН, тоді прилад»
([[feedback-controls-have-three-distinct-diseases]]).

## ВИГРАШ, А НЕ СТЕЛЯ
Модель `w(n) = a + k·(f(n) − f̄)` — дві корпусні константи, LOO по мережах.
При k=0 вироджується в нинішню константу ⟹ ГІРШЕ бути не може за побудовою
(на навчальних; LOO чесний). MSE(w) розкладається точно:
    ⟨e1,e2⟩ = Z0 + w·NUM + w²·DEN,  Z0=⟨a1,a2⟩, NUM=⟨a1,Δ⟩+⟨a2,Δ⟩, DEN=‖Δ‖²

📣 ПЕРЕДБАЧЕНО: пломба відтворить 0.14. Півподіл дасть корел 0.3…0.5 — краще
   за `vout`, бо міряє ту саму величину, що й потрібна. Оракульна σ² покаже
   стелю формули; чекаю 0.5…0.7. Надійність `w_орак` — 0.4…0.6, тобто стеля
   кореляції 0.63…0.77, і тоді нинішні 0.28 — це справді мало, є куди рости.
   🎯 Виграш чекаю +0.3…0.8% (частина від стелі бленду ×1.0243). Якщо ж
   оракульна σ² теж дає ≤0.2 — формула НЕ ТА, і гілка закривається остаточно,
   бо кращої σ² за оракульну не буває.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/wsplit.py [cuda:N] [NETS] [NDRAW]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, predict91,      # noqa: E402
                        W_BLEND, ALPHA)
# 🚨 ІСТИНА З БАНКУ, НІКОЛИ `truth_last` ([[reference-bank-truth-is-the-right-target]]):
#    `truth_last(wc)` міряє ОБРІЗАНУ мережу, грейдер — ПОВНУ, зсув 1.5e-04…6.1e-04
#    більший за обидва шуми. Плюс банк у ~30× швидший.
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L_ = 32
NAMES = ['⛔1 vout/‖Δ‖²  (пломба ≈0.14)', '🎯2 півподіл/‖Δ‖², усер. по жеребах',
         '🎯3 півподіл/‖Δ‖² з ОДНОГО жеребу (як у сабміті)',
         '⛔4 σ²_ОРАКУЛ/‖Δ‖²  (СТЕЛЯ ФОРМУЛИ)',
         '⛔5 НЕГ зсунутий', '⛔6 ЧИСТИЙ ШУМ']
NFT = len(NAMES)
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
NETS = len(FS)

X = np.zeros((NETS, NFT))
# покомпонентні суми для точного MSE(w) та для LOO
Z0 = np.zeros(NETS); NUM = np.zeros(NETS); DEN = np.zeros(NETS)
# половини жеребів — для НАДІЙНОСТІ w_орак
NUM_H = np.zeros((NETS, 2)); DEN_H = np.zeros((NETS, 2))
SEAL_SPLIT = []
rng0 = np.random.default_rng(11)
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, 8, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        s_vo = s_sp = s_dn = 0.0
        OU = []
        for d in range(NDRAW):
            sd = 77_000_000 + 733 * n + d
            pr, _, _, (ours, m_deep, vout) = predict91(
                wc, npair, DEV, seed=sd, cl=cl, diag=True, raw=True)
            dl = (1.0 + ALPHA) * m_deep - ours
            a1, a2 = ours - T1, ours - T2
            z0 = float((a1 * a2).sum())
            nu = float((a1 * dl).sum()) + float((a2 * dl).sum())
            de = float((dl * dl).sum())
            Z0[n] += z0; NUM[n] += nu; DEN[n] += de
            h = 0 if d < NDRAW // 2 else 1
            NUM_H[n, h] += nu; DEN_H[n, h] += de
            OU.append(ours)
            # 🔑 ПІВПОДІЛ: два прогони по npair/2 = ті самі флопи, що один
            #    по npair. Різні seed ⟹ незалежні половини.
            _, _, _, (oA, _, _) = predict91(wc, npair // 2, DEV,
                                            seed=sd * 2 + 1, cl=cl,
                                            diag=True, raw=True)
            _, _, _, (oB, _, _) = predict91(wc, npair // 2, DEV,
                                            seed=sd * 2 + 2, cl=cl,
                                            diag=True, raw=True)
            s2sp = float(((oA - oB) ** 2).sum()) / 4.0
            s_vo += vout; s_sp += s2sp; s_dn += de
            if d == 0:
                # 🚨 У САБМІТІ ЖЕРЕБ ОДИН. Усереднення по NDRAW завищує якість
                #    ознаки — тож канал 3 бере рівно те, що доступне наживо.
                X[n, 2] = s2sp / max(de, 1e-300)
                SEAL_SPLIT.append((s2sp, vout, de))
        # σ²_ОРАКУЛ: розкид `ours` по жеребах (незміщена оцінка по NDRAW)
        OU = torch.stack(OU)
        s2_or = float(OU.var(0, unbiased=True).sum())
        X[n, 0] = s_vo / max(s_dn, 1e-300)
        X[n, 1] = s_sp / max(s_dn, 1e-300)
        X[n, 3] = s2_or / max(s_dn / NDRAW, 1e-300)
        X[n, 5] = rng0.standard_normal()
    del W, wc, OU
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

X[:, 4] = np.roll(X[:, 1], 1)                     # ⛔НЕГ

w_or = -NUM / (2.0 * np.maximum(DEN, 1e-300))
w_h = [-NUM_H[:, h] / (2.0 * np.maximum(DEN_H[:, h], 1e-300)) for h in (0, 1)]

print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, банкова істина.')
print(f'🔑 w_орак: сер {w_or.mean():.4f} розкид {w_or.std():.4f} '
      f'[{w_or.min():.4f}…{w_or.max():.4f}]   (сабміт {W_BLEND}, sub92 0.0883)')

# ── ⛔ ПЛОМБА ПІВПОДІЛУ: σ̂²_split мусить бути ТОГО САМОГО ПОРЯДКУ, що σ²_орак
r_sp = X[:, 1] / np.maximum(X[:, 3], 1e-300)
r_vo = X[:, 0] / np.maximum(X[:, 3], 1e-300)
print(f'\n⛔ ПЛОМБА ОЦІНОК σ²  (відношення до ОРАКУЛЬНОЇ, медіана по мережах)')
print(f'   півподіл / оракул = {np.median(r_sp):.3f}   '
      f'{"✅" if 0.7 < np.median(r_sp) < 1.4 else "🚨 зміщена"}')
print(f'   vout     / оракул = {np.median(r_vo):.3f}   '
      f'{"(очікувалось ~1.9 через суміш антитетичних)" if r_vo.mean()>1.4 else ""}')
print(f'   📐 корел(півподіл, оракул) ПО МЕРЕЖАХ = '
      f'{np.corrcoef(X[:, 1], X[:, 3])[0, 1]:+.3f}  '
      f'⟸ чи вловлює півподіл РОЗКИД, а не лише рівень')

# ── 🔑 ПЛОМБА САМОЇ ТЕОРІЇ: формула каже w* = σ²/‖Δ‖² ТОЧНО, множник 1.
#    Якщо середнє каналу збігається з середнім `w_орак` — теорія стоїть.
print(f'\n🔑 ПЛОМБА ТЕОРІЇ `w* = σ²/‖Δ‖²` (множник мусить бути ≈1, не підганяється)')
for j in (0, 1, 2, 3):
    print(f'   {NAMES[j]:44s} сер {X[:, j].mean():.4f} '
          f'⟹ множник до w_орак {w_or.mean()/max(X[:, j].mean(),1e-30):6.3f}')

# ── 🚨 НАДІЙНІСТЬ ЕТАЛОНА
rh = float(np.corrcoef(w_h[0], w_h[1])[0, 1])
rel = 2.0 * rh / (1.0 + rh) if rh > -1 else float('nan')
print(f'\n🚨 НАДІЙНІСТЬ ЕТАЛОНА `w_орак` (половини жеребів, Спірмен-Браун)')
print(f'   корел половин {rh:+.3f} ⟹ надійність {rel:+.3f} ⟹ '
      f'**СТЕЛЯ БУДЬ-ЯКОЇ КОРЕЛЯЦІЇ {np.sqrt(max(rel,0)):.3f}**')

se = 1.0 / np.sqrt(max(NETS - 3, 1))
print(f'\n{"канал":38s} {"корел":>8s} {"корел/стеля":>12s}   (похибка ±{se:.2f})')
print('─' * 66)
CR = np.zeros(NFT)
for j, nm in enumerate(NAMES):
    if X[:, j].std() < 1e-300:
        continue
    CR[j] = float(np.corrcoef(X[:, j], w_or)[0, 1])
    fr = (f'{CR[j]/np.sqrt(rel):+12.3f}' if rel > 0.01 else f'{"—":>12s}')
    print(f'{nm:38s} {CR[j]:+8.3f} {fr}'
          + ('  🟢' if abs(CR[j]) > 2 * se else ''))


# ── 🎯 ВИГРАШ: w(n) = a + k·(f(n) − f̄), LOO ПО МЕРЕЖАХ
def mse_of(wv):
    return float(np.sum(Z0 + wv * NUM + wv * wv * DEN))


def fit_ak(f, idx):
    u = f[idx] - f[idx].mean()
    A = np.array([[np.sum(DEN[idx]), np.sum(u * DEN[idx])],
                  [np.sum(u * DEN[idx]), np.sum(u * u * DEN[idx])]])
    b = -0.5 * np.array([np.sum(NUM[idx]), np.sum(u * NUM[idx])])
    try:
        return np.linalg.solve(A, b), f[idx].mean()
    except np.linalg.LinAlgError:
        return np.array([w_or.mean(), 0.0]), f[idx].mean()


base_c = mse_of(np.full(NETS, W_BLEND))
base_92 = mse_of(np.full(NETS, 0.0883))
orac = mse_of(w_or)
print(f'\n🎯 ВИГРАШ (LOO по мережах, модель `w = a + k·(f − f̄)`)')
print(f'   база sub91 (w={W_BLEND})           {base_c:.6e}   ×1.0000')
print(f'   база sub92 (w=0.0883)              {base_92:.6e}   '
      f'×{base_c/base_92:.4f}')
print(f'   ⛔ОРАКУЛ (стеля персоналізації)     {orac:.6e}   '
      f'×{base_92/orac:.4f} над sub92')
print(f'\n{"канал":38s} {"LOO MSE":>13s} {"× над sub92":>12s}')
print('─' * 66)
for j, nm in enumerate(NAMES):
    if X[:, j].std() < 1e-300:
        continue
    wv = np.zeros(NETS)
    for n in range(NETS):
        idx = np.array([m for m in range(NETS) if m != n])
        (a, k), fb = fit_ak(X[:, j], idx)
        wv[n] = a + k * (X[n, j] - fb)
    wv = np.clip(wv, 0.0, 0.5)
    m = mse_of(wv)
    print(f'{nm:38s} {m:13.6e} {base_92/m:12.4f}'
          + ('  🟢' if base_92 / m > 1.002 else ''))

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/wsplit.npz',
         X=X, w_or=w_or, Z0=Z0, NUM=NUM, DEN=DEN,
         NUM_H=NUM_H, DEN_H=DEN_H, CR=CR)
print(f'\n📊 БАЛ 1.954e-07 (sub87) · sub92 ≈1.946e-07 · '
      f'стеля бленду ×{base_92/orac:.4f} ⟹ {1.946e-07/(base_92/orac):.4e}')
