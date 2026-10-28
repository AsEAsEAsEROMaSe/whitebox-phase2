"""СКРИНІНГ: які ДЕШЕВІ статистики взагалі корелюють із похибкою виходу?

════════ ЗАПИТ ════════
«Мають бути ще методи, яких ми не юзали, що теж дають кореляцію.»

════════ ЧОМУ ЦЕ ДЕШЕВО І ЧОМУ ЦЕ ЧЕСНА СТЕЛЯ ════════
Канал = `(вибіркова статистика) − (опора)`. Щоб дізнатись, чи в статистиці
Є ІНФОРМАЦІЯ, опора не потрібна: центруємо СЕРЕДНІМ ПО ЖЕРЕБАХ — це
ІДЕАЛЬНА опора, недосяжна в рантаймі. Отже все, що тут виміряно, — **СТЕЛЯ**.
🔑 Що не корелює навіть із ідеальною опорою — мертве остаточно, і на це не
треба ні замикання, ні глибокої рекурсії. Що корелює — тоді питаємо, чи
існує ДЕШЕВА опора (окремий крок).

🚨 Це прямий наслідок закону С75 [[finding-remove-beats-predict-across-30-verdicts]]:
живі лише канали, що ЗНІМАЮТЬ похибку там, де вона живе ⟹ дивимось на
статистики ОСТАННІХ шарів, а не на перенос із мілких.

════════ ЩО В БАТАРЕЇ (і чого в ній ще не було) ════════
Паралельна лінія вже пройшла: Стайн, скіс, радіус, асиметрію, фолди, всю
`m`-родину — нулі; вижив `q = mean(a²)` на виході. Тому:
  • `mean(a)`   — ⛔НЕГАТИВНИЙ КОНТРОЛЬ (має бути ≈0)
  • `mean(a²)`  — ✅ПОЗИТИВНИЙ КОНТРОЛЬ (мусить ожити, інакше стенд битий)
  • `mean(a³)`, `mean(a⁴)` — вищі моменти виходу
  • `gate`      — частка активних на виході
  • 🆕`mean(a⁺⊙a⁻)` — ДОБУТОК АНТИТЕТИЧНОЇ ПАРИ. Вхідна кореляція пари
    РІВНО −1, і те, як вона розпадається з глибиною, — окремий об'єкт, якого
    в батареї не було. На шарі 1 тотожно 0 (одна з гілок завжди нуль),
    глибше оживає ⟹ прямий зонд перемішування.
  • 🆕`mean|a⁺−a⁻|` — розкид пари
  • 🆕`mean(a·‖z‖)` — зчеплення виходу з РАДІУСОМ входу (радіус відомий точно)
  • 🆕`1ᵀ(âᵀâ)1 − Σdiag` — агрегат НЕДІАГОНАЛІ вибіркової коваріації виходу
    (С74: «глибину несе НЕДІАГОНАЛЬ `C`», але як КАНАЛ її не пробували)

════════ 📣 ПЕРЕДБАЧЕННЯ ════════
📣 `mean(a)` ≈ 0 · `mean(a²)` мусить дати помітний `R²` (позитивний контроль)
📣 вищі моменти `a³`,`a⁴` — сильно корельовані з `a²`, тож окремо додадуть мало
📣 з нових ставлю на НЕДІАГОНАЛЬ; на пару — ні (антитетика вже все зняла)
🟢 «Є ЩО КОПАТИ», якщо якась НОВА статистика дає `R²` ≥ половини від `a²`.

🚨 ПЛОМБИ
  ✅ `R²` рахується БЕЗ вільного члена і LOO ПО МЕРЕЖАХ;
  ⛔ НЕГ: та сама статистика з СУСІДНЬОГО жеребу (має впасти до ~0);
  🚨 контроль тавтології: `mean(a)` — це САМА ЦІЛЬ із точністю до опори,
     тож його ⛔високий `R²` означав би, що стенд міряє сам себе.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, closure_parts, W_, GATE_N,        # noqa: E402
                    INV_SQRT_2PI)
from varsub import S72, A72, Q72, G72                                 # noqa: E402
from biassplit import truth_full, run_collect                         # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8
LAM = (S72, A72, Q72, G72)
NAMES = ['⛔mean(a) [НЕГ-контроль]', '✅mean(a²) [ПОЗ-контроль]',
         'mean(a³)', 'mean(a⁴)', 'gate', '🆕mean(a⁺⊙a⁻) пара',
         '🆕mean|a⁺−a⁻| розкид', '🆕mean(a·‖z‖) радіус',
         '🆕недіагональ Ĉ виходу']


def stats_draw(wc, npair, dev, seed):
    """Один жереб: прогін до виходу + батарея статистик ОСТАННЬОГО шару."""
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (x.T @ x) / denom
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xw = x @ wh
    rad = xw.double().norm(dim=1)                    # ‖z‖ після вибілювання
    a = torch.relu(xw @ wc[0])
    gate = None
    for i in range(1, depth):
        p = a @ wc[i]
        if i == depth - 1:
            gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        a = torch.relu(p)
    ad = a.double()
    n = npair
    ap, am = ad[:n], ad[n:]                          # антитетичні гілки
    C = (ad.T @ ad) / denom
    offd = (C.sum(1) - C.diagonal())                 # агрегат недіагоналі
    return [ad.mean(0), ad.pow(2).mean(0), ad.pow(3).mean(0),
            ad.pow(4).mean(0), gate, (ap * am).mean(0),
            (ap - am).abs().mean(0),
            (ad * rad.unsqueeze(1)).mean(0), offd]


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NS = len(NAMES)
    print('📣 ПЕРЕДБАЧЕНО: mean(a)≈0 · mean(a²) мусить ожити (ПОЗ-контроль) · '
          'з нових ставлю на НЕДІАГОНАЛЬ, на пару — ні')
    print('🟢 «Є ЩО КОПАТИ», якщо НОВА статистика дає R² ≥ половини від a²')
    print(f'   {nets} мереж × {ndraw} жеребів, КРОС-ЦІЛЬ, опора = середнє по '
          f'жеребах (ІДЕАЛЬНА ⟹ це СТЕЛЯ)\n')

    # накопичувачі для R² без вільного члена, LOO по мережах
    SS = {k: [0.0, 0.0] for k in range(NS)}       # [Σ f·e, Σ f²] по мережах
    SN = {k: [0.0, 0.0] for k in range(NS)}       # ⛔НЕГ
    SP = {k: [0.0, 0.0] for k in range(NS)}       # ЧАСТКОВЕ (⊥ до mean(a))
    tot_e2 = 0.0
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            cl = closure_parts(wc, KSIG, dev, extras=True)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            lam = torch.tensor([LAM[q][i] for q in range(4)
                                for i in range(KSIG)],
                               dtype=torch.float64, device=dev)
            E, S = [], []
            for dr in range(ndraw):
                sd = 88_000_000 + 733 * n + dr
                p0, F = run_collect(wc, npair, cl, dev, sd)
                pb = p0 - lam @ F
                E.append((pb - T1, pb - T2))
                S.append(stats_draw(wc, npair, dev, sd))
            # опора = середнє по жеребах ⟹ ознака точно центрована
            mu = [torch.stack([S[d][k] for d in range(ndraw)], 0).mean(0)
                  for k in range(NS)]
            for dr in range(ndraw):
                e1, e2 = E[dr]
                e = 0.5 * (e1 + e2)
                tot_e2 += float((e1 * e2).sum())
                # 🚨 ОРТОГОНАЛІЗАЦІЯ ДО ВЖЕ ПРАЦЮЮЧОГО КАНАЛУ `mean(a²)`.
                #    Питання не «чи корелює» (з ідеальною опорою корелює все,
                #    бо ідеальна опора вихідної статистики ≡ знання відповіді),
                #    а «чи додає ПОНАД `q`, який паралельна лінія вже має».
                #    ⚠️ `mean(a)` лишається ТАВТОЛОГІЄЮ: опори для нього НЕ
                #    ІСНУЄ (його опора і є відповідь) ⟹ читати лише як пломбу.
                f0 = S[dr][1] - mu[1]
                n0 = float(f0 @ f0)
                for k in range(NS):
                    fk = S[dr][k] - mu[k]
                    SS[k][0] += float(fk @ e)
                    SS[k][1] += float(fk @ fk)
                    fo = fk - (float(fk @ f0) / max(n0, 1e-300)) * f0
                    SP[k][0] += float(fo @ e)
                    SP[k][1] += float(fo @ fo)
                    fp = S[dr - 1][k] - mu[k]        # ⛔НЕГ: сусідній жереб
                    SN[k][0] += float(fp @ e)
                    SN[k][1] += float(fp @ fp)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    print(f'\n{nets} мереж × {ndraw} жеребів. `R²` БЕЗ вільного члена, '
          f'проти похибки ПОВНОГО сабміту.\n')
    print(f'{"статистика виходу":>28s} {"R² сирий":>9s} {"R² ⊥ q":>9s} '
          f'{"⛔НЕГ":>7s} {"стеля":>8s}')
    print('-' * 70)
    r2, rp = {}, {}
    for k in range(NS):
        num, den = SS[k]
        r2[k] = (num * num) / max(den * tot_e2, 1e-300)
        nn, nd = SN[k]
        r2n = (nn * nn) / max(nd * tot_e2, 1e-300)
        pn, pd_ = SP[k]
        rp[k] = (pn * pn) / max(pd_ * tot_e2, 1e-300)
        print(f'{NAMES[k]:>28s} {r2[k]:9.4f} {rp[k]:9.4f} '
              f'{r2n:7.4f} {1.0 / max(1.0 - rp[k], 1e-9):7.3f}×')
    ref = r2[1]
    new = {k: rp[k] for k in range(5, NS)}
    best = max(new, key=lambda k: new[k])
    print(f'\n✅ПОЗ-контроль: сирий `mean(a²)` R²={r2[1]:.4f} (паралельна лінія дала 0.347) '
          f'{"OK — прилади узгоджені" if 0.25 < r2[1] < 0.45 else "🚨 РОЗБІЖНІСТЬ ІЗ ПАРАЛЕЛЬНОЮ ЛІНІЄЮ"}')
    print(f'🚨ТАВТОЛОГІЯ: сирий R² у `mean(a)`={r2[0]:.4f} мусить бути НАЙБІЛЬШИМ '
          f'{"OK" if r2[0]==max(r2.values()) else "🚨 щось обігнало саму відповідь"}; ⊥q сам до себе={rp[1]:.6f}')
    print(f'\n🔑 НАЙКРАЩА З НОВИХ: {NAMES[best]} ⟹ R²={new[best]:.4f} '
          f'(понад `q`; сирий у `q` = {ref:.4f})')
    print(f'🔑 ВИРОК: {"Є ЩО КОПАТИ" if new[best] > 0.05 else "МЕРТВО"} '
          f'(🟢гейт: додає ⊥q понад 0.05)')
    print('📊 усе тут — СТЕЛЯ з ІДЕАЛЬНОЮ опорою; реальна опора буде гіршою')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
