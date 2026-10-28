"""ПРОФІЛЬ ДРИЖАННЯ ПО ГЛИБИНІ + ЧИ ЕКСТРАПОЛЮЄТЬСЯ ВІН ІЗ ПЕРШИХ ШАРІВ.

════════ ЗВІДКИ (ідея юзера) ════════
«середнє виводиться із сум, нейрони дрижать; на кінцях і початку мають дрижати
менше, ніж усередині» + «дрижання конусне, як нитка на вітру ⟹ в теорії можна
екстраполювати з перших шарів на останні».

Непрямі дані є, ЧИСТОЇ КРИВОЇ НЕМА. `varmap.py` міряв ПРИЗ (точна опора до
шару K: 1.72× на K=8 → 44.8× на K=30), `lyapunov_dirs.py` міряв НАПРЯМКИ
(перекриття топ-6 із M₁: 0.914 на 5 шарі → 0.161 на 30). Але сам профіль
`J(ℓ) = ‖x̄_ℓ − m_ℓ‖ / ‖m_ℓ‖` не міряний жодного разу.

════════ ДВА РІЗНІ ПИТАННЯ, ЯКІ ЛЕГКО ЗЛИТИ ════════
🚨 «Екстраполювати дрижання» означає ДВІ несумісні речі:
  (A) МАГНІТУДА `E‖e_ℓ‖²` — передбачувана з ляпуновського росту, АЛЕ
      НЕ ВІДНІМАЄТЬСЯ: `E[e]=0`, знання розміру не дає знаку. Максимум,
      що з неї — оптимальний `λ`, а він уже в оптимумі (0.3%).
  (B) ЗНАЧЕННЯ ЖЕРЕБУ `⟨e_31, u₁⟩` — ось це віднімається і йде в бал.
Міряємо ОБИДВІ і не плутаємо.

════════ ЩО САМЕ МІРЯЄМО ════════
A. `J(ℓ)` — відносне дрижання середнього по шарах (форма кривої: горб чи ні).
B. `COH(ℓ)` — частка шуму в головному напрямку + PR-ранг шуму
   (пояснює, ЧОМУ на кінці не гаситься усередненням по 256 нейронах).
C. ЕКСТРАПОЛЯЦІЯ МАГНІТУДИ: пряма `log J(ℓ) = a + b·ℓ` по ℓ=1..8 →
   передбачення `J(31)`. Відношення передбачене/факт.
D. ЕКСТРАПОЛЯЦІЯ ЗНАЧЕННЯ (те, що може дати бал). OOS-`R²` для
   `y = ⟨e_31, u₁⟩` за ДОСТУПНИМИ ознаками (вибірка мінус замикання):
     M1 — лише шар 8            (чи досить однієї глибокої точки)
     M2 — шари 1..8             (≈ те, що робить сабміт)
     M3 — M2 + ознаки НАХИЛУ    (власне гіпотеза «нитки на вітру»)
     ORC — ⟨e_8, u_8⟩ ІСТИННЕ   (стеля: чи впирається в замикання, чи в перенос)

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 A: `J(ℓ)` МОНОТОННО РОСТЕ, без горба посередині, з прискоренням після ℓ≈20.
     J(0) ≲ 0.2·J(1) — вибілювання прибиває вхід.
📣 B: `COH` росте з глибиною, PR-ранг шуму падає до ~2-3 (колапс конуса).
📣 C: екстраполяція ПРОМАХНЕТЬСЯ в 2-10× (нахил на 1..8 не той, що на 20..31,
     бо ляпуновський режим встановлюється пізно).
📣 D: M2 ≈ 0.10-0.15 (це і є наші 11%), M3 − M2 ≤ 0.02, ORC ≈ 0.40-0.45
     (бо `varmap` дає 1.72× на K=8 ⟹ ρ²≈0.42).
🟢 ЖИВЕ, якщо M3 − M2 ≥ 0.05 (нахил несе НОВЕ понад те, що вже беремо).
🟢 АБО якщо ORC − M2 ≥ 0.20 — тоді впирається в ТОЧНІСТЬ замикання, а не
   в перенос, і це інша (вже відома) стіна.

🚨 ПЛОМБИ
  ✅ ІСТИНА ЗБІЖНА: половинки істини розходяться ≪ самого дрижання;
  ✅ ШАР 0 ТОЧНИЙ: `‖m_0 − σ/√2π‖/‖m_0‖ ≈ 0` (перевіряє і істину, і код);
  ✅ ОДНОРІДНІСТЬ `f(2z)=2f(z)` — інакше весь конусний опис неправомірний;
  ⛔ ROLL: ознаки, зсунуті на сусідній жереб, мусять дати `R² ≈ 0`.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N)

L_ = 32
# 🚨 захищений розбір: цей модуль ІМПОРТУЄТЬСЯ з `jitview.py`, у якого інший
#    порядок аргументів — сирий `int(sys.argv[3])` там падав би на імпорті.
DEV = (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].startswith('cuda')
       else 'cuda:4')
NDRAW = (int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit()
         else 128)
TR_PAIRS, CHUNK = 524_288, 32768
KSIG = 8
FEAT_L = (1, 2, 3, 4, 6, 8)
PROF_L = (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 27, 29, 30, 31)
# 🚨 K=31 — ТАВТОЛОГІЧНА ПЛОМБА: перенос порожній ⟹ R² мусить бути рівно 1.
#    Без неї стеля вже одного разу вийшла заниженою вчетверо.
ORC_K = (8, 16, 24, 31)
# 🔑 ТА САМА МАШИНЕРІЯ, АЛЕ З ЗАМИКАННЯМ ЗАМІСТЬ ІСТИНИ. Різниця CLO↔ORC на
#    ОДНОМУ шарі — це РІВНО ціна неточності замикання, без домішки переносу.
#    `0` = наш реальний якір (шар 0, де істина ТОЧНА: σ/√2π).
CLO_K = (0, 1, 2, 4, 8)


def truth_means(wc, dev, seed, halves=False):
    """Істина на ВСІХ шарах. `halves=True` ⟹ ще й дві половини для пломби."""
    depth = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
           for w in wc]
    acc_h = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
             for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            a = torch.relu(a @ wc[i])
            s = a.double().sum(0)
            acc[i] += s
            if done < TR_PAIRS // 2:
                acc_h[i] += s
        done += k
    full = [t / float(2 * TR_PAIRS) for t in acc]
    half = [t / float(2 * (TR_PAIRS // 2)) for t in acc_h] if halves else None
    return full, half


def draw(wc, npair, dev, seed, m_cl, f_cl):
    """Один жереб рівно як у сабміті.

    Повертає (x̄ по ВСІХ шарах, ознаки, затвори по шарах). Затвори потрібні,
    щоб ПІСЛЯ прогону зробити чесний перенос оракула `v ← (v@W)·затвор`."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xw = x @ wh

    a = torch.relu(xw @ wc[0])
    out = [a.double().mean(0)]
    gates = [None]
    feats = []
    for i in range(1, len(wc)):
        p = a @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        a = torch.relu(p)
        xm = a.double().mean(0)
        if i in FEAT_L and i - 1 < len(m_cl):
            feats.append(float((xm - m_cl[i - 1]).sum()))
            feats.append(float((gate - f_cl[i - 1]).sum()))
        out.append(xm)
        gates.append(gate)
    return out, feats, gates


def transport(v, wc, gates, k_from, k_to):
    """Той самий перенос, що в сабміті: `v ← (v @ W_i) · затвор_i`."""
    for i in range(k_from + 1, k_to + 1):
        v = (v @ wc[i].double()) * gates[i]
    return v


def r2_oos(X, y, intercept=True):
    """Чесний OOS: фіт на першій половині жеребів, вимір на другій.

    🚨 `intercept=True` МОВЧКИ ПРИБИРАЄ ЗМІЩЕННЯ ЗАМИКАННЯ: `x̄_K − m_замик,K`
    = шум вибірки + СТАЛА похибка замикання, і вільний член з'їдає сталу
    задарма. У САБМІТІ вільного члена НЕМА — там зміщення переноситься і
    псує результат. Тому міряємо ОБИДВА: з ним (скільки СИГНАЛУ) і без нього
    (скільки з нього реально можна взяти)."""
    h = len(y) // 2
    A = np.hstack([X, np.ones((X.shape[0], 1))]) if intercept else X
    b, *_ = np.linalg.lstsq(A[:h], y[:h], rcond=None)
    r = y[h:] - A[h:] @ b
    d = (y[h:].var() if intercept else (y[h:] ** 2).mean())
    e = (r.var() if intercept else (r ** 2).mean())
    return float(1.0 - e / max(d, 1e-300))


def zsc(X):
    X = X - X.mean(0, keepdims=True)
    s = X.std(0)
    s[s == 0] = 1.0
    return X / s


def main(nets=16, dev=DEV, ndraw=NDRAW):
    torch.manual_seed(23)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print('📣 ПЕРЕДБАЧЕНО: A монотонний ріст без горба · B PR-ранг шуму →2-3 ·')
    print('📣 C екстраполяція промахне в 2-10× · D M2≈0.10-0.15, '
          'M3−M2≤0.02, ORC≈0.40-0.45')
    print('🟢 ЖИВЕ при M3−M2 ≥ 0.05\n', flush=True)

    PROF, COHV, PRV = [], [], []
    R2 = {'M1': [], 'M2': [], 'M3': [], 'ROLL': [], 'SUB': []}
    R2.update({f'ORC{k}': [] for k in ORC_K})
    R2.update({f'CLO{k}': [] for k in CLO_K})
    R2.update({f'NOI{k}': [] for k in CLO_K})
    BS = {k: [] for k in CLO_K if k >= 1}
    EXTR = []
    seal_tr, seal_l0, seal_hom = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            mt, mh = truth_means(wc, dev, 310_000 + 877 * n, halves=True)
            _, m_cl, f_cl, _ = closure_parts(wc, min(KSIG, d - 1), dev,
                                             extras=True)

            # ✅ПЛОМБА ОДНОРІДНОСТІ
            g = torch.Generator(device=dev).manual_seed(999)
            zz = torch.randn(64, W_, device=dev, generator=g)
            a1, a2 = zz, zz * 2.0
            for i in range(d):
                a1 = torch.relu(a1 @ wc[i])
                a2 = torch.relu(a2 @ wc[i])
            seal_hom.append(float((a2 - 2.0 * a1).abs().max()
                                  / a1.abs().max()))
            # ✅ПЛОМБА ШАРУ 0: m_0 мусить дорівнювати σ/√2π
            sg0 = wc[0].pow(2).sum(0).sqrt().double()
            seal_l0.append(float((mt[0] - sg0 * INV_SQRT_2PI).norm()
                                 / mt[0].norm()))

            D, FT = [], []
            ORCV = {k: [] for k in ORC_K}
            CLOV = {k: [] for k in CLO_K}
            for k in range(ndraw):
                xb, ft, gts = draw(wc, npair, dev, 51_000_000 + 977 * n + k,
                                   m_cl, f_cl)
                D.append([xb[l] - mt[l] for l in range(d)])
                FT.append(ft)
                # 🔑 ЧЕСНИЙ ОРАКУЛ: істинна похибка шару K, ПЕРЕНЕСЕНА
                #    рівно тим самим ланцюгом, що в сабміті.
                for kk in ORC_K:
                    if kk <= d - 1:
                        ORCV[kk].append(
                            transport(xb[kk] - mt[kk], wc, gts, kk, d - 1))
                # 🔑 ТЕ САМЕ, АЛЕ З ДОСТУПНОЮ ОПОРОЮ (повний ВЕКТОР, як у
                #    сабміті, а не стиснутий у суму)
                for kk in CLO_K:
                    ref = (sg0 * INV_SQRT_2PI if kk == 0
                           else m_cl[kk - 1].double())
                    CLOV[kk].append(transport(xb[kk] - ref, wc, gts,
                                              kk, d - 1))

            # ---------- A. профіль + B. когерентність ----------
            prof, coh, pr, mag = [], [], [], []
            for l in range(d):
                E = torch.stack([D[k][l] for k in range(ndraw)], 0)
                nrm2 = float(E.pow(2).sum(1).mean())
                prof.append(np.sqrt(nrm2) / float(mt[l].norm()))
                mag.append(np.sqrt(nrm2))
                Ec = E - E.mean(0, keepdim=True)
                C = (Ec.T @ Ec) / float(ndraw)
                ev = torch.linalg.eigvalsh(C.double()).clamp_min(0)
                pr.append(float(ev.sum() ** 2 / ev.pow(2).sum()))
                coh.append(float(ev[-1] / ev.sum()))
            PROF.append(prof)
            COHV.append(coh)
            PRV.append(pr)
            # ✅ПЛОМБА ІСТИНИ: розбіг половин проти самого дрижання
            seal_tr.append(float((mt[d - 1] - mh[d - 1]).norm()) / mag[d - 1])

            # ---------- C. екстраполяція МАГНІТУДИ ----------
            ll = np.arange(1, KSIG + 1, dtype=np.float64)
            lg = np.log(np.array([mag[int(l)] for l in ll]))
            b1, b0 = np.polyfit(ll, lg, 1)
            EXTR.append(float(np.exp(b0 + b1 * (d - 1)) / mag[d - 1]))

            # ---------- D. екстраполяція ЗНАЧЕННЯ ----------
            E31 = torch.stack([D[k][d - 1] for k in range(ndraw)], 0)
            Ec = E31 - E31.mean(0, keepdim=True)
            _, V1 = torch.linalg.eigh((Ec.T @ Ec).double())
            y = (Ec @ V1[:, -1]).cpu().numpy().astype(np.float64)

            X = zsc(np.array(FT, dtype=np.float64))
            n8 = 2 * (len(FEAT_L) - 1)                 # ознаки шару 8
            # нахил ‖x̄_ℓ − m_замик,ℓ‖ по ℓ=1..8, ПОЖЕРЕБНО (гіпотеза «нитки»)
            slope = []
            for k in range(ndraw):
                v = np.abs(np.array([FT[k][2 * j] for j in range(len(FEAT_L))]))
                v = np.log(np.maximum(v, 1e-30))
                s_, _ = np.polyfit(np.array(FEAT_L, dtype=np.float64), v, 1)
                slope.append(s_)
            sl = zsc(np.array(slope)[:, None])
            x8 = X[:, n8:n8 + 2]
            XM3 = np.hstack([X, sl, sl * x8[:, :1], sl * x8[:, 1:2]])
            R2['M1'].append(r2_oos(x8, y))
            R2['M2'].append(r2_oos(X, y))
            R2['M3'].append(r2_oos(XM3, y))
            R2['ROLL'].append(r2_oos(np.roll(X, 1, axis=0), y))
            u1 = V1[:, -1]
            for kk in ORC_K:
                if not ORCV[kk]:
                    continue
                o = (torch.stack(ORCV[kk], 0) @ u1).cpu().numpy()
                R2[f'ORC{kk}'].append(r2_oos(o[:, None].astype(np.float64), y))
            cols = []
            for kk in CLO_K:
                c = (torch.stack(CLOV[kk], 0) @ u1).cpu().numpy()
                c = c[:, None].astype(np.float64)
                R2[f'CLO{kk}'].append(r2_oos(c, y))
                R2[f'NOI{kk}'].append(r2_oos(c, y, intercept=False))
                cols.append(c)
                # 🔑 ЗМІЩЕННЯ ЗАМИКАННЯ проти ШУМУ ВИБІРКИ на тому самому шарі
                if kk >= 1:
                    bias = float((m_cl[kk - 1].double() - mt[kk]).norm())
                    sig = float(torch.stack([D[q][kk] for q in range(ndraw)],
                                            0).pow(2).sum(1).mean() ** 0.5)
                    BS[kk].append(bias / sig)
            # SUB — усі доступні канали разом: рівно те, що вміє наша схема
            R2['SUB'].append(r2_oos(np.hstack(cols), y))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    P = np.array(PROF)
    CH = np.array(COHV)
    PR = np.array(PRV)
    print(f'\n{nets} мереж × {ndraw} жеребів, npair={npair}\n')
    print(f'✅ПЛОМБА ОДНОРІДНОСТІ: {np.max(seal_hom):.2e} '
          f'{"✅" if np.max(seal_hom) < 1e-5 else "🛑"}')
    print(f'✅ПЛОМБА ШАР 0 = σ/√2π: {np.mean(seal_l0)*100:.4f}% '
          f'{"✅" if np.mean(seal_l0) < 0.02 else "🛑"}')
    print(f'✅ПЛОМБА ІСТИНИ (розбіг половин / дрижання): '
          f'{np.mean(seal_tr):.4f} {"✅" if np.mean(seal_tr) < 0.35 else "🛑"}')

    print('\nA+B. ПРОФІЛЬ ДРИЖАННЯ ПО ГЛИБИНІ')
    print(f'{"шар":>4} {"J(ℓ)=‖e‖/‖m‖":>14} {"vs J(1)":>9} '
          f'{"когерент.":>10} {"PR-ранг шуму":>13}')
    for l in PROF_L:
        if l >= P.shape[1]:
            continue
        print(f'{l:>4} {np.mean(P[:, l])*100:>13.4f}% '
              f'{np.mean(P[:, l])/np.mean(P[:, 1]):>9.3f} '
              f'{np.mean(CH[:, l])*100:>9.1f}% {np.mean(PR[:, l]):>13.2f}')

    print(f'\nC. ЕКСТРАПОЛЯЦІЯ МАГНІТУДИ (пряма по ℓ=1..8 → ℓ=31)')
    e = np.array(EXTR)
    print(f'   передбачене/факт: медіана {np.median(e):.3f}× '
          f'(розкид {np.percentile(e,10):.3f}…{np.percentile(e,90):.3f})')
    print(f'   {"✅ЕКСТРАПОЛЮЄТЬСЯ" if 0.5 < np.median(e) < 2 else "🛑ПРОМАХ"}'
          f' — але магнітуда НЕ віднімається (E[e]=0)')

    print(f'\nD. ЕКСТРАПОЛЯЦІЯ ЗНАЧЕННЯ (OOS R² для ⟨e_31, u₁⟩)')
    print('   ── повний ВЕКТОР, перенесений тим самим ланцюгом ──')
    print(f'   {"шар K":>6} {"ЗАМИК.":>8} {"ІСТИНА":>8} {"ЦІНА":>7} '
          f'{"БЕЗ вільн.чл.":>14} {"зміщ/шум":>9}')
    for kk in CLO_K:
        c = np.mean(R2[f'CLO{kk}'])
        ni = np.mean(R2[f'NOI{kk}'])
        o = np.mean(R2[f'ORC{kk}']) if f'ORC{kk}' in R2 else float('nan')
        gap = (o - c) if o == o else float('nan')
        bs = np.mean(BS[kk]) if kk in BS else float('nan')
        tag = '  ← наш якір' if kk == 0 else ''
        print(f'   {kk:>6} {c:>8.4f} {o:>8.4f} {gap:>7.4f} {ni:>14.4f} '
              f'{bs:>9.2f}{tag}')
    for kk in ORC_K:
        if kk in CLO_K:
            continue
        v = np.mean(R2[f'ORC{kk}'])
        tag = ' ← ✅ТАВТОЛОГІЯ, мусить бути 1.0' if kk == 31 else ''
        print(f'   {kk:>6} {"—":>11} {v:>9.4f}{tag}')
    print('\n   РОЗКИД зміщ/шум ПО МЕРЕЖАХ (чи варта адаптивна λ):')
    for kk in sorted(BS):
        a = np.array(BS[kk])
        print(f'      шар {kk:>2}: сер {a.mean():.2f} · '
              f'мін {a.min():.2f} · макс {a.max():.2f} · '
              f'ст.в/сер {a.std()/a.mean():.3f}')
    print(f'\n   SUB (усі доступні канали разом): '
          f'{np.mean(R2["SUB"]):.4f}')
    print('   ── стиснуті в СУМУ ознаки (слабша версія, для контролю) ──')
    for k in ('M1', 'M2', 'M3', 'ROLL'):
        v = np.array(R2[k])
        print(f'   {k:>6}: {np.mean(v):>7.4f}  (медіана {np.median(v):>7.4f})')
    dm = np.mean(R2['M3']) - np.mean(R2['M2'])
    do = np.mean(R2['ORC8']) - np.mean(R2['CLO8'])
    ds = np.mean(R2['ORC8']) - np.mean(R2['SUB'])
    rl = np.mean(R2['ROLL'])
    taut = np.mean(R2['ORC31'])
    print(f'\n   ✅ПЛОМБА ТАВТОЛОГІЇ ORC31 = {taut:.4f} '
          f'{"✅" if taut > 0.99 else "🛑 СТЕНД ЗЛАМАНИЙ"}')
    print(f'   ⛔ROLL {rl:.4f} {"✅" if abs(rl) < 0.15 else "🛑 тавтологія"}')
    print(f'   M3 − M2 = {dm:+.4f}  (🟢гейт ≥ 0.05 — гіпотеза «нитки»)')
    print(f'   ORC8 − CLO8 = {do:+.4f}  ← ЧИСТА ціна неточності замикання')
    print(f'   ORC8 − SUB  = {ds:+.4f}  ← що лишається понад УСІ наші канали')
    live = dm >= 0.05
    print(f'\n🔑 ВИРОК НАХИЛУ: {"ЖИВЕ" if live else "МЕРТВО"}')
    coh31 = np.mean(CH[:, -1])
    print(f'   якби взяли все з M3: {1/(1-min(np.mean(R2["M3"]),0.99)):.2f}× '
          f'по когерентній компоненті ({coh31*100:.0f}% шуму) ⟹ '
          f'{1/(1-coh31*max(np.mean(R2["M3"]),0)):.3f}× у MSE')
    print('📊 БАЛ: 1.954e-07 (діагностика, стендова — бал не чіпає)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
