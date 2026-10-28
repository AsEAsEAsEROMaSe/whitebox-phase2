"""РОЗТИН ЗМІЩЕННЯ: що саме створює ПЛОСКУ підлогу MSE — і чи вона наша ціна.

════════ ЗВІДКИ ПИТАННЯ (форензика лідера, С75) ════════
Бал = `MSE × max(0.1, C/B)`. Лідер **1.25e-08**. Хоч би який у нього множник,
множник ≥ 0.1 ⟹ його ПОВНА MSE ≤ **1.25e-07**.
Наша MSE 1.95e-06 ділиться (грейдерний диференціал sub87: вибірка ↓3.75% ⟹
MSE +2.09% замість +3.90%) на шум **54%** і ПЛОСКЕ зміщення **46%**:

    V = 1.05e-06   ·   B = 0.90e-06

🔑🔑 **САМЕ ЛИШЕ НАШЕ ЗМІЩЕННЯ У 7.2× БІЛЬШЕ ЗА ВСЮ MSE ЛІДЕРА.**
⟹ напрямок «ще трохи варіанс-редукції» не веде до нього НАВІТЬ у межі
нескінченної вибірки. Спершу мусить зникнути плоска підлога.

Із цих 46% зміряно прямо лише обрізання (16%). Решта **84% приписана
«замиканню» ВІДНІМАННЯМ, а не абляцією** — тобто джерело не встановлене.
Цей стенд встановлює його поіменно.

════════ ЧОМУ ЗМІЩЕННЯ ВЗАГАЛІ Є (читано в коді сабміту) ════════
`v` = оцінка похибки вибірки, яку ми ВІДНІМАЄМО. Її насіння
`x̄₀ − ‖w₀‖/√(2π)` — РІВНО похибка, середнє нуль: `‖w₀‖/√(2π)` точна.
Але чотири родини поправок мають вигляд `λ·(вибіркове − ЗАМИКАННЯ)`, і

    E[вибіркове] = істина ≠ замикання   ⟹   E[поправки] = λ·(похибка замикання)

тобто кожна родина ВЛИВАЄ детермінований зсув, який НЕ спадає з `n`.
🚨 І це стосується ВСІХ восьми шарів: навіть на шарі 1 опорні `s_замик`,
`q_замик`, `Φ_замик` рахуються через НЕДІАГОНАЛЬ `C₁` з ряду Мелера K=2,
тобто вже наближені. Точне в усій схемі рівно одне — `‖w₀‖/√(2π)`.

════════ ПРИЛАД ════════
Перенос ЛІНІЙНИЙ за `v` ⟹ внесок кожної родини на кожному шарі несеться
власним рядком поруч із `v`, і БУДЬ-ЯКА абляція `λ` — це лінійна комбінація
з ОДНОГО прогону:  `pred(λ) = pred₀ − λᵀF`.

Розділення зміщення й шуму — БЕЗ свіпу по `n`, через `D` жеребів:
    E[Σ(p_d−T₁)(p_d−T₂)] = ‖b‖² + V          (крос-ціль знімає шум істини)
    E[Σ(p̄ −T₁)(p̄ −T₂)] = ‖b‖² + V/D
  ⟹ ‖b‖² = (D·b2 − mse)/(D−1)   — незміщено, точно.

🚨 ІСТИНА БЕРЕТЬСЯ З **ПОВНОЇ** МЕРЕЖІ, не з обрізаної. Попередні стенди
брали `truth_last(wc)` — від обрізаної, і зміщення обрізання було НЕВИДИМЕ
за побудовою.

🚨🔑 І ЦЕ НЕ ДРІБНИЦЯ: `arena62/shrink_both.py`, яким фітились УСІ 32 `λ`
сабміту, бере `truth_moments(wc, …)` — істину **ОБРІЗАНОЇ** мережі. Отже
регресія мінімізувала `‖e_вибірки − λᵀF‖²`, тоді як грейдер міряє
`‖e_вибірки + b_обрізання − λᵀF‖²`. Якщо `F` корелює з `b_обрізання`
(а обидва живуть уздовж `m`), `λ` стоять не там. Тому тут міряємо ОБИДВІ
істини: розбіжність між ними і є доказ механізму.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 повний сабміт: `‖b‖²/MSE` ≈ **0.40–0.55** (грейдер каже 0.46);
📣 обрізання дає ≈ **16%** зміщення (незалежна звірка з sub88);
📣 моя гіпотеза: решту несуть родини `λ` — найбільше `Q`/`G` на шарах 5–8,
   де і `λ` найбільші (0.019→0.042), і похибка замикання найбільша.
🟢 ЖИВЕ, якщо якась абляція ЗМЕНШУЄ ПОВНУ MSE (зміщення+шум) на >5%.

🚨 ПЛОМБИ
  ✅ ТОТОЖНІСТЬ: мій прогін із рідними λ = `run72` із тими самими λ (≤1e-9);
  ✅ ЧИСТИЙ МК (без `v`, без обрізання) НЕЗМІЩЕНИЙ ⟹ `‖b‖²/MSE` < 3%;
     якщо ні — прилад має власну підлогу і решта чисел забруднена;
  ✅ ‖b‖² ≥ 0 у кожному варіанті.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, closure_parts, run_side1 as run72,  # noqa: E402
                    W_, GATE_N, INV_SQRT_2PI, _phi)
from varsub import S72, A72, Q72, G72                                   # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8
TRUTH_PAIRS = 262_144
CHUNK = 32_768
LAM = (S72, A72, Q72, G72)
FAM = ('S: δs', 'A: δE[x]', 'Q: δE[x²]', 'G: δмаска')


def truth_full(W, dev, seed):
    """Істина ПОВНОЇ мережі (не обрізаної!) — інакше зміщення обрізання невидиме."""
    acc = torch.zeros(W[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TRUTH_PAIRS:
        k = min(CHUNK, TRUTH_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for w in W:
            x = torch.relu(x @ w)
        acc += x.sum(0).double()
        done += k
    return acc / (2.0 * done)


def run_collect(wc, npair, cl, dev, seed, use_v=True, white=True):
    """Порт сабміту з ЗБОРОМ перенесених внесків усіх 4 родин × ksig шарів.

    Повертає (pred₀, F), де `pred₀` — прогноз БЕЗ жодної λ-поправки, а
    `F[q*ksig + i-1]` — внесок родини `q` шару `i`, уже перенесений до виходу.
    Тоді `pred(λ) = pred₀ − Σ λ·F` для будь-якого набору λ (перенос лінійний).
    """
    s_cl, m_cl, f_cl, q_cl = cl
    ksig = len(s_cl)
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    if white:
        cov = (x.T @ x) / denom
        ev, V = torch.linalg.eigh(cov.double())
        wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
        w0 = wh @ wc[0]
    else:
        w0 = wc[0]
    xa = torch.relu(x @ w0)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = xa.double().mean(0) - sg0 * INV_SQRT_2PI
    if not use_v:
        v = torch.zeros_like(v)
    F = torch.zeros(4 * ksig, wc[0].shape[1], dtype=torch.float64, device=dev)
    for i in range(1, depth):
        p = xa @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        F = (F @ wd) * gate.unsqueeze(0)
        if i <= ksig:
            pd = p.double()
            mh = pd.mean(0)
            sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
            F[0 * ksig + i - 1] += _phi(mh / sh) * (sh - s_cl[i - 1])
            xa = torch.relu(p)
            xd = xa.double()
            F[1 * ksig + i - 1] += xd.mean(0) - m_cl[i - 1]
            F[2 * ksig + i - 1] += xd.pow(2).mean(0) - q_cl[i - 1]
            F[3 * ksig + i - 1] += gate - f_cl[i - 1]
        else:
            xa = torch.relu(p)
    return xa.double().mean(0) - v, F


def split(mse, b2, D):
    """(‖b‖², V) із крос-цільових сум. Незміщено."""
    bb = (D * b2 - mse) / (D - 1.0)
    return bb, mse - bb


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print('📣 ПЕРЕДБАЧЕНО: ‖b‖²/MSE повного сабміту 0.40–0.55 (грейдер 0.46); '
          'обрізання ≈16% зміщення')
    print('🟢 ЖИВЕ, якщо абляція зменшує ПОВНУ MSE на >5%')
    print(f'   мереж {nets} × {ndraw} жеребів, істина {TRUTH_PAIRS} пар × 2 '
          f'(КРОС-ЦІЛЬ, з ПОВНОЇ мережі)\n')

    # варіанти λ: маска (4, ksig) множників на рідні коефіцієнти
    VAR = [('ПОВНИЙ сабміт', (1, 1, 1, 1)),
           ('− родина S (δs)', (0, 1, 1, 1)),
           ('− родина A (δE[x])', (1, 0, 1, 1)),
           ('− родина Q (δE[x²])', (1, 1, 0, 1)),
           ('− родина G (δмаска)', (1, 1, 1, 0)),
           ('лише S+A (Q,G геть)', (1, 1, 0, 0)),
           ('ВСІ λ = 0 (тільки перенос)', (0, 0, 0, 0))]

    acc = {}   # ключ -> [mse, b2]   (проти ПОВНОЇ істини)
    acc_c = {}  # те саме проти істини ОБРІЗАНОЇ мережі — ціль, у яку фітили λ

    def add(d, k, m, b):
        a = d.setdefault(k, [0.0, 0.0])
        a[0] += m; a[1] += b

    nout = 0
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            cl_t = closure_parts(wc, KSIG, dev, extras=True)
            cl_f = closure_parts(W, KSIG, dev, extras=True)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            # істина ОБРІЗАНОЇ мережі — та сама ціль, у яку фітили λ
            C1 = truth_full(wc, dev, 310_000 + 7919 * n)
            C2 = truth_full(wc, dev, 410_000 + 7919 * n)
            nout += T1.shape[0]

            lam = torch.tensor([LAM[q][i] for q in range(4)
                                for i in range(KSIG)],
                               dtype=torch.float64, device=dev)
            # ── ПЛОМБА ТОТОЖНОСТІ (лише на першій мережі, перший жереб)
            if n == 0:
                p0, F0 = run_collect(wc, npair, cl_t, dev, 5_000_000)
                mine = p0 - lam @ F0
                ref, _ = run72(wc, npair, cl_t[0], cl_t[1], dev, 5_000_000,
                               lam_s=S72, lam_a=A72, lam_q=Q72, lam_g=G72,
                               q_ext=cl_t[3], p_ext=cl_t[2])
                rel = float((mine - ref).abs().max() / ref.abs().max())
                print(f'  ✅ПЛОМБА ТОТОЖНОСТІ (мій порт vs run72): {rel:.2e} '
                      f'{"OK" if rel < 1e-9 else "🚨 РОЗБІЖНІСТЬ"}\n')

            sums = {}
            for dr in range(ndraw):
                sd = 88_000_000 + 733 * n + dr
                p0, F = run_collect(wc, npair, cl_t, dev, sd)
                for nm, msk in VAR:
                    mm = torch.tensor([msk[q] * LAM[q][i] for q in range(4)
                                       for i in range(KSIG)],
                                      dtype=torch.float64, device=dev)
                    sums.setdefault(nm, []).append(p0 - mm @ F)
                # 🔑 ЧЕСНА БАЗА «що взагалі купує вся КЗ-машинерія»: той самий
                #    бюджет, те саме обрізання, але БЕЗ `v` і БЕЗ λ.
                pnv, _ = run_collect(wc, npair, cl_t, dev, sd, use_v=False)
                sums.setdefault('обрізаний МК (без v, без λ)', []).append(pnv)
                # 🔑🔑 ВИРІШАЛЬНА ПЕРЕХРЕСНА ПЕРЕВІРКА ПЛОСКОЇ ПІДЛОГИ.
                #   Грейдерний диференціал sub87 стверджує зміщення 46%.
                #   Цей стенд міряє 7.6%. Розсудити можна ПРЯМО: подвоїти `n`.
                #     зміщення 7.6% ⟹ MSE(2n)/MSE(n) виграш 1.86×
                #     зміщення 46%  ⟹ виграш лише 1.37×
                #   Різниця величезна ⟹ одне число розсудить два прилади.
                p2, F2 = run_collect(wc, 2 * npair, cl_t, dev, sd + 500_000)
                sums.setdefault('ПОВНИЙ сабміт @ 2n', []).append(p2 - lam @ F2)
                # без обрізання (та сама кількість пар ⟹ шум співмірний).
                # 🚨 УВАГА: ці рядки витрачають ~1.35× бюджету (пари взято з
                #    обрізаної конфігурації) ⟹ їхнє «vs повн» НЕ є балом.
                pf0, Ff = run_collect(W, npair, cl_f, dev, sd)
                sums.setdefault('⛔без обрізання, всі λ=0', []).append(pf0)
                sums.setdefault('⛔без обрізання, повний λ', []).append(
                    pf0 - lam @ Ff)
                # ✅ЧИСТИЙ МК: без `v`, без λ, без обрізання
                pm, _ = run_collect(W, npair, cl_f, dev, sd, use_v=False)
                sums.setdefault('✅МК+вибілювання', []).append(pm)
                # ✅✅ЄДИНА ТОЧКА, ДЕ ЗМІЩЕННЯ ТОЧНО НУЛЬ: антитетичний МК БЕЗ
                #   вибілювання. Вибілювання робить вибіркову коваріацію
                #   ТОТОЖНО `I` ⟹ обумовлює вибірку ⟹ САМЕ ПО СОБІ зміщує.
                pn, _ = run_collect(W, npair, cl_f, dev, sd, use_v=False,
                                    white=False)
                sums.setdefault('✅ЧИСТИЙ МК без вибіл. (ПЛОМБА)', []).append(pn)

            for nm, lst in sums.items():
                P = torch.stack(lst, 0)
                pb = P.mean(0)
                add(acc, nm,
                    float((((P - T1) * (P - T2)).sum(1)).mean()),
                    float(((pb - T1) * (pb - T2)).sum()))
                add(acc_c, nm,
                    float((((P - C1) * (P - C2)).sum(1)).mean()),
                    float(((pb - C1) * (pb - C2)).sum()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    print(f'\n{nets} мереж × {ndraw} жеребів. MSE на НЕЙРОН, крос-ціль, '
          f'істина з ПОВНОЇ мережі.\n')
    print(f'{"варіант":>30s} {"MSE":>10s} {"‖b‖²":>10s} {"шум":>10s} '
          f'{"b/MSE":>7s} {"vs повн":>8s} | {"ціль ФІТУ":>9s} {"vs повн":>8s}')
    print('-' * 100)
    order = ([v[0] for v in VAR] + ['ПОВНИЙ сабміт @ 2n',
             'обрізаний МК (без v, без λ)',
             '⛔без обрізання, всі λ=0',
             '⛔без обрізання, повний λ', '✅МК+вибілювання',
             '✅ЧИСТИЙ МК без вибіл. (ПЛОМБА)'])
    ref_mse = acc['ПОВНИЙ сабміт'][0] / nout
    ref_cut = acc_c['ПОВНИЙ сабміт'][0] / nout
    best = (1.0, None)
    for nm in order:
        m, b2 = acc[nm]
        mse = m / nout
        bb, vv = split(m, b2, ndraw)
        bb, vv = bb / nout, vv / nout
        g = ref_mse / mse
        mc_ = acc_c[nm][0] / nout
        if nm != 'ПОВНИЙ сабміт' and not nm.startswith('✅') and g > best[0]:
            best = (g, nm)
        print(f'{nm:>30s} {mse:10.3e} {bb:10.3e} {vv:10.3e} '
              f'{bb / max(mse, 1e-30):7.3f} {g:7.3f}× | '
              f'{mc_:9.3e} {ref_cut / mc_:7.3f}×')

    mfull, bfull = acc['ПОВНИЙ сабміт']
    bb, _ = split(mfull, bfull, ndraw)
    frac = bb / max(mfull, 1e-30)
    mmc, bmc = acc['✅ЧИСТИЙ МК без вибіл. (ПЛОМБА)']
    bmc2, _ = split(mmc, bmc, ndraw)
    fmc = bmc2 / max(mmc, 1e-30)
    mw, bw = acc['✅МК+вибілювання']
    bw2, _ = split(mw, bw, ndraw)
    print(f'\n🔑 ЦІНА ВИБІЛЮВАННЯ У ЗМІЩЕННІ: ‖b‖² {bmc2 / nout:.3e} (без) → '
          f'{bw2 / nout:.3e} (з), MSE {mmc / nout:.3e} → {mw / nout:.3e}')
    print(f'\n✅ПЛОМБА ЧИСТОГО МК: ‖b‖²/MSE = {fmc:+.4f} '
          f'{"OK (|·|<0.03)" if abs(fmc) < 0.03 else "🚨 ПРИЛАД МАЄ ВЛАСНУ ПІДЛОГУ"}')
    # 🔑 ДОКАЗ МЕХАНІЗМУ: якщо проти ЦІЛІ ФІТУ (обрізана істина) λ виграють,
    #    а проти СПРАВЖНЬОЇ програють — значить фіт мінімізував не ту величину.
    gf = acc_c['ВСІ λ = 0 (тільки перенос)'][0] / max(acc_c['ПОВНИЙ сабміт'][0], 1e-30)
    gt = acc['ВСІ λ = 0 (тільки перенос)'][0] / max(acc['ПОВНИЙ сабміт'][0], 1e-30)
    print(f'🔑 λ=0 проти ЦІЛІ ФІТУ (обрізана істина): {gf:.3f}×   '
          f'проти СПРАВЖНЬОЇ: {gt:.3f}×')
    print(f'   ⟹ {"🚨 λ ФІТИЛИСЬ НЕ В ТУ ЦІЛЬ" if gf < 1.0 < gt else "механізм НЕ підтверджено"}')
    print(f'🔑 ЗМІЩЕННЯ ПОВНОГО САБМІТУ: {frac:.3f} від MSE '
          f'(📣передбачено 0.40–0.55, грейдер 0.46)')
    mtrim = acc['⛔без обрізання, повний λ'][0] / nout
    btrim, _ = split(*acc['⛔без обрізання, повний λ'], ndraw)
    print(f'🔑 ОБРІЗАННЯ несе {(bb - btrim) / max(bb, 1e-30):.3f} зміщення '
          f'(📣незалежна звірка з sub88: 0.16)')
    m2 = acc['ПОВНИЙ сабміт @ 2n'][0] / nout
    g2 = ref_mse / m2
    bbn = bb / nout                      # ‖b‖² на нейрон (bb — сума!)
    pred_mine = ref_mse / (bbn + (ref_mse - bbn) / 2.0)
    pred_grad = ref_mse / (0.46 * ref_mse + 0.54 * ref_mse / 2.0)
    print(f'\n🔑🔑 РОЗСУД ДВОХ ПРИЛАДІВ (подвоєння `n`): виміряно {g2:.3f}×')
    print(f'     📣 цей стенд (зміщення {frac:.3f}) передбачав {pred_mine:.3f}×')
    print(f'     📣 грейдерний диференціал (зміщення 0.46) — {pred_grad:.3f}×')
    print(f'     ⟹ {"ЦЕЙ СТЕНД" if abs(g2 - pred_mine) < abs(g2 - pred_grad) else "🚨 ГРЕЙДЕР"} ближче')
    mnv = acc['обрізаний МК (без v, без λ)'][0] / nout
    print(f'🔑 ЩО КУПУЄ ВСЯ КЗ-МАШИНЕРІЯ (той самий бюджет): '
          f'{mnv / ref_mse:.3f}× (МК без v проти повного сабміту)')
    print(f'\n🔑 НАЙКРАЩА АБЛЯЦІЯ: {best[1]} ⟹ {best[0]:.3f}×')
    print(f'🔑 ВИРОК: {"ЖИВЕ" if best[0] > 1.05 else "МЕРТВО"} '
          f'(🟢гейт >1.05×)')
    print(f'📊 БАЛ: 1.954e-07 → {1.954e-07 / max(best[0], 1e-9):.3e} '
          f'(якщо перенесеться на грейдер)')
    print(f'📊 СТЕЛЯ «зміщення = 0»: {1.954e-07 * (1 - frac):.3e}  '
          f'(лідер 1.25e-08)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
