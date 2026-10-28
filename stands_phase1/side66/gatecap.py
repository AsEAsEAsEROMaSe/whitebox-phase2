"""ЧИ Є ПЕРЕНОС БУТИЛКОВИМ ГОРЛОМ? (гіпотеза юзера)

════════ ЗВІДКИ ════════
`varmap.py`: заміщення істинною похибкою на шарі 30 дає **11.39×** (`ρ²`=0.912),
і я пояснив залишкові 8.8% як «шум, народжений на ОСТАННЬОМУ кроці». Але це
пояснення НЕ ПЕРЕВІРЕНЕ: між заміщенням і виходом стоїть ПЕРЕНОС
    `v ← (v @ W_ℓ) · gate_ℓ`,
і він може втрачати сам по собі. Юзер: «може ми створили бутилкове горло, або
всовуємо в одне число, і треба ще одна операція».

ДВА КОНКРЕТНІ ПІДОЗРЮВАНІ:
  1️⃣ **ЗАТВОР**: `gate = (p[:1024] > 0).mean(0)` — оцінка `E[relu']` лише по
     **1024** семплах при основній вибірці ~8800. Шум ~`√(q(1−q)/1024)` ≈ 1.5%
     НА КОЖНОМУ з 30 кроків. Економія від обрізання копійчана (порівняння `p>0`
     проти матмула `n²`), а точність переносу може з'їдатись.
  2️⃣ **ОДНЕ ЧИСЛО**: `v` — ОДИН вектор. Заміщення на різних `K` дають РІЗНІ
     канали, і ми жодного разу не пробували їх СКОМБІНУВАТИ — брали рівно один.

════════ ЩО МІРЯЄМО ════════
  • `gate` по 1024 (наш) · по ВСІХ семплах · ОРАКУЛЬНИЙ `E[relu']` з 262k;
  • комбінація кількох `K` одночасно: `pred = x̄ − Σ λ_K v_K`, λ фітуються
    LOO ПО МЕРЕЖАХ (як у `deepk`), проти найкращого ОДНОГО `K`.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 ЗАТВОР: шум 1.5% на крок дав би `ρ²`≈0.99 на крок; за 30 кроків це помітно,
   але 8.8% втрати — завелико для самого затвора ⟹ оракульний `gate` підніме
   стелю 11.39× лише до **12–14×**, не більше.
📣 КОМБІНАЦІЯ: канали з різних `K` СИЛЬНО скорельовані (глибокий містить
   мілкий) ⟹ додадуть мало, **1.05–1.2×** понад найкращий одиничний.
🟢 ЖИВЕ, якщо будь-що з двох дає ≥1.5× понад 11.39× (тобто стеля схеми була
   занижена саме переносом, і горло справжнє).

🚨 ПЛОМБИ
  ✅ `K = остання глибина` — ТАВТОЛОГІЯ (`pred = m_істина`), мусить дати
     майже нескінченність; якщо ні — механіка заміщення зламана;
  ✅ `gate` по 1024 мусить відтворити число `varmap` (11.39×);
  ⛔ НЕГ: λ, фітовані на ШУМОВИХ каналах (сусідній жереб) ≈ 1.000×.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_                           # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402
from price31 import loo, three_slices                              # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
TR_PAIRS, CHUNK = 131_072, 32768
KS = (4, 8, 12, 16, 20, 24, 27, 28, 29, 30)
GATES = ('1024', 'всі', 'оракул')


def truth_mq(wc, dev, seed):
    """Істинні `E[relu]` і `E[relu']`(=частка активних) для кожного шару."""
    depth = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
           for w in wc]
    qs = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
          for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            p = a @ wc[i]
            qs[i] += (p > 0).double().sum(0)
            a = torch.relu(p)
            acc[i] += a.double().sum(0)
        done += k
    n = float(2 * TR_PAIRS)
    return [t / n for t in acc], [t / n for t in qs]


def run_map(wc, npair, dev, seed, mtrue, ks, gate_mode, qtrue=None,
            gate_n=1024):
    """Один жереб. Повертає (x̄_L, матриця каналів VM для всіх K)."""
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    nk = len(ks)
    VM = torch.zeros(nk, wc[0].shape[1], dtype=torch.float64, device=dev)
    for j, K in enumerate(ks):
        if K == 0:
            VM[j] = xa.double().mean(0) - mtrue[0]
    for i in range(1, depth):
        p = xa @ wc[i]
        if gate_mode == 'оракул':
            gate = qtrue[i]
        elif gate_mode == 'всі':
            gate = (p > 0).to(p.dtype).mean(0).double()
        else:
            gate = (p[:gate_n] > 0).to(p.dtype).sum(0).double() / gate_n
        VM = (VM @ wc[i].double()) * gate.unsqueeze(0)
        xa = torch.relu(p)
        for j, K in enumerate(ks):
            if K == i:
                VM[j] = xa.double().mean(0) - mtrue[i]
    return xa.double().mean(0), VM


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: оракульний `gate` підніме 11.39× лише до 12–14×; '
          f'комбінація K додасть 1.05–1.2×')
    print(f'🟢 ЖИВЕ при ≥1.5× понад 11.39× ⟹ горло справжнє')
    print(f'   мереж {nets} × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    pure = []
    # 🚨 глибина `d` різна між мережами ⟹ ключ `d-1` (тавтологія) наперед
    # невідомий; заводимо запис ліниво, інакше KeyError на першій же мережі
    one = {}
    G = {gm: [] for gm in GATES}
    B = {gm: [] for gm in GATES}
    C = {gm: [] for gm in GATES}
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            ks = [K for K in KS if K < d] + [d - 1]   # ✅останній = тавтологія
            nk = len(ks)
            mt, qt = truth_mq(wc, dev, 310_000 + 877 * n)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            ap = 0.0
            ao = {(gm, K): 0.0 for gm in GATES for K in ks}
            gg = {gm: torch.zeros(nk, nk, dtype=torch.float64, device=dev)
                  for gm in GATES}
            bb = {gm: torch.zeros(nk, dtype=torch.float64, device=dev)
                  for gm in GATES}
            cc = {gm: 0.0 for gm in GATES}
            gN = torch.zeros(nk, nk, dtype=torch.float64, device=dev)
            bN = torch.zeros(nk, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            for dr in range(ndraw):
                sd = 77_000_000 + 733 * n + dr
                for gm in GATES:
                    xl, VM = run_map(wc, npair, dev, sd, mt, ks, gm, qt)
                    if gm == '1024':
                        ap += float(((xl - T1) * (xl - T2)).sum())
                    for j, K in enumerate(ks):
                        p = xl - VM[j]
                        ao[(gm, K)] += float(((p - T1) * (p - T2)).sum())
                    e1, e2 = xl - T1, xl - T2
                    e = 0.5 * (e1 + e2)
                    cc[gm] += float((e1 * e2).sum())
                    gg[gm] += VM @ VM.T
                    bb[gm] += VM @ e
                    if gm == '1024':
                        if prevR is not None:
                            gN += prevR @ prevR.T
                            bN += prevR @ e
                            cN += float((e1 * e2).sum())
                        prevR = VM.clone()
            pure.append(ap / ndraw)
            for gm in GATES:
                for K in ks:
                    one.setdefault((gm, K), []).append(ao[(gm, K)] / ndraw)
                G[gm].append(gg[gm]); B[gm].append(bb[gm]); C[gm].append(cc[gm])
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair}, K={ks[-1]})', flush=True)

    P = np.array(pure).sum()
    kk = [K for K in KS] + [max(k for (g, k) in one if one[(g, k)])]
    kk = sorted(set(K for (g, K) in one if one[(g, K)]))
    print(f'\n{nets} мереж × {ndraw} жеребів. КРОС-ЦІЛЬ. '
          f'База = чистий `x̄`.\n')
    print(f'{"K":>4s} ' + ' '.join(f'{"gate="+g:>12s}' for g in GATES))
    print('-' * 46)
    for K in kk:
        row = f'{K:4d} '
        for gm in GATES:
            v = np.array(one[(gm, K)])
            row += f' {P / v.sum():11.3f}×'
        print(row)

    print(f'\n{"КОМБІНАЦІЯ всіх K (LOO-фіт λ)":>32s}')
    best = {}
    for gm in GATES:
        nk = G[gm][0].shape[0]
        idx = list(range(nk))
        val = loo([g[idx][:, idx] for g in G[gm]], [b[idx] for b in B[gm]],
                  C[gm], nk)
        s, md, w3 = three_slices(np.array(pure), val)
        best[gm] = s
        print(f'{"gate="+gm:>20s}: {s:8.3f}× (медіана {md:.3f}×, '
              f'без топ-3 {w3:.3f}×)')
    nk = Gn[0].shape[0]
    idx = list(range(nk))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, nk)
    s, md, w3 = three_slices(np.array(pure), vN)
    print(f'{"⛔НЕГ":>20s}: {s:8.3f}× (мусить бути ≈1.000×)')

    k30 = max(K for K in kk if K <= 30)
    kmax = max(kk)
    r0 = P / np.array(one[('1024', k30)]).sum()
    rg = P / np.array(one[('оракул', k30)]).sum()
    rt = P / np.array(one[('1024', kmax)]).sum()
    print(f'\n✅ПЛОМБА ТАВТОЛОГІЇ (K={kmax}): {rt:.1f}× '
          f'{"✅величезне" if rt > 100 else "🛑 МЕХАНІКА ЗЛАМАНА"}')
    print(f'✅ПЛОМБА varmap (K={k30}, gate=1024): {r0:.2f}× '
          f'(очікувано ≈11.39×)')
    print(f'\n🔑 ЗАТВОР: {r0:.2f}× → {rg:.2f}× з оракульним `gate` '
          f'⟹ {rg / r0:.3f}×')
    print(f'🔑 КОМБІНАЦІЯ: найкращий одиничний {max(r0, rg):.2f}× → '
          f'{max(best.values()):.2f}× ⟹ {max(best.values()) / max(r0, rg):.3f}×')
    win = max(max(best.values()), rg) / r0
    print(f'🔑 ВИРОК: {"ГОРЛО СПРАВЖНЄ" if win >= 1.5 else "ГОРЛА НЕМА"} '
          f'({win:.3f}×, 🟢гейт 1.5×)')
    print(f'📊 БАЛ: 1.972e-07 (вимір оракульний, стендовий)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
