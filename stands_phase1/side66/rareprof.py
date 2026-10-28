"""ЧИ НЕСУТЬ ПРОМІЖНІ СТАНИ БІЛЬШЕ, НІЖ БЕРЕ СКАЛЯРНА λ (ідея юзера, С76).

════════ ЗВІДКИ ПИТАННЯ ════════
Юзер: «а якщо дати не тільки вихід, а й ПРОМІЖНІ СТАНИ?» Попередній вирок
(«форма траєкторії x̄ несе нуль») стосувався траєкторії ПО СЕМПЛАХ. Тут інше:
проміжні стани ПО ГЛИБИНІ, того самого проходу.

Що вже зміряно і чому це НЕ закриває питання:
  * `jitprof.py` подавав проміжні стани, АЛЕ (а) стиснуті в СУМУ, (б) лінійно,
    (в) фіт ПО ЖЕРЕБАХ усередині мережі. Дало 0.0359 без вільного члена.
  * `selfinfo.py` показав, що вибіркові моменти ТОГО САМОГО проходу знають
    58–76% шуму середнього — але при ЗНАННІ ІСТИННИХ значень ознак.

════════ ДЕ САМЕ ОПОРА Є, А ДЕ ЇЇ НЕМА (це і є вісь стенду) ════════
🚨 Глибше шару 8 замикання НЕ рахується ⟹ `x̄_ℓ` нема з чим порівняти:
   ознака без опори — це число, а не похибка. Тому «дати трансформеру всі 32
   шари» упирається не в модель, а у ВІДСУТНІСТЬ ЕТАЛОНА.
🚨 Фолдові ознаки (`x̄^A − x̄^B`) аналітично марні: для iid різниця половин
   НЕЗАЛЕЖНА від їх суми ⟹ несе розмір шуму, але не знак. Не міряємо.
🟢 Лишається те, де опора Є: шари 1..8. Наша схема бере звідти `λ·δ` зі
   СКАЛЯРНОЮ λ на (родину, шар) — тобто однаковою для ВСІХ 256 нейронів.
   А `whocarries.py` зміряв, що похибка розподілена по нейронах РІЗКО
   нерівномірно: рідкісні (`w=min(q,1−q)<0.005`, 29.4% нейронів) несуть 43%
   шуму — 9.955× проти 6.280× у частих і 7.932× у випадкової групи.
   ⟹ скалярна λ усереднює те, що усереднювати не можна.

════════ ЩО МІРЯЄМО (ВИГРАШ, не стеля) ════════
Перенос ЛІНІЙНИЙ за `v` ⟹ розщеплюємо КОЖЕН рядок внеску на G підрядків за
групою рідкісності нейрона `w = min(gate, 1−gate)` того самого шару. Тоді
`pred(λ) = pred₀ − λᵀF` для 4·8·G коефіцієнтів, усі з ОДНОГО прогону.
  BASE      — рідні λ сабміту (база, 32 числа)
  FIT32     — перефіт тих самих 32 (контроль приладу)
  FITG      — 32·G за групами рідкісності          ← ідея «проміжні стани»
  ⛔ROLLG   — ті самі 32·G, але групи ЗСУНУТІ по нейронах (та сама ємність,
              зруйнована структура) — головний контроль
  NL32/NLG  — плюс МОДУЛЯЦІЯ λ скалярними ознаками жеребу (‖δ‖/‖опора‖ на
              шарі 8): справжня нелінійність, теж без зайвих прогонів
Усі λ фітуються **LOO ПО МЕРЕЖАХ** (мережа виміру не бачила ні своїх жеребів,
ні своєї істини) — бо в сабміті λ глобальні. MSE — КРОС-ЦІЛЬОВА (дві незалежні
істини), тож шум істини не входить в очікування.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 FIT32 = 0.99–1.02× (рідні λ фітились на більшому наборі — перефіт не мусить
   давати нічого; це перевірка приладу, а не результат).
📣 FITG  = **1.00–1.04×**. Чому не більше: закон «стеля каналу = шар його
   точності» дає для каналів на 1..8 стелю 1.72×, і родина `Q` уже бере 1.20×
   з неї; профілювання — це тонша настройка ВЖЕ ВЗЯТОГО каналу, а не новий.
📣 NL    ≤ FITG + 0.01.
📣 ⛔ROLLG ≈ 1.00× (±0.02) — інакше виграш є від ЄМНОСТІ, а не від структури.
🟢 ЖИВЕ, якщо FITG ≥ 1.05× ПРИ ROLLG ≤ 1.02×. Інакше клас «поелементний
   профіль λ» закрито, і разом із ним — уся форма «дати моделі проміжні стани
   на шарах з опорою».

🚨 ПЛОМБИ
  ✅ ТОТОЖНІСТЬ: `pred₀ − λ_рідніᵀF` (з однаковою λ в усіх групах) мусить
     збігтися з `run72(рідні λ)` до ~1e-12 — інакше збір зламав базу;
  ✅ СУМА ГРУП: Σ_g F[(q,i,g)] мусить дорівнювати нерозщепленому рядку;
  ✅ ‖b‖² ≥ 0 і крос-MSE > 0;
  ⛔ ROLLG (див. вище).
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, closure_parts, run_side1 as run72,   # noqa: E402
                    W_, GATE_N, INV_SQRT_2PI, _phi)
from biassplit import truth_full                                         # noqa: E402
from varsub import S72, A72, Q72, G72                                    # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSIG = 8
LAM = (S72, A72, Q72, G72)
FAM = ('S: δs', 'A: δE[x]', 'Q: δE[x²]', 'G: δмаска')
# межі груп рідкісності — РІВНО ті, на яких `whocarries.py` зміряв
# нерівномірність внеску (9.955 / 3.380 / 4.004 / 6.280)
EDGES = (0.005, 0.05, 0.2)
G_ = len(EDGES) + 1
NZ = 3                       # скалярні ознаки жеребу для модуляції λ
RIDGE = 1e-8                 # відносний, зафіксований ДО прогону
# сітка усадки для ВКЛАДЕНОГО вибору (фіксована наперед, не за результатом)
RGRID = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def run_collect_g(wc, npair, cl, dev, seed, edges, roll=0):
    """Порт сабміту + збір внесків, РОЗЩЕПЛЕНИХ за групою рідкісності нейрона.

    Повертає (pred₀, F, z), де F має 4·ksig·G рядків, уже перенесених до
    виходу, а `z` — NZ скалярних ознак жеребу (‖δ‖/‖опора‖ на шарі ksig).
    `roll` ≠ 0 ⟹ ⛔контроль: мітки груп зсунуті по нейронах.
    """
    s_cl, m_cl, f_cl, q_cl = cl
    ksig = len(s_cl)
    depth = len(wc)
    G = len(edges) + 1
    bnd = torch.tensor(edges, dtype=torch.float64, device=dev)

    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (x.T @ x) / denom
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    w0 = wh @ wc[0]

    xa = torch.relu(x @ w0)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = xa.double().mean(0) - sg0 * INV_SQRT_2PI

    F = torch.zeros(4 * ksig * G, wc[0].shape[1], dtype=torch.float64,
                    device=dev)
    zf = torch.zeros(NZ, dtype=torch.float64, device=dev)
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
            xa = torch.relu(p)
            xd = xa.double()
            d4 = (_phi(mh / sh) * (sh - s_cl[i - 1]),
                  xd.mean(0) - m_cl[i - 1],
                  xd.pow(2).mean(0) - q_cl[i - 1],
                  gate - f_cl[i - 1])
            wmin = torch.minimum(gate, 1.0 - gate)
            gid = torch.bucketize(wmin, bnd)
            if roll:
                gid = torch.roll(gid, roll)
            for q, dd in enumerate(d4):
                for gg in range(G):
                    F[(q * ksig + i - 1) * G + gg] += dd * (gid == gg)
            if i == ksig:
                # ознаки жеребу: РОЗМІР розбіжності з опорою, безрозмірний
                # (доступний у сабміті: чисельник і знаменник обидва свої)
                zf[0] = d4[2].norm() / q_cl[i - 1].norm().clamp_min(1e-30)
                zf[1] = d4[1].norm() / m_cl[i - 1].norm().clamp_min(1e-30)
                zf[2] = d4[0].norm() / s_cl[i - 1].norm().clamp_min(1e-30)
        else:
            xa = torch.relu(p)
    return xa.double().mean(0) - v, F, zf


def expand(F, z, zmu, zsd):
    """Додати рядки МОДУЛЯЦІЇ: (нормована ознака) × (нерозщеплений рядок).

    Центрування/масштаб z НЕ міняють простір моделі (бо нерозщеплені рядки
    вже в ньому як лінійна комбінація групових) — лише обумовленість."""
    ksig = KSIG
    base = F.reshape(4 * ksig, G_, -1).sum(1)          # (32, n) сума груп
    zn = (z - zmu) / zsd
    return torch.cat([F] + [zn[j] * base for j in range(NZ)], 0)


def p_full(nr):
    """Матриця обмеження «однакова λ в усіх групах» (nr×32) для FIT32."""
    P = np.zeros((nr, 4 * KSIG))
    for r in range(4 * KSIG):
        for g in range(G_):
            P[r * G_ + g, r] = 1.0
    return P


def solve(Gm, b, ridge, th0):
    """Ridge У ДІАГОНАЛЬНО НОРМОВАНИХ координатах, З ЦЕНТРОМ У ДІЮЧІЙ БАЗІ.

    🚨 Два окремі рішення, обидва потрібні:
    (1) нормування — без нього однаковий ridge карав би родини по-різному
        (масштаби `S` і `Q` різняться на порядки);
    (2) 🔑 центр `θ0` = рідні λ сабміту, а НЕ нуль. Інакше при сильній усадці
        варіант сходиться до `λ=0` (гірше за базу), і сітка усадки не містить
        самої бази — тоді «×BASE < 1» означає лише, що фітер не дотягнувся,
        а не що ідея погана. З центром у базі α→∞ дає РІВНО BASE, тож число
        читається прямо: це ПРИРІСТ понад діюче.
    """
    s = np.sqrt(np.clip(np.diag(Gm), 1e-300, None))
    Gn = Gm / np.outer(s, s)
    th = np.linalg.solve(Gn + ridge * np.eye(Gm.shape[0]),
                         b / s + ridge * (th0 * s))
    return th / s


def main(nets=24, dev=DEV, ndraw=NDRAW):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'СТЕНД: {nets} мереж × {ndraw} жеребів, G={G_} груп рідкісності '
          f'{EDGES}, LOO ПО МЕРЕЖАХ, крос-ціль, dev={dev}')
    print('📣 ПЕРЕДБАЧЕНО: FIT32 0.99–1.02× · FITG 1.00–1.04× · '
          'NL ≤ FITG+0.01 · ⛔ROLLG ≈1.00×')
    print('🟢 ЖИВЕ при FITG ≥ 1.05× і ROLLG ≤ 1.02×\n', flush=True)

    NRG = 4 * KSIG * G_
    NR = NRG + NZ * 4 * KSIG
    # накопичувачі ПО МЕРЕЖАХ (для LOO): нормальні рівняння + крос-скаляри
    acc = {k: [] for k in ('Gm', 'b1', 'b2', 'c12', 'Gr', 'r1', 'r2')}
    Zs, seal_id, seal_sum = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            cl = closure_parts(wc, KSIG, dev, extras=True)
            s_cl, m_cl, f_cl, q_cl = cl
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)

            Fs, Frs, D1, D2, Zn = [], [], [], [], []
            for k in range(ndraw):
                sd = 61_000_000 + 977 * n + k
                p0, F, z = run_collect_g(wc, npair, cl, dev, sd, EDGES)
                pr0, Fr, _ = run_collect_g(wc, npair, cl, dev, sd, EDGES,
                                           roll=7)
                if n == 0 and k == 0:
                    # ✅ПЛОМБА ТОТОЖНОСТІ: рідні λ через мій збір == run72
                    lam = torch.tensor(
                        [LAM[q][i] for q in range(4) for i in range(KSIG)
                         for _ in range(G_)], dtype=torch.float64, device=dev)
                    mine = p0 - lam @ F
                    ref, _ = run72(wc, npair, s_cl, m_cl, dev, sd,
                                   lam_s=S72, lam_a=A72, lam_q=Q72,
                                   lam_g=G72, q_ext=q_cl, p_ext=f_cl)
                    seal_id.append(float((mine - ref).abs().max()
                                         / ref.abs().max()))
                    # ✅ПЛОМБА СУМИ ГРУП
                    seal_sum.append(float((F.reshape(4 * KSIG, G_, -1).sum(1)
                                           - Fr.reshape(4 * KSIG, G_, -1)
                                           .sum(1)).abs().max()
                                          / F.abs().max()))
                Fs.append(F); Frs.append(Fr); Zn.append(z)
                D1.append(p0 - T1); D2.append(p0 - T2)
            Fst = torch.stack(Fs, 0)
            Frst = torch.stack(Frs, 0)
            Zt = torch.stack(Zn, 0)
            Zs.append(Zt.cpu().numpy())
            d1 = torch.stack(D1, 0); d2 = torch.stack(D2, 0)
            # накопичення робимо ПІСЛЯ, коли відомі zmu/zsd ⟹ тримаємо сирі
            acc.setdefault('raw', []).append(
                (Fst.cpu(), Frst.cpu(), Zt.cpu(), d1.cpu(), d2.cpu()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  ({time.time()-t0:.0f}s)', flush=True)

    Za = np.concatenate(Zs, 0)
    zmu = torch.tensor(Za.mean(0)); zsd = torch.tensor(Za.std(0) + 1e-30)
    print(f'\n  ознаки жеребу z: сер {Za.mean(0)} · ст.в/сер '
          f'{Za.std(0)/np.abs(Za.mean(0))}')

    Gm, b1, b2, c12, Gr, r1, r2 = [], [], [], [], [], [], []
    for (Fst, Frst, Zt, d1, d2) in acc['raw']:
        for src, GG, B1, B2 in ((Fst, Gm, b1, b2), (Frst, Gr, r1, r2)):
            Fe = torch.stack([expand(src[k], Zt[k], zmu, zsd)
                              for k in range(src.shape[0])], 0)
            # 🚨 GEMM, не einsum: (k,r,i) → (r, k·i); einsum тут ішов би
            #    хвилинами на кожній мережі
            M = Fe.permute(1, 0, 2).reshape(Fe.shape[1], -1)
            GG.append((M @ M.T).numpy())
            B1.append((M @ d1.reshape(-1)).numpy())
            B2.append((M @ d2.reshape(-1)).numpy())
        c12.append(float((d1 * d2).sum()))
    Gm = np.array(Gm); b1 = np.array(b1); b2 = np.array(b2)
    Gr = np.array(Gr); r1 = np.array(r1); r2 = np.array(r2)
    c12 = np.array(c12)
    ND = nets * ndraw

    lam_base = np.array([LAM[q][i] for q in range(4) for i in range(KSIG)
                         for _ in range(G_)] + [0.0] * (NR - NRG))
    P32 = np.zeros((NR, 4 * KSIG)); P32[:NRG] = p_full(NRG)
    PNL = np.zeros((NR, 4 * KSIG * (1 + NZ)))
    PNL[:NRG, :4 * KSIG] = p_full(NRG)
    PNL[NRG:, 4 * KSIG:] = np.eye(NZ * 4 * KSIG)
    PG = np.zeros((NR, NRG)); PG[:NRG] = np.eye(NRG)
    PGNL = np.eye(NR)

    def mse_of(lam, GA, B1, B2, idx):
        """крос-MSE варіанта `lam` на мережах `idx` (сума по жеребах)."""
        s = 0.0
        for n in idx:
            s += (c12[n] - lam @ B1[n] - lam @ B2[n] + lam @ GA[n] @ lam)
        return s / (len(idx) * ndraw)

    def loo(P, GA, B1, B2, ridge=None, lam0=None):
        """LOO по мережах: λ фітується на решті, міряється на своїй.

        `ridge=None` ⟹ ВКЛАДЕНИЙ вибір: усередині кожного зовнішнього фолду
        сила усадки обирається другим LOO по ТРЕНУВАЛЬНИХ мережах. Так вирок
        не залежить від мого довільного числа і не бачить тестової мережі."""
        # усе — одразу в координатах θ (λ = Pθ): `lam@B = θ@(PᵀB)`,
        # `lam@G@lam = θ@(PᵀGP)@θ` ⟹ проєктуємо ОДИН раз, не в циклі
        Gp = np.array([P.T @ GA[n] @ P for n in range(nets)])
        c1 = np.array([P.T @ B1[n] for n in range(nets)])
        c2 = np.array([P.T @ B2[n] for n in range(nets)])
        Gs = Gp.sum(0); bs = 0.5 * (c1.sum(0) + c2.sum(0))
        # θ0 — рідні λ у координатах цього варіанта; ✅перевіряємо, що
        # P·θ0 ВІДТВОРЮЄ базу точно (інакше «×BASE» ні з чим не порівнювати)
        th0 = np.linalg.lstsq(P, lam_base if lam0 is None else lam0,
                              rcond=None)[0]
        rec = float(np.abs(P @ th0 - (lam_base if lam0 is None else lam0)).max())
        assert rec < 1e-12, f'θ0 не відтворює базу: {rec:.2e}'

        def err(th, m):
            return c12[m] - th @ c1[m] - th @ c2[m] + th @ Gp[m] @ th

        tot, picked = 0.0, []
        for n in range(nets):
            Gtr = Gs - Gp[n]; btr = bs - 0.5 * (c1[n] + c2[n])
            if ridge is None:
                best, rr = None, RGRID[0]
                for r in RGRID:
                    e = sum(err(solve(Gtr - Gp[m],
                                      btr - 0.5 * (c1[m] + c2[m]), r, th0), m)
                            for m in range(nets) if m != n)
                    if best is None or e < best:
                        best, rr = e, r
            else:
                rr = ridge
            picked.append(rr)
            tot += err(solve(Gtr, btr, rr, th0), n)
        return tot / (nets * ndraw), picked

    base = mse_of(lam_base, Gm, b1, b2, range(nets))
    VAR = (('FIT32 (перефіт 32, LOO)', P32, Gm, b1, b2),
           ('FITG (32×%d за рідкісністю, LOO)' % G_, PG, Gm, b1, b2),
           ('⛔ROLLG (та сама ємність, зсунуті групи)', PG, Gr, r1, r2),
           ('NL32 (32 + модуляція z, LOO)', PNL, Gm, b1, b2),
           ('NLG (усе разом, LOO)', PGNL, Gm, b1, b2))
    res = [('BASE (рідні λ сабміту)', base)]
    picks = {}
    for nm, P, GA, B1, B2 in VAR:
        m, pk = loo(P, GA, B1, B2)
        res.append((nm, m))
        picks[nm] = pk

    print(f'\n✅ПЛОМБА ТОТОЖНОСТІ (мій збір == run72): {seal_id[0]:.2e} '
          f'{"✅" if seal_id[0] < 1e-9 else "🛑 ЗБІР ЗЛАМАВ БАЗУ"}')
    print(f'✅ПЛОМБА СУМИ ГРУП (Σ_g не залежить від групування): '
          f'{seal_sum[0]:.2e} {"✅" if seal_sum[0] < 1e-12 else "🛑"}')
    print(f'\n{"варіант":<42} {"крос-MSE":>12} {"×BASE":>8} {"усадка":>9}')
    for nm, m in res:
        pk = picks.get(nm)
        tag = '' if pk is None else f'{np.median(pk):.0e}'
        print(f'{nm:<42} {m:>12.5e} {base/m:>8.4f} {tag:>9}')

    print('\n   ЧУТЛИВІСТЬ ДО УСАДКИ (діагностика, вирок береться з вкладеного '
          'вибору вище)')
    print(f'   {"ridge":>8} {"FIT32":>9} {"FITG":>9} {"⛔ROLLG":>9}')
    for r in RGRID:
        a = base / loo(P32, Gm, b1, b2, ridge=r)[0]
        b_ = base / loo(PG, Gm, b1, b2, ridge=r)[0]
        c_ = base / loo(PG, Gr, r1, r2, ridge=r)[0]
        print(f'   {r:>8.0e} {a:>9.4f} {b_:>9.4f} {c_:>9.4f}')

    fg = base / res[2][1]
    rl = base / res[3][1]
    f32 = base / res[1][1]
    print(f'\n   FITG / FIT32 = {f32 and fg / f32:.4f}× ← чистий внесок '
          f'ГРУПУВАННЯ (мій фітер в обох однаковий)')
    live = fg >= 1.05 and rl <= 1.02
    print(f'\n⛔ROLLG = {rl:.4f}× {"✅контроль чистий" if rl <= 1.02 else "🛑 виграш від ЄМНОСТІ"}')
    print(f'🔑 ВИРОК: FITG = {fg:.4f}× при гейті 1.05 ⟹ '
          f'{"🟢 ЖИВЕ" if live else "⛔ МЕРТВО"}')
    print(f'   ⟹ клас «проміжні стани на шарах з опорою, поелементно» '
          f'{"вимагає сабміту" if live else "ЗАКРИТО"}')
    print(f'\n📊 БАЛ: 1.954e-07 (діагностика; естиматор не чіпаємо до вироку)')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
