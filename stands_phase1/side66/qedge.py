"""ЧИМ ЖИВЕ λ-КАНАЛ: ШУМОМ ВИБІРКИ ЧИ ЗМІЩЕННЯМ ЗАМИКАННЯ (С84)

## ЗВІДКИ ПИТАННЯ — З ДВОХ ПРОВАЛІВ ПОСПІЛЬ
1. `edgeksig.py`: точніша `m_cl` (Еджворт) ⟹ MSE ГІРШЕ ×0.875, монотонно з `λ`.
   ⛔пломби чисті: γ=0 ≡ база РІВНО, обернений знак теж гірший (×0.927).
2. `cutcheck.py`: обрізання ні до чого — на `wc` виграш точності ТОЙ САМИЙ
   (шар 8: ×1.529 проти ×1.535 на повних `W`).
⟹ `m_cl` СПРАВДІ точніша, і сабміт від цього СПРАВДІ гіршає. Отже питання не
в точності, а в тому, ЩО САМЕ канал використовує.

## 🔑 РОЗКЛАД, ЯКИЙ ВИРІШУЄ
Ознака `δ = x̄ − m_cl` розпадається ТОЧНО на дві частини:
    `δ = (x̄ − істина)` + `(істина − m_cl)` = **ШУМ ВИБІРКИ** + **ЗМІЩЕННЯ**
Перша залежить від жеребу й має нульове середнє; друга детермінована для
мережі. Обидві переносяться до виходу тим самим ланцюгом воріт.
🎯 Проганяємо сабміт ТРИЧІ: з повною `δ`, лише зі ШУМОВОЮ частиною, лише зі
ЗМІЩЕННЯМ. Хто дає виграш — той і годує канал.
📌 «Істина» для проміжних шарів береться як середнє `x̄` по D незалежних
жеребах (D=32) ⟹ це ОРАКУЛ, у рантаймі недоступний; тут він потрібен лише
щоб розділити внески.

⛔ ПЛОМБИ:
 (а) `δ_шум + δ_зміщ ≡ δ_повна` ⟹ MSE(шум)+MSE(зміщ) конструкція мусить
     відтворювати MSE(повна) при однакових `λ` — звіряємо ТОТОЖНІСТЬ векторів;
 (б) `λ=0` у всіх трьох ⟹ ОДИН І ТОЙ САМИЙ MSE;
 (в) істина — БАНК, крос-ціль (шум банку знято);
 (г) оракульна «істина» шару має власний шум ~1/√(32·npair) — друкуємо його.

📣 ПЕРЕДБАЧЕНО: якщо канал живе на ШУМІ (як завжди вважали), то варіант «лише
   шум» дасть ≈ повну, а «лише зміщення» ≈ нічого. Тоді провал Еджворта
   незрозумілий і треба шукати далі. Якщо ж «лише зміщення» дає більшу
   частину — то λ-канали сабміту 60 сесій працювали НЕ на тому, що ми думали,
   і це міняє мішень: не гасити шум точніше, а КРАЩЕ оцінювати зміщення.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/whatfeeds.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS,
                        ALPHA, W_BLEND)


def true_cumulants(wc, dev, ns, bs):
    """ІСТИННІ γ₁,γ₂ маргіналів `h` пошарово, крізь ОБРІЗАНІ ваги."""
    d = len(wc)
    S1 = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    S2 = [torch.zeros_like(x) for x in S1]; S3 = [torch.zeros_like(x) for x in S1]
    S4 = [torch.zeros_like(x) for x in S1]
    gen = torch.Generator(device=dev); gen.manual_seed(4242)
    done = 0
    while done < ns:
        b = min(bs, ns - done)
        a = torch.randn(b, W_, device=dev, generator=gen)
        for i in range(d):
            h = a @ wc[i]; hd = h.double()
            S1[i] += hd.sum(0); S2[i] += (hd * hd).sum(0)
            S3[i] += (hd ** 3).sum(0); S4[i] += (hd ** 4).sum(0)
            a = torch.relu(h)
        done += b
    g1, g2 = [], []
    for i in range(d):
        mh = S1[i] / ns
        v = (S2[i] / ns - mh * mh).clamp_min(1e-30)
        k3 = S3[i] / ns - 3 * mh * (S2[i] / ns) + 2 * mh ** 3
        k4 = (S4[i] / ns - 4 * mh * (S3[i] / ns) + 6 * mh ** 2 * (S2[i] / ns)
              - 3 * mh ** 4 - 3 * v * v)
        g1.append(k3 / v ** 1.5); g2.append(k4 / v ** 2)
    return g1, g2

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
L_, NDRAW, NORA = 32, 8, 32
CS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів (+{NORA} для оракула), {DEV}\n',
      flush=True)


def feats(wc, npair, dev, seed):
    """Сирі ВИБІРКОВІ статистики шарів 1..KSIG (без віднімання замикання)."""
    depth = len(wc)
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    x = torch.cat([torch.relu(z @ (wh @ wc[0])), torch.relu(-(z @ (wh @ wc[0])))], 0)
    den = float(2 * npair)
    out = []
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        pd = p.double()
        mh = pd.sum(0) / den
        sh = (pd.pow(2).sum(0) / den - mh * mh).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
        if i <= KSIG:
            # 🚨 `sh` БЕЗ множника φ: у сабміті ознака = φ(t)·(sh − s_cl), тобто
            #    φ — СПІЛЬНИЙ множник різниці. Перша версія рахувала
            #    φ·sh − s_cl і дала MSE у 37× гіршу за базу ⟹ стенд не
            #    відтворював сабміт. 🎓 РОЗКЛАДАТИ ТРЕБА ТЕ, ЩО В ДУЖКАХ.
            out.append((sh, x.double().sum(0) / den,
                        x.double().pow(2).sum(0) / den, gate,
                        _phi(mh / sh)))
    return out


def run(wc, npair, dev, seed, cl, DL):
    """Прогін сабміту, де δ-ознаки ПІДМІНЕНІ на `DL` (список кортежів по 4)."""
    depth = len(wc)
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    den = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / den - sg0 * INV_SQRT_2PI
    m_deep = cl[4]
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        x = torch.relu(p)
        if i <= KSIG:
            d = DL[i - 1]
            v = (v + SP.LAM_S[i-1]*d[0] + SP.LAM_A[i-1]*d[1]
                 + SP.LAM_Q[i-1]*d[2] + SP.LAM_G[i-1]*d[3])
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


LAM0 = (SP.LAM_S, SP.LAM_A, SP.LAM_Q, SP.LAM_G)
# 0: сабміт · 1: q очищено ОРАКУЛОМ (стеля) · 2: q очищено ЕДЖВОРТОМ
#            · 3: ⛔контроль — Еджворт з ОБЕРНЕНИМ знаком
NV = 4
MSE = np.zeros((len(FS), NV, len(CS)))
SEAL_ID = 0.0
SEAL_RUN = 0.0
SEAL_L0 = np.zeros((len(FS), NV))
ORNOISE = np.zeros(len(FS))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        s_cl, m_cl, q_cl, P_cl, _ = cl
        # 🔑 ЛОКАЛЬНА поправка до q: γ істинні, але C ланцюга НЕ міняється.
        #    Саме цим відрізняється від ⛔edgeksig, де поправка входила в C і
        #    поширювалась глибше, псуючи все нижче.
        g1t, g2t = true_cumulants(wc, DEV, 1 << 18, 1 << 13)
        QE = []
        _m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=DEV)
        _C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=DEV)
        for _i in range(len(wc)):
            _w = wc[_i].double()
            _cp = _w.T @ (_C @ _w)
            _s = _cp.diagonal().clamp_min(1e-20).sqrt()
            _mp = _m @ _w
            _t = _mp / _s
            _ph, _Ph = _phi(_t), _Phi(_t)
            if 1 <= _i <= KSIG:
                QE.append((g1t[_i] / 3.0) * _s * _s * _ph
                          - (g2t[_i] / 12.0) * _s * _s * _t * _ph)
            _m = _mp * _Ph + _s * _ph
            if _i < len(wc) - 1:
                _d2 = (_mp * _mp + _s * _s) * _Ph + _mp * _s * _ph
                _C = (_cp * torch.outer(_Ph, _Ph)
                      + (0.5 * _cp * _cp) * torch.outer(_ph / _s, _ph / _s))
                _C = _C + torch.diag((_d2 - _m * _m) - _C.diagonal())
        # ── ОРАКУЛ: «істина» проміжних шарів = середнє по NORA жеребах
        acc = [[torch.zeros_like(t) for t in tup]
               for tup in feats(wc, npair, DEV, 5_000_000)]
        acc2 = [[torch.zeros_like(t) for t in tup] for tup in acc]
        for d in range(NORA):
            ft = feats(wc, npair, DEV, 5_000_000 + 977 * n + d)
            for k in range(len(ft)):
                for j in range(5):
                    acc[k][j] += ft[k][j] / NORA
                    acc2[k][j] += ft[k][j] ** 2 / NORA
        # ⛔(г) шум самого оракула
        ORNOISE[n] = float(sum(((acc2[k][1] - acc[k][1] ** 2).clamp_min(0)
                                / NORA).sum() for k in range(len(acc))).sqrt())
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        REF = (s_cl, m_cl, q_cl, P_cl)
        for d in range(NDRAW):
            sd = 95_000_000 + 733 * n + d
            ft = feats(wc, npair, DEV, sd)
            DLS = []
            for vi in range(NV):
                L = []
                for k in range(len(ft)):
                    row = []
                    for j in range(4):
                        full = ft[k][j] - REF[j][k]
                        bias = acc[k][j] - REF[j][k]
                        if j == 2 and vi == 1:
                            u = full - bias                 # ОРАКУЛ
                        elif j == 2 and vi == 2:
                            u = full - QE[k]                # ЕДЖВОРТ
                        elif j == 2 and vi == 3:
                            u = full + QE[k]                # ⛔обернений знак
                        else:
                            u = full
                        row.append(u * ft[k][4] if j == 0 else u)
                    L.append(tuple(row))
                DLS.append(L)
            # ⛔ПЛОМБА (д): `run` з ПОВНОЮ δ ≡ `predict91`. 🚨 Саме її бракувало:
            #    без неї перша версія рахувала ознаку `S` як φ·sh − s_cl
            #    замість φ·(sh − s_cl) і давала MSE у 37× гіршу за базу.
            if d == 0:
                SP.LAM_S, SP.LAM_A, SP.LAM_Q, SP.LAM_G = LAM0
                _p1 = run(wc, npair, DEV, sd, cl, DLS[0])
                _p2, _ = SP.predict91(wc, npair, DEV, seed=sd, cl=cl)
                SEAL_RUN = max(SEAL_RUN, float((_p1 - _p2).abs().max()))
            # ⛔ПЛОМБА (а): шум + зміщення ≡ повна, ТОТОЖНО
            if d == 0:
                SEAL_ID = max(SEAL_ID, max(
                    float((DLS[1][k][j] + DLS[2][k][j] - DLS[0][k][j]).abs().max())
                    for k in range(len(ft)) for j in range(4)))
            for k, c in enumerate(CS):
                SP.LAM_S = tuple(c*x for x in LAM0[0])
                SP.LAM_A = tuple(c*x for x in LAM0[1])
                SP.LAM_Q = tuple(c*x for x in LAM0[2])
                SP.LAM_G = tuple(c*x for x in LAM0[3])
                for vi in range(NV):
                    pr = run(wc, npair, DEV, sd, cl, DLS[vi])
                    e = float(((pr - T1) * (pr - T2)).sum())
                    MSE[n, vi, k] += e / NDRAW
                    if k == 0:
                        SEAL_L0[n, vi] = MSE[n, vi, 0]
    SP.LAM_S, SP.LAM_A, SP.LAM_Q, SP.LAM_G = LAM0
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/whatfeeds.npz', MSE=MSE, CS=CS)
print(f'\n⛔ПЛОМБА (д) run(повна δ) ≡ predict91: max|Δ| = {SEAL_RUN:.2e} '
      f'{"✅" if SEAL_RUN < 1e-15 else "🚨 СТЕНД НЕ ВІДТВОРЮЄ САБМІТ"}')
print(f'⛔ПЛОМБА (а) шум+зміщення ≡ повна: max|Δ| = {SEAL_ID:.2e} '
      f'{"✅" if SEAL_ID < 1e-15 else "🚨 РОЗКЛАД НЕ ТОТОЖНИЙ"}')
sp = float(np.abs(SEAL_L0 - SEAL_L0[:, :1]).max())
print(f'⛔ПЛОМБА (б) при λ=0 всі три однакові: max|Δ| = {sp:.2e} '
      f'{"✅" if sp < 1e-18 else "🚨"}')
print(f'⛔ПЛОМБА (г) шум оракула (32 жереби): {ORNOISE.mean():.3e}\n')

NM = ['⛔ сабміт', '🟢 q ОРАКУЛ (стеля)', '🎯 q ЕДЖВОРТ', '⛔ Едж, знак −']
m = [MSE[:, v, :].mean(0) for v in range(NV)]
print(f'{"c (×λ)":>8s} ' + ' '.join(f'{x:>13s}' for x in NM))
print('─' * 92)
for k, c in enumerate(CS):
    print(f'{c:8.2f} ' + ' '.join(f'{m[v][k]:13.5e}' for v in range(NV)))
b0 = m[0][0]
print(f'\n{"варіант":>22s} {"оптимум c":>10s} {"MSE":>13s} {"× до λ=0":>10s} '
      f'{"частка виграшу":>15s}')
print('─' * 76)
gains = []
for v in range(NV):
    k = int(np.argmin(m[v])); g = b0 / m[v][k]
    gains.append(g)
    print(f'{NM[v]:>22s} {CS[k]:10.2f} {m[v][k]:13.5e} {g:10.4f} '
          f'{(g-1)/max(gains[0]-1,1e-12)*100:14.1f}%')
print('\n🎯 ВИРОК — ДЕ САМЕ СИДИТЬ ШКІДЛИВЕ ЗМІЩЕННЯ')
for v in range(1, 5):
    part = (gains[v] - gains[0]) / max(gains[5] - gains[0], 1e-12) * 100
    print(f'   {NM[v]:>20s}: ×{gains[v]/gains[0]:.4f} до сабміту '
          f'({part:5.1f}% усього ефекту очищення)')
print(f'   {"🟢 ВСІ разом":>20s}: ×{tot:.4f}  ⟹ бал 1.954e-07 → '
      f'{1.954e-07/max(tot,1e-9):.4e}')
print('   🚨 це ОРАКУЛ (зміщення з 32 жеребів). Практична цінність — у тому,')
print('      ЯКА родина винна: її одну треба навчитись чистити аналітично.')
print(f'\n📊 БАЛ 1.954e-07 · діагностика; «зміщення» — ОРАКУЛ, у рантаймі його нема')
