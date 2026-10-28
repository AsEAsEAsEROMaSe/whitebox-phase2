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
            p3 = pd.pow(3).sum(0) / den
            p4 = pd.pow(4).sum(0) / den
            p2 = pd.pow(2).sum(0) / den
            k3 = p3 - 3 * mh * p2 + 2 * mh ** 3
            k4 = p4 - 4 * mh * p3 + 6 * mh ** 2 * p2 - 3 * mh ** 4 - 3 * sh ** 4
            out.append((sh, x.double().sum(0) / den,
                        x.double().pow(2).sum(0) / den, gate,
                        _phi(mh / sh), k3 / sh ** 3, k4 / sh ** 4))
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
# 0: сабміт · 1: q чищено γ З ТОГО Ж ПРОХОДУ (реалізовне!) · 2: ⛔знак −
NV = 3
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
        # ── ОРАКУЛ: «істина» проміжних шарів = середнє по NORA жеребах
        acc = [[torch.zeros_like(t) for t in tup]
               for tup in feats(wc, npair, DEV, 5_000_000)]
        acc2 = [[torch.zeros_like(t) for t in tup] for tup in acc]
        for d in range(NORA):
            ft = feats(wc, npair, DEV, 5_000_000 + 977 * n + d)
            for k in range(len(ft)):
                for j in range(7):
                    acc[k][j] += ft[k][j] / NORA
                    acc2[k][j] += ft[k][j] ** 2 / NORA
        # ⛔(г) шум самого оракула
        ORNOISE[n] = float(sum(((acc2[k][1] - acc[k][1] ** 2).clamp_min(0)
                                / NORA).sum() for k in range(len(acc))).sqrt())
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        REF = (s_cl, m_cl, q_cl, P_cl)
        # опори замикання для поправки: s, t=mp/s, φ(t) на шарах 1..KSIG
        SS, TT_, PH = [], [], []
        _m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=DEV)
        _C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=DEV)
        for _i in range(len(wc)):
            _w = wc[_i].double()
            _cp = _w.T @ (_C @ _w)
            _s = _cp.diagonal().clamp_min(1e-20).sqrt()
            _mp = _m @ _w; _t = _mp / _s
            _ph, _Ph = _phi(_t), _Phi(_t)
            if 1 <= _i <= KSIG:
                SS.append(_s); TT_.append(_t); PH.append(_ph)
            _m = _mp * _Ph + _s * _ph
            if _i < len(wc) - 1:
                _d2 = (_mp * _mp + _s * _s) * _Ph + _mp * _s * _ph
                _C = (_cp * torch.outer(_Ph, _Ph)
                      + (0.5 * _cp * _cp) * torch.outer(_ph / _s, _ph / _s))
                _C = _C + torch.diag((_d2 - _m * _m) - _C.diagonal())
        for d in range(NDRAW):
            sd = 95_000_000 + 733 * n + d
            ft = feats(wc, npair, DEV, sd)
            DLS = []
            for vi in range(NV):
                L = []
                for k in range(len(ft)):
                    row = []
                    # 🔑 поправка до q рахується на ОПОРАХ ЗАМИКАННЯ (детерм.),
                    #    але з γ, знятими з ЦЬОГО ж проходу вибірки
                    qc = ((ft[k][5] / 3.0) * SS[k] * SS[k] * PH[k]
                          - (ft[k][6] / 12.0) * SS[k] * SS[k] * TT_[k] * PH[k])
                    for j in range(4):
                        full = ft[k][j] - REF[j][k]
                        if j == 2 and vi == 1:
                            full = full - qc
                        elif j == 2 and vi == 2:
                            full = full + qc
                        row.append(full * ft[k][4] if j == 0 else full)
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

NM = ['⛔ сабміт', '🎯 q чищено γ З ПРОХОДУ', '⛔ той самий, знак −']
m = [MSE[:, v, :].mean(0) for v in range(NV)]
print(f'{"c (×λ)":>8s} ' + ' '.join(f'{x:>21s}' for x in NM))
print('─' * 76)
for k, c in enumerate(CS):
    print(f'{c:8.2f} ' + ' '.join(f'{m[v][k]:21.5e}' for v in range(NV)))
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
g = gains[1] / gains[0]
print('\n🎯 ВИРОК — ЧИ РЕАЛІЗОВНА ЧИСТКА `q` БЕЗ ІСТИНИ')
print(f'   γ З ТОГО Ж ПРОХОДУ: ×{g:.4f} до сабміту ⟹ бал 1.946e-07 → '
      f'{1.946e-07/max(g,1e-9):.4e}  (від sub92)')
print(f'   ⛔ обернений знак:  ×{gains[2]/gains[0]:.4f} '
      f'{"🚨 ЗНАК НЕ ТОЙ" if gains[2] > gains[1] else "✅"}')
print('   📌 стеля з ОРАКУЛОМ ×1.0896, з ІСТИННИМИ γ ×1.0040')
if g >= 1.003 and gains[2] < gains[1]:
    print('   ⟹ 🟢 РОБИТИ ФАЙЛ: додається до sub92, ціна ~0.1% флопів')
else:
    print('   ⟹ ⛔ НЕ РОБИТИ: виграш нижче шуму / знак не тримається')
print(f'\n📊 БАЛ 1.954e-07 · діагностика; «зміщення» — ОРАКУЛ, у рантаймі його нема')
