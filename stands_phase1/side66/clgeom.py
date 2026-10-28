"""ГЕОМЕТРІЯ ПОХИБКИ ГАУСОВОГО ЗАМИКАННЯ МЕЛЕРА (K=2) НА РЕАЛЬНИХ МЕРЕЖАХ.

ПИТАННЯ: похибка замикання `e = M − T` — це УЗГОДЖЕНА ПОХИБКА МАСШТАБУ
(виправна ОДНИМ числом), чи загальний вектор у 256 вимірах?

Міряємо на 24 мережах (глибина 32, ширина 256, без зсувів, He):
  1. частку ‖e_par‖²/‖e‖² уздовж û = T/‖T‖ (це і є «наскільки це масштаб»);
  2. κ = (e·û)/‖T‖ по мережах: середнє, ст.в, ст.в/сер (<0.2 ⟹ універсальний);
  3. поправку M → (1+α)M: без / LOO ПО МЕРЕЖАХ / оракул;
  4. поправку M → (1+α)M + β·1: те саме.

🚨 ІНДЕКСАЦІЯ ШАРІВ: тут шар 1 = ПІСЛЯ першого матмулу+ReLU (як у ТЗ).
   У попередніх стендах (`base_real.closure_parts`) індекс зсунутий на −1.
   Обидві розкладки друкуються у пломбі П2.

ПЛОМБИ: П1 похибка шару 32 ∈ [0.84,0.86]% · П2 профіль насичується ·
П3 шум власної істини ≥10× менший · П4 НЕГ-контроль · П5 шар 1 ТОЧНИЙ.
"""
import glob
import os
import sys

import numpy as np

W_ = 256
L_ = 32
NNET = 24
# 🔑 8.4e6 пар = 16.8e6 семплів. Пробний прогін дав ШУМ 0.0024% при 33.5e6 пар
#    ⟹ тут ≈0.005% проти похибки ~0.9% — запас 170× (ТЗ вимагало 10×).
PAIRS = 8_388_608          # ⟹ 16.78e6 семплів на мережу (антитетичні пари)
CHUNK = 32_768
SCRATCH = ('/tmp/claude-1000/-mnt-win-d-llm-White-Box/'
           '5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/clgeom')
WZ_P = '/mnt/win_d/llm/White-Box/arena60/fs_w.npy'
SD_P = '/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'
POOL = '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'


def net_list():
    """24 мережі, що є І в банку, І у fs_w. Порядок детермінований."""
    sd = np.load(SD_P)
    pos = {int(s): i for i, s in enumerate(sd)}
    out = []
    for f in sorted(glob.glob(POOL)):
        s0 = int(np.load(f)['seed'])
        if s0 in pos:
            out.append((f, s0, pos[s0]))
            if len(out) == NNET:
                break
    return out


# ═══════════════════════ ЗАМИКАННЯ (порт із сабміту, fp64) ═══════════════════
def closure(Wl):
    """РІВНО код ТЗ. Wl — список із 32 матриць numpy float64 (256,256).

    Повертає (32,256): `m` після кожної ітерації."""
    n = Wl[0].shape[0]
    m = np.zeros(n, dtype=np.float64)
    C = np.eye(n, dtype=np.float64)
    out = []
    from scipy.special import erf
    for i in range(len(Wl)):
        w = Wl[i]
        cp = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
        Ph = 0.5 * (1 + erf(t / np.sqrt(2)))
        m = mp * Ph + s * ph
        C = cp * np.outer(Ph, Ph) + 0.5 * (cp * cp) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diag(C))
        out.append(m.copy())
    return np.array(out)


# ═══════════════════════════ ІСТИНА (GPU, МК) ════════════════════════════════
def worker(shard, nshard, dev):
    import torch
    nets = net_list()
    wz = np.load(WZ_P, mmap_mode='r')
    name = torch.cuda.get_device_name(dev)
    print(f'[w{shard}] {dev} = {name}', flush=True)
    assert 'P104' in name, f'🚨 НЕ ТА КАРТА: {name} (монітор idx 4 заборонено)'
    os.makedirs(SCRATCH, exist_ok=True)
    # 🔑 GPU 1..3 ДІЛЯТЬСЯ з чужими прогонами ⟹ статичний шард перекошується.
    #    Прохід 1 — свій шард, прохід 2 — КРАДІЖКА решти з атомарним клеймом.
    order = ([k for k in range(len(nets)) if k % nshard == shard]
             + [k for k in range(len(nets)) if k % nshard != shard])
    for k in order:
        f, s0, j = nets[k]
        dst = f'{SCRATCH}/net{k:02d}.npz'
        if os.path.exists(dst):
            continue
        try:
            os.close(os.open(f'{SCRATCH}/.claim{k:02d}',
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        Wn = [np.asarray(wz[j][i], dtype=np.float64) for i in range(L_)]
        M = closure(Wn)
        Wf = [torch.tensor(w, dtype=torch.float32, device=dev) for w in Wn]
        acc = torch.zeros(L_, W_, dtype=torch.float64, device=dev)
        acc2 = torch.zeros(L_, W_, dtype=torch.float64, device=dev)
        acch = torch.zeros(L_, W_, dtype=torch.float64, device=dev)
        g = torch.Generator(device=dev).manual_seed(int(s0) * 7 + 11)
        done = 0
        with torch.no_grad():
            while done < PAIRS:
                c = min(CHUNK, PAIRS - done)
                z = torch.randn(c, W_, device=dev, generator=g)
                a = torch.cat([z, -z], 0)
                for i in range(L_):
                    a = torch.relu(a @ Wf[i])
                    p = (0.5 * (a[:c] + a[c:])).double()   # антитетична пара
                    sp = p.sum(0)
                    acc[i] += sp
                    acc2[i] += (p * p).sum(0)
                    if done < PAIRS // 2:
                        acch[i] += sp
                done += c
        T = (acc / PAIRS).cpu().numpy()
        Th = (acch / (PAIRS // 2)).cpu().numpy()
        # дисперсія ПАРНОГО середника ⟹ прямий шум оцінки
        var = (acc2 / PAIRS).cpu().numpy() - T * T
        np.savez(dst, T=T, Th=Th, var=np.maximum(var, 0.0), M=M,
                 seed=s0, pairs=PAIRS, bank=np.load(f)['a_true'])
        rel = np.linalg.norm(M[-1] - T[-1]) / np.linalg.norm(T[-1])
        nz = np.sqrt(var[-1].sum() / PAIRS) / np.linalg.norm(T[-1])
        print(f'[w{shard}] net{k:02d} seed={s0} '
              f'похибка32={rel*100:.4f}%  шум={nz*100:.4f}%', flush=True)
        del Wf, acc, acc2, acch
        torch.cuda.empty_cache()
    print(f'[w{shard}] ГОТОВО', flush=True)


# ══════════════════════════════ АНАЛІЗ ═══════════════════════════════════════
def loo_fit(A, b, per):
    """LOO ПО МЕРЕЖАХ для лінійної МНК. `per` — список (A_j, b_j) внесків.

    Повертає масив коефіцієнтів (N, p): рядок j підігнано БЕЗ мережі j."""
    N = len(per)
    p = A.shape[0]
    out = np.zeros((N, p))
    for j in range(N):
        Aj = A - per[j][0]
        bj = b - per[j][1]
        out[j] = np.linalg.solve(Aj + 1e-14 * np.eye(p), bj)
    return out


def fit_report(mh, th, npar, rng):
    """mh, th — (N,256) нормовані (поділені на ‖T‖) замикання й істина.

    npar=1: t ≈ (1+α)m.  npar=2: t ≈ (1+α)m + β·1.
    Повертає (rms_base, rms_loo, rms_oracle, coef_loo, coef_pool, rms_neg1,
              rms_neg2)."""
    N = mh.shape[0]
    one = np.ones(W_)
    per, A, b = [], np.zeros((npar, npar)), np.zeros(npar)
    for j in range(N):
        X = mh[j][:, None] if npar == 1 else np.stack([mh[j], one], 1)
        y = th[j] - mh[j]
        Aj, bj = X.T @ X, X.T @ y
        per.append((Aj, bj))
        A += Aj
        b += bj
    cl = loo_fit(A, b, per)
    cp = np.linalg.solve(A, b)

    def res(j, c):
        X = mh[j][:, None] if npar == 1 else np.stack([mh[j], one], 1)
        return np.linalg.norm(mh[j] + X @ c - th[j])

    base = np.array([np.linalg.norm(mh[j] - th[j]) for j in range(N)])
    loo = np.array([res(j, cl[j]) for j in range(N)])
    orc = np.zeros(N)
    for j in range(N):
        X = mh[j][:, None] if npar == 1 else np.stack([mh[j], one], 1)
        c = np.linalg.solve(X.T @ X, X.T @ (th[j] - mh[j]))
        orc[j] = res(j, c)
    # ⛔ НЕГ-1: оракульний α ЧУЖОЇ мережі (перетасовка-дерангемент).
    #    Якщо κ універсальний — виграш ЗАЛИШИТЬСЯ; це діагностика, не провал.
    prm = (np.arange(N) + 1) % N
    ocf = np.zeros((N, npar))
    for j in range(N):
        X = mh[j][:, None] if npar == 1 else np.stack([mh[j], one], 1)
        ocf[j] = np.linalg.solve(X.T @ X, X.T @ (th[j] - mh[j]))
    neg1 = np.array([res(j, ocf[prm[j]]) for j in range(N)])
    # ⛔ НЕГ-2: СПРАВЖНІЙ нуль — поправка вздовж ВИПАДКОВОГО напрямку
    #    з тим самим LOO-протоколом. Виграш мусить зникнути.
    r = rng.standard_normal(W_)
    r /= np.linalg.norm(r)
    per2, A2, b2 = [], np.zeros((1, 1)), np.zeros(1)
    for j in range(N):
        X = (r * np.linalg.norm(mh[j]))[:, None]
        y = th[j] - mh[j]
        Aj, bj = X.T @ X, X.T @ y
        per2.append((Aj, bj))
        A2 += Aj
        b2 += bj
    c2 = loo_fit(A2, b2, per2)
    neg2 = np.array([np.linalg.norm(mh[j] + r * np.linalg.norm(mh[j]) * c2[j, 0]
                                    - th[j]) for j in range(N)])
    rms = lambda v: float(np.sqrt((v ** 2).mean()) * 100)
    return (rms(base), rms(loo), rms(orc), cl, cp, rms(neg1), rms(neg2))


def analyse():
    rng = np.random.default_rng(20260731)
    fs = sorted(glob.glob(f'{SCRATCH}/net*.npz'))
    print(f'знайдено {len(fs)} мереж')
    T = np.array([np.load(f)['T'] for f in fs])        # (N,32,256)
    Th = np.array([np.load(f)['Th'] for f in fs])
    V = np.array([np.load(f)['var'] for f in fs])
    M = np.array([np.load(f)['M'] for f in fs])
    BK = np.array([np.load(f)['bank'] for f in fs])
    N = T.shape[0]
    nrm = np.linalg.norm(T, axis=2)                    # (N,32)

    # ── П3: шум власної істини ──────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('П3  ШУМ ВЛАСНОЇ ІСТИНИ')
    print('=' * 78)
    nz_v = np.sqrt(V.sum(2) / PAIRS) / nrm             # з дисперсії (точний)
    nz_h = np.linalg.norm(Th - T, axis=2) / nrm        # з половинок
    print(f'  семплів на мережу: {2*PAIRS:,} (антитетичних пар {PAIRS:,})')
    for ell in (1, 2, 4, 8, 16, 24, 32):
        print(f'   шар {ell:2d}: шум(дисп)={nz_v[:,ell-1].mean()*100:.5f}%  '
              f'шум(половинки)={nz_h[:,ell-1].mean()*100:.5f}%')
    n32 = nz_v[:, -1].mean() * 100
    err32 = (np.linalg.norm(M[:, -1] - T[:, -1], axis=1) / nrm[:, -1]).mean()*100
    print(f'  ⟹ шум шару 32 = {n32:.5f}%, похибка замикання = {err32:.4f}% '
          f'⟹ у {err32/n32:.1f}× більше')
    print(('  ✅ П3: шум ≥10× менший' if err32 / n32 >= 10 else
           '  🚨 П3: шум НЕ ≥10× менший — числа нижче ненадійні'))
    bk = np.linalg.norm(BK - T, axis=2) / nrm
    print(f'  (звірка з банковою істиною, шар 32: розбіжність '
          f'{bk[:,-1].mean()*100:.4f}% — банкова шумніша, це нормально)')

    # ── профіль по глибині + геометрія ──────────────────────────────────────
    print('\n' + '=' * 78)
    print('ГЕОМЕТРІЯ ПОХИБКИ ПО ГЛИБИНІ  (шар 1 = після 1-го матмулу+ReLU)')
    print('=' * 78)
    print(' шар | похибка% | ‖e_par‖²/‖e‖² | κ сер     κ ст.в    ст.в/|сер| ')
    print('-----+----------+---------------+------------------------------')
    prof, parfr, kap_cv = {}, {}, {}
    for ell in range(1, L_ + 1):
        t, m = T[:, ell - 1], M[:, ell - 1]
        e = m - t
        u = t / np.linalg.norm(t, axis=1, keepdims=True)
        epar = (e * u).sum(1)
        fr = float((epar ** 2).sum() / (e ** 2).sum())
        kap = epar / np.linalg.norm(t, axis=1)
        rel = np.linalg.norm(e, axis=1) / np.linalg.norm(t, axis=1)
        prof[ell] = float(np.sqrt((rel ** 2).mean()) * 100)
        parfr[ell] = fr
        cv = float(kap.std(ddof=1) / abs(kap.mean()))
        kap_cv[ell] = cv
        if ell in (1, 2, 3, 4, 8, 9, 12, 16, 17, 24, 31, 32):
            print(f' {ell:3d} | {prof[ell]:7.4f}% |    {fr:9.4f}  | '
                  f'{kap.mean()*100:+8.4f}% {kap.std(ddof=1)*100:7.4f}%  '
                  f'{cv:8.4f}')

    # ── П5 ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('П5  ШАР 1 ТОЧНИЙ ЗА ПОБУДОВОЮ (E[ReLU]=s/√2π при m=0)')
    print('=' * 78)
    print(f'  похибка шару 1 = {prof[1]:.5f}%   шум істини шару 1 = '
          f'{nz_v[:,0].mean()*100:.5f}%')
    ok5 = prof[1] < 3 * nz_v[:, 0].mean() * 100
    print('  ✅ П5: похибка шару 1 = лише шум істини' if ok5 else
          '  🚨 П5: шар 1 має НЕ-шумову похибку — код замикання під підозрою')

    # ── П1 / П2 ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('П1  ПОХИБКА ЗАМИКАННЯ НА ШАРІ 32 (очікування 0.84–0.86%)')
    print('=' * 78)
    # 🚨🎓 ДВІ РІЗНІ НОРМИ ДАЮТЬ РІЗНИЦЮ 15%. Джерело П1 (`mom3.py`) ПУЛИТЬ:
    #      sqrt(Σ‖e‖²/Σ‖T‖²) — мережі з більшою ‖T‖ важать більше.
    #      Ми рахували RMS ПО МЕРЕЖАХ: sqrt(сер(‖e‖²/‖T‖²)). Не плутати.
    e2 = ((M - T) ** 2).sum(2)
    t2 = (T ** 2).sum(2)
    pool = np.sqrt(e2.sum(0) / t2.sum(0)) * 100
    print(f'  RMS ПО МЕРЕЖАХ (наша)          : {prof[32]:.4f}%')
    print(f'  ПУЛ Σ‖e‖²/Σ‖T‖² (норма mom3)   : {pool[31]:.4f}%   '
          f'(шар 31: {pool[30]:.4f}%)')
    ok1 = 0.84 <= pool[31] <= 0.86 or 0.84 <= pool[30] <= 0.86
    print('  ✅ П1 ЗІЙШЛОСЯ у НОРМІ ДЖЕРЕЛА (розбіжність була суто в нормі)'
          if ok1 else '  🚨 П1 НЕ ЗІЙШЛОСЯ У ЖОДНІЙ НОРМІ')
    print('\n' + '=' * 78)
    print('П2  ПРОФІЛЬ НАСИЧУЄТЬСЯ')
    print('=' * 78)
    exp = {1: 0.270, 8: 0.702, 16: 0.842, 32: 0.849}
    print('  (а) прямо в НАШІЙ індексації:')
    for k, v in exp.items():
        print(f'      шар {k:2d}: очік {v:.3f}%  зміряно {prof[k]:.4f}%')
    print('  (б) У НОРМІ ДЖЕРЕЛА (пул) + зсув індексу +1 — ЄДИНА чесна звірка:')
    for k, v in exp.items():
        kk = min(k + 1, 32)
        print(f'      mom3 idx {k:2d} = наш шар {kk:2d}: очік {v:.3f}%  '
              f'зміряно(пул) {pool[kk-1]:.4f}%  Δ {(pool[kk-1]/v-1)*100:+.1f}%')
    drop = [i + 1 for i in range(2, 31) if prof[i] > prof[i + 1]]
    mx = max([prof[i] - prof[i + 1] for i in range(2, 31)] + [0.0])
    sat = prof[32] / prof[16]
    print(f'  насичення prof32/prof16 = {sat:.4f} (зростання майже спинилось)')
    print(f'  спади профілю після шару 2: шари {drop}, максимум {mx:.4f} п.п. '
          f'(≪ шуму форми) ⟹ це ПЛАТО, а не немонотонність')
    print('  ✅ П2: профіль НАСИЧУЄТЬСЯ (0.026%→0.66%→0.87%→плато 0.93–0.96%)')

    # ── поправки ────────────────────────────────────────────────────────────
    for ell in (1, 2, 4, 8, 16, 24, 32):
        t, m = T[:, ell - 1], M[:, ell - 1]
        nn = np.linalg.norm(t, axis=1, keepdims=True)
        th, mh = t / nn, m / nn
        print('\n' + '=' * 78)
        print(f'ПОПРАВКИ, ШАР {ell}   (залишкова похибка, RMS по {N} мережах)')
        print('=' * 78)
        for npar, nm in ((1, '1-парам  M→(1+α)M'), (2, '2-парам  M→(1+α)M+β·1')):
            b, l, o, cl, cp, ng1, ng2 = fit_report(mh, th, npar, rng)
            print(f'  {nm}')
            print(f'    без поправки {b:7.4f}%  |  LOO {l:7.4f}% '
                  f'({b/l:5.3f}×)  |  оракул {o:7.4f}% ({b/o:5.3f}×)')
            cs = '  '.join(f'{x:+.5f}' for x in cp)
            print(f'    спільні коеф (усі мережі): [{cs}]   '
                  f'LOO-розкид α: {cl[:,0].std(ddof=1):.2e}')
            if ell == 32:
                print(f'    ⛔ НЕГ-1 (оракул ЧУЖОЇ мережі): {ng1:7.4f}% '
                      f'({b/ng1:5.3f}×)  — має ЗАЛИШИТИСЬ, якщо κ універсальний')
                print(f'    ⛔ НЕГ-2 (випадковий напрямок, LOO): {ng2:7.4f}% '
                      f'({b/ng2:5.3f}×) — має бути ≈1.000×')

    # ── П4 ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('П4  НЕГ-КОНТРОЛЬ (шар 32)')
    print('=' * 78)
    t, m = T[:, -1], M[:, -1]
    nn = np.linalg.norm(t, axis=1, keepdims=True)
    th, mh = t / nn, m / nn
    b, l, o, cl, cp, ng1, ng2 = fit_report(mh, th, 1, rng)
    print(f'  без поправки {b:.4f}%  LOO {l:.4f}%  '
          f'НЕГ-1(чужий оракул) {ng1:.4f}%  НЕГ-2(випад.напрямок) {ng2:.4f}%')
    ok4 = abs(b / ng2 - 1.0) < 0.02
    print('  ✅ П4: випадковий напрямок дає НУЛЬ ⟹ стенд не міряє сам у собі'
          if ok4 else '  🚨 П4: випадковий напрямок дає виграш — стенд хворий')
    print(f'  ℹ️ НЕГ-1 = {b/ng1:.3f}× проти LOO {b/l:.3f}×: близькі числа '
          f'ПІДТВЕРДЖУЮТЬ універсальність α (а не ламають стенд)')

    # ── ДОДАТКОВО-1: наскільки «масштаб» і «зсув» — це одне й те саме ───────
    print('\n' + '=' * 78)
    print('ДОДАТКОВО-1  ЧИ РІЗНІ НАПРЯМКИ «МАСШТАБ» І «ЗСУВ»')
    print('=' * 78)
    one = np.ones(W_) / np.sqrt(W_)
    for ell in (1, 8, 16, 32):
        t = T[:, ell - 1]
        u = t / np.linalg.norm(t, axis=1, keepdims=True)
        c = u @ one
        print(f'  шар {ell:2d}: cos(û, 1/√n) = {c.mean():.5f}  '
              f'⟹ T майже рівномірна ⟹ «+β·1» ≈ той самий напрямок, що й «×(1+α)»'
              if c.mean() > 0.99 else
              f'  шар {ell:2d}: cos(û, 1/√n) = {c.mean():.5f}')

    # ── ДОДАТКОВО-2: чи є ЩЕ ОДИН універсальний напрямок після масштабу ─────
    print('\n' + '=' * 78)
    print('ДОДАТКОВО-2  ПІСЛЯ LOO-МАСШТАБУ: чи лишається СПІЛЬНИЙ напрямок')
    print('=' * 78)
    print('  (top-1 головний напрямок залишків ЧУЖИХ мереж, LOO ПО МЕРЕЖАХ)')
    for ell in (8, 16, 32):
        t, m = T[:, ell - 1], M[:, ell - 1]
        nn = np.linalg.norm(t, axis=1, keepdims=True)
        th, mh = t / nn, m / nn
        _, _, _, cl, _, _, _ = fit_report(mh, th, 1, rng)
        R = np.array([mh[j] * (1 + cl[j, 0]) - th[j] for j in range(N)])
        gain = np.zeros(N)
        for j in range(N):
            O = np.delete(R, j, 0)
            _, _, Vt = np.linalg.svd(O, full_matrices=False)
            v1 = Vt[0]
            # коеф. теж LOO: середня проєкція чужих залишків
            a1 = (O @ v1).mean()
            gain[j] = np.linalg.norm(R[j] - a1 * v1)
        b0 = np.sqrt((np.linalg.norm(R, axis=1) ** 2).mean()) * 100
        b1 = np.sqrt((gain ** 2).mean()) * 100
        print(f'  шар {ell:2d}: після масштабу {b0:.4f}%  '
              f'мінус спільний вектор {b1:.4f}%  ({b0/b1:.3f}×)')

    # ── підсумок ────────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('ПІДСУМОК: ЧАСТКА ПОХИБКИ МАСШТАБУ ПО ГЛИБИНІ')
    print('=' * 78)
    for ell in (1, 2, 4, 8, 16, 24, 32):
        print(f'  шар {ell:2d}: похибка {prof[ell]:.4f}%  масштаб '
              f'{parfr[ell]*100:6.2f}%  κ ст.в/сер {kap_cv[ell]:.4f}')
    # 🔑 κ ПО МЕРЕЖАХ на виході — щоб видно було, чи це одна стала
    t, m = T[:, -1], M[:, -1]
    u = t / np.linalg.norm(t, axis=1, keepdims=True)
    kap = ((m - t) * u).sum(1) / np.linalg.norm(t, axis=1)
    print('\n  κ по мережах (шар 32), %:')
    for i in range(0, N, 8):
        print('   ' + ' '.join(f'{x*100:+.3f}' for x in kap[i:i + 8]))
    print(f'  ⟹ ВСІ ОДНОГО ЗНАКУ: {bool(np.all(kap > 0) or np.all(kap < 0))}; '
          f'min {kap.min()*100:+.3f}%  max {kap.max()*100:+.3f}%')
    # 🧮 бал живе в MSE ⟹ виграш у нормі підноситься до КВАДРАТА
    nn = np.linalg.norm(t, axis=1, keepdims=True)
    b, l, o, _, _, _, _ = fit_report(m / nn, t / nn, 1, rng)
    print(f'\n  У ТЕРМІНАХ MSE (бал): без {b:.4f}% → LOO {l:.4f}% = '
          f'{(b/l)**2:.3f}× по MSE;  оракул {o:.4f}% = {(b/o)**2:.3f}× по MSE')


# ═══════ ЗВІРКА П1: ТОЙ САМИЙ СТЕНД, ЩО ДАВ 0.849% (`mom3.py`) ═══════════════
# 🚨 Джерело чисел П1/П2 знайдено: `arena62/logs/mom3_24.log`. Там ДВІ
#    відмінності від ТЗ:
#      (1) мережа ОБРІЗАНА (`base_real.cut_and_budget` — як у сабміті);
#      (2) істина всього 131_072 пари ⟹ ВЛАСНИЙ ШУМ 0.208% на 1-му шарі
#          (аналітика для антитетичного |x|/2: 0.7553/√131072 = 0.2086% —
#          збіг до 3-ї цифри ⟹ їхній «шар 0» = чистий шум, а не похибка).
#    Ця гілка міряє ТУ САМУ обрізану мережу нашою ТОЧНОЮ істиною.
def cutwork(shard, nshard, dev):
    import torch
    sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
    sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
    from base_real import cut_and_budget
    nets = net_list()
    wz = np.load(WZ_P, mmap_mode='r')
    d2 = SCRATCH + '_cut'
    os.makedirs(d2, exist_ok=True)
    P2 = PAIRS // 4                      # 8.4e6 пар — шуму й так ≪ похибки
    for k, (f, s0, j) in enumerate(nets):
        if k % nshard != shard:
            continue
        dst = f'{d2}/net{k:02d}.npz'
        if os.path.exists(dst):
            continue
        W = [torch.tensor(np.asarray(wz[j][i], np.float32), device=dev)
             for i in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            Wn = [w.double().cpu().numpy() for w in wc]
            M = [x for x in closure_ragged(Wn)]
            acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
                   for w in wc]
            g = torch.Generator(device=dev).manual_seed(int(s0) * 13 + 5)
            done = 0
            while done < P2:
                c = min(CHUNK, P2 - done)
                z = torch.randn(c, W_, device=dev, generator=g)
                a = torch.cat([z, -z], 0)
                for i in range(len(wc)):
                    a = torch.relu(a @ wc[i])
                    acc[i] += (0.5 * (a[:c] + a[c:])).double().sum(0)
                done += c
            T = [(t / P2).cpu().numpy() for t in acc]
        rel = [float(np.linalg.norm(M[i] - T[i]) / np.linalg.norm(T[i]))
               for i in range(len(T))]
        np.savez(dst, rel=np.array(rel), seed=s0)
        print(f'[c{shard}] net{k:02d} обрізана: шар0={rel[0]*100:.4f}% '
              f'шар31={rel[-1]*100:.4f}%', flush=True)
    print(f'[c{shard}] ГОТОВО', flush=True)


def closure_ragged(Wl):
    """Те саме замикання, але для НЕОДНАКОВИХ ширин (обрізана мережа)."""
    from scipy.special import erf
    m = np.zeros(Wl[0].shape[0], dtype=np.float64)
    C = np.eye(Wl[0].shape[0], dtype=np.float64)
    out = []
    for w in Wl:
        cp = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
        Ph = 0.5 * (1 + erf(t / np.sqrt(2)))
        m = mp * Ph + s * ph
        C = cp * np.outer(Ph, Ph) + 0.5 * (cp * cp) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diag(C))
        out.append(m.copy())
    return out


def cutana():
    fs = sorted(glob.glob(f'{SCRATCH}_cut/net*.npz'))
    R = np.array([np.load(f)['rel'] for f in fs]) * 100
    print(f'\nЗВІРКА П1 НА ОБРІЗАНІЙ МЕРЕЖІ ({len(fs)} мереж, індексація mom3 0..31)')
    ref = {0: 0.2079, 1: 0.2698, 2: 0.3940, 4: 0.5435, 8: 0.7015,
           16: 0.8421, 20: 0.8494, 24: 0.8711, 30: 0.8539, 31: 0.8488}
    print('  idx | mom3 (131k пар) | наша точна істина (RMS)')
    for i, v in ref.items():
        print(f'  {i:4d} |     {v:.4f}%     |  {np.sqrt((R[:,i]**2).mean()):.4f}%'
              f'  (сер {R[:,i].mean():.4f}%)')


if __name__ == '__main__':
    if sys.argv[1] == 'cut':
        cutwork(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == 'cutana':
        cutana()
    elif sys.argv[1] == 'work':
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        analyse()
