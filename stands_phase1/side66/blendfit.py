"""ПІДГІН `w` ДЛЯ БЛЕНДУ sub91 + КАЛІБРУВАННЯ ЦІНИ. (С77)

Ідея: бленд ЛІНІЙНИЙ по `w`, а естиматор ДЕТЕРМІНОВАНИЙ (фіксований `_SEED`).
⟹ достатньо ДВОХ прогонів на мережу:
    pred(w) = (1−w)·pred(0) + w·pred(1),
де pred(0) — чиста наша оцінка (при зниженому util), pred(1) — саме
`(1+α)·M_last`. Далі будь-яке `w` рахується офлайн ТОЧНО.

Друкує: ціну (F/B, λR/B, множник) обох файлів, відносну похибку замикання
ДО і ПІСЛЯ поправки (П4), оптимальне `w` і свіп.

Запуск: python3 blendfit.py <nets> <off>
"""
import glob
import statistics as st
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)

import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
from bench import load_estimator                        # noqa: E402
import importlib.util                                   # noqa: E402

B = 272_000_000_000
LAM = 1e11


def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.split('/')[-1][:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(nets=24, off=0):
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    fs = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][off:off + nets]

    m87 = load_mod(SK + '/estimator_sub87.py')
    m91 = load_mod(SK + '/estimator_sub91.py')
    e87 = m87.Estimator()
    print(f'НАБІР off={off} nets={len(fs)}  '
          f'seeds: {[int(np.load(q)["seed"]) for q in fs]}', flush=True)
    print(f'sub91: _TARGET_UTIL={m91._TARGET_UTIL}  α={m91._ALPHA}  '
          f'w(у файлі)={m91._W_BLEND}\n', flush=True)

    P0, P1, TR, A87 = [], [], [], []
    c87, c91 = [], []
    for f in fs:
        zz = np.load(f)
        s0 = int(zz['seed'])
        truth = np.asarray(zz['a_true'])[-1].astype(np.float64)
        j = int(np.where(sd == s0)[0][0])
        Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
        mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)

        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            r = e87.predict(mlp, B)
        d = c.summary_dict()
        c87.append((c.flops_used / B, LAM * d['residual_wall_time_s'] / B,
                    len(c._op_log)))
        A87.append(np.asarray(r)[-1].astype(np.float64))

        m91._W_BLEND = 0.0
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            r0 = m91.Estimator().predict(mlp, B)
        d = c.summary_dict()
        c91.append((c.flops_used / B, LAM * d['residual_wall_time_s'] / B,
                    len(c._op_log)))
        m91._W_BLEND = 1.0
        with flops.BudgetContext(flop_budget=B, quiet=True):
            r1 = m91.Estimator().predict(mlp, B)
        m91._W_BLEND = 0.0

        P0.append(np.asarray(r0)[-1].astype(np.float64))
        P1.append(np.asarray(r1)[-1].astype(np.float64))
        TR.append(truth)
        print(f'  … {s0}', flush=True)

    P0, P1, TR, A87 = map(np.array, (P0, P1, TR, A87))
    # pred(1) = (1+α)·M ⟹ сире замикання M = pred(1)/(1+α)
    Mraw = P1 / (1.0 + m91._ALPHA)

    print('\n' + '=' * 78)
    print('ЦІНА (сирі складові, множник = max(0.1, F/B + λR/B))')
    print('=' * 78)
    for nm, cc in (('sub87', c87), ('sub91(глибоке зам.)', c91)):
        fb = st.median([q[0] for q in cc])
        rb = st.median([q[1] for q in cc])
        print(f'  {nm:22s} F/B={fb:.5f}  λR/B={rb:.5f}  сума={fb+rb:.5f}  '
              f'множник={max(0.1, fb+rb):.5f}  оп={cc[0][2]}')
    dfb = (st.median([q[0] for q in c91]) - st.median([q[0] for q in c87]))
    print(f'  ⟹ Δ(F/B) = {dfb:+.5f}   Δоп = {c91[0][2]-c87[0][2]:+d}')

    print('\n' + '=' * 78)
    print('П4  ВІДНОСНА ПОХИБКА ЗАМИКАННЯ (RMS ПО МЕРЕЖАХ, банкова істина)')
    print('=' * 78)
    nn = np.linalg.norm(TR, axis=1)
    rel_raw = np.linalg.norm(Mraw - TR, axis=1) / nn
    rel_cor = np.linalg.norm(P1 - TR, axis=1) / nn
    rel_our = np.linalg.norm(P0 - TR, axis=1) / nn
    r = lambda v: float(np.sqrt((v ** 2).mean()) * 100)
    print(f'  ДО поправки  : {r(rel_raw):.4f}%   (очікування 0.9–1.0%)')
    print(f'  ПІСЛЯ (1+α)  : {r(rel_cor):.4f}%   (очікування 0.58–0.62%)')
    print(f'  наша (sub91 w=0): {r(rel_our):.4f}%')
    print(f'  ⟹ MSE-виграш поправки {(r(rel_raw)/r(rel_cor))**2:.3f}×;  '
          f'відношення MSE замик/наша = '
          f'{(r(rel_cor)/r(rel_our))**2:.2f}×')

    print('\n' + '=' * 78)
    print('ПІДГІН w (MSE проти банкової істини, пул по мережах)')
    print('=' * 78)
    e0 = P0 - TR
    e1 = P1 - TR
    v0 = float((e0 * e0).mean())
    v1 = float((e1 * e1).mean())
    cv = float((e0 * e1).mean())
    print(f'  MSE(наша)={v0:.4e}  MSE(замик_corr)={v1:.4e}  '
          f'коваріація={cv:.4e}  кор={cv/np.sqrt(v0*v1):+.4f}')
    # мінімум по w: d/dw ‖(1-w)e0 + w e1‖² = 0 ⟹ w* = (v0-cv)/(v0+v1-2cv)
    wstar = (v0 - cv) / (v0 + v1 - 2 * cv)
    print(f'  🎯 теоретичне w* = {wstar:.4f}   '
          f'(наївне v0/(v0+v1) = {v0/(v0+v1):.4f})')
    print('\n   w      MSE       виграш проти w=0')
    best = (None, 1e99)
    for w in (0.0, 0.02, 0.05, 0.08, 0.12, 0.2, round(wstar, 4)):
        e = (1 - w) * e0 + w * e1
        mse = float((e * e).mean())
        if mse < best[1]:
            best = (w, mse)
        print(f'  {w:5.4f}  {mse:.4e}   {v0/mse:6.4f}×')
    print(f'  ⟹ найкраще w = {best[0]} ({v0/best[1]:.4f}× по MSE)')
    # скільки мереж покращились при w*
    for w in (0.05, round(wstar, 4)):
        e = (1 - w) * e0 + w * e1
        pm = (e * e).mean(1)
        p0 = (e0 * e0).mean(1)
        print(f'  w={w}: кращих {int((pm < p0).sum())}/{len(fs)} мереж')
    # звірка: sub87 vs sub91(w=0) — скільки коштує знижений util
    eA = A87 - TR
    print(f'\n  sub87 MSE={float((eA*eA).mean()):.4e}  vs sub91(w=0) '
          f'{v0:.4e}  ⟹ util коштує {v0/float((eA*eA).mean()):.4f}× по MSE')

    print('\n' + '=' * 78)
    print('ПО-МЕРЕЖНО: sub87 проти sub91 з `w` У ФАЙЛІ (множники однакові = 0.1)')
    print('=' * 78)
    wf = 0.0651
    eB = (1 - wf) * e0 + wf * e1
    mA = (eA * eA).mean(1)
    mB = (eB * eB).mean(1)
    print('   seed      MSE sub87     MSE sub91     виграш')
    for i, f in enumerate(fs):
        s0 = int(np.load(f)['seed'])
        fl = '✅' if mB[i] < mA[i] else '🚨'
        print(f'  {s0}  {mA[i]:.4e}  {mB[i]:.4e}  {mA[i]/mB[i]:7.4f}× {fl}')
    print(f'\n  П5: кращих {int((mB < mA).sum())}/{len(fs)}')
    print(f'  БАЛ як відношення СУМ: {mA.sum()/mB.sum():.4f}×   '
          f'(медіана відношень {float(np.median(mA/mB)):.4f}×)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24,
         off=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
