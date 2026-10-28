"""ПЛОМБА ПРИЛАДУ: ПОРТ САБМІТА sub97 У numpy + РОЗКЛАД MSE НА V/N І b². (С88)

Мета — ВІДТВОРИТИ ВІДОМЕ ГРЕЙДЕРНЕ ЧИСЛО ДО БУДЬ-ЯКИХ ВИСНОВКІВ:
    sub97: бал 1.873e-07, множник 0.1 ⟹ MSE = 1.869e-06
    підгін по трьох точках грейдера дав V/N = 1.514e-06 (81%) + b² = 3.548e-07
Якщо порт не відтворює 1.869e-06 — усі висновки про «прибрати дисперсію»
стоять на піску, і це треба сказати ПРЯМО.

МЕТОД: на кожній мережі беремо K незалежних жеребів сабміта.
    Var  = дисперсія прогнозу по жеребах (усереднена по 256 виходах)
    b²   = ‖сер.прогноз − істина‖²/256, з КРОС-ЦІЛЛЮ (дві незалежні істини),
           щоб шум істини не додавався: b² = ⟨(p̄−T₁)·(p̄−T₂)⟩
    MSE  = Var + b²   (перевіряємо проти прямого ⟨(p−T₁)(p−T₂)⟩)

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 П1 (60%): MSE порту ляже в 1.6…2.2e-06 ⟹ прилад збігається з грейдером.
 П2 (50%): частка b² у MSE вийде БІЛЬШОЮ за заявлені 19%, бо `_W_BLEND`=0.0883
           виглядає завеликим для оптимуму (моя алгебра дає w*≈0.04).
"""
import math
import os
import sys
import time

import numpy as np

W_, L_ = 256, 32
WZ_P = '/mnt/win_d/llm/White-Box/arena60/fs_w.npy'
NNET = int(os.environ.get('NNET', '8'))
NDRAW = int(os.environ.get('NDRAW', '16'))
LOG2T = int(os.environ.get('LOG2T', '19'))       # пар на КОЖНУ з двох істин
BUDGET = 272_000_000_000
CHUNK = 8192

_TARGET_UTIL = 0.0875
_ALPHA = -0.00751
_W_BLEND = float(os.environ.get('WBLEND', '0.0883'))
_MIN_PAIRS = 128
_PILOT_PAIRS = 128
_GATE_N = 1024
_KSIG = 8
_ISQ2PI = 0.3989422804014327
_C3, _C4 = 1.0 / 3.0, 1.0 / 12.0
_LAM_S = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
          0.0088285, 0.0007922, 0.0015044, 0.0001062)
_LAM_A = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
          -0.0025822, -0.0030778, 0.0006822, 0.0068061)
_LAM_Q = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
          0.0189783, 0.0250559, 0.0340802, 0.0415433)
_LAM_G = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
          -0.0146129, -0.0100612, 0.0167223, 0.0168109)
_erf = np.frompyfunc(math.erf, 1, 1)
WGRID = (0.0, 0.02, 0.04, 0.0883, 0.15, 0.25)


def _phi(x):
    return np.exp(-0.5 * x * x) * _ISQ2PI


def _Phi(x):
    return 0.5 * (1.0 + _erf(x * 0.7071067811865476).astype(np.float64))


def _closure_parts(w_cut, ksig, kdeep):
    n = w_cut[0].shape[0]
    m = np.zeros(n); C = np.eye(n)
    out = []; mo = []; qo = []; po = []; ao = []; bo = []
    for i in range(kdeep):
        w = w_cut[i]
        cp = w.T @ (C @ w)
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        if 1 <= i <= ksig:
            out.append(s)
        mp = m @ w
        t = mp / s
        ph = _phi(t); Ph = _Phi(t)
        if 1 <= i <= ksig:
            po.append(Ph); s2ph = (s * s) * ph
            ao.append(s2ph); bo.append(s2ph * t)
        m = mp * Ph + s * ph
        if i == kdeep - 1 and i > ksig:
            break
        C = cp * np.outer(Ph, Ph) + (0.5 * (cp * cp)) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diag(C))
        if 1 <= i <= ksig:
            mo.append(m); qo.append(np.diag(C) + m * m)
    return out, mo, qo, po, m, ao, bo


def predict(Wl, seed, wblend=_W_BLEND):
    depth = len(Wl); width = Wl[0].shape[0]
    per_full = 2.0 * depth * width * width
    rng = np.random.default_rng(seed)
    zp = rng.standard_normal((_PILOT_PAIRS, width), dtype=np.float32)
    xp = np.concatenate([zp, -zp], 0)
    alive = []; a = xp
    for i in range(depth):
        h = a @ Wl[i]
        if i < depth - 1:
            alive.append(h.max(0) >= 0.0)
        a = np.maximum(h, 0.0)
    w_cut = []
    for i in range(depth):
        wi = Wl[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        w_cut.append(np.ascontiguousarray(wi))
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in w_cut)
    used = 2.0 * _PILOT_PAIRS * per_full
    left = _TARGET_UTIL * float(BUDGET) - used
    n_pairs = max(_MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0))))

    z = rng.standard_normal((n_pairs, width), dtype=np.float32)
    zd = z.astype(np.float64)
    cov = (zd.T @ zd) / float(n_pairs)
    ev, evec = np.linalg.eigh(cov)
    whiten = (evec * (1.0 / np.sqrt(np.maximum(ev, 1e-8)))) @ evec.T
    denom = float(2 * n_pairs)
    ksig = min(_KSIG, depth - 1)
    wc64 = [w.astype(np.float64) for w in w_cut]
    s_cl, m_cl, q_cl, P_cl, m_deep, a_cl, b_cl = _closure_parts(
        wc64, ksig, depth)

    w0 = (whiten @ wc64[0]).astype(np.float32)
    y = z @ w0
    x = np.concatenate([np.maximum(y, 0.0), np.maximum(-y, 0.0)], 0)
    sg0 = np.sqrt(np.maximum((wc64[0] * wc64[0]).sum(0), 1e-30))
    v = x.astype(np.float64).sum(0) / denom - sg0 * _ISQ2PI
    out_last = None
    for i in range(1, depth):
        p = x @ w_cut[i]
        gate = (p[:_GATE_N] > 0.0).sum(0).astype(np.float64) / _GATE_N
        v = (v @ wc64[i]) * gate
        if i <= ksig:
            pd = p.astype(np.float64)
            p2 = pd * pd
            mh = pd.sum(0) / denom
            e2 = p2.sum(0) / denom
            sh2 = np.maximum(e2 - mh * mh, 1e-30)
            sh = np.sqrt(sh2)
            e3 = (p2 * pd).sum(0) / denom
            e4 = (p2 * p2).sum(0) / denom
            mh2 = mh * mh
            k3 = e3 - 3.0 * mh * e2 + 2.0 * (mh2 * mh)
            k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                  - 3.0 * (mh2 * mh2) - 3.0 * (sh2 * sh2))
            dq = ((k3 / (sh2 * sh)) * (_C3 * a_cl[i - 1])
                  - (k4 / (sh2 * sh2)) * (_C4 * b_cl[i - 1]))
            x = np.maximum(p, 0.0)
            xd = x.astype(np.float64)
            xm = xd.sum(0) / denom
            xq = (xd * xd).sum(0) / denom
            v = (v + (_LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + _LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + _LAM_Q[i - 1] * ((xq - q_cl[i - 1]) - dq)
                 + _LAM_G[i - 1] * (gate - P_cl[i - 1]))
        else:
            x = np.maximum(p, 0.0)
        if i == depth - 1:
            out_last = x.astype(np.float64).sum(0) / denom
    ours = out_last - v
    return ours + wblend * ((1.0 + _ALPHA) * m_deep - ours), n_pairs, ours, m_deep


def truth(Wl, seed, npair):
    acc = np.zeros(W_); n = 0
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        c = min(CHUNK, npair - done)
        z = rng.standard_normal((c, W_), dtype=np.float32)
        a = np.concatenate([z, -z], 0)
        for i in range(L_):
            a = np.maximum(a @ Wl[i], 0.0)
        acc += a.astype(np.float64).sum(0); n += 2 * c; done += c
    return acc / n


def main():
    wz = np.load(WZ_P, mmap_mode='r')
    NT = 1 << LOG2T
    rec = []
    for net in range(NNET):
        t0 = time.time()
        Wl = [np.ascontiguousarray(np.asarray(wz[net][i], dtype=np.float32))
              for i in range(L_)]
        T1 = truth(Wl, 90000 + net, NT)
        T2 = truth(Wl, 190000 + net, NT)
        OU = np.zeros((NDRAW, W_))
        for d in range(NDRAW):
            _, npair, OU[d], MD = predict(Wl, 31337 + net * 97 + d)
        Y = (1.0 + _ALPHA) * MD
        row = []
        for w in WGRID:
            P = OU + w * (Y[None] - OU)
            mse = ((P - T1[None]) * (P - T2[None])).mean()
            var = P.var(0, ddof=1).mean()
            b2 = ((P.mean(0) - T1) * (P.mean(0) - T2)).mean() - var / NDRAW
            row += [mse, var, b2]
        ecl = Y - 0.5 * (T1 + T2)
        rel_cl = np.linalg.norm(ecl) / np.linalg.norm(T1)
        rec.append(row + [(T1 * T2).mean(), rel_cl, npair])
        i0 = 3 * list(WGRID).index(0.0883)
        print(f'[net{net}] N={2*npair} | w=0.0883: MSE={row[i0]:.4e} '
              f'Var={row[i0+1]:.4e} b²={row[i0+2]:.4e} | w=0: MSE={row[0]:.4e} '
              f'Var={row[1]:.4e} b²={row[2]:.4e} | замик={rel_cl*100:.3f}% '
              f'RMS²={(T1*T2).mean():.4f} [{time.time()-t0:.0f}s]', flush=True)
    a = np.array(rec)
    nw = len(WGRID)
    print(f'\n═══ {NNET} мереж × {NDRAW} жеребів, істина 2×{2*NT} семплів ═══')
    print(f'  RMS(істина)² = {a[:,3*nw].mean():.5f}  (пам\'ять 0.88404)')
    print(f'  похибка замикання після α = {100*a[:,3*nw+1].mean():.3f}%')
    print('   w      MSE        Var        b²        бал(×0.1)')
    best = None
    for k, w in enumerate(WGRID):
        mse = a[:, 3 * k].mean(); var = a[:, 3 * k + 1].mean()
        b2 = a[:, 3 * k + 2].mean()
        print(f'  {w:.4f}  {mse:.4e}  {var:.4e}  {b2:+.3e}  {0.1*mse:.4e}')
        if best is None or mse < best[1]:
            best = (w, mse)
    print(f'  ⟹ найкраще w={best[0]:.4f}, MSE={best[1]:.4e}, '
          f'бал={0.1*best[1]:.4e} проти sub97 1.873e-07')
    v0 = a[:, 1].mean(); b0 = a[:, 2].mean()
    print(f'  🔑 ПІДЛОГА без бленду (w=0): Var={v0:.4e} b²={b0:.3e} ⟹ якби '
          f'дисперсія зникла, бал={0.1*max(b0,0):.3e}')


if __name__ == '__main__':
    sys.exit(main())
