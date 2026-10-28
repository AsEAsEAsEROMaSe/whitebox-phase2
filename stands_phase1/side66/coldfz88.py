"""«ХОЛОДНЕ ЗАМОРОЖЕННЯ»: ЗНИЗИТИ ЦІНУ СЕМПЛА, А НЕ ДИСПЕРСІЮ. (С88)

МЕХАНІЗМ (виведений, не вгаданий). Бал на підлозі множника дорівнює
    score = 0.1·MSE ≈ 0.1·[ V·c/(0.0875·B) + b² ],
де `V` — дисперсія НА СЕМПЛ, `c` — ЦІНА семпла. Отже добуток `V·c` — інваріант:
у 2 рази дешевший семпл рівносильний удвічі меншій дисперсії. Дисперсійні
канали закриті (антитетика на глибині 32 дає лише ×1.09 — зміряно `mccal88`),
тому б'ю по `c`.

Ціна семпла = Σ_k 2·n_k·n_{k+1}. Пілот сабміта вирізає лише нейрони, що НІ РАЗУ
не спрацювали (зміряно: лишається 82.6%). Але замикання дає `t_i = m_i/s_i`
для КОЖНОГО нейрона, а внесок нейрона в дисперсію шару = `s²·V(t)`, і `V(t)`
падає надзвичайно швидко: V(3)=0.998, V(0)=0.341, V(−1)=0.069, V(−2)=0.0057.
⟹ «холодні» нейрони (`t<τ`) майже не флуктуюють: їх можна замінити на
   КОНСТАНТУ (їхнє замикальне середнє) + НЕЗАЛЕЖНИЙ шум із правильною
   дисперсією, і не рахувати їх на семплах узагалі.
   Внесок у наступний шар:  p ← W_hotᵀa_hot + (W_coldᵀμ_cold) + ξ,
   ξ_j ~ N(0, Σ_{i∈cold} W_ij²·Var(a_i)) — обидва доданки коштують ОДИН матвек
   на шар, а не матмул на кожен семпл.

ЩО ЛАМАЄТЬСЯ: втрачаються КОРЕЛЯЦІЇ холодних із гарячими. Саме там живе
похибка замикання (С87: позадіагональ 1.69% проти діагоналі 0.23%) ⟹ гілка
може померти саме тут. Це і міряємо.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 П1 (90%): при τ=−1 ціна впаде в 1.6…1.9×, при τ=0 — у 2.4…2.8×.
 П2 (45%): при τ=−1 повний бал покращає (виграш ціни переважить зсув).
 П3 (25%): при τ=0 бал теж покращає — сумніваюсь, бо викидається вже 5%
           дисперсії шару і всі парні кореляції холодних.
 П4 (70%): БЕЗ додавання шуму ξ бал буде помітно ГІРШИЙ, ніж із ним.
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
LOG2T = int(os.environ.get('LOG2T', '18'))
BUDGET = 272_000_000_000
CHUNK = 8192
_TARGET_UTIL = 0.0875
_ALPHA = -0.00751
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
TAUS = (-9.9, -1.5, -1.0, -0.5, 0.0, 0.5)


def _phi(x):
    return np.exp(-0.5 * x * x) * _ISQ2PI


def _Phi(x):
    return 0.5 * (1.0 + _erf(x * 0.7071067811865476).astype(np.float64))


def closure_full(w_cut):
    """Повний ланцюг замикання: на КОЖЕН шар вертає (t, μ_a, Var_a, s, Φ, ...)."""
    n = w_cut[0].shape[0]
    m = np.zeros(n); C = np.eye(n)
    T = []; MU = []; VA = []; S = []; PH = []; MO = []; QO = []
    AO = []; BO = []
    for i in range(len(w_cut)):
        w = w_cut[i]
        cp = w.T @ (C @ w)
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = _phi(t); Ph = _Phi(t)
        m = mp * Ph + s * ph
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        T.append(t); MU.append(m.copy()); VA.append(np.maximum(d, 0.0))
        S.append(s); PH.append(Ph)
        s2ph = (s * s) * ph
        AO.append(s2ph); BO.append(s2ph * t)
        if i < len(w_cut) - 1:
            C = (cp * np.outer(Ph, Ph)
                 + (0.5 * (cp * cp)) * np.outer(ph / s, ph / s))
            C = C + np.diag(d - np.diag(C))
        MO.append(m.copy()); QO.append(d + m * m)
    return T, MU, VA, S, PH, MO, QO, AO, BO


def predict(Wl, seed, tau, noise=True, orc=None):
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
    wc64 = [w.astype(np.float64) for w in w_cut]
    T, MU, VA, S, PH, MO, QO, AO, BO = closure_full(wc64)
    if orc is not None:                    # ОРАКУЛ: μ,Var холодних із МК-істини
        om, ov = orc
        MU = [om[i][alive[i]] if i < depth - 1 else om[i] for i in range(depth)]
        VA = [ov[i][alive[i]] if i < depth - 1 else ov[i] for i in range(depth)]

    # ── ХОЛОДНА МАСКА (шар i = після i-го матмулу; останній не морозимо) ──
    hot = [np.ones(w_cut[i].shape[1], dtype=bool) for i in range(depth)]
    for i in range(depth - 1):
        hot[i] = T[i] > tau
        if hot[i].sum() < 8:
            hot[i] = T[i] > (np.sort(T[i])[-8])
    # ціна семпла з урахуванням заморожування
    per_cut = 0.0
    nin = width
    for i in range(depth):
        nout = int(hot[i].sum()) if i < depth - 1 else w_cut[i].shape[1]
        per_cut += 2.0 * nin * nout
        if noise and i < depth - 1 and int((~hot[i]).sum()) > 0:
            per_cut += 18.0 * nout        # генерація+масштаб+додавання ξ
        nin = nout
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

    h0 = hot[0]
    w0 = (whiten @ wc64[0][:, h0]).astype(np.float32)
    y = z @ w0
    x = np.concatenate([np.maximum(y, 0.0), np.maximum(-y, 0.0)], 0)
    sg0 = np.sqrt(np.maximum((wc64[0][:, h0] ** 2).sum(0), 1e-30))
    v = x.astype(np.float64).sum(0) / denom - sg0 * _ISQ2PI
    out_last = None
    for i in range(1, depth):
        prev = hot[i - 1]
        Wi = wc64[i][prev, :]
        cold = ~prev
        last = (i == depth - 1)
        sel = hot[i] if not last else np.ones(w_cut[i].shape[1], dtype=bool)
        Wis = np.ascontiguousarray(Wi[:, sel])
        p = x @ Wis.astype(np.float32)
        pd = p.astype(np.float64)
        if cold.any():
            Wc = wc64[i][cold, :][:, sel]
            pd = pd + (MU[i - 1][cold] @ Wc)
            if noise:
                sg = np.sqrt(np.maximum((Wc * Wc).T @ VA[i - 1][cold], 0.0))
                pd = pd + rng.standard_normal(pd.shape) * sg
        gate = (pd[:_GATE_N] > 0.0).sum(0).astype(np.float64) / _GATE_N
        v = (v @ Wis) * gate
        if i <= ksig:
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
            dq = ((k3 / (sh2 * sh)) * (_C3 * AO[i][sel])
                  - (k4 / (sh2 * sh2)) * (_C4 * BO[i][sel]))
            xd = np.maximum(pd, 0.0)
            xm = xd.sum(0) / denom
            xq = (xd * xd).sum(0) / denom
            v = (v + (_LAM_S[i - 1] * _phi(mh / sh)) * (sh - S[i][sel])
                 + _LAM_A[i - 1] * (xm - MO[i][sel])
                 + _LAM_Q[i - 1] * ((xq - QO[i][sel]) - dq)
                 + _LAM_G[i - 1] * (gate - PH[i][sel]))
            x = xd.astype(np.float32)
        else:
            x = np.maximum(p if not cold.any() else pd.astype(np.float32), 0.0)
        if last:
            out_last = x.astype(np.float64).sum(0) / denom
    ours = out_last - v
    return ours, n_pairs, MU[-1], per_cut


def truth(Wl, seed, npair):
    acc = np.zeros(W_); n = 0
    rng = np.random.default_rng(seed); done = 0
    while done < npair:
        c = min(CHUNK, npair - done)
        z = rng.standard_normal((c, W_), dtype=np.float32)
        a = np.concatenate([z, -z], 0)
        for i in range(L_):
            a = np.maximum(a @ Wl[i], 0.0)
        acc += a.astype(np.float64).sum(0); n += 2 * c; done += c
    return acc / n


def layer_stats(Wl, seed, npair):
    s1 = np.zeros((L_, W_)); s2 = np.zeros((L_, W_)); n = 0
    rng = np.random.default_rng(seed); done = 0
    while done < npair:
        c = min(CHUNK, npair - done)
        z = rng.standard_normal((c, W_), dtype=np.float32)
        a = np.concatenate([z, -z], 0)
        for i in range(L_):
            a = np.maximum(a @ Wl[i], 0.0)
            ad = a.astype(np.float64)
            s1[i] += ad.sum(0); s2[i] += (ad * ad).sum(0)
        n += 2 * c; done += c
    m = s1 / n
    return m, np.maximum(s2 / n - m * m, 0.0)


ORACLE = os.environ.get('ORACLE', '0') == '1'


def main():
    wz = np.load(WZ_P, mmap_mode='r')
    NT = 1 << LOG2T
    noise = os.environ.get('NOISE', '1') == '1'
    res = {t: [] for t in TAUS}
    for net in range(NNET):
        t0 = time.time()
        Wl = [np.ascontiguousarray(np.asarray(wz[net][i], dtype=np.float32))
              for i in range(L_)]
        T1 = truth(Wl, 90000 + net, NT); T2 = truth(Wl, 190000 + net, NT)
        ORC = layer_stats(Wl, 555 + net, 1 << 16) if ORACLE else None
        line = []
        for tau in TAUS:
            OU = np.zeros((NDRAW, W_))
            for d in range(NDRAW):
                OU[d], npair, MD, pc = predict(Wl, 31337 + net * 97 + d, tau,
                                               noise, ORC)
            best = None
            for w in (0.0, 0.02, 0.04, 0.0883):
                P = OU + w * ((1 + _ALPHA) * MD[None] - OU)
                m = ((P - T1[None]) * (P - T2[None])).mean()
                if best is None or m < best[1]:
                    best = (w, m)
            mse0 = ((OU - T1[None]) * (OU - T2[None])).mean()
            var0 = OU.var(0, ddof=1).mean()
            res[tau].append([mse0, var0, best[1], best[0], 2 * npair, pc])
            line.append(f'τ={tau:+.1f} N={2*npair} MSE={mse0:.3e}')
        print(f'[net{net}] ' + ' | '.join(line) + f'  [{time.time()-t0:.0f}s]',
              flush=True)
    print(f'\n═══ {NNET} мереж × {NDRAW} жеребів, шум ξ={"так" if noise else "НІ"} ═══')
    base = np.array(res[TAUS[0]])
    print('  τ      ціна семпла    N      MSE(w=0)    ×база   MSE(кращий w)  ×база')
    for t in TAUS:
        a = np.array(res[t])
        print(f'  {t:+.1f}   {a[:,5].mean():.3e}  {a[:,4].mean():7.0f}  '
              f'{a[:,0].mean():.4e}  {base[:,0].mean()/a[:,0].mean():5.3f}   '
              f'{a[:,2].mean():.4e}   {base[:,2].mean()/a[:,2].mean():5.3f}   '
              f'(медіана ×{np.median(base[:,0]/a[:,0]):.3f})')


if __name__ == '__main__':
    sys.exit(main())
