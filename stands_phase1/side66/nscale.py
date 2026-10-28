#!/usr/bin/env python3
"""Як відносна похибка гаусового замикання (Мелер K=2) залежить від ШИРИНИ `n`.

Гіпотеза: err% ∝ c/n  (p = 1).

Стенд:
  * замикання — ПОРТ діючого сабміту (`base72.closure_parts`), numpy float64;
  * істина — Монте-Карло на GPU, матмули fp32, акумуляція суми у float64,
    антитетичні пари (z, −z);
  * метрика err% = 100·RMS(m_замик − істина)/RMS(істина) по компонентах виходу.

Пломби П1..П5 друкуються явно з вердиктом.
"""
import os
import sys
import glob
import time
import math

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

import torch  # noqa: E402

L_ = 32
PROF_LAYERS = (1, 2, 4, 8, 16, 24, 32)
WZ_PATH = '/mnt/win_d/llm/White-Box/arena60/fs_w.npy'
SD_PATH = '/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'
POOL = '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


# ─────────────────────────── ЗАМИКАННЯ (порт сабміту) ──────────────────────
def closure_m(W, layers=None):
    """Мелер K=2 з точною діагоналлю. Повертає dict {шар: m} (шари 1-based).

    Порт `_closure_parts` сабміту / `side66/base72.py:closure_parts`, fp64.
    """
    want = set(layers) if layers is not None else {len(W)}
    n0 = W[0].shape[0]
    m = np.zeros(n0, dtype=np.float64)
    C = np.eye(n0, dtype=np.float64)
    out = {}
    for i in range(len(W)):
        w = np.asarray(W[i], dtype=np.float64)
        cp = w.T @ (C @ w)
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = np.exp(-0.5 * t * t) / SQRT2PI
        Ph = 0.5 * (1.0 + torch.erf(torch.from_numpy(t / SQRT2)).numpy())
        m = mp * Ph + s * ph
        C = cp * np.outer(Ph, Ph) + (0.5 * cp * cp) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diag(C))
        if (i + 1) in want:
            out[i + 1] = m.copy()
    return out


# ──────────────────────────────── ІСТИНА (МК) ──────────────────────────────
def truth_mc(Wt, dev, seed, npair, layers=PROF_LAYERS, chunk=None):
    """E[a_l] на заданих шарах. Матмули fp32, сума fp64, антитетичні пари.

    Повертає (dict {шар: mean}, dict {шар: mean_half1}, dict {шар: mean_half2},
              dict {шар: E‖a‖²}).
    """
    n = Wt[0].shape[0]
    if chunk is None:
        chunk = max(1024, min(32768, int(2 ** 24 // max(n, 1))))
    want = sorted(set(layers))
    half = npair // 2
    acc = {l: torch.zeros(Wt[-1].shape[1] if l == len(Wt) else n,
                          dtype=torch.float64, device=dev) for l in want}
    accA = {l: torch.zeros_like(acc[l]) for l in want}
    sq = {l: torch.zeros((), dtype=torch.float64, device=dev) for l in want}
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    done = 0
    doneA = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.randn((k, n), generator=g, device=dev, dtype=torch.float32)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(Wt):
            x = torch.relu(x @ w)
            l = i + 1
            if l in acc:
                s = x.sum(0).double()
                acc[l] += s
                if done < half:
                    accA[l] += s
                sq[l] += (x.double() * x.double()).sum()
        if done < half:
            doneA += k
        done += k
    tot = 2.0 * done
    mean = {l: (acc[l] / tot).cpu().numpy() for l in want}
    nA = 2.0 * doneA
    h1 = {l: (accA[l] / nA).cpu().numpy() for l in want}
    h2 = {l: ((acc[l] - accA[l]) / (tot - nA)).cpu().numpy() for l in want}
    nrm = {l: float(sq[l]) / tot for l in want}
    return mean, h1, h2, nrm


def rel_err(pred, truth):
    return 100.0 * float(np.sqrt(np.mean((pred - truth) ** 2))
                         / np.sqrt(np.mean(truth ** 2)))


# ──────────────────────────────── ПРОГОНИ ──────────────────────────────────
def make_net(n, seed):
    rng = np.random.default_rng(seed)
    sd = math.sqrt(2.0 / n)
    return [rng.standard_normal((n, n)).astype(np.float64) * sd
            for _ in range(L_)]


def run_group(tag, nets, dev, npair, layers, log, every=1):
    """nets: список списків матриць. Повертає dict з err/noise/prof/norm.

    Мертві мережі (ReLU згасив усе, RMS(істина) ≈ 0) відбраковуються ЯВНО:
    відносна метрика там 0/0.
    """
    layers = tuple(sorted(set(layers)))
    lt = L_ if L_ in layers else max(layers)
    errs, noises, dead = [], [], 0
    prof = {l: [] for l in layers}
    pnz = {l: [] for l in layers}
    norm = {l: [] for l in layers}
    for j, W in enumerate(nets):
        t0 = time.time()
        cl = closure_m(W, layers)
        Wt = [torch.tensor(np.asarray(w, np.float32), device=dev) for w in W]
        with torch.no_grad():
            mean, h1, h2, nrm = truth_mc(Wt, dev, 5_000_003 + 7919 * j,
                                         npair, layers)
        del Wt
        n = W[0].shape[0]
        # мертва мережа: масштаб виходу впав більш ніж у 1e6 разів від шару 1
        r1 = float(np.sqrt(np.mean(mean[layers[0]] ** 2)))
        rL = float(np.sqrt(np.mean(mean[lt] ** 2)))
        if not np.isfinite(rL) or rL <= 1e-6 * max(r1, 1e-300):
            dead += 1
            log(f'    [{tag}] мережа {j + 1}/{len(nets)}: ⚠️ МЕРТВА '
                f'(RMS вих/RMS шар1 = {rL / max(r1, 1e-300):.2e}) — відкинута')
            continue
        for l in layers:
            prof[l].append(rel_err(cl[l], mean[l]))
            pnz[l].append(0.5 * rel_err(h1[l], h2[l]))
            norm[l].append(nrm[l] / n)
        errs.append(prof[lt][-1])
        noises.append(pnz[lt][-1])
        if (j % every) == 0 or j == len(nets) - 1:
            log(f'    [{tag}] мережа {j + 1}/{len(nets)}: err={errs[-1]:.4f}%  '
                f'шум={noises[-1]:.4f}%  ‖a_L‖²/n={norm[lt][-1]:.4f}  '
                f'({time.time() - t0:.1f}s)')
    return dict(err=np.array(errs), nz=np.array(noises), dead=dead,
                prof={l: np.array(v) for l, v in prof.items()},
                pnz={l: np.array(v) for l, v in pnz.items()},
                norm={l: np.array(v) for l, v in norm.items()})


def gmean(a):
    return float(np.exp(np.mean(np.log(np.asarray(a, float)))))


def bank_nets(k=24):
    WZ = np.load(WZ_PATH, mmap_mode='r')
    SD = np.load(SD_PATH)
    have = {int(v): i for i, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(POOL))
          if int(np.load(f)['seed']) in have][:k]
    out = []
    for f in fs:
        s0 = int(np.load(f)['seed'])
        W = [np.asarray(WZ[have[s0]][q], np.float64) for q in range(L_)]
        out.append((W, f))
    return out


def loglog_fit(ns, errs_per_net):
    """МНК log(err) ~ a + b·log(n). Повертає (p_all, se_all, p_mean, se_mean)."""
    X = np.concatenate([np.full(len(ee), math.log(n))
                        for n, ee in zip(ns, errs_per_net)])
    Y = np.concatenate([np.log(np.asarray(ee, float))
                        for ee in errs_per_net])
    A = np.vstack([np.ones_like(X), X]).T
    beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    r = Y - A @ beta
    cov = (float(r @ r) / (len(Y) - 2)) * np.linalg.inv(A.T @ A)

    # рівнева (кластерна) регресія: 1 точка на `n` ⇒ SE не занижена
    Xm = np.log(np.asarray(ns, float))
    Ym = np.array([math.log(gmean(ee)) for ee in errs_per_net])
    Am = np.vstack([np.ones_like(Xm), Xm]).T
    bm = np.linalg.lstsq(Am, Ym, rcond=None)[0]
    rm = Ym - Am @ bm
    if len(Ym) > 2:
        covm = (float(rm @ rm) / (len(Ym) - 2)) * np.linalg.inv(Am.T @ Am)
        sem = math.sqrt(covm[1, 1])
    else:
        sem = float('nan')
    return -beta[1], math.sqrt(cov[1, 1]), -bm[1], sem


TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def main():
    logp = '/mnt/win_d/llm/White-Box/arena62/logs/nscale.log'
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    fh = open(logp, 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        fh.write(s + '\n')
        fh.flush()

    gid = int(os.environ.get('GPU', 1))
    dev = torch.device(f'cuda:{gid}' if torch.cuda.is_available() else 'cpu')
    log(f'# стенд: {dev}  torch={torch.__version__}  '
        f'{torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}')
    log('# 📣 ПЕРЕДБАЧЕНО (гіпотеза p=1, якір 0.849% на n=256): '
        'n=128→1.70 · 64→3.40 · 32→6.79 · 16→13.6 · 8→27.2\n')

    NPAIR = int(os.environ.get('NPAIR', 2_000_000))
    NBANK = int(os.environ.get('NBANK', 24))
    # більше мереж там, де розкид по мережах великий (і рахувати дешево)
    NN = {8: 64, 16: 48, 32: 32, 64: 24, 128: 16, 256: 16}
    for k in NN:
        NN[k] = max(12, int(NN[k] * float(os.environ.get('NSCALE', 1.0))))

    # ── П1 + П2: РЕАЛЬНІ МЕРЕЖІ БАНКУ (n=256) ─────────────────────────────
    log('══ П1/П2: РЕАЛЬНІ МЕРЕЖІ БАНКУ, n=256, L=32 ══')
    bn = bank_nets(NBANK)
    log(f'  мереж знайдено: {len(bn)}')
    allw = np.concatenate([np.asarray(W).ravel() for W, _ in bn[:4]])
    wstd = float(allw.std())
    tgt = math.sqrt(2.0 / 256)
    ok2 = abs(wstd / tgt - 1.0) < 0.02
    log(f'  П2 std ваг банку = {wstd:.6f} · sqrt(2/256) = {tgt:.6f} · '
        f'відношення {wstd / tgt:.4f}  {"✅" if ok2 else "🚨"}')

    RB = run_group('банк', [W for W, _ in bn], dev, NPAIR, PROF_LAYERS, log)
    e_b = RB['err']
    E1 = float(e_b.mean())
    sem_b = float(e_b.std(ddof=1) / math.sqrt(len(e_b)))
    ok1 = 0.84 <= E1 <= 0.86 or abs(E1 - 0.849) <= 2 * sem_b
    bt = np.array([rel_err(closure_m(W, [L_])[L_],
                           np.load(f)['a_true'][-1].astype(np.float64))
                   for W, f in bn])
    log(f'\n  П1 ГОЛОВНА: банк n=256 → {E1:.4f}% ± {sem_b:.4f} (SEM, '
        f'{len(e_b)} мереж), очікувано 0.84–0.86%  {"✅" if ok1 else "🚨"}')
    log(f'     проти БАНКОВОЇ істини a_true[-1]: {bt.mean():.4f}% '
        f'(Δ стендів {abs(bt.mean() - E1):.4f} п.п.)')
    r3 = E1 / max(RB['nz'].mean(), 1e-12)
    log(f'  П3 шум власної істини (шар 32) = {RB["nz"].mean():.4f}% ⇒ '
        f'похибка/шум = {r3:.1f}×  {"✅" if r3 >= 10 else "🚨"}')
    log('  П5 профіль по глибині (банк, n=256):  [шар: похибка% (шум%)]')
    for l in PROF_LAYERS:
        log(f'     {l:2d}: {RB["prof"][l].mean():.4f}% ({RB["pnz"][l].mean():.4f}%)')
    sat = RB['prof'][32].mean() / max(RB['prof'][16].mean(), 1e-12)
    ok5 = sat < 1.15
    log(f'     насичення err(32)/err(16) = {sat:.3f}  {"✅" if ok5 else "🚨"}')
    if not ok1:
        log('\n🚨🚨 П1 НЕ ПРОЙШЛА — стенд/масштаб під сумнівом.')

    # ── РОЗГОРТКА ПО `n` ──────────────────────────────────────────────────
    NS = [8, 16, 32, 64, 128, 256]
    log('\n══ РОЗГОРТКА ПО ШИРИНІ (синтетичні He-мережі, iid гаус, без зсувів) ══')
    R = {}
    for n in NS:
        k = NN[n]
        log(f'  n={n} ({k} мереж):')
        nets = [make_net(n, 900_000 + 131 * n + j) for j in range(k)]
        lay = tuple(range(1, L_ + 1)) if n in (32, 256) else PROF_LAYERS
        ev = 1 if n >= 128 else max(1, k // 6)
        R[n] = run_group(f'n={n}', nets, dev, NPAIR, lay, log, every=ev)
        e = R[n]['err']
        log(f'    → n={n}: сер={e.mean():.4f}%  геом={gmean(e):.4f}%  '
            f'SEM={e.std(ddof=1) / math.sqrt(len(e)):.4f}  '
            f'шум={R[n]["nz"].mean():.4f}%  живих={len(e)}/{k} '
            f'(мертвих {R[n]["dead"]})')

    # ── ПЛОМБИ ────────────────────────────────────────────────────────────
    log('\n══ ПЛОМБИ ══')
    ok3 = True
    for n in NS:
        r = R[n]['err'].mean() / max(R[n]['nz'].mean(), 1e-12)
        ok3 &= r >= 10
        log(f'  П3 n={n:3d}: похибка {R[n]["err"].mean():8.4f}% / шум '
            f'{R[n]["nz"].mean():7.4f}% = {r:7.1f}×  {"✅" if r >= 10 else "🚨"}')
    log('  П4 E‖a_l‖²/n. Теорія: E[‖a_{l+1}‖²]=‖a_l‖² ТОЧНО (He: E[relu²]='
        '‖a‖²/n) ⇒ зберігається В СЕРЕДНЬОМУ ПО МЕРЕЖАХ. Але для СКІНЧЕННОЇ '
        '`n` це мультиплікативне блукання: log(‖a_l‖²/n) ~ N(−σ²l/2, σ²l), '
        'σ²≈c/n ⇒ вибіркове середнє при малих `n` НЕ ОЦІНЮЄТЬСЯ (домінують '
        'рідкісні викиди). Тому сіль: (a) шар 1 мусить дати рівно 1;')
    log('      (b) знос log-медіани мусить масштабуватись як 1/n ⇒ '
        'n·|E log(‖a₃₂‖²/n)| ≈ const.')
    ok4 = True
    for n in NS:
        nm = R[n]['norm']
        row = [f'{l}:{float(nm[l].mean()):.3f}' for l in PROF_LAYERS]
        med = [f'{l}:{float(np.median(nm[l])):.3f}' for l in PROF_LAYERS]
        lg = float(np.mean(np.log(np.maximum(nm[32], 1e-300))))
        inv = n * abs(lg) / L_          # мусить бути ~const по `n`
        g1 = abs(float(nm[1].mean()) - 1.0) < 0.05
        fin = bool(np.all(np.isfinite(nm[32])))
        ok4 &= (g1 and fin)
        log(f'     n={n:3d} сер  {"  ".join(row)}')
        log(f'          мед  {"  ".join(med)}  | n·|Δlog|/L={inv:.2f}  '
            f'шар1={float(nm[1].mean()):.4f} {"✅" if g1 and fin else "🚨"}')

    # ── ТАБЛИЦЯ ───────────────────────────────────────────────────────────
    log('\n══ ТАБЛИЦЯ ══')
    log(f'  {"n":>5} {"похибка%":>9} {"геом%":>9} {"SEM":>7} {"шум%":>8} '
        f'{"мереж":>6} {"прогноз p=1":>11} {"факт/прогноз":>12}')
    for n in NS:
        e = R[n]['err']
        pred = 0.849 * 256.0 / n
        log(f'  {n:5d} {e.mean():9.4f} {gmean(e):9.4f} '
            f'{e.std(ddof=1) / math.sqrt(len(e)):7.4f} {R[n]["nz"].mean():8.4f} '
            f'{len(e):6d} {pred:11.3f} {e.mean() / pred:12.3f}')
    log(f'  {"256б":>5} {e_b.mean():9.4f} {gmean(e_b):9.4f} {sem_b:7.4f} '
        f'{RB["nz"].mean():8.4f} {len(e_b):6d}   ← РЕАЛЬНІ мережі банку')

    # ── ПІДГІН ────────────────────────────────────────────────────────────
    log('\n══ ПІДГІН err% ∝ n^(−p), лог-лог МНК ══')
    for lo in (8, 16, 32, 64):
        sub = [n for n in NS if n >= lo]
        if len(sub) < 3:
            continue
        p, se, pm, sem = loglog_fit(sub, [R[n]['err'] for n in sub])
        t = TCRIT.get(len(sub) - 2, 2.0)
        log(f'  n ≥ {lo:3d} ({len(sub)} точок): '
            f'p(усі мережі) = {p:.4f} ± {se:.4f} · '
            f'p(рівневий) = {pm:.4f} ± {sem:.4f} '
            f'⇒ 95% ДІ [{pm - t * sem:.3f}, {pm + t * sem:.3f}]  '
            f'{"✅ p=1 всередині" if abs(pm - 1) <= t * sem else "🚨 p=1 ПОЗА ДІ"}')
    p, se, pm, sem = loglog_fit(NS, [R[n]['err'] for n in NS])
    log(f'\n  🎯 ГОЛОВНЕ (усі 6 `n`): p = {pm:.4f} ± {sem:.4f} '
        f'(95% ДІ [{pm - 2.776 * sem:.3f}, {pm + 2.776 * sem:.3f}]), '
        f'відхил від 1 = {(pm - 1.0) / max(sem, 1e-12):+.2f}σ')

    # ── ПРОФІЛІ ───────────────────────────────────────────────────────────
    log('\n══ ПРОФІЛЬ ПО ГЛИБИНІ (похибка%, у дужках шум істини%) ══')
    log(f'  {"шар":>4} {"n=256 банк":>18} {"n=256 синт":>18} {"n=32 синт":>18} '
        f'{"32/256":>7}')
    for l in PROF_LAYERS:
        a = R[256]['prof'][l].mean()
        b = RB['prof'][l].mean()
        c = R[32]['prof'][l].mean()
        log(f'  {l:4d} {b:11.4f} ({RB["pnz"][l].mean():.4f}) '
            f'{a:11.4f} ({R[256]["pnz"][l].mean():.4f}) '
            f'{c:11.4f} ({R[32]["pnz"][l].mean():.4f}) {c / max(a, 1e-12):7.2f}')

    log('\n══ ПІДСУМОК ПЛОМБ ══')
    for nm, v in (('П1 банк 0.84–0.86%', ok1), ('П2 std ваг', ok2),
                  ('П3 шум ≥10× менший', ok3),
                  ('П4 норма (сер. по мережах)', ok4),
                  ('П5 насичення профілю', ok5)):
        log(f'  {nm:32s} {"✅" if v else "🚨"}')
    fh.close()


if __name__ == '__main__':
    main()
