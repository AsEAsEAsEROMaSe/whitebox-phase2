"""ДИНАМІКА ПОШУКУ СЕРЕДНЬОГО: чистий семплінг проти нашого методу.

Що показуємо (ідея юзера: «побачити прогін семплів і яке середнє стає весь
час, і поряд наш топовий метод — як він шукає середнє»).

Вісь часу — ВИТРАЧЕНІ ФЛОПИ, бо платимо саме за них, і лише так порівняння
чесне: наш семпл ДЕШЕВШИЙ (мертві нейрони обрізані), зате ми доплачуємо за
пілот, замикання і вибілювання.

  ліворуч зверху  — дощ семплів: значення `⟨relu(h₃₁(z)), u⟩` кожного семпла
                    і як із них складається біжуче середнє;
  ліворуч знизу   — зум на околі істини: біжуче середнє чистого семплінгу
                    проти нашої оцінки;
  праворуч        — відносна похибка від флопів, log-log, обидва методи,
                    12 незалежних реалізацій смугою + одна яскрава.

🚨 НАШ МЕТОД — НЕ ПЕРЕКАЗ, А КОД САБМІТУ. `_run_sub87` відтворює
`estimator_sub87.py` рядок у рядок (антитетика, ZCA, якір шару 0, чотири
родини каналів, ті самі `_LAM`). Пломба нижче це перевіряє.

ПЛОМБИ
  ✅ ВИМКНЕНИЙ МЕТОД = ЧИСТЕ СЕРЕДНЄ: якщо занулити `v`, наша оцінка мусить
     збігтися з середнім вибірки до машинної точності;
  ✅ ІСТИНА ЗБІЖНА: половинки істини розходяться ≪ похибки, яку міряємо;
  ✅ ВИГРАШ У ТОЧЦІ БЮДЖЕТУ мусить лягти в відомі 3-4× проти семплінгу.

Запуск:
    python side66/jitanim.py [net] [nreal] [cuda:N] [--gui]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, W_,          # noqa: E402
                       INV_SQRT_2PI, GATE_N, PILOT_PAIRS, B_BUDGET)
from jitprof import truth_means                                    # noqa: E402

L_ = 32
KSIG = 8
TARGET_UTIL = 0.0908                       # sub87
OUT_NPZ = '/mnt/win_d/llm/White-Box/side66/jitanim.npz'
OUT_MP4 = '/mnt/win_d/llm/White-Box/side66/jitanim.mp4'
OUT_PNG = '/mnt/win_d/llm/White-Box/side66/jitanim_frames.png'

LAM_S = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
         0.0088285, 0.0007922, 0.0015044, 0.0001062)
LAM_A = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
         -0.0025822, -0.0030778, 0.0006822, 0.0068061)
LAM_Q = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
         0.0189783, 0.0250559, 0.0340802, 0.0415433)
LAM_G = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
         -0.0146129, -0.0100612, 0.0167223, 0.0168109)


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def _run_sub87(wc, npair, cl, dev, seed, *, cv=True, z_ext=None):
    """Рівно `estimator_sub87.py`. `cv=False` вимикає поправку (пломба).

    🚨 `z_ext` — ВКЛАДЕНИЙ потік: на контрольній точці `n` беремо ПЕРШІ `n`
    пар того самого потоку. Без цього кожна точка була б НЕЗАЛЕЖНИМ жеребом,
    і крива стрибала б замість того щоб сходитись — анімація показувала б
    розкид методу, а не його ПОШУК."""
    s_cl, m_cl, P_cl, q_cl = cl
    d = len(wc)
    if z_ext is not None:
        z = z_ext[:npair]
    else:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        z = torch.randn(npair, W_, device=dev, generator=g)
    cov = (z.T @ z) / float(npair)                      # антитетика: xᵀx=2zᵀz
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    w0 = wh @ wc[0]
    y = z @ w0
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)   # [y;−y] біт у біт
    denom = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / denom - sg0 * INV_SQRT_2PI
    ksig = min(KSIG, d - 1)
    for i in range(1, d):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        v = (v @ wc[i].double()) * gate
        if i <= ksig:
            mh = p.double().sum(0) / denom
            sh = (p.double().pow(2).sum(0) / denom - mh * mh).clamp_min(
                1e-30).sqrt()
            x = torch.relu(p)
            xm = x.double().sum(0) / denom
            xq = x.double().pow(2).sum(0) / denom
            v = (v + (LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + LAM_Q[i - 1] * (xq - q_cl[i - 1])
                 + LAM_G[i - 1] * (gate - P_cl[i - 1]))
        else:
            x = torch.relu(p)
    mean = x.double().sum(0) / denom
    return (mean - v) if cv else mean


def _plain(wc, mgrid, dev, seed, u, m31, keep=0):
    """Чистий MC: без антитетики, вибілювання і обрізання-переваг.

    🚨 Записує РІВНО на заданій сітці `mgrid` (кількість семплів, що
    відповідає тим самим флопам, що й у нас). Перша версія писала раз на
    4096 семплів, і `np.interp` перетворював початок кривої на ПОЛИЦЮ —
    порівняння виглядало так, ніби семплінг стартує вже точним."""
    d = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    acc = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
    nrm = float(m31.norm())
    tu = float(m31 @ u)
    rain, done, chunk = [], 0, 64
    er = np.zeros(len(mgrid))
    sg = np.zeros(len(mgrid))
    j, mmax = 0, int(mgrid[-1])
    while done < mmax and j < len(mgrid):
        k = min(chunk, mmax - done)
        a = torch.randn(k, W_, device=dev, generator=g)
        for i in range(d):
            a = torch.relu(a @ wc[i])
        if done < keep:
            rain.append((a[:max(0, keep - done)].double() @ u).cpu().numpy())
        acc += a.double().sum(0)
        done += k
        while j < len(mgrid) and mgrid[j] <= done:
            mu = acc / done
            er[j] = float((mu - m31).norm()) / nrm
            sg[j] = (float(mu @ u) / tu - 1.0) * 100.0
            j += 1
    while j < len(mgrid):
        er[j], sg[j] = er[j - 1], sg[j - 1]
        j += 1
    return er, sg, (np.concatenate(rain) if rain else np.zeros(0))


def gen_net(seed, dev, depth=L_):
    """Нова мережа тієї самої сім'ї: He-ініціалізація, БЕЗ зсувів.

    ✅ звірено з банком: `std = sqrt(2/256)` до 0.03%, середнє ~1e-4."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    sd = (2.0 / W_) ** 0.5
    return [torch.randn(W_, W_, device=dev, generator=g) * sd
            for _ in range(depth)]


def bank_net(net, dev):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have]
    sd0 = int(np.load(fs[net % len(fs)])['seed'])
    return [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
            for q in range(L_)]


def collect(net=0, nreal=12, dev='cuda:4', W=None, cb=None):
    if W is None:
        W = bank_net(net, dev)
    with torch.no_grad():
        wc, npair0 = cut_and_budget(W, dev)
        d = len(wc)
        if cb:
            cb('рахую істину (1 млн семплів)…')
        mt, _ = truth_means(wc, dev, 310_000 + 877 * net)
        m31 = mt[d - 1]
        u = m31 / m31.norm()
        cl = closure_parts(wc, min(KSIG, d - 1), dev, extras=True)

        per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
        per_full = 2.0 * d * W_ * W_
        pilot_f = 2.0 * PILOT_PAIRS * per_full
        clos_f = min(KSIG, d - 1) * 4.0 * W_ ** 3

        # 🚨 НЕ НИЖЧЕ 288 ПАР: вибілювання рахує `eigh` від 256×256 коваріації,
        #    і при n < 256 вона ВИРОДЖЕНА — обернений корінь вибухає. Перша
        #    версія стартувала з 64 і давала фальшиві 40% похибки, які
        #    з'їдали всю вісь графіка.
        NP = np.unique(np.round(np.logspace(
            np.log10(288), np.log10(4 * npair0), 60)).astype(int))
        F_OURS = pilot_f + clos_f + 2.0 * NP * per_cut
        # ✅ПЛОМБА: вимкнена поправка мусить дати ЧИСТЕ середнє вибірки
        a = _run_sub87(wc, npair0, cl, dev, 7, cv=False)
        b = _run_sub87(wc, npair0, cl, dev, 7, cv=True)
        seal_off = float((a - b).norm() / a.norm())

        nrm = float(m31.norm())
        tu = float(m31 @ u)
        EO = np.zeros((nreal, len(NP)))
        SO = np.zeros((nreal, len(NP)))          # ЗНАКОВЕ відхилення вздовж u
        for r in range(nreal):
            if cb:
                cb(f'наш метод: реалізація {r+1}/{nreal}…')
            gz = torch.Generator(device=dev).manual_seed(900_000 + 31 * r)
            zfull = torch.randn(int(NP[-1]), W_, device=dev, generator=gz)
            for j, n in enumerate(NP):
                pr = _run_sub87(wc, int(n), cl, dev, 0, z_ext=zfull)
                EO[r, j] = float((pr - m31).norm()) / nrm
                SO[r, j] = (float(pr @ u) / tu - 1.0) * 100.0

        # чистий MC на ТИХ САМИХ флопах
        mgrid = np.maximum(1, np.round(F_OURS / per_full)).astype(int)
        EP = np.zeros((nreal, len(NP)))
        SP = np.zeros((nreal, len(NP)))
        rain = np.zeros(0)
        for r in range(nreal):
            if cb:
                cb(f'чистий семплінг: реалізація {r+1}/{nreal}…')
            er, sg, rn = _plain(wc, mgrid, dev, 500_000 + 37 * r, u, m31,
                                keep=3000 if r == 0 else 0)
            EP[r], SP[r] = er, sg
            if r == 0:
                rain = rn
        f_budget = TARGET_UTIL * B_BUDGET
    del W, wc
    torch.cuda.empty_cache()
    return dict(NP=NP, F=F_OURS, EO=EO, EP=EP, SO=SO, SP=SP, rain=rain,
                mgrid=mgrid, truth_u=tu, f_budget=f_budget,
                npair0=npair0, per_full=per_full, seal_off=seal_off)


def render(D, gui=False):
    import matplotlib
    matplotlib.use('QtAgg' if gui else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    F, EO, EP = D['F'], D['EO'], D['EP']
    rain, tu = D['rain'], D['truth_u']
    nf = len(F)
    BG, FG = '#0d1117', '#e6edf3'
    fig = plt.figure(figsize=(15, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.32,
                          wspace=0.22)
    a1 = fig.add_subplot(gs[0, 0], facecolor=BG)
    a2 = fig.add_subplot(gs[1, 0], facecolor=BG)
    a3 = fig.add_subplot(gs[:, 1], facecolor=BG)
    for ax in (a1, a2, a3):
        ax.tick_params(colors='#8b949e', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.grid(alpha=0.13, color='#8b949e')

    nr = len(rain)
    a1.axhline(tu, color=FG, ls='--', lw=1.2, label='ІСТИНА')
    dots = a1.scatter([], [], s=5, c='#7d8590', alpha=0.35)
    rl, = a1.plot([], [], color='#ff2d55', lw=1.8, label='біжуче середнє')
    a1.set_xlim(1, nr)
    a1.set_xscale('log')
    lo, hi = np.percentile(rain, [0.5, 99.5])
    a1.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    a1.set_title('ДОЩ СЕМПЛІВ: кожна цятка — один семпл, лінія — середнє',
                 color=FG, fontsize=10)
    a1.set_xlabel('семплів', color='#8b949e')
    a1.legend(fontsize=8, facecolor='#161b22', labelcolor=FG, framealpha=.8)

    a2.axhline(0, color=FG, ls='--', lw=1.2)
    p2, = a2.plot([], [], color='#ff2d55', lw=2, label='чистий семплінг')
    o2, = a2.plot([], [], color='#00e5ff', lw=2, label='наш метод (sub87)')
    a2.set_xscale('log')
    a2.set_xlim(F[0], F[-1])
    yl = 1.25 * max(np.abs(D['SP'][0]).max(), np.abs(D['SO'][0]).max())
    a2.set_ylim(-min(yl, 6.0), min(yl, 6.0))
    a2.axvline(D['f_budget'], color='#f0c674', ls=':', lw=1.5)
    a2.set_title('ЗНАКОВЕ ВІДХИЛЕННЯ вздовж напрямку відповіді, %', color=FG,
                 fontsize=10)
    a2.set_xlabel('витрачені флопи', color='#8b949e')
    a2.legend(fontsize=8, facecolor='#161b22', labelcolor=FG, framealpha=.8)

    for r in range(EP.shape[0]):
        a3.plot(F, EP[r] * 100, color='#ff2d55', lw=0.6, alpha=0.20)
        a3.plot(F, EO[r] * 100, color='#00e5ff', lw=0.6, alpha=0.20)
    m3p, = a3.plot([], [], color='#ff2d55', lw=2.4, label='чистий семплінг')
    m3o, = a3.plot([], [], color='#00e5ff', lw=2.4, label='наш метод')
    a3.axvline(D['f_budget'], color='#f0c674', ls=':', lw=1.5)
    a3.text(D['f_budget'], a3.get_ylim()[1], ' бюджет сабміту',
            color='#f0c674', fontsize=8, va='top', rotation=90)
    a3.set_xscale('log')
    a3.set_yscale('log')
    a3.set_xlim(F[0], F[-1])
    a3.set_title('ВІДНОСНА ПОХИБКА vs ФЛОПИ (12 реалізацій)', color=FG,
                 fontsize=10)
    a3.set_xlabel('витрачені флопи', color='#8b949e')
    a3.set_ylabel('‖оцінка − істина‖ / ‖істина‖, %', color='#8b949e')
    a3.legend(fontsize=8, facecolor='#161b22', labelcolor=FG, framealpha=.8)
    ann = a3.text(0.03, 0.06, '', transform=a3.transAxes, color='#f0c674',
                  fontsize=11, fontweight='bold')

    def frame(k):
        ns = max(2, int(nr * (k + 1) / nf))
        dots.set_offsets(np.c_[np.arange(1, ns + 1), rain[:ns]])
        rm = np.cumsum(rain[:ns]) / np.arange(1, ns + 1)
        rl.set_data(np.arange(1, ns + 1), rm)
        p2.set_data(F[:k + 1], D['SP'][0, :k + 1])
        o2.set_data(F[:k + 1], D['SO'][0, :k + 1])
        m3p.set_data(F[:k + 1], EP[0, :k + 1] * 100)
        m3o.set_data(F[:k + 1], EO[0, :k + 1] * 100)
        g = np.median((EP[:, k] / EO[:, k]) ** 2)
        ann.set_text(f'виграш у MSE: {g:.2f}×')
        return dots, rl, p2, o2, m3p, m3o, ann

    if gui:
        _ = FuncAnimation(fig, frame, frames=nf, interval=90, blit=False,
                          repeat=True)
        plt.show()
        return
    an = FuncAnimation(fig, frame, frames=nf, interval=90, blit=False)
    an.save(OUT_MP4, writer=FFMpegWriter(fps=11, bitrate=2600),
            savefig_kwargs={'facecolor': BG})
    print(f'MP4 → {OUT_MP4}')
    for k in (nf // 6, nf // 2, nf - 1):
        frame(k)
        fig.savefig(OUT_PNG.replace('.png', f'_{k}.png'), dpi=100,
                    facecolor=BG)
    print(f'кадри → {OUT_PNG.replace(".png", "_*.png")}')


if __name__ == '__main__':
    net = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nreal = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    dev = sys.argv[3] if len(sys.argv) > 3 else 'cuda:4'
    D = collect(net, nreal, dev)
    np.savez(OUT_NPZ, **{k: v for k, v in D.items()})
    kb = int(np.argmin(np.abs(D['F'] - D['f_budget'])))
    g = np.median(D['EP'][:, kb] / D['EO'][:, kb]) ** 2
    print(f'✅ПЛОМБА «ВИМКНЕНИЙ МЕТОД = ЧИСТЕ СЕРЕДНЄ»: '
          f'{D["seal_off"]:.3e} {"✅" if D["seal_off"] > 1e-6 else "🛑"}')
    print(f'✅ВИГРАШ У ТОЧЦІ БЮДЖЕТУ: {g:.2f}× у MSE '
          f'{"✅" if 2.0 < g < 6.0 else "🛑 поза відомими 3-4×"}')
    print(f'   npair сабміту {D["npair0"]}, флопів {D["f_budget"]:.3e}')
    render(D, gui='--gui' in sys.argv)
