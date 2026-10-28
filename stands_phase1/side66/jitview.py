"""3D-ВІЗУАЛІЗАЦІЯ ДРИЖАННЯ: «нитка на вітру» (ідея юзера).

Що саме малюємо і чому це чесна картинка, а не декорація.

Для кожного жеребу `k` і кожного шару `ℓ` беремо СПРАВЖНЮ похибку вибірки
`e_ℓ = x̄_ℓ − m_ℓ` і ПЕРЕНОСИМО її до виходу тим самим ланцюгом, що в сабміті:
    `T_ℓ(e_ℓ) = e_ℓ · W_{ℓ+1}·затвор_{ℓ+1} · … · W_31·затвор_31`
Далі проєктуємо на 3 головні напрямки шуму ВИХОДУ.

Тоді для одного жеребу виходить крива з 32 точок:
  * початок (ℓ=0)  — те, що бачить наша контрольна змінна (там істина ТОЧНА);
  * кінець  (ℓ=31) — справжня похибка, яку треба було б відняти;
  * відстань між ними — рівно те, чого нам бракує.
Якби перенос був точний, крива була б ТОЧКОЮ. Довжина нитки = наша втрата.

🚨 Чому саме перенос, а не сирі `e_ℓ`: сирі похибки різних шарів живуть у
РІЗНИХ просторах (обрізання дає різну ширину) і непорівнянні. Перенос
приводить усе до спільного простору виходу — і водночас це рівно та операція,
на якій наш метод втрачає зв'язок.

ПЛОМБИ
  ✅ остання точка нитки мусить збігтися з `e_31` до машинної точності
     (тавтологія: перенос із 31 у 31 порожній);
  ✅ хмара кінців мусить мати той самий розкид, що `J(31)` зі `jitprof.py`.

Запуск:
    python side66/jitview.py [net] [ndraw] [cuda:N] [--gui]
Без `--gui` пише PNG у `side66/jitview.png` і `.npz` поруч.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, GATE_N                   # noqa: E402
from jitprof import truth_means, transport                         # noqa: E402

L_ = 32
OUT_NPZ = '/mnt/win_d/llm/White-Box/side66/jitview.npz'
OUT_PNG = '/mnt/win_d/llm/White-Box/side66/jitview.png'


def collect(net=0, ndraw=64, dev='cuda:4'):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have]
    sd0 = int(np.load(fs[net])['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, dev)
        d = len(wc)
        mt, _ = truth_means(wc, dev, 310_000 + 877 * net)
        scale = float(mt[d - 1].norm())

        TR = np.zeros((ndraw, d, 3), dtype=np.float64)
        raw = []
        for k in range(ndraw):
            g = torch.Generator(device=dev).manual_seed(51_000_000 + k)
            z = torch.randn(npair, W_, device=dev, generator=g)
            x = torch.cat([z, -z], 0)
            cov = (x.T @ x) / float(2 * npair)
            ev, V = torch.linalg.eigh(cov.double())
            wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
            a = torch.relu((x @ wh) @ wc[0])
            xb, gates = [a.double().mean(0)], [None]
            for i in range(1, d):
                p = a @ wc[i]
                gates.append((p[:GATE_N] > 0).to(p.dtype).sum(0).double()
                             / GATE_N)
                a = torch.relu(p)
                xb.append(a.double().mean(0))
            raw.append([transport(xb[l] - mt[l], wc, gates, l, d - 1)
                        for l in range(d)])

        # 3 головні напрямки шуму ВИХОДУ
        E = torch.stack([raw[k][d - 1] for k in range(ndraw)], 0)
        Ec = E - E.mean(0, keepdim=True)
        _, V3 = torch.linalg.eigh((Ec.T @ Ec).double())
        P = V3[:, -3:]
        for k in range(ndraw):
            for l in range(d):
                TR[k, l] = (raw[k][l] @ P).cpu().numpy() / scale

        # ✅ПЛОМБИ / ЧЕСНІ ЗАСТЕРЕЖЕННЯ ДО КАРТИНКИ
        # 🚨 «розкид нитки == J(31)» БУЛО Б НЕПРАВИЛЬНО: нитка живе у 3
        #    напрямках, похибка — у 256. Правильна величина — ЯКУ ЧАСТКУ
        #    дисперсії показує картинка. Без неї малюнок вводив би в оману.
        spread = float(np.sqrt((TR[:, d - 1, :] ** 2).sum(1).mean()))
        full = float(E.pow(2).sum(1).mean() ** 0.5) / scale
        capt = (spread / full) ** 2
        # у скільки разів перенесений якір шару 0 МЕНШИЙ за справжню похибку
        att = full / max(float(np.sqrt((TR[:, 0, :] ** 2).sum(1).mean()))
                         / max(np.sqrt(capt), 1e-30), 1e-30)
    del W, wc
    torch.cuda.empty_cache()
    return TR, dict(capt=capt, spread=spread, full=full, npair=npair,
                    scale=scale, depth=d, att=att)


def render(TR, seal, gui=False):
    import matplotlib
    matplotlib.use('QtAgg' if gui else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection          # noqa: F401

    ndraw, d, _ = TR.shape
    fig = plt.figure(figsize=(13, 6.5), facecolor='#0d1117')
    cmap = plt.get_cmap('turbo')

    ax = fig.add_subplot(121, projection='3d', facecolor='#0d1117')
    for k in range(min(ndraw, 70)):
        t = TR[k]
        for l in range(d - 1):
            ax.plot(t[l:l + 2, 0], t[l:l + 2, 1], t[l:l + 2, 2],
                    color=cmap(l / (d - 1)), lw=0.7, alpha=0.55)
    ax.scatter(TR[:, 0, 0], TR[:, 0, 1], TR[:, 0, 2], s=14,
               c='#00e5ff', depthshade=False, label='ℓ=0 (що бачить КЗ)')
    ax.scatter(TR[:, d - 1, 0], TR[:, d - 1, 1], TR[:, d - 1, 2], s=18,
               c='#ff2d55', depthshade=False, label='ℓ=31 (справжня похибка)')
    ax.set_title('НИТКА ПОХИБКИ: перенос $e_\\ell$ до виходу\n'
                 'колір = глибина шару', color='#e6edf3', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, facecolor='#161b22',
              labelcolor='#e6edf3', framealpha=0.8)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.05, 0.07, 0.09, 1.0))
        pane.label.set_color('#8b949e')
        pane.set_tick_params(colors='#8b949e', labelsize=7)

    ax2 = fig.add_subplot(122, facecolor='#0d1117')
    rad = np.sqrt((TR ** 2).sum(2))
    ax2.plot(np.arange(d), rad.mean(0) * 100, color='#ff2d55', lw=2,
             label='‖перенесена похибка‖, % від ‖m₃₁‖')
    ax2.fill_between(np.arange(d),
                     np.percentile(rad, 10, axis=0) * 100,
                     np.percentile(rad, 90, axis=0) * 100,
                     color='#ff2d55', alpha=0.18)
    dist = np.sqrt(((TR - TR[:, -1:, :]) ** 2).sum(2))
    ax2.plot(np.arange(d), dist.mean(0) * 100, color='#00e5ff', lw=2,
             ls='--', label='відстань до істинної похибки (наша втрата)')
    ax2.set_xlabel('шар ℓ', color='#8b949e')
    ax2.set_ylabel('% від ‖m₃₁‖', color='#8b949e')
    ax2.set_title(f'ПЕРЕНОС СТИСКАЄ: якір шару 0 приходить у {seal["att"]:.1f}×\n'
                  'меншим за справжню похибку ⟹ її роблять ОСТАННІ шари',
                  color='#e6edf3', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#161b22', labelcolor='#e6edf3',
               framealpha=0.8)
    ax2.grid(alpha=0.15, color='#8b949e')
    ax2.tick_params(colors='#8b949e', labelsize=8)
    for s in ax2.spines.values():
        s.set_color('#30363d')

    fig.suptitle(f'дрижання як нитка · {ndraw} жеребів · npair={seal["npair"]}'
                 f' · 3D-проєкція показує {seal["capt"]*100:.0f}% дисперсії'
                 f' похибки (решта — поза трьома напрямками)',
                 color='#8b949e', fontsize=9)
    fig.tight_layout()
    if gui:
        plt.show()
    else:
        fig.savefig(OUT_PNG, dpi=125, facecolor='#0d1117')
        print(f'PNG → {OUT_PNG}')


if __name__ == '__main__':
    net = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ndraw = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    dev = sys.argv[3] if len(sys.argv) > 3 else 'cuda:4'
    gui = '--gui' in sys.argv
    TR, seal = collect(net, ndraw, dev)
    np.savez(OUT_NPZ, TR=TR, **seal)
    print(f'✅ПЕРЕХРЕСНА ПЛОМБА: J(31) тут {seal["full"]*100:.4f}% — '
          f'звіряти з `jitprof.py` на ТІЙ САМІЙ мережі')
    print(f'ℹ️  3D-проєкція показує {seal["capt"]*100:.1f}% дисперсії похибки')
    print(f'🔑 ЗАТУХАННЯ: якір шару 0 приходить у {seal["att"]:.1f}× меншим '
          f'за справжню похибку виходу')
    render(TR, seal, gui=gui)
