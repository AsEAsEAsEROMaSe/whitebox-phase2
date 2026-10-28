"""ІНТЕРАКТИВНИЙ СТЕНД: криві діючого топу `sub87` проти давнього `v2`.

Той самий стиль, що `jitgui.py`, але питання інше. `jitgui` показує, ЯК шукається
середнє. Тут показано, ЧИМ платиться точність — три панелі, які разом і є вся
арифметика змагання:

  ⬅  ВІДНОСНА ПОХИБКА vs ФЛОПИ — товсте = медіана, тонке = окремі жереби;
  ↗  БАЛ = MSE × max(0.1, C/B) — з вертикаллю на підлозі 0.1. Мінімум обох
      кривих сидить РІВНО на зламі, і видно, чому зайвий бюджет купувати марно;
  ↘  MSE × C/B — інваріант. Поки діє закон 1/C, це ПРЯМА ГОРИЗОНТАЛЬ.
      Будь-який підйом = ми впираємось у щось, що семплами не б'ється.

🚨 КРОС-ЦІЛЬ ОБОВ'ЯЗКОВА. Перший захід на цей свіп дав фальшиве «плато зміщення»
на рівні 6.9e-7 в ОБОХ естиматорів. Виявилось — це власний шум банкової `a_true`
(≈3e-7 по MSE), а метадані `avar_true/(2·pairs)` занижують його у ~7×. Тут MSE
рахується як `E[(p−T1)(p−T2)]` по ДВОХ незалежних істинах ⟹ шум істини входить
у добуток із нульовим середнім і плато не вигадується.

Запуск:  ./torch_venv/bin/python side66/curvegui.py [cuda:N]
"""
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('QtAgg')

from PyQt5 import QtCore, QtWidgets                               # noqa: E402
from matplotlib.backends.backend_qtagg import (                   # noqa: E402
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure                              # noqa: E402

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
import jitanim                                                    # noqa: E402
from base_real import cut_and_budget, closure_parts, W_           # noqa: E402

BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
RED, CYAN, GOLD = '#ff2d55', '#00e5ff', '#f0c674'
B_ = 272_000_000_000
# 🚨 residual (пітонівський час) GPU-порт відтворити не може. Беремо ЗМІРЯНІ
# на flopscope добавки: sub87 C/B=0.0980 при аналітичних 0.0908 ⟹ +0.0072;
# v2 0.0965 при 0.092 ⟹ +0.0045. Позначено на панелі, щоб не видавати за точне.
RES = {'sub87': 0.0072, 'v2': 0.0045}
GRID = 14


def _v2(wc_full, npair, dev, z_ext):
    """`estimator_whiten_v2`: вибілена антитетика, БЕЗ обрізання й поправок."""
    z = z_ext[:npair]
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc_full[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    for i in range(1, len(wc_full)):
        x = torch.relu(x @ wc_full[i])
    return x.double().sum(0) / float(2 * npair)


def _truth(W, dev, seed, pairs=1 << 18, ch=1 << 15):
    acc = torch.zeros(W[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    done = 0
    while done < pairs:
        k = min(ch, pairs - done)
        g = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        a = torch.cat([g, -g], 0)
        for m in W:
            a = torch.relu(a @ m)
        acc += a.double().sum(0)
        done += k
    return acc / (2.0 * pairs)


def collect(W, dev, nreal=6, cb=None):
    """Свіп по npair із ВКЛАДЕНИМ потоком (кожна точка — префікс того самого
    жеребу), тож криві СХОДЯТЬСЯ, а не стрибають."""
    with torch.no_grad():
        wc, npair0 = cut_and_budget(W, dev)
        # ✅ closure_parts(extras=True) вертає РІВНО (s, m, P, q) — саме той
        # порядок, який чекає `_run_sub87`. Перестановки НЕ треба.
        cl = closure_parts(wc, 8, dev, extras=True)
        if cb:
            cb('дві незалежні істини…')
        T1 = _truth(W, dev, 4_100_001)
        T2 = _truth(W, dev, 8_200_003)
        f_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
        f_full = 2.0 * len(W) * W_ * W_
        grid = np.unique(np.round(np.geomspace(npair0 / 8, npair0 * 16, GRID))
                         ).astype(int)
        out = {'grid': grid, 'npair0': npair0,
               'cb': {'sub87': grid * 2 * f_cut / B_ + RES['sub87'],
                      'v2': grid * 2 * f_full / B_ + RES['v2']},
               'sub87': [], 'v2': []}
        for r in range(nreal):
            if cb:
                cb(f'жереб {r+1}/{nreal}…')
            g = torch.Generator(device=dev).manual_seed(90_000 + 7717 * r)
            zbig = torch.randn(int(grid[-1]), W_, device=dev, generator=g)
            e87, e2 = [], []
            for n in grid:
                p = jitanim._run_sub87(wc, int(n), cl, dev, 0, z_ext=zbig)
                e87.append(float(((p - T1) * (p - T2)).mean()))
                q = _v2(W, int(n), dev, zbig)
                e2.append(float(((q - T1) * (q - T2)).mean()))
            out['sub87'].append(e87)
            out['v2'].append(e2)
            del zbig
            torch.cuda.empty_cache()
        out['sub87'] = np.array(out['sub87'])
        out['v2'] = np.array(out['v2'])
        out['norm'] = float((((T1 + T2) / 2) ** 2).mean())
    return out


class Win(QtWidgets.QMainWindow):
    def __init__(self, dev='cuda:1'):
        super().__init__()
        self.dev = dev
        self.D = None
        self.setWindowTitle('Криві: діючий топ sub87 проти v2')
        self.resize(1600, 900)
        self.fig = Figure(figsize=(15.5, 7.4), facecolor=BG)
        self.canvas = FigureCanvas(self.fig)

        self.b_new = QtWidgets.QPushButton('🎲 НОВА МЕРЕЖА')
        self.b_bank = QtWidgets.QPushButton('з банку №')
        self.sp = QtWidgets.QSpinBox(); self.sp.setRange(0, 99)
        self.nreal = QtWidgets.QSpinBox()
        self.nreal.setRange(1, 16); self.nreal.setValue(6)
        self.nreal.setPrefix('жеребів: ')
        self.status = QtWidgets.QLabel('натисни «з банку №» або «НОВА МЕРЕЖА»')
        self.status.setStyleSheet(f'color:{GOLD};font-size:13px')

        bar = QtWidgets.QHBoxLayout()
        for w in (self.b_new, self.b_bank, self.sp, self.nreal):
            bar.addWidget(w)
        bar.addStretch(1)
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.canvas, 1)
        box.addLayout(bar)
        box.addWidget(self.status)
        c = QtWidgets.QWidget(); c.setLayout(box)
        c.setStyleSheet(f'background:{BG};color:{FG}')
        self.setCentralWidget(c)
        self.b_new.clicked.connect(lambda: self._go(None))
        self.b_bank.clicked.connect(lambda: self._go(self.sp.value()))

    def _msg(self, t):
        self.status.setText(t)
        QtWidgets.QApplication.processEvents()

    def _go(self, net):
        self._msg('рахую…')
        W = (jitanim.bank_net(net, self.dev) if net is not None
             else jitanim.gen_net(np.random.randint(1 << 30), self.dev))
        self.D = collect(W, self.dev, self.nreal.value(), cb=self._msg)
        self._draw(net)

    def _sty(self, ax, t, xl, yl):
        ax.set_facecolor(BG)
        ax.set_title(t, color=FG, fontsize=11)
        ax.set_xlabel(xl, color=MUT, fontsize=9)
        ax.set_ylabel(yl, color=MUT, fontsize=9)
        ax.tick_params(colors=MUT, labelsize=8)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.grid(alpha=.12, color=FG)

    def _draw(self, net):
        D = self.D
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.15],
                                   hspace=.34, wspace=.22)
        a1 = self.fig.add_subplot(gs[:, 0])
        a2 = self.fig.add_subplot(gs[0, 1])
        a3 = self.fig.add_subplot(gs[1, 1])
        nrm = D['norm']
        for nm, col in (('v2', RED), ('sub87', CYAN)):
            m = np.maximum(D[nm], 1e-18)
            # 🚨 ФЛОПИ У КОЖНОГО СВОЇ: sub87 жене ОБРІЗАНІ матриці, тож на тому
            # самому npair коштує менше. Перша версія малювала обидві криві
            # проти ОДНАКОВОЇ (повної) сітки флопів — тобто порівнювала при
            # рівних СЕМПЛАХ, і тим ЗАНИЖУВАЛА sub87, зсуваючи перетин ліворуч.
            fl = (D['cb'][nm] - RES[nm]) * B_
            for row in m:
                a1.plot(fl, 100 * np.sqrt(row / nrm), color=col, lw=.7, alpha=.28)
            a1.plot(fl, 100 * np.sqrt(np.median(m, 0) / nrm), color=col, lw=2.6,
                    label=('v2 (3.09e-7)' if nm == 'v2' else 'sub87 (1.954e-7)'))
            cb = D['cb'][nm]
            med = np.median(m, 0)
            a2.plot(cb, med * np.maximum(0.1, cb), color=col, lw=2.4)
            a2.plot(cb, med * np.maximum(0.1, cb), 'o', color=col, ms=3)
            a3.plot(cb, med * cb, color=col, lw=2.4)
        a1.set_xscale('log'); a1.set_yscale('log')
        self._sty(a1, 'ВІДНОСНА ПОХИБКА vs ФЛОПИ  (товсте = медіана)',
                  'флопи семплінгу', '‖оцінка − істина‖ / ‖істина‖, %')
        a1.legend(facecolor=BG, edgecolor='#30363d', labelcolor=FG, fontsize=9)
        a2.set_xscale('log'); a2.set_yscale('log')
        a2.axvline(0.1, color=GOLD, ls=':', lw=1.4)
        a2.text(0.101, a2.get_ylim()[1] * .55, ' підлога множника 0.1',
                color=GOLD, fontsize=8)
        self._sty(a2, 'БАЛ = MSE × max(0.1, C/B)   ← мінімум РІВНО на зламі',
                  'C/B (ціна)', 'бал')
        a3.set_xscale('log'); a3.set_yscale('log')
        a3.axvline(0.1, color=GOLD, ls=':', lw=1.4)
        self._sty(a3, 'ІНВАРІАНТ MSE × C/B  — горизонталь = закон 1/C тримається',
                  'C/B (ціна)', 'MSE × C/B')
        i0 = int(np.argmin(np.abs(D['cb']['sub87'] - 0.1)))
        g = (np.median(D['v2'], 0)[i0] / np.median(D['sub87'], 0)[i0])
        a1.text(.03, .06, f'виграш sub87 над v2 на підлозі: {g:.2f}×',
                transform=a1.transAxes, color=GOLD, fontsize=13, weight='bold')
        self.fig.tight_layout()
        self.canvas.draw()
        self._msg(f'мережа {"банк №"+str(net) if net is not None else "нова"} · '
                  f'npair робочий {D["npair0"]} · жеребів {D["sub87"].shape[0]} · '
                  f'крос-ціль (2 незалежні істини) · residual — зміряна добавка')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Win(sys.argv[1] if len(sys.argv) > 1 else 'cuda:1')
    w.show()
    sys.exit(app.exec_())
