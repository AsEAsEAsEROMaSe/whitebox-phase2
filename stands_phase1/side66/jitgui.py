"""ІНТЕРАКТИВНИЙ СТЕНД: як чистий семплінг і наш метод шукають середнє.

PyQt5 + вбудований matplotlib. Кнопкою генеруються НОВІ мережі тієї самої
сім'ї (He, без зсувів — звірено з банком до 0.03%), тож картинку можна
дивитись не на одному прикладі, а на скільки завгодно.

Панелі:
  ⬆ дощ семплів + біжуче середнє + ІСТИНА; вертикальні риски — ПЕРЕТИНИ
    середнього з істиною (те, що помітив юзер: у якийсь момент воно
    перетинає і далі коливається навколо);
  ⬇ знакове відхилення обох методів від істини, вісь — витрачені флопи;
  ➡ відносна похибка vs флопи, кілька реалізацій.

🚨 ПЕРЕТИН ≠ СИГНАЛ. У точці перетину оцінка ТОЧНА, але дізнатись про це
зсередини не можна: `x̄ − m` не спостережуване без `m`. Стенд для того й
зроблений, щоб це було видно очима, а не на віру.

Запуск:
    python side66/jitgui.py [cuda:N]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('QtAgg')

from PyQt5 import QtCore, QtWidgets                               # noqa: E402
from matplotlib.backends.backend_qtagg import (                   # noqa: E402
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure                              # noqa: E402

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import jitanim                                                    # noqa: E402

BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
RED, CYAN, GOLD = '#ff2d55', '#00e5ff', '#f0c674'


class Win(QtWidgets.QMainWindow):
    def __init__(self, dev='cuda:4'):
        super().__init__()
        self.dev = dev
        self.D = None
        self.k = 0
        self.setWindowTitle('Пошук середнього: семплінг проти sub87')
        self.resize(1550, 900)

        self.fig = Figure(figsize=(15, 7.4), facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self._axes()

        self.b_new = QtWidgets.QPushButton('🎲 НОВА МЕРЕЖА')
        self.b_bank = QtWidgets.QPushButton('з банку №')
        self.sp = QtWidgets.QSpinBox()
        self.sp.setRange(0, 99)
        self.b_play = QtWidgets.QPushButton('▶ пуск')
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nreal = QtWidgets.QSpinBox()
        self.nreal.setRange(1, 24)
        self.nreal.setValue(6)
        self.nreal.setPrefix('реалізацій: ')
        self.status = QtWidgets.QLabel('натисни «НОВА МЕРЕЖА»')
        self.status.setStyleSheet(f'color:{GOLD};font-size:13px')

        bar = QtWidgets.QHBoxLayout()
        for w in (self.b_new, self.b_bank, self.sp, self.nreal, self.b_play):
            bar.addWidget(w)
        bar.addWidget(self.sl, 3)
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.canvas, 1)
        box.addLayout(bar)
        box.addWidget(self.status)
        c = QtWidgets.QWidget()
        c.setLayout(box)
        c.setStyleSheet(f'background:{BG};color:{FG}')
        self.setCentralWidget(c)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.b_new.clicked.connect(self._new)
        self.b_bank.clicked.connect(self._bank)
        self.b_play.clicked.connect(self._play)
        self.sl.valueChanged.connect(self._scrub)
        self.seed = 1

    # ---------- оформлення осей ----------
    def _axes(self):
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.15, 1],
                                   hspace=0.34, wspace=0.22,
                                   left=0.06, right=0.985,
                                   top=0.93, bottom=0.09)
        self.a1 = self.fig.add_subplot(gs[0, 0], facecolor=BG)
        self.a2 = self.fig.add_subplot(gs[1, 0], facecolor=BG)
        self.a3 = self.fig.add_subplot(gs[:, 1], facecolor=BG)
        for ax in (self.a1, self.a2, self.a3):
            ax.tick_params(colors=MUT, labelsize=8)
            for s in ax.spines.values():
                s.set_color('#30363d')
            ax.grid(alpha=0.13, color=MUT)
        self.a1.set_title('ДОЩ СЕМПЛІВ · риски = перетини середнього з істиною',
                          color=FG, fontsize=10)
        self.a1.set_xlabel('семплів', color=MUT)
        self.a2.set_title('ЗНАКОВЕ ВІДХИЛЕННЯ вздовж напрямку відповіді, %',
                          color=FG, fontsize=10)
        self.a2.set_xlabel('витрачені флопи', color=MUT)
        self.a3.set_title('ВІДНОСНА ПОХИБКА vs ФЛОПИ', color=FG, fontsize=10)
        self.a3.set_xlabel('витрачені флопи', color=MUT)
        self.a3.set_ylabel('‖оцінка − істина‖ / ‖істина‖, %', color=MUT)

    # ---------- обчислення ----------
    def _say(self, m):
        self.status.setText(m)
        QtWidgets.QApplication.processEvents()

    def _compute(self, W=None, net=0, tag=''):
        for b in (self.b_new, self.b_bank, self.b_play):
            b.setEnabled(False)
        self.timer.stop()
        self.b_play.setText('▶ пуск')
        try:
            self.D = jitanim.collect(net, self.nreal.value(), self.dev,
                                     W=W, cb=self._say)
        except Exception as e:                       # noqa: BLE001
            self._say(f'🛑 {type(e).__name__}: {e}')
            for b in (self.b_new, self.b_bank, self.b_play):
                b.setEnabled(True)
            return
        self._load(tag)
        for b in (self.b_new, self.b_bank, self.b_play):
            b.setEnabled(True)

    def _new(self):
        self.seed += 1
        W = jitanim.gen_net(10_000 + 7919 * self.seed, self.dev)
        self._compute(W=W, net=0, tag=f'НОВА мережа #{self.seed}')

    def _bank(self):
        n = self.sp.value()
        self._compute(W=jitanim.bank_net(n, self.dev), net=n,
                      tag=f'банк №{n}')

    # ---------- малювання ----------
    def _load(self, tag):
        D = self.D
        F, rain, tu = D['F'], D['rain'], float(D['truth_u'])
        for ax in (self.a1, self.a2, self.a3):
            for ln in list(ax.lines) + list(ax.collections):
                ln.remove()
            for t in list(ax.texts):
                t.remove()
        nr = len(rain)
        rm = np.cumsum(rain) / np.arange(1, nr + 1)
        # 🔑 ПЕРЕТИНИ: там, де знак (біжуче середнє − істина) міняється
        cr = np.where(np.diff(np.sign(rm - tu)) != 0)[0] + 1
        self.cross = cr
        self.rm = rm

        self.a1.axhline(tu, color=FG, ls='--', lw=1.2)
        self.d1 = self.a1.scatter([], [], s=5, c='#7d8590', alpha=0.35)
        self.l1, = self.a1.plot([], [], color=RED, lw=1.8)
        self.a1.set_xscale('log')
        self.a1.set_xlim(1, nr)
        lo, hi = np.percentile(rain, [0.5, 99.5])
        self.a1.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

        self.a2.axhline(0, color=FG, ls='--', lw=1.2)
        self.p2, = self.a2.plot([], [], color=RED, lw=2,
                                label='чистий семплінг')
        self.o2, = self.a2.plot([], [], color=CYAN, lw=2,
                                label='наш метод (sub87)')
        self.a2.axvline(D['f_budget'], color=GOLD, ls=':', lw=1.5)
        self.a2.set_xscale('log')
        self.a2.set_xlim(F[0], F[-1])
        yl = min(6.0, 1.25 * max(np.abs(D['SP'][0]).max(),
                                 np.abs(D['SO'][0]).max()))
        self.a2.set_ylim(-yl, yl)
        self.a2.legend(fontsize=8, facecolor='#161b22', labelcolor=FG,
                       framealpha=.8)

        for r in range(D['EP'].shape[0]):
            self.a3.plot(F, D['EP'][r] * 100, color=RED, lw=0.6, alpha=0.2)
            self.a3.plot(F, D['EO'][r] * 100, color=CYAN, lw=0.6, alpha=0.2)
        self.m3p, = self.a3.plot([], [], color=RED, lw=2.4,
                                 label='чистий семплінг')
        self.m3o, = self.a3.plot([], [], color=CYAN, lw=2.4, label='наш метод')
        self.a3.axvline(D['f_budget'], color=GOLD, ls=':', lw=1.5)
        self.a3.set_xscale('log')
        self.a3.set_yscale('log')
        self.a3.set_xlim(F[0], F[-1])
        self.a3.legend(fontsize=8, facecolor='#161b22', labelcolor=FG,
                       framealpha=.8)
        self.ann = self.a3.text(0.03, 0.06, '', transform=self.a3.transAxes,
                                color=GOLD, fontsize=12, fontweight='bold')

        self.sl.setRange(0, len(F) - 1)
        kb = int(np.argmin(np.abs(F - D['f_budget'])))
        g = float(np.median((D['EP'][:, kb] / D['EO'][:, kb]) ** 2))
        self.tag = (f'{tag} · npair {D["npair0"]} · перетинів {len(cr)}'
                    f' (перший на {cr[0] if len(cr) else "—"} семплі)'
                    f' · виграш у бюджеті {g:.2f}×')
        self._frame(len(F) - 1)
        self.sl.setValue(len(F) - 1)

    def _frame(self, k):
        D, F = self.D, self.D['F']
        self.k = k = int(np.clip(k, 0, len(F) - 1))
        nr = len(D['rain'])
        ns = max(2, int(nr * (k + 1) / len(F)))
        xs = np.arange(1, ns + 1)
        self.d1.set_offsets(np.c_[xs, D['rain'][:ns]])
        self.l1.set_data(xs, self.rm[:ns])
        for t in list(self.a1.texts):
            t.remove()
        vis = self.cross[self.cross < ns]
        self.a1.set_xlabel(f'семплів (перетинів досі: {len(vis)})', color=MUT)
        self.p2.set_data(F[:k + 1], D['SP'][0, :k + 1])
        self.o2.set_data(F[:k + 1], D['SO'][0, :k + 1])
        self.m3p.set_data(F[:k + 1], D['EP'][0, :k + 1] * 100)
        self.m3o.set_data(F[:k + 1], D['EO'][0, :k + 1] * 100)
        g = float(np.median((D['EP'][:, k] / D['EO'][:, k]) ** 2))
        self.ann.set_text(f'виграш у MSE: {g:.2f}×')
        self.status.setText(self.tag)
        self.canvas.draw_idle()

    def _scrub(self, v):
        if self.D is not None:
            self._frame(v)

    def _play(self):
        if self.D is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.b_play.setText('▶ пуск')
        else:
            self.timer.start(90)
            self.b_play.setText('⏸ пауза')

    def _tick(self):
        k = (self.k + 1) % len(self.D['F'])
        self.sl.setValue(k)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda:4'
    app = QtWidgets.QApplication(sys.argv)
    w = Win(dev)
    w.show()
    sys.exit(app.exec_())
