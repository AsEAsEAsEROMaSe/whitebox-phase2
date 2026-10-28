"""ІНТЕРАКТИВНИЙ СТЕНД: ДЕ ХТО СИДИТЬ НА ОДНІЙ ПРЯМІЙ (С80). Темна тема.

Юзер: «якщо є зміщена сіра лінія, то синя гуляє навколо якоїсь віртуальної.
має існувати механізм, щоб зрозуміти відношення. хочу інтерактивну з кнопками»
+ «в темній темі. і хочу оте посемплову візуалізацію ще окрім цих».

🔑 ГОЛОВНА ДУМКА, ЯКУ СТЕНД ПОКАЗУЄ РУКАМИ:
   усі методи — це ТОЧКИ НА ОДНІЙ ПРЯМІЙ `похибка = σ/√N_еф`.
   Різниця між нами, атрактором і лідером — лише в тому, ДЕ хто на ній сидить:
       атрактор     N_еф ≈ 2        (уся конструкція варта двох семплів)
       наш топовий  N_еф ≈ 18 000   (машинерія купує ×2.2 понад сирий МК)
       лідер        N_еф ≈ 440 000
   А «віртуальна лінія», навколо якої гуляє синя, — це МЕДІАНА, і розмах
   навколо неї заданий χ² з `k` ступенями волі (`√(2/k)`).

## П'ЯТЬ ПАНЕЛЕЙ
 1 — одна пряма: усі методи як точки на `σ/√N`
 2 — гуляння: що таке «віртуальна лінія» і чому розмах не спадає
 3 — `N_еф` стовпчиками
 4 — чому рух по прямій не міняє бал
 5 — 🆕 ПОСЕМПЛОВА ХОДА: як біжуче середнє йде до істини, семпл за семплом,
     у кадрі, прибитому до атрактора. Повзунок гортає кількість семплів.

## КНОПКИ
  [1..5]  панель · [МК][атрактор][наш][лідер][ціль] криві · [◀ мережа ▶]
  [усі мережі] · [log/lin] · [сирі семпли] (для панелі 5) · [зберегти PNG]
  повзунок — кількість семплів на панелі 5 (лог-шкала, 1…16384)

⚠️ Дані: `side66/logs/oneline.npz` (готує `oneline.py`) і
   `side66/logs/walk.npz` (готує `walkdata.py`). GPU не потрібен.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/linegui.py
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
matplotlib.rcParams.update({
    'figure.facecolor': '#15181d', 'axes.facecolor': '#1b1f26',
    'savefig.facecolor': '#15181d', 'text.color': '#e6e6e6',
    'axes.labelcolor': '#e6e6e6', 'axes.edgecolor': '#4a5160',
    'xtick.color': '#c8c8c8', 'ytick.color': '#c8c8c8',
    'grid.color': '#333a45', 'legend.facecolor': '#22262e',
    'legend.edgecolor': '#4a5160', 'axes.titlecolor': '#f0f0f0'})
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure                             # noqa: E402
from PyQt5 import QtWidgets, QtCore                              # noqa: E402

D1 = '/mnt/win_d/llm/White-Box/side66/logs/oneline.npz'
D2 = '/mnt/win_d/llm/White-Box/side66/logs/walk.npz'
SHOT = '/mnt/win_d/llm/White-Box/side66/shots/linegui.png'
QSS = """
QWidget { background:#15181d; color:#e6e6e6; font-size:12px; }
QPushButton { background:#262b34; border:1px solid #414855; border-radius:4px;
              padding:6px 10px; }
QPushButton:hover { background:#313846; }
QPushButton:checked { background:#2d6ea8; border:1px solid #4b9fe0; }
QLabel { color:#9fb4c8; }
QSlider::groove:horizontal { height:6px; background:#262b34; border-radius:3px; }
QSlider::handle:horizontal { background:#4b9fe0; width:14px; margin:-5px 0;
                             border-radius:7px; }
"""
BLUE, GREY, RED, GREEN = '#4b9fe0', '#8a93a3', '#e5484d', '#46c07a'


class Win(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        z = np.load(D1)
        self.NS, self.ERR, self.att = z['NS'], z['ERR'], z['att']
        self.ndraw, self.nets = int(z['ndraw']), len(z['att'])
        self.prod_mse, self.prod_n = float(z['prod_mse']), int(z['prod_n'])
        self.lead, self.nrm2 = float(z['leader']), z['nrm2']
        w = np.load(D2)
        self.walk, self.raw = w['walk'], w['raw']
        self.watt, self.wmu, self.K = w['att'], w['mu'], int(w['K'])
        self.panel, self.net, self.logy, self.showraw = 0, -1, True, True
        self.kk = self.K
        self.show_ = {'МК': True, 'атрактор': True, 'наш': True,
                      'лідер': True, 'ціль': True}

        self.fig = Figure(figsize=(13, 7.5))
        self.cv = FigureCanvasQTAgg(self.fig)
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.cv)

        r1 = QtWidgets.QHBoxLayout()
        for i, nm in enumerate(('1 · одна пряма', '2 · гуляння', '3 · N_еф',
                                '4 · бал не рухається',
                                '5 · ПОСЕМПЛОВА ХОДА')):
            b = QtWidgets.QPushButton(nm)
            b.clicked.connect(lambda _, k=i: self.set_panel(k))
            r1.addWidget(b)
        root.addLayout(r1)

        r2 = QtWidgets.QHBoxLayout()
        for nm in self.show_:
            b = QtWidgets.QPushButton(nm)
            b.setCheckable(True)
            b.setChecked(True)
            b.clicked.connect(lambda st, k=nm: self.toggle(k, st))
            r2.addWidget(b)
        br = QtWidgets.QPushButton('сирі семпли')
        br.setCheckable(True)
        br.setChecked(True)
        br.clicked.connect(self.flip_raw)
        bl = QtWidgets.QPushButton('log / lin')
        bl.clicked.connect(self.flip_log)
        r2.addWidget(br)
        r2.addWidget(bl)
        root.addLayout(r2)

        r3 = QtWidgets.QHBoxLayout()
        for nm, fn in (('◀ мережа', lambda: self.step(-1)),
                       ('мережа ▶', lambda: self.step(+1)),
                       ('усі мережі', lambda: self.set_net(-1)),
                       ('зберегти PNG', self.save)):
            b = QtWidgets.QPushButton(nm)
            b.clicked.connect(fn)
            r3.addWidget(b)
        self.lab = QtWidgets.QLabel('')
        r3.addWidget(self.lab, 1)
        root.addLayout(r3)

        r4 = QtWidgets.QHBoxLayout()
        r4.addWidget(QtWidgets.QLabel('семплів на панелі 5:'))
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setRange(0, 1000)
        self.sl.setValue(1000)
        self.sl.valueChanged.connect(self.set_k)
        r4.addWidget(self.sl, 1)
        self.klab = QtWidgets.QLabel(f'{self.K}')
        r4.addWidget(self.klab)
        root.addLayout(r4)

        self.setStyleSheet(QSS)
        self.setWindowTitle('Одна пряма — де хто сидить (С80)')
        self.resize(1450, 1000)
        self.draw()

    def _err(self):
        if self.net < 0:
            return self.ERR
        s = self.net * self.ndraw
        return self.ERR[:, s:s + self.ndraw]

    def _prod_rel(self):
        nn = self.nrm2 if self.net < 0 else self.nrm2[self.net:self.net + 1]
        return float(np.sqrt(256 * self.prod_mse / np.median(nn)))

    def set_panel(self, k):
        self.panel = k
        self.draw()

    def toggle(self, k, st):
        self.show_[k] = st
        self.draw()

    def flip_log(self):
        self.logy = not self.logy
        self.draw()

    def flip_raw(self, st):
        self.showraw = st
        self.draw()

    def step(self, d):
        self.net = max(-1, min(self.nets - 1, self.net + d))
        self.draw()

    def set_net(self, n):
        self.net = n
        self.draw()

    def set_k(self, v):
        self.kk = max(1, int(round(self.K ** (v / 1000.0))))
        self.klab.setText(f'{self.kk}')
        if self.panel == 4:
            self.draw()

    def save(self):
        self.fig.savefig(SHOT, dpi=110)
        self.lab.setText(f'збережено → {SHOT}')

    def draw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        E = self._err()
        med = np.median(E, 1)
        sig, att = med[0], float(np.median(
            self.att if self.net < 0 else self.att[self.net:self.net + 1]))
        prod = self._prod_rel()
        ne = {'атрактор': (sig / att) ** 2, 'наш': (sig / prod) ** 2,
              'лідер': (sig / prod) ** 2 * self.lead, 'ціль': (sig / 2e-4) ** 2}
        who = 'усі мережі' if self.net < 0 else f'мережа {self.net+1}/{self.nets}'
        self.lab.setText(f'{who}   ·   σ={100*sig:.2f}%   ·   '
                         f'машинерія ×{ne["наш"]/self.prod_n:.2f}')

        if self.panel == 0:
            if self.show_['МК']:
                ax.fill_between(self.NS, np.percentile(E, 10, 1),
                                np.percentile(E, 90, 1), alpha=.22, color=BLUE)
                ax.plot(self.NS, med, 'o-', color=BLUE, lw=2, label='МК')
            if self.show_['атрактор']:
                ax.axhline(att, color=GREY, ls='--', lw=2,
                           label=f'атрактор {100*att:.1f}% (N_еф≈{ne["атрактор"]:.0f})')
            if self.show_['наш']:
                ax.plot([ne['наш']], [prod], 'o', color=BLUE, ms=15, mec='w',
                        zorder=6, label=f'наш топовий N_еф={ne["наш"]:,.0f}')
            if self.show_['лідер']:
                ax.plot([ne['лідер']], [prod / np.sqrt(self.lead)], 'o',
                        color=RED, ms=15, mec='w', zorder=6,
                        label=f'лідер N_еф={ne["лідер"]:,.0f}')
            if self.show_['ціль']:
                ax.axhline(2e-4, color=GREEN, ls='-.', lw=1.6, label='ціль 0.02%')
            ax.set_xscale('log')
            ax.set_xlabel('ефективні семпли  N_еф')
            ax.set_ylabel('відносна похибка')
            ax.set_title('Усі методи — точки на ОДНІЙ прямій σ/√N')

        elif self.panel == 1:
            R = E / med[:, None]
            for i, n in enumerate(self.NS):
                ax.scatter([n] * R.shape[1], R[i], s=4, alpha=.22, color=BLUE)
            w = float(np.median(np.std(R[3:], 1)))
            ax.axhline(1, color=RED, lw=2, label='віртуальна лінія = медіана')
            for s in (+1, -1):
                ax.axhline(1 + s * w, color=RED, ls=':', lw=1.5)
            ax.plot([], [], ls=':', color=RED,
                    label=f'±{100*w:.0f}%  ⟹  k ≈ {0.5/max(w,1e-9)**2:.1f}')
            ax.set_xscale('log')
            ax.set_xlabel('N'); ax.set_ylabel('похибка ÷ медіана')
            ax.set_title('«Гуляння» = χ² з k ступенями волі; розмах НЕ спадає з N')

        elif self.panel == 2:
            nm = ['атрактор', 'один\nсемпл', 'наш\nтоповий', 'лідер', 'ціль']
            vv = [ne['атрактор'], 1, ne['наш'], ne['лідер'], ne['ціль']]
            ax.bar(nm, vv, color=[GREY, '#c8c8c8', BLUE, RED, GREEN], alpha=.9)
            for i, v in enumerate(vv):
                ax.text(i, v * 1.5, f'{v:,.0f}', ha='center', color='#e6e6e6')
            ax.set_yscale('log'); ax.set_ylabel('N_еф')
            ax.set_title(f'Наша машинерія купує ×{ne["наш"]/self.prod_n:.2f} '
                         'понад сирий МК')
            ax.grid(alpha=.25, axis='y')
            self.cv.draw()
            return

        elif self.panel == 3:
            Ng = np.logspace(0, 6, 300)
            sc = (sig / np.sqrt(Ng)) ** 2 * np.maximum(
                0.1, 0.0997 * Ng / self.prod_n)
            sc = sc / sc[np.argmin(np.abs(Ng - self.prod_n))]
            ax.plot(Ng, sc, lw=2.5, color=BLUE)
            ax.axvline(self.prod_n, color=BLUE, ls='--', label=f'ми: N={self.prod_n}')
            ax.axvline(0.1 / 0.0997 * self.prod_n, color=GREY, ls=':',
                       label='підлога множника 10%')
            ax.set_xscale('log'); ax.set_xlabel('N (семплів)')
            ax.set_ylabel('бал, віднормований до нашого')
            ax.set_title('Вище підлоги MSE×бюджет = const ⟹ бал ПЛАСКИЙ')

        else:
            # ── 5 · ПОСЕМПЛОВА ХОДА ─────────────────────────────────
            j = max(0, self.net) if self.net >= 0 else 0
            j = min(j, len(self.walk) - 1)
            mu = float(self.wmu[j])
            wk = self.walk[j][:self.kk] / mu
            rw = self.raw[j][:self.kk] / mu
            at = self.watt[j] / mu
            if self.showraw:
                ax.scatter(rw[:, 0], rw[:, 1], s=3, alpha=.10, color='#5d6b7d',
                           label=f'сирі семпли (розкид ±{100*np.std(rw):.0f}%)')
            ax.plot(wk[:, 0], wk[:, 1], '-', color=BLUE, lw=1.1, alpha=.9,
                    label='хода біжучого середнього')
            ax.scatter(wk[-1, 0], wk[-1, 1], s=140, color=BLUE, ec='w',
                       zorder=6, label=f'де ми при N={self.kk}')
            ax.scatter([0], [0], s=260, marker='*', color=GREEN, ec='w',
                       zorder=7, label='істина E[o]')
            ax.axhline(at[1], color=GREY, ls='--', lw=2,
                       label=f'атрактор (нерухомий, {100*abs(at[1]):.1f}%)')
            ax.set_xlabel('уздовж E[o]   (частка ‖E[o]‖)')
            ax.set_ylabel('уздовж осі атрактора   (частка ‖E[o]‖)')
            ax.set_title(f'ПОСЕМПЛОВА ХОДА, мережа {j+1}:  сірий стоїть, '
                         f'синій блукає й сходиться\nN={self.kk}   '
                         f'похибка {100*np.hypot(*wk[-1]):.3f}%')
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=.25)
            ax.legend(fontsize=9, loc='upper right')
            self.cv.draw()
            return

        if self.logy:
            ax.set_yscale('log')
        ax.grid(alpha=.25, which='both')
        ax.legend(fontsize=9)
        self.cv.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Win()
    w.show()
    sys.exit(app.exec_())
