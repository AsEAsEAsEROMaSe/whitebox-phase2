"""ПЕРЕГЛЯДАЧ МАТРИЦЬ ЗАДАЧІ — ІНТЕРАКТИВНИЙ СТЕНД (С83)

## НАВІЩО, ЯКЩО Є СТАТИЧНІ КАРТИНКИ
Статична теплова карта показує ОДНЕ нормування. А весь сьогоднішній результат
з'явився саме з ПОРІВНЯННЯ нормувань: сира матриця показує лише масштаб мережі,
а z по рядку відкриває пожеребну структуру, і кореляції в цих двох зрізах мають
ПРОТИЛЕЖНІ знаки ([[finding-per-network-personalization-needs-an-unobservable-projection]]).
Тут перемикач нормування — головна ручка, а не оздоба.

## ЧОТИРИ ВКЛАДКИ
 1 МАТРИЦІ    — теплова карта будь-якої величини, 4 нормування, значення під
                курсором, клік по рядку/стовпцю → зріз збоку
 2 КОРЕЛЯЦІЇ  — матриця корел у зрізі МІЖ / ВСЕРЕДИНІ + діаграма розсіяння пари
 3 КОНСТРУКТОР ОЗНАК — 🎯 головне: складаєш `A`, `A/B`, `A−B`, `A·B` з наявних
                величин і одразу бачиш корел МІЖ/ВСЕРЕДИНІ та **чесний виграш
                LOO по мережах** для ваги бленду `w` і масштабу виходу `c`.
                Саме цей цикл я проганяв руками пів сесії.
 4 ПРОФІЛІ    — λ сабміта (4 родини × 8 шарів), пошарова усадка формула↔оракул

## ⛔ ПЛОМБИ ВБУДОВАНО
Внизу вкладки 3 завжди стоять два контролі: ⛔НЕГ (ознака зі зсунутої мережі) і
⛔ЧИСТИЙ ШУМ. Якщо кандидат не б'є їх — виграшу немає, хоч би що показував корел.
`sub92` (w=0.0883, c=1) — база; усе рахується проти неї.

⚠️ Дані (усі вже пораховані, GPU НЕ потрібен):
   `arena62/logs/wcjoint2.npz` · `wvout.npz` · `perlayer.npz` · `wpred.npz`

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/mxgui.py
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
matplotlib.rcParams.update({
    'figure.facecolor': '#14181b', 'axes.facecolor': '#1a2024',
    'axes.edgecolor': '#39454a', 'axes.labelcolor': '#c8d4d6',
    'text.color': '#e2eaeb', 'xtick.color': '#93a3a8', 'ytick.color': '#93a3a8',
    'grid.color': '#2b353a', 'font.size': 8, 'figure.autolayout': False,
})
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,  # noqa: E402
                                               NavigationToolbar2QT)
from matplotlib.figure import Figure                              # noqa: E402
from PyQt5 import QtCore, QtWidgets                               # noqa: E402

LOG = '/mnt/win_d/llm/White-Box/arena62/logs/'
W_SUB, ALPHA = 0.0883, -0.00751


def load():
    """Усі пожеребні величини (48 мереж × 16 жеребів) в один словник."""
    A = np.load(LOG + 'wcjoint2.npz')
    B = np.load(LOG + 'wvout.npz')
    d = {}
    for k in ('OO', 'OD', 'DD', 'OT', 'DT', 'TT', 'SP', 'MM'):
        d[k] = A[k].astype(float)
    for k, nm in (('VO', 'VO'), ('NU', 'NU'), ('ZZ', 'ZZ'), ('DE', 'DE')):
        d[nm] = B[k].astype(float)
    n, m = d['OO'].shape
    d['NP'] = np.repeat(B['NP'][:, None], m, 1).astype(float)
    # похідні, які реально вживались у дослідженні
    d['‖pr‖²'] = d['OO'] + 2 * W_SUB * d['OD'] + W_SUB ** 2 * d['DD']
    d['⟨pr,T⟩'] = d['OT'] + W_SUB * d['DT']
    d['w_опт'] = -d['NU'] / (2 * d['DE'])
    d['c_опт'] = d['⟨pr,T⟩'] / (2 * d['‖pr‖²'])
    d['σ²=−NU/2'] = -d['NU'] / 2
    d['‖b‖²=DD−σ²'] = d['DD'] + d['NU'] / 2
    return d


NICE = {'OO': '‖ours‖²', 'OD': '⟨ours,Δ⟩', 'DD': '‖Δ‖²', 'OT': '⟨ours,T₁+T₂⟩',
        'DT': '⟨Δ,T₁+T₂⟩', 'TT': '⟨T₁,T₂⟩', 'SP': 'півподіл σ̂²',
        'MM': '‖m_deep‖²', 'VO': 'vout', 'NU': 'NU (лін. член MSE)',
        'ZZ': '⟨a₁,a₂⟩', 'DE': '‖Δ‖² (з wvout)', 'NP': 'n_pairs'}

NORMS = ['сире', 'z по МЕРЕЖІ (рядку)', 'z по ЖЕРЕБУ (стовпцю)', 'log10|·|']


def apply_norm(M, mode):
    if mode == 1:
        s = M.std(1, keepdims=True)
        return (M - M.mean(1, keepdims=True)) / np.maximum(s, 1e-300)
    if mode == 2:
        s = M.std(0, keepdims=True)
        return (M - M.mean(0, keepdims=True)) / np.maximum(s, 1e-300)
    if mode == 3:
        return np.log10(np.maximum(np.abs(M), 1e-300))
    return M


def dm(M):
    return M - M.mean(1, keepdims=True)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, w=7.4, h=4.6):
        self.fig = Figure(figsize=(w, h))
        super().__init__(self.fig)


class TabMatrix(QtWidgets.QWidget):
    """Вкладка 1: теплова карта + зрізи."""

    def __init__(self, D):
        super().__init__()
        self.D = D
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.sel = QtWidgets.QComboBox()
        self.sel.addItems([NICE.get(k, k) for k in D])
        self.keys = list(D)
        self.norm = QtWidgets.QComboBox()
        self.norm.addItems(NORMS)
        self.norm.setCurrentIndex(1)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(['RdBu_r', 'viridis', 'magma', 'coolwarm'])
        for w, t in ((self.sel, 'величина'), (self.norm, 'нормування'),
                     (self.cmap, 'палітра')):
            bar.addWidget(QtWidgets.QLabel(t))
            bar.addWidget(w, 2)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.cv = Canvas(9.6, 5.0)
        lay.addWidget(NavigationToolbar2QT(self.cv, self))
        lay.addWidget(self.cv, 1)
        self.stat = QtWidgets.QLabel('наведи курсор на карту')
        self.stat.setStyleSheet('font-family:monospace;color:#8fd3dd;padding:3px')
        lay.addWidget(self.stat)
        for w in (self.sel, self.norm, self.cmap):
            w.currentIndexChanged.connect(self.draw)
        self.cv.mpl_connect('motion_notify_event', self.hover)
        self.cv.mpl_connect('button_press_event', self.click)
        self.pick = (None, None)
        self.draw()

    def cur(self):
        return self.D[self.keys[self.sel.currentIndex()]]

    def draw(self):
        M = self.cur()
        Z = apply_norm(M, self.norm.currentIndex())
        f = self.cv.fig
        f.clear()
        gs = f.add_gridspec(2, 2, width_ratios=[3, 1.25], height_ratios=[3, 1.1],
                            hspace=0.28, wspace=0.22, left=.07, right=.97,
                            top=.93, bottom=.1)
        self.ax = f.add_subplot(gs[0, 0])
        v = np.abs(Z).max() if self.norm.currentIndex() in (1, 2) else None
        im = self.ax.imshow(Z, aspect='auto', cmap=self.cmap.currentText(),
                            vmin=-2.5 if v else None, vmax=2.5 if v else None,
                            interpolation='nearest')
        self.ax.set_ylabel('мережа')
        self.ax.set_xlabel('жереб')
        self.ax.set_title(f'{NICE.get(self.keys[self.sel.currentIndex()])}'
                          f'   ·   {NORMS[self.norm.currentIndex()]}', fontsize=9)
        f.colorbar(im, ax=self.ax, fraction=0.04)
        # зрізи
        r, c = self.pick
        axr = f.add_subplot(gs[0, 1])
        axc = f.add_subplot(gs[1, 0])
        if r is not None:
            axr.plot(Z[r], np.arange(Z.shape[1]), '.-', color='#5fc0cc', lw=1)
            axr.set_title(f'мережа {r}', fontsize=8)
            axr.invert_yaxis()
        else:
            axr.text(.5, .5, 'клік по карті\n→ зріз мережі', ha='center',
                     va='center', color='#6f7f85', fontsize=8)
            axr.set_xticks([]); axr.set_yticks([])
        if c is not None:
            axc.plot(Z[:, c], '.-', color='#d9a05b', lw=1)
            axc.set_title(f'жереб {c} — профіль по мережах', fontsize=8)
        else:
            axc.text(.5, .5, 'клік по карті → профіль жеребу', ha='center',
                     va='center', color='#6f7f85', fontsize=8)
            axc.set_xticks([]); axc.set_yticks([])
        for a in (axr, axc):
            a.grid(alpha=.25)
        # статистика розкиду
        ax4 = f.add_subplot(gs[1, 1])
        wsd = dm(M).std()
        bsd = M.mean(1).std()
        ax4.barh([0, 1], [bsd, wsd], color=['#5fc0cc', '#d9695b'])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['МІЖ', 'ВСЕР'], fontsize=7)
        ax4.set_title(f'розкид  всер/між = {wsd/max(bsd,1e-300):.3f}', fontsize=8)
        ax4.grid(alpha=.25, axis='x')
        self.cv.draw_idle()

    def hover(self, e):
        if e.inaxes is not getattr(self, 'ax', None) or e.xdata is None:
            return
        M = self.cur()
        c = int(round(e.xdata)); r = int(round(e.ydata))
        if 0 <= r < M.shape[0] and 0 <= c < M.shape[1]:
            Z = apply_norm(M, self.norm.currentIndex())
            self.stat.setText(f'мережа {r:2d} · жереб {c:2d}   '
                              f'сире {M[r,c]:+.6e}   нормоване {Z[r,c]:+.3f}')

    def click(self, e):
        if e.inaxes is not getattr(self, 'ax', None) or e.xdata is None:
            return
        self.pick = (int(round(e.ydata)), int(round(e.xdata)))
        self.draw()


class TabCorr(QtWidgets.QWidget):
    """Вкладка 2: кореляції у двох зрізах + розсіяння пари."""

    def __init__(self, D):
        super().__init__()
        self.D = D
        self.keys = list(D)
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['МІЖ мережами (середні по жеребах)',
                            'ВСЕРЕДИНІ мережі (пожеребні відхилення)'])
        bar.addWidget(QtWidgets.QLabel('зріз'))
        bar.addWidget(self.mode, 2)
        self.a = QtWidgets.QComboBox(); self.b = QtWidgets.QComboBox()
        for w in (self.a, self.b):
            w.addItems([NICE.get(k, k) for k in D])
        self.a.setCurrentIndex(self.keys.index('DD'))
        self.b.setCurrentIndex(self.keys.index('w_опт'))
        bar.addWidget(QtWidgets.QLabel('пара'))
        bar.addWidget(self.a, 1); bar.addWidget(self.b, 1)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.cv = Canvas(9.6, 5.2)
        lay.addWidget(self.cv, 1)
        self.note = QtWidgets.QLabel('')
        self.note.setStyleSheet('font-family:monospace;color:#8fd3dd;padding:3px')
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        for w in (self.mode, self.a, self.b):
            w.currentIndexChanged.connect(self.draw)
        self.draw()

    def vecs(self):
        f = (lambda M: M.mean(1)) if self.mode.currentIndex() == 0 else \
            (lambda M: dm(M).ravel())
        return [f(self.D[k]) for k in self.keys]

    def draw(self):
        V = self.vecs()
        good = [i for i, v in enumerate(V) if np.std(v) > 1e-300]
        C = np.corrcoef(np.array([V[i] for i in good]))
        lbl = [NICE.get(self.keys[i], self.keys[i]) for i in good]
        f = self.cv.fig; f.clear()
        ax = f.add_subplot(1, 2, 1)
        im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(lbl)))
        ax.set_xticklabels(lbl, rotation=90, fontsize=6)
        ax.set_yticks(range(len(lbl)))
        ax.set_yticklabels(lbl, fontsize=6)
        for i in range(len(lbl)):
            for j in range(len(lbl)):
                ax.text(j, i, f'{C[i,j]:+.2f}', ha='center', va='center',
                        fontsize=4.6,
                        color='white' if abs(C[i, j]) > .55 else '#1a2024')
        ax.set_title(self.mode.currentText(), fontsize=8.5)
        f.colorbar(im, ax=ax, fraction=0.045)
        ax2 = f.add_subplot(1, 2, 2)
        ia, ib = self.a.currentIndex(), self.b.currentIndex()
        x, y = V[ia], V[ib]
        if np.std(x) < 1e-300 or np.std(y) < 1e-300:
            ax2.text(.5, .5, 'величина детермінована\n(нульова дисперсія в цім зрізі)',
                     ha='center', va='center', color='#d9695b', fontsize=9)
            r = float('nan')
        else:
            ax2.plot(x, y, '.', ms=3.4, color='#5fc0cc', alpha=.7)
            r = float(np.corrcoef(x, y)[0, 1])
            p = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax2.plot(xs, np.polyval(p, xs), '-', color='#d9a05b', lw=1.2)
        ax2.set_xlabel(NICE.get(self.keys[ia], self.keys[ia]))
        ax2.set_ylabel(NICE.get(self.keys[ib], self.keys[ib]))
        ax2.set_title(f'корел = {r:+.3f}', fontsize=9)
        ax2.grid(alpha=.25)
        f.subplots_adjust(left=.13, right=.97, bottom=.24, top=.9, wspace=.42)
        self.cv.draw_idle()
        self.note.setText(
            'МІЖ = сигнал, який можна взяти · ВСЕРЕДИНІ = пожеребний шум. '
            'Якщо знаки протилежні — спільний фіт візьме НЕ ту компоненту '
            '(так померли і вага бленду, і масштаб виходу).')


class TabBuild(QtWidgets.QWidget):
    """Вкладка 3: конструктор ознак із чесним LOO та вбудованими пломбами."""

    OPS = ['A', 'A / B', 'A − B', 'A · B', 'A / √B', 'log A']

    def __init__(self, D):
        super().__init__()
        self.D = D
        self.keys = list(D)
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.op = QtWidgets.QComboBox(); self.op.addItems(self.OPS)
        self.op.setCurrentIndex(1)
        self.a = QtWidgets.QComboBox(); self.b = QtWidgets.QComboBox()
        for w in (self.a, self.b):
            w.addItems([NICE.get(k, k) for k in D])
        self.a.setCurrentIndex(self.keys.index('VO'))
        self.b.setCurrentIndex(self.keys.index('DD'))
        self.tgt = QtWidgets.QComboBox()
        self.tgt.addItems(['вага бленду  w', 'масштаб виходу  c'])
        self.per = QtWidgets.QCheckBox('ознака ПОЖЕРЕБНА (як у сабміті)')
        self.per.setChecked(True)
        for w, t in ((self.op, 'форма'), (self.a, 'A'), (self.b, 'B'),
                     (self.tgt, 'ціль')):
            bar.addWidget(QtWidgets.QLabel(t)); bar.addWidget(w, 1)
        bar.addWidget(self.per)
        go = QtWidgets.QPushButton('порахувати  ▶')
        go.setStyleSheet('font-weight:600;padding:4px 14px')
        bar.addWidget(go)
        lay.addLayout(bar)
        self.cv = Canvas(9.6, 3.4)
        lay.addWidget(self.cv, 1)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet('font-family:monospace;font-size:11px;'
                               'background:#11161a;color:#d6e2e4')
        lay.addWidget(self.out, 1)
        go.clicked.connect(self.run)
        self.run()

    def feat(self):
        A = self.D[self.keys[self.a.currentIndex()]]
        B = self.D[self.keys[self.b.currentIndex()]]
        o = self.op.currentIndex()
        F = (A, A / np.where(np.abs(B) < 1e-300, 1e-300, B), A - B, A * B,
             A / np.sqrt(np.maximum(np.abs(B), 1e-300)),
             np.log(np.maximum(np.abs(A), 1e-300)))[o]
        return F if self.per.isChecked() else \
            np.repeat(F.mean(1)[:, None], F.shape[1], 1)

    def loo(self, F):
        """Чесний LOO по мережах для обраної цілі. Повертає × над sub92."""
        D = self.D
        N, M = F.shape
        if self.tgt.currentIndex() == 0:                      # вага бленду
            P, Q, R = D['DE'], D['NU'], D['ZZ']

            def mse(w):
                return float(np.sum(R + w * Q + w * w * P))
            base = mse(W_SUB)
            sgn, lo, hi = -0.5, 0.0, 0.5
        else:                                                 # масштаб виходу
            P, Q, R = D['‖pr‖²'], D['⟨pr,T⟩'], D['TT']

            def mse(c):
                return float(np.sum(c * c * P - c * Q + R))
            base = mse(1.0)
            sgn, lo, hi = 0.5, 0.99, 1.01
        wv = np.zeros((N, M))
        for n in range(N):
            i = np.array([m for m in range(N) if m != n])
            fb = F[i].mean(); u = F[i] - fb
            G = np.array([[P[i].sum(), (u * P[i]).sum()],
                          [(u * P[i]).sum(), (u * u * P[i]).sum()]])
            b = sgn * np.array([Q[i].sum(), (u * Q[i]).sum()])
            try:
                a, k = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                a, k = (W_SUB if self.tgt.currentIndex() == 0 else 1.0), 0.0
            wv[n] = np.clip(a + k * (F[n] - fb), lo, hi)
        return base / mse(wv), base

    def run(self):
        F = self.feat()
        D = self.D
        tgt = D['w_опт'] if self.tgt.currentIndex() == 0 else D['c_опт']
        cb = float(np.corrcoef(F.mean(1), tgt.mean(1))[0, 1]) \
            if F.mean(1).std() > 1e-300 else float('nan')
        cw = float(np.corrcoef(dm(F).ravel(), dm(tgt).ravel())[0, 1]) \
            if dm(F).std() > 1e-300 else float('nan')
        g, base = self.loo(F)
        rng = np.random.default_rng(7)
        gn, _ = self.loo(np.roll(F, 1, axis=0))
        gs, _ = self.loo(np.repeat(rng.standard_normal(F.shape[0])[:, None],
                                   F.shape[1], 1))
        # стеля: оракул по мережах
        orc = np.repeat(tgt.mean(1)[:, None], F.shape[1], 1)
        if self.tgt.currentIndex() == 0:
            cap = base / float(np.sum(D['ZZ'] + orc * D['NU'] + orc**2 * D['DE']))
        else:
            cap = base / float(np.sum(orc**2 * D['‖pr‖²'] - orc * D['⟨pr,T⟩']
                                      + D['TT']))
        nm = (f'{self.OPS[self.op.currentIndex()]}'
              f'   A={NICE.get(self.keys[self.a.currentIndex()])}'
              f'   B={NICE.get(self.keys[self.b.currentIndex()])}')
        ok = g > 1.002 and g > gn and g > gs
        self.out.setPlainText(
            f'ОЗНАКА   {nm}\n'
            f'режим    {"ПОЖЕРЕБНА (як у сабміті)" if self.per.isChecked() else "усереднена по 16 жеребах (у сабміті НЕДОСТУПНА)"}\n'
            f'ціль     {self.tgt.currentText()}\n'
            f'{"─"*78}\n'
            f'корел МІЖ мережами        {cb:+.4f}      ⟸ сигнал\n'
            f'корел ВСЕРЕДИНІ мережі    {cw:+.4f}      ⟸ пожеребний шум\n'
            f'{"  🚨 ЗНАКИ ПРОТИЛЕЖНІ — спільний фіт візьме НЕ ту компоненту" if cb*cw<0 else ""}\n'
            f'{"─"*78}\n'
            f'ВИГРАШ LOO по мережах     ×{g:.4f}   {"🟢" if ok else "⛔"}\n'
            f'⛔ НЕГ (зсунута мережа)    ×{gn:.4f}\n'
            f'⛔ ЧИСТИЙ ШУМ              ×{gs:.4f}\n'
            f'⛔ СТЕЛЯ (оракул мережі)   ×{cap:.4f}\n'
            f'{"─"*78}\n'
            f'{"🟢 КАНДИДАТ Б\'Є ОБИДВІ ПЛОМБИ" if ok else "⛔ не б`є пломби — виграшу немає, хоч би що показував корел"}\n'
            f'📊 бал: sub92 1.946e-07 → {1.946e-07/max(g,1e-9):.4e}'
            f'   (стеля {1.946e-07/max(cap,1e-9):.4e})')
        f = self.cv.fig; f.clear()
        ax = f.add_subplot(1, 3, 1)
        ax.plot(F.mean(1), tgt.mean(1), '.', ms=4, color='#5fc0cc')
        ax.set_title(f'МІЖ мережами  r={cb:+.3f}', fontsize=8.5)
        ax.set_xlabel('ознака'); ax.grid(alpha=.25)
        ax2 = f.add_subplot(1, 3, 2)
        ax2.plot(dm(F).ravel(), dm(tgt).ravel(), '.', ms=2, color='#d9695b',
                 alpha=.5)
        ax2.set_title(f'ВСЕРЕДИНІ  r={cw:+.3f}', fontsize=8.5)
        ax2.set_xlabel('ознака (відхилення)'); ax2.grid(alpha=.25)
        ax3 = f.add_subplot(1, 3, 3)
        bars = ax3.bar(['ознака', 'НЕГ', 'шум', 'стеля'], [g, gn, gs, cap],
                       color=['#5fc0cc' if ok else '#8a9296', '#6f7f85',
                              '#6f7f85', '#d9a05b'])
        ax3.axhline(1.0, color='#e2eaeb', lw=1, ls='--')
        ax3.set_title('× над sub92', fontsize=8.5)
        ax3.grid(alpha=.25, axis='y')
        for b, v in zip(bars, [g, gn, gs, cap]):
            ax3.text(b.get_x() + b.get_width() / 2, v, f'{v:.4f}', ha='center',
                     va='bottom', fontsize=6.5)
        f.subplots_adjust(left=.08, right=.98, bottom=.18, top=.88, wspace=.32)
        self.cv.draw_idle()


class TabProf(QtWidgets.QWidget):
    """Вкладка 4: λ сабміта і пошарова усадка."""

    LAM_S = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
             0.0088285, 0.0007922, 0.0015044, 0.0001062)
    LAM_A = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
             -0.0025822, -0.0030778, 0.0006822, 0.0068061)
    LAM_Q = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
             0.0189783, 0.0250559, 0.0340802, 0.0415433)
    LAM_G = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
             -0.0146129, -0.0100612, 0.0167223, 0.0168109)

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.cv = Canvas(9.6, 6.0)
        lay.addWidget(NavigationToolbar2QT(self.cv, self))
        lay.addWidget(self.cv, 1)
        self.draw()

    def draw(self):
        L = np.array([self.LAM_S, self.LAM_A, self.LAM_Q, self.LAM_G])
        f = self.cv.fig; f.clear()
        ax = f.add_subplot(2, 2, 1)
        im = ax.imshow(L, aspect='auto', cmap='RdBu_r',
                       vmin=-np.abs(L).max(), vmax=np.abs(L).max())
        ax.set_yticks(range(4))
        ax.set_yticklabels(['σ', 'μ', 'E[a²]', 'маска'], fontsize=8)
        ax.set_xticks(range(8)); ax.set_xticklabels(range(1, 9), fontsize=7)
        ax.set_title('λ сабміта: 4 родини × 8 шарів', fontsize=8.5)
        for i in range(4):
            for j in range(8):
                ax.text(j, i, f'{L[i,j]:+.4f}', ha='center', va='center',
                        fontsize=5, color='white' if abs(L[i, j]) > .02
                        else '#1a2024')
        f.colorbar(im, ax=ax, fraction=.045)
        ax2 = f.add_subplot(2, 2, 2)
        for r, nm in zip(L, ['σ', 'μ', 'E[a²]', 'маска']):
            ax2.plot(range(1, 9), r, 'o-', lw=1.3, ms=3, label=nm)
        ax2.axhline(0, color='#93a3a8', lw=.7)
        ax2.legend(fontsize=7); ax2.grid(alpha=.25)
        ax2.set_xlabel('шар'); ax2.set_title('профілі по глибині', fontsize=8.5)
        try:
            P = np.load(LOG + 'perlayer.npz')
            LF, LO = P['LF'], P['LO']
            ax3 = f.add_subplot(2, 2, 3)
            im = ax3.imshow(LO, aspect='auto', cmap='viridis')
            ax3.set_title('λ ОРАКУЛЬНИЙ (мережі × шари)', fontsize=8.5)
            ax3.set_xlabel('шар'); f.colorbar(im, ax=ax3, fraction=.045)
            ax4 = f.add_subplot(2, 2, 4)
            cc = [float(np.corrcoef(LF[:, k], LO[:, k])[0, 1])
                  for k in range(LF.shape[1])]
            ax4.plot(range(1, len(cc) + 1), cc, 'o-', color='#5fc0cc', lw=1.3,
                     ms=3)
            ax4.axhline(0, color='#93a3a8', lw=.7)
            ax4.set_title('корел формула↔оракул по мережах', fontsize=8.5)
            ax4.set_xlabel('шар'); ax4.grid(alpha=.25)
        except FileNotFoundError:
            pass
        f.subplots_adjust(left=.09, right=.97, bottom=.09, top=.94,
                          hspace=.42, wspace=.28)
        self.cv.draw_idle()


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Матриці White-Box — інтерактивний стенд (С83)')
        self.resize(1320, 880)
        D = load()
        t = QtWidgets.QTabWidget()
        t.addTab(TabMatrix(D), '1  Матриці')
        t.addTab(TabCorr(D), '2  Кореляції')
        t.addTab(TabBuild(D), '3  Конструктор ознак')
        t.addTab(TabProf(), '4  Профілі')
        self.setCentralWidget(t)
        self.statusBar().showMessage(
            'дані: 48 мереж × 16 жеребів, банкова істина bank_s53/truth2.npz, '
            'крос-ціль · база sub92 (w=0.0883)')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("""
      QWidget{background:#14181b;color:#dbe5e7;font-size:12px}
      QComboBox,QPushButton,QCheckBox{background:#1e262a;border:1px solid #354247;
        padding:3px 6px;border-radius:2px}
      QComboBox:hover,QPushButton:hover{border-color:#5fc0cc}
      QPushButton:focus,QComboBox:focus{outline:2px solid #5fc0cc}
      QTabBar::tab{background:#1a2024;padding:6px 14px;border:1px solid #354247}
      QTabBar::tab:selected{background:#26333a;color:#8fd3dd;font-weight:600}
      QLabel{color:#93a3a8}
    """)
    w = Main()
    w.show()
    sys.exit(app.exec_())
