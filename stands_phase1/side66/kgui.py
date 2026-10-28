"""ІНТЕРАКТИВНИЙ СТЕНД: ПОЛЕ, ЙОГО ПЕРЕКІС І РЯД ЕДЖВОРТА (С78).

Той самий стиль, що `jitgui.py` / `curvegui.py`. Питання тут — ідея юзера про
«щільне поле, як потік води»: на вхід приходить гаусова хмара, кожен шар її
деформує, а ReLU СКЛАДАЄ навпіл. Наше замикання несе тільки центр і ширину —
тобто вважає, що хмара лишається гаусовою. Вона не лишається.

  ⬅⬆ ЩІЛЬНІСТЬ НА ШАРІ — реальна гістограма стандартизованого `p_j` проти
      гаусової кривої і проти Еджвортових кривих 1/2/3 порядку. ВИДНО ПЕРЕКІС.
  ➡⬆ ПОХИБКА `E[ReLU]` по координатах — скільки кожен порядок знімає.
  ⬅⬇ ПРОФІЛЬ `γ₃, γ₄, γ₅` по 32 шарах — де поле найбільш негаусове.
  ➡⬇ СТЕЛЯ vs ПОРЯДОК — бари з лініями «наш MSE» і «лідер», плюс S/N кожного
      кумулянта на РОБОЧІЙ вибірці (те, що вирішує, чи стеля стає балом).

🚨 S/N ВИЯВИВСЯ ВЕЛИКИМ (κ₃ 9.3, κ₄ 3.4, κ₅ 1.1) — попри моє передбачення, що
   CLT зробить кумулянти невимірними. Причина: після ReLU активації невід'ємні
   й сильно пов'язані, тож «сума 256 незалежних» не діє і поле негаусове куди
   сильніше, ніж 1/√n.

Запуск:  ./torch_venv/bin/python side66/kgui.py [cuda:N]
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

BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
RED, CYAN, GOLD, GRN = '#ff2d55', '#00e5ff', '#f0c674', '#3fb950'
VIO = '#bc8cff'
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
W_, D = 256, 32
NB = 81                       # бінів гістограми
LO, HI = -4.0, 4.0
S2PI = float(1.0 / np.sqrt(2 * np.pi))
SQ2 = float(np.sqrt(2.0))
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')


# ─────────────────────────── рахунок ───────────────────────────
def _moments(W, seed, npairs, whiten=False):
    """Сирі моменти 1..5, ворота і E[relu] по всіх шарах."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(7, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done, CH = 0, 1 << 14
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            pd = p.double()
            q = pd * pd
            Z[L][0] += pd.sum(0); Z[L][1] += q.sum(0)
            Z[L][2] += (q * pd).sum(0); Z[L][3] += (q * q).sum(0)
            Z[L][4] += (q * q * pd).sum(0)
            Z[L][5] += (p > 0).double().sum(0)
            x = torch.relu(p)
            Z[L][6] += x.double().sum(0)
        done += b
    N = float(2 * npairs)
    out = []
    for L in range(D):
        a1, a2, a3, a4, a5 = (Z[L][i] / N for i in range(5))
        s2 = (a2 - a1 * a1).clamp_min(1e-30)
        s = s2.sqrt()
        mu3 = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
        mu4 = a4 - 4.0 * a1 * a3 + 6.0 * a1 * a1 * a2 - 3.0 * a1 ** 4
        mu5 = (a5 - 5.0 * a1 * a4 + 10.0 * a1 * a1 * a3
               - 10.0 * a1 ** 3 * a2 + 4.0 * a1 ** 5)
        out.append(dict(m=a1, s=s, k3=mu3, k4=mu4 - 3.0 * s2 * s2,
                        k5=mu5 - 10.0 * s2 * mu3,
                        g=Z[L][5] / N, relu=Z[L][6] / N))
    return out


def _hist(W, MM, seed, npairs, layer):
    """Гістограма СТАНДАРТИЗОВАНОГО p на шарі `layer`, окремо по координатах."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    H = torch.zeros(W_, NB, dtype=torch.float64, device=DEV)
    m, s = MM[layer]['m'], MM[layer]['s']
    done, CH, tot = 0, 1 << 14, 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(layer + 1):
            p = x @ W[L]
            if L < layer:
                x = torch.relu(p)
        u = (p.double() - m) / s                       # (2b, 256)
        idx = ((u - LO) / (HI - LO) * (NB - 1)).round().clamp(0, NB - 1).long()
        H.scatter_add_(1, idx.T, torch.ones_like(idx.T, dtype=torch.float64))
        tot += 2 * b
        done += b
    return (H / (tot * (HI - LO) / (NB - 1))).cpu().numpy()


def _edge_pdf(u, g3, g4, g5):
    """Щільність Еджворта до 3-го порядку (список: гаус, +κ₃, +κ₄, +κ₅)."""
    ph = np.exp(-0.5 * u * u) * S2PI
    He3 = u ** 3 - 3 * u
    He4 = u ** 4 - 6 * u ** 2 + 3
    He5 = u ** 5 - 10 * u ** 3 + 15 * u
    He6 = u ** 6 - 15 * u ** 4 + 45 * u ** 2 - 15
    He7 = u ** 7 - 21 * u ** 5 + 105 * u ** 3 - 105 * u
    He9 = (u ** 9 - 36 * u ** 7 + 378 * u ** 5 - 1260 * u ** 3 + 945 * u)
    t1 = g3 / 6 * He3
    t2 = g4 / 24 * He4 + g3 * g3 / 72 * He6
    t3 = g5 / 120 * He5 + g3 * g4 / 144 * He7 + g3 ** 3 / 1296 * He9
    return [ph, ph * (1 + t1), ph * (1 + t1 + t2), ph * (1 + t1 + t2 + t3)]


def _relu_err(M):
    """Похибка E[relu] для 4 порядків, покоординатно (у частках від істини)."""
    m, s = M['m'], M['s']
    t = m / s
    s2 = s * s; s3 = s2 * s; s4 = s2 * s2
    Ph = 0.5 * (1.0 + torch.erf(t / SQ2))
    ph = torch.exp(-0.5 * t * t) * S2PI
    t2 = t * t; t3 = t2 * t
    He3 = t3 - 3 * t
    He5 = t3 * t2 - 10 * t3 + 15 * t
    He7 = t3 * t2 * t2 - 21 * t3 * t2 + 105 * t3 - 105 * t
    c3 = -M['k3'] * t * ph / (6 * s2)
    c4 = M['k4'] * ph * (t2 - 1) / (24 * s3)
    c33 = M['k3'] ** 2 * ph * (t2 * t2 - 6 * t2 + 3) / (72 * s4 * s)
    c5 = -M['k5'] * ph * He3 / (120 * s4)
    c34 = -M['k3'] * M['k4'] * ph * He5 / (144 * s4 * s2)
    c333 = -M['k3'] ** 3 * ph * He7 / (1296 * s4 * s4)
    base = m * Ph + s * ph
    tr = M['relu']
    out = [base, base + c3, base + c3 + c4 + c33,
           base + c3 + c4 + c33 + c5 + c34 + c333]
    # 🚨 НЕ ділити покоординатно: мертві нейрони мають E[relu]≈0 і дають 1e25.
    #    Масштаб — ТИПОВА величина відповіді на шарі (медіана).
    sc = float(tr.abs().median().clamp_min(1e-30))
    return [((o - tr).abs() / sc).cpu().numpy() for o in out]


class Data:
    """Усе, що треба для однієї мережі. Рахується раз на зміну мережі."""

    def __init__(self, net, logn=17, npair=4096):
        self.net, self.logn, self.npair = net, logn, npair
        self.W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
                  for k in range(D)]
        with torch.no_grad():
            self.MM = _moments(self.W, 91_000 + 13 * net, 1 << logn)
            self.SM = _moments(self.W, 55_000 + 7 * net, npair, whiten=True)
        self.g3 = np.array([float((M['k3'] / M['s'] ** 3).abs().mean())
                            for M in self.MM])
        self.g4 = np.array([float((M['k4'] / M['s'] ** 4).abs().mean())
                            for M in self.MM])
        self.g5 = np.array([float((M['k5'] / M['s'] ** 5).abs().mean())
                            for M in self.MM])
        # S/N кожного кумулянта на РОБОЧІЙ вибірці
        self.snr = []
        for k, pw in (('k3', 3), ('k4', 4), ('k5', 5)):
            sg = nz = 0.0
            for L in range(D):
                sk = self.MM[L]['s'] ** pw
                a, b = self.MM[L][k] / sk, self.SM[L][k] / sk
                sg += float((a * a).sum()); nz += float((b - a).pow(2).sum())
            self.snr.append(np.sqrt(sg / max(nz, 1e-300)))
        self._hc = {}

    def hist(self, layer):
        if layer not in self._hc:
            with torch.no_grad():
                self._hc[layer] = _hist(self.W, self.MM,
                                        91_000 + 13 * self.net,
                                        1 << (self.logn - 2), layer)
        return self._hc[layer]

    def err(self, layer):
        return _relu_err(self.MM[layer])


# ─────────────────────────── GUI ───────────────────────────
class Win(QtWidgets.QMainWindow):
    NAMES = ('гаус (m,s)', '+κ₃', '+κ₄, κ₃²', '+κ₅, κ₃κ₄, κ₃³')
    COLS = (MUT, CYAN, GOLD, GRN)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Поле, перекіс і ряд Еджворта — С78')
        self.resize(1500, 950)
        self.net, self.layer, self.coord = 0, 16, 0
        self.show_ord = [True, True, True, True]
        self.D = Data(self.net)
        self._build()
        self.redraw()

    def _build(self):
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)
        self.fig = Figure(figsize=(15, 8.6), facecolor=BG)
        self.cv = FigureCanvas(self.fig)
        v.addWidget(self.cv, 1)

        bar = QtWidgets.QHBoxLayout(); v.addLayout(bar)

        def lab(txt):
            q = QtWidgets.QLabel(txt); q.setStyleSheet(f'color:{MUT}')
            bar.addWidget(q); return q

        lab('мережа')
        self.sn = QtWidgets.QSpinBox(); self.sn.setRange(0, 47)
        self.sn.valueChanged.connect(self.on_net); bar.addWidget(self.sn)

        lab('   ШАР')
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setRange(0, D - 1); self.sl.setValue(self.layer)
        self.sl.valueChanged.connect(self.on_layer); bar.addWidget(self.sl, 2)
        self.ll = lab(f'{self.layer:2d}')

        lab('   нейрон')
        self.sc = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sc.setRange(0, W_ - 1)
        self.sc.valueChanged.connect(self.on_coord); bar.addWidget(self.sc, 1)
        self.lc = lab('  0')

        b = QtWidgets.QPushButton('найперекошеніший')
        b.clicked.connect(self.worst); bar.addWidget(b)

        self.cb = []
        for i, nm in enumerate(self.NAMES):
            c = QtWidgets.QCheckBox(nm); c.setChecked(True)
            c.setStyleSheet(f'color:{self.COLS[i]}')
            c.stateChanged.connect(self.on_ord)
            bar.addWidget(c); self.cb.append(c)

    def on_net(self, v):
        self.net = v
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.D = Data(self.net)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.redraw()

    def on_layer(self, v):
        self.layer = v; self.ll.setText(f'{v:2d}'); self.redraw()

    def on_coord(self, v):
        self.coord = v; self.lc.setText(f'{v:3d}'); self.redraw()

    def on_ord(self):
        self.show_ord = [c.isChecked() for c in self.cb]; self.redraw()

    def worst(self):
        M = self.D.MM[self.layer]
        j = int((M['k3'] / M['s'] ** 3).abs().argmax())
        self.sc.setValue(j)

    def _ax(self, ax, ttl):
        ax.set_facecolor(BG); ax.set_title(ttl, color=FG, fontsize=10)
        ax.tick_params(colors=MUT, labelsize=8)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.grid(alpha=.15, color=MUT)

    def redraw(self):
        self.fig.clear()
        D_ = self.D
        L, j = self.layer, self.coord
        M = D_.MM[L]
        g3 = float(M['k3'][j] / M['s'][j] ** 3)
        g4 = float(M['k4'][j] / M['s'][j] ** 4)
        g5 = float(M['k5'][j] / M['s'][j] ** 5)

        # ── 1. щільність
        ax = self.fig.add_subplot(221)
        self._ax(ax, f'ЩІЛЬНІСТЬ поля: шар {L}, нейрон {j}   '
                     f'γ₃={g3:+.3f}  γ₄={g4:+.3f}  γ₅={g5:+.3f}')
        u = np.linspace(LO, HI, NB)
        h = D_.hist(L)[j]
        ax.bar(u, h, width=(HI - LO) / (NB - 1), color='#21262d',
               edgecolor='#30363d', linewidth=.4, label='справжня щільність')
        uf = np.linspace(LO, HI, 400)
        for i, c in enumerate(_edge_pdf(uf, g3, g4, g5)):
            if self.show_ord[i]:
                ax.plot(uf, c, color=self.COLS[i], lw=2 if i else 1.4,
                        ls='--' if i == 0 else '-', label=self.NAMES[i])
        ax.axvline(0, color=RED, lw=.8, alpha=.5)
        ax.set_xlabel('(p − m)/s', color=MUT, fontsize=8)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # ── 2. похибка E[relu] по координатах
        ax = self.fig.add_subplot(222)
        self._ax(ax, f'ПОХИБКА E[ReLU] по 256 нейронах шару {L} (відсортовано)')
        errs = D_.err(L)
        for i, e in enumerate(errs):
            if self.show_ord[i]:
                ax.semilogy(np.sort(e)[::-1], color=self.COLS[i], lw=1.6,
                            label=f'{self.NAMES[i]}  медіана {np.median(e):.2e}')
        ax.set_ylim(1e-7, 5.0)
        ax.set_xlabel('нейрон (за спаданням похибки), у частках типової відповіді',
                      color=MUT, fontsize=8)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # ── 3. профіль негаусовості по шарах
        ax = self.fig.add_subplot(223)
        self._ax(ax, 'НЕГАУСОВІСТЬ ПО ШАРАХ: середнє |γ_k| (шар 0 = рівно гаус)')
        for arr, c, nm in ((D_.g3, CYAN, '|γ₃| перекіс'),
                           (D_.g4, GOLD, '|γ₄| хвости'),
                           (D_.g5, GRN, '|γ₅|')):
            ax.semilogy(arr, color=c, lw=1.8, marker='o', ms=2.5, label=nm)
        ax.axvline(L, color=RED, lw=1.2, alpha=.7)
        ax.set_ylim(1e-4, 2.0)          # шар 0 = машинний нуль, не розтягувати
        ax.text(0.3, 1.5e-4, 'шар 0: РІВНО гаус\n(вхід не складений)',
                color=MUT, fontsize=7)
        ax.set_xlabel('шар', color=MUT, fontsize=8)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # ── 4. стеля vs порядок + S/N
        ax = self.fig.add_subplot(224)
        self._ax(ax, 'СТЕЛЯ КЛАСУ (24 мережі) і S/N на РОБОЧІЙ вибірці')
        # 24 мережі × 4 жереби, крос-істина (`field.py`, лог `field24k5.log`)
        ceil = np.array([1.116, 0.2027, 0.0496, 0.02558])     # % відн. похибки
        gain = ('—', '0.8×', '12.7×', '47.6×')                # запас до нашого MSE
        xs = np.arange(4)
        ax.bar(xs, ceil, color=[c for c in self.COLS], alpha=.85)
        for x, y, g in zip(xs, ceil, gain):
            ax.text(x, y * 1.18, f'{y:.4f}%', color=FG, ha='center', fontsize=8)
            ax.text(x, y * 0.72, g, color=BG, ha='center', fontsize=9,
                    fontweight='bold')
        ax.axhline(0.177, color=RED, ls='--', lw=1.4, label='наш MSE = 0.177%')
        ax.axhline(0.02, color=VIO, ls=':', lw=1.6, label='ЛІДЕР = 0.02%')
        ax.set_yscale('log'); ax.set_xticks(xs)
        ax.set_xticklabels(['гаус', '+κ₃', '+κ₄', '+κ₅'], color=MUT, fontsize=9)
        ax.set_ylabel('відносна похибка', color=MUT, fontsize=8)
        sn = D_.snr
        ax.set_ylim(0.012, 3.0)
        ax.text(.02, .90, f'S/N цієї мережі:  κ₃ {sn[0]:.1f}   κ₄ {sn[1]:.1f}   '
                          f'κ₅ {sn[2]:.2f}   (треба >1)',
                transform=ax.transAxes, color=GOLD, fontsize=9)
        ax.legend(fontsize=7, loc='upper right', facecolor=BG,
                  labelcolor=FG, edgecolor='#30363d')

        self.fig.tight_layout()
        self.cv.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Win(); w.show()
    sys.exit(app.exec_())
