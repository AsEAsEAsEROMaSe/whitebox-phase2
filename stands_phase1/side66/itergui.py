"""ІНТЕРАКТИВНИЙ СТЕНД: ЯК ІТЕРАЦІЯ СХОДИТЬСЯ ДО АТРАКТОРА (С79).

Юзер: «візуалізуй прохід і те, як криві йдуть або сходяться».

Чотири панелі — і разом вони показують увесь життєвий цикл сигналу:

  1. ЗБІЖНІСТЬ — `cos(x_t, x*)` для сотень різних стартів. Усі криві падають на
     одну: скільки б стартів не брати, вони приходять в ОДНЕ місце.
  2. ТРАЄКТОРІЇ — ті самі шляхи, спроєктовані у площину (PCA навколо атрактора).
     Видно ГЕОМЕТРІЮ сходження, а не лише швидкість.
  3. ПРОХІД КРІЗЬ 32 ШАРИ — як змінюється норма всередині ОДНОГО проходу, для
     гаусового входу і для самого атрактора. Це два різні режими однієї мережі.
  4. НОРМА ПО ІТЕРАЦІЯХ — геометричне згасання `λ^t` у лог-шкалі.

🔑 ГОЛОВНЕ, ЩО ТУТ ВИДНО: мережа критична на ГАУСОВОМУ вході (норма зберігається)
   і згасальна на ВЛАСНОМУ виході (`λ≈0.78`), бо після першого ж проходу вхід
   стає невід'ємним. Той самий оператор, два різні режими.

⛔ ПЛОМБИ в заголовку: `cos(f(x*), x*)` (мусить бути 1), мінімум виходу (≥0),
   і `cos(x*, E[o])` — наскільки атрактор збігається з тим, що ми РАХУЄМО.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/itergui.py [cuda:N]
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
NSTART, ITERS = 240, 45
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
NNET = int(WZ.shape[0])


def _f(W, x):
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x


def _pass_norms(W, x):
    """Норма ‖x‖ пошарово всередині ОДНОГО проходу (нормовано на старт)."""
    out = [1.0]
    n0 = float(x.norm())
    for L in range(D):
        x = torch.relu(x @ W[L])
        out.append(float(x.norm()) / max(n0, 1e-30))
    return np.array(out)


class Data:
    def __init__(self, net):
        self.net = net
        W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
             for k in range(D)]
        g = torch.Generator(device=DEV).manual_seed(1200 + 29 * net)
        with torch.no_grad():
            x = torch.randn(NSTART, W_, device=DEV, generator=g)
            x = x / x.norm(dim=1, keepdim=True)
            traj, lam = [x.clone()], []
            for _ in range(ITERS):
                y = _f(W, x)
                n = y.norm(dim=1, keepdim=True).clamp_min(1e-30)
                lam.append(n.squeeze(1).clone())
                x = y / n
                traj.append(x.clone())
            self.X = torch.stack(traj)                 # (T+1, NSTART, 256)
            self.LAM = torch.stack(lam).cpu().numpy()  # (T, NSTART)
            xs = x[0:1]
            self.xstar = xs.squeeze(0).cpu().numpy()
            fx = _f(W, xs)
            self.seal_fix = float((fx / fx.norm() * xs).sum())
            self.seal_neg = float(x.min())
            # істинне E[o] — для порівняння з атрактором
            acc, done, N = torch.zeros(W_, dtype=torch.float64, device=DEV), 0, 1 << 15
            gg = torch.Generator(device=DEV).manual_seed(500 + net)
            while done < N:
                b = min(1 << 13, N - done)
                z = torch.randn(b, W_, device=DEV, generator=gg)
                acc += _f(W, torch.cat([z, -z], 0)).double().sum(0)
                done += b
            T = (acc / (2 * N)).float()
            a, t = xs.squeeze(0), T
            ac, tc = a - a.mean(), t - t.mean()
            self.cos_tgt = float((ac * tc).sum() / (ac.norm() * tc.norm()))
            # прохід крізь шари: гаусів вхід проти атрактора
            zg = torch.randn(1, W_, device=DEV, generator=g)
            self.np_gauss = _pass_norms(W, zg / zg.norm())
            self.np_attr = _pass_norms(W, xs)
        # cos із фінальним напрямком по кроках
        xf = self.X[-1]
        self.COS = (self.X * xf[None]).sum(2).cpu().numpy()      # (T+1, NSTART)
        # PCA навколо атрактора для панелі траєкторій
        A = (self.X - xf[None]).reshape(-1, W_).cpu().numpy()
        U, S, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
        self.P = (A - A.mean(0)) @ Vt[:2].T
        self.P = self.P.reshape(self.X.shape[0], NSTART, 2)
        del W
        torch.cuda.empty_cache()


class Win(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ітерація мережі: як криві сходяться до атрактора')
        self.setStyleSheet(f'background:{BG};color:{FG}')
        self.net = 0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.D = Data(self.net)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.fig = Figure(figsize=(15.5, 8.8), facecolor=BG)
        self.cv = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.cv)
        row = QtWidgets.QHBoxLayout()
        self.lab = QtWidgets.QLabel()
        self.lab.setStyleSheet(f'color:{MUT};font-size:11px')
        row.addWidget(self.lab)
        row.addStretch(1)
        b = QtWidgets.QPushButton('інша мережа')
        b.setStyleSheet('background:#21262d;padding:5px 12px')
        b.clicked.connect(self._next)
        row.addWidget(b)
        lay.addLayout(row)
        self._draw()

    def _next(self):
        self.net = (self.net + 1) % NNET
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.D = Data(self.net)
        QtWidgets.QApplication.restoreOverrideCursor()
        self._draw()

    def _ax(self, gs, ttl):
        ax = self.fig.add_subplot(gs)
        ax.set_facecolor(BG)
        ax.set_title(ttl, color=FG, fontsize=9.5)
        ax.tick_params(colors=MUT, labelsize=8)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.grid(alpha=.15, color=MUT)
        return ax

    def _draw(self):
        d = self.D
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, hspace=.32, wspace=.20,
                                   left=.055, right=.985, top=.91, bottom=.09)
        T = d.COS.shape[0]
        steps = np.arange(T)

        # 1. ЗБІЖНІСТЬ
        ax = self._ax(gs[0, 0], f'ЗБІЖНІСТЬ: {NSTART} різних стартів, '
                                'усі падають на одну криву')
        gap = np.clip(1.0 - d.COS, 1e-9, None)
        for k in range(0, NSTART, 2):
            ax.semilogy(steps, gap[:, k], color=CYAN, lw=.5, alpha=.13)
        ax.semilogy(steps, np.median(gap, 1), color=GOLD, lw=2.2,
                    label='медіана')
        ax.set_xlabel('номер ітерації (один прохід = 32 шари)', color=MUT,
                      fontsize=8)
        ax.set_ylabel('1 − cos(x_t, x*)   ⟵ відстань до атрактора', color=MUT,
                      fontsize=8)
        ax.axhline(1e-7, color=GRN, ls=':', lw=1.1,
                   label='межа fp32 — нижче вже шум округлення')
        ax.legend(fontsize=7.5, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # 2. ТРАЄКТОРІЇ
        ax = self._ax(gs[0, 1], 'ТРАЄКТОРІЇ у площині (PCA навколо атрактора) — '
                                'геометрія сходження')
        for k in range(0, NSTART, 2):
            ax.plot(d.P[:, k, 0], d.P[:, k, 1], color=VIO, lw=.5, alpha=.28)
        ax.plot(d.P[0, ::2, 0], d.P[0, ::2, 1], 'o', color=CYAN, ms=2.4,
                label='старти')
        ax.plot([0], [0], '*', color=RED, ms=17, label='АТРАКТОР  x*')
        ax.set_xlabel('гол. компонента 1', color=MUT, fontsize=8)
        ax.set_ylabel('гол. компонента 2', color=MUT, fontsize=8)
        ax.legend(fontsize=7.5, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # 3. ПРОХІД КРІЗЬ ШАРИ
        ax = self._ax(gs[1, 0], 'ПРОХІД КРІЗЬ 32 ШАРИ: два режими одного оператора')
        L = np.arange(D + 1)
        ax.semilogy(L, d.np_gauss, color=GRN, lw=2, marker='o', ms=3,
                    label='ГАУСІВ вхід (наша задача)')
        ax.semilogy(L, d.np_attr, color=GOLD, lw=2, marker='s', ms=3,
                    label='АТРАКТОР (невід\'ємний вхід)')
        ax.axhline(1.0, color=RED, ls='--', lw=1, alpha=.7)
        ax.set_xlabel('шар', color=MUT, fontsize=8)
        ax.set_ylabel('‖x‖ / ‖x₀‖', color=MUT, fontsize=8)
        ax.legend(fontsize=7.5, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # 4. НОРМА ПО ІТЕРАЦІЯХ
        lam = d.LAM
        lm0 = float(np.median(lam[-1]))
        # 🚨 ПІДПИС МУСИТЬ ЧИТАТИ ЗМІРЯНЕ: λ гуляє 0.33…1.82 між мережами, і
        #    жорстке «згасає» БРЕХАЛО на мережах із λ>1 (там сигнал росте).
        word = 'ЗГАСАЄ' if lm0 < 1 else 'РОСТЕ'
        ax = self._ax(gs[1, 1], f'РОЗТЯГ λ за ітерацію: сигнал {word} геометрично')
        ax.plot(np.arange(1, lam.shape[0] + 1), np.median(lam, 1), color=CYAN,
                lw=2, marker='o', ms=3, label='λ = ‖f(x)‖/‖x‖')
        ax.fill_between(np.arange(1, lam.shape[0] + 1), lam.min(1), lam.max(1),
                        color=CYAN, alpha=.15)
        ax.axhline(1.0, color=RED, ls='--', lw=1.2, label='λ=1 (без згасання)')
        lm = lm0
        ax.set_xlabel('номер ітерації', color=MUT, fontsize=8)
        ax.set_ylabel('λ', color=MUT, fontsize=8)
        tail = (f'за 10 проходів лишається {lm**10*100:.1f}%' if lm < 1
                else f'за 10 проходів зростає у {lm**10:.0f}×')
        ax.text(.42, .14, f'λ = {lm:.4f}\n{tail}',
                transform=ax.transAxes, color=GOLD, fontsize=9.5)
        ax.legend(fontsize=7.5, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        self.lab.setText(
            f'мережа {self.net}/{NNET}   ·   ПЛОМБИ: cos(f(x*),x*) = '
            f'{d.seal_fix:.6f} (треба 1)   ·   min виходу = {d.seal_neg:.1e} '
            f'(треба ≥0)   ·   cos(x*, E[o]) центр. = {d.cos_tgt:.4f}')
        self.cv.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Win()
    w.resize(1560, 940)
    w.show()
    sys.exit(app.exec_())
