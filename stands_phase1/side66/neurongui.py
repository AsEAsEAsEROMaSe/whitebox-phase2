"""ІНТЕРАКТИВНИЙ СТЕНД: КРИВА ОДНОГО НЕЙРОНА (С79).

Питання юзера: «хочу побачити криву, яку формує площина на деякому нейроні
при прогоні семплів».

Нейрон — це ГІПЕРПЛОЩИНА: 256 ваг, зсуву НЕМА. Її «криву» видно двома
незалежними способами, і обидва тут одночасно.

  ЗЛІВА — ГЕОМЕТРІЯ. Ведемо семпл по великому колу `z(θ)=√d·(cosθ·u+sinθ·v)`
  і дивимось передактивацію `p_j(θ)`. На ШАРІ 0 це РІВНО синусоїда
  `A cosθ + B sinθ` (бо `p = z·w` лінійне) ⟹ рівно ДВА нулі, хоч як крути.
  Глибше ReLU вже порізав простір на конуси, і та сама площина дає
  КУСКОВО-синусоїдальну ЛАМАНУ: кожен злам — це місце, де якийсь нейрон
  попереднього шару перемкнувся.

  ЗПРАВА — СТАТИСТИКА. Та сама площина, але на iid-вибірці: щільність `p_j`.
  Це і є «поле», яке наше замикання носить як `(m, C)`. Видно перекіс `γ₃`,
  через який гаусова формула `m·Φ(t)+s·φ(t)` промахується.

🔑 ЗВ'ЯЗОК МІЖ ПАНЕЛЯМИ: заштрихована частина кола (`p>0`) — це рівно ворота
   `q = P(p>0)`, а вони ж є заштрихованою площею під щільністю. Ліва панель
   показує ДЕ нейрон відкритий, права — ЯК ЧАСТО.

🚨 ПЛОМБИ, ВБУДОВАНІ В СТЕНД (друкуються в заголовку; якщо не збігається —
   зламаний СТЕНД, а не мережа цікава):
   * шар 0, число нулів на колі = РІВНО 2 (теорема, не вимір);
   * шар 0, ворота `q = 0.500` (бо `m = 0` строго);
   * сума гістограми = 1.000.

Запуск:  /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/neurongui.py [cuda:N]
"""
import math
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
NB = 81                        # бінів гістограми
LO, HI = -4.5, 4.5             # у одиницях (p−m)/s
# 🚨 РОЗДІЛЬНА ЗДАТНІСТЬ. С63 зміряла ~450–550 зламів на коло НА ШАР ⟹ на шарі
#    31 їх ~14 000. З 6000 точок вони НЕ роздільні (я так і зробив спершу).
#    Тому: багато точок + окреме вікно зуму на 1/60 кола.
NTH = 24000                    # точок на великому колі
ZOOM = 60                      # у скільки разів звужене вікно вставки
S2PI = float(1.0 / np.sqrt(2 * np.pi))
SQ2 = float(np.sqrt(2.0))
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
NNET = int(WZ.shape[0])


# ─────────────────────────── рахунок ───────────────────────────
def _scan(W, seed, npairs):
    """m, s, γ₃, ворота q, точне E[relu] — для ВСІХ шарів × нейронів.

    Антитетика (`z` і `−z`) ⟹ усі непарні моменти ВХОДУ занулено точно.
    🚨 матмули fp32 (на P104 fp64 у 32× повільніше), накопичення fp64.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = torch.zeros(D, 5, W_, dtype=torch.float64, device=DEV)
    done, CH = 0, 1 << 13
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            pd = p.double()
            Z[L][0] += pd.sum(0)
            Z[L][1] += (pd * pd).sum(0)
            Z[L][2] += (pd * pd * pd).sum(0)
            Z[L][3] += (p > 0).double().sum(0)
            x = torch.relu(p)
            Z[L][4] += x.double().sum(0)
        done += b
    N = float(2 * npairs)
    a1, a2, a3 = Z[:, 0] / N, Z[:, 1] / N, Z[:, 2] / N
    q, er = Z[:, 3] / N, Z[:, 4] / N
    s = (a2 - a1 * a1).clamp_min(1e-30).sqrt()
    mu3 = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
    return a1, s, mu3 / s ** 3, q, er


def _hist(W, seed, npairs, m, s):
    """Гістограми `(p−m)/s` для ВСІХ шарів × нейронів: (D, 256, NB)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    H = torch.zeros(D, W_, NB, dtype=torch.float32, device=DEV)
    mf = [m[L].float() for L in range(D)]
    sf = [s[L].float() for L in range(D)]
    done, CH = 0, 1 << 13
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            u = (p - mf[L]) / sf[L]
            idx = ((u - LO) / (HI - LO) * (NB - 1)).round().long().clamp(0, NB - 1)
            H[L].scatter_add_(1, idx.T.contiguous(),
                              torch.ones(W_, u.shape[0], device=DEV))
            x = torch.relu(p)
        done += b
    return H / H.sum(2, keepdim=True).clamp_min(1.0)


def _circle(W, seed, L):
    """`p(θ)` вздовж ОДНОГО великого кола, лише для шару `L`: (NTH, 256).

    🚨 Радіус √d: саме там живе гаусова маса (`E‖z‖ = 15.984` при d=256).
       Через позитивну однорідність радіус не змінює ФОРМУ — лише масштаб.
    🔑 Тримати всі 32 шари одразу = 786 МБ; прогін же коштує 3.4e10 флопів
       (мить на GPU) ⟹ дешевше ПЕРЕРАХУВАТИ на зміну шару, ніж зберігати.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    u = torch.randn(W_, device=DEV, generator=g)
    u = u / u.norm()
    v = torch.randn(W_, device=DEV, generator=g)
    v = v - (v @ u) * u
    v = v / v.norm()
    th = torch.linspace(0.0, 2.0 * np.pi, NTH, device=DEV)
    x = float(np.sqrt(W_)) * (torch.cos(th)[:, None] * u[None, :]
                              + torch.sin(th)[:, None] * v[None, :])
    # 🔑 ЗЛАМИ РАХУЄМО ТОЧНО, А НЕ З ФОРМИ КРИВОЇ. Крива нейрона шару L
    #    ламається РІВНО там, де якийсь нейрон шарів <L перемкнув знак.
    #    Це підрахунок без порогів і без вільних параметрів.
    sw = torch.zeros(NTH - 1, dtype=torch.bool, device=DEV)
    for i in range(L + 1):
        p = x @ W[i]
        if i < L:
            g = p > 0
            sw |= (g[1:] != g[:-1]).any(1)
        x = torch.relu(p)
    return th.cpu().numpy(), p, int(sw.sum())


def _zeros_on_circle(y):
    """Скільки разів крива перетинає нуль (замкнене коло ⟹ завжди парне)."""
    sg = np.sign(y)
    sg[sg == 0] = 1.0
    return int(np.count_nonzero(np.diff(sg)))


# 🚨 ЧОМУ НЕ «ПЕРЕТИНИ НУЛЯ» ЯК МІРА ЛАМАНОСТІ: на глибині більшість нейронів
#    РІДКІСНІ (шар 31 — 56.6% мають `w<0.005`), тобто на цьому колі не
#    перетинають нуль ЖОДНОГО разу. Нуль перетинів там означає «завжди
#    закритий», а не «гладкий».
# ⛔ І ЧОМУ НЕ ДРУГА РІЗНИЦЯ (моя перша спроба, ПРОВАЛИЛА ВЛАСНУ ПЛОМБУ):
#    у fp32 `d2` гладкої синусоїди ~|y|·Δθ² = 7e-8 ТОНЕ в шумі округлення
#    ~|y|·1e-7 ⟹ поріг «6× медіани» дав 280 хибних зламів там, де теорія
#    вимагає рівно 0. Тому зламки рахуються в `_circle` ТОЧНО — по
#    перемиканнях нейронів попередніх шарів.


class Data:
    def __init__(self, net, logn=15, cseed=0):
        self.net = net
        self.cseed = cseed
        self.W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
                  for k in range(D)]
        with torch.no_grad():
            m, s, g3, q, er = _scan(self.W, 1000 + 7 * net, 1 << logn)
            self.H = _hist(self.W, 4000 + 11 * net, 1 << (logn - 1), m, s)
            self.th, P0, k0 = _circle(self.W, self._cs(), 0)
        self.m = m.cpu().numpy()
        self.s = s.cpu().numpy()
        self.g3 = g3.cpu().numpy()
        self.q = q.cpu().numpy()
        self.er = er.cpu().numpy()
        self._cache = {0: (P0.cpu().numpy(), k0)}
        # ⛔ПЛОМБИ: шар 0 — рівно 2 нулі на колі (теорема), ворота рівно 0.5
        self.seal_z = max(_zeros_on_circle(self._cache[0][0][:, j])
                          for j in range(0, W_, 17))
        self.seal_k = k0
        self.seal_q = float(np.abs(self.q[0] - 0.5).max())

    def _cs(self):
        return 900 + 13 * self.net + 101 * self.cseed

    def curve(self, L, j):
        """Крива шару `L` і число її зламів; шар рахується на вимогу."""
        if L not in self._cache:
            with torch.no_grad():
                _, P, k = _circle(self.W, self._cs(), L)
            self._cache[L] = (P.cpu().numpy(), k)
        P, k = self._cache[L]
        return P[:, j], k


# ─────────────────────────── вікно ───────────────────────────
class Win(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Крива одного нейрона — площина, що ріже потік')
        self.setStyleSheet(f'background:{BG};color:{FG}')
        self.net, self.cseed = 0, 0
        self.L, self.j = 8, 0
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

        row.addWidget(QtWidgets.QLabel('шар'))
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setRange(0, D - 1)
        self.sl.setValue(self.L)
        self.sl.setFixedWidth(230)
        self.sl.valueChanged.connect(self._set_L)
        row.addWidget(self.sl)

        row.addWidget(QtWidgets.QLabel('нейрон'))
        self.sn = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sn.setRange(0, W_ - 1)
        self.sn.setValue(self.j)
        self.sn.setFixedWidth(230)
        self.sn.valueChanged.connect(self._set_j)
        row.addWidget(self.sn)

        for txt, fn in (('найрідкісніший', self._rare),
                        ('найперекошеніший', self._skew),
                        ('інше коло', self._newcirc),
                        ('інша мережа', self._newnet)):
            b = QtWidgets.QPushButton(txt)
            b.setStyleSheet('background:#21262d;padding:5px 10px')
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self._draw()

    # ── керування ──
    def _set_L(self, v):
        self.L = v
        self._draw()

    def _set_j(self, v):
        self.j = v
        self._draw()

    def _rare(self):
        w = np.minimum(self.D.q[self.L], 1.0 - self.D.q[self.L])
        self.sn.setValue(int(np.argmin(w)))

    def _skew(self):
        self.sn.setValue(int(np.argmax(np.abs(self.D.g3[self.L]))))

    def _reload(self, **kw):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.D = Data(self.net, cseed=self.cseed)
        QtWidgets.QApplication.restoreOverrideCursor()
        self._draw()

    def _newcirc(self):
        self.cseed += 1
        self._reload()

    def _newnet(self):
        self.net = (self.net + 1) % NNET
        self._reload()

    # ── малювання ──
    def _ax(self, gs, ttl):
        ax = self.fig.add_subplot(gs)
        ax.set_facecolor(BG)
        ax.set_title(ttl, color=FG, fontsize=9)
        ax.tick_params(colors=MUT, labelsize=8)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.grid(alpha=.15, color=MUT)
        return ax

    def _draw(self):
        L, j = self.L, self.j
        d, th = self.D, self.D.th
        y, nk = d.curve(L, j)
        m, s = float(d.m[L, j]), float(d.s[L, j])
        t = m / s
        g3, q, er = float(d.g3[L, j]), float(d.q[L, j]), float(d.er[L, j])
        nz = _zeros_on_circle(y)

        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3, hspace=.34, wspace=.24,
                                   left=.05, right=.985, top=.90, bottom=.08)

        # ── 1. КРИВА ПО КОЛУ ──
        ttl = (f'ГЕОМЕТРІЯ: передактивація $p_{{{j}}}(\\theta)$ вздовж великого '
               f'кола  ·  шар {L}  ·  ЗЛАМІВ {nk}  ·  перетинів нуля {nz}')
        if L == 0:
            ttl += '   ⟵ ПЛОМБА: рівно 2 нулі й 0 зламів (чиста синусоїда)'
        ax = self._ax(gs[0, :2], ttl)
        ax.plot(th, y, color=CYAN, lw=.9)
        ax.fill_between(th, 0, y, where=(y > 0), color=CYAN, alpha=.20,
                        label='нейрон ВІДКРИТИЙ (p>0)')
        ax.axhline(0, color=RED, lw=1.0, alpha=.8)
        ax.axhline(m, color=GOLD, ls='--', lw=1.0, alpha=.8,
                   label=f'середнє m = {m:+.3f}')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xlabel('θ — кут уздовж великого кола, рад', color=MUT, fontsize=8)
        ax.set_ylabel('p', color=MUT, fontsize=8)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor='#30363d',
                  loc='upper right')
        # 🔑 ВСТАВКА: без зуму зламів не видно — на шарі 31 їх ~14 000
        k = NTH // ZOOM
        ins = ax.inset_axes([.045, .045, .38, .33])
        ins.set_facecolor('#161b22')
        ins.plot(th[:k], y[:k], color=GOLD, lw=1.0)
        # 🚨 НУЛЬ МАЛЮВАТИ ТІЛЬКИ ЯКЩО ВІН У КАДРІ. Інакше `axhline(0)` тягне
        #    ylim до нуля, крива стискається в горизонтальну рису, і зламки —
        #    те, заради чого вставка й існує — стають невидимі.
        y0, y1 = float(y[:k].min()), float(y[:k].max())
        if y0 < 0.0 < y1:
            ins.axhline(0, color=RED, lw=.7, alpha=.7)
        pad = .08 * max(y1 - y0, 1e-9)
        ins.set_ylim(y0 - pad, y1 + pad)
        ins.tick_params(colors=MUT, labelsize=6)
        for sp in ins.spines.values():
            sp.set_color(GOLD)
        ins.set_title(f'зум ×{ZOOM} — видно окремі зламки', color=GOLD,
                      fontsize=7)

        # ── 2. ЩІЛЬНІСТЬ ──
        ax = self._ax(gs[0, 2], 'СТАТИСТИКА: щільність $p$ на iid-вибірці')
        h = d.H[L, j].cpu().numpy()
        u = np.linspace(LO, HI, NB)
        ax.bar(u, h, width=(HI - LO) / (NB - 1), color='#21262d',
               edgecolor='#30363d', linewidth=.4, label='справжня')
        uf = np.linspace(LO, HI, 400)
        gau = np.exp(-.5 * uf * uf) * S2PI * (HI - LO) / (NB - 1)
        ax.plot(uf, gau, color=GOLD, lw=1.5, label='гаус (що несе замикання)')
        edg = gau * (1.0 + g3 / 6.0 * (uf ** 3 - 3.0 * uf))
        ax.plot(uf, edg, color=GRN, lw=1.5, ls='--', label='+ перекіс κ₃')
        # поріг ReLU у цих координатах — це −t
        ax.axvline(-t, color=RED, lw=1.2)
        ax.fill_between(uf, 0, np.interp(uf, u, h), where=(uf > -t),
                        color=CYAN, alpha=.18)
        ax.set_xlabel('(p − m)/s        ⟵ червона лінія = поріг ReLU',
                      color=MUT, fontsize=8)
        ax.legend(fontsize=6.5, facecolor=BG, labelcolor=FG,
                  edgecolor='#30363d')

        # ── 3. ЩО ВИХОДИТЬ ПІСЛЯ ReLU ──
        ax = self._ax(gs[1, 0], 'СКЛАДКА: $relu(p)$ — це те, що йде далі')
        r = np.maximum(y, 0.0)
        ax.plot(th, r, color=VIO, lw=1.1)
        ax.fill_between(th, 0, r, color=VIO, alpha=.20)
        ax.axhline(r.mean(), color=CYAN, lw=1.2,
                   label=f'середнє по колу = {r.mean():.4f}')
        ax.axhline(er, color=GRN, ls='--', lw=1.2,
                   label=f'істинне E[relu] = {er:.4f}')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xlabel('θ', color=MUT, fontsize=8)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # ── 4. ВОРОТА ВСЬОГО ШАРУ ──
        ax = self._ax(gs[1, 1], f'ВОРОТА всіх 256 нейронів шару {L}')
        qs = np.sort(d.q[L])
        ax.plot(np.arange(W_), qs, color=CYAN, lw=1.5)
        rk = int(np.searchsorted(qs, q))
        ax.plot([rk], [q], 'o', color=RED, ms=7,
                label=f'нейрон {j}:  q = {q:.4f}')
        ax.axhline(.005, color=GOLD, ls=':', lw=1.0)
        ax.axhline(.995, color=GOLD, ls=':', lw=1.0,
                   label='межа «рідкісних» w=0.005')
        ax.set_ylim(-.02, 1.02)
        ax.set_xlabel('нейрони, впорядковані за q', color=MUT, fontsize=8)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor='#30363d')

        # ── 5. ЩО КАЖЕ ФОРМУЛА ──
        ax = self._ax(gs[1, 2], 'ЧОМУ ФОРМУЛА ПРОМАХУЄТЬСЯ')
        ax.axis('off')
        # 🚨 `np.math.erf` НЕ ІСНУЄ в сучасному numpy (я вже попадався на
        #    `np.math.gamma` у `lineval.py`) ⟹ беремо зі стандартного `math`.
        Ph = .5 * (1.0 + math.erf(t / SQ2))
        ph = float(np.exp(-.5 * t * t) * S2PI)
        gauss = m * Ph + s * ph
        edge = gauss - (g3 * s ** 3) * t * ph / (6.0 * s * s)
        # 🚨 НОРМУВАТИ НА `s`, А НЕ НА `er`. У рідкісного нейрона `E[relu]`
        #    прямує до нуля (тут зміряно рівно 0.000000), і відносна похибка
        #    вибухає в 1e19% — це ділення на нуль, а не результат. Масштаб
        #    передактивації `s` завжди додатний, тож похибка в його одиницях
        #    порівнянна між нейронами будь-якої рідкісності.
        e_g = 100.0 * (gauss - er) / s
        e_e = 100.0 * (edge - er) / s
        w = min(q, 1.0 - q)
        rare = '   ⟵ РІДКІСНИЙ' if w < 0.005 else ''
        txt = (f'$t = m/s$ = {t:+.4f}\n'
               f'ворота $q=\\Phi(t)$ = {q:.4f}   (w = {w:.4f}){rare}\n'
               f'перекіс $\\gamma_3$ = {g3:+.4f}\n'
               f'масштаб $s$ = {s:.4f}\n'
               f'\n'
               f'істинне  E[relu] = {er:.6f}\n'
               f'гаусова формула  = {gauss:.6f}\n'
               f'+ поправка κ₃    = {edge:.6f}\n'
               f'\n'
               f'похибка у частках $s$:\n'
               f'   гаус  {e_g:+.4f}%\n'
               f'   +κ₃   {e_e:+.4f}%\n')
        ax.text(.0, .98, txt, color=FG, fontsize=10, va='top', family='monospace')
        ax.text(.0, .17,
                'Ліва панель — ДЕ нейрон відкритий.\n'
                'Права верхня — ЯК ЧАСТО. Це та сама\nплоща, лише в різних координатах.',
                color=MUT, fontsize=8.5, va='top')

        self.lab.setText(
            f'мережа {self.net}/{NNET}   ·   ⛔ПЛОМБИ: нулів на шарі 0 = '
            f'{self.D.seal_z} (треба 2), зламів {self.D.seal_k} (треба 0)   ·   '
            f'max|q₀−0.5| = {self.D.seal_q:.2e}'
            f'   ·   Σгісто = {float(self.D.H[L, j].sum()):.3f}')
        self.cv.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Win()
    w.resize(1560, 940)
    w.show()
    sys.exit(app.exec_())
