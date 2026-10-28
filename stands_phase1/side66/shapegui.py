"""🖥️ ФОРМА РОЗПОДІЛУ `h` ПО ШАРАХ — PyQt + matplotlib. (С86)

## ЗВІДКИ — ЗАПИТ ЮЗЕРА
«Візуалізуй форму для мене на кожному слої» + «в pyqt вбудуй матплотліб і
кнопки генерувати нові мережі».

## ЩО ВИДНО
Уся наша похибка сидить в одному місці:
    `E[|h|] = σ·√(2/π) · (1 − γ₄/24 + …)`
перший множник ТОЧНИЙ (з `WᵀCW`), дужка — невідома ФОРМА. Тут вона напряму.

  ①  `p(t)` проти `φ(t)`, лог-шкала — хвости, де все й вирішується
  ②  `p(t) − φ(t)` — власне ФОРМА, очищена від гаусової частини
  ③  `γ₃` (зі знаком і модуль) та `γ₄` по 32 шарах
  ④  похибка `E|t|` у % — ЗМІРЯНА проти передбачення Еджворта `−γ₄/24`

🔑 Кожен нейрон стандартизується ОКРЕМО (`t = (h_i−μ_i)/σ_i`) — інакше суміш
256 різних нормувань дасть артефакт змішування, а не форму.

⛔ПЛОМБА: шар 1 (`h₁ = zᵀW₀`) СТРОГО гаусів ⟹ `p−φ ≈ 0`, `γ₃=γ₄=0`.
   Усе, що там видно, — шум приладу, і це масштабна лінійка для решти шарів.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/shapegui.py [cuda:N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
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
import jitanim                                                    # noqa: E402

BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
RED, CYAN, GOLD, GRN = '#ff2d55', '#00e5ff', '#f0c674', '#3fb950'
L_, W_ = 32, 256
NB, LO, HI = 320, -6.0, 6.0
EDGE = torch.linspace(LO, HI, NB + 1)
CTR = ((EDGE[1:] + EDGE[:-1]) * 0.5).numpy()
DT = float(EDGE[1] - EDGE[0])
PHI = np.exp(-CTR ** 2 / 2) / np.sqrt(2 * np.pi)
E_G = float(np.sqrt(2 / np.pi))                     # E|t| для гаусової


def collect_shape(W, dev, ns, batch=1 << 14, seed=4242, cb=None):
    """Два проходи: (1) μ,σ понейронно, (2) гістограма `t` і κ₃,κ₄."""
    ed = EDGE.to(dev)
    M1 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    M2 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            hd = h.double()
            M1[i] += hd.sum(0); M2[i] += (hd * hd).sum(0)
            a = torch.relu(h)
        done += b
        if cb:
            cb(f'прохід 1/2 · {100*done//ns}%')
    MU = [m / ns for m in M1]
    SG = [(m2 / ns - m * m).clamp_min(1e-30).sqrt() for m, m2 in zip(MU, M2)]

    HIS = [torch.zeros(NB, dtype=torch.float64, device=dev) for _ in range(L_)]
    C3 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    C4 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            t = (h.double() - MU[i]) / SG[i]        # 🔑 ПОНЕЙРОННО
            t2 = t * t
            C3[i] += (t2 * t).sum(0); C4[i] += (t2 * t2).sum(0)
            idx = torch.bucketize(t.reshape(-1), ed) - 1
            ok = (idx >= 0) & (idx < NB)
            HIS[i] += torch.bincount(idx[ok], minlength=NB).double()
            a = torch.relu(h)
        done += b
        if cb:
            cb(f'прохід 2/2 · {100*done//ns}%')
    tot = float(ns) * W_
    P = np.stack([(h / tot / DT).cpu().numpy() for h in HIS])
    G3 = np.stack([(c / ns).cpu().numpy() for c in C3])
    G4 = np.stack([((c / ns) - 3.0).cpu().numpy() for c in C4])
    # похибка E|t|: зміряна проти Еджвортового −γ₄/24
    EM = np.array([(np.abs(CTR) * P[i]).sum() * DT for i in range(L_)])
    return dict(P=P, G3=G3, G4=G4, EM=(EM / E_G - 1.0) * 100.0,
                ED=(-G4.mean(1) / 24.0) * 100.0)


class Win(QtWidgets.QMainWindow):
    def __init__(self, dev='cuda:1'):
        super().__init__()
        self.dev = dev
        self.D = None
        self.seed = 0
        self.setWindowTitle('Форма розподілу h по шарах — де живе вся похибка')
        self.resize(1560, 900)

        self.fig = Figure(figsize=(15.4, 7.2), facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self._axes()

        self.b_new = QtWidgets.QPushButton('🎲 НОВА МЕРЕЖА')
        self.b_bank = QtWidgets.QPushButton('з банку №')
        self.sp = QtWidgets.QSpinBox(); self.sp.setRange(0, 99)
        self.lg = QtWidgets.QSpinBox()
        self.lg.setRange(14, 22); self.lg.setValue(18)
        self.lg.setPrefix('вибірка 2^')
        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setRange(1, L_); self.sl.setValue(L_)
        self.lab = QtWidgets.QLabel('шар 32')
        self.lab.setMinimumWidth(70)
        self.msg = QtWidgets.QLabel('натисни «НОВА МЕРЕЖА» або «з банку»')

        for w in (self.b_new, self.b_bank, self.sp, self.lg, self.lab,
                  self.msg):
            w.setStyleSheet(f'color:{FG};background:#161b22;padding:4px;'
                            f'border:1px solid #30363d;border-radius:4px')
        bar = QtWidgets.QHBoxLayout()
        for w in (self.b_new, self.b_bank, self.sp, self.lg, self.lab):
            bar.addWidget(w)
        bar.addWidget(self.msg, 1)
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(bar)
        box.setStyleSheet(f'background:{BG}')
        self.setCentralWidget(box)

        self.b_new.clicked.connect(self._new)
        self.b_bank.clicked.connect(self._bank)
        self.sl.valueChanged.connect(self._scrub)
        lay.addWidget(self.sl)

    def _axes(self):
        gs = self.fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22,
                                   left=0.06, right=0.985,
                                   top=0.93, bottom=0.09)
        self.a1 = self.fig.add_subplot(gs[0, 0])
        self.a2 = self.fig.add_subplot(gs[0, 1])
        self.a3 = self.fig.add_subplot(gs[1, 0])
        self.a4 = self.fig.add_subplot(gs[1, 1])
        ttl = ('① щільність p(t) проти гаусової φ(t)',
               '② ФОРМА: p(t) − φ(t)',
               '③ негаусовість по шарах',
               '④ похибка E|t|, % — зміряна проти Еджворта')
        for ax, t in zip((self.a1, self.a2, self.a3, self.a4), ttl):
            ax.set_facecolor('#010409')
            ax.set_title(t, color=FG, fontsize=10.5, loc='left')
            for s in ax.spines.values():
                s.set_color('#30363d')
            ax.tick_params(colors=MUT, labelsize=8)
            ax.grid(alpha=0.13, color=MUT, lw=0.5)

    def _say(self, m):
        self.msg.setText(m)
        QtWidgets.QApplication.processEvents()

    def _compute(self, W, tag):
        for b in (self.b_new, self.b_bank):
            b.setEnabled(False)
        try:
            self.D = collect_shape(W, self.dev, 1 << self.lg.value(),
                                   seed=4242 + self.seed, cb=self._say)
            self.tag = tag
            self._draw()
            self._say(f'{tag} · готово')
        except Exception as e:                              # noqa: BLE001
            self._say(f'🛑 {type(e).__name__}: {e}')
        finally:
            for b in (self.b_new, self.b_bank):
                b.setEnabled(True)

    def _new(self):
        self.seed += 1
        W = jitanim.gen_net(10_000 + 7919 * self.seed, self.dev)
        self._compute(W, f'🎲 НОВА мережа #{self.seed}')

    def _bank(self):
        n = self.sp.value()
        self._compute(jitanim.bank_net(n, self.dev), f'банк №{n}')

    def _scrub(self, v):
        self.lab.setText(f'шар {v}')
        if self.D is not None:
            self._draw()

    def _draw(self):
        D, k = self.D, self.sl.value() - 1
        for ax in (self.a1, self.a2, self.a3, self.a4):
            ax.clear()
        self._axes_style()
        p = D['P'][k]

        # ① щільність, лог — видно хвости
        self.a1.semilogy(CTR, np.maximum(PHI, 1e-9), color=MUT, lw=1.6,
                         label='гаусова φ(t)')
        self.a1.semilogy(CTR, np.maximum(p, 1e-9), color=CYAN, lw=1.4,
                         label=f'зміряна, шар {k+1}')
        self.a1.set_ylim(1e-7, 1.0); self.a1.set_xlim(LO, HI)
        self.a1.set_xlabel('t = (h − μ)/σ', color=MUT, fontsize=9)
        self.a1.legend(facecolor='#161b22', edgecolor='#30363d',
                       labelcolor=FG, fontsize=8.5)

        # ② форма
        for j, c in ((0, '#2d3742'), (7, '#44506080')):
            if j != k:
                self.a2.plot(CTR, D['P'][j] - PHI, color=c, lw=1.0)
        self.a2.axhline(0, color=MUT, lw=0.8)
        self.a2.plot(CTR, p - PHI, color=RED, lw=1.8)
        self.a2.fill_between(CTR, 0, p - PHI, color=RED, alpha=0.16)
        self.a2.set_xlim(-4.5, 4.5)
        self.a2.set_xlabel('t   (сірим — шари 1 і 8 для масштабу)',
                           color=MUT, fontsize=9)

        # ③ негаусовість по шарах
        x = np.arange(1, L_ + 1)
        self.a3.plot(x, D['G4'].mean(1), color=GOLD, lw=1.8, marker='o',
                     ms=3, label='γ₄ (ексцес) — ВИННИЙ')
        self.a3.plot(x, np.abs(D['G3']).mean(1), color=CYAN, lw=1.3,
                     ls='--', label='⟨|γ₃|⟩ модуль')
        self.a3.plot(x, D['G3'].mean(1), color=RED, lw=1.6,
                     label='⟨γ₃⟩ ЗІ ЗНАКОМ ≈ 0')
        self.a3.axvline(k + 1, color=GRN, lw=1.0, alpha=0.55)
        self.a3.axhline(0, color=MUT, lw=0.8)
        self.a3.set_xlabel('шар', color=MUT, fontsize=9)
        self.a3.legend(facecolor='#161b22', edgecolor='#30363d',
                       labelcolor=FG, fontsize=8.5, loc='upper left')

        # ④ похибка E|t| проти Еджворта
        self.a4.plot(x, D['EM'], color=RED, lw=2.0, marker='o', ms=3,
                     label='зміряна похибка E|t|')
        self.a4.plot(x, D['ED'], color=GOLD, lw=1.5, ls='--',
                     label='Еджворт −γ₄/24')
        self.a4.axvline(k + 1, color=GRN, lw=1.0, alpha=0.55)
        self.a4.axhline(0, color=MUT, lw=0.8)
        self.a4.set_xlabel('шар', color=MUT, fontsize=9)
        self.a4.set_ylabel('%', color=MUT, fontsize=9)
        self.a4.legend(facecolor='#161b22', edgecolor='#30363d',
                       labelcolor=FG, fontsize=8.5, loc='lower left')
        self.a4.text(0.98, 0.95,
                     f'шар {k+1}:  зміряно {D["EM"][k]:+.3f}%   '
                     f'Еджворт {D["ED"][k]:+.3f}%',
                     transform=self.a4.transAxes, ha='right', va='top',
                     color=FG, fontsize=9.5)
        self.fig.suptitle(
            f'{self.tag}   ·   вибірка 2^{self.lg.value()}   ·   '
            f'⛔пломба шар 1: γ₄ = {self.D["G4"][0].mean():+.4f} (нуль)',
            color=FG, fontsize=11)
        self.canvas.draw_idle()

    def _axes_style(self):
        ttl = ('① щільність p(t) проти гаусової φ(t)',
               '② ФОРМА: p(t) − φ(t)',
               '③ негаусовість по шарах',
               '④ похибка E|t|, % — зміряна проти Еджворта')
        for ax, t in zip((self.a1, self.a2, self.a3, self.a4), ttl):
            ax.set_facecolor('#010409')
            ax.set_title(t, color=FG, fontsize=10.5, loc='left')
            for s in ax.spines.values():
                s.set_color('#30363d')
            ax.tick_params(colors=MUT, labelsize=8)
            ax.grid(alpha=0.13, color=MUT, lw=0.5)


if __name__ == '__main__':
    dv = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
    assert dv not in ('cuda:0', 'cuda:4'), '🚫 cuda:0 ТАБУ, cuda:4 — монітор'
    app = QtWidgets.QApplication(sys.argv)
    w = Win(dv)
    w.show()
    sys.exit(app.exec_())
