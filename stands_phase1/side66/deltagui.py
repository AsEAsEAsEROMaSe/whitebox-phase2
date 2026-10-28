"""ІНТЕРАКТИВНИЙ СТЕНД: куди вказує `Δ = p₈₇ − p_v2`.

GUI-версія `deltafig.py` — ті самі кнопки, що в `jitgui.py`/`curvegui.py`, щоб
дивитись не на одну мережу, а на скільки завгодно, включно з НОВИМИ (не з банку).

«Вказувати на середнє» означає дивитись рівно ПРОТИ власної похибки `e₈₇`:
щоб поправка зменшила помилку, `Δ` мусить мати компоненту вздовж `−e₈₇`.
    180° — точно НА істину · 90° — нікуди · 0° — точно ВІД істини.

🚨 У 256 вимірах два НЕЗАЛЕЖНІ вектори лежать під 90° ± 3.6°, тож самого піка на
90° мало — поруч завжди малюється ЕТАЛОН із випадкових векторів. Якщо гістограми
збігаються центром, `Δ` не несе напрямку.
🔑 Уже зміряно на 12 мережах: центр 88.8° (еталон 89.8°), АЛЕ розкид 24° проти
3.4° ⟹ **структура Є, знака НЕМА**. Тому в заголовку стоїть `t` для середнього
косинуса: якщо на якійсь мережі `|t| > 3`, у `Δ` там З'ЯВИВСЯ сталий знак —
це був би привід копати.

Запуск:  CUDA_DEVICE_ORDER=PCI_BUS_ID ./torch_venv/bin/python side66/deltagui.py cuda:3
"""
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('QtAgg')

from PyQt5 import QtWidgets                                       # noqa: E402
from matplotlib.backends.backend_qtagg import (                   # noqa: E402
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure                              # noqa: E402

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
import jitanim                                                    # noqa: E402
import curvegui                                                   # noqa: E402
from base_real import cut_and_budget, closure_parts, W_           # noqa: E402

BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
CYAN, GOLD, GRN = '#00e5ff', '#f0c674', '#3fb950'


def collect(W, dev, draws=32, cb=None):
    if cb:
        cb('дві незалежні істини…')
    T1 = curvegui._truth(W, dev, 4_100_001).cpu().numpy()
    T2 = curvegui._truth(W, dev, 8_200_003).cpu().numpy()
    tm = (T1 + T2) / 2.0
    with torch.no_grad():
        wc, npair = cut_and_budget(W, dev)
        cl = closure_parts(wc, 8, dev, extras=True)
        ang, rad, aref, cs = [], [], [], []
        sd_, se_, sv_ = [], [], []
        for r in range(draws):
            if cb and r % 8 == 0:
                cb(f'жереб {r+1}/{draws}…')
            g = torch.Generator(device=dev).manual_seed(51_000 + 977 * r)
            z = torch.randn(npair, W_, device=dev, generator=g)
            p87 = jitanim._run_sub87(wc, npair, cl, dev, 0, z_ext=z).cpu().numpy()
            pv2 = curvegui._v2(W, npair, dev, z).double().cpu().numpy()
            e87, ev2, d = p87 - tm, pv2 - tm, p87 - pv2
            c = float(d @ e87 / (np.linalg.norm(d) * np.linalg.norm(e87)))
            cs.append(c)
            ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            rad.append(float(np.linalg.norm(d) / np.linalg.norm(e87)))
            q = np.random.default_rng(7000 + r).standard_normal(e87.size)
            aref.append(np.degrees(np.arccos(np.clip(
                float(q @ e87 / (np.linalg.norm(q) * np.linalg.norm(e87))), -1, 1))))
            s = np.linalg.norm(e87)
            sd_.append(d / s); se_.append(e87 / s); sv_.append(ev2 / s)
    cs = np.array(cs)
    return dict(ang=np.array(ang), rad=np.array(rad), aref=np.array(aref),
                D=np.concatenate(sd_), E=np.concatenate(se_),
                V=np.concatenate(sv_), cos=cs, npair=npair,
                t=float(cs.mean() / (cs.std(ddof=1) / np.sqrt(len(cs)) + 1e-30)))


class Win(QtWidgets.QMainWindow):
    def __init__(self, dev='cuda:3'):
        super().__init__()
        self.dev = dev
        self.setWindowTitle('Куди вказує Δ = p₈₇ − p_v2')
        self.resize(1600, 920)
        self.fig = Figure(figsize=(15.5, 8.4), facecolor=BG)
        self.canvas = FigureCanvas(self.fig)

        self.b_new = QtWidgets.QPushButton('🎲 НОВА МЕРЕЖА')
        self.b_bank = QtWidgets.QPushButton('з банку №')
        self.sp = QtWidgets.QSpinBox(); self.sp.setRange(0, 99)
        self.nd = QtWidgets.QSpinBox()
        self.nd.setRange(8, 256); self.nd.setValue(32)
        self.nd.setPrefix('жеребів: ')
        self.status = QtWidgets.QLabel('натисни «з банку №» або «НОВА МЕРЕЖА»')
        self.status.setStyleSheet(f'color:{GOLD};font-size:13px')

        bar = QtWidgets.QHBoxLayout()
        for w in (self.b_new, self.b_bank, self.sp, self.nd):
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
        W = (jitanim.bank_net(net, self.dev) if net is not None
             else jitanim.gen_net(np.random.randint(1 << 30), self.dev))
        self.D = collect(W, self.dev, self.nd.value(), cb=self._msg)
        del W
        torch.cuda.empty_cache()
        self._draw(net)

    def _sty(self, ax, t, xl='', yl=''):
        ax.set_facecolor(BG)
        ax.set_title(t, color=FG, fontsize=11, pad=9)
        ax.set_xlabel(xl, color=MUT, fontsize=9)
        ax.set_ylabel(yl, color=MUT, fontsize=9)
        ax.tick_params(colors=MUT, labelsize=8)
        for s in ax.spines.values():
            s.set_color('#30363d')
        ax.grid(alpha=.12, color=FG)

    def _draw(self, net):
        D = self.D
        ang, aref = D['ang'], D['aref']
        r2 = float(np.corrcoef(D['D'], D['E'])[0, 1] ** 2)
        rv = float(np.corrcoef(D['V'], D['E'])[0, 1])
        share = float(np.mean(-np.cos(np.radians(ang))))
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, hspace=.34, wspace=.2)

        a1 = self.fig.add_subplot(gs[0, 0], projection='polar')
        a1.set_facecolor(BG)
        a1.scatter(np.radians(ang), D['rad'], s=14, color=GOLD, alpha=.5,
                   edgecolors='none')
        a1.set_theta_zero_location('E')
        a1.set_thetamin(0); a1.set_thetamax(180)
        a1.set_thetagrids([0, 45, 90, 135, 180],
                          ['0°\nВІД істини', '45°', '90°\nНІКУДИ', '135°',
                           '180°\nНА істину'], color=FG, fontsize=8.5)
        a1.plot([np.pi, np.pi], [0, D['rad'].max()], color=GRN, lw=2.4, alpha=.8)
        a1.plot([np.pi / 2, np.pi / 2], [0, D['rad'].max()], color=MUT, lw=1.3, ls=':')
        a1.plot([np.radians(ang.mean())] * 2, [0, D['rad'].max()], color=CYAN, lw=2)
        a1.set_rlabel_position(112)
        a1.tick_params(colors=MUT, labelsize=7.5)
        a1.grid(alpha=.16, color=FG)
        a1.set_title('КУДИ ВКАЗУЄ Δ · кут до власної похибки, радіус |Δ|/|e₈₇| '
                     '· бірюзова = середнє', color=FG, fontsize=11, pad=17)

        a2 = self.fig.add_subplot(gs[1, 0])
        bins = np.linspace(min(40, ang.min() - 5), max(140, ang.max() + 5), 61)
        a2.hist(aref, bins=bins, color=MUT, alpha=.55,
                label='ЕТАЛОН: випадковий вектор')
        a2.hist(ang, bins=bins, color=GOLD, alpha=.75, label='Δ = p₈₇ − p_v2')
        a2.axvline(90, color=MUT, ls=':', lw=1.3)
        self._sty(a2, f'ГІСТОГРАМА КУТА · Δ ширша за еталон у '
                      f'{ang.std()/max(aref.std(),1e-9):.1f}× — структура Є, знак?',
                  'кут між Δ і e₈₇, градуси', 'жеребів')
        a2.legend(facecolor=BG, edgecolor='#30363d', labelcolor=FG, fontsize=9)

        a3 = self.fig.add_subplot(gs[0, 1])
        a3.scatter(D['E'], D['D'], s=3, color=GOLD, alpha=.16, edgecolors='none')
        self._sty(a3, f'Δ ПРОТИ e₈₇ по координатах · R² = {r2:.4f}',
                  'наша похибка e₈₇ (норм.)', 'Δ (норм.)')

        a4 = self.fig.add_subplot(gs[1, 1])
        a4.scatter(D['E'], D['V'], s=3, color=CYAN, alpha=.16, edgecolors='none')
        self._sty(a4, f'для контрасту: e_v2 ПРОТИ e₈₇ · ρ = {rv:.3f}',
                  'наша похибка e₈₇ (норм.)', 'похибка v2 (норм.)')

        t = D['t']
        self.fig.text(.5, .972, f'Δ під {ang.mean():.1f}° (еталон {aref.mean():.1f}°) '
                      f'· на істину {100*share:+.2f}% довжини Δ · '
                      f't(сер. косинуса) = {t:+.2f}', ha='center',
                      color=(GRN if abs(t) > 3 else GOLD), fontsize=13, weight='bold')
        self.fig.tight_layout()
        self.canvas.draw()
        self._msg(f'мережа {"банк №"+str(net) if net is not None else "нова"} · '
                  f'npair {D["npair"]} · жеребів {len(ang)} · крос-ціль · '
                  f'{"🟢 |t|>3 — У Δ З’ЯВИВСЯ ЗНАК, КОПАТИ" if abs(t) > 3 else "⛔ знака нема"}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Win(sys.argv[1] if len(sys.argv) > 1 else 'cuda:3')
    w.show()
    sys.exit(app.exec_())
