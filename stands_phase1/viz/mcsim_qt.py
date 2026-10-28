#!/usr/bin/env python3
"""17.10.2026. Жива симуляція Монте-Карло (PyQt6 + pyqtgraph + torch на GPU): точки без ліміту, кнопки Старт / Стоп / Скинути.
Мережа d (дамп), шар L. Щотакту пакет B точок z~N(0,I) через L шарів, накопичуємо середнє передактивації p_j для всіх 1024 нейронів шару L
і вклади 1024 ваг v_i = h_i·W_ij для обраного нейрона j. Істина = банк 1e9 (pre_mean[L]), ланцюг = /home/trim/wblogs/live/d<d>_l<L>.npz.
Панелі: 1) |похибка|/σ проти N: 3 нейрони, RMS по 1024, ланцюг (пунктир), 1/√N · 2) вклади ваг нейрона j: RMS однієї, |Σ|, ланцюг
3) розсіювання: похибка МК проти похибки ланцюга по 1024 нейронах (corr у заголовку) · 4) знакові траєкторії 3 нейронів.
Запуск: /home/trim/notqa-venv/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/mcsim_qt.py [d=0] [L=8]   (GPU 2 за UUID уже в коді)"""
import sys, os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
import numpy as np, torch, pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QCheckBox
from PyQt6.QtCore import Qt, QTimer

DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"; BANK = "/home/trim/wb2data/kp_p2/mlp_{:05d}.npz"
di = int(sys.argv[1]) if len(sys.argv) > 1 else 0; L = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
Wn = np.load(DUMP.format(di))["weights"]; W = [torch.tensor(Wn[l], dtype=torch.float32, device=dev) for l in range(L + 1)]
kp = np.load(BANK.format(di)); mu_true = kp["pre_mean"][L].astype(np.float64)
lv = np.load("/home/trim/wblogs/live/d%d_l%02d.npz" % (di, L)); mu_c = lv["mu"].astype(np.float64); sig = np.sqrt(np.diag(lv["C"])).astype(np.float64)
e_c = (mu_c - mu_true) / sig                          # похибка ланцюга у σ, 1024
js = [int(np.argmax(np.abs(e_c))), int(np.argsort(np.abs(e_c))[512]), int(np.argmin(np.abs(mu_c / sig)))]
h_true_prev = kp["mean"][L - 1].astype(np.float64); r_true = kp["mean"][L].astype(np.float64)
from math import erf
_t = mu_c / sig; r_chain = mu_c * 0.5 * (1 + np.vectorize(erf)(_t / np.sqrt(2))) + sig * np.exp(-_t ** 2 / 2) / np.sqrt(2 * np.pi)
lvp = np.load("/home/trim/wblogs/live/d%d_l%02d.npz" % (di, L - 1)); _mp = lvp["mu"].astype(np.float64); _sp = np.sqrt(np.diag(lvp["C"])).astype(np.float64); _tp = _mp / _sp
h_chain_prev = _mp * 0.5 * (1 + np.vectorize(erf)(_tp / np.sqrt(2))) + _sp * np.exp(-_tp ** 2 / 2) / np.sqrt(2 * np.pi)
mse_chain = float(((r_chain - r_true) ** 2).mean()); mse_chain_pre = float(((mu_c - mu_true) ** 2).mean())
COL = ["#0f8b8d", "#d9731f", "#3d5a98"]; MUTE = "#8a97a4"
pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")

class Sim:
    def __init__(self):
        self.g = torch.Generator(device=dev).manual_seed(777); self.reset()
    def reset(self):
        self.n = 0; self.acc_p = torch.zeros(1024, dtype=torch.float64, device=dev); self.acc_v = torch.zeros(1024, dtype=torch.float64, device=dev); self.acc_r = torch.zeros(1024, dtype=torch.float64, device=dev); self.acc_p2 = torch.zeros(1024, dtype=torch.float64, device=dev)
        self.hist = []   # (N, e3[3] у σ, rms1024 у σ, rms_w, sum_w)
    def step(self, B, j):
        z = torch.randn(B, 1024, generator=self.g, device=dev); h = z
        for l in range(L): h = torch.relu(h @ W[l])
        p = h @ W[L]; self.acc_p += p.double().sum(0); self.acc_r += torch.relu(p).double().sum(0); self.acc_p2 += (p.double() ** 2).sum(0); self.acc_v += (h * W[L][:, j][None]).double().sum(0); self.n += B
        pm = (self.acc_p / self.n).cpu().numpy(); e = (pm - mu_true) / sig
        vm = (self.acc_v / self.n).cpu().numpy(); ew = vm - (h_chain_prev if blind.isChecked() else h_true_prev) * Wn[L][:, j]
        rm = (self.acc_r / self.n).cpu().numpy(); self.mse = float(((rm - r_true) ** 2).mean()); self.mse_pre = float(((pm - mu_true) ** 2).mean())
        dch = (pm - mu_c) / sig; var_hat = (self.acc_p2 / self.n).cpu().numpy() - pm ** 2; se = np.sqrt(np.maximum(var_hat, 0) / self.n) / sig
        self.pm = pm; self.dch = dch; self.se = se; self.mse_vs_chain = float(((rm - r_chain) ** 2).mean()); self.mse_pre_vs_chain = float(((pm - mu_c) ** 2).mean())
        self.hist.append((self.n, e[js], float(np.sqrt((e ** 2).mean())), float(np.sqrt((ew ** 2).mean())), float(abs(ew.sum())), dch[js], float(np.sqrt((dch ** 2).mean())), se[js], float(np.sqrt((se ** 2).mean()))))
        return e

sim = Sim(); app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Жива симуляція Монте-Карло: мережа %d, шар %d, %s" % (di, L, dev)); cw = QWidget(); g = QGridLayout(cw); w.setCentralWidget(cw)
bar = QHBoxLayout(); btn = QPushButton("▶ Старт"); rst = QPushButton("Скинути"); bs = QSpinBox(); bs.setRange(256, 65536); bs.setSingleStep(256); bs.setValue(4096)
jn = QSpinBox(); jn.setRange(0, 1023); jn.setValue(js[0]); lab = QLabel(); blind = QCheckBox("істина невідома (як у задачі)"); ghost = QCheckBox("накласти криву з істиною")
for x in (btn, rst, QLabel("пакет:"), bs, QLabel("нейрон для ваг:"), jn, blind, ghost, lab): bar.addWidget(x)
bar.addStretch(); g.addLayout(bar, 0, 0, 1, 2)
score = QLabel("MSE середнього шару %d (1024 нейрони): натисни Старт" % L); score.setStyleSheet("font: bold 12pt 'IBM Plex Mono'; color:#18212b; background:#e7ecf1; padding:6px; border-radius:4px"); g.addWidget(score, 1, 0, 1, 2)
lab.setStyleSheet("font: 10pt 'IBM Plex Mono'"); cw.setStyleSheet("QLabel{font: 10pt 'IBM Plex Mono'}")

def mk(title, xl, yl, logy=True):
    pw = pg.PlotWidget(title=title); pw.setLogMode(x=True, y=logy); pw.showGrid(x=True, y=True, alpha=0.25); pw.setLabel("bottom", xl); pw.setLabel("left", yl)
    pw.addLegend(offset=(10, -10)); pw.getAxis("bottom").enableAutoSIPrefix(False); pw.getAxis("left").enableAutoSIPrefix(False); return pw
p1 = mk("1. |похибка p_j| / σ проти N", "N точок", "|похибка| / σ"); p2 = mk("2. вклади 1024 ваг нейрона: похибка біжучого середнього", "N точок", "|похибка вкладу|")
p3 = pg.PlotWidget(title="3. 1024 нейрони шару: похибка МК проти похибки ланцюга (у σ)"); p3.showGrid(x=True, y=True, alpha=0.25); p3.setLabel("bottom", "ланцюг − істина, /σ"); p3.setLabel("left", "МК − істина, /σ"); p3.getAxis("bottom").enableAutoSIPrefix(False); p3.getAxis("left").enableAutoSIPrefix(False)
p4 = mk("4. знакова похибка / σ, три нейрони на спільних точках", "N точок", "похибка / σ", logy=False)
g.addWidget(p1, 2, 0); g.addWidget(p2, 2, 1); g.addWidget(p3, 3, 0); g.addWidget(p4, 3, 1)
c1 = [p1.plot(pen=pg.mkPen(COL[a], width=2), name="нейрон %d (t=%.1f)" % (js[a], mu_c[js[a]] / sig[js[a]])) for a in range(3)]
c1r = p1.plot(pen=pg.mkPen("#18212b", width=2.5), name="RMS по 1024 нейронах"); c1t = p1.plot(pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine), name="1/√N")
tl = [p1.addLine(y=np.log10(abs(e_c[js[a]]) + 1e-12), pen=pg.mkPen(COL[a], style=Qt.PenStyle.DashLine)) for a in range(3)]
tl.append(p1.addLine(y=np.log10(np.sqrt((e_c ** 2).mean())), pen=pg.mkPen("#18212b", width=1.5, style=Qt.PenStyle.DashLine)))
c1g = [p1.plot(pen=pg.mkPen(COL[a], width=1, style=Qt.PenStyle.DotLine)) for a in range(3)]; c4g = [p4.plot(pen=pg.mkPen(COL[a], width=1, style=Qt.PenStyle.DotLine)) for a in range(3)]
c1se = p1.plot(pen=pg.mkPen("#c0396b", width=2, style=Qt.PenStyle.DashLine), name="оцінка шуму s/√N з самих точок (RMS)")
c2r = p2.plot(pen=pg.mkPen(COL[1], width=2.5), name="RMS однієї ваги"); c2s = p2.plot(pen=pg.mkPen(COL[0], width=2.5), name="|Σ 1024| = похибка p_j")
c2a = p2.plot(pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine), name="RMS·√1024"); c2b = p2.plot(pen=pg.mkPen(MUTE, style=Qt.PenStyle.DashLine), name="RMS·1024")
l2 = pg.InfiniteLine(angle=0, pen=pg.mkPen("#18212b", width=1.5, style=Qt.PenStyle.DashLine)); p2.addItem(l2)
sc = pg.ScatterPlotItem(size=5, pen=None, brush=(61, 90, 152, 120)); p3.addItem(sc); p3.addLine(x=0, pen=pg.mkPen(MUTE)); p3.addLine(y=0, pen=pg.mkPen(MUTE)); diag = p3.plot(pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine))
c4 = [p4.plot(pen=pg.mkPen(COL[a], width=2), name="нейрон %d" % js[a]) for a in range(3)]; c4u = p4.plot(pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine), name="±2/√N"); c4d = p4.plot(pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine)); p4.addLine(y=0, pen=pg.mkPen(MUTE))

def chain_w_line():
    j = jn.value(); l2.setPos(np.log10(abs(e_c[j] * sig[j]) + 1e-12))
chain_w_line()

def redraw():
    H = sim.hist; N = np.array([x[0] for x in H], float); E3 = np.array([x[1] for x in H]); R = np.array([x[2] for x in H]); RW = np.array([x[3] for x in H]); SW = np.array([x[4] for x in H])
    B = blind.isChecked(); D3 = np.array([x[5] for x in H]); RD = np.array([x[6] for x in H]); SE3 = np.array([x[7] for x in H]); RSE = np.array([x[8] for x in H])
    for it in tl: it.setVisible(not B)
    G = B and ghost.isChecked()
    for a in range(3): c1g[a].setData(N, np.abs(E3[:, a]) + 1e-9 if G else []); c4g[a].setData(N, E3[:, a] if G else [])
    if B: E3, R = D3, RD
    l2.setVisible(not B); p2.setTitle("2. вклади 1024 ваг нейрона: " + ("|МК − ланцюг| по вагах (істини нема)" if B else "похибка біжучого середнього"))
    for a in range(3): c1[a].setData(N, np.abs(E3[:, a]) + 1e-9); c4[a].setData(N, E3[:, a])
    c1se.setData(N, RSE); c1r.setData(N, R); c1t.setData(N, 1 / np.sqrt(N)); c4u.setData(N, 2 / np.sqrt(N)); c4d.setData(N, -2 / np.sqrt(N))
    c2r.setData(N, RW); c2s.setData(N, SW + 1e-12); c2a.setData(N, RW * 32); c2b.setData(N, RW * 1024)
    e = sim.last
    if B:
        sc.setData(mu_c / sig, sim.pm / sig); lim = np.abs(mu_c / sig).max(); diag.setData([-lim, lim], [-lim, lim])
        p3.setTitle("3. 1024 нейрони: середнє МК проти ланцюга (обидва у σ, істини нема) · RMS різниці %.1e · оцінка шуму МК %.1e" % (RD[-1], RSE[-1]))
        p1.setTitle("1. |МК − ланцюг| / σ проти N (істини нема; хто з двох помиляється, невідомо)"); p4.setTitle("4. (МК − ланцюг) / σ, три нейрони")
        p3.setLabel("bottom", "ланцюг μ/σ"); p3.setLabel("left", "МК середнє/σ")
        score.setText("без істини видно лише:  MSE(МК − ланцюг) після ReLU %.3e   передактивація %.3e   ·   власний шум МК з точок: Var/N = %.3e   ·   різниця ≈ шум ⟹ ланцюг і МК ще не розрізнити" % (sim.mse_vs_chain, sim.mse_pre_vs_chain, float((sim.se ** 2 * sig ** 2).mean())))
        lab.setText("N = %d · RMS(МК − ланцюг) %.2e σ · оцінка шуму МК %.2e σ" % (sim.n, RD[-1], RSE[-1])); decades(N[-1]); return
    p1.setTitle("1. |похибка p_j| / σ проти N"); p4.setTitle("4. знакова похибка / σ, три нейрони на спільних точках"); p3.setLabel("bottom", "ланцюг − істина, /σ"); p3.setLabel("left", "МК − істина, /σ")
    sc.setData(e_c, e); lim = max(np.abs(e_c).max(), np.abs(e).max()); diag.setData([-lim, lim], [-lim, lim])
    p3.setTitle("3. 1024 нейрони шару: похибка МК проти ланцюга (у σ), corr %.3f · RMS МК %.1e · ланцюг %.1e" % (np.corrcoef(e_c, e)[0, 1], R[-1], np.sqrt((e_c ** 2).mean())))
    score.setText("MSE середнього шару %d  ·  після ReLU:  МК %.3e   ланцюг (гаусова формула з μ,C, без κ3) %.3e   ×%.2f     |   передактивація μ:  МК %.3e   ланцюг %.3e   ×%.1f" % (L, sim.mse, mse_chain, sim.mse / mse_chain, sim.mse_pre, mse_chain_pre, sim.mse_pre / mse_chain_pre))
    decades(N[-1])
    lab.setText("N = %d · RMS МК %.2e σ · ланцюг %.2e σ · відношення ×%.1f · N до рівності ≈ %.1e" % (sim.n, R[-1], np.sqrt((e_c ** 2).mean()), R[-1] / np.sqrt((e_c ** 2).mean()), (1 / np.sqrt((e_c ** 2).mean())) ** 2))

def decades(nmax):
    lo, hi = 2, int(np.ceil(np.log10(max(nmax, 100))))
    tk = [[(k, "10^%d" % k if k > 5 else "%d" % 10 ** k) for k in range(lo, hi + 1)], [(np.log10(m * 10 ** k), "") for k in range(lo, hi + 1) for m in (2, 5)]]
    for pw in (p1, p2, p4): pw.getAxis("bottom").setTicks(tk)
timer = QTimer(); timer.setInterval(0)
def tick():
    sim.last = sim.step(bs.value(), jn.value()); redraw(); app.processEvents()
timer.timeout.connect(tick)
def toggle():
    if timer.isActive(): timer.stop(); btn.setText("▶ Старт")
    else: timer.start(); btn.setText("■ Стоп")
btn.clicked.connect(toggle)
def do_reset(): sim.reset(); chain_w_line()
rst.clicked.connect(do_reset); jn.valueChanged.connect(do_reset); blind.stateChanged.connect(lambda _: do_reset()); ghost.stateChanged.connect(lambda _: sim.hist and redraw())
w.resize(1700, 1000); w.show(); sys.exit(app.exec())
