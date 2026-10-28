#!/usr/bin/env python3
"""17.10.2026. Один шар, по кожному нейрону (жива симуляція, PyQt6 + pyqtgraph + torch GPU).
Семплюємо справжній вхід шару L (z через L шарів), робимо ОДИН крок: p = h·W_L, r = ReLU(p). Для кожного з 1024 нейронів порівнюємо:
  МК       середнє r_j по N точках (шум 1/√N)
  гаус-крок  μ·Φ(t)+σ·φ(t) з μ_j, σ_j, зміряних на тих самих точках (похибка форми, НЕ падає з N)
  істина   банк 1e9 (mean[L])
Панелі: 1) похибка по нейронах (у σ), відсортовано за t · 2) та сама похибка проти N: RMS по 1024 для МК і для гаус-кроку ·
3) обраний нейрон: гістограма p_j проти гауса з тими ж μ, σ · 4) розсіювання: похибка гаус-кроку проти γ3 нейрона.
Запуск: /home/trim/notqa-venv/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/onestep_qt.py [дамп=0] [шар=8]"""
import sys, os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
import numpy as np, torch, pyqtgraph as pg
from math import erf
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox
from PyQt6.QtCore import Qt, QTimer

DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"; BANK = "/home/trim/wb2data/kp_p2/mlp_{:05d}.npz"
di = int(sys.argv[1]) if len(sys.argv) > 1 else 0; L = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
Wn = np.load(DUMP.format(di))["weights"]; W = [torch.tensor(Wn[l], dtype=torch.float32, device=dev) for l in range(L + 1)]
kp = np.load(BANK.format(di)); r_true = kp["mean"][L].astype(np.float64); mu_true = kp["pre_mean"][L].astype(np.float64)
sd_true = np.sqrt(np.maximum(kp["pre_m2"][L].astype(np.float64) - mu_true ** 2, 1e-12)) if kp["pre_m2"][L].mean() > kp["pre_mean"][L].var() else np.sqrt(kp["pre_m2"][L].astype(np.float64))
g3_true = kp["pre_m3"][L].astype(np.float64)   # (центральний чи ні, уточнюється нижче за масштабом)

Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / np.sqrt(2)))); phi = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
t_true = mu_true / sd_true; gs_true = mu_true * Phi(t_true) + sd_true * phi(t_true); e_gs_true = (gs_true - r_true) / sd_true   # гаус-крок з ТОЧНИХ μ, σ: чиста похибка форми
COL = dict(mc="#0f8b8d", gs="#d9731f", ink="#18212b", mute="#8a97a4", k3="#c0396b")
pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")

class Sim:
    def __init__(self): self.g = torch.Generator(device=dev).manual_seed(777); self.reset()
    def reset(self):
        self.n = 0; z = lambda: torch.zeros(1024, dtype=torch.float64, device=dev)
        self.S1, self.S2, self.S3, self.R1 = z(), z(), z(), z(); self.hist = []; self.keep = None
    def step(self, B, j):
        z = torch.randn(B, 1024, generator=self.g, device=dev); h = z
        for l in range(L): h = torch.relu(h @ W[l])
        p = (h @ W[L]).double(); r = torch.relu(p); self.n += B
        self.S1 += p.sum(0); self.S2 += (p ** 2).sum(0); self.S3 += (p ** 3).sum(0); self.R1 += r.sum(0)
        pj = p[:, j].cpu().numpy(); self.keep = pj if self.keep is None or len(self.keep) > 200000 else np.concatenate([self.keep, pj])
        n = self.n; mu = (self.S1 / n); var = (self.S2 / n - mu ** 2).clamp_min(1e-12); sd = var.sqrt()
        m3 = self.S3 / n - 3 * mu * var - mu ** 3; g3 = (m3 / sd ** 3).cpu().numpy()
        mu_n, sd_n = mu.cpu().numpy(), sd.cpu().numpy(); t = mu_n / sd_n
        gs = mu_n * Phi(t) + sd_n * phi(t)                    # гаус-крок з зміряних μ, σ
        mc = (self.R1 / n).cpu().numpy()
        self.e_mc = (mc - r_true) / sd_n; self.e_gs = (gs - r_true) / sd_n; self.t = t; self.g3 = g3; self.mu_n = mu_n; self.sd_n = sd_n
        self.hist.append((n, float(np.sqrt((self.e_mc ** 2).mean())), float(np.sqrt((self.e_gs ** 2).mean())), float(np.sqrt(((mu_n - mu_true) / sd_n) ** 2).mean())))

sim = Sim(); app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Один шар по нейронах: мережа %d, крок %d → %d" % (di, L, L + 1)); cw = QWidget(); G = QGridLayout(cw); w.setCentralWidget(cw)
bar = QHBoxLayout(); btn = QPushButton("▶ Старт"); rst = QPushButton("Скинути"); bs = QSpinBox(); bs.setRange(256, 65536); bs.setSingleStep(256); bs.setValue(4096)
jn = QSpinBox(); jn.setRange(0, 1023); jn.setValue(0); lab = QLabel()
for x in (btn, rst, QLabel("пакет:"), bs, QLabel("нейрон:"), jn, lab): bar.addWidget(x)
bar.addStretch(); G.addLayout(bar, 0, 0, 1, 2); cw.setStyleSheet("QLabel{font: 10pt 'IBM Plex Mono'}")
def P(title, xl, yl, logx=False, logy=False):
    pw = pg.PlotWidget(title=title); pw.setLogMode(x=logx, y=logy); pw.showGrid(x=True, y=True, alpha=0.25); pw.setLabel("bottom", xl); pw.setLabel("left", yl); pw.addLegend(offset=(10, 10)); pw.getAxis("bottom").enableAutoSIPrefix(False); pw.getAxis("left").enableAutoSIPrefix(False); return pw
p1 = P("1. похибка E[ReLU] по 1024 нейронах, у σ (нейрони відсортовано за t = μ/σ)", "нейрон (за t)", "похибка / σ")
c1m = p1.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=(15, 139, 141, 140), symbolPen=None, name="МК (шум, падає з N)")
c1g = p1.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=(217, 115, 31, 160), symbolPen=None, name="гаус-крок зі зміряних μ, σ (форма, не падає)")
c1x = p1.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=(192, 57, 107, 200), symbolPen=None, name="гаус-крок з ТОЧНИХ μ, σ банку (чиста похибка форми)")
p1.addLine(y=0, pen=pg.mkPen(COL["mute"])); c1t = p1.plot(pen=pg.mkPen(COL["mute"], width=1, style=Qt.PenStyle.DashLine), name="t/10 (для орієнтиру)")
p2 = P("2. RMS похибки по 1024 нейронах проти N", "N точок", "RMS / σ", True, True)
c2m = p2.plot(pen=pg.mkPen(COL["mc"], width=2.5), name="МК: E[ReLU] з точок"); c2g = p2.plot(pen=pg.mkPen(COL["gs"], width=2.5), name="гаус-крок зі зміряних μ, σ"); c2u = p2.plot(pen=pg.mkPen(COL["mute"], style=Qt.PenStyle.DotLine), name="1/√N")
c2p = p2.plot(pen=pg.mkPen(COL["ink"], width=1.5, style=Qt.PenStyle.DashLine), name="похибка самого μ (вхід кроку)")
p2.addLine(y=np.log10(np.sqrt((e_gs_true ** 2).mean())), pen=pg.mkPen(COL["k3"], width=2, style=Qt.PenStyle.DashLine)); c2x = p2.plot(pen=pg.mkPen(COL["k3"], width=2, style=Qt.PenStyle.DashLine), name="гаус-крок з точних μ, σ: RMS %.1e σ (плато форми)" % np.sqrt((e_gs_true ** 2).mean()))
p3 = P("3. обраний нейрон: гістограма передактивації p_j проти гауса з тими ж μ, σ", "p_j", "щільність")
c3h = pg.PlotCurveItem(np.array([0.0, 1.0]), np.array([0.0]), stepMode="center", fillLevel=0, brush=(15, 139, 141, 80), pen=pg.mkPen(COL["mc"], width=1.5)); p3.addItem(c3h); c3g = p3.plot(pen=pg.mkPen(COL["ink"], width=1.5, style=Qt.PenStyle.DashLine), name="гаус"); l3 = pg.InfiniteLine(angle=90, pen=pg.mkPen(COL["k3"], width=1.5)); p3.addItem(l3)
p4 = P("4. похибка гаус-кроку (точні μ, σ) проти асиметрії γ3 (кожна точка = нейрон)", "γ3 нейрона", "похибка гаус-кроку / σ")
c4 = p4.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=(217, 115, 31, 120), symbolPen=None); p4.addLine(y=0, pen=pg.mkPen(COL["mute"])); p4.addLine(x=0, pen=pg.mkPen(COL["mute"]))
G.addWidget(p1, 1, 0); G.addWidget(p2, 1, 1); G.addWidget(p3, 2, 0); G.addWidget(p4, 2, 1)

def redraw():
    o = np.argsort(sim.t); c1m.setData(sim.e_mc[o]); c1g.setData(sim.e_gs[o]); c1x.setData(e_gs_true[o]); c1t.setData(sim.t[o] / 10); p1.setYRange(-0.02, 0.02)
    H = sim.hist; N = np.array([x[0] for x in H], float); c2m.setData(N, [x[1] for x in H]); c2g.setData(N, [x[2] for x in H]); c2u.setData(N, 1 / np.sqrt(N)); c2p.setData(N, [x[3] for x in H])
    j = jn.value(); y, x = np.histogram(sim.keep, bins=100, density=True); c3h.setData(x, y); xx = np.linspace(x[0], x[-1], 300); c3g.setData(xx, phi((xx - sim.mu_n[j]) / sim.sd_n[j]) / sim.sd_n[j]); l3.setPos(0)
    p3.setTitle("3. нейрон %d: t=%.2f, γ3=%.2f · похибка гаус-кроку %.1e σ · МК %.1e σ (червона лінія = нуль ReLU)" % (j, sim.t[j], sim.g3[j], sim.e_gs[j], sim.e_mc[j]))
    c4.setData(sim.g3, e_gs_true); p4.setTitle("4. похибка гаус-кроку з точних μ, σ проти γ3 передактивації (corr %.2f)" % np.corrcoef(sim.g3, e_gs_true)[0, 1])
    lab.setText("N = %d · RMS МК %.2e σ · RMS гаус-кроку %.2e σ · гаус-крок перестав падати ⟹ це похибка форми, не шуму" % (sim.n, H[-1][1], H[-1][2]))
timer = QTimer(); timer.setInterval(0)
def tick(): sim.step(bs.value(), jn.value()); redraw(); app.processEvents()
timer.timeout.connect(tick)
def toggle():
    if timer.isActive(): timer.stop(); btn.setText("▶ Старт")
    else: timer.start(); btn.setText("■ Стоп")
btn.clicked.connect(toggle); rst.clicked.connect(lambda: sim.reset()); jn.valueChanged.connect(lambda _: setattr(sim, "keep", None))
w.resize(1700, 1000); w.show(); sys.exit(app.exec())
