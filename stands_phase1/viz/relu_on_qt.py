#!/usr/bin/env python3
"""17.10.2026. Графік: середнє виходу при поступовому вмиканні ReLU по нейронах (дані wb2lean/dig/relu_on.py → /home/trim/wblogs/relu_on_d<d>.npz).
Запуск: /usr/bin/python3 relu_on_qt.py [дамп=0]"""
import sys, numpy as np, pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel
from PyQt6.QtCore import Qt
d = int(sys.argv[1]) if len(sys.argv) > 1 else 0; z = np.load("/home/trim/wblogs/relu_on_d%d.npz" % d); ks = z["ks"]; R = z["rows"]
import os; RR = np.load("/home/trim/wblogs/relu_on_d%d_rand.npz" % d)["rows"] if os.path.exists("/home/trim/wblogs/relu_on_d%d_rand.npz" % d) else None
pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")
app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Поступове вмикання ReLU по нейронах: мережа %d, одні ваги, 65536 спільних точок" % d); cw = QWidget(); G = QGridLayout(cw); w.setCentralWidget(cw)
def P(t, xl, yl, logy=False):
    p = pg.PlotWidget(title=t); p.setLogMode(y=logy); p.showGrid(x=True, y=True, alpha=0.25); p.setLabel("bottom", xl); p.setLabel("left", yl); p.getAxis("bottom").enableAutoSIPrefix(False); p.getAxis("left").enableAutoSIPrefix(False)
    for l in range(17): p.addLine(x=1024 * l, pen=pg.mkPen("#c9d2db", width=1))
    return p
p1 = P("1. |E[out]| (лог): суцільна = послідовно, пунктир = випадковий порядок", "увімкнених нейронів (1024 на шар, зліва направо шари 0…15)", "|E[out]|", True)
p1.addLegend(offset=(-10,10)); p1.plot(ks, R[:, 1] + 1e-12, pen=pg.mkPen("#0f8b8d", width=2), name="послідовно (шар за шаром)")
if RR is not None: p1.plot(ks, RR[:, 1] + 1e-12, pen=pg.mkPen("#0f8b8d", width=2, style=Qt.PenStyle.DashLine), name="випадковий порядок по всій мережі")
p2 = P("2. напрямок: cos(E[out] часткове, E[out] повне ReLU)", "увімкнених нейронів", "cos"); p2.addLegend(offset=(10,10)); p2.plot(ks, R[:, 3], pen=pg.mkPen("#d9731f", width=2), name="послідовно")
if RR is not None: p2.plot(ks, RR[:, 3], pen=pg.mkPen("#d9731f", width=2, style=Qt.PenStyle.DashLine), name="випадково"); p2.addLine(y=0, pen=pg.mkPen("#8a97a4"))
p3 = P("3. середнє по 1024 виходах, E[out] усереднене по нейронах", "увімкнених нейронів", "mean_j E[out_j]"); p3.plot(ks, R[:, 2], pen=pg.mkPen("#3d5a98", width=2))
if RR is not None: p3.plot(ks, RR[:, 2], pen=pg.mkPen("#3d5a98", width=2, style=Qt.PenStyle.DashLine)); p3.addLine(y=0, pen=pg.mkPen("#8a97a4"))
p4 = P("4. RMS(E[out] часткове − повне) (лог)", "увімкнених нейронів", "RMS", True); p4.plot(ks, R[:, 4] + 1e-12, pen=pg.mkPen("#c0396b", width=2))
if RR is not None: p4.plot(ks, RR[:, 4] + 1e-12, pen=pg.mkPen("#c0396b", width=2, style=Qt.PenStyle.DashLine))
G.addWidget(p1, 0, 0); G.addWidget(p2, 0, 1); G.addWidget(p3, 1, 0); G.addWidget(p4, 1, 1)
lab = QLabel("вимкнений нейрон = тотожність · повне ReLU (k=16384): |E[out]| = %.2f, cos = 1, RMS до банку %.1e (шум 65k точок) · k=0 (лінійна мережа): E[out] ≈ 0, |E[out]| = %.1f це шум" % (R[-1, 1], R[-1, 4], R[0, 1]))
lab.setStyleSheet("font: 10pt 'IBM Plex Mono'; padding:4px"); G.addWidget(lab, 2, 0, 1, 2)
w.resize(1700, 950); w.show()
if "--png" in sys.argv: app.processEvents(); w.grab().save("/mnt/win_d/llm/White-Box-P2/viz/relu_on.png"); sys.exit(0)
sys.exit(app.exec())
