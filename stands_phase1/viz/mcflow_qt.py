#!/usr/bin/env python3
"""17.10.2026. Як Монте-Карло гасить похибку: сирі числа з rawflow_d0_L8.npz (мережа 0, шар 8, 10^6 точок, 3 нейрони, 1024 ваги).
Чотири панелі (pyqtgraph):
 1. біжуче середнє передактивації p_j після N точок: похибка у σ проти N, обвідна 1/√N, горизонталі = похибка ланцюга
 2. вклади ваг v_i = h_i·W_ij для нейрона 236: 24 найбільших |W| окремо (похибка біжучого середнього) та RMS усіх 1024 і |сума|
 3. похибка E[h] попереднього шару по 1024 нейронах: МК 10^6 проти ланцюга (розсіювання, corr ≈ 0.11)
 4. траєкторії трьох нейронів разом (спільні семпли ⟹ корельовані) і матриця кореляцій
Запуск: /usr/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/mcflow_qt.py [npz]"""
import sys, numpy as np, pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel
from PyQt6.QtCore import Qt

f = sys.argv[1] if len(sys.argv) > 1 else "/home/trim/wblogs/rawflow_d0_L8.npz"
d = np.load(f); ks = d["ks"].astype(float); p = d["p"]; v = d["v"]; js = d["js"]; mt = d["mu_true"]; mc = d["mu_chain"]; s = d["s"]; t = d["t"]
Wj = d["Wj"]; hm = d["h_mean_mc"]; ht = d["h_mean_true"]; hc = d["h_mean_chain"]; vt = d["v_truth"]
E = (p - mt) / s                      # (21,3) похибка МК у σ
ech = (mc - mt) / s                   # (3,) похибка ланцюга у σ
e = v - vt                            # (21,1024,3) похибка вкладів
rms = np.sqrt((e ** 2).mean(1)); sm = np.abs(e.sum(1)); q = np.sqrt((e ** 2).sum(1))
o = np.argsort(-np.abs(Wj[:, 0]))
ec = (hc - ht)[:, None] * Wj; em = (hm - ht)[:, None] * Wj
corr = np.corrcoef(E.T)
Ncross = (s / np.abs(mc - mt)) ** 2

pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")
COL = ["#0f8b8d", "#d9731f", "#3d5a98"]; MUTE = "#8a97a4"
app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Як Монте-Карло гасить похибку: мережа 0, шар 8"); cw = QWidget(); g = QGridLayout(cw); w.setCentralWidget(cw)
hdr = QLabel("нейрони %d (t=%.2f) · %d (t=%.2f) · %d (t=%.2f) · σ≈0.55 · похибка ланцюга у σ: %.1e · %.1e · %.1e · МК зрівняється з ланцюгом при N ≈ %.1e · %.1e · %.1e"
             % (js[0], t[0], js[1], t[1], js[2], t[2], abs(ech[0]), abs(ech[1]), abs(ech[2]), Ncross[0], Ncross[1], Ncross[2]))
hdr.setStyleSheet("font: 10pt 'IBM Plex Mono'; color:#223; padding:4px"); g.addWidget(hdr, 0, 0, 1, 2)

def logplot(title, xl, yl):
    pw = pg.PlotWidget(title=title); pw.setLogMode(x=True, y=True); pw.showGrid(x=True, y=True, alpha=0.25); pw.setLabel("bottom", xl); pw.setLabel("left", yl); pw.addLegend(offset=(10, -10)); pw.getAxis("bottom").enableAutoSIPrefix(False); pw.getAxis("left").enableAutoSIPrefix(False); return pw

# 1. біжуче середнє нейрона
p1 = logplot("1. біжуче середнє p_j: |похибка| у σ проти числа точок N", "N точок", "|похибка| / σ")
for a in range(3):
    p1.plot(ks, np.abs(E[:, a]) + 1e-9, pen=pg.mkPen(COL[a], width=2), symbol="o", symbolSize=5, symbolBrush=COL[a], name="нейрон %d, t=%.1f" % (js[a], t[a]))
    p1.addLine(y=np.log10(abs(ech[a])), pen=pg.mkPen(COL[a], width=1.2, style=Qt.PenStyle.DashLine))
p1.plot(ks, 1 / np.sqrt(ks), pen=pg.mkPen(MUTE, width=1.5, style=Qt.PenStyle.DotLine), name="1/√N (теорія шуму)")
p1.addLine(x=np.log10(65536), pen=pg.mkPen("#c0396b", width=1, style=Qt.PenStyle.DashDotLine))
txt = pg.TextItem("бюджет ≈ 65k точок", color="#c0396b", anchor=(0, 1)); txt.setPos(np.log10(65536), np.log10(0.2)); p1.addItem(txt)
txt2 = pg.TextItem("пунктир = похибка ланцюга (стала, N не залежить)", color=MUTE, anchor=(1, 0)); txt2.setPos(np.log10(1e6), np.log10(0.05)); p1.addItem(txt2)
g.addWidget(p1, 1, 0)

# 2. вклади ваг
p2 = logplot("2. вклади ваг v_i = h_i·W_i%d (1024 штук): похибка біжучого середнього" % js[0], "N точок", "|похибка вкладу|")
for i in o[:24]:
    p2.plot(ks, np.abs(e[:, i, 0]) + 1e-12, pen=pg.mkPen((217, 115, 31, 60), width=1))
p2.plot(ks, rms[:, 0], pen=pg.mkPen(COL[1], width=2.5), name="RMS по 1024 вагах (одна вага)")
p2.plot(ks, sm[:, 0], pen=pg.mkPen(COL[0], width=2.5), name="|Σ 1024 похибок| = похибка p_j")
p2.plot(ks, rms[:, 0] * 32, pen=pg.mkPen(MUTE, width=1.5, style=Qt.PenStyle.DotLine), name="RMS·√1024: незалежні знаки")
p2.plot(ks, rms[:, 0] * 1024, pen=pg.mkPen(MUTE, width=1, style=Qt.PenStyle.DashLine), name="RMS·1024: якби всі в один бік")
p2.addLine(y=np.log10(abs(ec[:, 0].sum())), pen=pg.mkPen("#18212b", width=1.2, style=Qt.PenStyle.DashLine))
t3 = pg.TextItem("ланцюг: сума вкладів %.1e (RMS однієї ваги %.1e, коеф %.2f)" % (abs(ec[:, 0].sum()), np.sqrt((ec[:, 0] ** 2).mean()), abs(ec[:, 0].sum()) / np.sqrt((ec[:, 0] ** 2).sum())), color="#18212b", anchor=(0, 1))
t3.setPos(np.log10(30), np.log10(abs(ec[:, 0].sum()))); p2.addItem(t3)
g.addWidget(p2, 1, 1)

# 3. розсіювання похибок E[h] попереднього шару
p3 = pg.PlotWidget(title="3. попередній шар, 1024 нейрони: похибка E[h] ланцюга проти МК 10⁶ (corr %.2f)" % np.corrcoef(hm - ht, hc - ht)[0, 1])
p3.showGrid(x=True, y=True, alpha=0.25); p3.setLabel("bottom", "похибка ланцюга E[h_i] − істина"); p3.setLabel("left", "похибка МК 10⁶ E[h_i] − істина")
sc = pg.ScatterPlotItem(hc - ht, hm - ht, size=5, pen=None, brush=(61, 90, 152, 120)); p3.addItem(sc)
lim = max(np.abs(hc - ht).max(), np.abs(hm - ht).max()); p3.plot([-lim, lim], [-lim, lim], pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine)); p3.addLine(x=0, pen=pg.mkPen(MUTE)); p3.addLine(y=0, pen=pg.mkPen(MUTE))
t4 = pg.TextItem("RMS: ланцюг %.1e · МК %.1e\nмайже незалежні ⟹ МК не повторює помилку ланцюга, але й не виправляє її" % (np.sqrt(((hc - ht) ** 2).mean()), np.sqrt(((hm - ht) ** 2).mean())), color="#223", anchor=(0, 0)); t4.setPos(-lim, lim); p3.addItem(t4)
g.addWidget(p3, 2, 0)

# 4. три траєкторії разом, знакові
p4 = pg.PlotWidget(title="4. три нейрони на ТИХ САМИХ точках: знакова похибка у σ (corr %.2f · %.2f · %.2f)" % (corr[0, 1], corr[0, 2], corr[1, 2]))
p4.setLogMode(x=True, y=False); p4.showGrid(x=True, y=True, alpha=0.25); p4.setLabel("bottom", "N точок"); p4.setLabel("left", "похибка / σ"); p4.addLegend(offset=(-10, 10)); p4.getAxis("bottom").enableAutoSIPrefix(False)
for a in range(3):
    p4.plot(ks, E[:, a], pen=pg.mkPen(COL[a], width=2), symbol="o", symbolSize=5, symbolBrush=COL[a], name="нейрон %d" % js[a])
p4.plot(ks, 2 / np.sqrt(ks), pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine), name="±2/√N"); p4.plot(ks, -2 / np.sqrt(ks), pen=pg.mkPen(MUTE, style=Qt.PenStyle.DotLine))
p4.addLine(y=0, pen=pg.mkPen(MUTE))
t5 = pg.TextItem("спільні семпли ⟹ похибки нейронів корельовані через W (236 проти 196: −0.71)\nусереднення 1024 вкладів гасить лише НЕЗАЛЕЖНУ частину: ×32, не ×1024", color="#223", anchor=(0, 1)); t5.setPos(np.log10(3), -0.55); p4.addItem(t5)
g.addWidget(p4, 2, 1)

w.resize(1700, 1000); w.show(); sys.exit(app.exec())
