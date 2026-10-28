#!/usr/bin/env python3
"""18.10.2026, ідея юзера: основна мережа працює як завжди, а дзеркальна отримує те, що ReLU відрізав, і назад нічого не віддає.
  основна:     p_l = h_{l-1} W_l ,  h_l = ReLU(p_l)
  відрізане:   c_l = ReLU(-p_l) = h_l - p_l   (≥ 0)
  дзеркальна:  m_l = m_{l-1} W_l + c_l        (лінійна, сама нічого не ріже)
  лінійна:     λ_l = λ_{l-1} W_l , λ_{-1} = z  (мережа без жодного ReLU)
Тотожність: h_l = λ_l + m_l ТОЧНО, а E[λ] = 0 ТОЧНО ⟹ усе середнє основної мережі осідає в дзеркальній.
Внесок шару k у середнє виходу: E[c_k]·W_{k+1}⋯W_15 (лінійність середнього).
Запуск: /home/trim/notqa-venv/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/mirror_qt.py [дамп=0] [N=65536]   (--png шлях)"""
import sys, os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
import numpy as np, torch, pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QLabel, QSlider, QSpinBox
from PyQt6.QtCore import Qt
args = [a for a in sys.argv[1:] if not a.startswith("--")]; png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
if png: args = [a for a in args if a != png]
di = int(args[0]) if args else 0; N = int(args[1]) if len(args) > 1 else 65536; dev = "cuda"; B = 8192; JN = 0
Wn = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % di)["weights"]; W = [torch.tensor(Wn[l], dtype=torch.float64, device=dev) for l in range(16)]
truth = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % di)["mean"][15].astype(np.float64)
z64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
S = {k: z64(16, 1024) for k in ("h", "m", "l", "c", "h2", "m2", "l2", "hl")}; ident = 0.0; samp = {k: [] for k in ("h", "m", "l")}
g = torch.Generator(device=dev).manual_seed(777)
for b in range(N // B):
    lam = torch.randn(B, 1024, generator=g, device=dev, dtype=torch.float64); h = lam.clone(); m = z64(B, 1024)
    for l in range(16):
        p = h @ W[l]; hn = torch.relu(p); c = hn - p; lam = lam @ W[l]; m = m @ W[l] + c; h = hn
        ident = max(ident, float(((h - lam - m).abs().max() / (lam.abs().max() + 1e-30))))
        for k, v in (("h", h), ("m", m), ("l", lam), ("c", c)): S[k][l] += v.sum(0)
        S["h2"][l] += (h * h).sum(0); S["m2"][l] += (m * m).sum(0); S["l2"][l] += (lam * lam).sum(0); S["hl"][l] += (h * lam).sum(0)
        if b < 4:
            for k, v in (("h", h), ("m", m), ("l", lam)): samp[k].append((l, v[:, :64].cpu().numpy()))
E = {k: (S[k] / N).cpu().numpy() for k in ("h", "m", "l", "c")}
sd = {k: np.sqrt(np.maximum((S[k + "2"] / N).cpu().numpy() - E[k] ** 2, 0)) for k in ("h", "m", "l")}
corr_hl = ((S["hl"] / N).cpu().numpy() - E["h"] * E["l"]) / (sd["h"] * sd["l"] + 1e-30)
SAMP = {k: [np.concatenate([a for (l, a) in samp[k] if l == L]) for L in range(16)] for k in samp}
# внесок кожного шару в середнє виходу: E[c_k] W_{k+1}..W_15
Wc = [w.cpu().numpy() for w in W]; contrib = []
for k in range(16):
    v = E["c"][k].copy()
    for l in range(k + 1, 16): v = v @ Wc[l]
    contrib.append(v)
contrib = np.array(contrib); out = E["h"][15]; cum = np.cumsum(contrib, 0)
rms = lambda a: np.sqrt((a ** 2).mean())

pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")
COL = dict(h="#0f8b8d", m="#c0396b", l="#8a97a4", ink="#18212b", acc="#d9731f")
app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Дзеркальна мережа: що осідає з відрізаного ReLU · мережа %d, N=%d" % (di, N)); cw = QWidget(); G = QGridLayout(cw); w.setCentralWidget(cw)
bar = QHBoxLayout(); sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(0, 15); sl.setValue(15); jn = QSpinBox(); jn.setRange(0, 63); lab = QLabel()
for x in (QLabel("шар L:"), sl, QLabel("нейрон (0..63):"), jn, lab): bar.addWidget(x)
G.addLayout(bar, 0, 0, 1, 2); cw.setStyleSheet("QLabel{font: 10pt 'IBM Plex Mono'}")
def P(t, xl, yl, logy=False):
    p = pg.PlotWidget(title=t); p.showGrid(x=True, y=True, alpha=0.25); p.setLabel("bottom", xl); p.setLabel("left", yl); p.setLogMode(y=logy); p.addLegend(offset=(10, 10))
    for a in ("bottom", "left"): p.getAxis(a).enableAutoSIPrefix(False)
    return p
x = np.arange(16)
p1 = P("1. по шарах: середнє (суцільні) і розкид (пунктир), RMS по 1024 нейронах", "шар", "величина (лог)", True)
for k, nm in (("h", "основна h"), ("m", "дзеркальна m"), ("l", "лінійна λ (без ReLU)")):
    p1.plot(x, [rms(E[k][l]) for l in x], pen=pg.mkPen(COL[k], width=2.5), symbol="o", symbolSize=6, symbolBrush=COL[k], name=nm + ": |середнє|")
    p1.plot(x, [rms(sd[k][l]) for l in x], pen=pg.mkPen(COL[k], width=1.5, style=Qt.PenStyle.DashLine), name=nm + ": розкид")
p1.plot(x, [rms(E["c"][l]) for l in x], pen=pg.mkPen(COL["acc"], width=2), symbol="t", symbolBrush=COL["acc"], name="відрізане c шару (середнє)")
p2 = P("2. звідки середнє виходу: внесок відрізаного на шарі k, перенесений лінійно до виходу", "шар k", "частка / cos")
share = np.array([contrib[k] @ out / (out @ out) for k in range(16)])
p2.addItem(pg.BarGraphItem(x=x, height=share, width=0.7, brush=(192, 57, 107, 170), name="проєкція внеску на середнє виходу (сума = 1)"))
p2.plot(x, [cum[k] @ out / np.linalg.norm(cum[k]) / np.linalg.norm(out) for k in x], pen=pg.mkPen(COL["ink"], width=2), symbol="o", symbolSize=6, name="cos(накопичене до шару k, середнє виходу)")
p2.plot(x, [np.linalg.norm(contrib[k]) / np.linalg.norm(out) for k in x], pen=pg.mkPen(COL["acc"], width=1.5, style=Qt.PenStyle.DashLine), symbol="t", name="|внесок| / |середнє виходу|"); p2.addLine(y=0, pen=pg.mkPen(COL["l"]))
p3 = P("", "середнє основної h_j", "середнє дзеркальної m_j / лінійної λ_j")
s3m = p3.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=(192, 57, 107, 150), symbolPen=None, name="дзеркальна"); s3l = p3.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=(138, 151, 164, 150), symbolPen=None, name="лінійна (має бути 0)"); d3 = p3.plot(pen=pg.mkPen(COL["ink"], style=Qt.PenStyle.DotLine))
p4 = P("", "значення на одному семплі", "щільність"); c4 = {k: p4.plot(pen=pg.mkPen(COL[k], width=2), name=nm) for k, nm in (("h", "основна h_j"), ("m", "дзеркальна m_j"), ("l", "лінійна λ_j"))}
G.addWidget(p1, 1, 0); G.addWidget(p2, 1, 1); G.addWidget(p3, 2, 0); G.addWidget(p4, 2, 1)
foot = QLabel("Тотожність h = λ + m перевірена на кожному семплі: max відносна похибка %.1e. Середнє виходу проти банку: RMS %.2e (шум МК). corr(h, λ) на виході: медіана %.3f ⟹ лінійна мережа як контрольна змінна дає редукцію дисперсії ×%.4f." % (ident, rms(out - truth), np.median(corr_hl[15]), 1 / (1 - np.median(corr_hl[15] ** 2))))
foot.setWordWrap(True); G.addWidget(foot, 3, 0, 1, 2)
def draw():
    L = sl.value(); j = jn.value(); s3m.setData(E["h"][L], E["m"][L]); s3l.setData(E["h"][L], E["l"][L]); lo, hi = E["h"][L].min(), E["h"][L].max(); d3.setData([lo, hi], [lo, hi])
    p3.setTitle("3. шар %d, 1024 нейрони: середнє дзеркальної = середнє основної (діагональ), лінійна ≈ 0 ± шум %.1e" % (L, rms(E["l"][L])))
    for k in c4:
        v = SAMP[k][L][:, j]; y, e = np.histogram(v, bins=120, density=True); c4[k].setData(0.5 * (e[1:] + e[:-1]), y)
    p4.setTitle("4. шар %d, нейрон %d на окремих семплах: σ основної %.2g · σ дзеркальної %.2g · σ лінійної %.2g (m і λ гасять одна одну)" % (L, j, sd["h"][L][j], sd["m"][L][j], sd["l"][L][j]))
    lab.setText(" L=%d" % L)
sl.valueChanged.connect(lambda _: draw()); jn.valueChanged.connect(lambda _: draw()); draw(); w.resize(1700, 1050); w.show()
if png: app.processEvents(); w.grab().save(png); sys.exit(0)
sys.exit(app.exec())
