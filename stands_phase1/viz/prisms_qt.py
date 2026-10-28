#!/usr/bin/env python3
"""18.10.2026, метафора юзера: мережа як поле призм з різним кутом.
Без біасів кожна межа ReLU проходить через нуль ⟹ кожна грань є КЛИНОМ з центру, а мережа всередині клину лінійна (призма зі своїм кутом).
Тому досить дивитись на коло напрямків: межа грані = кут, де хоч один нейрон перемикається.
Режими: іграшка (вхід 2D, ширина 16, глибина 8) · справжня мережа (зріз площиною через два випадкові напрямки в z-просторі 1024).
Панелі: 1) площина входу, грані-клини до шару L, пучок точок · 2) кількість граней на колі проти шару (усі нейрони / лише призми) ·
3) куди йдуть промені: образ кола напрямків після шару L (PCA), колір = кут входу · 4) частка відкриття кожного нейрона шару L: скло, мертві, призми.
Запуск: /home/trim/notqa-venv/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/prisms_qt.py   (--png шлях для знімка)"""
import sys, os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
import numpy as np, pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QLabel, QSlider, QComboBox
from PyQt6.QtCore import Qt
try:
    import torch; DEV = "cuda" if torch.cuda.is_available() else None
except Exception: torch = None; DEV = None
NT = 4096; TH = np.linspace(0, 2 * np.pi, NT, endpoint=False); GLASS = 0.995
pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")

def stack(X, W):   # прохід шарами; віддає генератор (l, передактивація>0, стан) на GPU, якщо є
    if DEV and X.shape[1] > 64:
        h = torch.tensor(X, dtype=torch.float32, device=DEV)
        for l, w in enumerate(W):
            p = h @ torch.tensor(w, dtype=torch.float32, device=DEV); h = torch.relu(p); yield l, p > 0, h
    else:
        h = X
        for l, w in enumerate(W): p = h @ w; h = np.maximum(p, 0); yield l, p > 0, h
cpu = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else x

def build(mode):
    rng = np.random.RandomState(3); nt = 4096 if mode == 0 else 65536; th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    if mode == 0:
        dims = [2] + [16] * 8; W = [rng.randn(dims[i], dims[i + 1]) * np.sqrt(2 / dims[i]) for i in range(8)]
        U = np.eye(2); full = rng.randn(40000, 2)
    else:
        Wn = np.load("/home/trim/wb2data/dumps_p2/mlp_0000.npz")["weights"]; W = [np.asarray(Wn[l], dtype=np.float32) for l in range(16)]
        U = np.linalg.qr(rng.randn(1024, 2))[0].T * np.sqrt(1024); full = rng.randn(8192, 1024).astype(np.float32)
    beam2 = rng.randn(1500, 2)
    fopen = [cpu(m.float().mean(0)) if hasattr(m, "float") else m.mean(0) for _, m, _ in stack(full, W)]
    glass = [(f > GLASS) | (f < 1 - GLASS) for f in fopen]
    circ = (np.stack([np.cos(th), np.sin(th)], 1) @ U).astype(np.float32)
    D, Dp, edges, edgesp, hs = [], [], [], [], []; acc = np.zeros(nt, bool); accp = np.zeros(nt, bool)
    for l, m, h in stack(circ, W):
        m = cpu(m); ch = m != np.roll(m, 1, 0); e = ch.any(1); ep = ch[:, ~glass[l]].any(1)
        acc |= e; accp |= ep; D.append(int(acc.sum())); Dp.append(int(accp.sum())); edges.append(np.where(e)[0]); edgesp.append(acc.copy()); hs.append(cpu(h)[::nt // 1024].astype(np.float64))
    seg = [np.cumsum(a) % 12 for a in edgesp]
    return dict(W=W, hs=hs, fopen=fopen, glass=glass, D=D, Dp=Dp, seg=seg, edges=edges, beam2=beam2, L=len(W), nt=nt, th=th)

PAL = np.array([pg.intColor(i, 12, maxValue=235, minValue=200, sat=70).getRgb()[:3] for i in range(12)], np.uint8)
app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Поле призм: грані ReLU-мережі без біасів"); cw = QWidget(); G = QGridLayout(cw); w.setCentralWidget(cw)
bar = QHBoxLayout(); cb = QComboBox(); cb.addItems(["іграшка: вхід 2D, 16×8", "справжня мережа 0: зріз площиною, 1024×16"]); sl = QSlider(Qt.Orientation.Horizontal); lab = QLabel(); info = QLabel()
for x in (QLabel("мережа:"), cb, QLabel("  шар L:"), sl, lab): bar.addWidget(x)
G.addLayout(bar, 0, 0, 1, 2); cw.setStyleSheet("QLabel{font: 10pt 'IBM Plex Mono'}")
def P(t, xl, yl):
    p = pg.PlotWidget(title=t); p.showGrid(x=True, y=True, alpha=0.2); p.setLabel("bottom", xl); p.setLabel("left", yl)
    for a in ("bottom", "left"): p.getAxis(a).enableAutoSIPrefix(False)
    return p
p1 = P("", "напрямок 1", "напрямок 2"); p1.setAspectLocked(True); img = pg.ImageItem(); p1.addItem(img); rays = pg.PlotCurveItem(pen=pg.mkPen((30, 30, 40, 150), width=1), connect="pairs"); p1.addItem(rays)
sc1 = pg.ScatterPlotItem(size=3, brush=(20, 30, 50, 120), pen=None); p1.addItem(sc1)
p2 = P("2. скільки граней-клинів перетинає коло напрямків", "шар", "граней на колі (лог)"); p2.setLogMode(y=True); p2.addLegend(offset=(10, 10))
c2a = p2.plot(pen=pg.mkPen("#3d5a98", width=2.5), symbol="o", symbolSize=6, name="межі від усіх нейронів"); c2p = p2.plot(pen=pg.mkPen("#d9731f", width=2.5), symbol="o", symbolSize=6, name="межі лише від призм (без скла)"); v2 = pg.InfiniteLine(angle=90, pen=pg.mkPen("#c0396b", width=1.5)); p2.addItem(v2)
p3 = P("", "головна вісь 1", "головна вісь 2"); p3.setAspectLocked(True); sc3 = pg.ScatterPlotItem(size=3, pen=None); p3.addItem(sc3); o3 = pg.ScatterPlotItem(size=10, brush="#18212b", pen=None); p3.addItem(o3)
p4 = P("", "нейрон шару L (відсортовано)", "частка пучка, для якої нейрон відкритий"); c4 = p4.plot(pen=None, symbol="o", symbolSize=3)
for y in (GLASS, 1 - GLASS): p4.addLine(y=y, pen=pg.mkPen("#8a97a4", style=Qt.PenStyle.DashLine))
G.addWidget(p1, 1, 0); G.addWidget(p2, 1, 1); G.addWidget(p3, 2, 0); G.addWidget(p4, 2, 1); info.setWordWrap(True); G.addWidget(info, 3, 0, 1, 2)
S = {}; HUE = np.array([pg.hsvColor(i / 1024, 0.75, 0.85).getRgb() for i in range(1024)])
gx = np.linspace(-3.2, 3.2, 360); GX, GY = np.meshgrid(gx, gx); ANG = ((np.arctan2(GY, GX) % (2 * np.pi)) / (2 * np.pi) * NT).astype(int) % NT; R = np.hypot(GX, GY)

def load(mode):
    info.setText("рахую…"); app.processEvents(); S.clear(); S.update(build(mode)); sl.blockSignals(True); sl.setRange(0, S["L"] - 1); sl.setValue(0); sl.blockSignals(False)
    c2a.setData(np.arange(S["L"]), np.maximum(S["D"], 1)); c2p.setData(np.arange(S["L"]), np.maximum(S["Dp"], 1)); sc1.setData(S["beam2"][:, 0], S["beam2"][:, 1]); draw()
def draw():
    L = sl.value(); seg = S["seg"][L]; rgb = PAL[seg[(ANG * S["nt"]) // NT]].astype(float); rgb[R > 3.1] = 244; img.setImage(rgb.transpose(1, 0, 2).astype(np.uint8), levels=(0, 255)); img.setRect(-3.2, -3.2, 6.4, 6.4)
    ch = S["edges"][L]; ch = ch[:: max(1, len(ch) // 400)]; th = S["th"]; xs = np.zeros(2 * len(ch)); ys = np.zeros(2 * len(ch)); xs[1::2] = 3.1 * np.cos(th[ch]); ys[1::2] = 3.1 * np.sin(th[ch]); rays.setData(xs, ys)
    p1.setTitle("1. площина входу: %d граней-клинів на колі з %d точок після шару %d · лінії = ребра шару %d (до 400) · точки = пучок" % (S["D"][L], S["nt"], L, L))
    h = S["hs"][L]; nr = np.linalg.norm(h, axis=1, keepdims=True) + 1e-30; u = h / nr; c = u - u.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); y = c @ Vt[:2].T
    sc3.setData(y[:, 0], y[:, 1], brush=[pg.mkBrush(*x) for x in HUE]); o3.setData([0], [0]); mu = u.mean(0); cosmin = float((u @ (mu / np.linalg.norm(mu))).min())
    p3.setTitle("3. напрямки виходу після шару %d (PCA), колір = кут входу · найширший від осі %.1f°" % (L, np.degrees(np.arccos(np.clip(cosmin, -1, 1)))))
    f = np.sort(S["fopen"][L]); g = S["glass"][L]; c4.setData(f, symbolBrush=[pg.mkBrush("#8a97a4") if (x > GLASS or x < 1 - GLASS) else pg.mkBrush("#d9731f") for x in f])
    ng_open = int((S["fopen"][L] > GLASS).sum()); ng_dead = int((S["fopen"][L] < 1 - GLASS).sum()); n = len(f)
    p4.setTitle("4. шар %d: скло (майже завжди відкрите) %d · мертві %d · призми %d з %d" % (L, ng_open, ng_dead, n - ng_open - ng_dead, n)); lab.setText(" L = %d" % L); v2.setPos(L)
    info.setText("Без біасів кожне ребро ReLU проходить через нуль, тож грань = клин, а всередині клину мережа лінійна (призма). Скло: нейрон, у чиє ребро пучок майже не влучає (відкритий >%.1f%% або <%.1f%%), він лінійний і не дає κ3. "
                 "Помаранчеве на панелі 2 = скільки ребер лишиться, якщо скло викинути. Панель 3: глибше промені сходяться у вузький конус (фокусування)." % (100 * GLASS, 100 * (1 - GLASS)))
cb.currentIndexChanged.connect(load); sl.valueChanged.connect(lambda _: draw())
w.resize(1700, 1050); load(0); w.show()
if "--png" in sys.argv:
    out = sys.argv[sys.argv.index("--png") + 1]
    for mode, L in ((0, 7), (1, 15)):
        cb.setCurrentIndex(mode); sl.setValue(L); app.processEvents(); w.grab().save(out.replace(".png", "_%d.png" % mode))
    sys.exit(0)
sys.exit(app.exec())
