#!/usr/bin/env python3
"""17.10.2026. Чотири кумулянти вектора стану h (після ReLU шару L) на реальній мережі: середнє, коваріація, асиметрія, ексцес.
Семплюємо N точок на GPU, рахуємо точно з вибірки, малюємо в pyqtgraph.
Запуск: /home/trim/notqa-venv/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/cumulants_qt.py [дамп=0] [шар=8] [N=131072]"""
import sys, os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
import numpy as np, torch, pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel
from PyQt6.QtCore import Qt

DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"
di = int(sys.argv[1]) if len(sys.argv) > 1 else 0; L = int(sys.argv[2]) if len(sys.argv) > 2 else 8; N = int(sys.argv[3]) if len(sys.argv) > 3 else 131072
dev = "cuda" if torch.cuda.is_available() else "cpu"
Wn = np.load(DUMP.format(di))["weights"]; W = [torch.tensor(Wn[l], dtype=torch.float32, device=dev) for l in range(min(L + 2, 16))]
g = torch.Generator(device=dev).manual_seed(777)
# семплюємо потоково (пам'ять GPU спільна з іншими вікнами): прохід 1 = μ, C, частка відкриття; прохід 2 (той самий сід) = центровані моменти
B = 8192; nb = N // B; HAS_P = L + 1 < 16
def gen(): 
    gg = torch.Generator(device=dev).manual_seed(777)
    for _ in range(nb):
        h = torch.randn(B, 1024, generator=gg, device=dev)
        for l in range(L + 1): h = torch.relu(h @ W[l])
        yield h.double(), ((h @ W[L + 1]).double() if HAS_P else None)
S1 = torch.zeros(1024, dtype=torch.float64, device=dev); S2 = torch.zeros(1024, 1024, dtype=torch.float64, device=dev); AL = torch.zeros(1024, dtype=torch.float64, device=dev)
P1 = torch.zeros(1024, dtype=torch.float64, device=dev); P2 = torch.zeros(1024, dtype=torch.float64, device=dev)
for h, p in gen():
    S1 += h.sum(0); S2 += h.T @ h; AL += (h > 0).double().sum(0)
    if HAS_P: P1 += p.sum(0); P2 += (p ** 2).sum(0)
mu = S1 / N; C = S2 / N - mu[:, None] * mu[None, :]; sd = C.diag().clamp_min(0).sqrt(); alive = AL / N
corr = C / (sd[:, None] * sd[None, :] + 1e-12)
pm = P1 / N; psd = (P2 / N - pm ** 2).clamp_min(0).sqrt() if HAS_P else None
K = 48; idx = torch.argsort(sd, descending=True)[:K]; j = int(torch.argsort(sd)[700]); jp = int(torch.argsort(psd)[512]) if HAS_P else 0
Z3 = torch.zeros(1024, dtype=torch.float64, device=dev); Z4 = torch.zeros(1024, dtype=torch.float64, device=dev); K3 = torch.zeros(K, K, dtype=torch.float64, device=dev); K4 = torch.zeros(K, K, dtype=torch.float64, device=dev)
Q3 = torch.zeros(1024, dtype=torch.float64, device=dev); Q4 = torch.zeros(1024, dtype=torch.float64, device=dev); hj_l = []; pj_l = []
for h, p in gen():
    z = (h - mu) / (sd + 1e-12); Z3 += z.pow(3).sum(0); Z4 += z.pow(4).sum(0); zk = z[:, idx]; K3 += (zk ** 2).T @ zk; K4 += (zk ** 2).T @ (zk ** 2); hj_l.append(h[:, j].cpu())
    if HAS_P: zp = (p - pm) / psd; Q3 += zp.pow(3).sum(0); Q4 += zp.pow(4).sum(0); pj_l.append(p[:, jp].cpu())
g3 = Z3 / N; g4 = Z4 / N - 3; k3_aab = K3 / N; k4_aabb = K4 / N - 1 - 2 * corr[idx][:, idx] ** 2
mu_n, sd_n, g3_n, g4_n, alive_n = [x.cpu().numpy() for x in (mu, sd, g3, g4, alive)]
corr_n = corr.cpu().numpy(); k3_n = k3_aab.cpu().numpy(); k4_n = k4_aabb.cpu().numpy(); hj = torch.cat(hj_l).numpy()
p = pm if HAS_P else None
if HAS_P: pg3 = (Q3 / N).cpu().numpy(); pg4 = (Q4 / N - 3).cpu().numpy(); pj = torch.cat(pj_l).numpy()

pg.setConfigOptions(antialias=True, background="#f4f6f8", foreground="#223")
COL = dict(mu="#0f8b8d", cov="#3d5a98", k3="#d9731f", k4="#c0396b", mute="#8a97a4", ink="#18212b")
app = QApplication(sys.argv); w = QMainWindow(); w.setWindowTitle("Кумулянти вектора стану: мережа %d, після ReLU шару %d, N=%d" % (di, L, N)); cw = QWidget(); G = QGridLayout(cw); w.setCentralWidget(cw)
hdr = QLabel("h = стан після ReLU шару %d (1024 нейрони) · частка мертвих (ніколи не відкрились) %.1f%% · медіана t=μ/σ передактивації наступного шару %.2f" % (L, 100 * (alive_n == 0).mean(), float(np.median(np.abs((pm / psd).cpu().numpy()))) if p is not None else 0))
hdr.setStyleSheet("font: 10pt 'IBM Plex Mono'; padding:4px"); G.addWidget(hdr, 0, 0, 1, 3)

def P(title, xl, yl):
    pw = pg.PlotWidget(title=title); pw.showGrid(x=True, y=True, alpha=0.25); pw.setLabel("bottom", xl); pw.setLabel("left", yl); pw.getAxis("bottom").enableAutoSIPrefix(False); pw.getAxis("left").enableAutoSIPrefix(False); return pw
def heat(title, M, cmap, lo, hi):
    pw = pg.PlotWidget(title=title); im = pg.ImageItem(M.T); im.setLookupTable(pg.colormap.get(cmap).getLookupTable()); im.setLevels((lo, hi)); pw.addItem(im); pw.setAspectLocked(True); pw.hideAxis("bottom"); pw.hideAxis("left"); return pw

# 1. середнє
o = np.argsort(mu_n); p1 = P("1. κ1: середнє μ_a по 1024 нейронах (відсортовано)", "нейрон (за μ)", "μ, σ"); p1.addLegend(offset=(10, 10))
p1.plot(mu_n[o], pen=pg.mkPen(COL["mu"], width=2), name="μ"); p1.plot(sd_n[o], pen=pg.mkPen(COL["cov"], width=1.5), name="σ (√діагоналі C)")
p1.plot(np.where(alive_n[o] == 0)[0], mu_n[o][alive_n[o] == 0], pen=None, symbol="x", symbolPen=COL["mute"], symbolSize=5, name="мертві")
G.addWidget(p1, 1, 0)
# 2. коваріація
p2 = heat("2. κ2: кореляційна матриця corr_ab, 1024×1024 (синій −, білий 0, червоний +)", corr_n, "CET-D1", -0.25, 0.25)
off = corr_n[np.triu_indices(1024, 1)]; p2.setTitle("2. κ2: кореляції 1024×1024 · поза діагоналлю RMS %.3f, max |ρ| %.2f · ефект. ранг C %.1f" % (np.sqrt((off ** 2).mean()), np.abs(off).max(), float((C.trace() ** 2 / (C ** 2).sum()).cpu())))
G.addWidget(p2, 1, 1)
# 3. гістограма одного нейрона проти гауса
p3 = P("3. один нейрон h_j після ReLU: гістограма проти гауса з тими ж μ, σ", "h_j", "щільність")
y, x = np.histogram(hj, bins=120, density=True); p3.plot(x, y, stepMode="center", fillLevel=0, brush=(217, 115, 31, 90), pen=pg.mkPen(COL["k3"], width=1.5))
xx = np.linspace(x[0], x[-1], 400); p3.plot(xx, np.exp(-(xx - mu_n[j]) ** 2 / (2 * sd_n[j] ** 2)) / (sd_n[j] * np.sqrt(2 * np.pi)), pen=pg.mkPen(COL["ink"], width=1.5, style=Qt.PenStyle.DashLine))
p3.setTitle("3. нейрон %d після ReLU: γ3=%.2f, γ4=%.2f, відкритий %.0f%% (пунктир = гаус з тими ж μ, σ)" % (j, g3_n[j], g4_n[j], 100 * alive_n[j]))
G.addWidget(p3, 1, 2)
# 4. γ3, γ4 по нейронах проти частки відкриття
p4 = P("4. діагоналі κ3, κ4: γ3, γ4 нейрона проти частки відкриття (майже мертві, <2%, не показано)", "частка семплів, де нейрон відкритий", "γ (нормовано)"); p4.addLegend(offset=(-10, 10)); p4.setYRange(-2, 8)
live = alive_n > 0.02; p4.plot(alive_n[live], g3_n[live], pen=None, symbol="o", symbolSize=4, symbolBrush=(217, 115, 31, 110), symbolPen=None, name="γ3 (асиметрія)")
p4.plot(alive_n[live], g4_n[live], pen=None, symbol="o", symbolSize=4, symbolBrush=(192, 57, 107, 110), symbolPen=None, name="γ4 (ексцес)"); p4.addLine(y=0, pen=pg.mkPen(COL["mute"]))
if p is not None:
    p4.plot([0.5] * 1024, pg3, pen=None, symbol="t", symbolSize=3, symbolBrush=(15, 139, 141, 60), symbolPen=None, name="γ3 передактивації наступного шару (при 0.5)")
    p4.plot([0.52] * 1024, pg4, pen=None, symbol="t", symbolSize=3, symbolBrush=(61, 90, 152, 60), symbolPen=None, name="γ4 передактивації наступного шару (при 0.52)")
G.addWidget(p4, 2, 0)
# 5. зріз κ3_aab
p5 = heat("5. κ3_aab: зріз третього кумулянта для %d найживіших нейронів (гаус = 0)" % K, k3_n, "CET-D1", -0.3, 0.3); G.addWidget(p5, 2, 1)
# 6. зріз κ4_aabb
p6 = heat("6. κ4_aabb: зріз четвертого кумулянта (гаус = 0), ті самі %d нейронів" % K, k4_n, "CET-D1", -0.5, 0.5); G.addWidget(p6, 2, 2)
foot = QLabel("що з цього несе ланцюг v37: κ1, κ2 повністю · κ3 як 4 молодих джерела рангу 1024 + стиснені старі (панель 5 це його точний зріз) · κ4 лише діагональ і зрізи за законами λ·C, κ31 (панель 6 поза діагоналлю) · "
              "те, чого не несе: залежності вище парних («море»), саме вони роблять передактивацію наступного шару негаусовою (трикутники на панелі 4)")
foot.setWordWrap(True); foot.setStyleSheet("font: 10pt 'IBM Plex Mono'; padding:4px; color:#223"); G.addWidget(foot, 3, 0, 1, 3)
w.resize(1800, 1000); w.show(); sys.exit(app.exec())
