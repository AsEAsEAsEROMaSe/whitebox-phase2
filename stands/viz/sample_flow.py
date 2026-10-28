"""sample_flow (22.10.2026, white-box-b9): сортуємо ВАГИ шару і дивимось, що стає з входом і виходом.

Мережа рахує p_j = Σ_a h_a · W[a, j], вихід ReLU(p). Сортування ваг нейрона j (стовпця W) дає
посортований стовпець Ws[:, j] і маску перестановки π_j (яка вага стояла на якому місці). Щоб вихід
лишився тим самим, вхід мусить пройти ту саму маску: H[k, j] = h[π_j(k)], тобто кожен нейрон бачить
СВІЙ переставлений вхід, і вектор входу стає матрицею 1024×1024.

Вкладка «Сортування ваг», стовпці: вхід · ваги · добутки вхід×вага · вихід.
  ряд 1  як є:                         h        W        h_a·W[a,j]          ReLU(h·W)
  ряд 2  ваги посортовані + маски:     H = h[π] Ws       H·Ws (по стовпцях)  той самий вихід
  ряд 3  ваги посортовані БЕЗ масок:   h        Ws       h_a·Ws[a,j]         хибний вихід
Режим сортування: у кожному нейроні окремо або всі 1024² ваг шару разом.
Вкладка «Усі шари» і «Останній шар проти істини» як раніше.
usage: python3 sample_flow.py"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

TRUTH = "/home/trim/ext/fc/repo/data/mini_truth.npz"
N, L, S = 1024, 16, 32
SCALE = (2.0 / N) ** 0.5


def weights(seed):
    """ваги як у грейдері: p = h @ W, W[a, j] = вага від входу a до нейрона j"""
    wss, _, _ = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(wss)
    return [(rng.standard_normal((N, N)) * SCALE).astype(np.float32) for _ in range(L)]


def heat(title, cmap, levels):
    pl = pg.PlotItem(title=title)
    im = pg.ImageItem(); pl.addItem(im); pl.invertY(True); pl.setAspectLocked(True)
    pl.hideAxis("bottom"); pl.hideAxis("left")
    im.setColorMap(pg.colormap.get(cmap))
    bar = pg.ColorBarItem(values=levels, colorMap=pg.colormap.get(cmap), interactive=False, width=12)
    bar.setImageItem(im, insert_in=pl)
    return pl, im, bar


def show(im, bar, M, lv):
    im.setImage(M.T); im.setLevels(lv); bar.setLevels(lv)


def grid(v):
    return v.reshape(S, S)


def masked_layer(h, Ws, order, how):
    """шар через посортовані ваги і маску: вхід іде на ваги за маскою, сума на своєму нейроні (індекси справжні)"""
    if how == 0:                                          # order[k, j] = який вхід стоїть біля k-ї ваги нейрона j
        return np.maximum((h[order] * Ws).sum(axis=0), 0.0)
    if how == 1:                                          # order[a, k] = до якого нейрона йде k-та вага входу a
        return np.maximum(np.bincount(order.reshape(-1), weights=(h[:, None] * Ws).reshape(-1).astype(np.float64), minlength=N), 0.0).astype(np.float32)
    a_idx, j_idx = np.unravel_index(order, (N, N))        # order = позиції всіх ваг у посортованому порядку
    return np.maximum(np.bincount(j_idx, weights=(h[a_idx] * Ws.reshape(-1)).astype(np.float64), minlength=N), 0.0).astype(np.float32)


def sort_weights(W, how):
    """посортовані ваги і що потрібно, щоб із ними відтворити h·W: (Ws, H або None, scatter j або None)"""
    if how == 0:                                          # кожен стовпець (нейрон) окремо
        order = np.argsort(-W, axis=0)
        return np.take_along_axis(W, order, axis=0), order, None
    if how == 1:                                          # кожен рядок (вхід) окремо
        order = np.argsort(-W, axis=1)
        return np.take_along_axis(W, order, axis=1), None, order
    flat = np.argsort(-W, axis=None)                      # уся матриця
    return W.reshape(-1)[flat].reshape(N, N), None, flat


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сортування ваг шару: що стає з входом і виходом")
        self.t = np.load(TRUTH, allow_pickle=False)
        self.net = -1
        top = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout(); top.addLayout(bar)
        self.netc = QtWidgets.QComboBox()
        for i in range(12):
            self.netc.addItem(f"mini {i}: {self.t['mlp_name'][i]}")
        self.layer = QtWidgets.QSpinBox(); self.layer.setRange(0, L - 1); self.layer.setValue(0)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 10 ** 6); self.seed.setValue(1)
        self.how = QtWidgets.QComboBox(); self.how.addItems(["у кожному нейроні (стовпці W)", "у кожному вході (рядки W)", "усі ваги шару разом"])
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["середнє 64 семплів", "середнє 1024 семплів", "середнє 4096 семплів"])
        for lab, w in (("мережа", self.netc), ("шар", self.layer), ("сід семпла", self.seed),
                       ("сортувати ваги", self.how), ("остання вкладка", self.mode)):
            bar.addWidget(QtWidgets.QLabel(lab)); bar.addWidget(w)
        bar.addStretch(1)
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.info)
        tabs = QtWidgets.QTabWidget(); top.addWidget(tabs, 1)

        g0 = pg.GraphicsLayoutWidget(); tabs.addTab(g0, "Мережа з посортованими вагами: вхід і виходи шарів")
        self.net_cells = {}
        cmi0 = pg.colormap.get("inferno")
        for r0, key, name in ((0, "orig", "справжня мережа"), (2, "mask", "посортовані ваги + маски"), (4, "sort", "посортовані ваги без масок")):
            g0.addLabel(f"<b>{name}</b>", angle=-90, row=r0, col=0, rowspan=2)
            cells = []
            for i in range(L + 1):
                pl = g0.addPlot(row=r0 + i // 9, col=1 + i % 9, title=("вхід z" if i == 0 else f"ш{i - 1}"))
                im = pg.ImageItem(); im.setColorMap(cmi0); pl.addItem(im); pl.invertY(True); pl.setAspectLocked(True)
                pl.hideAxis("bottom"); pl.hideAxis("left")
                cells.append((pl, im))
            self.net_cells[key] = cells

        g1 = pg.GraphicsLayoutWidget(); tabs.addTab(g1, "Один шар: вхід · ваги · добутки · вихід")
        wl = (-4 * SCALE, 4 * SCALE)
        rows = ("як є", "ваги посортовані + маски", "ваги посортовані без масок")
        titles = (("вхід h (32×32)", "ваги W (рядок = вхід, стовпець = нейрон)", "добутки h_a · W[a, j]", "вихід ReLU(h·W) (32×32)"),
                  ("вхід, переставлений маскою під кожен нейрон", "ваги посортовані", "добутки (ті самі числа, інший порядок)", "вихід: сума стовпців → ReLU"),
                  ("вхід h як є", "ваги посортовані", "добутки без маски", "вихід без маски (хибний)"))
        cm = (("inferno", "CET-D1", "CET-D1", "inferno"),) * 3
        self.cells = []
        for r in range(3):
            g1.addLabel(f"<b>{rows[r]}</b>", angle=-90, row=r, col=0)
            row = []
            for c in range(4):
                cell = heat(titles[r][c], cm[r][c], wl)
                g1.addItem(cell[0], row=r, col=c + 1)
                row.append(cell)
            self.cells.append(row)

        g2 = pg.GraphicsLayoutWidget(); tabs.addTab(g2, "Усі шари: активації сирі і посортовані")
        self.all_raw, self.all_srt = [], []
        cmi = pg.colormap.get("inferno")
        for half, store, r0 in (("сирі", self.all_raw, 0), ("посортовані", self.all_srt, 2)):
            for l in range(L):
                pl = g2.addPlot(row=r0 + l // 8, col=l % 8, title=f"шар {l} {half}")
                im = pg.ImageItem(); im.setColorMap(cmi); pl.addItem(im); pl.invertY(True); pl.setAspectLocked(True)
                pl.hideAxis("bottom"); pl.hideAxis("left")
                store.append(im)

        g3 = pg.GraphicsLayoutWidget(); tabs.addTab(g3, "Останній шар проти істини")
        self.tr3 = [heat(tt, c, (0, 3)) for tt, c in (("істина датасету (1e9 семплів), посортована", "inferno"),
                                                       ("середнє наших семплів у тому ж порядку", "inferno"),
                                                       ("різниця (наше − істина)", "CET-D1"))]
        for c, cell in enumerate(self.tr3):
            g3.addItem(cell[0], row=0, col=c)

        for w in (self.netc, self.mode, self.how):
            w.currentIndexChanged.connect(self.redraw)
        self.layer.valueChanged.connect(self.redraw); self.seed.valueChanged.connect(self.redraw)
        self.resize(1750, 1050)
        self.redraw()

    def load(self, k):
        if k != self.net:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.Ws = weights(int(self.t["mlp_seed"][k]))
            self.truth = self.t["all_layer_means"][k].astype(np.float32)
            self.net = k
            QtWidgets.QApplication.restoreOverrideCursor()

    def redraw(self):
        self.load(self.netc.currentIndex())
        l = self.layer.value()
        z = np.random.default_rng(self.seed.value()).standard_normal(N).astype(np.float32)
        hs = [z]
        for W in self.Ws:
            hs.append(np.maximum(hs[-1] @ W, 0.0))
        key = (self.net, self.how.currentIndex())
        if getattr(self, "_srt_key", None) != key:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            srt = [sort_weights(Wl, self.how.currentIndex()) for Wl in self.Ws]
            self.Wsorted = [x[0] for x in srt]
            self.orders = [(x[1] if x[1] is not None else x[2]).astype(np.int32) for x in srt]
            self._srt_key = key
            QtWidgets.QApplication.restoreOverrideCursor()
        gs = [z]
        for Wl in self.Wsorted:
            gs.append(np.maximum(gs[-1] @ Wl, 0.0))
        ms = [z]
        for Wl, od in zip(self.Wsorted, self.orders):
            ms.append(masked_layer(ms[-1], Wl, od, self.how.currentIndex()))
        mdiff = [float(np.max(np.abs(a - b))) for a, b in zip(hs[1:], ms[1:])]
        lim0 = float(np.percentile(np.concatenate(hs[1:]), 99.5))
        for key2, seq in (("orig", hs), ("mask", ms), ("sort", gs)):
            for i, (pl, im) in enumerate(self.net_cells[key2]):
                v = seq[i]
                lv = (-3.0, 3.0) if i == 0 else (0.0, lim0)
                im.setImage(grid(v).T, levels=lv)
                if i > 0:
                    pl.setTitle(f"ш{i - 1} сер {float(v.mean()):.3g}", size="8pt")
        corr = [float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan") for a, b in zip(hs[1:], gs[1:])]
        h, W = hs[l], self.Ws[l]
        y = np.maximum(h @ W, 0.0)
        wl = (-4 * SCALE, 4 * SCALE)
        hl = (-3.0, 3.0) if l == 0 else (0.0, float(np.percentile(h, 99.5)))
        yl = (0.0, float(np.percentile(y, 99.5)))
        prod = h[:, None] * W
        pl = float(np.percentile(np.abs(prod), 99.5)); pl = (-pl, pl)
        how = self.how.currentIndex()
        if how == 0:
            order = np.argsort(-W, axis=0)                     # маска: для нейрона j порядок входів за спаданням ваги
            Wsrt = np.take_along_axis(W, order, axis=0)        # кожен стовпець посортований
            H = h[order]                                       # H[k, j] = h[π_j(k)]
            ys = np.maximum((H * Wsrt).sum(axis=0), 0.0)       # той самий вихід
            what = "у кожному нейроні (стовпці W) окремо: маска = 1024 перестановки по 1024"
        elif how == 1:
            order = np.argsort(-W, axis=1)                     # маска: для входу a, до якого нейрона йде k-та вага
            Wsrt = np.take_along_axis(W, order, axis=1)        # кожен рядок посортований
            H = np.repeat(h[:, None], N, axis=1)               # вхід той самий, а вихід треба розкидати маскою
            ys = np.maximum(np.bincount(order.reshape(-1), weights=(H * Wsrt).reshape(-1).astype(np.float64), minlength=N), 0.0).astype(np.float32)
            what = "у кожному вході (рядки W) окремо: маска каже, до якого нейрона йде кожна вага"
        else:
            flat = np.argsort(-W, axis=None)                   # одна маска на всі 1024² ваг
            Wsrt = W.reshape(-1)[flat].reshape(N, N)           # уся матриця посортована
            a_idx, j_idx = np.unravel_index(flat, (N, N))
            H = h[a_idx].reshape(N, N)                         # вхід, що стоїть біля кожної ваги
            ys = np.maximum(np.bincount(j_idx, weights=(H.reshape(-1) * Wsrt.reshape(-1)).astype(np.float64), minlength=N), 0.0).astype(np.float32)
            what = "усі 1024² ваг шару разом: маска = одна перестановка на мільйон позицій (рядок і стовпець кожної ваги)"
        yw = np.maximum(h @ Wsrt, 0.0)                          # посортовані ваги без маски
        c = self.cells
        show(c[0][0][1], c[0][0][2], grid(h), hl); show(c[0][1][1], c[0][1][2], W, wl)
        show(c[0][2][1], c[0][2][2], prod, pl); show(c[0][3][1], c[0][3][2], grid(y), yl)
        show(c[1][0][1], c[1][0][2], H, hl); show(c[1][1][1], c[1][1][2], Wsrt, wl)
        show(c[1][2][1], c[1][2][2], H * Wsrt, pl); show(c[1][3][1], c[1][3][2], grid(ys), yl)
        show(c[2][0][1], c[2][0][2], grid(h), hl); show(c[2][1][1], c[2][1][2], Wsrt, wl)
        show(c[2][2][1], c[2][2][2], h[:, None] * Wsrt, pl); show(c[2][3][1], c[2][3][2], grid(yw), (0.0, max(yl[1], float(np.percentile(yw, 99.5)))))
        same = float(np.max(np.abs(ys - y)))
        bad = float(np.linalg.norm(yw - y) / max(np.linalg.norm(y), 1e-30))
        # усі шари
        lim = float(np.percentile(np.concatenate(hs[1:]), 99.5))
        for i in range(L):
            self.all_raw[i].setImage(grid(hs[i + 1]).T, levels=(0, lim))
            self.all_srt[i].setImage(grid(np.sort(hs[i + 1])).T, levels=(0, lim))
        # останній шар проти істини
        B = [64, 1024, 4096][self.mode.currentIndex()]
        Z = np.random.default_rng(self.seed.value()).standard_normal((B, N)).astype(np.float32)
        hb = Z
        for Wl in self.Ws:
            hb = np.maximum(hb @ Wl, 0.0)
        out = hb.mean(axis=0); tr = self.truth[-1]; o = np.argsort(tr)
        tl = (0.0, float(np.percentile(tr, 99.5)))
        show(self.tr3[0][1], self.tr3[0][2], grid(tr[o]), tl); show(self.tr3[1][1], self.tr3[1][2], grid(out[o]), tl)
        d = out[o] - tr[o]; dl = float(np.max(np.abs(d))) or 1.0
        show(self.tr3[2][1], self.tr3[2][2], grid(d), (-dl, dl))
        mse = float(np.mean((out - tr) ** 2)); theo = float(self.t["avg_variance"][self.net]) / B
        self.info.setText(
            f"<b>Посортовані ваги + маски</b> (вхід іде на ваги за маскою, сума на своєму нейроні): макс. різниця зі справжньою мережею по шарах "
            + " ".join(f"{x:.0e}" for x in mdiff) + " ⟹ та сама мережа (лише округлення fp32). Ціна: маска переставляє 1024² входів на шар, стільки ж, скільки саме множення.<br>"
            f"<b>Посортовані ваги без масок:</b> середнє виходу останнього шару "
            f"{float(gs[-1].mean()):.3f} проти справжнього {float(hs[-1].mean()):.3f}; кореляція виходів шарів зі справжніми: "
            + " ".join("—" if c != c else f"{c:.2f}" for c in corr) + "<br>"
            f"<b>Шар {l}, один семпл. Сортування ваг {what}.</b><br>"
            f"З масками вихід той самий: максимальна різниця <b>{same:.1e}</b>. Але вхід перестає бути вектором 1024: "
            f"кожен нейрон бачить свій переставлений вхід, це матриця 1024×1024, а маска займає стільки ж, скільки самі ваги.<br>"
            f"Без масок (посортовані ваги, вхід як є): відносна похибка виходу <b>{bad:.2f}</b>, тобто вся інформація шару в масках.<br>"
            f"Остання вкладка: середнє {B} семплів, MSE {mse:.3e} (очікувано ≈ {theo:.2e}; сабміт 337568 дає 1.9e-8).")


if __name__ == "__main__":
    pg.setConfigOptions(background="w", foreground="k", antialias=True, imageAxisOrder="col-major")
    app = QtWidgets.QApplication(sys.argv)
    w = App(); w.show()
    sys.exit(app.exec())
