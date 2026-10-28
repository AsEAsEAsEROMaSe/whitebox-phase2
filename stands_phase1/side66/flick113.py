#!/usr/bin/env python3
"""ЖИВА МЕРЕЖА ФАЗИ 2 (1024 × 16) — PyQt5, реальний час, ТРИ ВИДИ.

Питання ЮЗЕРА (С113): «як колись в реалтайм гнали семпли і я бачив як нейрони
мерехтять» → «потрібно як нейромережа звична також показати. типу і ваги і
нейрони і окремо».

Це НЕ програвання запису. Кожен кадр — свіжий `z ~ N(0,I)`, прогнаний крізь
16 шарів He-init без біасів, 16 384 нейрони.

┌─ КАРТА ─────────────────────────────────────────────────────────────────┐
│ Усі нейрони одразу. Рядок = шар, піксель = нейрон, тепле = відкритий.    │
│ Тут видно ЗАМЕРЗАННЯ — суть задачі: нейрон, чиї ворота застрягли,        │
│ інформації більше не несе.                                              │
├─ СХЕМА ─────────────────────────────────────────────────────────────────┤
│ Звичний вигляд мережі: кружки й ребра. Показано ПІДМЕРЕЖУ (K нейронів на │
│ шар), бо 1024×1024 ребер на шар намалювати не можна. Ребро світиться     │
│ пропорційно до |h·w| — тобто до РЕАЛЬНОГО потоку сигналу, а не до ваги.  │
├─ ВАГИ ──────────────────────────────────────────────────────────────────┤
│ Сама матриця W_l як зображення + гістограма + вхідний вектор одного      │
│ нейрона. Ваги СТАТИЧНІ; міняється лише те, які з них зараз працюють.     │
└─────────────────────────────────────────────────────────────────────────┘

🔒 ПЛОМБИ (у рядку стану; 🚨 ⟹ зламаний СТЕНД, а не мережа цікава):
  * шар 0: частка `p>0` → 0.5000 (вхід гаусів, m=0 строго — теорема);
  * шар 0: замерзлих РІВНО 0;
  * He-init: sd(W) = √(2/n) = 0.044194 при n=1024, середнє 0;
  * rms передактивацій шару 0 = √2 = 1.41421 ТОЧНО (бо E[p²]=2·E[h²] при He);
    глибші шари плавають 1.42…1.65 — це флуктуація РЕАЛІЗАЦІЇ W, не поламка,
    і її розкид падає як 1/√n (0.068 при n=1024 проти 0.180 при n=256).

📏 ФАЗА 1 ПРОТИ ФАЗИ 2 (зміряно окремо, 2048 семплів, на виході):
        медіана |t|   замерзлих   PR       cos(двох станів)
  256×32    3.400       62.5%    24.6%        0.9790
  1024×16   2.409       34.7%    47.9%        0.9237

Запуск:
    /usr/bin/python3 /mnt/win_d/llm/White-Box-P2/side66/flick113.py
    ... --width 256 --depth 32     # стара форма, для порівняння
    ... --nodes 18                 # більше кружків на схемі
    ... --selftest                 # без вікна

Залежності: numpy + PyQt5. GPU не потрібен (один семпл = 16.7 МФЛОП).
"""
import argparse
import sys
import time

import numpy as np

GROUND = (10, 14, 17)
PANEL = (14, 22, 27)
RULE = (31, 45, 54)
COLD = (47, 127, 158)
ZERO = (12, 18, 22)
WARM = (240, 166, 60)
ROSE = (194, 112, 138)
LIVE = (95, 208, 196)


def _lut(scale=2.2):
    out = np.zeros((256, 3), np.uint8)
    for i, v in enumerate(np.linspace(-4.0, 4.0, 256)):
        w = min(1.0, abs(v) / scale)
        a = np.array(ZERO, float)
        b = np.array(COLD if v <= 0 else WARM, float)
        out[i] = np.clip(a + (b - a) * w, 0, 255).astype(np.uint8)
    return out


LUT_SIG = _lut()
LUT_W = _lut(3.0)
LUT_GATE = np.where((np.arange(256) >= 128)[:, None],
                    np.array(WARM), np.array((20, 29, 35))).astype(np.uint8)


class Net:
    """He-init MLP без біасів. Ваги з сіда, у пам'яті; нічого не пишеться."""

    def __init__(self, width, depth, seed):
        self.w, self.d, self.seed = width, depth, seed
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / width)
        self.W = [(rng.standard_normal((width, width)) * s).astype(np.float32)
                  for _ in range(depth)]
        self.rng = np.random.default_rng(seed ^ 0x5EED)
        self.rms = np.zeros(depth, np.float32)   # 0 ⟹ перший вимір бере значення як є
        self.he_sd = float(s)

    def step(self, n=1):
        """n свіжих семплів. Вертає z, P, H останнього і лічильник воріт усіх n."""
        z = self.rng.standard_normal((n, self.w)).astype(np.float32)
        h = z
        P = np.empty((self.d, self.w), np.float32)
        H = np.empty((self.d, self.w), np.float32)
        G = np.empty((self.d, self.w), np.int32)
        for l, W in enumerate(self.W):
            p = h @ W
            P[l] = p[-1]
            G[l] = (p > 0).sum(0)
            h = np.maximum(p, 0.0)
            H[l] = h[-1]
            r = float(np.sqrt((p * p).mean()))
            self.rms[l] = r if self.rms[l] == 0.0 else 0.97 * self.rms[l] + 0.03 * r
        return z[-1], P, H, G


def selftest():
    print("── самоперевірка стенду ──")
    net = Net(256, 8, 12345)
    n = 4096
    _, P, H, G = net.step(n)
    f0 = G[0] / n
    print(f"🔒 шар 0, частка p>0 : {f0.mean():.4f}      (теорія 0.5000)")
    assert abs(f0.mean() - 0.5) < 0.01
    fz0 = int(((G[0] == 0) | (G[0] == n)).sum())
    print(f"🔒 шар 0, замерзлих  : {fz0}               (теорія 0)")
    assert fz0 == 0
    sd = float(np.std(net.W[0]))
    print(f"🔒 He-init sd(W)     : {sd:.6f}      (теорія {net.he_sd:.6f})")
    assert abs(sd - net.he_sd) / net.he_sd < 0.02
    print(f"🔒 rms шару 0        : {net.rms[0]:.5f}      (теорія 1.41421)")
    assert abs(net.rms[0] / 1.4142135 - 1.0) < 0.05
    print(f"🔒 H = relu(P)       : {float(np.abs(H - np.maximum(P, 0)).max()):.1e}")
    assert np.allclose(H, np.maximum(P, 0))
    print(f"🔒 форми             : P{P.shape}  H{H.shape}  G{G.shape}")
    print("✅ усі пломби пройдено")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--seed", type=int, default=6113113)
    ap.add_argument("--nodes", type=int, default=14)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    from PyQt5 import QtCore, QtGui, QtWidgets

    def qc(t, al=255):
        return QtGui.QColor(t[0], t[1], t[2], al)

    def img_from(rgb):
        arr = np.ascontiguousarray(rgb.astype(np.uint8))
        h, w = arr.shape[:2]
        return QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888), arr

    # ── ВИД 1: КАРТА ─────────────────────────────────────────────────────
    class MapView(QtWidgets.QWidget):
        picked = QtCore.pyqtSignal(int, int)

        def __init__(self, w, d):
            super().__init__()
            self.w, self.d = w, d
            self.buf = np.zeros((d, w, 3), np.float32)
            self.hover = None
            self.setMouseTracking(True)
            self.setMinimumHeight(300)

        def push(self, rgb, decay):
            self.buf *= decay
            np.maximum(self.buf, rgb.astype(np.float32), out=self.buf)
            self.update()

        def paintEvent(self, _):
            img, _keep = img_from(self.buf)
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), qc(GROUND))
            r = self.rect().adjusted(54, 4, -4, -22)
            p.drawImage(r, img)
            p.setPen(qc((113, 136, 143)))
            f = p.font()
            f.setPointSize(7)
            p.setFont(f)
            step = max(1, self.d // 8)
            for l in range(self.d):
                if l == 0 or l == self.d - 1 or l % step == 0:
                    y = r.top() + (l + 0.5) * r.height() / self.d
                    p.drawText(2, int(y) + 3, f"шар {l}")
            p.drawText(r.left(), r.bottom() + 15, "нейрон 0")
            p.drawText(r.right() - 34, r.bottom() + 15, str(self.w - 1))
            if self.hover:
                j, l = self.hover
                p.setPen(QtGui.QPen(qc(LIVE), 1))
                cw = r.width() / self.w
                ch = r.height() / self.d
                p.drawRect(QtCore.QRectF(r.left() + j * cw - 1, r.top() + l * ch,
                                         max(2.0, cw + 2), ch))
            p.end()

        def _cell(self, ev):
            r = self.rect().adjusted(54, 4, -4, -22)
            j = int((ev.x() - r.left()) / max(1, r.width()) * self.w)
            l = int((ev.y() - r.top()) / max(1, r.height()) * self.d)
            return max(0, min(self.w - 1, j)), max(0, min(self.d - 1, l))

        def mouseMoveEvent(self, ev):
            self.hover = self._cell(ev)
            self.update()

        def mousePressEvent(self, ev):
            j, l = self._cell(ev)
            self.picked.emit(l, j)

    # ── ВИД 2: СХЕМА ─────────────────────────────────────────────────────
    class GraphView(QtWidgets.QWidget):
        picked = QtCore.pyqtSignal(int, int)

        def __init__(self, net, K):
            super().__init__()
            self.K = K
            self.set_net(net)
            self.zv = None
            self.hv = None
            self.thr = 0.25
            self.setMinimumHeight(320)
            self.setMouseTracking(True)
            self.hover = None

        def set_net(self, net):
            self.net = net
            rng = np.random.default_rng(net.seed ^ 0xA11CE)
            self.sel = [np.sort(rng.choice(net.w, self.K, replace=False))
                        for _ in range(net.d + 1)]
            self.subW = [net.W[l][np.ix_(self.sel[l], self.sel[l + 1])]
                         for l in range(net.d)]
            self.update()

        def feed(self, z, H):
            self.zv = z[self.sel[0]]
            self.hv = [H[l][self.sel[l + 1]] for l in range(self.net.d)]
            self.update()

        def _geom(self):
            r = self.rect().adjusted(14, 26, -14, -18)
            cols = self.net.d + 1
            xs = [r.left() + (i + 0.5) * r.width() / cols for i in range(cols)]
            ys = [r.top() + (k + 0.5) * r.height() / self.K for k in range(self.K)]
            return xs, ys, r

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            p.fillRect(self.rect(), qc(GROUND))
            xs, ys, r = self._geom()
            f = p.font()
            f.setPointSize(7)
            p.setFont(f)
            p.setPen(qc((113, 136, 143)))
            p.drawText(int(xs[0]) - 14, r.top() - 10, "вхід z")
            for i in range(1, self.net.d + 1):
                if i == self.net.d or (i - 1) % max(1, self.net.d // 6) == 0:
                    p.drawText(int(xs[i]) - 12, r.top() - 10, f"ш{i - 1}")

            vals = [self.zv] + (self.hv or [])
            if self.zv is None:
                p.setPen(qc((113, 136, 143)))
                p.drawText(r.center(), "натисни Пуск")
                p.end()
                return

            # ребра: яскравість ∝ |h·w| (реальний потік), колір ∝ знаку ваги
            for l in range(self.net.d):
                a = np.abs(vals[l])[:, None] * np.abs(self.subW[l])
                m = float(a.max()) or 1.0
                for i in range(self.K):
                    if vals[l][i] <= 0 and l > 0:
                        continue
                    for j in range(self.K):
                        s = a[i, j] / m
                        if s < self.thr:
                            continue
                        col = WARM if self.subW[l][i, j] > 0 else COLD
                        p.setPen(QtGui.QPen(qc(col, int(28 + 200 * s ** 1.6)),
                                            0.4 + 2.0 * s ** 2))
                        p.drawLine(QtCore.QPointF(xs[l], ys[i]),
                                   QtCore.QPointF(xs[l + 1], ys[j]))

            # кружки
            for c in range(self.net.d + 1):
                v = vals[c]
                sc = float(np.abs(v).max()) or 1.0
                for k in range(self.K):
                    val = float(v[k])
                    on = val > 0
                    s = min(1.0, abs(val) / sc)
                    col = qc(WARM, int(60 + 195 * s)) if on else qc((26, 37, 44))
                    p.setBrush(QtGui.QBrush(col))
                    p.setPen(QtGui.QPen(qc(RULE if not on else WARM, 200), 1))
                    rad = 5.0 + 4.0 * s if on else 4.0
                    p.drawEllipse(QtCore.QPointF(xs[c], ys[k]), rad, rad)
            if self.hover is not None:
                c, k = self.hover
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(qc(LIVE), 1.4))
                p.drawEllipse(QtCore.QPointF(xs[c], ys[k]), 11, 11)
                lay = c - 1
                nid = self.sel[c][k]
                p.setPen(qc(LIVE))
                p.drawText(int(xs[c]) - 26, int(ys[k]) - 15,
                           ("z[%d]" % nid) if lay < 0 else ("ш%d·%d" % (lay, nid)))
            p.setPen(qc((90, 110, 118)))
            p.drawText(r.left(), r.bottom() + 13,
                       f"підмережа {self.K}×{self.net.d + 1} із {self.net.w} на шар · "
                       f"тепле ребро = вага>0 · яскравість = |h·w| · "
                       f"порогом сховано слабкі {int(self.thr * 100)}%")
            p.end()

        def _cell(self, ev):
            xs, ys, _ = self._geom()
            c = int(np.argmin([abs(ev.x() - x) for x in xs]))
            k = int(np.argmin([abs(ev.y() - y) for y in ys]))
            return c, k

        def mouseMoveEvent(self, ev):
            self.hover = self._cell(ev)
            self.update()

        def mousePressEvent(self, ev):
            c, k = self._cell(ev)
            if c > 0:
                self.picked.emit(c - 1, int(self.sel[c][k]))

    # ── ВИД 3: ВАГИ ──────────────────────────────────────────────────────
    class WeightView(QtWidgets.QWidget):
        def __init__(self, net):
            super().__init__()
            self.layer = 0
            self.col = 0
            self.set_net(net)
            self.setMouseTracking(True)
            self.setMinimumHeight(320)

        def set_net(self, net):
            self.net = net
            self.rebuild()

        def rebuild(self):
            W = self.net.W[self.layer]
            sd = float(W.std())
            idx = np.clip(W / (3.0 * sd) * 4.0 * 32.0 + 128.0, 0, 255).astype(np.uint8)
            self.rgb = LUT_W[idx]
            self.sd = sd
            self.mean = float(W.mean())
            hist, edges = np.histogram(W[::7, ::7], bins=61,
                                       range=(-4 * sd, 4 * sd), density=True)
            self.hist, self.edges = hist, edges
            self.update()

        def set_layer(self, l):
            self.layer = int(l)
            self.rebuild()

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), qc(GROUND))
            R = self.rect().adjusted(10, 26, -10, -10)
            side = min(R.height(), int(R.width() * 0.52))
            mat = QtCore.QRect(R.left(), R.top(), side, side)
            img, _keep = img_from(self.rgb)
            p.drawImage(mat, img)
            p.setPen(QtGui.QPen(qc(RULE), 1))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(mat)
            f = p.font()
            f.setPointSize(7)
            p.setFont(f)
            p.setPen(qc((113, 136, 143)))
            p.drawText(mat.left(), mat.top() - 10,
                       f"W[{self.layer}]  {self.net.w}×{self.net.w} — "
                       f"рядок = вхідний нейрон, стовпець = вихідний")
            x = mat.left() + int((self.col + 0.5) / self.net.w * mat.width())
            p.setPen(QtGui.QPen(qc(LIVE), 1))
            p.drawLine(x, mat.top(), x, mat.bottom())

            rp = QtCore.QRect(mat.right() + 18, R.top(),
                              R.right() - mat.right() - 18, R.height())
            if rp.width() < 120:
                p.end()
                return
            hh = int(rp.height() * 0.42)
            hb = QtCore.QRect(rp.left(), rp.top(), rp.width(), hh)
            p.fillRect(hb, qc(PANEL))
            m = float(self.hist.max()) or 1.0
            n = len(self.hist)
            for i, v in enumerate(self.hist):
                bx = hb.left() + i * hb.width() / n
                bh = v / m * (hb.height() - 16)
                mid = 0.5 * (self.edges[i] + self.edges[i + 1])
                p.fillRect(QtCore.QRectF(bx, hb.bottom() - bh,
                                         hb.width() / n - 0.6, bh),
                           qc(WARM if mid > 0 else COLD, 210))
            g = np.exp(-0.5 * (0.5 * (self.edges[:-1] + self.edges[1:]) / self.sd) ** 2)
            g = g / g.max() * (hb.height() - 16)
            path = QtGui.QPainterPath()
            for i in range(n):
                bx = hb.left() + (i + 0.5) * hb.width() / n
                yy = hb.bottom() - g[i]
                path.lineTo(bx, yy) if i else path.moveTo(bx, yy)
            p.setPen(QtGui.QPen(qc(LIVE, 200), 1.2))
            p.drawPath(path)
            p.setPen(qc((113, 136, 143)))
            p.drawText(hb.left(), hb.top() - 10, "розподіл ваг проти гауса")
            ok = abs(self.sd - self.net.he_sd) / self.net.he_sd < 0.02
            p.setPen(qc(LIVE if ok else ROSE))
            p.drawText(hb.left() + 4, hb.top() + 13,
                       f"{'🔒' if ok else '🚨'} sd {self.sd:.6f}  теорія "
                       f"{self.net.he_sd:.6f}  ·  середнє {self.mean:+.2e}")

            cb = QtCore.QRect(rp.left(), hb.bottom() + 22, rp.width(),
                              rp.height() - hh - 22)
            p.fillRect(cb, qc(PANEL))
            col = self.net.W[self.layer][:, self.col]
            p.setPen(qc((113, 136, 143)))
            p.drawText(cb.left(), cb.top() - 10,
                       f"вхідні ваги нейрона {self.col} — усі {self.net.w} чисел")
            mid = cb.center().y()
            p.setPen(QtGui.QPen(qc(RULE), 1))
            p.drawLine(cb.left(), mid, cb.right(), mid)
            sc = (cb.height() * 0.44) / (4 * self.sd)
            step = max(1, self.net.w // max(1, cb.width()))
            for i in range(0, self.net.w, step):
                xx = cb.left() + i / self.net.w * cb.width()
                yy = mid - float(col[i]) * sc
                p.setPen(qc(WARM if col[i] > 0 else COLD, 150))
                p.drawLine(QtCore.QPointF(xx, mid), QtCore.QPointF(xx, yy))
            p.end()

        def mousePressEvent(self, ev):
            R = self.rect().adjusted(10, 26, -10, -10)
            side = min(R.height(), int(R.width() * 0.52))
            if R.left() <= ev.x() <= R.left() + side:
                self.col = int(max(0, min(self.net.w - 1,
                                          (ev.x() - R.left()) / side * self.net.w)))
                self.update()

    # ── ГРАФІК ОДНОГО НЕЙРОНА ───────────────────────────────────────────
    class Trace(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.hist = np.zeros(260, np.float32)
            self.n = 0
            self.label = "нейрон не обрано — клікни на карті або схемі"
            self.setFixedHeight(92)

        def push(self, v):
            self.hist[:-1] = self.hist[1:]
            self.hist[-1] = v
            self.n += 1
            self.update()

        def reset(self, label):
            self.hist[:] = 0
            self.n = 0
            self.label = label
            self.update()

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), qc(PANEL))
            W, H = self.width(), self.height()
            mid = H * 0.5
            p.setPen(QtGui.QPen(qc(RULE), 1))
            p.drawLine(0, int(mid), W, int(mid))
            if self.n:
                sc = H * 0.42 / 3.0
                path = QtGui.QPainterPath()
                for i, v in enumerate(self.hist):
                    x = i / (len(self.hist) - 1) * W
                    y = mid - max(-3.0, min(3.0, float(v))) * sc
                    path.lineTo(x, y) if i else path.moveTo(x, y)
                p.setPen(QtGui.QPen(qc(WARM), 1.4))
                p.drawPath(path)
            f = p.font()
            f.setPointSize(8)
            p.setFont(f)
            p.setPen(qc((113, 136, 143)))
            p.drawText(8, 15, self.label)
            p.setPen(qc((60, 78, 86)))
            p.drawText(W - 66, 15, "p/rms +3")
            p.drawText(W - 66, H - 6, "−3")
            p.end()

    # ── ГОЛОВНЕ ВІКНО ────────────────────────────────────────────────────
    class App(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.w, self.d = a.width, a.depth
            self.net = Net(self.w, self.d, a.seed)
            self.cnt = np.zeros((self.d, self.w), np.int64)
            self.tot = 0
            self.mode = 0
            self.sorted = False
            self.order = np.tile(np.arange(self.w), (self.d, 1))
            self.watch = None
            self.frames = 0
            self.t0 = time.time()
            self.build()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(33)

        def build(self):
            self.setWindowTitle(f"Жива мережа — {self.w}×{self.d}, seed {self.net.seed}")
            self.setStyleSheet("""
                QWidget{background:#0a0e11;color:#e3eaea;
                    font-family:'IBM Plex Sans','DejaVu Sans';font-size:12px}
                QPushButton{background:#182229;border:1px solid #1f2d36;
                    border-radius:3px;padding:6px 13px}
                QPushButton:hover{background:#1e2b33}
                QPushButton:checked{background:#22323a;border-color:#3d5763;color:#fff}
                QLabel#tag{color:#71888f;font-family:'IBM Plex Mono','DejaVu Sans Mono';
                    font-size:10px;letter-spacing:1px}
                QLabel#stat{color:#a8bcc3;
                    font-family:'IBM Plex Mono','DejaVu Sans Mono'}
                QTabBar::tab{background:#0e161b;color:#a8bcc3;padding:7px 20px;
                    border:1px solid #1f2d36;border-bottom:0}
                QTabBar::tab:selected{background:#22323a;color:#fff}
                QTabWidget::pane{border:1px solid #1f2d36;top:-1px}
                QSlider::groove:horizontal{height:3px;background:#1f2d36}
                QSlider::handle:horizontal{background:#f0a63c;width:11px;
                    margin:-5px 0;border-radius:2px}
                QSpinBox{background:#182229;border:1px solid #1f2d36;padding:3px 6px}
            """)
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(14, 12, 14, 12)
            v.setSpacing(9)
            v.addWidget(QtWidgets.QLabel(
                f"<b style='font-size:15px'>Жива мережа</b><span style='color:#71888f'>"
                f"   {self.w} × {self.d} · He-init без біасів · "
                f"кожен кадр — свіжий z ~ N(0,I)</span>"))

            self.tabs = QtWidgets.QTabWidget()
            self.map = MapView(self.w, self.d)
            self.graph = GraphView(self.net, a.nodes)
            self.wv = WeightView(self.net)
            self.map.picked.connect(self.on_pick)
            self.graph.picked.connect(self.on_pick)
            self.tabs.addTab(self.map, "Карта — усі 16 384")
            self.tabs.addTab(self.graph, "Схема — нейрони і ребра")
            self.tabs.addTab(self.wv, "Ваги — матриця W")
            v.addWidget(self.tabs, 1)

            c = QtWidgets.QHBoxLayout()
            c.setSpacing(7)
            self.bplay = QtWidgets.QPushButton("Пауза")
            self.bplay.setCheckable(True)
            self.bplay.setChecked(True)
            self.bplay.clicked.connect(
                lambda: self.bplay.setText("Пауза" if self.bplay.isChecked() else "Пуск"))
            c.addWidget(self.bplay)
            self.bmode = []
            for i, nm in enumerate(("Сигнал", "Ворота", "Замерзлі")):
                b = QtWidgets.QPushButton(nm)
                b.setCheckable(True)
                b.setChecked(i == 0)
                b.clicked.connect(lambda _, k=i: self.set_mode(k))
                self.bmode.append(b)
                c.addWidget(b)
            self.bsort = QtWidgets.QPushButton("Упорядкувати")
            self.bsort.setCheckable(True)
            self.bsort.clicked.connect(self.toggle_sort)
            c.addWidget(self.bsort)
            b = QtWidgets.QPushButton("Нова мережа")
            b.clicked.connect(self.reseed)
            c.addWidget(b)
            c.addSpacing(10)
            c.addWidget(self._tag("ШАР W"))
            self.spin = QtWidgets.QSpinBox()
            self.spin.setRange(0, self.d - 1)
            self.spin.valueChanged.connect(self.wv.set_layer)
            c.addWidget(self.spin)
            c.addWidget(self._tag("ТЕМП"))
            self.sfps = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.sfps.setRange(1, 60)
            self.sfps.setValue(12)
            self.sfps.setFixedWidth(96)
            c.addWidget(self.sfps)
            c.addWidget(self._tag("СЛІД"))
            self.sdec = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.sdec.setRange(0, 95)
            self.sdec.setValue(45)
            self.sdec.setFixedWidth(80)
            c.addWidget(self.sdec)
            c.addWidget(self._tag("ПОРІГ РЕБЕР"))
            self.sthr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.sthr.setRange(0, 80)
            self.sthr.setValue(25)
            self.sthr.setFixedWidth(80)
            self.sthr.valueChanged.connect(
                lambda x: setattr(self.graph, "thr", x / 100.0))
            c.addWidget(self.sthr)
            c.addStretch(1)
            v.addLayout(c)

            self.trace = Trace()
            v.addWidget(self.trace)
            self.stat = QtWidgets.QLabel("накопичую статистику…")
            self.stat.setObjectName("stat")
            v.addWidget(self.stat)
            self.resize(1280, 860)

        def _tag(self, t):
            lb = QtWidgets.QLabel(t)
            lb.setObjectName("tag")
            return lb

        def set_mode(self, k):
            self.mode = k
            for i, b in enumerate(self.bmode):
                b.setChecked(i == k)

        def toggle_sort(self):
            self.sorted = self.bsort.isChecked()
            self.order = (np.argsort(self.cnt, axis=1, kind="stable")
                          if self.sorted and self.tot
                          else np.tile(np.arange(self.w), (self.d, 1)))

        def reseed(self):
            s = int(np.random.default_rng().integers(1, 10 ** 7))
            self.net = Net(self.w, self.d, s)
            self.cnt[:] = 0
            self.tot = 0
            self.frames = 0
            self.t0 = time.time()
            self.watch = None
            self.graph.set_net(self.net)
            self.wv.set_net(self.net)
            self.trace.reset("нейрон не обрано")
            self.setWindowTitle(f"Жива мережа — {self.w}×{self.d}, seed {s}")

        def on_pick(self, l, j):
            self.watch = (l, j)
            self.wv.col = j
            self.spin.setValue(l)
            self.wv.update()
            self.trace.reset(f"шар {l} · нейрон {j}")

        def tick(self):
            if not self.bplay.isChecked():
                return
            n = max(1, self.sfps.value() // 6)
            z, P, H, G = self.net.step(n)
            self.cnt += G
            self.tot += n
            self.frames += 1
            rms = np.maximum(self.net.rms, 1e-12)[:, None]

            page = self.tabs.currentIndex()
            if page == 0:
                idx = np.clip((P / rms) * 32.0 + 128.0, 0, 255).astype(np.uint8)
                if self.sorted:
                    idx = np.take_along_axis(idx, self.order, axis=1)
                if self.mode == 0:
                    rgb = LUT_SIG[idx]
                else:
                    rgb = LUT_GATE[idx]
                    if self.mode == 2 and self.tot > 8:
                        fz = (self.cnt == 0) | (self.cnt == self.tot)
                        if self.sorted:
                            fz = np.take_along_axis(fz, self.order, axis=1)
                        rgb = (rgb.astype(np.float32) * 0.5)
                        rgb[fz] = ROSE
                        rgb = rgb.astype(np.uint8)
                self.map.push(rgb, self.sdec.value() / 100.0)
            elif page == 1:
                self.graph.feed(z, H)

            if self.watch:
                l, j = self.watch
                self.trace.push(float(P[l, j] / rms[l, 0]))
            if self.frames % 6 == 0:
                self.refresh_stat(P)

        def refresh_stat(self, P):
            f = self.cnt / max(1, self.tot)
            frozen = ((f <= 0.0) | (f >= 1.0)).mean() * 100.0
            fz_last = ((f[-1] <= 0.0) | (f[-1] >= 1.0)).mean() * 100.0
            sps = self.tot / max(1e-9, time.time() - self.t0)
            seal0 = f[0].mean()
            fz0 = int(((f[0] <= 0.0) | (f[0] >= 1.0)).sum())
            rms_ok = abs(float(self.net.rms[0]) / 1.4142135 - 1.0) < 0.05
            ok = "🔒" if abs(seal0 - 0.5) < 0.02 and fz0 == 0 and rms_ok else "🚨"
            self.stat.setText(
                f"семплів {self.tot}   ·   {sps:5.0f} семплів/с   ·   "
                f"відкрито зараз {(P > 0).mean() * 100:4.1f} %   ·   замерзлих "
                f"{frozen:4.1f} %   ·   на виході {fz_last:4.1f} %      "
                f"{ok} шар 0: ворота {seal0:.4f}, замерзлих {fz0} · "
                f"rms\u2080 {self.net.rms[0]:.3f}/\u221a2")

        def keyPressEvent(self, e):
            if e.key() == QtCore.Qt.Key_Space:
                self.bplay.setChecked(not self.bplay.isChecked())
                self.bplay.setText("Пауза" if self.bplay.isChecked() else "Пуск")
            elif e.key() == QtCore.Qt.Key_Escape:
                self.close()

    app = QtWidgets.QApplication(sys.argv)
    win = App()
    win.show()
    if a.smoke:                      # прогін усіх трьох видів без людини
        for i, nm in enumerate(("Карта", "Схема", "Ваги")):
            win.tabs.setCurrentIndex(i)
            for _ in range(4):
                win.tick()
            app.processEvents()
            pm = win.tabs.currentWidget().grab()
            print(f"🔒 вид «{nm}»: намальовано {pm.width()}×{pm.height()} px")
            assert pm.width() > 10 and pm.height() > 10
        win.on_pick(3, 77)
        for _ in range(6):
            win.tick()
        app.processEvents()
        print(f"🔒 клік по нейрону: історія {win.trace.n} точок, "
              f"W-стовпець {win.wv.col}, шар {win.wv.layer}")
        assert win.trace.n >= 5 and win.wv.col == 77
        win.bsort.setChecked(True); win.toggle_sort()
        win.set_mode(2); win.tabs.setCurrentIndex(0)
        for _ in range(3):
            win.tick()
        app.processEvents()
        print("🔒 режим «Замерзлі» + упорядкування: без винятків")
        print("✅ димовий прогін пройдено")
        return
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
