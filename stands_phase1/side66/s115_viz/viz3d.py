#!/usr/bin/env python3
"""White-Box · 3D-візуалізація стану мережі по шарах. Чотири вкладки.

1 «трійками»  точка = СЕМПЛ.  1024 → 341 трійка → вектор у R³
              ⊙ біле кільце = середнє по 6144 семплах
              ⊙ золоте кільце = ІСТИННЕ середнє (банк N=2²³) ⟹ видно похибку МК
2 «нейрони»   точка = НЕЙРОН. x значення · y увага по всіх · z сума топ-k схожих
3 «семпли»    точка = СЕМПЛ.  те саме, але схожість між СТАНАМИ семплів
4 «зони»      простір розбито на 32×32 = 1024 ЗОНИ, зона = ОДНЕ число ембедінга,
              і всередині зони стоять усі 16 шарів (вертикальний стовпчик).
              Канал обирається: m · істина · похибка · s · t · √dd · emp · h[0]

Пробіл пуск · ←/→ шар · Tab вкладка · F навести камеру
"""
import sys, numpy as np
sys.path.append("/home/trim/vizpkg")
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import cm

D1 = np.load("/home/trim/wblogs/viz3d.npz", allow_pickle=True)
P, NRM, COS = D1["P"].astype(np.float32), D1["NRM"], D1["COS"]
TRUEMU = D1["TRUEMU"].astype(np.float32)
NAME = str(D1["name"]); D, NS, _ = P.shape
MU = P.mean(axis=1)
CEN = [P[i] - MU[i] for i in range(D)]
COV = np.array([np.cov(c.T) for c in CEN])
SPREAD = np.array([np.sqrt((c*c).sum(1).mean()) for c in CEN], np.float32)
PERS = np.zeros(D, np.float32)
for i in range(D-1):
    a, b = CEN[i], CEN[i+1]
    PERS[i] = np.mean([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(3)])
PERS[D-1] = PERS[D-2]

D2 = np.load("/home/trim/wblogs/viz_attn.npz", allow_pickle=True)
AX, AY, AZ, AT = D2["X"], D2["Y"], D2["Z"], D2["T"]
KS2 = list(D2["KS"]); TAU2 = float(D2["tau"]); NSAMP = AX.shape[1] - 1

D3 = np.load("/home/trim/wblogs/viz_sattn.npz", allow_pickle=True)
SV, SY, SZ, SST = D3["V"], D3["Y"], D3["Z"], D3["ST"]
TAUS3 = list(D3["TAUS"]); KS3 = list(D3["KS"])

D4 = np.load("/home/trim/wblogs/viz_zone.npz", allow_pickle=True)
CHAN = D4["CHAN"].astype(np.float32)                 # (NCH, 16, 1024)
CHNAME = [str(x) for x in D4["NAMES"]]
CHSIGN = np.asarray(D4["SIGNED"], bool)
ORDS = {"індекс (сирий)": None, "ранг |m| у своєму шарі": D4["ORD_SELF"],
        "ранг |m| за шаром 15": D4["ORD_L15"], "ранг t у своєму шарі": D4["ORD_T"]}
ORDK = list(ORDS.keys())
GZ = 32                                              # 32×32 зон
IDX = np.arange(GZ*GZ)
ZX = (IDX // GZ).astype(np.float32)                  # колонка зони
ZY = (IDX %  GZ).astype(np.float32)                  # рядок зони
STEPS = 60

def lerp(a, b, f): return a*(1.0-f) + b*f
def _sqrtm(C, inv=False):
    ev, U = np.linalg.eigh(C); ev = np.maximum(ev, 1e-12)
    s = 1.0/np.sqrt(ev) if inv else np.sqrt(ev)
    return (U*s) @ U.T
def rings(c, r, n=64):
    th = np.linspace(0, 2*np.pi, n); cs, sn, z0 = np.cos(th), np.sin(th), np.zeros(n)
    seg = []
    for a in (np.stack([cs,sn,z0],1), np.stack([cs,z0,sn],1), np.stack([z0,cs,sn],1)):
        p = np.asarray(c, np.float64) + r*a
        seg.append(np.repeat(p, 2, axis=0)[1:-1])
    return np.concatenate(seg, 0).astype(np.float32)

class Viz(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("White-Box · %s · %d семплів · 16 шарів" % (NAME, NS))
        self.resize(1650, 1000)
        self.pos = 0.0; self.playing = False
        self.sub = np.linspace(0, NS-1, 48).astype(int)
        self._last = [None]*4; self._fitted = [False]*4

        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root); outer.setContentsMargins(6,6,6,6)
        self.tabs = QtWidgets.QTabWidget(); outer.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._ontab)
        self.tabs.addTab(self._tab1(), "  трійками · семпли  ")
        self.tabs.addTab(self._tab2(), "  нейрони · увага  ")
        self.tabs.addTab(self._tab3(), "  семпли · увага  ")
        self.tabs.addTab(self._tab4(), "  зони · 16 шарів  ")

        bar = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("▶  пуск"); self.btn.setFixedWidth(110)
        self.btn.clicked.connect(self.toggle); bar.addWidget(self.btn)
        b2 = QtWidgets.QPushButton("⟲"); b2.setFixedWidth(40)
        b2.clicked.connect(lambda: self.slider.setValue(0)); bar.addWidget(b2)
        b3 = QtWidgets.QPushButton("⌖ навести камеру (F)"); b3.setFixedWidth(180)
        b3.clicked.connect(lambda: self.fit()); bar.addWidget(b3)
        bar.addWidget(QtWidgets.QLabel("шар"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, (D-1)*STEPS); self.slider.valueChanged.connect(self.onslide)
        bar.addWidget(self.slider, 1)
        bar.addWidget(QtWidgets.QLabel("швидкість"))
        self.spd = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.spd.setFixedWidth(120)
        self.spd.setRange(1, 40); self.spd.setValue(10); bar.addWidget(self.spd)
        outer.addLayout(bar)
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.tick); self.timer.start(16)
        self.refresh()

    # ── допоміжне ──
    def _view(self, dist, elev, az, gsize, gsp):
        v = gl.GLViewWidget(); v.setBackgroundColor(pg.mkColor(12,13,18))
        v.setCameraPosition(distance=dist, elevation=elev, azimuth=az)
        g = gl.GLGridItem(); g.setSize(gsize,gsize); g.setSpacing(gsp,gsp)
        g.setColor((70,75,95,85)); v.addItem(g)
        a = gl.GLAxisItem(); a.setSize(gsize*0.22,gsize*0.22,gsize*0.22); v.addItem(a)
        return v, g
    def _hud(self):
        h = QtWidgets.QLabel(); h.setFont(QtGui.QFont("monospace", 10))
        h.setStyleSheet("color:#cfe3ff;background:#161822;padding:9px;border-radius:6px")
        h.setTextFormat(QtCore.Qt.RichText); return h
    def _note(self, t):
        n = QtWidgets.QLabel(t); n.setWordWrap(True)
        n.setStyleSheet("color:#8d93a8;font-size:11px"); return n
    def _sizer(self, sl, v=45):
        sl.addWidget(QtWidgets.QLabel("розмір точки"))
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(10,140); s.setValue(v)
        s.valueChanged.connect(self.refresh); sl.addWidget(s); return s
    def colors(self, v, alpha=0.72):
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        t = np.clip((v-lo)/max(hi-lo, 1e-9), 0, 1)
        c = cm.turbo(t).astype(np.float32); c[:,3] = alpha; return c

    # ───────────── вкладка 1 ─────────────
    def _tab1(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        self.v1, self.g1 = self._view(115, 18, 45, 200, 10); lay.addWidget(self.v1, 1)
        self.s1 = gl.GLScatterPlotItem(pos=np.zeros((NS,3),np.float32), size=4.2, pxMode=True)
        self.v1.addItem(self.s1)
        self.ring1 = gl.GLLinePlotItem(mode='lines', color=(1,1,1,0.95), width=2.2, antialias=True)
        self.v1.addItem(self.ring1)
        self.ring1t = gl.GLLinePlotItem(mode='lines', color=(1.0,0.82,0.25,0.95), width=2.6, antialias=True)
        self.v1.addItem(self.ring1t)
        self.r1 = gl.GLLinePlotItem(mode='lines', color=(0.55,0.72,1.0,0.10), width=1.0, antialias=True)
        self.r1.setVisible(False); self.v1.addItem(self.r1)
        self.t1 = gl.GLLinePlotItem(mode='lines', color=(1.0,0.80,0.35,0.75), width=1.6, antialias=True)
        self.t1.setVisible(False); self.v1.addItem(self.t1)

        side = QtWidgets.QWidget(); side.setFixedWidth(300)
        sl = QtWidgets.QVBoxLayout(side); sl.setSpacing(8)
        self.hud1 = self._hud(); sl.addWidget(self.hud1)
        self.cbC = QtWidgets.QCheckBox("центрувати"); self.cbC.setChecked(True)
        self.cbC.stateChanged.connect(self._refit); sl.addWidget(self.cbC)
        self.cbP = QtWidgets.QCheckBox("прибрати пульс (повна коваріація)"); self.cbP.setChecked(True)
        self.cbP.stateChanged.connect(self.refresh); sl.addWidget(self.cbP)
        self.cbR = QtWidgets.QCheckBox("промені від центра"); self.cbR.stateChanged.connect(self.refresh)
        sl.addWidget(self.cbR)
        self.cbT = QtWidgets.QCheckBox("сліди 48 семплів"); self.cbT.stateChanged.connect(self.refresh)
        sl.addWidget(self.cbT)
        self.cmb1 = QtWidgets.QComboBox(); self.cmb1.addItems(["колір: cos(h,середнє)","колір: ‖h‖","колір: сталий"])
        self.cmb1.currentIndexChanged.connect(self.refresh); sl.addWidget(self.cmb1)
        sl.addWidget(QtWidgets.QLabel("розмір кілець"))
        self.rsz = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.rsz.setRange(2,120); self.rsz.setValue(18)
        self.rsz.valueChanged.connect(self.refresh); sl.addWidget(self.rsz)
        self.psz1 = self._sizer(sl, 42)
        sl.addWidget(self._note("⊙ біле — середнє по 6144 семплах\n"
            "⊙ золоте — ІСТИННЕ середнє (банк N=2²³)\n\n"
            "Розрив між ними = похибка МК. Зміряно: 1.0–1.8% розкиду\n"
            "хмари при теоретичних 1/√6144 = 1.28%. Зменш кільця\n"
            "і наблизь колесом, щоб побачити."))
        sl.addStretch(1); lay.addWidget(side); return w

    # ───────────── вкладка 2 ─────────────
    def _tab2(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        self.v2, self.g2 = self._view(13, 20, 40, 24, 2); lay.addWidget(self.v2, 1)
        self.s2 = gl.GLScatterPlotItem(pos=np.zeros((1024,3),np.float32), size=5.0, pxMode=True)
        self.v2.addItem(self.s2)
        self.s2b = gl.GLScatterPlotItem(pos=np.zeros((1024,3),np.float32), size=5.0, pxMode=True)
        self.s2b.setVisible(False); self.v2.addItem(self.s2b)
        self.ring2 = gl.GLLinePlotItem(mode='lines', color=(1,1,1,0.95), width=2.2, antialias=True)
        self.v2.addItem(self.ring2)
        side = QtWidgets.QWidget(); side.setFixedWidth(300)
        sl = QtWidgets.QVBoxLayout(side); sl.setSpacing(8)
        self.hud2 = self._hud(); sl.addWidget(self.hud2)
        sl.addWidget(QtWidgets.QLabel("семпл"))
        self.smp = QtWidgets.QComboBox(); self.smp.addItem("середнє по всіх")
        for i in range(NSAMP): self.smp.addItem("семпл %d" % i)
        self.smp.currentIndexChanged.connect(self.refresh); sl.addWidget(self.smp)
        sl.addWidget(QtWidgets.QLabel("k у топ-k"))
        self.kk = QtWidgets.QComboBox()
        for k in KS2: self.kk.addItem("k = %d" % k)
        self.kk.setCurrentIndex(1); self.kk.currentIndexChanged.connect(self._refit); sl.addWidget(self.kk)
        self.cmb2 = QtWidgets.QComboBox()
        self.cmb2.addItems(["колір: t = m/s нейрона","колір: власне значення","колір: сталий"])
        self.cmb2.currentIndexChanged.connect(self.refresh); sl.addWidget(self.cmb2)
        self.cbN = QtWidgets.QCheckBox("нормувати осі"); self.cbN.setChecked(True)
        self.cbN.stateChanged.connect(self._refit); sl.addWidget(self.cbN)
        self.cbF = QtWidgets.QCheckBox("перехід розчиненням"); self.cbF.setChecked(True)
        self.cbF.stateChanged.connect(self.refresh); sl.addWidget(self.cbF)
        self.psz2 = self._sizer(sl)
        sl.addWidget(self._note("Точка = НЕЙРОН (1024).\nСхожість — cos вхідних ваг W[:,j].\n"
            "A = softmax(cos/τ), τ = %.3f ⟹ ~12 сусідів.\n\n"
            "🚨 індекс нейрона МІЖ шарами нічого не значить\n(corr ≈ 0) ⟹ перехід розчиненням,\n"
            "а не морфінгом." % TAU2))
        sl.addStretch(1); lay.addWidget(side); return w

    # ───────────── вкладка 3 ─────────────
    def _tab3(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        self.v3, self.g3 = self._view(14, 20, 40, 24, 2); lay.addWidget(self.v3, 1)
        self.s3 = gl.GLScatterPlotItem(pos=np.zeros((NS,3),np.float32), size=4.0, pxMode=True)
        self.v3.addItem(self.s3)
        self.ring3 = gl.GLLinePlotItem(mode='lines', color=(1,1,1,0.95), width=2.4, antialias=True)
        self.v3.addItem(self.ring3)
        side = QtWidgets.QWidget(); side.setFixedWidth(300)
        sl = QtWidgets.QVBoxLayout(side); sl.setSpacing(8)
        self.hud3 = self._hud(); sl.addWidget(self.hud3)
        sl.addWidget(QtWidgets.QLabel("власне значення семпла"))
        self.vv = QtWidgets.QComboBox(); self.vv.addItems(["⟨h, m̂⟩ — проєкція на середнє", "‖h‖ — норма"])
        self.vv.currentIndexChanged.connect(self._refit); sl.addWidget(self.vv)
        sl.addWidget(QtWidgets.QLabel("τ′ (на Z-нормованому cos)"))
        self.tt = QtWidgets.QComboBox()
        for i, t in enumerate(TAUS3): self.tt.addItem("τ′ = %.1f  (~%.0f сусідів)" % (t, SST[15,2+i]))
        self.tt.currentIndexChanged.connect(self._refit); sl.addWidget(self.tt)
        sl.addWidget(QtWidgets.QLabel("k у топ-k"))
        self.k3 = QtWidgets.QComboBox()
        for k in KS3: self.k3.addItem("k = %d" % k)
        self.k3.setCurrentIndex(1); self.k3.currentIndexChanged.connect(self._refit); sl.addWidget(self.k3)
        self.cmb3 = QtWidgets.QComboBox()
        self.cmb3.addItems(["колір: cos(h,середнє)","колір: власне значення","колір: сталий"])
        self.cmb3.currentIndexChanged.connect(self.refresh); sl.addWidget(self.cmb3)
        self.cbN3 = QtWidgets.QCheckBox("нормувати осі"); self.cbN3.setChecked(True)
        self.cbN3.stateChanged.connect(self._refit); sl.addWidget(self.cbN3)
        self.cbC3 = QtWidgets.QCheckBox("центрувати (відняти середнє)")
        self.cbC3.stateChanged.connect(self._refit); sl.addWidget(self.cbC3)
        self.psz3 = self._sizer(sl, 40)
        sl.addWidget(self._note("Точка = СЕМПЛ (6144). ⊙ біле кільце = середнє.\n"
            "Схожість — cos(h_s,h_r) у 1024-вимірному стані.\n\n"
            "🚨 сирий cos на глибині зібраний біля 0.92 (sd 0.013)\n"
            "⟹ фіксована τ вироджується. Тому Z-нормування:\n"
            "logit = (cos−сер)/sd/τ′. Так шари порівнянні."))
        sl.addStretch(1); lay.addWidget(side); return w

    # ───────────── вкладка 4 · зони ─────────────
    def _tab4(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        self.v4, self.g4 = self._view(70, 22, 40, 60, 4); lay.addWidget(self.v4, 1)
        self.g4.setVisible(False)
        self.s4 = gl.GLScatterPlotItem(pos=np.zeros((GZ*GZ*D,3),np.float32), size=3.0, pxMode=True)
        self.v4.addItem(self.s4)
        self.l4 = gl.GLLinePlotItem(mode='lines', color=(0.60,0.72,1.0,0.16), width=1.0, antialias=True)
        self.l4.setVisible(False); self.v4.addItem(self.l4)
        self.lat4 = gl.GLLinePlotItem(mode='lines', color=(0.42,0.48,0.66,0.42), width=1.0, antialias=True)
        self.v4.addItem(self.lat4)
        self.cur4 = gl.GLLinePlotItem(mode='lines', color=(1.0,0.84,0.30,0.55), width=1.4, antialias=True)
        self.v4.addItem(self.cur4)

        side = QtWidgets.QWidget(); side.setFixedWidth(300)
        sl = QtWidgets.QVBoxLayout(side); sl.setSpacing(8)
        self.hud4 = self._hud(); sl.addWidget(self.hud4)
        sl.addWidget(QtWidgets.QLabel("канал — що саме за число"))
        self.ch4 = QtWidgets.QComboBox(); self.ch4.addItems(CHNAME)
        self.ch4.currentIndexChanged.connect(self._refit); sl.addWidget(self.ch4)
        sl.addWidget(QtWidgets.QLabel("порядок зон"))
        self.or4 = QtWidgets.QComboBox(); self.or4.addItems(ORDK)
        self.or4.setCurrentIndex(1); self.or4.currentIndexChanged.connect(self._refit)
        sl.addWidget(self.or4)
        self.nr4 = QtWidgets.QComboBox()
        self.nr4.addItems(["шкала: спільна на всі шари", "шкала: своя на кожен шар"])
        self.nr4.currentIndexChanged.connect(self.refresh); sl.addWidget(self.nr4)
        sl.addWidget(QtWidgets.QLabel("рельєф (зсув усередині зони)"))
        self.rl4 = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.rl4.setRange(0,100); self.rl4.setValue(55)
        self.rl4.valueChanged.connect(self.refresh); sl.addWidget(self.rl4)
        sl.addWidget(QtWidgets.QLabel("розсув шарів по висоті"))
        self.dz4 = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.dz4.setRange(4,60); self.dz4.setValue(14)
        self.dz4.valueChanged.connect(self._refit); sl.addWidget(self.dz4)
        self.cbL4 = QtWidgets.QCheckBox("сполучити шари в зоні (нитки)")
        self.cbL4.stateChanged.connect(self.refresh); sl.addWidget(self.cbL4)
        self.cbH4 = QtWidgets.QCheckBox("підсвітити поточний шар"); self.cbH4.setChecked(True)
        self.cbH4.stateChanged.connect(self.refresh); sl.addWidget(self.cbH4)
        self.cbA4 = QtWidgets.QCheckBox("решітка на КОЖНОМУ шарі")
        self.cbA4.stateChanged.connect(self.refresh); sl.addWidget(self.cbA4)
        self.psz4 = self._sizer(sl, 30)
        self.note4 = self._note("")
        sl.addWidget(self.note4)
        sl.addStretch(1); lay.addWidget(side); return w

    def _lattice(self, levels, dz):
        seg = []
        e = np.arange(GZ+1, dtype=np.float32) - 0.5
        for z in levels:
            for u in e:
                seg += [[u, -0.5, z], [u, GZ-0.5, z]]
                seg += [[-0.5, u, z], [GZ-0.5, u, z]]
        return np.asarray(seg, np.float32)

    def _norm4(self, v, signed, lo, hi):
        if signed:
            r = max(abs(lo), abs(hi), 1e-12)
            return np.clip(0.5 + 0.5*v/r, 0, 1)
        return np.clip((v-lo)/max(hi-lo, 1e-12), 0, 1)

    def draw4(self, i, j, f):
        ci = self.ch4.currentIndex(); signed = bool(CHSIGN[ci])
        A = CHAN[ci]                                   # (16,1024)
        perm = ORDS[ORDK[self.or4.currentIndex()]]
        V = A if perm is None else np.take_along_axis(A, perm, axis=1)
        dz = self.dz4.value()/10.0
        rel = self.rl4.value()/100.0*0.86

        if self.nr4.currentIndex() == 0:
            lo, hi = np.percentile(V, 0.5), np.percentile(V, 99.5)
            U = self._norm4(V, signed, lo, hi)
        else:
            U = np.empty_like(V)
            for L in range(D):
                a, b = np.percentile(V[L], 0.5), np.percentile(V[L], 99.5)
                U[L] = self._norm4(V[L], signed, a, b)

        off = (U - 0.5)*rel                            # зсув усередині зони по x
        pts = np.empty((D, GZ*GZ, 3), np.float32)
        pts[:, :, 0] = ZX[None, :] + off
        pts[:, :, 1] = ZY[None, :]
        pts[:, :, 2] = (np.arange(D, dtype=np.float32)*dz)[:, None]
        flat = pts.reshape(-1, 3)
        self._last[3] = flat
        if not self._fitted[3]: self._fitted[3] = True; QtCore.QTimer.singleShot(0, self.fit)

        cmap = cm.coolwarm if signed else cm.turbo
        col = cmap(U.reshape(-1)).astype(np.float32)
        Lc = i if f < 0.5 else j
        if self.cbH4.isChecked():
            al = np.full(D, 0.30, np.float32); al[Lc] = 1.0
            if Lc > 0: al[Lc-1] = 0.55
            if Lc < D-1: al[Lc+1] = 0.55
            col[:, 3] = np.repeat(al, GZ*GZ)
            sz = np.repeat(np.where(np.arange(D) == Lc, 2.2, 1.0).astype(np.float32), GZ*GZ)
            sz = sz*self.psz4.value()/10.0
        else:
            col[:, 3] = 0.78; sz = self.psz4.value()/10.0
        self.s4.setData(pos=flat, color=col, size=sz)

        if self.cbL4.isChecked():
            seg = np.empty(((D-1)*GZ*GZ*2, 3), np.float32)
            seg[0::2] = pts[:-1].reshape(-1, 3); seg[1::2] = pts[1:].reshape(-1, 3)
            self.l4.setData(pos=seg); self.l4.setVisible(True)
        else: self.l4.setVisible(False)

        lv = list(np.arange(D)*dz) if self.cbA4.isChecked() else [-0.6*dz, (D-1+0.6)*dz]
        self.lat4.setData(pos=self._lattice(lv, dz))
        self.cur4.setData(pos=self._lattice([self.pos*dz], dz))

        v = V[Lc]
        zero = 100.0*float((np.abs(A[Lc]) <= 1e-12).mean())
        raw = perm is None
        self.note4.setText(
            ("🚨 порядок СИРИЙ: зона j на шарі L і на L+1 — РІЗНІ нейрони\n"
             "(corr стовпців W ≈ 0, зміряно С115) ⟹ вертикаль тут не має\n"
             "сенсу як «еволюція одного числа». Це чесна картинка розподілу,\n"
             "але не траєкторії.\n\n" if raw else
             "зона = РАНГ у своєму шарі ⟹ вертикаль порівнянна: видно, як\n"
             "міняється ПРОФІЛЬ розподілу з глибиною, а не доля нейрона.\n\n")
            + "1024 зони = 32×32 · у кожній 16 рівнів = 16 шарів\n"
              "колір і зсув усередині зони = значення каналу")
        self.note4.setStyleSheet("color:%s;font-size:11px" % ("#e08f8f" if raw else "#8d93a8"))
        self.hud4.setText(
            "<b style='color:#fff;font-size:13px'>шар %5.2f</b> / 15<br><br>"
            "<span style='color:#ffd24a'>%s</span><br><br>"
            "мін <b>%+.4g</b><br>медіана <b>%+.4g</b><br>макс <b>%+.4g</b><br>"
            "σ <b>%.4g</b><br><br>нулів (|v|≤1e-12) <b>%.1f%%</b><br><br>"
            "<span style='color:#9fb8d8'>зон 1024 · рівнів 16<br>точок %d</span>"
            % (self.pos, CHNAME[ci], float(v.min()), float(np.median(v)), float(v.max()),
               float(v.std()), zero, GZ*GZ*D))

    # ───────────── камера ─────────────
    def fit(self, pts=None):
        ti = self.tabs.currentIndex()
        view, grid = [(self.v1,self.g1),(self.v2,self.g2),(self.v3,self.g3),(self.v4,self.g4)][ti]
        if pts is None: pts = self._last[ti]
        if pts is None or len(pts) == 0: return
        c = pts.mean(0)
        r = float(np.percentile(np.linalg.norm(pts-c, axis=1), 97)) or 1.0
        view.opts["center"] = pg.Vector(float(c[0]), float(c[1]), float(c[2]))
        view.setCameraPosition(distance=r*3.4)
        grid.resetTransform(); grid.setSize(r*7, r*7); grid.setSpacing(r*0.7, r*0.7)
        grid.translate(float(c[0]), float(c[1]), float(c[2])-r*1.8); view.update()

    def refresh(self):
        i = int(np.floor(self.pos)); j = min(i+1, D-1); f = self.pos - i
        [self.draw1, self.draw2, self.draw3, self.draw4][self.tabs.currentIndex()](i, j, f)

    def _mark(self, ti, item, c, r):
        item.setData(pos=rings(c, r))
        self._last[ti] = self._last[ti]

    # ───────────── малювання ─────────────
    def draw1(self, i, j, f):
        p = lerp(P[i], P[j], f); mu = lerp(MU[i], MU[j], f)
        tm = lerp(TRUEMU[i], TRUEMU[j], f)
        c = lerp(COS[i], COS[j], f); n = lerp(NRM[i], NRM[j], f)
        b = p - p.mean(0)
        if self.cbP.isChecked() and i != j:
            M = _sqrtm(np.cov(b.T), inv=True) @ _sqrtm(lerp(COV[i], COV[j], f))
            b = (b @ M).astype(np.float32)
        cc = self.cbC.isChecked()
        base = b if cc else b + mu
        mpt = np.zeros(3, np.float32) if cc else mu
        tpt = (tm - mu) if cc else tm
        self._last[0] = base
        if not self._fitted[0]: self._fitted[0] = True; QtCore.QTimer.singleShot(0, self.fit)
        m = self.cmb1.currentIndex()
        col = self.colors(c if m == 0 else n) if m < 2 else \
              np.tile(np.array([[0.45,0.72,1.0,0.62]],np.float32),(NS,1))
        self.s1.setData(pos=base, color=col, size=self.psz1.value()/10.0)
        rr = SPREAD[i]*self.rsz.value()/1000.0
        self.ring1.setData(pos=rings(mpt, rr)); self.ring1t.setData(pos=rings(tpt, rr*0.8))
        if self.cbR.isChecked():
            seg = np.empty((NS*2,3), np.float32); seg[0::2] = mpt; seg[1::2] = base
            self.r1.setData(pos=seg); self.r1.setVisible(True)
        else: self.r1.setVisible(False)
        if self.cbT.isChecked():
            pts = []
            for s in self.sub:
                tr = P[:, s, :] - (MU if cc else 0.0)
                for k in range(D-1): pts += [tr[k], tr[k+1]]
            self.t1.setData(pos=np.asarray(pts, np.float32)); self.t1.setVisible(True)
        else: self.t1.setVisible(False)
        d = base - base.mean(0); ev = np.linalg.eigvalsh(np.cov(d.T))
        gap = float(np.linalg.norm(tm - mu))
        self.hud1.setText(
            "<b style='color:#fff;font-size:13px'>шар %5.2f</b> / 15<br><br>"
            "cos(h,середнє) <b>%.4f</b><br>кут <b>%.1f°</b><br>‖h‖ <b>%.1f</b><br>"
            "розкид 3D <b>%.2f</b><br>анізотропія <b>%.2f</b><br><br>"
            "<span style='color:#ffd24a'>⊙ похибка МК<br>‖емпір−істина‖ <b>%.4f</b><br>"
            "= <b>%.2f%%</b> розкиду<br>теорія 1/√N = 1.28%%</span><br><br>"
            "<span style='color:#9fb8d8'>сталість шар→шар <b>%+.2f</b></span>"
            % (self.pos, c.mean(), np.degrees(np.arccos(np.clip(c.mean(),-1,1))), n.mean(),
               float(np.sqrt((d*d).sum(1).mean())), ev[-1]/max(ev[0],1e-9),
               gap, 100*gap/SPREAD[i], float(lerp(PERS[i], PERS[j], f))))

    def _cloud2(self, L, s, ki):
        x, y = AX[L,s], AY[L,s]; z = AZ[L,s,:,ki]
        pts = np.stack([x,y,z],1).astype(np.float32)
        if self.cbN.isChecked():
            sg = np.array([max(x.std(),1e-9),max(y.std(),1e-9),max(z.std(),1e-9)],np.float32)
            pts = pts/sg
        m = self.cmb2.currentIndex()
        col = self.colors(AT[L] if m==0 else x) if m<2 else \
              np.tile(np.array([[0.45,0.72,1.0,0.7]],np.float32),(1024,1))
        return pts, col, x

    def draw2(self, i, j, f):
        s = self.smp.currentIndex(); ki = self.kk.currentIndex(); sz = self.psz2.value()/10.0
        if self.cbF.isChecked():
            pa, ca, _ = self._cloud2(i, s, ki); ca = ca.copy(); ca[:,3] *= max(1.0-f,0.0)
            self.s2.setData(pos=pa, color=ca, size=sz); self._last[1] = pa
            if i != j and f > 1e-3:
                pb, cb, _ = self._cloud2(j, s, ki); cb = cb.copy(); cb[:,3] *= f
                self.s2b.setData(pos=pb, color=cb, size=sz); self.s2b.setVisible(True)
            else: self.s2b.setVisible(False)
            L = i if f < 0.5 else j
            x, y, z, t = AX[L,s], AY[L,s], AZ[L,s,:,ki], AT[L]; pts = pa
        else:
            self.s2b.setVisible(False)
            x = lerp(AX[i,s],AX[j,s],f); y = lerp(AY[i,s],AY[j,s],f)
            z = lerp(AZ[i,s,:,ki],AZ[j,s,:,ki],f); t = lerp(AT[i],AT[j],f)
            pts = np.stack([x,y,z],1).astype(np.float32)
            if self.cbN.isChecked():
                sg = np.array([max(x.std(),1e-9),max(y.std(),1e-9),max(z.std(),1e-9)],np.float32)
                pts = pts/sg
            m = self.cmb2.currentIndex()
            col = self.colors(t if m==0 else x) if m<2 else \
                  np.tile(np.array([[0.45,0.72,1.0,0.7]],np.float32),(1024,1))
            self.s2.setData(pos=pts, color=col, size=sz); self._last[1] = pts
        if not self._fitted[1]: self._fitted[1] = True; QtCore.QTimer.singleShot(0, self.fit)
        r = float(np.percentile(np.linalg.norm(pts-pts.mean(0),axis=1),97))
        self.ring2.setData(pos=rings(pts.mean(0), r*0.05))
        self.hud2.setText(
            "<b style='color:#fff;font-size:13px'>шар %5.2f</b> / 15<br><br>%s · k = %d<br><br>"
            "мертвих (h=0) <b>%.1f%%</b><br>corr(знач,увага) <b>%+.3f</b><br>"
            "corr(знач,топ-k) <b>%+.3f</b><br><br>σ знач <b>%.3f</b><br>σ увага <b>%.3f</b><br>"
            "σ топ-k <b>%.3f</b><br><br><span style='color:#9fb8d8'>медіана |t| <b>%.2f</b></span>"
            % (self.pos, self.smp.currentText(), KS2[ki], 100*float((x<=1e-9).mean()),
               float(np.corrcoef(x,y)[0,1]), float(np.corrcoef(x,z)[0,1]),
               x.std(), y.std(), z.std(), float(np.median(np.abs(t)))))

    def draw3(self, i, j, f):
        vi = self.vv.currentIndex(); ti_ = self.tt.currentIndex(); ki = self.k3.currentIndex()
        x = lerp(SV[i,:,vi], SV[j,:,vi], f)
        y = lerp(SY[i,:,ti_,vi], SY[j,:,ti_,vi], f)
        z = lerp(SZ[i,:,ki,vi], SZ[j,:,ki,vi], f)
        cs = lerp(COS[i], COS[j], f)
        pts = np.stack([x,y,z],1).astype(np.float64)
        if self.cbC3.isChecked(): pts = pts - pts.mean(0)
        if self.cbN3.isChecked():
            sg = pts.std(0); sg[sg < 1e-12] = 1.0; pts = pts/sg
        pts = pts.astype(np.float32)
        m = self.cmb3.currentIndex()
        col = self.colors(cs if m==0 else x) if m<2 else \
              np.tile(np.array([[0.45,0.72,1.0,0.55]],np.float32),(NS,1))
        self.s3.setData(pos=pts, color=col, size=self.psz3.value()/10.0)
        self._last[2] = pts
        if not self._fitted[2]: self._fitted[2] = True; QtCore.QTimer.singleShot(0, self.fit)
        c = pts.mean(0); r = float(np.percentile(np.linalg.norm(pts-c,axis=1),97))
        self.ring3.setData(pos=rings(c, r*0.06))
        L = i if f < 0.5 else j
        self.hud3.setText(
            "<b style='color:#fff;font-size:13px'>шар %5.2f</b> / 15<br><br>"
            "%s<br>τ′ = %.1f · k = %d<br><br>"
            "cos(h_s,h_r) <b>%.4f</b> ± <b>%.4f</b><br>ефект. сусідів <b>%.0f</b><br><br>"
            "corr(знач,увага) <b>%+.3f</b><br>corr(знач,топ-k) <b>%+.3f</b><br>"
            "corr(увага,топ-k) <b>%+.3f</b>"
            % (self.pos, "проєкція ⟨h,m̂⟩" if vi==0 else "норма ‖h‖",
               TAUS3[ti_], KS3[ki], SST[L,0], SST[L,1], SST[L,2+ti_],
               float(np.corrcoef(x,y)[0,1]), float(np.corrcoef(x,z)[0,1]),
               float(np.corrcoef(y,z)[0,1])))

    def _ontab(self, _):
        if not hasattr(self, "v4"): return      # ще будуємось: addTab смикає сигнал
        self.refresh(); self.fit()
    def _refit(self): self.refresh(); QtCore.QTimer.singleShot(0, self.fit)
    def onslide(self, v): self.pos = v/float(STEPS); self.refresh()
    def toggle(self):
        self.playing = not self.playing
        self.btn.setText("❚❚  пауза" if self.playing else "▶  пуск")
    def tick(self):
        if not self.playing: return
        v = self.slider.value() + max(1, self.spd.value()//3)
        self.slider.setValue(0 if v > self.slider.maximum() else v)
    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Space: self.toggle()
        elif k == QtCore.Qt.Key_Right: self.slider.setValue(min(self.slider.maximum(), self.slider.value()+STEPS))
        elif k == QtCore.Qt.Key_Left:  self.slider.setValue(max(0, self.slider.value()-STEPS))
        elif k == QtCore.Qt.Key_F: self.fit()
        elif k == QtCore.Qt.Key_Tab: self.tabs.setCurrentIndex((self.tabs.currentIndex()+1) % 4)

app = QtWidgets.QApplication(sys.argv); app.setStyle("Fusion")
pal = QtGui.QPalette()
for r_, c_ in ((QtGui.QPalette.Window,(24,26,34)),(QtGui.QPalette.WindowText,(210,218,235)),
               (QtGui.QPalette.Base,(18,20,27)),(QtGui.QPalette.Button,(38,42,55)),
               (QtGui.QPalette.ButtonText,(215,225,240))):
    pal.setColor(r_, QtGui.QColor(*c_))
app.setPalette(pal)
v = Viz(); v.show(); sys.exit(app.exec_())
