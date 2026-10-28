"""NETVIEW — мережа як КАРТА, якою рухаються (мова стара, з viz_app.py).

Що взято зі старої версії:
  · розмір нейрона кодує значення, не лише колір
  · ребра ЗНАКОВІ: додатні блакитним, від'ємні рожевим, прозорість за силою
  · між колонками — САМІ МАТРИЦІ ВАГ тепловими квадратами (червоне +, синє −)
  · просторі координати (CX=340, CY=22): це карта, а не картинка в екран
  · відсікання за видимою областю, мінімапа, підказка під курсором

Що показуємо своє (знахідка С47):
  розмір і яскравість нейрона = ЖИВІСТЬ 2·min(p,1−p). Прибитий нейрон = дрібна тьмяна цятка,
  живий = велика яскрава. Тому мережа на очах «худне» з глибиною.
  Відтінок = куди прибитий: блакитний — переважно пропускає, рожевий — переважно ріже.
"""
import numpy as np, torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QImage

CX, CY = 340.0, 22.0
BG   = QColor('#0B0E14'); LINE = QColor('#1c2432'); MUT = QColor('#7D8590')
INK  = QColor('#E2E8F0'); OPEN = QColor('#39BAE6'); SHUT = QColor('#F07178')
GOLD = QColor('#FFB454'); PANEL= QColor('#11161f')
TOPK = 4

def weight_qimage(Wl):
    """Матриця ваг → теплокарта: додатні червоним, від'ємні синім, нуль темний."""
    n = Wl.shape[0]
    a = Wl / (np.abs(Wl).max() + 1e-12)
    img = np.zeros((n, n, 4), np.uint8); img[..., 3] = 255
    base = np.array([20, 24, 33], np.float32)
    pos  = np.array([88, 74, 199], np.float32)      # BGR: червонястий
    neg  = np.array([189, 144, 30], np.float32)     # BGR: синій
    m = np.clip(np.abs(a) * 2.2, 0, 1)[..., None]
    col = np.where(a[..., None] >= 0, pos, neg)
    img[..., :3] = (base * (1 - m) + col * m).astype(np.uint8)
    return QImage(img.tobytes(), n, n, 4 * n, QImage.Format_ARGB32).copy()

class NetView(QtWidgets.QWidget):
    info = QtCore.pyqtSignal(str)

    def __init__(self, W, dev, D=256, L=32):
        super().__init__()
        self.W, self.dev, self.D, self.L = W, dev, D, L
        self.setMinimumSize(600, 400); self.setMouseTracking(True)
        self.s = .1; self.ox = -60.; self.oy = -40.
        self.drag = None; self.pin = None; self.hover = None
        self.accum = True; self.frame = 0; self.n = 0
        self.gates = np.full((L, D), .5, np.float32)
        self.mode = 'живість'
        self.maps = {'живість': np.zeros((L, D), np.float32)}
        self._acc_reset()
        self.show_edges = True; self.show_mats = True
        self._mat = {}
        # топ-K вхідних ребер на нейрон: (джерело, вага)
        self.esrc, self.ew = [], []
        for l in range(L):
            w = W[l]
            v, i = torch.topk(w.abs(), TOPK, dim=0)
            self.esrc.append(i.cpu().numpy())
            self.ew.append(torch.gather(w, 0, i).cpu().numpy())
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick)
        self.reset_view()

    def set_weights(self, W):
        """Підмінити мережу: перерахувати ребра, скинути кеш матриць і накопичення."""
        self.W = W
        self.esrc, self.ew = [], []
        for l in range(self.L):
            w = W[l]
            v, i = torch.topk(w.abs(), TOPK, dim=0)
            self.esrc.append(i.cpu().numpy())
            self.ew.append(torch.gather(w, 0, i).cpu().numpy())
        self._mat = {}
        self.pin = None; self.hover = None
        self._acc_reset(); self.n = 0; self.frame = 0
        self.gates = np.full((self.L, self.D), .5, np.float32)
        self.maps = {m: np.zeros((self.L, self.D), np.float32) for m in self.MODES}
        self.update()

    # ---------- геометрія ----------
    def X(self, l): return (l * CX - self.ox) * self.s
    def Y(self, j): return (j * CY - self.oy) * self.s

    def reset_view(self):
        vw, vh = max(self.width(), 400), max(self.height(), 300)
        self.s = min(vw / (self.L * CX + 700), vh / (self.D * CY + 120))
        self.ox, self.oy = -60., -40.
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.s <= .02: self.reset_view()

    # ---------- симуляція ----------
    def start(self, accum):
        self.accum = accum
        for k in self.S: self.S[k][:] = 0
        self.n = 0; self.frame = 0
        self.timer.start(90)
    def stop(self): self.timer.stop()

    def _acc_reset(self):
        z = lambda: torch.zeros(self.L, self.D, dtype=torch.float64, device=self.dev)
        self.S = dict(g=z(), a=z(), a2=z(), h=z(), h2=z(), r=z(), infl=z())

    def tick(self):
        """Один кадр симуляції. Рахуємо ВСІ карти одразу — перемикання режиму
        миттєве й не потребує перерахунку."""
        nb = 400 if self.accum else 1
        g = torch.Generator(device=self.dev).manual_seed(80000 + self.frame)
        x = torch.randn(nb, self.D, device=self.dev, generator=g)
        acts, pres = [], []
        a = x
        for l in range(self.L):
            h = a @ self.W[l]
            h.requires_grad_(True); h.retain_grad()
            pres.append(h)
            a = torch.relu(h); acts.append(a)
        # ВНЕСОК: як фліктуація нейрона рухає вихідне середнє (одна зворотна хода)
        out = acts[-1].mean(0)
        (out * out).sum().backward()
        for l in range(self.L):
            hh = pres[l].detach(); aa = acts[l].detach()
            self.S['g'][l] += (hh > 0).double().sum(0)
            self.S['a'][l] += aa.double().sum(0)
            self.S['a2'][l] += (aa.double() ** 2).sum(0)
            self.S['h'][l] += hh.double().sum(0)
            self.S['h2'][l] += (hh.double() ** 2).sum(0)
            self.S['r'][l] += torch.relu(hh).double().sum(0)
            self.S['infl'][l] += pres[l].grad.detach().abs().double().sum(0)
        self.n += nb; self.frame += 1
        if not self.accum:                      # без накопичення — тільки цей кадр
            for k in self.S: self.S[k] /= 1.0
        self._recompute()
        self.update()

    def _recompute(self):
        """Перерахувати всі карти з накопичених сум (n семплів)."""
        n = max(self.n, 1)
        S = self.S
        p = (S['g'] / n)
        mu = S['h'] / n; sd = (S['h2'] / n - mu * mu).clamp(min=1e-30).sqrt()
        ma = S['a'] / n; va = (S['a2'] / n - ma * ma).clamp(min=0)
        z = mu / sd
        Phi = .5 * (1 + torch.erf(z / 1.4142135623730951))
        phi = torch.exp(-.5 * z * z) / 2.5066282746310002
        clo = (mu * Phi + sd * phi - S['r'] / n).abs()          # похибка гаусового замикання
        infl = S['infl'] / n
        def nrm(M):                                             # нормуємо ПО ШАРУ
            M = M.float(); m = M.abs().amax(1, keepdim=True).clamp(min=1e-30)
            return (M / m).cpu().numpy()
        self.maps = {
            'живість':      (2 * torch.minimum(p, 1 - p)).float().cpu().numpy(),
            'активація':    nrm(ma),
            'дисперсія':    nrm(va.sqrt()),
            'внесок':       nrm(infl * va.sqrt()),               # чутливість × розмах
            'похибка':      nrm(clo),
        }
        self.gates = p.float().cpu().numpy()
        live = float((self.maps['живість'] > .1).mean())
        self.info.emit('кадр %d · %s · живих %.0f%% · режим: %s'
                       % (self.frame, ('накопичено %s семплів' % f'{self.n:,}')
                          if self.accum else 'ОДИН семпл на кадр', 100 * live, self.mode))

    # ---------- миша ----------
    def wheelEvent(self, e):
        k = pow(1.0015, e.angleDelta().y())
        ns = min(40., max(.02, self.s * k))
        wx = e.x() / self.s + self.ox; wy = e.y() / self.s + self.oy
        self.s = ns
        self.ox = wx - e.x() / ns; self.oy = wy - e.y() / ns
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton: self.reset_view(); return
        h = self.hit(e.x(), e.y())
        if h: self.pin = None if self.pin == h else h
        else: self.drag = (e.x(), e.y(), self.ox, self.oy)
        self.update()

    def mouseMoveEvent(self, e):
        if self.drag:
            x0, y0, ox, oy = self.drag
            self.ox = ox - (e.x() - x0) / self.s; self.oy = oy - (e.y() - y0) / self.s
        else:
            h = self.hit(e.x(), e.y())
            new = (h[0], h[1], e.x(), e.y()) if h else None
            if new != self.hover: self.hover = new
            else: return
        self.update()

    def mouseReleaseEvent(self, e): self.drag = None
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape: self.pin = None; self.update()

    def hit(self, mx, my):
        wx = mx / self.s + self.ox; wy = my / self.s + self.oy
        l = int(round(wx / CX)); j = int(round(wy / CY))
        if 0 <= l < self.L and 0 <= j < self.D:
            if abs(wx - l * CX) * self.s < 9 and abs(wy - j * CY) * self.s < 9:
                return (l, j)
        return None

    # ---------- кольори ----------
    MODES = ('живість', 'активація', 'дисперсія', 'внесок', 'похибка')
    HINT = {'живість':   'наскільки нейрон реально перемикається (2·min(p,1−p))',
            'активація': 'середня активація |a| — нормовано в межах шару',
            'дисперсія': 'розмах активації — ЗВІДКИ береться шум Монте-Карло',
            'внесок':    'чутливість виходу × розмах — які нейрони РУХАЮТЬ відповідь',
            'похибка':   'похибка гаусового замикання на нейроні — де ламається аналітика'}

    def set_mode(self, m):
        self.mode = m
        if self.n: self._recompute()
        self.update()

    def _col(self, l, j):
        M = self.maps.get(self.mode)
        b = float(M[l, j]) if M is not None else 0.
        b = max(0., min(1., b))
        if self.mode == 'живість':
            c = QColor(OPEN if self.gates[l, j] >= .5 else SHUT)
        elif self.mode == 'похибка':
            c = QColor(GOLD)
        else:
            c = QColor(OPEN if self.gates[l, j] >= .5 else SHUT)
        c.setAlphaF(.12 + .88 * b)
        return c, b

    def _matimg(self, l):
        if l not in self._mat:
            self._mat[l] = weight_qimage(self.W[l].cpu().numpy())
        return self._mat[l]

    # ---------- малювання ----------
    def paintEvent(self, _):
        p = QPainter(self); p.fillRect(self.rect(), BG)
        s = self.s
        p.setRenderHint(QPainter.Antialiasing, s > .06)
        l0 = max(0, int(self.ox / CX) - 1)
        l1 = min(self.L - 1, int((self.ox + self.width() / s) / CX) + 1)
        j0 = max(0, int(self.oy / CY) - 2)
        j1 = min(self.D - 1, int((self.oy + self.height() / s) / CY) + 2)

        if self.show_mats and s > .012: self._matrices(p, l0, l1)
        if self.show_edges and s * 3 > .35 and not self.pin: self._edges(p, l0, l1)
        self._neurons(p, l0, l1, j0, j1)
        if self.pin: self._pin(p)
        self._minimap(p); self._tip(p); self._legend(p)
        p.end()

    def _matrices(self, p, l0, l1):
        # матриці мають сенс лише коли видно КОЛОНКУ ЦІЛКОМ (оглядовий масштаб).
        # При наближенні вони або їдуть за екран, або лізуть на підписи — тоді ховаємо.
        y0, y1 = self.Y(0), self.Y(self.D - 1)
        if y0 < -4 or y1 > self.height() - 46: return
        side = min(CX * .60 * self.s, (y1 - y0) * .92)
        if side < 8: return
        ymid = (y0 + y1) / 2
        p.setFont(QFont('monospace', 9))
        for l in range(max(1, l0), l1 + 1):
            x = (self.X(l - 1) + self.X(l)) / 2
            r = QRectF(x - side / 2, ymid - side / 2, side, side)
            p.drawImage(r, self._matimg(l))
            p.setPen(QPen(LINE)); p.setBrush(Qt.NoBrush); p.drawRect(r)
            if side > 60:
                p.setPen(QPen(MUT)); p.drawText(QPointF(r.x(), r.y() - 6), 'W%d  256×256' % l)

    def _edges(self, p, l0, l1):
        for l in range(max(1, l0), l1 + 1):
            xa, xb = self.X(l - 1) + 3, self.X(l) - 3
            src, w = self.esrc[l], self.ew[l]
            wmax = np.abs(w).max() + 1e-12
            for sign in (1, -1):
                for bucket in (0, 1, 2):
                    lines = []
                    for k in range(TOPK):
                        for j in range(0, self.D, 1 if self.s * 3 > 1.2 else 3):
                            wv = w[k, j]
                            if (wv > 0) != (sign > 0): continue
                            rel = abs(wv) / wmax
                            if rel < .55: continue        # як у старій: слабкі ребра не малюємо
                            mag = min(2, int(rel * 3))
                            if mag != bucket: continue
                            lines.append(QLineF(xa, self.Y(int(src[k, j])), xb, self.Y(j)))
                    if lines:
                        c = QColor(OPEN if sign > 0 else SHUT)
                        c.setAlphaF(.06 + .10 * bucket)
                        p.setPen(QPen(c, max(.6, .5 * self.s * 3)))
                        p.drawLines(lines)

    def _neurons(self, p, l0, l1, j0, j1):
        p.setPen(Qt.NoPen); p.setFont(QFont('monospace', 9))
        fast = self.s * 3 < .9
        for l in range(l0, l1 + 1):
            x = self.X(l)
            if fast:
                for j in range(j0, j1 + 1):
                    c, b = self._col(l, j)
                    pen = QPen(c); pen.setWidthF(max(1.2, 2.6 * b)); p.setPen(pen)
                    p.drawPoint(QPointF(x, self.Y(j)))
                p.setPen(Qt.NoPen)
            else:
                for j in range(j0, j1 + 1):
                    c, b = self._col(l, j)
                    p.setBrush(QBrush(c))
                    r = max(.7, min(5.0, (1.0 + 3.0 * b) * self.s * 3))
                    p.drawEllipse(QPointF(x, self.Y(j)), r, r)
            # підписи: прибиті до низу вікна + пропуск, коли колонки тісніші за напис
            every = max(1, int(np.ceil(52.0 / max(CX * self.s, 1e-6))))
            if self.s > .04 and (l % every == 0 or l == self.L - 1):
                yl = min(self.Y(self.D) + 20, self.height() - 42)
                p.setPen(QPen(MUT))
                p.drawText(QPointF(x - 16, yl),
                           'вихід' if l == self.L - 1 else 'шар %d' % l)
                p.setPen(Qt.NoPen)

    def _pin(self, p):
        l, j = self.pin
        if l >= 1:
            xa, xb = self.X(l - 1) + 3, self.X(l) - 3
            for k in range(TOPK):
                wv = self.ew[l][k, j]
                c = QColor(OPEN if wv > 0 else SHUT); c.setAlphaF(.95)
                p.setPen(QPen(c, 2.2))
                p.drawLine(QLineF(xa, self.Y(int(self.esrc[l][k, j])), xb, self.Y(j)))
        if l + 1 < self.L:
            xa, xb = self.X(l) + 3, self.X(l + 1) - 3
            src, w = self.esrc[l + 1], self.ew[l + 1]
            for k in range(TOPK):
                for dj in np.where(src[k] == j)[0]:
                    wv = w[k, dj]
                    c = QColor(OPEN if wv > 0 else SHUT); c.setAlphaF(.95)
                    p.setPen(QPen(c, 2.2))
                    p.drawLine(QLineF(xa, self.Y(j), xb, self.Y(int(dj))))
        p.setPen(QPen(GOLD, 2)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(self.X(l), self.Y(j)), max(6., 4 * self.s * 3), max(6., 4 * self.s * 3))

    def _minimap(self, p):
        w, h = 190, 120
        r = QRectF(self.width() - w - 14, 14, w, h)
        p.setBrush(QBrush(PANEL)); p.setPen(QPen(LINE)); p.drawRect(r)
        sx = w / (self.L * CX); sy = h / (self.D * CY)
        g = self.gates
        p.setPen(Qt.NoPen)
        step = 4
        for l in range(0, self.L, 1):
            for j in range(0, self.D, step):
                b = 2 * min(float(g[l, j]), 1 - float(g[l, j]))
                if b < .12: continue
                c = QColor(OPEN if g[l, j] >= .5 else SHUT); c.setAlphaF(.25 + .75 * b)
                p.setBrush(QBrush(c))
                p.drawRect(QRectF(r.x() + l * CX * sx, r.y() + j * CY * sy, 2.0, 2.0))
        vx = r.x() + self.ox * sx; vy = r.y() + self.oy * sy
        vw = self.width() / self.s * sx; vh = self.height() / self.s * sy
        p.setBrush(Qt.NoBrush); p.setPen(QPen(GOLD, 1.2))
        p.drawRect(QRectF(vx, vy, vw, vh))

    def _tip(self, p):
        if not self.hover: return
        l, j, mx, my = self.hover
        pr = float(self.gates[l, j]); b = 2 * min(pr, 1 - pr)
        txt = ['шар %d · нейрон %d' % (l, j),
               'p(відкритий) = %.4f' % pr,
               'живість = %.3f  %s' % (b, 'ПРИБИТИЙ' if b < .1 else 'живий')]
        p.setFont(QFont('monospace', 9))
        fm = p.fontMetrics(); wd = max(fm.width(t) for t in txt) + 16
        r = QRectF(mx + 14, my + 10, wd, 16 * len(txt) + 10)
        if r.right() > self.width(): r.moveLeft(mx - wd - 14)
        p.setBrush(QBrush(PANEL)); p.setPen(QPen(LINE)); p.drawRect(r)
        p.setPen(QPen(INK))
        for i, t in enumerate(txt):
            p.drawText(QPointF(r.x() + 8, r.y() + 18 + 16 * i), t)

    def _legend(self, p):
        p.setFont(QFont('monospace', 9)); p.setPen(QPen(MUT))
        p.drawText(12, 18, 'колесо — зум до курсора · тягнути — пан · клік — пін нейрона '
                           '· Esc — зняти · права кнопка — скинути вид')
        y = self.height() - 12
        p.drawText(12, y - 18, 'РОЗМІР І ЯСКРАВІСТЬ = %s · %s'
                   % (self.mode.upper(), self.HINT.get(self.mode, '')))
        if self.mode == 'похибка':
            p.setPen(QPen(GOLD)); p.drawText(12, y, 'золоте — там гаусове замикання ПОМИЛЯЄТЬСЯ найбільше')
        else:
            p.setPen(QPen(OPEN)); p.drawText(12, y, 'блакитний — переважно ПРОПУСКАЄ / вага +')
            p.setPen(QPen(SHUT)); p.drawText(360, y, 'рожевий — переважно РІЖЕ / вага −')
