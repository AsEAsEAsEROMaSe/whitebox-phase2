#!/usr/bin/env python3
"""17.10.2026. Локальна карта ліній обрахунку (PyQt6). Те саме, що артефакт lines.html: величини як лінії через шари,
вузли обрахунку, лінія Монте-Карло, закриті гілки зі зміряними межами, відкрите питання.
Запуск: /usr/bin/python3 /mnt/win_d/llm/White-Box-P2/viz/lines_qt.py
Колесо миші: масштаб · ліва кнопка: тягнути · F: вписати · D: темна/світла тема · S: зберегти PNG."""
import sys, math, random
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QGraphicsPathItem, QGraphicsSimpleTextItem, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QPainterPath, QFont, QImage
from PyQt6.QtCore import Qt, QPointF, QRectF

LIGHT = dict(bg="#f3f5f7", panel="#ffffff", ink="#18212b", mute="#5b6875", line="#c9d2db", mu="#0f8b8d", cov="#3d5a98", k3="#d9731f",
             old="#8a5bb0", k4="#c0396b", mc="#6b7a88", truth="#18212b", dead="#9aa6b2", cut="#2e9b4f")
DARK = dict(bg="#0f151c", panel="#151d26", ink="#e4eaf0", mute="#93a1af", line="#2c3945", mu="#35b9b5", cov="#7d9be0", k3="#f0954a",
            old="#b58ad8", k4="#e86a96", mc="#8e9cab", truth="#e4eaf0", dead="#55626f", cut="#5ccb7c")

DEAD = [
    ("mu", "ознака нейрона → знак похибки", "corr 0±0.1"), ("mu", "таблиця / гіпермережа", "≤×1.02"), ("mu", "поправка з величин замикання", "стеля ×1.027"),
    ("cov", "гладкий базис 128", "= випадковий"), ("cov", "Кронекер / тензорний потяг", "522 = гаус"), ("cov", "сферизація", "×0.74"),
    ("k3", "стиснення молодих до 512", "×0.24"), ("k3", "проєктор критичних напрямків", "×0.015"), ("k3", "вік 3 замість 4", "×0.91"), ("k3", "випадкові 50% точних", "42% стелі"),
    ("k4", "табличний γ₄", "провал «чужа»"), ("k4", "дзеркальні ланцюги λ±, g±", "×1.005"), ("k4", "предиктор з джерел", "×1.25 за ту ж ціну"),
    ("mc", "антитетика / RQMC / кубатура", "≤×1.4, вибілювання бере все"), ("mc", "2k точок як ліки", "шум ×1000"), ("mc", "«погані» семпли", "не існують, corr 0.01"),
    ("ink", "ідеальна поправка кроку", "стеля ×1.19"), ("ink", "Еджворт / Джонсон / КТФ", "×0.83…1.14"), ("ink", "точні парні кумулянти", "×20, ціна 16 бюджетів"),
]
KEYS = [
    "○ W: лінійний крок, усі лінії множаться на ваги; точний; для джерел κ3 це n³ кожне, звідси товщина",
    "⊗ хаб: джерела κ3 згортаються в зрізи D21, D3 і вливаються в μ та C; 23.6% ціни",
    "◆ ReLU: нелінійний крок через гаусові похідні; єдине місце народження похибки: 15…20% на шар, решта успадкована",
    "● народження: з C після ReLU стартує нове джерело κ3 рангу 1024",
    "▽ базис: з віку 5 джерело стискається у спільний базис 384, з віку 8 у 224",
    "↻ κ4: діагональ регенерується з μ, C щошару; поза діагоналлю закон λ·C і зріз κ31; пам'яті не має",
    "◇ вихід: промах 2.2e-8 від істини; Монте-Карло з мільйона точок ще шумніше за цей промах",
]


class View(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.dark = False
        self.build()

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def keyPressEvent(self, e):
        k = e.text().lower()
        if k == "f": self.fitInView(self.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)
        elif k == "d": self.dark = not self.dark; self.build()
        elif k == "s":
            r = self.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20); img = QImage(int(r.width() * 2), int(r.height() * 2), QImage.Format.Format_ARGB32)
            img.fill(QColor(self.C["bg"])); p = QPainter(img); p.setRenderHint(QPainter.RenderHint.Antialiasing); self.scene().render(p, QRectF(img.rect()), r); p.end()
            img.save("/mnt/win_d/llm/White-Box-P2/viz/lines.png"); print("збережено viz/lines.png")
        else: super().keyPressEvent(e)

    # ---- малювання ----
    def pen(self, col, w=1.0, dash=None):
        p = QPen(QColor(self.C[col] if col in self.C else col), w); p.setCapStyle(Qt.PenCapStyle.RoundCap); p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if dash: p.setDashPattern(dash)
        return p

    def path(self, pts, col, w, dash=None):
        pp = QPainterPath(QPointF(*pts[0]))
        for x, y in pts[1:]: pp.lineTo(x, y)
        it = QGraphicsPathItem(pp); it.setPen(self.pen(col, w, dash)); self.sc.addItem(it); return it

    def text(self, x, y, s, size=9, col="ink", anchor="l"):
        t = QGraphicsSimpleTextItem(s); f = QFont("IBM Plex Mono"); f.setPointSizeF(float(size)); t.setFont(f); t.setBrush(QBrush(QColor(self.C[col] if col in self.C else col)))
        w = t.boundingRect().width(); t.setPos(x - (w / 2 if anchor == "c" else w if anchor == "r" else 0), y - t.boundingRect().height() * 0.8); self.sc.addItem(t); return t

    def dot(self, x, y, r, col, fill=True):
        e = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r); e.setPen(self.pen("ink", 1.2)); e.setBrush(QBrush(QColor(self.C[col])) if fill else QBrush(QColor(self.C["panel"]))); self.sc.addItem(e)

    def build(self):
        self.C = DARK if self.dark else LIGHT; C = self.C
        self.sc = QGraphicsScene(); self.setScene(self.sc); self.setBackgroundBrush(QBrush(QColor(C["bg"])))
        L = 16; x0, x1 = 70, 1250; dx = (x1 - x0) / L; X = lambda l: x0 + l * dx
        yMu, yC, yK4, yK3, yMC = 120, 170, 70, 230, 430
        self.text(x0, 0, "Лінії обрахунку і контрольні точки", 16, "ink")
        self.text(x0, 14, "лінія = величина, що несеться через шари · товщина = ціна у флопах · вузли = місця обрахунку · під ланцюгом Монте-Карло · внизу закриті гілки зі зміряними межами", 9, "mute")
        for l in range(L + 1):
            self.path([(X(l), 40), (X(l), 470)], "line", 1); self.text(X(l), 36, ("шар %d" % l) if l < L else "вихід", 9, "mute", "c")
        def step(y, amp):
            p = [(x0, y)]
            for l in range(L): p += [(X(l) + dx * .55, y), (X(l) + dx * .62, y - amp), (X(l) + dx * .7, y)]
            return p + [(x1, y)]
        self.path(step(yC, 8), "cov", 4); self.path(step(yMu, 6), "mu", 3)
        for l in range(1, L):
            xa = X(l) + dx * .3; self.path([(xa, yC), (xa + 8, yK4), (X(l) + dx * .62, yK4), (X(l) + dx * .62, yMu - 6)], "k4", 1.6, [5, 3]); self.text(xa + 4, yK4 - 2, "↻", 8, "mute")
        for b in range(L - 1):
            yb = yK3 + b * 13; pts = [(X(b) + dx * .7, yC + 8), (X(b) + dx * .85, yb)]; age_end = min(b + 4, L)
            pts += [(X(l), yb) for l in range(b + 1, age_end + 1)]; self.path(pts, "k3", 3.2); self.dot(X(b) + dx * .7, yC + 8, 3.5, "k3")
            if age_end < L:
                self.path([(X(l), yb) for l in range(age_end, L + 1)], "old", 1.2); self.text(X(age_end) - 4, yb + 8, "▽", 8, "old")
                if b + 8 < L: self.path([(X(b + 8), yb), (x1, yb)], "old", .7)
            for l in range(b + 1, min(b + 4, L - 1) + 1): self.path([(X(l) + dx * .25, yb), (X(l) + dx * .35, yC + 4)], "k3", .8, [2, 2])
        for l in range(L):
            self.dot(X(l) + dx * .1, yC, 4, "panel", False); self.text(X(l) + dx * .1, yC - 4, "W", 8, "mute", "c")
            if l > 0: self.text(X(l) + dx * .35, yC + 8, "⊗", 10, "k3", "c")
            self.text(X(l) + dx * .62, yC + 28, "◆", 9, "ink", "c")
        self.text(X(0) + dx * .62, yC + 42, "ReLU", 8, "mute", "c")
        self.text(x1 + 10, yMu + 8, "◇ E[out]", 10, "mu"); self.text(x1 + 10, yMu + 22, "2.2e-8 від істини", 8, "mute")
        self.dot(x1 + 90, yMu, 4, "truth"); self.text(x1 + 98, yMu + 8, "істина", 9, "truth")
        self.text(X(5) + 6, yK3 + 5 * 13 - 2, "перенос 22% + хаб 24% = половина ціни", 8, "k3")
        self.text(X(10) + 6, yK3 + 2 * 13 - 2, "базис: ноги 10% + хаб 9% + перебудова 16%", 8, "old")
        self.text(X(2), yC + 52, "похибка: 15…20% народжується тут, 80% їде далі по лініях (cos 0.8…0.9)", 8, "mute")
        self.text(X(2), yK4 - 18, "κ4: без цього ×2 гірше · повні зрізи дали б ще ×1.4 (ціна n⁴)", 8, "k4")
        rnd = random.Random(7); pm = [(x0, yMC)]
        for l in range(L):
            for q in range(1, 5): pm.append((X(l) + dx * q / 4, yMC + (rnd.random() - .5) * 46))
        pm.append((x1, yMC + (rnd.random() - .5) * 46)); self.path(pm, "mc", 1.4, [3, 3])
        r = QGraphicsRectItem(x1, yMC - 30, 14, 60); c = QColor(C["mc"]); c.setAlphaF(.25); r.setBrush(QBrush(c)); r.setPen(QPen(Qt.PenStyle.NoPen)); self.sc.addItem(r)
        self.text(x1 + 20, yMC + 8, "10⁶ точок: шум 0.1% σ", 9, "mc"); self.text(x1 + 20, yMC + 22, "промах ланцюга 0.01…0.08% σ: не видно", 8, "mute")
        self.text(x0, yMC - 36, "Монте-Карло: одна точка через усі маски ReLU, усі залежності точні, але відхилення 56%", 10, "mc")
        for l in range(L): self.text(X(l) + dx * .62, yMC + 68, "◆", 8, "mc", "c")
        for x, lab, res in [(X(6) + dx * .5, "контрольна варіата", "×1.26"), (X(10) + dx * .5, "гібрид: ланцюг→семпли", "×0.07"), (X(13) + dx * .5, "МК-кумулянти 65k", "×0.04")]:
            self.path([(x, yC + 12), (x, yMC - 50)], "dead", 1, [2, 3]); self.text(x + 4, (yC + yMC) / 2, lab + " · " + res, 8, "dead")
        yD = 560; self.path([(x0, yD - 40), (x1 + 120, yD - 40)], "line", 1); self.text(x0, yD - 44, "ЗАКРИТІ ГІЛКИ (від якої лінії відходять · зміряна межа)", 10, "mute")
        cols = 4; cw = (x1 + 120 - x0) / cols
        for i, (col, lab, res) in enumerate(DEAD):
            cx = x0 + (i % cols) * cw; cy = yD + (i // cols) * 34
            self.path([(cx, cy), (cx + 26, cy), (cx + 40, cy + 8)], col, 1.6); self.text(cx + 44, cy + 12, "⛔", 9, "dead")
            self.text(cx + 58, cy + 8, lab, 9.5, "ink"); self.text(cx + 58, cy + 21, res, 8.5, "mute")
        yo = yD + 5 * 34 + 10; self.path([(x0, yo), (X(3), yo)], "k3", 3.2); self.path([(X(3), yo), (X(5), yo - 14)], "cut", 1.6, [4, 3])
        self.text(X(5) + 6, yo - 6, "? представлення молодого κ3, що не дорожчає з віком: єдиний шлях у топ, не знайдене", 10, "cut")
        yk = yo + 40
        for i, k in enumerate(KEYS): self.text(x0, yk + i * 16, k, 8.5, "mute")
        self.text(x0, yk + len(KEYS) * 16 + 6, "колесо: масштаб · тягнути: ліва кнопка · F: вписати · D: тема · S: зберегти PNG", 8, "mute")
        self.fitInView(self.sc.itemsBoundingRect().adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv); w = QMainWindow(); v = View(); w.setCentralWidget(v); w.setWindowTitle("Лінії обрахунку: v37"); w.resize(1500, 900); w.show()
    v.fitInView(v.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)
    sys.exit(app.exec())
