#!/usr/bin/env python
"""WHITE-BOX LAB — жива візуалізація висновків сесії 47.

Рахує САМ, на GPU, і малює по ходу. Не картинки з готових чисел, а справжні прогони.
Запуск:  /mnt/win_d/llm/White-Box/torch_venv/bin/python whitebox_lab.py

ПАНЕЛІ:
 1. ДРАБИНА     — вмикаємо ReLU по одному; відповідь не будується, а ВИГРИЗАЄТЬСЯ
 2. ДВА ВОРОГИ  — де живуть семпли (біля нуля) і де живе дисперсія (у товщі)
 3. ПОВОРОТИ    — приз 3.25× існує, але проксі його не бачить
 4. РУЧКА α     — перехід лінійна→ReLU це СХОДИНКА, не схил
"""
import sys, math, time
import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

BANK = '/mnt/win_d/llm/White-Box/netbank/bm16_w.npz'
D, L = 256, 32
BG, FG, ACC, ACC2, MUT = '#12141a', '#e6e8ee', '#5ec8f2', '#f2a65e', '#7d8494'

# ----------------------------------------------------------------------------- дані/мережа
def pick_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return 'cuda:%d' % (4 if n > 4 else n - 1)
    return 'cpu'

DEV = pick_device()
_dat = None
CUSTOM = None          # якщо задано — цю мережу бачать УСІ панелі, не лише вигляд

def weights(net, dtype=torch.float32):
    global _dat
    if CUSTOM is not None:
        return [w.to(dtype) for w in CUSTOM]
    if _dat is None:
        _dat = np.load(BANK)
    return [torch.tensor(_dat['w_%d_%d' % (net, i)].astype(np.float32),
                         device=DEV).to(dtype) for i in range(L)]

def make_net(seed):
    """Свіжа випадкова мережа з ТІЄЮ САМОЮ статистикою, що й у задачі:
       W ~ N(0, 2/d), без зсувів ⟹ критичність (лінійний шар подвоює дисперсію,
       ReLU ділить навпіл). Інакше мережа затухне або вибухне за 32 шари."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    sd = (2.0 / D) ** 0.5
    return [torch.randn(D, D, device=DEV, generator=g) * sd for _ in range(L)]

def fwd(W, x, alpha=0.0):
    a = x
    for li in range(L):
        h = a @ W[li]
        a = h if alpha == 1.0 else torch.maximum(h, alpha * h)
    return a

def zca(x):
    C = (x.t() @ x) / x.shape[0]
    e, V = torch.linalg.eigh(C.double().cpu())
    return (V @ torch.diag(e.clamp(min=1e-14).rsqrt()) @ V.t()).to(DEV).to(x.dtype)

def rand_orth(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = torch.randn(D, D, device=DEV, generator=g)
    Q, R = torch.linalg.qr(A.double())
    return (Q * torch.sign(R.diag())[None, :]).float()

# ----------------------------------------------------------------------------- воркер
class Worker(QtCore.QThread):
    say = QtCore.pyqtSignal(str)
    plot = QtCore.pyqtSignal(dict)
    fin = QtCore.pyqtSignal()

    def __init__(self, kind, scale, accum=True):
        super().__init__()
        self.kind, self.scale, self.accum, self.stop_flag = kind, scale, accum, False

    def run(self):
        try:
            getattr(self, 'exp_' + self.kind)()
        except Exception as e:
            self.say.emit('✖ ПОМИЛКА: %s' % e)
        self.fin.emit()

    # --- 1. драбина -------------------------------------------------------
    def exp_ladder(self):
        s = self.scale
        NP, CH = 40_000 * s, 20_000
        self.say.emit('Драбина: %s пар, fp64, хвости Q_k рахуються назад…' % f'{NP:,}')
        W = weights(0, torch.float64)
        Q = [None] * (L + 1)
        Q[L] = torch.eye(D, dtype=torch.float64, device=DEV)
        for k in range(L - 1, -1, -1):
            Q[k] = W[k] @ Q[k + 1]
        self.say.emit('  ‖Q_0‖ = %.3e   ‖Q_32‖ = %.3e   (відношення %.0f = 2^%.1f)'
                      % (Q[0].norm(), Q[L].norm(), Q[0].norm() / Q[L].norm(),
                         math.log2(float(Q[0].norm() / Q[L].norm()))))
        S = torch.zeros(L + 1, D, dtype=torch.float64, device=DEV); n = 0
        for c in range(max(1, NP // CH)):
            if self.stop_flag: return
            g = torch.Generator(device=DEV).manual_seed(5500 + c)
            z = torch.randn(CH, D, device=DEV, generator=g, dtype=torch.float64)
            x = torch.cat([z, -z], 0)
            a = x
            for k in range(L + 1):
                Mk = a @ Q[k]
                S[k] += ((Mk[:CH] + Mk[CH:]) * 0.5).sum(0)
                if k < L: a = torch.relu(a @ W[k])
            n += CH
            E = (S / n)
            ref = float((E[L] * E[L]).sum())
            y = [float((E[k] * E[k]).sum()) / ref for k in range(L + 1)]
            self.plot.emit(dict(kind='ladder', y=y, prog=(c + 1) / max(1, NP // CH)))
            self.say.emit('  накопичено %s пар' % f'{n:,}')
        self.say.emit('✔ пік на щаблі %d; спуск ×½ за шар (2^30 ≈ 1e9)' % int(np.argmax(y)))

    # --- 2. два вороги ----------------------------------------------------
    def exp_enemies(self):
        s = self.scale
        NP = 20_000 * s
        self.say.emit('Два вороги: розподіл передактивацій останнього шару, %s семплів…' % f'{NP:,}')
        W = weights(0)
        g = torch.Generator(device=DEV).manual_seed(4242)
        x = torch.randn(NP, D, device=DEV, generator=g)
        a = x
        for li in range(L - 1):
            a = torch.relu(a @ W[li])
        h = a @ W[L - 1]
        sd = h.std(0, keepdim=True)
        t = (h / sd).flatten()                      # передактивація в сигмах
        r = torch.relu(t)
        contrib = (r - r.mean()) ** 2               # внесок семпла у дисперсію
        NB = 80
        edges = torch.linspace(-4, 4, NB + 1, device=DEV)
        idx = (torch.bucketize(t, edges) - 1).clamp(0, NB - 1)
        one = torch.ones_like(t)
        dens = torch.zeros(NB, device=DEV).index_add_(0, idx, one).cpu().numpy()
        var = torch.zeros(NB, device=DEV).index_add_(0, idx, contrib).cpu().numpy()
        ctr = ((edges[:-1] + edges[1:]) / 2).cpu().numpy()
        # ЧЕСНЕ ПОРІВНЯННЯ: смуга |h|<0.5σ — це де ламається ReLU (аналітична біда).
        # Скільки там семплів і скільки звідти дисперсії?
        near = np.abs(ctr) < 0.5
        f_smp = dens[near].sum() / dens.sum()
        f_var = var[near].sum() / var.sum()
        self.plot.emit(dict(kind='enemies', x=ctr, dens=dens / dens.max(),
                            var=var / var.max(), f_smp=f_smp, f_var=f_var))
        self.say.emit('  смуга |h| < 0.5σ — саме там злам ReLU, там уся аналітична біда')
        self.say.emit('    у ній сидить %.1f%% УСІХ СЕМПЛІВ…' % (100 * f_smp))
        self.say.emit('    …але лише %.1f%% УСІЄЇ ДИСПЕРСІЇ' % (100 * f_var))
        self.say.emit('✔ семпли купчаться там, де аналітиці боляче, а дисперсія — деінде (×%.1f різниця)'
                      % (f_smp / max(f_var, 1e-9)))

    # --- 8. САМА МОДЕЛЬ ---------------------------------------------------
    def exp_model(self):
        """32 шари × 256 нейронів. БЕЗ накопичення — затвори ОДНОГО семпла (видно спалахи).
           З накопиченням — біжуче p_i (проявляється прибите тло)."""
        acc = self.accum
        FR, CH = 60, 2000 * self.scale
        self.say.emit('Модель: 32×256 = %d затворів. %s'
                      % (32 * D, 'НАКОПИЧЕННЯ: біжуча ймовірність відкриття'
                         if acc else 'БЕЗ НАКОПИЧЕННЯ: окремі семпли, кадр за кадром'))
        W = weights(0)
        acc_g = torch.zeros(L, D, device=DEV); nn = 0
        for f in range(FR):
            if self.stop_flag: return
            g = torch.Generator(device=DEV).manual_seed(70000 + f)
            a = torch.randn(CH if acc else 1, D, device=DEV, generator=g)
            G = torch.zeros(L, D, device=DEV)
            for li in range(L):
                h = a @ W[li]
                G[li] = (h > 0).float().mean(0)
                a = torch.relu(h)
            acc_g += G * a.shape[0]; nn += a.shape[0]
            img = (acc_g / nn) if acc else G
            im = img.cpu().numpy()
            live = float(((im > .05) & (im < .95)).mean()) if acc else float('nan')
            self.plot.emit(dict(kind='model', img=im, acc=acc, f=f + 1, FR=FR,
                                seen=nn, live=live))
            if f % 10 == 0:
                self.say.emit('  кадр %2d/%d   семплів усього %s' % (f + 1, FR, f'{nn:,}'))
            time.sleep(.06)
        if acc:
            self.say.emit('✔ проявилось ТЛО: більшість клітин майже чорні або майже білі —')
            self.say.emit('  це ПРИБИТІ нейрони, вони не перемикаються. Живих лише %.0f%%' % (100 * live))
            self.say.emit('✔ саме це видно на око як «тьмяна друга половина» — і це не артефакт')
        else:
            self.say.emit('✔ кожен кадр — один семпл. Ліва частина ЖИВЕ (клітини блимають),')
            self.say.emit('  права майже застигла: там перемикаються одиниці')

    # --- 6. прохід семплів -----------------------------------------------
    def exp_flow(self):
        acc = self.accum
        NS, NTR = 4000 * self.scale, 40
        self.say.emit('Прохід семплів крізь 32 шари, режим: %s'
                      % ('НАКОПИЧЕННЯ (біжуче середнє)' if acc else 'БЕЗ НАКОПИЧЕННЯ (кожен сам)'))
        W = weights(0)
        g = torch.Generator(device=DEV).manual_seed(777)
        z = torch.randn(NS, D, device=DEV, generator=g)
        x = torch.cat([z, -z], 0)                      # антитетичні пари
        a = x
        tr = np.zeros((NTR, L + 1))                    # окремі семпли
        run = np.zeros((L + 1, D))                     # біжуче середнє
        j = 7                                          # координата, за якою стежимо
        tr[:, 0] = a[:NTR, j].cpu().numpy()
        run[0] = ((a[:NS] + a[NS:]) * 0.5).mean(0).cpu().numpy()
        for li in range(L):
            a = torch.relu(a @ W[li])
            tr[:, li + 1] = a[:NTR, j].cpu().numpy()
            run[li + 1] = ((a[:NS] + a[NS:]) * 0.5).mean(0).cpu().numpy()
        # нормуємо кожен шар на власний масштаб, щоб бачити ФОРМУ, а не зростання
        sc = np.abs(run).mean(1) + 1e-30
        self.plot.emit(dict(kind='flow', tr=tr / sc[None, :], run=run[:, j] / sc,
                            acc=acc, spread=(np.abs(tr / sc[None, :]).std(0)).tolist()))
        s0, s1 = np.abs(tr[:, 1] / sc[1]).std(), np.abs(tr[:, L] / sc[L]).std()
        self.say.emit('  розкид окремих семплів: шар 1 = %.3f → шар 32 = %.3f' % (s0, s1))
        if acc:
            self.say.emit('✔ середнє йде гладко, хоча кожен окремий семпл скаче')
        else:
            self.say.emit('✔ окремі семпли не сходяться ні до чого — сходиться лише СЕРЕДНЄ')

    # --- 7. простий MC ----------------------------------------------------
    def exp_mc(self):
        acc = self.accum
        NP = 60_000 * self.scale
        self.say.emit('MC: %s. Істина 4М семплів…'
                      % ('НАКОПИЧЕННЯ (біжуча оцінка)' if acc else 'БЕЗ НАКОПИЧЕННЯ (окремі семпли)'))
        W = weights(0)
        S = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
        for c in range(20):
            g = torch.Generator(device=DEV).manual_seed(6100 + c)
            zz = torch.randn(100_000, D, device=DEV, generator=g)
            xx = torch.cat([zz, -zz], 0)
            S += fwd(W, xx).double().sum(0); n += xx.shape[0]
        TH = S / n
        self.say.emit('  істина готова. Женемо %s пар трьома способами…' % f'{NP:,}')
        # ОДНА реалізація сама скаче — усереднюємо квадрат помилки по REP незалежних прогонах,
        # інакше v2 не видно як переможця (криві перекриваються шумом)
        # ЧЕСНІСТЬ: вісь = ПРОГОНИ МЕРЕЖІ, не «пари». Звичайний MC витрачає 1 прогін на семпл,
        # антитетика й v2 — 2 прогони на пару. Порівнювати при рівних парах = дати їм 2× бюджет.
        # ОДНА реалізація скаче ⟹ усереднюємо квадрат помилки по REP незалежних прогонах.
        REP = 12
        kp = np.unique(np.logspace(1, math.log10(NP), 34).astype(int))   # пар
        ks = 2 * kp                                                       # прогонів мережі
        accum = {nm: np.zeros(len(kp)) for nm in ('MC', 'антитетика', 'v2')}
        for r in range(REP):
            if self.stop_flag: return
            g = torch.Generator(device=DEV).manual_seed(20250719 + 991 * r)
            z = torch.randn(NP, D, device=DEV, generator=g)
            z2 = torch.randn(NP, D, device=DEV, generator=g)
            yz = fwd(W, z).double()
            plain = torch.cat([yz, fwd(W, z2).double()], 0)               # 2·NP НЕЗАЛЕЖНИХ семплів
            anti = (yz + fwd(W, -z).double()) * 0.5                        # NP пар = 2·NP прогонів
            Xw = torch.cat([z, -z], 0); Xw = Xw @ zca(Xw)
            yv = fwd(W, Xw).double(); v2 = (yv[:NP] + yv[NP:]) * 0.5       # так само 2·NP
            for nm, Y, kk in (('MC', plain, ks), ('антитетика', anti, kp), ('v2', v2, kp)):
                if acc:
                    cs = torch.cumsum(Y, 0)
                    e = np.array([float(((cs[k - 1] / k) - TH).pow(2).sum()) for k in kk])
                else:
                    e = np.array([float((Y[k - 1] - TH).pow(2).sum()) for k in kk])
                accum[nm] += e
            self.say.emit('  прогін %d/%d' % (r + 1, REP))
            out = {'x': ks.tolist(), 'acc': acc}
            for nm in accum: out[nm] = list(np.sqrt(accum[nm] / (r + 1)))
            self.plot.emit(dict(kind='mc', **out))
        if acc:
            self.say.emit('  на %s ПРОГОНАХ МЕРЕЖІ (RMS по %d повторах):  MC %.2e   анти %.2e   v2 %.2e'
                          % (f'{ks[-1]:,}', REP, out['MC'][-1], out['антитетика'][-1], out['v2'][-1]))
            self.say.emit('✔ v2 краще за MC у %.2f× за нормою (%.2f× за MSE)'
                          % (out['MC'][-1] / out['v2'][-1], (out['MC'][-1] / out['v2'][-1]) ** 2))
            self.say.emit('✔ усі три падають як 1/√N — паралельні прямі на лог-графіку')
            self.say.emit('✔ v2 нижчий не через кращий НАХИЛ, а через нижчий СТАРТ (тихіші семпли)')
        else:
            self.say.emit('✔ без накопичення помилка НЕ падає — семпл скільки завгодно далеко')
            self.say.emit('✔ уся сила методу в УСЕРЕДНЕННІ, не в окремому семплі')

    # --- 5. прибиті затвори ----------------------------------------------
    def exp_pinned(self):
        s = self.scale
        N, CH = 50_000 * s, 25_000
        self.say.emit('Затвори: p_i = P(h_i>0) для кожного з 256 нейронів, пошарово…')
        W = weights(0)
        ssum = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(L)]
        n = 0
        for c in range(max(1, N // CH)):
            if self.stop_flag: return
            g = torch.Generator(device=DEV).manual_seed(1234 + c)
            a = torch.randn(CH, D, device=DEV, generator=g)
            for li in range(L):
                h = a @ W[li]
                ssum[li] += (h > 0).double().sum(0)
                a = torch.relu(h)
            n += CH
        hists, pinned, coins = [], [], []
        for li in range(L):
            p = (ssum[li] / n).cpu().numpy()
            hists.append(np.histogram(p, bins=20, range=(0, 1))[0].tolist())
            pinned.append(float(((p < .02) | (p > .98)).mean()))
            coins.append(float(((p > .4) & (p < .6)).mean()))
        for li in range(L):
            if self.stop_flag: return
            self.plot.emit(dict(kind='pinned', hist=hists[li], layer=li,
                                pinned=pinned, coins=coins))
            self.say.emit('  шар %2d:  прибитих %.3f   монеток %.3f' % (li, pinned[li], coins[li]))
            time.sleep(.10)
        self.say.emit('✔ шар 0: усі 256 нейронів рівно на ½ (теорема)')
        self.say.emit('✔ шар 31: прибито %.0f%%, монеток лише %.0f%%' % (100 * pinned[-1], 100 * coins[-1]))
        self.say.emit('✔ «затвори ½» — правда про СЕРЕДНЄ, неправда про МЕХАНІЗМ')

    # --- 3. повороти ------------------------------------------------------
    def exp_rot(self):
        s = self.scale
        M, NP = 15 * s, 25_000 * s
        self.say.emit('Повороти: істина 2М семплів…')
        W = weights(0)
        S = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
        for c in range(10):
            g = torch.Generator(device=DEV).manual_seed(6100 + c)
            z = torch.randn(100_000, D, device=DEV, generator=g)
            xx = torch.cat([z, -z], 0)
            S += fwd(W, xx).double().sum(0); n += xx.shape[0]
        TH = S / n
        self.say.emit('  істина готова, ‖E‖ = %.4f. Вибірка %s пар, %d поворотів…'
                      % (TH.norm(), f'{NP:,}', M))
        g = torch.Generator(device=DEV).manual_seed(31337)
        z = torch.randn(NP, D, device=DEV, generator=g)
        X = torch.cat([z, -z], 0)
        X = X @ zca(X)
        err, prox = [], []
        for m in range(M):
            if self.stop_flag: return
            Y = fwd(W, X @ rand_orth(20000 + m))
            p = ((Y[:NP] + Y[NP:]) * 0.5).double()
            err.append(float((p.mean(0) - TH).norm()))
            hh = NP // 2
            prox.append(float((p[:hh].mean(0) - p[hh:].mean(0)).norm()))
            self.plot.emit(dict(kind='rot', err=list(err), prox=list(prox)))
            self.say.emit('  поворот %2d/%d:  помилка %.3e   проксі %.3e'
                          % (m + 1, M, err[-1], prox[-1]))
        e = np.array(err)
        cc = np.corrcoef(np.array(prox), e)[0, 1] if len(e) > 2 else 0.0
        self.say.emit('✔ приз: макс/мін = %.2f×  ⟹  оракул %.2f× по MSE' %
                      (e.max() / e.min(), (e.mean() / e.min()) ** 2))
        self.say.emit('✔ проксі: корел = %+.3f  (±%.3f) ⟹ %s'
                      % (cc, 1 / max(1, len(e) - 3) ** 0.5,
                         'СЛІПИЙ' if abs(cc) < 2 / max(1, len(e) - 3) ** 0.5 else 'бачить?!'))

    # --- 4. ручка альфа ---------------------------------------------------
    def exp_alpha(self):
        s = self.scale
        NT, NP, REP = 400_000 * s, 20_000 * s, 3
        self.say.emit('Ручка α: критичність тримаємо (ваги × sqrt(2/(1+α²))/√2)…')
        W0 = weights(0)
        alphas = [1.0, 0.999, 0.99, 0.9, 0.6, 0.3, 0.1, 0.0]
        eps, rel = [], []
        for al in alphas:
            if self.stop_flag: return
            gg = (2.0 / (1.0 + al * al)) ** 0.5 / (2.0 ** 0.5)
            W = [w * gg for w in W0]
            S = torch.zeros(D, dtype=torch.float64, device=DEV); n = 0
            for c in range(max(1, NT // 100_000)):
                g = torch.Generator(device=DEV).manual_seed(770 + c + int(al * 1000))
                z = torch.randn(50_000, D, device=DEV, generator=g)
                xx = torch.cat([z, -z], 0)
                S += fwd(W, xx, al).double().sum(0); n += xx.shape[0]
            TH = S / n
            sig = float((TH * TH).mean())
            es = []
            for r in range(REP):
                g = torch.Generator(device=DEV).manual_seed(3300 + r)
                z = torch.randn(NP, D, device=DEV, generator=g)
                xx = torch.cat([z, -z], 0)
                es.append(float(((fwd(W, xx @ zca(xx), al).double().mean(0) - TH) ** 2).mean()))
            e = float(np.mean(es))
            if sig > 1e-300 and (1 - al) > 0:
                eps.append((1 - al) / 2); rel.append(e / sig)
                self.plot.emit(dict(kind='alpha', x=list(eps), y=list(rel)))
                self.say.emit('  α=%.3f  нелінійність=%.4f  відн.MSE=%.3e' % (al, eps[-1], rel[-1]))
            else:
                self.say.emit('  α=1.000  нелінійність=0       MSE = МАШИННИЙ НУЛЬ (%.1e)' % e)
        self.say.emit('✔ плато ⟹ малий параметр скорочується між сигналом і шумом')

# ----------------------------------------------------------------------------- полотно
class Canvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6), facecolor=BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.reset('Обери експеримент ліворуч і натисни ЗАПУСТИТИ')

    def style(self, title='', xl='', yl=''):
        a = self.ax
        a.set_facecolor(BG)
        for sp in a.spines.values(): sp.set_color('#2a2f3a')
        a.tick_params(colors=MUT, labelsize=9)
        a.grid(alpha=.15, color=MUT, linewidth=.6)
        a.set_title(title, color=FG, fontsize=12, pad=14)
        a.set_xlabel(xl, color=MUT, fontsize=10)
        a.set_ylabel(yl, color=MUT, fontsize=10)

    def reset(self, msg):
        self.fig.clf(); self.ax = self.fig.add_subplot(111)
        self.style()
        self.ax.text(.5, .5, msg, ha='center', va='center', color=MUT,
                     transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.draw()

    def draw_data(self, d):
        self.fig.clf(); self.ax = self.fig.add_subplot(111); a = self.ax
        k = d['kind']
        if k == 'ladder':
            y = np.array(d['y']); x = np.arange(len(y))
            a.semilogy(x[1:], y[1:], '-o', color=ACC, ms=5, lw=2)
            a.axhline(1.0, color=ACC2, ls='--', lw=1.4)
            a.text(len(y) - 1, 1.4, ' справжня відповідь', color=ACC2, fontsize=9, ha='right')
            pk = int(np.argmax(y))
            a.plot([pk], [y[pk]], 'o', color='#ff5d73', ms=11, mfc='none', mew=2)
            a.annotate('пік: щабель %d\n×%.0e більше за відповідь' % (pk, y[pk]),
                       (pk, y[pk]), (pk + 3, y[pk] * .5), color='#ff5d73', fontsize=9,
                       arrowprops=dict(color='#ff5d73', arrowstyle='->', lw=1.2))
            a.text(.98, .06, 'кожен увімкнений ReLU ділить відповідь ≈ навпіл',
                   transform=a.transAxes, ha='right', color=FG, fontsize=10)
            self.style('ДРАБИНА: відповідь не будується, а ВИГРИЗАЄТЬСЯ',
                       'скільки шарів має ReLU (решта — лінійні)',
                       '‖E[модель]‖² у частках від відповіді')
        elif k == 'enemies':
            x = d['x']
            a.fill_between(x, 0, d['dens'], color=ACC, alpha=.35, label='де СЕМПЛИ (густина)')
            a.plot(x, d['var'], color=ACC2, lw=2.4, label='звідки ДИСПЕРСІЯ')
            a.axvspan(-.5, .5, color='#ff5d73', alpha=.12)
            a.axvline(0, color='#ff5d73', lw=1.4, ls='--')
            a.text(.02, .93, 'злам ReLU\n(тут аналітична складність)', transform=a.transAxes,
                   color='#ff5d73', fontsize=9)
            a.text(.98, .60, 'у смузі |h|<0.5σ:\n%.0f%% усіх семплів\nале лише %.0f%% дисперсії'
                   % (100 * d['f_smp'], 100 * d['f_var']), transform=a.transAxes,
                   ha='right', color='#ff5d73', fontsize=10)
            a.legend(facecolor=BG, edgecolor='#2a2f3a', labelcolor=FG, fontsize=9, loc='upper right')
            self.style('ДВА ВОРОГИ: аналітика б\'ється біля нуля, дисперсія — у товщі',
                       'передактивація останнього шару, у сигмах', 'нормовано на максимум')
        elif k == 'rot':
            e = np.array(d['err']); p = np.array(d['prox'])
            a.scatter(p, e, s=70, color=ACC, alpha=.85, edgecolor='none')
            if len(e) > 2:
                a.axhline(e.min(), color='#4ad991', ls='--', lw=1.3)
                a.axhline(e.max(), color='#ff5d73', ls='--', lw=1.3)
                a.text(p.max(), e.min(), ' найкращий поворот ', color='#4ad991',
                       fontsize=9, ha='right', va='bottom')
                a.text(p.max(), e.max(), ' найгірший ', color='#ff5d73',
                       fontsize=9, ha='right', va='top')
                cc = np.corrcoef(p, e)[0, 1]
                a.text(.02, .04, 'вертикальний розкид = ПРИЗ (%.1f×)\nнахил ≈ 0 ⟹ проксі СЛІПИЙ (r=%+.2f)'
                       % (e.max() / e.min(), cc), transform=a.transAxes, color=FG, fontsize=10)
            self.style('ПОВОРОТИ: приз існує, але його не видно',
                       'проксі (розбіжність половин) — те, що МОЖНА порахувати',
                       'справжня помилка — те, чого МЕТОД НЕ ЗНАЄ')
        elif k == 'model':
            self.fig.clf(); a = self.fig.add_subplot(111); self.ax = a
            im = np.array(d['img'])
            a.imshow(im, aspect='auto', cmap='magma', vmin=0, vmax=1,
                     interpolation='nearest', origin='lower')
            a.set_facecolor(BG)
            for sp in a.spines.values(): sp.set_color('#2a2f3a')
            a.tick_params(colors=MUT, labelsize=9)
            a.set_xlabel('нейрон (256 у шарі)', color=MUT, fontsize=10)
            a.set_ylabel('шар (0 знизу → 31 угорі)', color=MUT, fontsize=10)
            if d['acc']:
                a.set_title('МОДЕЛЬ — ймовірність відкриття, накопичено по %s семплах  (живих %.0f%%)'
                            % (f"{d['seen']:,}", 100 * d['live']), color=FG, fontsize=12, pad=12)
                a.text(.5, -.14, 'чорне = завжди ріже,  біле = завжди пропускає,  кольорове = ЖИВЕ',
                       transform=a.transAxes, ha='center', color=FG, fontsize=10)
            else:
                a.set_title('МОДЕЛЬ — затвори ОДНОГО семпла, кадр %d з %d'
                            % (d['f'], d['FR']), color=FG, fontsize=12, pad=12)
                a.text(.5, -.14, 'дивись, ЯК МАЛО клітин змінюється між кадрами вгорі малюнка',
                       transform=a.transAxes, ha='center', color=ACC2, fontsize=10)
            a.axhline(15.5, color=ACC, lw=1, ls='--', alpha=.5)
            self.fig.tight_layout(); self.draw(); return
        elif k == 'flow':
            tr, run, acc = np.array(d['tr']), np.array(d['run']), d['acc']
            xs = np.arange(tr.shape[1])
            for i in range(tr.shape[0]):
                a.plot(xs, tr[i], color=ACC, alpha=.16, lw=.9)
            a.plot([], [], color=ACC, alpha=.5, lw=1.2, label='окремі семпли (40 шт.)')
            if acc:
                a.plot(xs, run, color=ACC2, lw=3, label='СЕРЕДНЄ по %s парах' % '4k+')
                a.text(.5, .95, 'кожен семпл скаче — а середнє йде гладко',
                       transform=a.transAxes, ha='center', va='top', color=ACC2, fontsize=11)
            else:
                a.text(.5, .95, 'без накопичення семпли не сходяться ні до чого',
                       transform=a.transAxes, ha='center', va='top', color='#ff5d73', fontsize=11)
            a.legend(facecolor=BG, edgecolor='#2a2f3a', labelcolor=FG, fontsize=9, loc='lower left')
            self.style('ПРОХІД СЕМПЛІВ крізь 32 шари (координата №7, шар нормовано на свій масштаб)',
                       'шар', 'значення активації')
        elif k == 'mc':
            x = np.array(d['x'])
            for nm, col in (('MC', '#7d8494'), ('антитетика', ACC2), ('v2', ACC)):
                a.loglog(x, d[nm], color=col, lw=2.4, label=nm)
            if d['acc']:
                y0 = d['MC'][0]
                a.loglog(x, y0 * (x / x[0]) ** -0.5, color='#4ad991', ls='--', lw=1.4,
                         label='теорія 1/√N')
                a.text(.5, .12, 'усі три падають ОДНАКОВО (1/√N).\nv2 нижчий не нахилом, а СТАРТОМ.',
                       transform=a.transAxes, ha='center', color=FG, fontsize=10)
            else:
                a.text(.5, .12, 'без накопичення помилка НЕ падає:\nодин семпл скільки завгодно далеко',
                       transform=a.transAxes, ha='center', color='#ff5d73', fontsize=11)
            a.legend(facecolor=BG, edgecolor='#2a2f3a', labelcolor=FG, fontsize=9)
            self.style('MC: %s' % ('НАКОПИЧЕННЯ — уся сила в усередненні' if d['acc']
                                   else 'БЕЗ НАКОПИЧЕННЯ — окремі семпли'),
                       'ПРОГОНІВ МЕРЕЖІ (однаковий бюджет для всіх трьох)  [лог]',
                       '‖оцінка − істина‖, RMS по 12 прогонах  [лог]')
        elif k == 'pinned':
            self.fig.clf()
            a = self.fig.add_subplot(211); self.ax = a
            hh = np.array(d['hist']); xs = np.linspace(.025, .975, len(hh))
            cols = ['#ff5d73' if (x < .1 or x > .9) else ACC for x in xs]
            a.bar(xs, hh, width=.045, color=cols)
            a.set_xlim(0, 1); a.set_ylim(0, max(140, hh.max() * 1.15))
            a.set_facecolor(BG)
            for sp in a.spines.values(): sp.set_color('#2a2f3a')
            a.tick_params(colors=MUT, labelsize=9); a.grid(alpha=.15, color=MUT, lw=.6)
            a.set_title('ШАР %d з 31 — скільки нейронів мають яку ймовірність бути ВІДКРИТИМИ'
                        % d['layer'], color=FG, fontsize=12, pad=12)
            a.set_xlabel('p = P(нейрон відкритий).   0 = завжди ріже,  1 = завжди пропускає',
                         color=MUT, fontsize=10)
            a.set_ylabel('скільки нейронів', color=MUT, fontsize=10)
            if d['layer'] == 0:
                a.text(.5, .72, 'усі 256 рівно на ½ — це ТЕОРЕМА', transform=a.transAxes,
                       ha='center', color='#4ad991', fontsize=12)
            elif d['layer'] > 20:
                a.text(.5, .72, 'два ГОРБИ по краях: нейрони ЗАСТИГЛИ', transform=a.transAxes,
                       ha='center', color='#ff5d73', fontsize=12)
            b = self.fig.add_subplot(212)
            ly = np.arange(len(d['pinned']))
            b.plot(ly, d['pinned'], color='#ff5d73', lw=2.2, label='прибиті (майже не перемикаються)')
            b.plot(ly, d['coins'], color=ACC, lw=2.2, label='монетки (0.4 < p < 0.6)')
            b.axvline(d['layer'], color=FG, lw=1.2, ls='--', alpha=.7)
            b.set_facecolor(BG); b.set_ylim(0, 1.02); b.set_xlim(0, L - 1)
            for sp in b.spines.values(): sp.set_color('#2a2f3a')
            b.tick_params(colors=MUT, labelsize=9); b.grid(alpha=.15, color=MUT, lw=.6)
            b.set_xlabel('глибина шару', color=MUT, fontsize=10)
            b.set_ylabel('частка нейронів', color=MUT, fontsize=10)
            b.legend(facecolor=BG, edgecolor='#2a2f3a', labelcolor=FG, fontsize=9)
            self.fig.tight_layout(); self.draw(); return
        elif k == 'alpha':
            x = np.array(d['x']); y = np.array(d['y'])
            a.semilogx(x, y, '-o', color=ACC, ms=6, lw=2)
            a.set_ylim(0, max(y.max() * 1.25, 1e-12))
            a.text(.03, .12, 'при α=1 (чиста лінійна) MSE = МАШИННИЙ НУЛЬ:\nзадача розв\'язується двома семплами',
                   transform=a.transAxes, color='#4ad991', fontsize=9)
            a.text(.5, .88, 'ПЛАТО: пів відсотка нелінійності\nвже дає повну складність задачі',
                   transform=a.transAxes, color=ACC2, fontsize=10, ha='center')
            self.style('РУЧКА α: перехід лінійна→ReLU це СХОДИНКА, не схил',
                       'сила нелінійності (1−α)/2  [лог]', 'відносна MSE')
        self.fig.tight_layout(); self.draw()

# ----------------------------------------------------------------------------- вікно
EXPS = [('net',     '0 · МЕРЕЖА',     'Сама мережа як мережа: 33 колонки × 256 нейронів,\nребра, масштаб, панорама, клік по нейрону.\nЖива: кадр за кадром або з накопиченням.'),
        ('model',   '0b · КАРТА',     'Та сама мережа тепловою картою — коли треба\nбачити всі 8192 затвори одразу, без ребер.'),
        ('ladder',  '1 · ДРАБИНА',    'Вмикаємо ReLU по одному. Відповідь не будується —\nвона ВИГРИЗАЄТЬСЯ: старт 0, стрибок ×1e9, далі\nкожен ReLU ділить навпіл.'),
        ('enemies', '2 · ДВА ВОРОГИ', 'Семпли скупчені біля нуля, де ламається ReLU.\nА дисперсія приходить із товщі. Аналітика й MC\nвоюють у різних місцях простору.'),
        ('rot',     '3 · ПОВОРОТИ',   'Ті самі семпли, різні повороти вибірки. Помилка\nміняється в рази — це ПРИЗ. Але обчислюваний\nпроксі з ним не корелює: оптимізатор сліпий.'),
        ('alpha',   '4 · РУЧКА α',    'Плавно вмикаємо нелінійність. Від машинного нуля\nдо повної складності — СХОДИНКА: малий параметр\nскорочується між сигналом і шумом.'),
        ('pinned',  '5 · ЗАТВОРИ',    'Найважливіше за день. «Затвори відкриті ½» —\nправда про середнє, неправда про механізм:\nз глибиною 3/4 нейронів ЗАСТИГАЮТЬ.'),
        ('flow',    '6 · ПРОХІД',     'Як окремі семпли йдуть крізь 32 шари.\nУвімкни НАКОПИЧЕННЯ — і побачиш, як з дикого\nрозкиду народжується гладке середнє.'),
        ('mc',      '7 · MC',         'Звичайний MC / антитетика / v2 — як помилка\nпадає з числом семплів. Без накопичення вона\nне падає взагалі.')]

class Lab(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WHITE-BOX LAB — сесія 47')
        self.resize(1500, 900)
        self.worker = None
        w = QtWidgets.QWidget(); self.setCentralWidget(w)
        lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(14)

        left = QtWidgets.QVBoxLayout(); left.setSpacing(10)
        ttl = QtWidgets.QLabel('ЕКСПЕРИМЕНТИ')
        ttl.setStyleSheet('color:%s;font-size:13px;font-weight:600;letter-spacing:2px' % MUT)
        left.addWidget(ttl)
        # список у прокрутці — інакше при 9 панелях керування виїжджає за край вікна
        inner = QtWidgets.QVBoxLayout(); inner.setSpacing(8); inner.setContentsMargins(0, 0, 8, 0)
        self.btns = []
        for kind, name, desc in EXPS:
            b = QtWidgets.QToolButton(); b.setText('%s\n%s' % (name, desc))
            b.setCheckable(True); b.setMinimumHeight(88); b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setStyleSheet("""QToolButton{background:#1a1d26;color:%s;border:1px solid #262b36;
                border-radius:10px;padding:10px 12px;text-align:left;font-size:11px}
                QToolButton:hover{border-color:%s}
                QToolButton:checked{border-color:%s;background:#1e2430;color:#fff}""" % (FG, MUT, ACC))
            b.clicked.connect(lambda _, k=kind: self.select(k))
            inner.addWidget(b); self.btns.append((kind, b))
        inner.addStretch(1)
        holder = QtWidgets.QWidget(); holder.setLayout(inner)
        scroll = QtWidgets.QScrollArea(); scroll.setWidget(holder); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea{background:transparent}'
                             'QScrollBar:vertical{background:#15181f;width:9px;border-radius:4px}'
                             'QScrollBar::handle:vertical{background:#2f3644;border-radius:4px;min-height:30px}'
                             'QScrollBar::add-line,QScrollBar::sub-line{height:0}')
        left.addWidget(scroll, 1)
        self.btns[0][1].setChecked(True); self.kind = EXPS[0][0]

        left.addSpacing(6)
        nl = QtWidgets.QLabel('мережа'); nl.setStyleSheet('color:%s;font-size:11px' % MUT)
        left.addWidget(nl)
        nrow = QtWidgets.QHBoxLayout(); nrow.setSpacing(6)
        self.net_box = QtWidgets.QComboBox()
        self.net_box.addItems(['банк 0', 'банк 1', 'банк 2'])
        self.net_box.setStyleSheet('background:#1a1d26;color:%s;border:1px solid #262b36;'
                                   'border-radius:8px;padding:7px' % FG)
        nrow.addWidget(self.net_box, 1)
        self.dice = QtWidgets.QPushButton('🎲 нова')
        self.dice.setCursor(QtCore.Qt.PointingHandCursor); self.dice.setMinimumHeight(34)
        self.dice.setToolTip('Згенерувати свіжу випадкову мережу W~N(0,2/d) — діє на ВСІ панелі')
        self.dice.setStyleSheet('QPushButton{background:#1a1d26;color:%s;border:1px solid #262b36;'
                                'border-radius:8px;font-size:11px;padding:0 10px}'
                                'QPushButton:hover{border-color:%s}' % (FG, ACC2))
        nrow.addWidget(self.dice)
        left.addLayout(nrow)

        lab = QtWidgets.QLabel('обсяг обчислень'); lab.setStyleSheet('color:%s;font-size:11px' % MUT)
        left.addWidget(lab)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(['швидко (×1)', 'середньо (×3)', 'ретельно (×8)'])
        self.scale.setStyleSheet('background:#1a1d26;color:%s;border:1px solid #262b36;'
                                 'border-radius:8px;padding:7px' % FG)
        left.addWidget(self.scale)
        self.acc = QtWidgets.QCheckBox('  НАКОПИЧЕННЯ  (панелі 0, 0b, 6, 7)')
        self.acc.setChecked(True); self.acc.setCursor(QtCore.Qt.PointingHandCursor)
        self.acc.setStyleSheet('QCheckBox{color:%s;font-size:11px;padding:6px}'
                               'QCheckBox::indicator{width:16px;height:16px}' % FG)
        left.addWidget(self.acc)
        # --- керування симуляцією мережі (лише для панелі 0) ---
        self.simbox = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(self.simbox); sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(6)
        ml = QtWidgets.QLabel('що фарбувати'); ml.setStyleSheet('color:%s;font-size:11px' % MUT)
        sv.addWidget(ml)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.setStyleSheet('background:#1a1d26;color:%s;border:1px solid #262b36;'
                                    'border-radius:8px;padding:7px' % FG)
        sv.addWidget(self.mode_box)
        self.mode_hint = QtWidgets.QLabel('')
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setStyleSheet('color:#6b7383;font-size:10px;padding:0 2px')
        sv.addWidget(self.mode_hint)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(6)
        self.step_btn = QtWidgets.QPushButton('⏭ крок')
        self.rst_btn = QtWidgets.QPushButton('⟲ скинути')
        for b in (self.step_btn, self.rst_btn):
            b.setCursor(QtCore.Qt.PointingHandCursor); b.setMinimumHeight(32)
            b.setStyleSheet('QPushButton{background:#1a1d26;color:%s;border:1px solid #262b36;'
                            'border-radius:8px;font-size:11px}QPushButton:hover{border-color:%s}'
                            % (FG, ACC))
            row.addWidget(b)
        sv.addLayout(row)
        left.addWidget(self.simbox)
        self.run = QtWidgets.QPushButton('▶  ЗАПУСТИТИ')
        self.run.setMinimumHeight(46); self.run.setCursor(QtCore.Qt.PointingHandCursor)
        self.run.setStyleSheet("""QPushButton{background:%s;color:#08121a;border:0;border-radius:10px;
            font-size:13px;font-weight:700;letter-spacing:1px}
            QPushButton:hover{background:#7ad6f7}QPushButton:disabled{background:#2a3038;color:%s}"""
            % (ACC, MUT))
        self.run.clicked.connect(self.go); left.addWidget(self.run)
        dv = QtWidgets.QLabel('пристрій: %s' % DEV); dv.setStyleSheet('color:#4a5160;font-size:10px')
        left.addWidget(dv)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(330); lay.addWidget(lw)

        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)
        from netview import NetView
        self.canvas = Canvas()
        self.netview = NetView(weights(0), DEV, D, L)
        self.netview.info.connect(self.status)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.canvas); self.stack.addWidget(self.netview)
        right.addWidget(self.stack, 1)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFixedHeight(190)
        self.log.setStyleSheet('background:#0d0f14;color:#9fb4c7;border:1px solid #1e2530;'
                               'border-radius:10px;padding:9px;font-family:monospace;font-size:11px')
        right.addWidget(self.log)
        rw = QtWidgets.QWidget(); rw.setLayout(right); lay.addWidget(rw, 1)
        self.setStyleSheet('QMainWindow{background:%s}' % BG)
        self.say('пристрій: %s   ·   банк: %s' % (DEV, BANK.split("/")[-1]))
        self.say('обери експеримент і тисни ЗАПУСТИТИ — усе рахується наживо')
        self.mode_box.addItems(self.netview.MODES)
        self.mode_box.currentTextChanged.connect(self.on_mode)
        self.step_btn.clicked.connect(self.netview.tick)
        self.rst_btn.clicked.connect(lambda: self.netview.start(self.acc.isChecked()))
        self.on_mode(self.netview.MODES[0])
        self.dice.clicked.connect(self.new_net)
        self.net_box.currentIndexChanged.connect(self.pick_bank)
        self.seed_n = 0
        self.select(EXPS[0][0])          # стек має відповідати обраній панелі вже на старті

    def on_mode(self, m):
        self.netview.set_mode(m)
        self.mode_hint.setText(self.netview.HINT.get(m, ''))

    def _swap(self, W, what):
        """Підмінити мережу для ВСІХ панелей і перезапустити вигляд."""
        global CUSTOM
        CUSTOM = W
        self.netview.set_weights(weights(0))
        self.say('── %s ──' % what)
        if self.kind == 'net':
            self.netview.start(self.acc.isChecked())
            self.run.setText('◼  ЗУПИНИТИ')
        else:
            self.canvas.reset('мережу змінено — натисни ЗАПУСТИТИ')

    def new_net(self):
        self.seed_n += 1
        seed = 424242 + 7919 * self.seed_n
        self._swap(make_net(seed), 'НОВА ВИПАДКОВА МЕРЕЖА, сід %d  (W~N(0,2/d), критична)' % seed)

    def pick_bank(self, i):
        global CUSTOM
        CUSTOM = None
        self._swap(weights(i), 'мережа з банку №%d' % i)

    def select(self, kind):
        self.kind = kind
        for k, b in self.btns: b.setChecked(k == kind)
        if kind != 'net':
            self.netview.stop()
            if not (self.worker and self.worker.isRunning()):
                self.run.setText('▶  ЗАПУСТИТИ')      # інакше лишається «ЗУПИНИТИ» від мережі
        self.simbox.setVisible(kind == 'net')
        self.stack.setCurrentIndex(1 if kind == 'net' else 0)

    def status(self, t):
        self.log.setPlainText(self.log.toPlainText().rsplit('\n', 1)[0] + '\n' + t
                              if '\n' in self.log.toPlainText() else t)

    def say(self, t):
        self.log.appendPlainText(t)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def go(self):
        if self.kind == 'net':
            if self.netview.timer.isActive():
                self.netview.stop(); self.run.setText('▶  ЗАПУСТИТИ'); return
            self.log.clear()
            self.say('МЕРЕЖА: %d колонок × %d нейронів. Режим: %s'
                     % (L + 1, D, 'НАКОПИЧЕННЯ — проявляється прибите тло'
                        if self.acc.isChecked() else 'ОДИН СЕМПЛ НА КАДР — видно, що блимає'))
            self.say('колесо — масштаб · тягнути — панорама · клік по нейрону · права кнопка — вписати')
            self.netview.reset_view(); self.netview.start(self.acc.isChecked())
            self.run.setText('◼  ЗУПИНИТИ'); return
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag = True; return
        self.log.clear()
        self.canvas.reset('рахую…')
        sc = [1, 3, 8][self.scale.currentIndex()]
        self.worker = Worker(self.kind, sc, self.acc.isChecked())
        self.worker.say.connect(self.say)
        self.worker.plot.connect(self.canvas.draw_data)
        self.worker.fin.connect(self.done)
        self.run.setEnabled(False); self.run.setText('◼  рахую…')
        self.worker.start()

    def done(self):
        self.run.setEnabled(True); self.run.setText('▶  ЗАПУСТИТИ')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    lab = Lab(); lab.show()
    sys.exit(app.exec_())
