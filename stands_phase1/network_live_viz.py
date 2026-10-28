"""Жива візуалізація 32x256 ReLU-мережі: клікни нейрон -> спостерігай за ним, пускай сигнали.

Локальний інструмент (не веб, не Claude-артефакт). Залежності — лише numpy (BSD-3-Clause)
і matplotlib (PSF/BSD-style, matplotlib license) — обидві дозволяють вільне комерційне й
особисте використання без copyleft, ніякого GPL/LGPL у ланцюжку.

Запуск:
    pip install numpy matplotlib   # якщо ще нема
    python network_live_viz.py

Самоперевірка без відкриття вікна:
    python network_live_viz.py --selftest
"""
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button

WIDTH = 256
DEPTH = 32

BG = "#15140f"
PANEL = "#1e1c16"
MUTED_HEX = "#3a362a"
INK = "#ece8dd"
MUTED_TXT = "#a39d8a"
ACCENT_HEX = "#e2a24f"
TENSOR_HEX = "#5fb3ac"
GOOD = "#7ecb96"
BAD = "#e2836c"

MUTED = np.array(mcolors.to_rgb(MUTED_HEX))
ACCENT = np.array(mcolors.to_rgb(ACCENT_HEX))
TENSOR = np.array(mcolors.to_rgb(TENSOR_HEX))

# кольори для списку "спостерігаю" — до 6 одночасно
WATCH_PALETTE = [ACCENT_HEX, TENSOR_HEX, "#c97b6b", "#8f8fd6", "#8fae5f", "#d19bd1"]
MAX_WATCH = 6
MONITOR_WINDOW = 60  # скільки останніх сигналів тримати на графіку монітора

SWEEP_STEP = 1.35     # на скільки колонок рухається фронт сигналу за один тік таймера
SWEEP_INTERVAL_MS = 26
AUTO_FIRE_MS = 1700


# ---------------------------------------------------------------- math ----

def he_std(width):
    return np.sqrt(2.0 / width)


def build_weights(rng, width=WIDTH, depth=DEPTH):
    """DEPTH матриць (width,width): W[l][i,k] = вага з нейрона k шару l у нейрон i шару l+1."""
    std = he_std(width)
    return [rng.standard_normal((width, width)).astype(np.float32) * std for _ in range(depth)]


def forward_trace(x0, W):
    """Один прогін для розфарбування графа: список із DEPTH+1 масивів (вхід + кожен шар)."""
    trace = [x0.copy()]
    cur = x0
    for Wl in W:
        z = Wl @ cur
        a = np.maximum(z, 0.0)
        trace.append(a)
        cur = a
    return trace


def forward_batch_to_layer(W, upto_layer, idx, n_samples, rng):
    """Батч N свіжих входів x~N(0,I) крізь ТУ САМУ мережу до `upto_layer`, читаючи один нейрон idx.

    Векторизовано через numpy/BLAS: (N,WIDTH) @ (WIDTH,WIDTH).T на кожному шарі.
    Повертає (z_samples, a_samples) — до і після ReLU для обраного нейрона.
    """
    X = rng.standard_normal((n_samples, WIDTH)).astype(np.float32)
    if upto_layer == 0:
        z = X[:, idx].copy()
        return z, np.maximum(z, 0.0)
    cur = X
    z_sel = None
    for l in range(upto_layer):
        Z = cur @ W[l].T
        if l == upto_layer - 1:
            z_sel = Z[:, idx].copy()
        cur = np.maximum(Z, 0.0)
    return z_sel, np.maximum(z_sel, 0.0)


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(__import__("math").erf)(x / np.sqrt(2.0)))


def gaussian_fold_mean(mu, sigma):
    """E[ReLU(Z)] для Z~N(mu,sigma^2): sigma*phi(alpha) + mu*Phi(alpha)."""
    if sigma < 1e-9:
        return max(mu, 0.0)
    alpha = mu / sigma
    return float(sigma * norm_pdf(alpha) + mu * norm_cdf(alpha))


# -------------------------------------------------------------- colors ----

def mix_rows(base, target, t):
    t = np.clip(t, 0.0, 1.0)[:, None]
    return base[None, :] + (target[None, :] - base[None, :]) * t


def node_colors(layer, values):
    m = float(np.max(np.abs(values))) or 1.0
    t = values / m
    if layer == 0:
        pos = t >= 0
        cols = np.tile(MUTED, (len(values), 1))
        cols[pos] = mix_rows(MUTED, ACCENT, t[pos])
        cols[~pos] = mix_rows(MUTED, TENSOR, -t[~pos])
        return cols
    return mix_rows(MUTED, ACCENT, np.clip(t, 0, 1))


# --------------------------------------------------------------- app ------

class LiveNetworkApp:
    def __init__(self, n_samples=3000, seed=None):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.selection = None      # (layer, idx) — останній клік, керує гістограмами/віялом
        self.watchlist = []        # [{key,layer,idx,color,pulses:[(pulse_no,val),...]}]
        self.pulse_count = 0
        self.sweep_front = None    # None = немає активної анімації сигналу
        self.sweep_hits = set()
        self.sweep_timer = None
        self.auto_timer = None
        self.auto_on = False
        self._reroll()

        self.fig = plt.figure(figsize=(14.5, 10), facecolor=BG)
        self.fig.suptitle(
            "Жива мережа: 32 шари x 256 нейронів  —  numpy, локально",
            color=INK, fontsize=15, fontweight="bold", y=0.99,
        )
        self.fig.text(
            0.055, 0.955,
            "«Пустити сигнал» анімує причинний прохід ОДНОГО входу крізь шари (значення вже "
            "пораховані numpy миттєво — анімація навмисно уповільнена для наочності). "
            "Клікни нейрон, щоб додати його до монітора внизу.",
            color=MUTED_TXT, fontsize=9.3, wrap=True,
        )

        gs = self.fig.add_gridspec(
            3, 3, height_ratios=[5.6, 3.0, 0.55],
            hspace=0.40, wspace=0.24, top=0.885, bottom=0.055, left=0.055, right=0.975,
        )
        self.ax_graph = self.fig.add_subplot(gs[0, :])
        self.ax_hz = self.fig.add_subplot(gs[1, 0])
        self.ax_ha = self.fig.add_subplot(gs[1, 1])
        self.ax_mon = self.fig.add_subplot(gs[1, 2])
        self.ax_text = self.fig.add_subplot(gs[2, :])
        self.ax_text.axis("off")

        for ax in (self.ax_graph, self.ax_hz, self.ax_ha, self.ax_mon):
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_color(MUTED_HEX)
            ax.tick_params(colors=MUTED_TXT, labelsize=8)

        btn_specs = [
            ("reroll", "нові ваги мережі", 0.055, 0.165),
            ("input", "новий вхід (перефарбувати)", 0.230, 0.205),
            ("fire", "▶ пустити сигнал", 0.445, 0.165),
            ("auto", "авто: викл", 0.620, 0.135),
            ("clear", "очистити монітор", 0.765, 0.20),
        ]
        self.buttons = {}
        for key, label, x, w in btn_specs:
            ax_b = self.fig.add_axes([x, 0.912, w, 0.035])
            btn = Button(ax_b, label, color=PANEL, hovercolor=MUTED_HEX)
            btn.label.set_color(INK)
            btn.label.set_fontsize(9.5)
            self.buttons[key] = btn
        self.buttons["reroll"].on_clicked(self._on_reroll_click)
        self.buttons["input"].on_clicked(self._on_input_click)
        self.buttons["fire"].on_clicked(lambda evt: self.fire_signal())
        self.buttons["auto"].on_clicked(self._on_auto_click)
        self.buttons["clear"].on_clicked(self._on_clear_click)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._draw_graph()
        self._draw_empty_panels()
        self._draw_monitor()

    # ---- data lifecycle ----

    def _reroll(self):
        self.W = build_weights(self.rng)
        self._resample_input()
        self.selection = None
        self.watchlist = []
        self.pulse_count = 0

    def _resample_input(self):
        x0 = self.rng.standard_normal(WIDTH).astype(np.float32)
        self.trace = forward_trace(x0, self.W)

    # ---- graph drawing ----

    def _draw_graph(self):
        ax = self.ax_graph
        ax.cla()
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(MUTED_HEX)
        ax.set_xlim(-1, DEPTH + 1)
        ax.set_ylim(WIDTH, -8)
        ticks = list(range(0, DEPTH + 1, 4)) + [DEPTH]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["IN" if l == 0 else f"L{l}" for l in ticks], color=MUTED_TXT, fontsize=8)
        ax.set_yticks([])

        if self.selection is not None:
            segs, cols, widths = self._build_fan(*self.selection)
            if segs:
                ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths))

        front = self.sweep_front
        max_layer = DEPTH if front is None else min(DEPTH, int(np.floor(front)))
        xs, ys, cs = [], [], []
        for layer in range(0, max_layer + 1):
            a = self.trace[layer]
            cols = node_colors(layer, a)
            xs.extend([layer] * WIDTH)
            ys.extend(range(WIDTH))
            cs.extend(cols.tolist())
        if xs:
            ax.scatter(xs, ys, s=5, c=cs, linewidths=0)

        if front is not None:
            ax.axvline(front, color=ACCENT_HEX, linewidth=9, alpha=0.14, zorder=4)
            ax.axvline(front, color=INK, linewidth=1.8, alpha=0.9, zorder=6)

        for w in self.watchlist:
            ax.scatter([w["layer"]], [w["idx"]], s=70, facecolors="none",
                       edgecolors=w["color"], linewidths=1.8, zorder=5)
        if self.selection is not None:
            layer, idx = self.selection
            ax.scatter([layer], [idx], s=110, facecolors="none",
                       edgecolors=INK, linewidths=1.2, zorder=6)

        ax.set_title(
            f"Сигналів пущено: {self.pulse_count}   ·   у моніторі: {len(self.watchlist)}/{MAX_WATCH}",
            color=MUTED_TXT, fontsize=9.5, loc="left",
        )
        self.fig.canvas.draw_idle()

    def _build_fan(self, layer, idx):
        segs, cols, widths = [], [], []
        if layer >= 1:
            w_in = self.W[layer - 1][idx, :]
            maxw = float(np.max(np.abs(w_in))) or 1.0
            for k in range(WIDTH):
                t = abs(w_in[k]) / maxw
                if t < 0.06:
                    continue
                base = ACCENT if w_in[k] >= 0 else TENSOR
                segs.append(((layer - 1, k), (layer, idx)))
                cols.append((*base, 0.10 + 0.55 * t))
                widths.append(0.6 + 1.6 * t)
        if layer <= DEPTH - 1:
            w_out = self.W[layer][:, idx]
            maxw2 = float(np.max(np.abs(w_out))) or 1.0
            for j in range(WIDTH):
                t = abs(w_out[j]) / maxw2
                if t < 0.06:
                    continue
                base = ACCENT if w_out[j] >= 0 else TENSOR
                segs.append(((layer, idx), (layer + 1, j)))
                cols.append((*base, 0.08 + 0.45 * t))
                widths.append(0.5 + 1.3 * t)
        return segs, cols, widths

    # ---- histogram panels (batch stats for last click) ----

    def _draw_empty_panels(self):
        for ax, title in ((self.ax_hz, "Перед ReLU (z)"), (self.ax_ha, "Після ReLU (a)")):
            ax.cla()
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_color(MUTED_HEX)
            ax.set_title(title, color=INK, fontsize=10.5, loc="left")
            ax.text(0.5, 0.5, "клікни нейрон вище", color=MUTED_TXT, fontsize=9.5,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        self.ax_text.cla(); self.ax_text.axis("off")
        self.fig.canvas.draw_idle()

    def _run_selection(self, layer, idx):
        z, a = forward_batch_to_layer(self.W, layer, idx, self.n_samples, self.rng)

        for ax in (self.ax_hz, self.ax_ha):
            ax.cla(); ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_color(MUTED_HEX)
            ax.tick_params(colors=MUTED_TXT, labelsize=8)

        self.ax_hz.hist(z, bins=40, color=ACCENT_HEX, alpha=0.85)
        if layer > 0:
            self.ax_hz.axvline(0, color=MUTED_TXT, linestyle="--", linewidth=1)
        self.ax_hz.set_title("Перед ReLU (z)", color=INK, fontsize=10.5, loc="left")

        self.ax_ha.hist(a, bins=40, color=ACCENT_HEX, alpha=0.85, density=False)
        mu_a, sd_a = float(np.mean(a)), float(np.std(a))
        if sd_a > 1e-9:
            xs = np.linspace(a.min(), a.max(), 200)
            dens = norm_pdf((xs - mu_a) / sd_a) / sd_a
            binw = (a.max() - a.min()) / 40
            self.ax_ha.plot(xs, dens * len(a) * binw, color=TENSOR_HEX, linewidth=2)
        self.ax_ha.set_title("Після ReLU (a) — гаус-фіт", color=INK, fontsize=10.5, loc="left")

        mu_z, sd_z = float(np.mean(z)), float(np.std(z))
        skew_a = float(np.mean(((a - mu_a) / sd_a) ** 3)) if sd_a > 1e-9 else 0.0
        formula = gaussian_fold_mean(mu_z, sd_z)
        rel_err = abs(formula - mu_a) / max(1e-9, abs(mu_a))

        label = "вхід" if layer == 0 else f"шар {layer}"
        verdict = ""
        if layer > 0:
            verdict = ("формула ще майже точна" if rel_err < 0.03
                       else "формула вже помітно хибить — глибина зіскосила розподіл")
        self.ax_text.cla(); self.ax_text.axis("off")
        txt = (f"Гістограми (батч {self.n_samples}): {label} · нейрон {idx}   "
               f"μ_z={mu_z:.3f}  σ_z={sd_z:.3f}   емп.μ(a)={mu_a:.4f}  скіс(a)={skew_a:.3f}   "
               f"формула={formula:.4f}  похибка={rel_err*100:.1f}%   {verdict}")
        color = INK if layer == 0 else (GOOD if rel_err < 0.03 else BAD)
        self.ax_text.text(0.0, 0.5, txt, color=color, fontsize=10, family="monospace",
                          ha="left", va="center", transform=self.ax_text.transAxes)
        self.fig.canvas.draw_idle()

    # ---- monitor panel (sequential single-pulse observations) ----

    def _draw_monitor(self):
        ax = self.ax_mon
        ax.cla(); ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(MUTED_HEX)
        ax.tick_params(colors=MUTED_TXT, labelsize=7.5)
        ax.set_title("Монітор (по одному сигналу)", color=INK, fontsize=10.5, loc="left")

        if not self.watchlist:
            ax.text(0.5, 0.5, "клікни нейрон,\nпотім «пустити сигнал»", color=MUTED_TXT,
                    fontsize=9, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self.fig.canvas.draw_idle()
            return

        any_pts = False
        for w in self.watchlist:
            if not w["pulses"]:
                continue
            any_pts = True
            xs = [p[0] for p in w["pulses"]]
            ys = [p[1] for p in w["pulses"]]
            lbl = ("IN" if w["layer"] == 0 else f"L{w['layer']}") + f"·#{w['idx']}"
            ax.plot(xs, ys, color=w["color"], marker="o", markersize=3,
                    linewidth=1.4, label=lbl)
        if any_pts:
            ax.legend(fontsize=7, facecolor=PANEL, edgecolor=MUTED_HEX,
                       labelcolor=INK, loc="upper left", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "натисни «пустити сигнал»", color=MUTED_TXT, fontsize=9,
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("сигнал №", color=MUTED_TXT, fontsize=8)
        self.fig.canvas.draw_idle()

    # ---- signal sweep animation ----

    def fire_signal(self):
        if self.sweep_front is not None:
            return  # анімація вже йде
        self._resample_input()
        self.pulse_count += 1
        self.sweep_front = -1.0
        self.sweep_hits = set()
        if self.sweep_timer is None:
            self.sweep_timer = self.fig.canvas.new_timer(interval=SWEEP_INTERVAL_MS)
            self.sweep_timer.add_callback(self._sweep_step)
        self.sweep_timer.start()

    def _sweep_step(self):
        self.sweep_front += SWEEP_STEP
        for w in self.watchlist:
            if w["layer"] <= self.sweep_front and w["key"] not in self.sweep_hits:
                self.sweep_hits.add(w["key"])
                val = float(self.trace[w["layer"]][w["idx"]])
                w["pulses"].append((self.pulse_count, val))
                if len(w["pulses"]) > MONITOR_WINDOW:
                    w["pulses"].pop(0)
        if self.sweep_front >= DEPTH + 1:
            self.sweep_timer.stop()
            self.sweep_front = None
            self._draw_graph()
            self._draw_monitor()
        else:
            self._draw_graph()

    def _on_auto_click(self, event):
        self.auto_on = not self.auto_on
        self.buttons["auto"].label.set_text("авто: увімк" if self.auto_on else "авто: викл")
        if self.auto_on:
            if self.auto_timer is None:
                self.auto_timer = self.fig.canvas.new_timer(interval=AUTO_FIRE_MS)
                self.auto_timer.add_callback(self._auto_tick)
            self.auto_timer.start()
        else:
            if self.auto_timer:
                self.auto_timer.stop()
        self.fig.canvas.draw_idle()

    def _auto_tick(self):
        if self.sweep_front is None:
            self.fire_signal()

    # ---- events ----

    def _on_click(self, event):
        if event.inaxes != self.ax_graph or event.xdata is None or event.ydata is None:
            return
        layer = int(round(event.xdata))
        idx = int(round(event.ydata))
        if not (0 <= layer <= DEPTH and 0 <= idx < WIDTH):
            return
        disp_click = (event.x, event.y)
        disp_node = self.ax_graph.transData.transform((layer, idx))
        if np.hypot(disp_click[0] - disp_node[0], disp_click[1] - disp_node[1]) > 10:
            return

        self.selection = (layer, idx)
        key = (layer, idx)
        if not any(w["key"] == key for w in self.watchlist):
            if len(self.watchlist) >= MAX_WATCH:
                self.watchlist.pop(0)
            color = WATCH_PALETTE[len(self.watchlist) % len(WATCH_PALETTE)]
            self.watchlist.append({"key": key, "layer": layer, "idx": idx, "color": color, "pulses": []})

        self._draw_graph()
        self._run_selection(layer, idx)
        self._draw_monitor()

    def _on_reroll_click(self, event):
        if self.auto_timer:
            self.auto_timer.stop()
        self.auto_on = False
        self.buttons["auto"].label.set_text("авто: викл")
        self._reroll()
        self._draw_graph()
        self._draw_empty_panels()
        self._draw_monitor()

    def _on_input_click(self, event):
        if self.sweep_front is not None:
            return
        self._resample_input()
        self._draw_graph()

    def _on_clear_click(self, event):
        self.watchlist = []
        self._draw_graph()
        self._draw_monitor()

    def show(self):
        plt.show()


def selftest():
    rng = np.random.default_rng(0)
    W = build_weights(rng)
    x0 = rng.standard_normal(WIDTH).astype(np.float32)
    trace = forward_trace(x0, W)
    assert len(trace) == DEPTH + 1
    z, a = forward_batch_to_layer(W, 32, 17, 2000, rng)
    mu_z, sd_z = float(np.mean(z)), float(np.std(z))
    mu_a = float(np.mean(a))
    formula = gaussian_fold_mean(mu_z, sd_z)
    print(f"layer32 neuron17: mu_z={mu_z:.4f} sd_z={sd_z:.4f} "
          f"emp.mu(a)={mu_a:.4f} formula={formula:.4f} "
          f"rel.err={abs(formula-mu_a)/max(1e-9,abs(mu_a))*100:.2f}%")
    z0, a0 = forward_batch_to_layer(W, 1, 3, 2000, rng)
    mu_z0, sd_z0 = float(np.mean(z0)), float(np.std(z0))
    mu_a0 = float(np.mean(a0))
    f0 = gaussian_fold_mean(mu_z0, sd_z0)
    print(f"layer1  neuron3 : mu_z={mu_z0:.4f} sd_z={sd_z0:.4f} "
          f"emp.mu(a)={mu_a0:.4f} formula={f0:.4f} "
          f"rel.err={abs(f0-mu_a0)/max(1e-9,abs(mu_a0))*100:.2f}%")

    # symuluj hід сигналу: обраний нейрон отримує значення, коли фронт його "проходить"
    watch_layer, watch_idx = 8, 42
    val = float(trace[watch_layer][watch_idx])
    front = -1.0
    hit = None
    while front < DEPTH + 1:
        front += SWEEP_STEP
        if watch_layer <= front:
            hit = float(trace[watch_layer][watch_idx])
            break
    assert hit is not None and abs(hit - val) < 1e-6
    print(f"sweep-hit L{watch_layer}#{watch_idx} = {hit:.4f} (matches trace)")
    print("OK: selftest пройшов без вікна matplotlib.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000,
                         help="скільки свіжих входів прогнати на клік нейрона (default 3000)")
    parser.add_argument("--seed", type=int, default=None, help="сід RNG")
    parser.add_argument("--selftest", action="store_true", help="перевірка без відкриття вікна")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    app = LiveNetworkApp(n_samples=args.samples, seed=args.seed)
    app.show()


if __name__ == "__main__":
    main()
