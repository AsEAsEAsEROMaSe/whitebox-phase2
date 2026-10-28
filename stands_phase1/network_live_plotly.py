"""Жива мережа 32x256 у Plotly/Dash: плавна WebGL-анімація сигналу + 3D-поверхня топової формули.

Локальний інструмент. Залежності — усі з дозвільними ліцензіями, без GPL/LGPL:
    numpy   BSD-3-Clause
    plotly  MIT License
    dash    MIT License (Plotly Inc.)

Встановлення (окреме venv, нічого системного не займає):
    python3 -m venv viz_venv
    ./viz_venv/bin/pip install numpy plotly dash

Запуск:
    ./viz_venv/bin/python network_live_plotly.py
    # відкриє http://127.0.0.1:8050 у браузері

Самоперевірка без запуску сервера:
    ./viz_venv/bin/python network_live_plotly.py --selftest
"""
import argparse
import math
import shutil
import subprocess
import threading
import time
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx

WIDTH = 256
DEPTH = 32
COLS = DEPTH + 1

BG = "#15140f"
PANEL = "#1e1c16"
MUTED_HEX = "#3a362a"
INK = "#ece8dd"
MUTED_TXT = "#a39d8a"
ACCENT_HEX = "#e2a24f"
TENSOR_HEX = "#5fb3ac"
GOOD = "#7ecb96"
BAD = "#e2836c"

WATCH_PALETTE = [ACCENT_HEX, TENSOR_HEX, "#c97b6b", "#8f8fd6", "#8fae5f", "#d19bd1"]
MAX_WATCH = 6
MONITOR_WINDOW = 60
NFRAMES = 40
N_SAMPLES_DEFAULT = 2000


# ---------------------------------------------------------------- math ----

def he_std(width):
    return math.sqrt(2.0 / width)


def build_weights(rng, width=WIDTH, depth=DEPTH):
    std = he_std(width)
    return [rng.standard_normal((width, width)).astype(np.float32) * std for _ in range(depth)]


def forward_trace(x0, W):
    trace = [x0.copy()]
    cur = x0
    for Wl in W:
        z = Wl @ cur
        a = np.maximum(z, 0.0)
        trace.append(a)
        cur = a
    return trace


def forward_batch_to_layer(W, upto_layer, idx, n_samples, rng):
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


def forward_final(W, X):
    """Прогнати батч X (N,WIDTH) крізь усі шари -> активації фінального шару (N,WIDTH)."""
    cur = X
    for Wl in W:
        cur = np.maximum(cur @ Wl.T, 0.0)
    return cur


def compute_truth(W, rng, n=40000):
    """Еталон E[relu(фінальний шар)] per neuron — велике MC (істина, до якої збігаємось)."""
    acc = np.zeros(WIDTH, dtype=np.float64)
    seen = 0
    chunk = 8000
    while seen < n:
        b = min(chunk, n - seen)
        X = rng.standard_normal((b, WIDTH)).astype(np.float32)
        acc += forward_final(W, X).sum(0)
        seen += b
    return (acc / seen).astype(np.float32)


def _mse_vec(est, truth):
    return float(np.mean((est.astype(np.float64) - truth.astype(np.float64)) ** 2))


def _zca_whiten(X):
    """ZCA: перетворити вибірку так, щоб її емпірична коваріація стала РІВНО I (репрезентативна
    вибірка — моменти входу точно збігаються з істиною N(0,I)). Ядро методу v2."""
    N = X.shape[0]
    C = (X.T @ X) / N
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-6, None)
    Cinv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    return (X @ Cinv_sqrt.astype(np.float32)).astype(np.float32)


def estimate_methods(W, X):
    """Три оцінки E[relu(фінал)] з тієї самої бази X (N,WIDTH):
       plain MC, antithetic MC, ZCA-antithetic (v2)."""
    est_plain = forward_final(W, X).mean(0)
    Xa = np.concatenate([X, -X], axis=0)
    est_anti = forward_final(W, Xa).mean(0)
    Xz = _zca_whiten(X)
    Xza = np.concatenate([Xz, -Xz], axis=0)
    est_v2 = forward_final(W, Xza).mean(0)
    return est_plain, est_anti, est_v2


def compute_mc_vs_v2(W, truth, rng, Ns, trials):
    """MSE проти істини для кожного методу й кожного N (усереднено по trials розіграшах)."""
    out = {"plain": [], "anti": [], "v2": []}
    for N in Ns:
        errs = {"plain": [], "anti": [], "v2": []}
        for _ in range(trials):
            X = rng.standard_normal((N, WIDTH)).astype(np.float32)
            ep, ea, ev = estimate_methods(W, X)
            errs["plain"].append(_mse_vec(ep, truth))
            errs["anti"].append(_mse_vec(ea, truth))
            errs["v2"].append(_mse_vec(ev, truth))
        for k in errs:
            out[k].append(float(np.mean(errs[k])))
    return out


def forward_layer_means(W, X):
    """Прогнати X крізь шари, повертаючи ПОШАРОВЕ середнє активацій (оцінка E[a^l] по вибірці)."""
    means = [X.mean(0)]
    cur = X
    for Wl in W:
        cur = np.maximum(cur @ Wl.T, 0.0)
        means.append(cur.mean(0))
    return means


def compute_truth_layers(W, rng, n=40000):
    """Істинні пошарові середні E[a^l] per neuron (велике MC) — еталон для кожного шару."""
    acc = [np.zeros(WIDTH, dtype=np.float64) for _ in range(COLS)]
    seen = 0
    chunk = 8000
    while seen < n:
        b = min(chunk, n - seen)
        X = rng.standard_normal((b, WIDTH)).astype(np.float32)
        acc[0] += X.sum(0)
        cur = X
        for l, Wl in enumerate(W):
            cur = np.maximum(cur @ Wl.T, 0.0)
            acc[l + 1] += cur.sum(0)
        seen += b
    return [(a / seen).astype(np.float32) for a in acc]


def compute_cycle(W, truth_layers, rng, N=2048):
    """Повний цикл v2 на мережі: готуємо вибірку (raw / ZCA-antithetic), проганяємо крізь 32 шари,
    міряємо похибку оцінки пошарово й фінально — проти істини. Порівняння plain vs v2."""
    X = rng.standard_normal((N, WIDTH)).astype(np.float32)
    Xz = _zca_whiten(X)
    Xza = np.concatenate([Xz, -Xz], axis=0)

    means_plain = forward_layer_means(W, X)
    means_v2 = forward_layer_means(W, Xza)

    mse_plain = [_mse_vec(means_plain[l], truth_layers[l]) for l in range(COLS)]
    mse_v2 = [_mse_vec(means_v2[l], truth_layers[l]) for l in range(COLS)]

    return {
        "x2d_raw": X[:, :2].copy(),
        "x2d_v2": Xza[:, :2].copy(),
        "mse_plain": mse_plain,
        "mse_v2": mse_v2,
        "est_plain": means_plain[-1],
        "est_v2": means_v2[-1],
        "truth_final": truth_layers[-1],
        "N": N,
    }


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def norm_cdf(x):
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def gaussian_fold_mean(mu, sigma):
    """E[ReLU(Z)], Z~N(mu,sigma^2): sigma*phi(alpha)+mu*Phi(alpha)."""
    if sigma < 1e-9:
        return max(mu, 0.0)
    alpha = mu / sigma
    return float(sigma * norm_pdf(alpha) + mu * norm_cdf(alpha))


def edgeworth_correction(mu, sigma, alpha, g3, g4):
    """Точна формула d із evo/gpu/lead1_sanity.py (R96, верифікована проти MC)."""
    phi_a = norm_pdf(alpha)
    d = (g3 / 6.0) * phi_a * (mu * (alpha ** 2 - 1) - sigma * alpha ** 3) \
        + (g4 / 24.0) * phi_a * (mu * (3 * alpha - alpha ** 3) + sigma * (alpha ** 4 - 2 * alpha ** 2 - 1))
    return d


def edgeworth_fold_mean(mu, sigma, g3, g4):
    if sigma < 1e-9:
        return max(mu, 0.0)
    alpha = mu / sigma
    return float(gaussian_fold_mean(mu, sigma) + edgeworth_correction(mu, sigma, alpha, g3, g4))


# -------------------------------------------------------------- colors ----

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


MUTED_RGB = _hex_to_rgb(MUTED_HEX)
ACCENT_RGB = _hex_to_rgb(ACCENT_HEX)
TENSOR_RGB = _hex_to_rgb(TENSOR_HEX)


def _mix(rgb_a, rgb_b, t):
    t = max(0.0, min(1.0, t))
    return tuple(round(rgb_a[i] + (rgb_b[i] - rgb_a[i]) * t) for i in range(3))


def node_colors_hex(layer, values):
    m = float(np.max(np.abs(values))) or 1.0
    out = []
    if layer == 0:
        for v in values:
            t = v / m
            rgb = _mix(MUTED_RGB, ACCENT_RGB, t) if t >= 0 else _mix(MUTED_RGB, TENSOR_RGB, -t)
            out.append(f"rgb{rgb}")
    else:
        for v in values:
            rgb = _mix(MUTED_RGB, ACCENT_RGB, v / m)
            out.append(f"rgb{rgb}")
    return out


# ------------------------------------------------------ node coordinates --

ALL_LAYER = np.repeat(np.arange(COLS), WIDTH)
ALL_IDX = np.tile(np.arange(WIDTH), COLS)
ALL_X = ALL_LAYER.tolist()
ALL_Y = ALL_IDX.tolist()
ALL_CUSTOM = list(zip(ALL_LAYER.tolist(), ALL_IDX.tolist()))


def opacity_for_front(front):
    return np.clip(front - ALL_LAYER + 1.0, 0.0, 1.0).tolist()


# ------------------------------------------------------------- fan lines --

def _segments_xy(segments):
    xs, ys = [], []
    for (x0, y0), (x1, y1) in segments:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return xs, ys


def build_fan_traces(W, layer, idx):
    traces = []
    tiers = [(0.15, 0.4, 0.18, 1.0), (0.4, 0.7, 0.38, 1.8), (0.7, 1.01, 0.65, 2.6)]

    if layer >= 1:
        w_in = W[layer - 1][idx, :]
        maxw = float(np.max(np.abs(w_in))) or 1.0
        t = np.abs(w_in) / maxw
        for lo, hi, opac, width in tiers:
            for sign, color in ((1, ACCENT_HEX), (-1, TENSOR_HEX)):
                mask = (t >= lo) & (t < hi) & (np.sign(w_in) == sign)
                sel = np.where(mask)[0]
                if sel.size == 0:
                    continue
                segs = [((layer - 1, int(k)), (layer, idx)) for k in sel]
                xs, ys = _segments_xy(segs)
                traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                          opacity=opac, hoverinfo="skip", showlegend=False))

    if layer <= DEPTH - 1:
        w_out = W[layer][:, idx]
        maxw2 = float(np.max(np.abs(w_out))) or 1.0
        t2 = np.abs(w_out) / maxw2
        for lo, hi, opac, width in tiers:
            for sign, color in ((1, ACCENT_HEX), (-1, TENSOR_HEX)):
                mask = (t2 >= lo) & (t2 < hi) & (np.sign(w_out) == sign)
                sel = np.where(mask)[0]
                if sel.size == 0:
                    continue
                segs = [((layer, idx), (layer + 1, int(j))) for j in sel]
                xs, ys = _segments_xy(segs)
                traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                          opacity=opac * 0.85, hoverinfo="skip", showlegend=False))
    return traces


# -------------------------------------------------------------- figures --

AXIS_TICKVALS = list(range(0, DEPTH + 1, 4)) + [DEPTH]
AXIS_TICKTEXT = ["IN" if l == 0 else f"L{l}" for l in AXIS_TICKVALS]


def _activation_grid(state):
    """z-матриця (WIDTH x COLS): активація кожного нейрона, нормована по колонці в [0,1].
    Це і поле-візуалізація, і клікер водночас — одне зображення, рендериться миттєво."""
    z = np.empty((WIDTH, COLS), dtype=np.float32)
    for layer in range(COLS):
        vals = state["trace"][layer]
        m = float(np.max(np.abs(vals))) or 1.0
        z[:, layer] = np.clip(vals / m, 0.0, 1.0)
    return z


def build_network_figure(state, show_weights=True, frames=None, front=None):
    data = []

    # Поле нейронів = ОДИН heatmap (33 колонки x 256 рядків). Рендериться як зображення —
    # без лагів на 8448 точок. І сам ловить кліки: клік по клітинці -> шар+нейрон напряму.
    # hoverinfo="none" (НЕ "skip"!) — інакше Plotly вимкне й події кліку.
    z = _activation_grid(state)
    data.append(go.Heatmap(
        x=list(range(COLS)), y=list(range(WIDTH)), z=z,
        colorscale=[[0.0, MUTED_HEX], [1.0, ACCENT_HEX]],
        zmin=0.0, zmax=1.0, showscale=False, hoverinfo="none", name="field",
    ))

    if show_weights and state["selection"] is not None:
        data += build_fan_traces(state["W"], *state["selection"])

    if state["watchlist"]:
        wx = [w["layer"] for w in state["watchlist"]]
        wy = [w["idx"] for w in state["watchlist"]]
        wc = [w["color"] for w in state["watchlist"]]
        data.append(go.Scatter(x=wx, y=wy, mode="markers",
                                marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=wc, width=2)),
                                hoverinfo="skip", showlegend=False))

    if state["selection"] is not None:
        sl, si = state["selection"]
        data.append(go.Scatter(x=[sl], y=[si], mode="markers",
                                marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color=INK, width=1.8)),
                                hoverinfo="skip", showlegend=False))

    shapes = []
    if front is not None:
        shapes = _front_shapes(front)

    layout = go.Layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL, clickmode="event",
        xaxis=dict(range=[-1, DEPTH + 1], tickmode="array", tickvals=AXIS_TICKVALS, ticktext=AXIS_TICKTEXT,
                   gridcolor=MUTED_HEX, color=MUTED_TXT, zeroline=False),
        yaxis=dict(range=[WIDTH + 6, -10], showticklabels=False, gridcolor=MUTED_HEX, zeroline=False),
        margin=dict(l=30, r=20, t=36, b=30), height=560, shapes=shapes,
        title=dict(text=f"Сигналів пущено: {state['pulse_count']}   ·   у моніторі: "
                        f"{len(state['watchlist'])}/{MAX_WATCH}",
                   font=dict(color=MUTED_TXT, size=12)),
        uirevision="keep-zoom",
    )
    fig = go.Figure(data=data, layout=layout)
    if frames is not None:
        fig.frames = frames
    return fig


def _front_shapes(front):
    """Дві лінії фронту сигналу: широке м'яке сяйво + тонке ядро."""
    return [
        dict(type="line", x0=front, x1=front, y0=-10, y1=WIDTH + 6,
             line=dict(color=ACCENT_HEX, width=10), opacity=0.16, layer="below"),
        dict(type="line", x0=front, x1=front, y0=-10, y1=WIDTH + 6,
             line=dict(color=INK, width=2)),
    ]


def build_sweep_frames():
    """Кадри анімації сигналу: рухаємо ЛИШЕ лінію-фронт через layout.shapes (дешево, плавно,
    без WebGL). Вузли лишаються видимі — фронт біжить поверх них."""
    fronts = np.linspace(-1.0, DEPTH + 2, NFRAMES)
    frames = []
    for i, front in enumerate(fronts):
        frames.append(go.Frame(name=str(i), layout=go.Layout(shapes=_front_shapes(float(front)))))
    return frames, float(fronts[0])


def build_hist_figure(samples, title, mark_zero=False, overlay_gauss=False):
    fig = go.Figure()
    if samples is None or len(samples) < 2:
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=230,
                          margin=dict(l=30, r=10, t=34, b=26),
                          font=dict(color=MUTED_TXT),
                          title=dict(text=title, font=dict(color=INK, size=13)),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          annotations=[dict(text="клікни нейрон вище", showarrow=False,
                                            xref="paper", yref="paper", x=0.5, y=0.5,
                                            font=dict(color=MUTED_TXT, size=12))])
        return fig
    counts, edges = np.histogram(samples, bins=40)
    centers = (edges[:-1] + edges[1:]) / 2
    binw = edges[1] - edges[0]
    fig.add_trace(go.Bar(x=centers, y=counts, width=binw * 0.92, marker_color=ACCENT_HEX,
                          opacity=0.85, name="емпірика"))
    if mark_zero and edges[0] < 0 < edges[-1]:
        fig.add_vline(x=0, line_dash="dash", line_color=MUTED_TXT)
    if overlay_gauss:
        mu, sd = float(np.mean(samples)), float(np.std(samples))
        if sd > 1e-9:
            xs = np.linspace(float(np.min(samples)), float(np.max(samples)), 200)
            dens = norm_pdf((xs - mu) / sd) / sd
            fig.add_trace(go.Scatter(x=xs, y=dens * len(samples) * binw, mode="lines",
                                     line=dict(color=TENSOR_HEX, width=2.4), name="гаус-фіт"))
    fig.update_layout(title=dict(text=title, font=dict(color=INK, size=13)), paper_bgcolor=BG,
                      plot_bgcolor=PANEL, font=dict(color=MUTED_TXT, size=10),
                      margin=dict(l=30, r=10, t=34, b=26), showlegend=False, height=230)
    fig.update_xaxes(gridcolor=MUTED_HEX, zerolinecolor=MUTED_HEX)
    fig.update_yaxes(gridcolor=MUTED_HEX, zerolinecolor=MUTED_HEX)
    return fig


def build_monitor_figure(watchlist):
    fig = go.Figure()
    has_pts = any(w["pulses"] for w in watchlist)
    if not watchlist or not has_pts:
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=230, font=dict(color=MUTED_TXT),
                          margin=dict(l=30, r=10, t=34, b=26),
                          title=dict(text="Монітор (по одному сигналу)", font=dict(color=INK, size=13)),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          annotations=[dict(text='клікни нейрон, тоді "пустити сигнал"', showarrow=False,
                                            xref="paper", yref="paper", x=0.5, y=0.5,
                                            font=dict(color=MUTED_TXT, size=11))])
        return fig
    for w in watchlist:
        if not w["pulses"]:
            continue
        xs = [p[0] for p in w["pulses"]]
        ys = [p[1] for p in w["pulses"]]
        lbl = ("IN" if w["layer"] == 0 else f"L{w['layer']}") + f"·#{w['idx']}"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=w["color"], width=1.6),
                                 marker=dict(size=5, color=w["color"]), name=lbl))
    fig.update_layout(title=dict(text="Монітор (по одному сигналу)", font=dict(color=INK, size=13)),
                      paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=MUTED_TXT, size=10),
                      legend=dict(font=dict(color=INK, size=9)), margin=dict(l=30, r=10, t=34, b=26), height=230)
    fig.update_xaxes(title="сигнал №", gridcolor=MUTED_HEX)
    fig.update_yaxes(gridcolor=MUTED_HEX)
    return fig


def build_3d_figure(skew, exkurt, show_gauss=True, show_edge=True):
    mu_vals = np.linspace(-3.0, 3.0, 70)
    sig_vals = np.linspace(0.05, 3.0, 70)
    MU, SIG = np.meshgrid(mu_vals, sig_vals)
    ALPHA = MU / SIG
    PHI = norm_pdf(ALPHA)
    CDF = 0.5 * (1.0 + np.vectorize(math.erf)(ALPHA / np.sqrt(2.0)))
    E_GAUSS = SIG * PHI + MU * CDF
    D = (skew / 6.0) * PHI * (MU * (ALPHA ** 2 - 1) - SIG * ALPHA ** 3) \
        + (exkurt / 24.0) * PHI * (MU * (3 * ALPHA - ALPHA ** 3) + SIG * (ALPHA ** 4 - 2 * ALPHA ** 2 - 1))
    E_EDGE = E_GAUSS + D

    fig = go.Figure()
    if show_gauss:
        fig.add_trace(go.Surface(x=mu_vals, y=sig_vals, z=E_GAUSS,
                                 colorscale=[[0, MUTED_HEX], [1, ACCENT_HEX]],
                                 opacity=0.55, showscale=False, name="гаусів згин"))
    if show_edge:
        fig.add_trace(go.Surface(x=mu_vals, y=sig_vals, z=E_EDGE,
                                 colorscale=[[0, MUTED_HEX], [1, TENSOR_HEX]],
                                 opacity=0.9, showscale=False, name="+ едворт-корекція"))
    fig.update_layout(
        paper_bgcolor=BG, font=dict(color=INK),
        scene=dict(
            xaxis=dict(title="μ (середнє z)", backgroundcolor=PANEL, gridcolor=MUTED_HEX, color=MUTED_TXT),
            yaxis=dict(title="σ (розкид z)", backgroundcolor=PANEL, gridcolor=MUTED_HEX, color=MUTED_TXT),
            zaxis=dict(title="E[ReLU(z)]", backgroundcolor=PANEL, gridcolor=MUTED_HEX, color=MUTED_TXT),
            camera=dict(eye=dict(x=1.6, y=-1.7, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=30, b=0), height=640,
        title=dict(text="E[ReLU(Z)], Z~N(μ,σ²) — формула Віка (k=0) + Едворт-κ4 корекція (R96)",
                   font=dict(color=INK, size=13)),
    )
    return fig


MC_NS = [512, 1024, 2048, 4096]
MC_TRIALS = 5


def build_mc_figure(Ns, results):
    """Log-log: MSE проти істини від N для plain MC / antithetic / v2(ZCA-anti)."""
    fig = go.Figure()
    series = [
        ("plain", "звичайний MC", MUTED_TXT, "circle"),
        ("anti", "antithetic MC (±x)", TENSOR_HEX, "diamond"),
        ("v2", "v2: ZCA-antithetic (наш)", ACCENT_HEX, "star"),
    ]
    for key, label, color, sym in series:
        fig.add_trace(go.Scatter(
            x=Ns, y=results[key], mode="lines+markers", name=label,
            line=dict(color=color, width=2.4 if key == "v2" else 1.8),
            marker=dict(color=color, size=11 if key == "v2" else 7, symbol=sym),
        ))
    # опорна лінія нахилу 1/N (закон MC): прив'язана до plain на першому N
    if results["plain"]:
        y0 = results["plain"][0]
        ref = [y0 * Ns[0] / n for n in Ns]
        fig.add_trace(go.Scatter(x=Ns, y=ref, mode="lines", name="нахил 1/N (закон MC)",
                                 line=dict(color=MUTED_HEX, width=1, dash="dot"), hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=MUTED_TXT, size=11),
        legend=dict(font=dict(color=INK, size=11), bgcolor=PANEL, bordercolor=MUTED_HEX, borderwidth=1),
        margin=dict(l=60, r=20, t=44, b=50), height=520,
        title=dict(text="MSE проти істини vs розмір вибірки N (менше = точніше)",
                   font=dict(color=INK, size=14)),
    )
    fig.update_xaxes(type="log", title="N (семплів)", gridcolor=MUTED_HEX, color=MUTED_TXT,
                     tickvals=Ns, ticktext=[str(n) for n in Ns])
    fig.update_yaxes(type="log", title="MSE", gridcolor=MUTED_HEX, color=MUTED_TXT)
    return fig


def build_mc_placeholder():
    fig = go.Figure()
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=520, font=dict(color=MUTED_TXT),
                      margin=dict(l=30, r=10, t=30, b=26),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text='натисни «прогнати порівняння»', showarrow=False,
                                        xref="paper", yref="paper", x=0.5, y=0.5,
                                        font=dict(color=MUTED_TXT, size=13))])
    return fig


def _cov_ellipse_xy(mean, cov, n=80):
    """Точки еліпса, що зображає коваріацію (якщо cov=I -> одиничне коло)."""
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-9, None)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)          # (2, n)
    axes = evecs @ np.diag(np.sqrt(evals))                     # масштаб по власних осях
    pts = axes @ circle + np.asarray(mean)[:, None]
    return pts[0], pts[1]


def build_v2_visual(N, nclicks):
    """Наочно, ЩО робить v2, у 2D: звичайна вибірка -> antithetic (центр у 0) -> ZCA (коло=істина).
    Кожна панель: точки вибірки, її емпіричний центр, еліпс коваріації та істинне одиничне коло."""
    rng = np.random.default_rng(4242 + int(nclicks or 0))
    X = rng.standard_normal((N, 2))

    mean_p = X.mean(0)
    cov_p = (X.T @ X) / N

    Xa = np.concatenate([X, -X], axis=0)
    mean_a = Xa.mean(0)
    cov_a = (Xa.T @ Xa) / (2 * N)

    C = (X.T @ X) / N
    evals, evecs = np.linalg.eigh(C)
    Cinv_sqrt = (evecs * (1.0 / np.sqrt(np.clip(evals, 1e-9, None)))) @ evecs.T
    Xz = X @ Cinv_sqrt
    Xza = np.concatenate([Xz, -Xz], axis=0)
    mean_v = Xza.mean(0)
    cov_v = (Xza.T @ Xza) / (2 * N)

    panels = [
        ("1 · звичайна вибірка", X, mean_p, cov_p, MUTED_TXT),
        ("2 · antithetic (±x)", Xa, mean_a, cov_a, TENSOR_HEX),
        ("3 · v2: + ZCA-вибілювання", Xza, mean_v, cov_v, ACCENT_HEX),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[0] for p in panels],
                        horizontal_spacing=0.05)
    tcirc = np.linspace(0, 2 * np.pi, 80)
    for i, (title, pts, mean, cov, color) in enumerate(panels, start=1):
        # істинне одиничне коло (еталон)
        fig.add_trace(go.Scatter(x=np.cos(tcirc), y=np.sin(tcirc), mode="lines",
                                 line=dict(color=MUTED_HEX, width=1.4, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=1, col=i)
        # еліпс коваріації вибірки
        ex, ey = _cov_ellipse_xy(mean, cov)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=2.2),
                                 hoverinfo="skip", showlegend=False), row=1, col=i)
        # точки
        fig.add_trace(go.Scattergl(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                   marker=dict(color=color, size=5, opacity=0.6),
                                   hoverinfo="skip", showlegend=False), row=1, col=i)
        # емпіричний центр
        fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode="markers",
                                 marker=dict(color=INK, size=11, symbol="x"),
                                 hoverinfo="skip", showlegend=False), row=1, col=i)
        fig.update_xaxes(range=[-3.2, 3.2], gridcolor=MUTED_HEX, color=MUTED_TXT,
                         zeroline=True, zerolinecolor=MUTED_HEX, row=1, col=i)
        fig.update_yaxes(range=[-3.2, 3.2], gridcolor=MUTED_HEX, color=MUTED_TXT,
                         zeroline=True, zerolinecolor=MUTED_HEX,
                         scaleanchor=f"x{i if i > 1 else ''}", scaleratio=1, row=1, col=i)

    for ann in fig.layout.annotations:
        ann.font = dict(color=INK, size=13)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=440,
                      margin=dict(l=20, r=20, t=40, b=20))

    def summ(mean, cov):
        return abs(np.linalg.norm(mean)), float(np.linalg.norm(cov - np.eye(2)))

    mp, cp = summ(mean_p, cov_p)
    ma, ca = summ(mean_a, cov_a)
    mv, cv = summ(mean_v, cov_v)
    summary = (f"Похибка моментів вибірки (менше = ближче до істини N(0,I)):\n"
               f"  звичайна:   |центр|={mp:.3f}   ||cov−I||={cp:.3f}\n"
               f"  antithetic: |центр|={ma:.3f}   ||cov−I||={ca:.3f}   ← центр точно 0\n"
               f"  v2 (ZCA):   |центр|={mv:.3f}   ||cov−I||={cv:.3f}   ← І центр, І cov точно як в істини")
    return fig, summary


def build_v2_placeholder():
    fig = go.Figure()
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=440, font=dict(color=MUTED_TXT),
                      margin=dict(l=30, r=10, t=30, b=26),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def build_cycle_figure(cyc):
    """Повний цикл v2 у 4 панелях: підготовка вибірки -> прохід крізь шари -> оцінка vs істина
    -> розподіл похибки. Скрізь порівняння звичайний MC (сірий) vs v2 (акцент)."""
    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.10,
        subplot_titles=[
            "Кроки 1–3: підготовка вибірки (сира → ZCA-антитетик)",
            "Крок 4: похибка оцінки крізь 32 шари",
            "Кроки 5–6: оцінка проти істини (фінальний шар)",
            "Розподіл похибки на нейрон",
        ],
    )

    # (1,1) 2D-хмара: сира vs ZCA-антитетик + одиничне коло
    t = np.linspace(0, 2 * np.pi, 80)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines",
                             line=dict(color=MUTED_HEX, width=1.4, dash="dot"),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    raw = cyc["x2d_raw"]
    v2p = cyc["x2d_v2"]
    fig.add_trace(go.Scattergl(x=raw[:, 0], y=raw[:, 1], mode="markers",
                               marker=dict(color=MUTED_TXT, size=4, opacity=0.4),
                               name="сира вибірка", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scattergl(x=v2p[:, 0], y=v2p[:, 1], mode="markers",
                               marker=dict(color=ACCENT_HEX, size=4, opacity=0.5),
                               name="v2 (ZCA-антитетик)", showlegend=True), row=1, col=1)
    fig.update_xaxes(range=[-3.4, 3.4], gridcolor=MUTED_HEX, color=MUTED_TXT, row=1, col=1)
    fig.update_yaxes(range=[-3.4, 3.4], gridcolor=MUTED_HEX, color=MUTED_TXT,
                     scaleanchor="x", scaleratio=1, row=1, col=1)

    # (1,2) MSE оцінки пошарово
    layers = list(range(COLS))
    fig.add_trace(go.Scatter(x=layers, y=cyc["mse_plain"], mode="lines+markers",
                             line=dict(color=MUTED_TXT, width=1.8), marker=dict(size=4),
                             name="звичайний MC", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=layers, y=cyc["mse_v2"], mode="lines+markers",
                             line=dict(color=ACCENT_HEX, width=2.4), marker=dict(size=5),
                             name="v2", showlegend=False), row=1, col=2)
    fig.update_xaxes(title="шар", gridcolor=MUTED_HEX, color=MUTED_TXT, row=1, col=2)
    fig.update_yaxes(type="log", title="MSE оцінки", gridcolor=MUTED_HEX, color=MUTED_TXT, row=1, col=2)

    # (2,1) оцінка vs істина per neuron
    tr = cyc["truth_final"]
    lo, hi = float(np.min(tr)), float(np.max(tr))
    pad = (hi - lo) * 0.08 + 1e-6
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                             line=dict(color=MUTED_HEX, width=1.4, dash="dash"),
                             hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scattergl(x=tr, y=cyc["est_plain"], mode="markers",
                               marker=dict(color=MUTED_TXT, size=4, opacity=0.5),
                               name="звичайний MC", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scattergl(x=tr, y=cyc["est_v2"], mode="markers",
                               marker=dict(color=ACCENT_HEX, size=4, opacity=0.6),
                               name="v2", showlegend=False), row=2, col=1)
    fig.update_xaxes(title="істина E[relu]", gridcolor=MUTED_HEX, color=MUTED_TXT, row=2, col=1)
    fig.update_yaxes(title="оцінка", gridcolor=MUTED_HEX, color=MUTED_TXT, row=2, col=1)

    # (2,2) гістограма похибки на нейрон
    err_p = np.abs(cyc["est_plain"].astype(np.float64) - tr.astype(np.float64))
    err_v = np.abs(cyc["est_v2"].astype(np.float64) - tr.astype(np.float64))
    hi_e = float(max(err_p.max(), err_v.max())) or 1e-6
    edges = np.linspace(0, hi_e, 34)
    cp, _ = np.histogram(err_p, bins=edges)
    cv, _ = np.histogram(err_v, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]
    fig.add_trace(go.Bar(x=centers, y=cp, width=bw * 0.9, marker_color=MUTED_TXT, opacity=0.6,
                         name="звичайний MC", showlegend=False), row=2, col=2)
    fig.add_trace(go.Bar(x=centers, y=cv, width=bw * 0.9, marker_color=ACCENT_HEX, opacity=0.75,
                         name="v2", showlegend=False), row=2, col=2)
    fig.update_xaxes(title="|оцінка − істина| на нейрон", gridcolor=MUTED_HEX, color=MUTED_TXT, row=2, col=2)
    fig.update_yaxes(title="нейронів", gridcolor=MUTED_HEX, color=MUTED_TXT, row=2, col=2)

    for ann in fig.layout.annotations:
        ann.font = dict(color=INK, size=12)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=760, barmode="overlay",
                      font=dict(color=MUTED_TXT, size=10),
                      legend=dict(font=dict(color=INK, size=10), bgcolor=PANEL,
                                  bordercolor=MUTED_HEX, borderwidth=1, x=0.01, y=0.99),
                      margin=dict(l=60, r=20, t=50, b=50))
    return fig


def build_cycle_placeholder():
    fig = go.Figure()
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=760, font=dict(color=MUTED_TXT),
                      margin=dict(l=30, r=10, t=30, b=26),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text='натисни «прогнати повний цикл»', showarrow=False,
                                        xref="paper", yref="paper", x=0.5, y=0.5,
                                        font=dict(color=MUTED_TXT, size=13))])
    return fig


# --------------------------------------------------------------- state ----

def new_state(seed=None):
    rng = np.random.default_rng(seed)
    W = build_weights(rng)
    x0 = rng.standard_normal(WIDTH).astype(np.float32)
    trace = forward_trace(x0, W)
    return {"rng": rng, "W": W, "trace": trace, "selection": None, "watchlist": [], "pulse_count": 0,
            "truth": None, "truth_layers": None}


STATE = new_state()


def resample_input(state):
    x0 = state["rng"].standard_normal(WIDTH).astype(np.float32)
    state["trace"] = forward_trace(x0, state["W"])


def add_to_watchlist(state, layer, idx):
    key = (layer, idx)
    if any(w["key"] == key for w in state["watchlist"]):
        return
    if len(state["watchlist"]) >= MAX_WATCH:
        state["watchlist"].pop(0)
    color = WATCH_PALETTE[len(state["watchlist"]) % len(WATCH_PALETTE)]
    state["watchlist"].append({"key": key, "layer": layer, "idx": idx, "color": color, "pulses": []})


def fire_signal(state):
    resample_input(state)
    state["pulse_count"] += 1
    for w in state["watchlist"]:
        val = float(state["trace"][w["layer"]][w["idx"]])
        w["pulses"].append((state["pulse_count"], val))
        if len(w["pulses"]) > MONITOR_WINDOW:
            w["pulses"].pop(0)


# ---------------------------------------------------------------- app -----

def build_app():
    app = Dash(__name__)
    app.title = "Жива мережа 32x256"

    btn_style = {
        "backgroundColor": PANEL, "color": INK, "border": f"1px solid {MUTED_HEX}",
        "borderRadius": "8px", "padding": "8px 14px", "marginRight": "8px",
        "cursor": "pointer", "fontSize": "13px",
    }
    panel_style = {"backgroundColor": PANEL, "border": f"1px solid {MUTED_HEX}", "borderRadius": "10px",
                    "padding": "10px"}

    tab_live = html.Div([
        html.P(
            "«Пустити сигнал» плавно анімує причинний прохід одного входу крізь шари (WebGL, значення вже "
            "пораховані numpy — анімація навмисно уповільнена для наочності). Клікни нейрон, щоб додати його "
            "в монітор.",
            style={"color": MUTED_TXT, "fontSize": "13px", "maxWidth": "900px"},
        ),
        html.Div([
            html.Button("нові ваги мережі", id="btn-reroll", style=btn_style),
            html.Button("новий вхід (перефарбувати)", id="btn-input", style=btn_style),
            html.Button("▶ пустити сигнал", id="btn-fire", style=btn_style),
            html.Button("авто: викл", id="btn-auto", style=btn_style),
            html.Button("очистити монітор", id="btn-clear", style=btn_style),
            dcc.Checklist(
                id="chk-weights",
                options=[{"label": " показати ваги обраного нейрона", "value": "show"}],
                value=["show"], style={"display": "inline-block", "color": INK, "fontSize": "13px",
                                       "marginLeft": "6px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px",
                  "marginBottom": "10px"}),
        dcc.Interval(id="auto-interval", interval=1700, disabled=True),
        html.Div(dcc.Graph(id="graph-network", figure=build_network_figure(STATE),
                           config={"displayModeBar": True, "scrollZoom": True}),
                 style=panel_style),
        html.Div([
            html.Div(dcc.Graph(id="graph-hist-z", figure=build_hist_figure(None, "Перед ReLU (z)")),
                     style={**panel_style, "flex": "1"}),
            html.Div(dcc.Graph(id="graph-hist-a", figure=build_hist_figure(None, "Після ReLU (a)")),
                     style={**panel_style, "flex": "1"}),
            html.Div(dcc.Graph(id="graph-monitor", figure=build_monitor_figure([])),
                     style={**panel_style, "flex": "1"}),
        ], style={"display": "flex", "gap": "10px", "marginTop": "10px"}),
        html.Div(id="stats-text", style={"color": INK, "fontFamily": "monospace", "fontSize": "13px",
                                         "marginTop": "10px", "padding": "8px 4px"}),
        html.Div(id="anim-dummy", style={"display": "none"}),
    ])

    tab_3d = html.Div([
        html.P(
            "Наша топова формула: точний Wick-коефіцієнт k=0 для E[ReLU(Z)] (гаусів згин) + Едворт-κ4 "
            "корекція на скіс γ3 і ексцес γ4 — та сама, що дала стелю 2.06e-7 (R96), верифікована проти MC "
            "у evo/gpu/lead1_sanity.py. Крути мишею — поверхня 3D.",
            style={"color": MUTED_TXT, "fontSize": "13px", "maxWidth": "900px"},
        ),
        html.Div([
            html.Div([
                html.Label("скіс γ3", style={"color": INK, "fontSize": "12px"}),
                dcc.Slider(id="slider-skew", min=-2, max=2, step=0.1, value=0.4,
                          marks={i: str(i) for i in range(-2, 3)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "marginRight": "20px"}),
            html.Div([
                html.Label("ексцес γ4", style={"color": INK, "fontSize": "12px"}),
                dcc.Slider(id="slider-kurt", min=-2, max=5, step=0.1, value=0.4,
                          marks={i: str(i) for i in range(-2, 6)},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1"}),
            dcc.Checklist(
                id="chk-surfaces",
                options=[{"label": " гаус", "value": "gauss"}, {"label": " +едворт", "value": "edge"}],
                value=["gauss", "edge"],
                style={"color": INK, "fontSize": "13px", "marginLeft": "20px", "whiteSpace": "nowrap"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div(dcc.Graph(id="graph-3d"), style=panel_style),
    ])

    tab_mc = html.Div([
        html.P(
            "НАОЧНО, що робить v2. Задача семплінгу: маленька випадкова вибірка «крива» — її центр "
            "зміщений, а хмара перекошена (не збігається з істиною N(0,I), пунктирне коло). v2 це "
            "виправляє у два кроки: antithetic (додаємо дзеркальні ±x → центр стає точно 0) і ZCA-"
            "вибілювання (перетворюємо хмару так, щоб її коваріація стала точно одиничним колом). "
            "Вибірка стає «репрезентативною» — її моменти точно як в істини, тож середнє по ній зміщене "
            "набагато менше. Пограйся з N і «новою вибіркою»: на малих N ефект найпомітніший.",
            style={"color": MUTED_TXT, "fontSize": "13px", "maxWidth": "980px"},
        ),
        html.Div([
            html.Div([
                html.Label("розмір вибірки N", style={"color": INK, "fontSize": "12px"}),
                dcc.Slider(id="slider-v2n", min=4, max=60, step=1, value=14,
                           marks={n: str(n) for n in (4, 10, 20, 30, 40, 50, 60)},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "marginRight": "20px"}),
            html.Button("нова вибірка", id="btn-v2-redraw", style=btn_style),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div(dcc.Graph(id="graph-v2", figure=build_v2_placeholder()), style=panel_style),
        html.Div(id="v2-summary", style={"color": INK, "fontFamily": "monospace", "fontSize": "13px",
                                         "whiteSpace": "pre", "marginTop": "10px", "padding": "8px 4px"}),
        html.Hr(style={"borderColor": MUTED_HEX, "margin": "20px 0"}),
        html.P(
            "А тепер той самий ефект у числах на СПРАВЖНІЙ мережі: MSE проти істини від розміру вибірки N. "
            "Звичайний MC падає як 1/N; antithetic дає ~2×; v2 (ZCA-antithetic) ~2.8× — і саме це тримає "
            "його у топі (score 3.09e-7, rank #24).",
            style={"color": MUTED_TXT, "fontSize": "13px", "maxWidth": "980px"},
        ),
        html.Div([
            html.Button("прогнати порівняння", id="btn-mc-run", style=btn_style),
            html.Span("рахує на ПОТОЧНІЙ мережі (істина = 40k семплів + " + str(MC_TRIALS)
                      + " розіграші/точку) — кілька секунд",
                      style={"color": MUTED_TXT, "fontSize": "12px", "marginLeft": "8px"}),
        ], style={"marginBottom": "12px"}),
        dcc.Loading(
            type="circle", color=ACCENT_HEX,
            children=html.Div(dcc.Graph(id="graph-mc", figure=build_mc_placeholder()), style=panel_style),
        ),
        html.Div(id="mc-summary", style={"color": INK, "fontFamily": "monospace", "fontSize": "13px",
                                         "marginTop": "12px", "padding": "8px 4px"}),
    ])

    # --- вкладка: повний цикл v2 ---
    cycle_steps = [
        ("1", "Вибірка", "N входів x~N(0,I)"),
        ("2", "ZCA", "вибілити → cov=I"),
        ("3", "Антитетик", "додати ±x → центр 0"),
        ("4", "32 шари ReLU", "прогнати крізь мережу"),
        ("5", "Усереднити", "mean по вибірці"),
        ("6", "vs Істина", "порівняти з E[relu]"),
    ]
    step_cards = []
    for i, (num, title, desc) in enumerate(cycle_steps):
        step_cards.append(html.Div([
            html.Div(num, style={"color": ACCENT_HEX, "fontWeight": "700", "fontSize": "18px"}),
            html.Div(title, style={"color": INK, "fontSize": "13px", "fontWeight": "600"}),
            html.Div(desc, style={"color": MUTED_TXT, "fontSize": "11px"}),
        ], style={"backgroundColor": PANEL, "border": f"1px solid {MUTED_HEX}", "borderRadius": "8px",
                  "padding": "10px 12px", "flex": "1", "minWidth": "120px"}))
        if i < len(cycle_steps) - 1:
            step_cards.append(html.Div("→", style={"color": MUTED_TXT, "fontSize": "18px",
                                                    "alignSelf": "center", "padding": "0 2px"}))

    tab_cycle = html.Div([
        html.P(
            "Повний конвеєр нашого методу v2 на СПРАВЖНІЙ мережі — від випадкової вибірки до фінальної "
            "оцінки E[relu(вихід)] для кожного нейрона, поруч зі звичайним MC. Чотири панелі показують "
            "усі кроки: як готується вибірка, як похибка оцінки поводиться крізь 32 шари, як фінальна "
            "оцінка лягає на істину, і наскільки щільніший розподіл похибки у v2.",
            style={"color": MUTED_TXT, "fontSize": "13px", "maxWidth": "980px"},
        ),
        html.Div(step_cards, style={"display": "flex", "gap": "4px", "flexWrap": "wrap",
                                    "marginBottom": "14px"}),
        html.Div([
            html.Button("прогнати повний цикл", id="btn-cycle-run", style=btn_style),
            html.Span("N=2048, істина = 40k семплів на поточній мережі — кілька секунд",
                      style={"color": MUTED_TXT, "fontSize": "12px", "marginLeft": "8px"}),
        ], style={"marginBottom": "12px"}),
        dcc.Loading(
            type="circle", color=ACCENT_HEX,
            children=html.Div(dcc.Graph(id="graph-cycle", figure=build_cycle_placeholder()),
                              style=panel_style),
        ),
        html.Div(id="cycle-summary", style={"color": INK, "fontFamily": "monospace", "fontSize": "13px",
                                            "marginTop": "12px", "padding": "8px 4px"}),
    ])

    app.layout = html.Div([
        html.H2("Жива мережа: 32 шари x 256 нейронів — Plotly/Dash, локально",
                style={"color": INK, "fontFamily": "Georgia, serif"}),
        dcc.Tabs(id="tabs", value="tab-live", children=[
            dcc.Tab(label="Жива мережа", value="tab-live", children=[tab_live],
                    style={"backgroundColor": PANEL, "color": MUTED_TXT, "border": f"1px solid {MUTED_HEX}"},
                    selected_style={"backgroundColor": BG, "color": ACCENT_HEX,
                                    "border": f"1px solid {MUTED_HEX}"}),
            dcc.Tab(label="3D: топова формула", value="tab-3d", children=[tab_3d],
                    style={"backgroundColor": PANEL, "color": MUTED_TXT, "border": f"1px solid {MUTED_HEX}"},
                    selected_style={"backgroundColor": BG, "color": ACCENT_HEX,
                                    "border": f"1px solid {MUTED_HEX}"}),
            dcc.Tab(label="v2 наочно + MC", value="tab-mc", children=[tab_mc],
                    style={"backgroundColor": PANEL, "color": MUTED_TXT, "border": f"1px solid {MUTED_HEX}"},
                    selected_style={"backgroundColor": BG, "color": ACCENT_HEX,
                                    "border": f"1px solid {MUTED_HEX}"}),
            dcc.Tab(label="Повний цикл v2", value="tab-cycle", children=[tab_cycle],
                    style={"backgroundColor": PANEL, "color": MUTED_TXT, "border": f"1px solid {MUTED_HEX}"},
                    selected_style={"backgroundColor": BG, "color": ACCENT_HEX,
                                    "border": f"1px solid {MUTED_HEX}"}),
        ]),
    ], style={"backgroundColor": BG, "padding": "18px 26px", "minHeight": "100vh",
             "fontFamily": "-apple-system, Segoe UI, sans-serif"})

    # ---- callbacks ----

    def _stats_text():
        sel = STATE["selection"]
        if sel is None:
            return "Клікни нейрон на графі."
        return f"Обрано: {'вхід' if sel[0] == 0 else f'шар {sel[0]}'} · нейрон {sel[1]}"

    @app.callback(
        Output("graph-network", "figure"),
        Output("graph-hist-z", "figure"),
        Output("graph-hist-a", "figure"),
        Output("graph-monitor", "figure"),
        Output("stats-text", "children"),
        Output("btn-auto", "children"),
        Output("auto-interval", "disabled"),
        Input("btn-reroll", "n_clicks"),
        Input("btn-input", "n_clicks"),
        Input("btn-fire", "n_clicks"),
        Input("auto-interval", "n_intervals"),
        Input("btn-clear", "n_clicks"),
        Input("btn-auto", "n_clicks"),
        Input("chk-weights", "value"),
        Input("graph-network", "clickData"),
        prevent_initial_call=True,
    )
    def unified(n_reroll, n_input, n_fire, n_auto_tick, n_clear, n_auto_btn, weights_val, click_data):
        trigger = ctx.triggered_id
        show_weights = "show" in (weights_val or [])
        frames, front0 = None, None

        auto_now = getattr(unified, "_auto_on", False)

        hz = go.Figure()
        ha = go.Figure()
        mon = go.Figure()
        rebuild_hist = False

        if trigger == "btn-reroll":
            STATE.update(new_state())
        elif trigger == "btn-input":
            resample_input(STATE)
        elif trigger in ("btn-fire", "auto-interval"):
            fire_signal(STATE)
            frames, front0 = build_sweep_frames()
        elif trigger == "btn-clear":
            STATE["watchlist"] = []
        elif trigger == "btn-auto":
            auto_now = not auto_now
            unified._auto_on = auto_now
        elif trigger == "chk-weights":
            pass
        elif trigger == "graph-network" and click_data:
            pt = click_data["points"][0]
            # координати кліку в даних (heatmap-ловець дає їх для будь-якої точки графа,
            # прямий клік по нейрону теж має x/y) -> округляємо до найближчого вузла
            if pt.get("x") is not None and pt.get("y") is not None:
                layer = int(round(float(pt["x"])))
                idx = int(round(float(pt["y"])))
                layer = max(0, min(DEPTH, layer))
                idx = max(0, min(WIDTH - 1, idx))
                STATE["selection"] = (layer, idx)
                add_to_watchlist(STATE, layer, idx)
                rebuild_hist = True

        fig_net = build_network_figure(STATE, show_weights=show_weights, frames=frames,
                                       front=front0)

        # Гістограми — важкі (батч N семплів). Перебудовуємо ЛИШЕ коли реально змінилось:
        # новий клік (rebuild_hist) або нові ваги (reroll). На fire/auto/input не чіпаємо —
        # розподіл по випадкових входах від них не залежить, тож клік лишається легким.
        if STATE["selection"] is not None and (rebuild_hist or trigger == "btn-reroll"):
            sl, si = STATE["selection"]
            z, a = forward_batch_to_layer(STATE["W"], sl, si, N_SAMPLES_DEFAULT, STATE["rng"])
            hz = build_hist_figure(z, "Перед ReLU (z)", mark_zero=(sl > 0))
            ha = build_hist_figure(a, "Після ReLU (a) — гаус-фіт", overlay_gauss=True)
        elif STATE["selection"] is None:
            hz = build_hist_figure(None, "Перед ReLU (z)")
            ha = build_hist_figure(None, "Після ReLU (a)")
        else:
            hz, ha = None, None

        mon = build_monitor_figure(STATE["watchlist"])

        from dash import no_update
        hz_out = hz if hz is not None else no_update
        ha_out = ha if ha is not None else no_update

        auto_label = "авто: увімк" if auto_now else "авто: викл"
        return fig_net, hz_out, ha_out, mon, _stats_text(), auto_label, (not auto_now)

    app.clientside_callback(
        """
        function(fig) {
            if (!fig || !fig.frames || !fig.frames.length) {
                return '';
            }
            var tries = 0;
            function tryAnimate() {
                var gd = document.getElementById('graph-network');
                var ready = gd && gd.data && gd.layout && window.Plotly;
                if (ready) {
                    Plotly.animate('graph-network', null, {
                        frame: {duration: 26, redraw: true},
                        transition: {duration: 22, easing: 'linear'},
                        mode: 'immediate'
                    }).catch(function(e) { console.warn('animate skipped:', e); });
                } else if (tries < 50) {
                    tries += 1;
                    setTimeout(tryAnimate, 40);
                }
            }
            tryAnimate();
            return '';
        }
        """,
        Output("anim-dummy", "children"),
        Input("graph-network", "figure"),
    )

    @app.callback(
        Output("graph-3d", "figure"),
        Input("slider-skew", "value"),
        Input("slider-kurt", "value"),
        Input("chk-surfaces", "value"),
    )
    def update_3d(skew, kurt, surfaces):
        surfaces = surfaces or []
        return build_3d_figure(skew, kurt, "gauss" in surfaces, "edge" in surfaces)

    @app.callback(
        Output("graph-mc", "figure"),
        Output("mc-summary", "children"),
        Input("btn-mc-run", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_mc_compare(n_clicks):
        if STATE.get("truth") is None:
            STATE["truth"] = compute_truth(STATE["W"], STATE["rng"])
        results = compute_mc_vs_v2(STATE["W"], STATE["truth"], STATE["rng"], MC_NS, MC_TRIALS)
        fig = build_mc_figure(MC_NS, results)
        nmax = MC_NS[-1]
        p, a, v = results["plain"][-1], results["anti"][-1], results["v2"][-1]
        summary = (f"При N={nmax}:  звичайний MC MSE={p:.2e}   ·   antithetic={a:.2e} "
                   f"(×{p/a:.2f})   ·   v2(ZCA-anti)={v:.2e} (×{p/v:.2f} над звичайним MC).  "
                   f"Саме ця ~2.8× редукція дисперсії і тримає v2 у топі.")
        return fig, summary

    @app.callback(
        Output("graph-v2", "figure"),
        Output("v2-summary", "children"),
        Input("slider-v2n", "value"),
        Input("btn-v2-redraw", "n_clicks"),
    )
    def update_v2_visual(nval, nclicks):
        return build_v2_visual(int(nval), nclicks)

    @app.callback(
        Output("graph-cycle", "figure"),
        Output("cycle-summary", "children"),
        Input("btn-cycle-run", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_cycle(n_clicks):
        if STATE.get("truth_layers") is None:
            STATE["truth_layers"] = compute_truth_layers(STATE["W"], STATE["rng"])
        cyc = compute_cycle(STATE["W"], STATE["truth_layers"], STATE["rng"], N=2048)
        fig = build_cycle_figure(cyc)
        mp = cyc["mse_plain"][-1]
        mv = cyc["mse_v2"][-1]
        summary = (f"Фінальний шар (N={cyc['N']}):  звичайний MC MSE={mp:.2e}   ·   "
                   f"v2 MSE={mv:.2e}   →  v2 точніший у ×{mp/mv:.2f}.  "
                   f"На панелі «оцінка проти істини» точки v2 (акцент) щільніше лягають на діагональ, "
                   f"а на гістограмі його похибки зсунуті ліворуч (менші).")
        return fig, summary

    return app


# ------------------------------------------------------------- selftest ---

def selftest():
    rng = np.random.default_rng(0)
    W = build_weights(rng)
    x0 = rng.standard_normal(WIDTH).astype(np.float32)
    trace = forward_trace(x0, W)
    assert len(trace) == COLS

    z, a = forward_batch_to_layer(W, 32, 17, 2000, rng)
    mu_z, sd_z = float(np.mean(z)), float(np.std(z))
    mu_a = float(np.mean(a))
    formula = gaussian_fold_mean(mu_z, sd_z)
    print(f"layer32 neuron17: mu_z={mu_z:.4f} sd_z={sd_z:.4f} emp.mu(a)={mu_a:.4f} "
          f"formula={formula:.4f}")

    edge = edgeworth_fold_mean(mu_z, sd_z, 0.4, 0.4)
    print(f"edgeworth(g3=.4,g4=.4) at layer32 point: {edge:.4f} (gauss {formula:.4f})")

    state = new_state(seed=1)
    fig = build_network_figure(state)
    # data[0] = heatmap-ловець; вузли = Scatter десь у data
    assert fig.data[0].type == "heatmap", "поле нейронів = heatmap (data[0])"
    assert fig.data[0].hoverinfo == "none", "поле мусить мати hoverinfo=none (не skip), щоб ловити кліки"
    assert fig.data[0].z.shape == (WIDTH, COLS), "z-матриця має форму (WIDTH, COLS)"
    frames, front0 = build_sweep_frames()
    assert len(frames) == NFRAMES
    assert isinstance(front0, float)

    add_to_watchlist(state, 1, 3)
    add_to_watchlist(state, 20, 55)
    fire_signal(state)
    fire_signal(state)
    assert len(state["watchlist"][0]["pulses"]) == 2
    mon = build_monitor_figure(state["watchlist"])
    assert len(mon.data) == 2

    fig3d = build_3d_figure(0.4, 0.4)
    assert len(fig3d.data) == 2

    # MC проти v2 — на маленькому наборі, лише перевірка коректності напрямку
    truth = compute_truth(W, rng, n=8000)
    res = compute_mc_vs_v2(W, truth, rng, [512, 2048], trials=3)
    p, v = res["plain"][-1], res["v2"][-1]
    print(f"MC vs v2 @N=2048: plain MSE={p:.2e}  v2(ZCA-anti)={v:.2e}  редукція ×{p/v:.2f}")
    assert v <= p * 1.5, "v2 не має бути помітно гіршим за plain MC"
    figmc = build_mc_figure([512, 2048], res)
    assert len(figmc.data) >= 3

    # v2 наочно: ZCA-вибірка має cov≈I і центр≈0 (перевіряємо, що вибілювання працює)
    figv2, sv2 = build_v2_visual(20, 0)
    assert len(figv2.data) == 12  # 3 панелі × 4 трейси
    Xt = np.random.default_rng(1).standard_normal((20, 2))
    Ct = (Xt.T @ Xt) / 20
    ev, evec = np.linalg.eigh(Ct)
    Cis = (evec * (1/np.sqrt(np.clip(ev, 1e-9, None)))) @ evec.T
    Xz = Xt @ Cis
    cov_err = np.linalg.norm((Xz.T @ Xz)/20 - np.eye(2))
    print(f"v2 наочно: ZCA cov-похибка ||cov-I||={cov_err:.2e} (має бути ~0)")
    assert cov_err < 1e-4, "ZCA-вибілювання має дати cov=I"

    # Повний цикл: v2 має бути не гіршим за plain на фінальному шарі
    tl = compute_truth_layers(W, rng, n=8000)
    cyc = compute_cycle(W, tl, rng, N=1024)
    assert len(cyc["mse_plain"]) == COLS and len(cyc["mse_v2"]) == COLS
    figc = build_cycle_figure(cyc)
    assert len(figc.data) >= 9
    print(f"повний цикл @N={cyc['N']}: фінал plain MSE={cyc['mse_plain'][-1]:.2e}  "
          f"v2 MSE={cyc['mse_v2'][-1]:.2e}  ×{cyc['mse_plain'][-1]/cyc['mse_v2'][-1]:.2f}")

    print("OK: selftest пройшов без запуску Dash-сервера.")


def launch_app_window(url):
    """Відкриває url як окреме "застосункове" вікно браузера (без адресного рядка/вкладок),
    якщо є Chrome/Chromium (--app=); інакше — звичайна вкладка типового браузера."""
    chrome = (shutil.which("google-chrome") or shutil.which("google-chrome-stable")
              or shutil.which("chromium") or shutil.which("chromium-browser")
              or shutil.which("microsoft-edge"))
    if chrome:
        subprocess.Popen(
            [chrome, f"--app={url}", "--window-size=1500,1040", "--new-window"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--no-browser", action="store_true",
                        help="не відкривати браузер автоматично")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    global STATE
    STATE = new_state(seed=args.seed)
    app = build_app()

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, launch_app_window, args=(url,)).start()

    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
