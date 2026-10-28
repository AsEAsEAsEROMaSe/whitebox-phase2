"""domain99 — ДОМЕН ЗАМИКАННЯ: чи лишається `diag(Cp)` ДОДАТНОЮ?
Написано сесією A (`kink99`), віддано сесії B для прогону на 50 мережах n=256.

════════════ ЩО САМЕ ЛОВИТЬ ════════════
`C = M2 − outer(ma, ma)`, тобто `Var = E[X²] − E[X]²`. Коли ворота насичені
(`|t|` велике), нейрон майже детермінований ⟹ обидва доданки величезні й майже
рівні, а відносна точність квадратури (~1e-9) з'їдає ВСЮ дисперсію.
⟹ `diag(Cp)` стає ВІД'ЄМНОЮ, а `s2 = max(diag(Cp), 1e-300)` це **МАСКУЄ**,
і далі рекурсія рахує сміття мовчки.

`[ЗМІРЯНО kink99, n=32, зсув входу ‖m₀‖ = c·√n]` при c=16 diag(Cp) від'ємна вже
на шарі 2 (−5.5e2 … −3.5e3), при c=32 — на шарі 1 (−5.4e3 … −2.7e4), 5–13
нейронів на шар. Похибка виходу при цьому 6.4% і 439% відповідно.
🔒 КОНТРОЛЬ, ЯКИЙ ВІДКИНУВ ОЧЕВИДНОГО ПІДОЗРЮВАНОГО: переповнення в `φ` НЕ
винне — захищена арифметика (клип аргументу) дає ТІ САМІ числа, нескінченних
значень 0. Причина саме в скороченні, а не в overflow.

════════════ ЩО МІРЯТИ ════════════
Для кожної мережі і кожного шару: `min(diag(Cp))`, кількість обрізаних
`max(·,1e-300)`, медіана `|t|`. Робоча точка (медіана |t| = 3.50) має бути
глибоко всередині домену — але це ТРЕБА ЗМІРЯТИ, бо жодна пломба проєкту цього
не перевіряла: усі стояли на РЕЗУЛЬТАТІ, а не на конструкції.

ПЕРЕДБАЧЕННЯ (A, до прогону): при n=256 у робочій точці від'ємних не буде
на жодній із 50 мереж — 85%. Якщо будуть — це тиха хвороба в сабміті.
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def psi(x):
    x = np.asarray(x, float)
    return phi(x) + x * ndtr(x)


def gl(a, b, npan, nnod):
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    return ((0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel(),
            (0.5 * (hi - lo) * ws[None, :]).ravel())


ZQ, WQ = gl(0.0, 40.0, 12, 20)


def trace(W, n, chunk=8192):
    """повертає (min diag Cp по шарах, к-сть обрізаних, медіана |t| по шарах)."""
    m = np.zeros(n)
    C = np.eye(n)
    mins, cuts, tmed = [], [], []
    for L in range(DEPTH):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        raw = np.diag(Cp).copy()                 # ДО обрізання — це і є предмет
        mins.append(float(raw.min()))
        cuts.append(int((raw <= 1e-300).sum()))
        s2 = np.maximum(raw, 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        tmed.append(float(np.median(np.abs(t))))
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
        tj = np.repeat(t, n)
        tk = np.tile(t, n)
        rr = rho.ravel()
        out = np.empty(n * n)
        q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-14))
        for o in range(0, n * n, chunk):
            sl = slice(o, min(o + chunk, n * n))
            x = ZQ[None, :] - tj[sl, None]
            u = (rr[sl, None] * x + tk[sl, None]) / q[sl, None]
            out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                       * psi(u)).sum(1)
        M2 = out.reshape(n, n) * (s[:, None] * s[None, :])
        np.fill_diagonal(M2, s2 * ((1 + t * t) * ndtr(t) + t * phi(t)))
        ma = s * psi(t)
        Cn = M2 - np.outer(ma, ma)
        C = 0.5 * (Cn + Cn.T)
        m = ma
    return np.array(mins), np.array(cuts), np.array(tmed)


print(f"domain99 · домен замикання · {NNET} мереж n=256 · мінісплит", flush=True)
bad_nets = 0
worst = []
for fpath in sorted(glob.glob(SRC + "/data-00000-of-00002.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(fpath, "rb")).read_all()
    x = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, 256, 256)
    names = tb.column("mlp_name").to_pylist()
    for i in range(min(tb.num_rows, NNET)):
        W = [np.ascontiguousarray(Wall[i][L], dtype=np.float64)
             for L in range(DEPTH)]
        mins, cuts, tmed = trace(W, 256)
        neg = int((mins < 0).sum())
        bad_nets += (neg > 0)
        worst.append(float(mins.min() / max(abs(mins).max(), 1e-300)))
        flag = "  🚨 ВІД'ЄМНА" if neg else ""
        print(f"  {i:>2d} {names[i]:<22s} min diag = {mins.min():.3e}"
              f"  шарів<0: {neg:>2d}  обрізано: {cuts.sum():>4d}"
              f"  мед|t| {np.median(tmed):.2f}{flag}", flush=True)

print(f"\nмереж із від'ємною diag(Cp): {bad_nets} із {NNET}")
print("⟹ ВИРОК: домен " + ("ПОРУШЕНО в робочій точці — тиха хвороба в сабміті"
                           if bad_nets else "ЧИСТИЙ у робочій точці"))
