"""euler101b — ВИРІШАЛЬНИЙ ВИМІР ДЛЯ ТОТОЖНОСТІ ЕЙЛЕРА: ЩО ТОЧНІШЕ ПРИ ГАУСОВОМУ
ПРИПУЩЕННІ — `E[ReLU(p)]` (наш нинішній функціонал) чи `ρ_p(0)` (потрібний новому)?

КОНТЕКСТ. `E[out] = E[Δout] = Σ_{l,j} ρ_{p_l[j]}(0) · E[g | p_l[j]=0]`,
де `g = B_l[j,a]·‖J_l[:,j]‖²`. Замикання вміє давати `(m,s)` ⟹ вміє давати
`ρ(0) = φ(t)/s`. Отже увесь новий шлях стоїть на тому, наскільки гаусова
оцінка ЩІЛЬНОСТІ В НУЛІ точніша (чи гірша) за гаусову оцінку `E[ReLU]`.

🚨 ОБИДВА ФУНКЦІОНАЛИ МІРЯЮТЬСЯ З ОДНИХ І ТИХ САМИХ ІСТИННИХ `(m,s)` ⟹
   ізолюється РІВНО негаусовість, а не похибка замикання. Це прямий аналог
   записаного «похибка ОДНОГО кроку з ТОЧНИМИ (μ,σ,ρ): E[relu] 0.139%».

ТРИ ВЕЛИЧИНИ НА КОЖЕН НЕЙРОН (l, j):
    E_relu_true  = mean(ReLU(p))                 з семплів
    E_relu_gauss = m·Φ(t) + s·φ(t)               гаус із ІСТИННИХ (m,s)
    ρ_true(0)    = щільність p у нулі            з семплів, ядром ε·s
    ρ_gauss(0)   = φ(t)/s                        гаус із ІСТИННИХ (m,s)

🔒 ПЛОМБА ШАРУ 0: там `p` гаусів ТОЧНО ⟹ обидві похибки мусять дати шум.
🔒 ПЛОМБА ЯДРА: `ρ_true` рахується на кількох `ε` ⟹ беремо плато, не одну точку.
🔒 ПЛОМБА МАСШТАБУВАННЯ: два `N` ⟹ якщо похибка `ρ` падає з `N`, це шум ядра.

ДРУГЕ ПИТАННЯ (факторизація): наскільки `E[g | p=0]` відрізняється від `E[g]`?
Якщо слабко — доданок факторизується на «щільність × середнє якобіанів», і
обидва множники мають власні рекурсії. Якщо сильно — потрібне спільне.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒 шар 0 дасть обидві похибки < 0.3% (шум ядра)                   80%
 (б) похибка `ρ(0)` на глибині БУДЕ БІЛЬШОЮ за похибку `E[ReLU]`       65%
     (мотив: `E[ReLU]` при `t≈3.5` майже не бачить форми — записано;
      а `ρ(0)` — суто локальна і формою живе)
 (в) якщо (б) справдиться, різниця буде ×2…×10                         50%
 (г) `E[g|p=0]` відрізнятиметься від `E[g]` більш ніж на 10%           60%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

W_ = 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
CHUNK = 16384
EPS = [0.02, 0.05, 0.10]
LAYERS = [0, 1, 2, 4, 8, 16, 24, 30, 31]

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не одну карту", flush=True)
    sys.exit(2)

f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, 32, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"euler101b · {NNET} мереж × {2*NS} семплів (антитетика) · "
      f"ε = {EPS}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def one_net(net, nsamp):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(32)]
    rng = cp.random.default_rng(9090 + net)
    L = len(LAYERS)
    S1 = cp.zeros((L, W_), dtype=cp.float64)
    S2 = cp.zeros((L, W_), dtype=cp.float64)
    RL = cp.zeros((L, W_), dtype=cp.float64)                 # Σ ReLU(p)
    # для щільності потрібні s — робимо два проходи
    done = 0
    while done < nsamp:
        nb = min(CHUNK, nsamp - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        k = 0
        for i in range(32):
            p = a @ Wg[i]
            if i in LAYERS:
                p64 = p.astype(cp.float64)
                S1[k] += p64.sum(0)
                S2[k] += (p64 * p64).sum(0)
                RL[k] += cp.maximum(p64, 0.0).sum(0)
                k += 1
            if i < 31:
                a = cp.maximum(p, 0.0)
        done += nb
    N = 2 * nsamp
    m = S1 / N
    s = cp.sqrt(cp.maximum(S2 / N - m * m, 1e-30))
    relu_true = RL / N
    t = m / s
    relu_g = m * ndtr(t) + s * phi(t)
    rho_g = phi(t) / s

    # другий прохід: щільність у нулі ядром ε·s
    rng = cp.random.default_rng(9090 + net)
    KA = {e: cp.zeros((L, W_), dtype=cp.float64) for e in EPS}
    done = 0
    while done < nsamp:
        nb = min(CHUNK, nsamp - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        k = 0
        for i in range(32):
            p = a @ Wg[i]
            if i in LAYERS:
                p64 = p.astype(cp.float64)
                for e in EPS:
                    w = e * s[k]
                    KA[e][k] += (cp.exp(-0.5 * (p64 / w) ** 2)
                                 / (SQ2PI * w)).sum(0)
                k += 1
            if i < 31:
                a = cp.maximum(p, 0.0)
        done += nb
    rho_true = {e: KA[e] / N for e in EPS}
    return relu_true, relu_g, rho_true, rho_g


def relerr(a, b):
    return float(cp.abs(a - b).mean() / cp.maximum(cp.abs(b).mean(), 1e-300))


for label, nsamp in (("N/4", NS // 4), ("N", NS)):
    acc = {L: [0.0, {e: 0.0 for e in EPS}] for L in LAYERS}
    for net in range(NNET):
        rt, rg, pt, pg = one_net(net, nsamp)
        for k, Ly in enumerate(LAYERS):
            acc[Ly][0] += relerr(rg[k], rt[k])
            for e in EPS:
                acc[Ly][1][e] += relerr(pg[k], pt[e][k])
    print(f"\n══ {label} = {2*nsamp} семплів, {NNET} мереж ══")
    print(f"{'шар':>4s}{'похибка E[ReLU]':>18s}" +
          "".join(f"{'ρ(0) ε='+str(e):>15s}" for e in EPS))
    for Ly in LAYERS:
        r = 100 * acc[Ly][0] / NNET
        cells = "".join(f"{100*acc[Ly][1][e]/NNET:>14.4f}%" for e in EPS)
        print(f"{Ly:>4d}{r:>17.4f}%{cells}", flush=True)
