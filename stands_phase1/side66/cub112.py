"""cub112 — ДЕТЕРМІНОВАНА КУБАТУРА ПРОТИ ВИБІЛЕНОЇ МК ЗА ОДНАКОВОЇ КІЛЬКОСТІ ТОЧОК.

════════ ЗВІДКИ ════════
Борд: #1 `ednacob` бал 1.84e-08 при MSE 3.63e-08 ⟹ множник **0.507**, а не
підлога 0.100. Тобто він витрачає ×5.07 бюджету. Якщо ті самі 0.507·B віддати
в МК, `V ∝ 1/N` дає MSE 3.65e-07 ⟹ бал 1.85e-07, тобто РІВНО наш.
⟹ **за однакового компуту його правило точніше у ×10.1 по MSE = ×3.18 по похибці.**
Реконструкція бюджету: 129 базисів × 256 = 33 024 точки × наша ціна 4.194e6
= 1.385e11 ⟹ C/B = 0.509 проти зміряних 0.507 (збіг 0.5%).

════════ ЧОМУ СТАРІ ВИРОКИ СЮДИ НЕ ПЕРЕНОСЯТЬСЯ ════════
«сферизація ×0.74» · «RQMC ×1.0156» · «4-ті моменти ×1.00» — усі троє міряли
ЗМЕНШЕННЯ ДИСПЕРСІЇ ВИПАДКОВИХ точок проти вибілювання при N=6144. Тут немає
дисперсії: набір детермінований, похибка є похибкою ПРАВИЛА.
🔑 Об'єднання ортонормованих базисів — це 2-дизайн, тобто рівно те, що вже
робить наше вибілювання (точні другі моменти). Kerdock/MUB дає СИЛЬНІШУ
рівномірність: |<v,w>| = 1/16 ТОЧНО між базисами замість флуктуацій ~N(0,1/256).

════════ ЩО МІРЯЄМО (усе за ОДНАКОВОЇ кількості точок N) ════════
  MCW    вибілена гаусова МК + антитетика  — база сабміту
  FRAME  об'єднання ВИПАДКОВИХ ортонормованих базисів, ±, × E‖z‖
  HAD    структуровані базиси H·D1·H·D2·H·D3, ±, × E‖z‖
  MUB    Kerdock: B_A = D_A·H, q_A(y)=Σ_{i<j}A_ij y_i y_j, A_β з GF(2^8), ±
Істина — `final_means` датасету (N=1e9 ⟹ відн. похибка 0.00074%, у ~200×
точніша за все, що ми міряємо).

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (записую, щоб не тлумачити заднім числом) ════════
 (а) 🔒FRAME ≈ MCW (×1.00–1.10) — бо 2-дизайн = вибілювання           75%
 (б) 🔒HAD ≈ FRAME — структура Адамара не додає порядку дизайну        70%
 (в) MUB > FRAME щонайменше у ×1.4 при K=64                           40%
 (г) перевага MUB РОСТЕ з K (при K=12 менша, ніж при K=64)            65%
 (д) 🔒MCW при 4×N падає РІВНО у 2.0 (пломба «шум, не зсув»)           90%
🎯 ЯКЩО (в) ПІДТВЕРДИТЬСЯ бодай на ×2 ⟹ MSE ×4 ⟹ бал ≈4.6e-08 = топ-3.
🎯 ЯКЩО (в) ПРОВАЛИТЬСЯ — це НЕ стіна, а адреса: значить виграш не в порядку
   дизайну, а в МЕРЕЖЕЗАЛЕЖНОМУ виборі фреймів (відкриті питання форуму O6/O8),
   і наступний стенд міряє саме поворот набору під конкретну W.
"""
import glob
import math
import os
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
E_NORM = 15.984382667          # E‖z‖ для n=256, точне число (С100, пломба 7.45e-15)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 50
REPL = int(sys.argv[2]) if len(sys.argv) > 2 else 8
KS = [6, 12, 17]                  # K базисів ⟹ N = 512·K точок
CKPT = "/home/trim/wblogs/cub112.npz"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("НЕ ЗАПІНОВАНО НА ОДНУ КАРТУ — вихід", flush=True)
    sys.exit(2)

# ─────────────────────── GF(2^8) і набір Кердока ───────────────────────
POLY = 0x11B


def gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= POLY
    return r


def gf_tr(a):
    """слід GF(2^8)→GF(2): Tr(a) = Σ_k a^(2^k)"""
    s, t = 0, a
    for _ in range(8):
        s ^= t
        t = gf_mul(t, t)
    return s


def alt_form(beta):
    """A_β[i,j] = Tr(β·(e_i·e_j² + e_i²·e_j)) — знакозмінна (A[i,i]=0)"""
    A = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        ei = 1 << i
        for j in range(8):
            ej = 1 << j
            v = gf_mul(ei, gf_mul(ej, ej)) ^ gf_mul(gf_mul(ei, ei), ej)
            A[i, j] = gf_tr(gf_mul(beta, v)) & 1
    return A


def gf2_rank(A):
    M = A.copy().astype(np.uint8)
    n, r = M.shape[0], 0
    for c in range(n):
        p = None
        for i in range(r, n):
            if M[i, c]:
                p = i
                break
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(n):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def build_mub():
    """Базиси B_A = D_A·H для A з ЛІНІЙНОГО підпростору знакозмінних форм,
    усі ненульові елементи якого НЕВИРОДЖЕНІ (умова взаємної незміщеності:
    A_β + A_γ = A_{β⊕γ} мусить мати повний ранг). Плюс стандартний базис.
    🚨 Перша редакція перевіряла ранг КОЖНОЇ форми, а треба ранг РІЗНИЦІ —
    пломба S4 це впіймала (|<v,w>| виходило 0…0.125 замість рівно 1/16)."""
    good = set(b for b in range(1, 256) if gf2_rank(alt_form(b)) == 8)
    rnd = np.random.default_rng(12345)
    best = [0]
    for _ in range(600):
        S = [0]
        for b in rnd.permutation(np.arange(1, 256)):
            b = int(b)
            if b in S:
                continue
            if all(((b ^ t) in good) for t in S if (b ^ t) != 0):
                S = S + [b ^ t for t in S]
        if len(S) > len(best):
            best = S
    H = np.ones((1, 1), dtype=np.float32)
    for _ in range(8):
        H = np.block([[H, H], [H, -H]])
    H /= 16.0
    ys = np.arange(256)
    bits = ((ys[:, None] >> np.arange(8)[None, :]) & 1).astype(np.uint8)
    bases = [np.eye(256, dtype=np.float32)]          # стандартний базис
    for b in best:
        A = alt_form(b)
        q = np.zeros(256, dtype=np.uint8)
        for a1 in range(8):
            for a2 in range(a1 + 1, 8):
                if A[a1, a2]:
                    q ^= bits[:, a1] & bits[:, a2]
        D = np.where(q[:, None] == 1, -1.0, 1.0).astype(np.float32)
        bases.append(D * H)
    return np.stack(bases), len(bases)


KER, NGOOD = build_mub()
print(f"MUB: побудовано {NGOOD} взаємно незміщених базисів", flush=True)

# 🔒 ПЛОМБА S4 — взаємна незміщеність
_g = cp.asarray(KER)
_mn, _mx = 1.0, 0.0
for _a in range(_g.shape[0]):
    for _b in range(_a + 1, _g.shape[0]):
        _c = cp.abs(_g[_a].T @ _g[_b])
        _mn = min(_mn, float(_c.min()))
        _mx = max(_mx, float(_c.max()))
print(f"🔒S4 |<v,w>| ПО ВСІХ парах базисів: min {_mn:.6f} max {_mx:.6f}"
      f"   (мусить бути рівно 1/16 = 0.062500)", flush=True)
_ort = float(cp.abs(_g[1].T @ _g[1] - cp.eye(256, dtype=cp.float32)).max())
print(f"🔒S4b ортонормованість базису: {_ort:.2e}", flush=True)
del _g

# ─────────────────────── дані ───────────────────────
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
tt = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(tt.type):
    tt = tt.flatten()
TRUE = np.asarray(tt).reshape(tb.num_rows, W_).astype(np.float64)
NNET = min(NNET, tb.num_rows)
print(f"cub112 · {NNET} мереж · {REPL} реплік · K={KS} ⟹ N={[512*k for k in KS]}",
      flush=True)


def hadamard(n=256):
    H = np.ones((1, 1), dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return cp.asarray(H / math.sqrt(n))


HG = hadamard()


def fwd_mean(V, Wg, chunk=16384):
    """середнє net(V) по рядках; накопичення у fp64"""
    acc = cp.zeros(W_, dtype=cp.float64)
    n = V.shape[0]
    for s in range(0, n, chunk):
        h = V[s:s + chunk]
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
        acc += h.sum(0, dtype=cp.float64)
    return acc / n


def pts_mcw(K, rng):
    """вибілена гаусова МК + антитетика; N = 512K точок"""
    npair = 256 * K
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    cov = (z.T.astype(cp.float64) @ z.astype(cp.float64)) / npair
    wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T.astype(cp.float32)
    z = z @ wh
    return cp.concatenate([z, -z], axis=0)


def pts_frame(K, rng):
    """K випадкових ортонормованих базисів, ±, × E‖z‖"""
    out = []
    for _ in range(K):
        g = rng.standard_normal((W_, W_), dtype=cp.float32)
        q, _r = cp.linalg.qr(g)
        out.append(q.T)                    # рядки = ортонормовані вектори
    V = cp.concatenate(out, axis=0)
    return cp.concatenate([V, -V], axis=0)


def pts_had(K, rng):
    """K структурованих ортогональних базисів H·D1·H·D2·H·D3, ±, × E‖z‖"""
    out = []
    for _ in range(K):
        q = HG
        for _k in range(3):
            d = (rng.integers(0, 2, size=W_) * 2 - 1).astype(cp.float32)
            q = (q * d[None, :]) @ HG
        out.append(q)
    V = cp.concatenate(out, axis=0)
    return cp.concatenate([V, -V], axis=0)


def pts_mub(K, rng, kg):
    """K базисів Кердока, ВИПАДКОВО ПОВЕРНУТИХ як ціле, ±, × E‖z‖.
    Поворот зберігає взаємну незміщеність ⟹ репліки законні."""
    g = rng.standard_normal((W_, W_), dtype=cp.float32)
    R, _r = cp.linalg.qr(g)
    idx = np.random.default_rng(int(rng.integers(0, 2**31))).permutation(
        kg.shape[0])[:K]
    V = cp.concatenate([kg[int(i)].T for i in idx], axis=0)   # рядки = вектори
    V = V @ R.T
    return cp.concatenate([V, -V], axis=0)


METHODS = ["MCW", "FRAME", "HAD", "MUB"]
res = np.full((NNET, len(KS), len(METHODS), REPL), np.nan)
done = np.zeros(NNET, dtype=bool)
if os.path.exists(CKPT):
    z = np.load(CKPT)
    if z["res"].shape == res.shape:
        res, done = z["res"], z["done"]
        print(f"підхоплено {int(done.sum())} мереж", flush=True)

KERG = cp.asarray(KER)
fails = 0
t0 = time.time()
for ni in range(NNET):
    if done[ni]:
        continue
    try:
        Wg = [cp.asarray(Wall[ni, i]) for i in range(DEPTH)]
        tr = cp.asarray(TRUE[ni])
        for ki, K in enumerate(KS):
            for mi, m in enumerate(METHODS):
                for r in range(REPL):
                    rng = cp.random.default_rng(9000 + 131 * r + 7 * ki + 3 * mi)
                    if m == "MCW":
                        V = pts_mcw(K, rng)
                        est = fwd_mean(V, Wg)
                    else:
                        V = (pts_frame(K, rng) if m == "FRAME" else
                             pts_had(K, rng) if m == "HAD" else
                             pts_mub(K, rng, KERG))
                        est = fwd_mean(V, Wg) * E_NORM
                    res[ni, ki, mi, r] = float(cp.mean((est - tr) ** 2))
                    del V
        done[ni] = True
        fails = 0
        np.savez(CKPT + ".tmp", res=res, done=done)
        os.replace(CKPT + ".tmp.npz", CKPT)
        print(f"  мережа {ni} готова  ({time.time()-t0:.0f} с)", flush=True)
    except Exception as e:                                    # noqa: BLE001
        fails += 1
        print(f"  ⚠️ мережа {ni}: {e}", flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        if fails >= 3:
            print("три невдачі поспіль — чистий вихід", flush=True)
            break

# ─────────────────────── звіт ───────────────────────
ok = done.copy()
print(f"\n=== ГОТОВО {int(ok.sum())} мереж ===")
MEAN_T2 = float(np.mean(TRUE[:NNET][ok] ** 2))
print(f"🔒 mean(true²) = {MEAN_T2:.6f}   (записано 0.9097)")


def rel(mse):
    return math.sqrt(mse / MEAN_T2) * 100.0


for ki, K in enumerate(KS):
    print(f"\n─── K={K}  ⟹  N = {512*K} точок ───")
    print(f"{'метод':>8} {'MSE':>14} {'відн.похибка':>14} {'×проти MCW':>12} "
          f"{'знак vs MCW':>12}")
    base = np.nanmean(res[ok, ki, 0, :], axis=1)
    for mi, m in enumerate(METHODS):
        cur = np.nanmean(res[ok, ki, mi, :], axis=1)
        mse = float(np.mean(cur))
        gain = math.sqrt(float(np.mean(base)) / mse) if mse > 0 else float("nan")
        wins = int(np.sum(cur < base))
        print(f"{m:>8} {mse:14.6e} {rel(mse):13.4f}% {gain:12.4f} "
              f"{wins:>7}/{int(ok.sum())}")

if len(KS) >= 2:
    b0 = float(np.mean(np.nanmean(res[ok, 0, 0, :], axis=1)))
    b1 = float(np.mean(np.nanmean(res[ok, -1, 0, :], axis=1)))
    print(f"\n🔒S3 пломба масштабування MCW: N×{512*KS[1]/(512*KS[0]):.2f} ⟹ "
          f"MSE ×{b0/b1:.3f}  (мусить бути ×{KS[-1]/KS[0]:.2f}, якщо це ШУМ)")
print(f"\nвсього {time.time()-t0:.0f} с")
