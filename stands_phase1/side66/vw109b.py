"""vw109b — КУТ 6 (ранг ТІЄЇ частини ансамбля, що робить відповідь) + КУТ 8.

КУТ 6. `vw109` зміряв ранг УСЬОГО ансамбля через ескіз (≥343, не сатурує) і
показав, що суміш із `k` компонент повертає лише 9.7% при k=64. Але кластеризація
є ГРУБИМ, НЕЛІНІЙНИМ урізанням — вона могла провалитись через свою грубість.
Точна алгебра каже, що питання інше:

    E[out]_a = Σ_i Cov(z_i , M_{ia})

тобто відповідь є ЗГОРТКОЮ ансамбля з входом, а вхід має лише 256 вимірів ⟹
частина ансамбля, яка бере участь у відповіді, має ранг ≤256 НЕЗАЛЕЖНО від того,
що повний ансамбль високорангова. Тут це міряється ПРЯМО: без ескізу (проби =
одинична матриця ⟹ маємо повну `M_s`), з ЛІНІЙНИМ урізанням до `k` мод.

    Δ_s = M_s − M̄ ,  X = [vec Δ_s] ,  G = XXᵀ ,  v_j — топ-власні вектори G
    Δ_s^(k) = Σ_{j≤k} v_j[s]·U_j ,  U_j = reshape(v_jᵀX)
    A_k = (1/N) Σ_s z_s @ Δ_s^(k) = Σ_{j≤k} g_j @ U_j ,  g_j = (1/N) Σ_s v_j[s] z_s

КУТ 8. Поекземплярна калібровка доведено НЕспостережувана: оптимум залежить від
ЛІНІЙНОГО функціонала реалізованого шуму, а все обчислюване зсередини оцінювача —
КВАДРАТИЧНЕ. Але величини ансамбля є функціями `W`, а не шуму ⟹ той аргумент на
них НЕ поширюється. Тут рахується помережна похибка ЗАМИКАННЯ (двомоментного,
Мелер K=2, без калібрувальних констант) і корелюється з величинами `vw109`.

🔒 ПЛОМБИ
 S1 ТАВТОЛОГІЯ   z_s @ M_s мусить дорівнювати прямому прогону ПОБІТОВО
 S2 ПОВНОТА      k = N мусить відтворити E[out] (з точністю до члена z̄@M̄)
 S3 РАНГ БЕЗ ЕСКІЗУ  точний PR повного ансамбля проти екстраполяції ескізу
 S4 КОНТРОЛЬ     ВИПАДКОВИЙ k-вимірний підпростір замість топ-k мод
 S5 ЗАМИКАННЯ    шар 0 мусить дати похибку ≈0 (там воно ТОЧНЕ)

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ
    k=32 мод дають ≥80% відповіді                              30%
    ранг відповідь-частини значно менший за ранг ансамбля      55%
    хоч одна величина ансамбля корелює з похибкою |r|>0.5      35%
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
TAG = sys.argv[1] if len(sys.argv) > 1 else "a"
NET0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NNET = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NSAMP = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
CHUNK = 256
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
OUT = f"/home/trim/wblogs/vw109b_{TAG}.json"

state = {}
if os.path.exists(OUT):
    try:
        state = json.load(open(OUT))
        print(f"↻ підхоплено {len(state)} мереж", flush=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"⚠️ чекпоінт нечитабельний ({e})", flush=True)


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, OUT)


import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 ВИДНО НЕ ОДНУ КАРТУ — вихід.", flush=True)
    sys.exit(2)
print(f"vw109b[{TAG}] · {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
      f" · мережі {NET0}..{NET0+NNET-1} · N={NSAMP}", flush=True)

Wall, TRUE = [], []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall.append(np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_))
    yy = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(yy.type):
        yy = yy.flatten()
    TRUE.append(np.asarray(yy).reshape(tb.num_rows, W_))
Wall = np.concatenate(Wall, 0)
TRUE = np.concatenate(TRUE, 0)

SQ2PI = math.sqrt(2.0 * math.pi)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure_cos_err(W, truth, chk=None):
    """Двомоментне гаусове замикання, Мелер K=2, БЕЗ калібрувальних констант.

    Повертає косинусну похибку (масштаб знімається оптимальним скаляром — саме
    так вимагає корпус міряти будь-яку поправку до `m`) і похибку на шарі 0.
    """
    m = cp.zeros(W_, dtype=cp.float64)
    C = (W[0].T @ W[0]).astype(cp.float64)
    err0 = {}
    for i in range(1, DEPTH + 1):
        s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
        t = m / s
        Ph, ph = ndtr(t), phi(t)
        ma = s * (ph + t * Ph)
        q = s * s * ((1.0 + t * t) * Ph + t * ph)          # E[a²]
        rho = C / cp.maximum(cp.outer(s, s), 1e-300)
        Ca = cp.outer(Ph, Ph) * C + 0.5 * cp.outer(ph, ph) * rho * rho \
            * cp.outer(s, s)
        cp.fill_diagonal(Ca, q - ma * ma)
        if chk is not None and i in chk:
            sm, ss = chk[i]
            err0[str(i)] = (float(cp.mean(cp.abs(m - sm)) / cp.mean(cp.abs(sm) + 1e-30)),
                       float(cp.mean(cp.abs(s - ss)) / cp.mean(ss)))
        if i == DEPTH:
            break
        Wi = W[i].astype(cp.float64)
        m = Wi.T @ ma
        C = Wi.T @ Ca @ Wi
    s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
    t = m / s
    pred = s * (phi(t) + t * ndtr(t))
    T = cp.asarray(truth, dtype=cp.float64)
    c = float(cp.dot(pred, T)) / math.sqrt(
        float(cp.dot(pred, pred)) * float(cp.dot(T, T)))
    return math.sqrt(max(0.0, 1.0 - c * c)), err0


def top_evecs_gram(G, k, iters=60, seed=5):
    d = G.shape[0]
    Q = cp.random.default_rng(seed).standard_normal((d, k), dtype=cp.float32)
    for _ in range(iters):
        Z = G @ Q
        for j in range(k):
            q = Z[:, j]
            if j:
                q = q - Q[:, :j] @ (Q[:, :j].T @ q)
            Q[:, j] = q / cp.maximum(cp.sqrt(cp.sum(q * q)), 1e-20)
    return Q


def one_net(net):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
         for L in range(DEPTH)]
    rng = cp.random.default_rng(9109 + net)
    z = rng.standard_normal((NSAMP, W_), dtype=cp.float32)

    masks, a = [], z
    for i in range(DEPTH):
        p = a @ W[i]
        mk = p > 0
        masks.append(mk)
        a = p * mk
    out_direct = a
    mo = out_direct.mean(0).astype(cp.float64)
    nrm = float(cp.sqrt(cp.sum(mo * mo)))

    # ── повна M посемплово, чанками ──────────────────────────────────────────
    X = cp.empty((NSAMP, W_ * W_), dtype=cp.float32)
    I = cp.eye(W_, dtype=cp.float32)
    s1 = 0.0
    for b0 in range(0, NSAMP, CHUNK):
        b1 = min(b0 + CHUNK, NSAMP)
        v = cp.broadcast_to(I[None], (b1 - b0, W_, W_)).copy()
        for i in range(DEPTH):
            v = (v.reshape(-1, W_) @ W[i]).reshape(b1 - b0, W_, W_)
            v = v * masks[i][b0:b1][:, None, :]
        # S1 ТАВТОЛОГІЯ: z_s @ M_s проти прямого прогону
        chk = cp.einsum("si,sij->sj", z[b0:b1], v)
        s1 = max(s1, float(cp.max(cp.abs(chk - out_direct[b0:b1]))
                            / cp.maximum(cp.mean(cp.abs(out_direct)), 1e-30)))
        X[b0:b1] = v.reshape(b1 - b0, -1)
        del v, chk
    mbar = X.mean(0)
    X -= mbar
    zb = z.mean(0).astype(cp.float64)
    base = mo - zb @ mbar.reshape(W_, W_).astype(cp.float64)   # = A_повне

    # точний PR повного ансамбля (S3)
    G = X @ X.T
    trG = float(cp.trace(G))
    pr_exact = trG * trG / max(float(cp.sum(G * G)), 1e-300)

    kmax = min(max(KS), NSAMP // 2)
    V = top_evecs_gram(G, kmax)
    Zc = z.astype(cp.float32)
    gj = (V.T @ Zc) / NSAMP                        # (k,256)
    Uj = V.T @ X                                   # (k, 65536)

    def acc(vv, uu, k):
        A = cp.zeros(W_, dtype=cp.float64)
        for j in range(k):
            A += (vv[j].astype(cp.float64)
                  @ uu[j].reshape(W_, W_).astype(cp.float64))
        return A

    res, resr = {}, {}
    Vr = top_evecs_gram(cp.eye(NSAMP, dtype=cp.float32), kmax, iters=1, seed=99)
    gr = (Vr.T @ Zc) / NSAMP
    Ur = Vr.T @ X
    for k in KS:
        if k > kmax:
            continue
        A = acc(gj, Uj, k)
        res[str(k)] = (float(cp.sum(A * mo)) / max(nrm * nrm, 1e-300),
                       float(cp.sqrt(cp.sum(A * A))) / max(nrm, 1e-300))
        Ar = acc(gr, Ur, k)
        resr[str(k)] = float(cp.sum(Ar * mo)) / max(nrm * nrm, 1e-300)
    # 🚨 ПОЗАВИБІРКОВЕ: моди з половини A, відповідь рахується на половині B.
    # Без цього крива ВНУТРІШНЬОВИБІРКОВА і при k→N є тавтологією.
    h = NSAMP // 2
    GA = X[:h] @ X[:h].T
    VA = top_evecs_gram(GA, kmax)
    UA = VA.T @ X[:h]                                  # моди-матриці з A
    nu = cp.maximum(cp.sum(UA * UA, 1), 1e-30)
    cB = (X[h:] @ UA.T) / nu[None, :]                  # коефіцієнти B у модах A
    moB = out_direct[h:].mean(0).astype(cp.float64)
    nrmB = float(cp.sqrt(cp.sum(moB * moB)))
    gB = (cB.T @ z[h:].astype(cp.float32)) / (NSAMP - h)
    oos = {}
    for k in KS:
        if k > kmax:
            continue
        A = acc(gB, UA, k)
        oos[str(k)] = float(cp.sum(A * moB)) / max(nrmB * nrmB, 1e-300)
    # 🔒 контроль позавибіркового: моди з A, але коефіцієнти від ПЕРЕМІШАНИХ B
    pm = cp.asarray(np.random.default_rng(13 + net).permutation(NSAMP - h))
    gBs = (cB[pm].T @ z[h:].astype(cp.float32)) / (NSAMP - h)
    oos_sh = {str(k): float(cp.sum(acc(gBs, UA, k) * moB))
              / max(nrmB * nrmB, 1e-300) for k in KS if k <= kmax}
    del GA, VA, UA, cB

    # S2 ПОВНОТА: усі моди разом
    Afull = acc(gj, Uj, kmax)
    s2 = float(cp.sqrt(cp.sum((Afull - base) ** 2))) / max(nrm, 1e-300)

    sp = {}
    aa = z
    for i in range(DEPTH):
        pp = aa @ W[i]
        if i + 1 in (1, 16, 31):
            sp[i + 1] = (pp.mean(0, dtype=cp.float64),
                         pp.astype(cp.float64).std(0))
        aa = pp * masks[i]
    cerr, cerr0 = closure_cos_err(W, TRUE[net], sp)
    del X, G, V, Uj, Vr, Ur
    cp.get_default_memory_pool().free_all_blocks()
    return {"S1": s1, "S2": s2, "pr_exact": pr_exact,
            "trunc": res, "trunc_rand": resr,
            "closure_cos_err": cerr, "closure_chk": cerr0,
            "oos": oos, "oos_shuf": oos_sh,
            "base_over_answer": float(cp.sqrt(cp.sum(base * base))) / nrm}


fails = 0
for net in range(NET0, NET0 + NNET):
    if str(net) in state:
        continue
    try:
        t0 = time.time()
        r = one_net(net)
        state[str(net)] = r
        save()
        fails = 0
        tr = r["trunc"]
        print(f"  ✅ {net}: PR точний {r['pr_exact']:.0f} · "
              f"k=8 {100*tr['8'][0]:.1f}% (рандом {100*r['trunc_rand']['8']:.1f}%) · "
              f"k=64 {100*tr['64'][0]:.1f}% ({100*r['trunc_rand']['64']:.1f}%) · "
              f"k=512 {100*tr.get('512',[float('nan')])[0]:.1f}% · "
              f"S1 {r['S1']:.1e} · S2 {r['S2']:.1e} · "
              f"ПОЗАВИБ k=64 {100*r['oos']['64']:.1f}% "
              f"(🔒переміш {100*r['oos_shuf']['64']:.1f}%) · "
              f"замик {100*r['closure_cos_err']:.3f}% "
              f"(🔒σ шар16 {100*r['closure_chk']['16'][1]:.2f}%) · "
              f"{time.time()-t0:.0f} с",
              flush=True)
    except Exception as e:                                    # noqa: BLE001
        fails += 1
        print(f"  ⚠️ {net} ВПАЛА ({type(e).__name__}: {e}) — {fails}/3",
              flush=True)
        import traceback
        traceback.print_exc()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(10)
        if fails >= 3:
            save()
            sys.exit(3)

save()
print(f"\n✅ ГОТОВО: {len(state)} мереж → {OUT}")
