"""vw109 — ВІРТУАЛЬНІ ВАГИ: прямий погляд на об'єкт, а не на нашу похибку (ідея ЮЗЕРА).

ТОЧНА ТОТОЖНІСТЬ, НА ЯКІЙ УСЕ СТОЇТЬ
    out(z) = z @ M_{D(z)},   M_D = W₀D₀W₁D₁ ⋯ W₃₁D₃₁,   D_i = diag(1[p_i>0])
тобто мережа Є лінійною мережею, чиї ваги залежать від входу. Звідси
    E[out] = M̄·E[z] + E[(M_D − M̄)z] = 0 + E[(M_D − M̄)z]
⟹ УСЯ ВІДПОВІДЬ Є КОВАРІАЦІЄЮ між тим, яку мережу отримав, і тим, який вхід.

ПРИЛАД: ЗАМОРОЖЕНІ ЗАТВОРИ + ПРОБИ
Зберігати M (256×256 на семпл) неможливо. Зберігаємо ЙОГО ДІЮ: женемо z,
запам'ятовуємо маски, тоді женемо p пробних векторів крізь ЗАМОРОЖЕНІ маски.
Дістаємо ескіз `X_s = vec(U @ M_{D(z_s)})` розміру 256·p — незміщений начерк
ансамбля (випадкове скетчування).

ЩО МІРЯЄМО (шість величин, яких у корпусі немає)
 A. ефективний ранг АНСАМБЛЯ ВІДОБРАЖЕНЬ (не стану, не зворотного оператора)
 B. пошарова атрибуція відповіді до мінливості затворів
 C. порядок взаємодії затворів (сума соло проти цілого)
 D. розклад E[out] на M̄ і флуктуацію
 E. умовна γ₄ всередині кластера ансамбля
 F. перенос спектральної форми між мережами (в аналізі)

🔒 П'ЯТЬ ПЛОМБ
 P1 ТАВТОЛОГІЯ: out із масок мусить збігтися з прямим прогоном ПОБІТОВО.
    (у step105 така сама пломба впіймала C[i] замість C[i+1] — без неї вирок
     був би хибним)
 P2 ІСТИНА: E[relu(p₃₁)] мусить збігтися з `final_means` датасету.
 P3 ПЕРЕМІШАНІ ЗАТВОРИ: маски від чужого семплу ⟹ E[out] РІВНО 0.
    Це пломба на САМУ ТОТОЖНІСТЬ, а не на результат.
 P4 ВИРОДЖЕНИЙ АНСАМБЛЬ: усі маски = 1 (лінійна мережа) ⟹ ранг РІВНО 1.
    І зворотний калібр: маски Бернуллі(½) незалежні від z ⟹ ранг ВЕЛИКИЙ.
 P5 МАСШТАБУВАННЯ: ранг при N і N/4 мусить СТОЯТИ. Росте ⟹ міряємо шум.

📣 ПЕРЕДБАЧЕННЯ, ЗАПИСАНІ ДО ПРОГОНУ (щоб не тлумачити заднім числом)
    ефективний ранг ансамбля 3…30, 90% маси у ≤30 модах            60%
    останні 8 шарів дають ≥60% відповіді                           70%
    P3 дає 0 до точності вибірки                                   95%
    умовна γ₄ падає нижче 0.10 при k=16                            45%
    🔑 НАЙЦІННІШИЙ РЕЗУЛЬТАТ — «ранг ВЕЛИКИЙ»: він убиває і суміш, і всю
       конструкцію віртуальних ваг ОДНИМ числом за один прогін.

ВИКОРИСТАННЯ
   CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID \
   CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 vw109.py <tag> <net0> <nnet> <N> <p>
"""
import glob
import json
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
NNET = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NSAMP = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
NPROBE = int(sys.argv[5]) if len(sys.argv) > 5 else 16
KCLUST = 16
OUT = f"/home/trim/wblogs/vw109_{TAG}.json"

state = {}
if os.path.exists(OUT):
    try:
        state = json.load(open(OUT))
        print(f"↻ підхоплено {len(state)} готових мереж із {OUT}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"⚠️ чекпоінт нечитабельний ({e}), починаю з нуля", flush=True)


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, OUT)


import cupy as cp                                             # noqa: E402

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 ВИДНО НЕ ОДНУ КАРТУ — пінування не спрацювало, вихід.", flush=True)
    sys.exit(2)
print(f"vw109[{TAG}] · карта "
      f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} · "
      f"мережі {NET0}..{NET0+NNET-1} · N={NSAMP} · проб={NPROBE}", flush=True)

# ── дані ─────────────────────────────────────────────────────────────────────
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
print(f"   завантажено {Wall.shape[0]} мереж, істина {TRUE.shape}", flush=True)


def eff_rank(Xc):
    """Ефективний ранг = (Σλ)²/Σλ² через коваріацію ескізу. Без eigh."""
    n, d = Xc.shape
    S = (Xc.T @ Xc) / n                       # d×d
    tr = float(cp.trace(S))
    fro = float(cp.sum(S * S))
    pr = tr * tr / max(fro, 1e-300)
    return pr, S


def spectrum_mass(S, ks=(1, 4, 8, 16, 32, 64)):
    ev = cp.linalg.eigvalsh(S.astype(cp.float64))
    ev = cp.sort(ev)[::-1]
    ev = cp.maximum(ev, 0.0)
    tot = float(cp.sum(ev))
    out = {}
    for k in ks:
        out[str(k)] = float(cp.sum(ev[:k])) / max(tot, 1e-300)
    return out, cp.asnumpy(ev[:64] / max(tot, 1e-300))


def top_evecs(S, k, iters=40, seed=11):
    """Топ-k власних векторів через підпросторову ітерацію.

    🚨 cusolver `eigh` з ВЕКТОРАМИ падає на цій карті (P104-100, sm_61):
    CUSOLVER_STATUS_EXECUTION_FAILED, і контекст після цього отруєний.
    Тут лише матмули й Грам-Шмідт — жодного LAPACK.
    """
    d = S.shape[0]
    rs = cp.random.default_rng(seed)
    Q = rs.standard_normal((d, k), dtype=cp.float32)
    for _ in range(iters):
        Z = S @ Q
        for j in range(k):
            q = Z[:, j]
            if j:
                q = q - Q[:, :j] @ (Q[:, :j].T @ q)
            Q[:, j] = q / cp.maximum(cp.sqrt(cp.sum(q * q)), 1e-20)
    return Q


def kmeans(Y, k, iters=25, seed=7):
    idx = cp.asarray(np.random.default_rng(seed).choice(
        int(Y.shape[0]), k, replace=False))
    C = Y[idx].copy()
    lab = cp.zeros(Y.shape[0], dtype=cp.int32)
    for _ in range(iters):
        d = (cp.sum(Y * Y, 1)[:, None] - 2.0 * (Y @ C.T)
             + cp.sum(C * C, 1)[None, :])
        lab = cp.argmin(d, 1).astype(cp.int32)
        for c in range(k):
            m = lab == c
            if int(m.sum()) > 0:
                C[c] = Y[m].mean(0)
    return lab


def g4_by_groups(P, lab, k):
    """медіанна понейронна γ₄ всередині груп, зважена розміром групи."""
    tot, num = 0.0, 0.0
    for c in range(k):
        m = lab == c
        n = int(m.sum())
        if n < 64:
            continue
        Q = P[m].astype(cp.float64)
        mu = Q.mean(0)
        d = Q - mu
        s2 = (d * d).mean(0)
        m4 = (d ** 4).mean(0)
        g4 = m4 / cp.maximum(s2 * s2, 1e-300) - 3.0
        tot += n * float(cp.median(cp.abs(g4)))
        num += n
    return tot / max(num, 1.0)


def one_net(net):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
         for L in range(DEPTH)]
    rng = cp.random.default_rng(9109 + net)
    z = rng.standard_normal((NSAMP, W_), dtype=cp.float32)

    # ── прямий прогін: маски + вихід ─────────────────────────────────────────
    masks, a = [], z
    for i in range(DEPTH):
        p = a @ W[i]
        mk = (p > 0)
        masks.append(mk)
        a = p * mk                                 # = relu(p), і для i=31 теж
        if i == DEPTH - 1:
            p31 = p
    out_direct = a
    mean_out = out_direct.mean(0)

    def prop(vec, sel):
        """z крізь маски: sel(i) → cupy-маска (NSAMP,W_) або скаляр 1."""
        v = vec
        for i in range(DEPTH):
            v = v @ W[i]
            s = sel(i)
            if s is not None:
                v = v * s
        return v

    # P1 ТАВТОЛОГІЯ: та сама послідовність через збережені маски
    o1 = prop(z, lambda i: masks[i])
    p1 = float(cp.max(cp.abs(o1 - out_direct)))

    # P2 ІСТИНА
    tv = cp.asarray(TRUE[net], dtype=cp.float64)
    mo = mean_out.astype(cp.float64)
    p2 = float(cp.sqrt(cp.mean((mo - tv) ** 2))
               / cp.maximum(cp.sqrt(cp.mean(tv ** 2)), 1e-300))

    # ── A. ансамбль віртуальних ваг через проби ──────────────────────────────
    U = rng.standard_normal((NPROBE, W_), dtype=cp.float32)
    U /= cp.sqrt(cp.sum(U * U, 1, keepdims=True))
    v = cp.broadcast_to(U[None, :, :], (NSAMP, NPROBE, W_)).copy()
    depth_pr = {}
    for i in range(DEPTH):
        v = (v.reshape(-1, W_) @ W[i]).reshape(NSAMP, NPROBE, W_)
        v = v * masks[i][:, None, :]
        if i in (4, 8, 16, 24, 31):
            Xi = v.reshape(NSAMP, NPROBE * W_)
            depth_pr[str(i)] = eff_rank(Xi - Xi.mean(0))[0]
            del Xi
    X = v.reshape(NSAMP, NPROBE * W_)
    del v
    xbar = X.mean(0)
    Xc = X - xbar
    pr, S = eff_rank(Xc)
    mass, ev = spectrum_mass(S)
    # D. розклад на M̄ і флуктуацію
    fluct = float(cp.sqrt(cp.mean(cp.sum(Xc * Xc, 1)))
                  / cp.maximum(cp.sqrt(cp.sum(xbar * xbar)), 1e-300))
    # P5 МАСШТАБУВАННЯ
    pr_q, _ = eff_rank(Xc[: NSAMP // 4])

    # P4 КАЛІБР: вироджений ансамбль (усі маски 1) і безструктурний (Бернуллі)
    def sketch_rank(sel):
        vv = cp.broadcast_to(U[None, :, :], (NSAMP, NPROBE, W_)).copy()
        for i in range(DEPTH):
            vv = (vv.reshape(-1, W_) @ W[i]).reshape(NSAMP, NPROBE, W_)
            s = sel(i)
            if s is not None:
                vv = vv * s
        XX = vv.reshape(NSAMP, NPROBE * W_)
        r, _ = eff_rank(XX - XX.mean(0))
        return r

    rng2 = cp.random.default_rng(4242 + net)
    bern = [(rng2.random((NSAMP, W_), dtype=cp.float32) > 0.5)
            for _ in range(DEPTH)]
    pr_lin = sketch_rank(lambda i: None)
    pr_bern = sketch_rank(lambda i: bern[i][:, None, :])

    # ── B/C. пошарова атрибуція ──────────────────────────────────────────────
    maj = [(mk.mean(0, dtype=cp.float32) > 0.5) for mk in masks]
    nrm = float(cp.sqrt(cp.sum(mean_out.astype(cp.float64) ** 2)))

    def emean(sel):
        return prop(z, sel).mean(0).astype(cp.float64)

    e_maj = emean(lambda i: maj[i][None, :])
    attr_none = float(cp.sqrt(cp.sum(e_maj ** 2))) / max(nrm, 1e-300)
    solo, proj = [], []
    for L in range(DEPTH):
        e = emean(lambda i, L=L: (masks[i] if i == L else maj[i][None, :]))
        solo.append(float(cp.sqrt(cp.sum(e ** 2))) / max(nrm, 1e-300))
        proj.append(float(cp.sum(e * mo)) / max(nrm * nrm, 1e-300))
    sum_solo = float(np.sum(proj))

    # P3 ПЕРЕМІШАНІ ЗАТВОРИ — підлога рахується З ВЛАСНОЇ дисперсії ЦІЄЇ
    # величини, а не з дисперсії out (то була не та підлога)
    p3s = []
    for rep in range(4):
        perm = cp.asarray(
            np.random.default_rng(31337 + 97 * rep + net).permutation(NSAMP))
        osh = prop(z, lambda i: masks[i][perm]).astype(cp.float64)
        e_sh = osh.mean(0)
        fl = float(cp.sqrt(cp.sum(osh.var(0) / NSAMP)))
        p3s.append((float(cp.sqrt(cp.sum(e_sh ** 2))) / max(nrm, 1e-300),
                    fl / max(nrm, 1e-300)))
        del osh
    p3 = float(np.mean([x[0] for x in p3s]))
    p3_floor = float(np.mean([x[1] for x in p3s]))

    # ── ВИРІШАЛЬНЕ: чи відтворює СУМІШ із k віртуальних мереж відповідь ──────
    # компонента c: маски = мажоритарні ВСЕРЕДИНІ c, вхід = z̄_c
    #   Σ_c p_c · (z̄_c @ M_c)   проти   E[out]
    def mixture(lab, k):
        acc = cp.zeros(W_, dtype=cp.float64)
        for c in range(k):
            m = lab == c
            n = int(m.sum())
            if n == 0:
                continue
            mc = [(mk[m].mean(0, dtype=cp.float32) > 0.5) for mk in masks]
            v = z[m].mean(0, dtype=cp.float32)[None, :]
            for i in range(DEPTH):
                v = (v @ W[i]) * mc[i][None, :]
            acc += (n / NSAMP) * v[0].astype(cp.float64)
        return (float(cp.sum(acc * mo)) / max(nrm * nrm, 1e-300),
                float(cp.sqrt(cp.sum(acc ** 2))) / max(nrm, 1e-300))

    # ── E. умовна γ₄ ─────────────────────────────────────────────────────────
    evec = top_evecs(S.astype(cp.float32), 8)
    Y = Xc @ evec
    Y /= cp.maximum(Y.std(0, keepdims=True), 1e-20)
    lab = kmeans(Y, KCLUST)
    sizes = [int((lab == c).sum()) for c in range(KCLUST)]
    g4_all = g4_by_groups(p31, cp.zeros(NSAMP, dtype=cp.int32), 1)
    g4_cl = g4_by_groups(p31, lab, KCLUST)
    # контроль: випадкове розбиття з ТИМИ САМИМИ розмірами
    rl = cp.asarray(np.random.default_rng(777 + net).permutation(
        np.repeat(np.arange(KCLUST), np.maximum(sizes, 0))[:NSAMP]
    ).astype(np.int32)) if sum(sizes) >= NSAMP else lab[
        cp.asarray(np.random.default_rng(777 + net).permutation(NSAMP))]
    g4_rnd = g4_by_groups(p31, rl, KCLUST)

    mix = {}
    for k in (1, 4, 16, 64):
        lb = (cp.zeros(NSAMP, dtype=cp.int32) if k == 1
              else kmeans(Y, k, seed=7 + k))
        mix[str(k)] = mixture(lb, k)
        rr = cp.asarray(np.random.default_rng(555 + k + net).integers(
            0, k, NSAMP).astype(np.int32))
        mix["rnd" + str(k)] = mixture(rr, k)

    return {
        "depth_pr": depth_pr, "mix": mix,
        "P1_taut": p1, "P2_truth": p2, "P3_shuffle": p3, "P3_floor": p3_floor,
        "P4_lin": pr_lin, "P4_bern": pr_bern, "P5_quarter": pr_q,
        "eff_rank": pr, "mass": mass, "ev64": [float(x) for x in ev],
        "fluct_over_mean": fluct,
        "attr_none": attr_none, "solo": solo, "proj": proj,
        "sum_solo": sum_solo,
        "g4_all": g4_all, "g4_clust": g4_cl, "g4_rand": g4_rnd,
        "sizes": sizes,
    }


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
        print(f"  ✅ {net}: ранг {r['eff_rank']:.1f} (лін {r['P4_lin']:.2f} · "
              f"бернуллі {r['P4_bern']:.0f} · чверть {r['P5_quarter']:.1f}) · "
              f"90% маси в {[k for k,v in r['mass'].items() if v>=0.9][:1]} · "
              f"P1 {r['P1_taut']:.1e} · P2 {100*r['P2_truth']:.3f}% · "
              f"P3 {r['P3_shuffle']:.4f}/{r['P3_floor']:.4f} · "
              f"СУМІШ k=16 {100*r['mix']['16'][0]:.1f}% "
              f"(рандом {100*r['mix']['rnd16'][0]:.1f}%) · "
              f"γ₄ {r['g4_all']:.3f}→{r['g4_clust']:.3f} "
              f"(рандом {r['g4_rand']:.3f}) · {time.time()-t0:.0f} с",
              flush=True)
    except Exception as e:                                    # noqa: BLE001
        fails += 1
        print(f"  ⚠️ мережа {net} ВПАЛА ({type(e).__name__}: {e}) — "
              f"невдача {fails}/3", flush=True)
        import traceback
        traceback.print_exc()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(10)
        if fails >= 3:
            print(f"🚨 ТРИ НЕВДАЧІ ПОСПІЛЬ — збережено {len(state)}.",
                  flush=True)
            save()
            sys.exit(3)

save()
print(f"\n✅ ГОТОВО: {len(state)} мереж → {OUT}")
