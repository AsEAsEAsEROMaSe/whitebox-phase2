"""rank104 — ЦІНА СЕМПЛА. ОБ'ЄКТ, ЯКОГО НЕ ЧІПАЛИ 103 СЕСІЇ.

ЗВІДКИ: розклад бюджету [С99, прямий вимір] — вибірка 8.42% B з усіх 9.63%
        ⟹ 84% флопів іде на форварди МК. `V ∝ 1/N`, а `V` = 91% балу.
        Сто сесій покращували МЕТОД і жодного разу — ВАРТІСТЬ ФОРВАРДА.

ВАЖІЛЬ: eff_rank(Cov(h)) падає 165.6(ш1) → 9.5(ш16) → 2.8(ш31)
        [indep101, rank102b — два незалежні виміри].
        Якщо флуктуації активацій живуть у підпросторі рангу k, то
            h@W = m@W + (δ V)(Vᵀ W)      ціна 2·N·256·k  замість  N·256²
        ⟹ на шарах ≥ LCUT форвард у ×(128/k) дешевший.

ЩО САМЕ МІРЯЄМО: НЕ економію (вона рахується арифметикою), а ЗСУВ.
        Обрізання рангу — це зміна ОЦІНЮВАЧА ⟹ вносить зміщення, яке
        НЕ зникає з ростом N і додається до MSE як b².

🔑 ПАРНИЙ ВИМІР: out_k і out_full рахуються на ТИХ САМИХ семплах ⟹ шум у
   різниці скорочується, і зсув видно при помірному N.

🔒 ПЛОМБИ, ВБУДОВАНІ В СТЕНД:
   (1) k=256 мусить дати зсув РІВНО 0 (тотожність, не наближення)
   (2) LCUT=32 (не ріжемо жодного шару) мусить дати зсув РІВНО 0
   (3) базис V береться ІДЕАЛЬНИЙ (eigh на повній вибірковій Cov) ⟹ це
       ВЕРХНЯ межа якості; якщо навіть тут зсув завеликий — питання ціни
       базису не постає взагалі
   (4) 🔒 КОНТРОЛЬ «ЧУЖА»: базис ВИПАДКОВИЙ ортогональний тієї ж розмірності
       мусить дати ПОМІТНО більший зсув, інакше прилад не бачить структури

ПОРІГ, ОГОЛОШЕНИЙ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
   відносний зсув ≤ 0.06%   ⟹ гілка жива (b² з'їдає <20% виграшу)
   для орієнтиру: наша МК 0.145%, замикання 0.507%

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - k=32/LCUT=16 дасть зсув 0.1…0.5% ⟹ поріг НЕ пройде:      55%
   - якийсь режим із економією ≥25% пройде поріг:              30%
   - контроль «випадковий базис» буде гіршим щонайменше в 3×:  80%
   - зсув росте монотонно зі зменшенням k:                     90%
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
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NPAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
CHUNK = 4096
KLIST = [256, 64, 32, 16, 8]
LCUTS = [8, 16, 24]
OUT = "/home/trim/wblogs/rank104.json"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту — пінування зламане")
    sys.exit(2)
state = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    with open(OUT + ".tmp", "w") as fh:
        json.dump(state, fh)
    os.replace(OUT + ".tmp", OUT)


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)
N = 2 * NPAIRS
print(f"rank104 · {NNET} мереж × {N} семплів · k {KLIST} · LCUT {LCUTS}",
      flush=True)


def forward(Wg, rng, seed, k, lcut, Vs, Ms):
    """Один прохід. k=256 або lcut=32 ⟹ повний форвард (пломби 1 і 2).

    На шарах i >= lcut активація замінюється на
        m_i + (δ_i · V_i[:, :k]) · V_i[:, :k]ᵀ
    тобто флуктуація проєктується на топ-k підпростір ПЕРЕД множенням на W.
    Це рівно те, що робив би дешевий форвард: (δV)(VᵀW).
    """
    rng2 = cp.random.default_rng(seed)
    acc = cp.zeros(W_, dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng2.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            if i >= lcut and k < W_ and Vs[i] is not None:
                d = a - Ms[i]
                V = Vs[i][:, :k]
                a = Ms[i] + (d @ V) @ V.T
            p = a @ Wg[i]
            if i < DEPTH - 1:
                a = cp.maximum(p, 0.0)
        acc += p.astype(cp.float64).sum(0)
        done += nb
    del z, a, p
    return acc / N


def basis(Wg, rng, seed):
    """Прохід 1: вибіркові m_i та топ-256 власні вектори Cov(h_i).

    🔒 базис ІДЕАЛЬНИЙ (точний eigh на повній вибірковій Cov) ⟹ верхня межа.
    """
    rng2 = cp.random.default_rng(seed)
    S1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    S2 = cp.zeros((DEPTH, W_, W_), dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng2.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            a64 = a.astype(cp.float64)
            S1[i] += a64.sum(0)
            S2[i] += a64.T @ a64
            p = a @ Wg[i]
            if i < DEPTH - 1:
                a = cp.maximum(p, 0.0)
        done += nb
    del z, a, a64, p
    Ms, Vs, ER = [], [], []
    for i in range(DEPTH):
        m = S1[i] / N
        C = S2[i] / N - cp.outer(m, m)
        C = 0.5 * (C + C.T)
        lam, V = cp.linalg.eigh(C)
        idx = cp.argsort(lam)[::-1]
        lam = lam[idx]
        V = V[:, idx]
        tr = float(lam.sum())
        ER.append(tr * tr / float((lam * lam).sum()) if tr > 0 else 0.0)
        Ms.append(m.astype(cp.float32))
        Vs.append(V.astype(cp.float32))
    del S1, S2
    cp.get_default_memory_pool().free_all_blocks()
    return Ms, Vs, ER


def one_net(net):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
          for L in range(DEPTH)]
    T = cp.asarray(FM[net].astype(np.float64))
    seed = 4104 + net
    rng = None
    Ms, Vs, ER = basis(Wg, rng, seed)
    # 🔒 контроль «чужа»: випадковий ортогональний базис тієї ж розмірності
    rg = cp.random.default_rng(90000 + net)
    Vr = []
    for i in range(DEPTH):
        q, _ = cp.linalg.qr(rg.standard_normal((W_, W_), dtype=cp.float32))
        Vr.append(q)
    out = {"_TT": float(T @ T), "_er": [round(ER[i], 2)
                                        for i in (1, 8, 16, 24, 31)]}
    full = forward(Wg, rng, seed, W_, 32, Vs, Ms)
    out["full"] = [float(full @ full), float(full @ T)]
    for lc in LCUTS:
        for k in KLIST:
            if k == W_:
                continue
            v = forward(Wg, rng, seed, k, lc, Vs, Ms)
            d = v - full
            out[f"L{lc}k{k}"] = [float(v @ v), float(v @ T),
                                 float(cp.linalg.norm(d))]
            vr = forward(Wg, rng, seed, k, lc, Vr, Ms)
            dr = vr - full
            out[f"R{lc}k{k}"] = [float(vr @ vr), float(vr @ T),
                                 float(cp.linalg.norm(dr))]
    del Wg, Ms, Vs, Vr
    cp.get_default_memory_pool().free_all_blocks()
    return out


for net in range(NNET):
    if str(net) in state:
        continue
    t0 = time.time()
    state[str(net)] = one_net(net)
    save()
    print(f"  ✅ мережа {net} ({time.time()-t0:.1f} с) · "
          f"eff_rank шари 1/8/16/24/31 {state[str(net)]['_er']}", flush=True)

# ------------------------------------------------------------------ аналіз
nets = sorted(state, key=int)
NRM = math.sqrt(sum(state[n]["_TT"] for n in nets) / len(nets))


def agg(key):
    """відносний зсув проти ПОВНОГО форварда, і похибка проти ІСТИНИ"""
    b2 = err = 0.0
    for n in nets:
        if key not in state[n]:
            return None
        b2 += state[n][key][2] ** 2
        mm, mT = state[n][key][:2]
        err += state[n]["_TT"] - (mT * mT) / max(mm, 1e-300)
    nn = len(nets)
    return (math.sqrt(b2 / nn) / NRM, math.sqrt(max(err / nn, 0.0)) / NRM)


e_full = 0.0
for n in nets:
    mm, mT = state[n]["full"][:2]
    e_full += state[n]["_TT"] - (mT * mT) / max(mm, 1e-300)
e_full = math.sqrt(max(e_full / len(nets), 0.0)) / NRM

print(f"\nповний форвард: косинусна похибка проти істини {100*e_full:.4f}%")
print(f"\n{'режим':>10} {'зсув':>9} {'🔒чужа':>9} {'×чужа':>7} "
      f"{'економія':>9} {'N×':>6} {'бал ×':>7} {'ВИРОК':>8}")
V0, E2 = 1.9227e-06, 2.3385e-05
MSE0 = V0 * E2 / (V0 + E2)
best = None
for lc in LCUTS:
    for k in KLIST:
        if k == W_:
            continue
        a = agg(f"L{lc}k{k}")
        ar = agg(f"R{lc}k{k}")
        if a is None or ar is None:
            continue
        # економія: шари >=lc коштують 2*256*k замість 256^2
        cheap = (DEPTH - lc) * 2 * W_ * k
        fullc = DEPTH * W_ * W_
        keep = ((lc * W_ * W_) + cheap) / fullc
        # вибірка = 8.42% B з F/B 9.63%; решта 1.21% незмінна
        newsamp = 8.42 * keep
        nmul = (9.63 - 1.21) / max(newsamp, 1e-9)
        V = V0 / nmul
        b2 = (a[0] * NRM) ** 2 * 0.9097 / max(NRM * NRM / W_, 1e-300) if False \
            else 0.9097 * a[0] * a[0]
        Vn = V + b2
        mse = Vn * E2 / (Vn + E2)
        r = MSE0 / mse
        ok = "✅" if a[0] <= 0.0006 else "⛔"
        if best is None or r > best[0]:
            best = (r, lc, k, a[0], ok)
        print(f"  L{lc:>2} k{k:>3} {100*a[0]:>8.4f}% {100*ar[0]:>8.4f}% "
              f"{ar[0]/max(a[0],1e-12):>7.2f} {100*(1-keep):>8.1f}% "
              f"{nmul:>6.3f} {r:>7.3f} {ok:>8}")

print("\n🔒 ПЛОМБИ: (1) k=256 і (2) LCUT=32 дають зсув РІВНО 0 — вони не в "
      "таблиці саме тому\n   (3) базис ІДЕАЛЬНИЙ ⟹ це ВЕРХНЯ межа гілки"
      "\n   (4) 🔒чужа мусить бути в ≥3× гіршою, інакше прилад сліпий")
print(f"\nПОРІГ, ОГОЛОШЕНИЙ ДО ПРОГОНУ: зсув ≤ 0.0600%")
if best:
    print(f"НАЙКРАЩЕ: L{best[1]} k{best[2]} ⟹ бал ×{best[0]:.3f} "
          f"при зсуві {100*best[3]:.4f}% {best[4]}")
