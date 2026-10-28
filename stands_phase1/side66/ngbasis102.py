"""ngbasis102 — ЧИ Є БАЗИС, У ЯКОМУ КОНЦЕНТРУЄТЬСЯ НЕГАУСОВІСТЬ?

ПІДСТАВА — ДВА СЬОГОДНІШНІ ВИМІРИ, ЯКІ РАЗОМ СТАВЛЯТЬ ПИТАННЯ:
  indep101:  96% негаусовості передактивацій сидить у ЗАЛЕЖНОСТІ між нейронами
             (γ₄ реальна 0.3947 проти перемішаної 0.0142); еф.ранг Cov = 2.8
  lowrank101: базис ІЗ ВЛАСНИХ ВЕКТОРІВ Cov знімає лише 17% похибки кроку при
             r=4 і 38% при r=16 ⟹ негаусовість НЕ там, де дисперсія

І ЦЕ Ж РОЗВ'ЯЗУЄ СУПЕРЕЧНІСТЬ ІЗ СИТОМ С99: сито міряло МАРГІНАЛЬНІ γ₃,γ₄, які
за побудовою не бачать 96% явища ⟹ його стеля 1.48e-8 є стелею МАРГІНАЛЬНОГО
ОПИСУ, а не класу.

ПИТАННЯ: чи існує напрямок/підпростір, у якому негаусовість зосереджена так
само, як дисперсія зосереджена у власному базисі Cov?

ТРИ БАЗИСИ, ОДНА Й ТА САМА МЕТРИКА (частка похибки кроку, знята моделлю):
  PCA   власні вектори Cov(h)            — те, що вже зміряно (×1.21 при r=4)
  NG    напрямки максимальної |γ₄|       — projection pursuit, степеневий метод
  RND   🔒 випадковий ортонормований      — КОНТРОЛЬ, має дати ~r/256

ЯК ШУКАЮТЬСЯ NG-НАПРЯМКИ (без ICA-бібліотек, чистий степеневий метод):
  вибілюємо h ⟹ y = Λ^{-1/2}Uᵀ(h−m); для вибіленого y будь-який напрямок має
  одиничну дисперсію ⟹ |γ₄(vᵀy)| — чиста міра негаусовості.
  ітерація FastICA-типу:  v ← E[y·(vᵀy)³] − 3v,  далі ортогоналізація Грама-Шмідта
  (це градієнт |κ₄| при одиничній нормі — стандартна конструкція)

🔒 ПЛОМБИ:
  (1) RND мусить дати частку ≈ r/256 (ізотропний рівень)
  (2) сума γ₄ по ВСІХ 256 напрямках однакова в будь-якому ортобазисі ⟹
      порівнюємо саме КОНЦЕНТРАЦІЮ, а не рівень
  (3) шар 1 (де негаусовість мізерна) мусить дати ≈нуль для всіх трьох

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) NG-напрямки дадуть |γ₄| у 3+ разів більший за PCA-напрямки          60%
 (б) NG зніме >35% похибки кроку при r=4 (проти 17% у PCA)               30%
 (в) 🔒 RND дасть ≈ r/256 = 1.6%                                         80%
 (г) NG-напрямки НЕ збігатимуться з головними PCA (|cos| < 0.3)          65%

ВИРОК: (б) < 25% ⟹ негаусовість ІЗОТРОПНА ⟹ носія не існує, і клас
закритий уже ВИМІРОМ, а не припущенням. (б) > 35% ⟹ носій є, і далі
питання лише, чи знаходиться він детерміновано з W.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
LS = [1, 16, 31]
RS = [4, 16]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"ngbasis102 · {NNET} мереж · {NS} семплів · шари {LS} · ранги {RS}",
      flush=True)


def psi(t):
    return cp.exp(-0.5 * t * t) / SQ2PI + t * ndtr(t)


def acts(Wg, layer, ns, seed):
    rng = cp.random.default_rng(seed)
    out = []
    n = 0
    while n < ns:
        nb = min(32768, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        h = cp.concatenate([z, -z], axis=0)
        for i in range(layer):
            h = cp.maximum(h @ Wg[i], 0.0)
        out.append(h)
        n += nb
    return cp.concatenate(out, axis=0).astype(cp.float64)


def whiten_basis(hc):
    """Λ^{-1/2}Uᵀ — вибілювач; повертає (whitener, dewhitener, lam, U)"""
    cov = (hc.T @ hc) / hc.shape[0]
    lam, U = cp.linalg.eigh(cov)
    lam = cp.maximum(lam, 1e-14)
    Wh = U * (1.0 / cp.sqrt(lam))[None, :]        # h_c @ Wh  ⟹ вибілене y
    De = U * cp.sqrt(lam)[None, :]                # y @ De.T  ⟹ назад у h
    return Wh, De, lam, U


def ng_directions(y, r, iters=24, seed=7):
    """напрямки максимальної |γ₄| у вибіленому просторі (FastICA-градієнт)"""
    rng = cp.random.default_rng(seed)
    V = rng.standard_normal((W_, r), dtype=cp.float64)
    V, _ = cp.linalg.qr(V)
    N = y.shape[0]
    for _ in range(iters):
        # E[y (vᵀy)³] − 3v  для всіх напрямків одразу
        P = y @ V                                  # (N, r)
        G = (y.T @ (P ** 3)) / N - 3.0 * V
        V, _ = cp.linalg.qr(G)                     # симетрична ортогоналізація
    return V


def step_err(hA, hB, w, U, r):
    """похибка кроку моделі: ядро в r напрямках `U` + гаусів залишок.
    U=None ⟹ повністю гаусова модель (r=0)."""
    mA = hA.mean(0)
    cA = hA - mA
    mp = mA @ w
    cov = (cA.T @ cA) / cA.shape[0]
    var_tot = cp.einsum("ij,ia,ja->a", cov, w, w)
    if U is None:
        s = cp.sqrt(cp.maximum(var_tot, 1e-30))
        pred = s * psi(mp / s)
    else:
        xi = cA @ U
        B = U.T @ w
        cxi = (xi.T @ xi) / xi.shape[0]
        var_k = cp.einsum("kl,ka,la->a", cxi, B, B)
        se = cp.sqrt(cp.maximum(var_tot - var_k, 1e-12))
        acc = cp.zeros(W_, dtype=cp.float64)
        for k in range(0, xi.shape[0], 16384):
            acc += psi((mp + xi[k:k + 16384] @ B) / se).sum(0)
        pred = se * (acc / xi.shape[0])
    truth = cp.maximum(hB @ w, 0.0).mean(0)
    nrm = cp.sqrt((truth * truth).mean())
    return float(cp.sqrt(((pred - truth) ** 2).mean()) / nrm)


res = {}
g4info = {}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    for L in LS:
        hA = acts(Wg, L, NS // 2, 4242 + net)
        hB = acts(Wg, L, NS // 2, 999983 + net)
        w = Wg[L].astype(cp.float64)
        mA = hA.mean(0)
        cA = hA - mA
        Wh, De, lam, U = whiten_basis(cA)
        y = cA @ Wh                                # вибілене

        res.setdefault((L, 0, "base"), []).append(step_err(hA, hB, w, None, 0))
        rng = cp.random.default_rng(555 + net)
        for r in RS:
            # PCA: r головних власних напрямків Cov
            Upca = U[:, -r:]
            # NG: r напрямків максимальної |γ₄|, повернуті у простір h
            Vng = ng_directions(y, r, seed=11 + net)
            Ung, _ = cp.linalg.qr(De @ Vng)
            # 🔒 RND: випадковий ортобазис
            Urnd, _ = cp.linalg.qr(rng.standard_normal((W_, r), dtype=cp.float64))
            for nm, UU in (("PCA", Upca), ("NG", Ung), ("RND", Urnd)):
                res.setdefault((L, r, nm), []).append(step_err(hA, hB, w, UU, r))
            # діагностика: |γ₄| у знайдених напрямках проти головних PCA
            for nm, VV in (("PCA", U[:, -r:]), ("NG", Ung)):
                p = cA @ VV
                p = p - p.mean(0)
                v = (p * p).mean(0)
                g4 = ((p ** 4).mean(0) - 3 * v * v) / cp.maximum(v * v, 1e-30)
                g4info.setdefault((L, r, nm), []).append(
                    float(cp.abs(g4).mean()))
            if r == RS[0]:
                cs = cp.abs(Ung.T @ U[:, -r:]).max()
                g4info.setdefault((L, r, "cos"), []).append(float(cs))
        del hA, hB, cA, y
        cp.get_default_memory_pool().free_all_blocks()
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'шар':>4s}{'r':>4s}{'база':>10s}{'PCA':>10s}{'NG':>10s}{'🔒RND':>10s}"
      f"{'×NG':>8s}{'×PCA':>8s}{'|γ₄|PCA':>9s}{'|γ₄|NG':>9s}{'🔒maxcos':>10s}")
print("=" * 102)
for L in LS:
    b = np.mean(res[(L, 0, "base")])
    for r in RS:
        p = np.mean(res[(L, r, "PCA")])
        n = np.mean(res[(L, r, "NG")])
        d = np.mean(res[(L, r, "RND")])
        cs = np.mean(g4info[(L, r, "cos")]) if (L, r, "cos") in g4info else float('nan')
        print(f"{L:>4d}{r:>4d}{100*b:>9.4f}%{100*p:>9.4f}%{100*n:>9.4f}%"
              f"{100*d:>9.4f}%{b/n:>8.3f}{b/p:>8.3f}"
              f"{np.mean(g4info[(L,r,'PCA')]):>9.4f}"
              f"{np.mean(g4info[(L,r,'NG')]):>9.4f}{cs:>10.3f}")
print("\n🔒 RND мусить лежати біля бази (ізотропний рівень r/256)")
print("ВИРОК: NG знімає <25% ⟹ негаусовість ІЗОТРОПНА, носія не існує")
print("       NG знімає >35% ⟹ носій Є, далі питання — чи знайти його з W")
