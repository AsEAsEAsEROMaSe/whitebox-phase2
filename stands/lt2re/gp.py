"""20.10.2026 white-box-16: генетичне програмування мат-ядер графового типу поверх ланцюга sb2 + LT2 (ідея юзера).
Кандидат = вираз f, що дає на кожному шарі l поправку f_l (n нейронів). Примітиви:
  листки: x<k> (власні 51 стовпці дампу шару l, tf + стандартизація на шар), u (LT2-корекція шару l / es_l), phi;
  графові: A(e) = e(l-1) @ W_l, Q(e) = e(l-1) @ W_l^2 (вираз на попередньому шарі, зібраний через граф ваг);
  бінарні + - * ; унарні tanh, sq, abs, Phi (норм. cdf), neg.
Поправка додається до бічного потоку: v_l = Phi_l*(v_{l-1} @ W_l) + es_l*z(f_l), вихід += c*v_15, z = стандартизація на шар.
Фітнес = частка залишку R = truth_15 - (pred_15 + u_15), знята МНК-коефіцієнтом c на TRAIN; вирок на VAL (інші мережі).
Контролі: random (листок-шум), чужа W (переставлені мережі) — мусять дати ~0 на VAL.
ПЕРЕДБАЧЕННЯ: найкращий вираз на VAL знімає ≤1% залишку (MSE ×1.00…1.02); шанс ≥2% ~10–15%.
usage: gp.py [env NTR=64 NVA=64 POP=300 GEN=25 SEED=0]"""
import glob, json, math, os, random, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
from lt2torch import LT2, tf

torch.set_num_threads(1)
dev = "cuda"
NTR = int(os.environ.get("NTR", 64)); NVA = int(os.environ.get("NVA", 64))
POP = int(os.environ.get("POP", 300)); GEN = int(os.environ.get("GEN", 25)); SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", "/home/trim/wb2data/lt2train/gp")
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SH = "/home/trim/wb2data/corpus14k"; DUMP = "/home/trim/wb2data/lt2dump/corpus"

loc = {}
for fn in sorted(os.listdir(SH)):
    if fn.startswith("shard") and fn.endswith(".npz"):
        with np.load(os.path.join(SH, fn), allow_pickle=False) as z:
            for r, i in enumerate(np.asarray(z["idx"], dtype=np.int64)):
                loc[int(i)] = (fn, r)
files = sorted(f for f in glob.glob(DUMP + "/corpus_*.npz") if not f.endswith(".tmp.npz") and int(f[-9:-4]) >= 2048)
rs = np.random.default_rng(1); rs.shuffle(files)
sets = {"tr": files[:NTR], "va": files[NTR:NTR + NVA]}
lt = LT2("/home/trim/wb2lean/lt2re/lt2_shipped.npz", dev)
es = lt.es


def load(fs):
    need = {}
    for f in fs:
        need.setdefault(loc[int(f[-9:-4])][0], []).append(f)
    T = {}
    for fn, ff in need.items():
        with np.load(os.path.join(SH, fn), allow_pickle=False) as z:
            M = z["moments"]
            for f in ff:
                T[f] = np.asarray(M[loc[int(f[-9:-4])][1], 0, 15], dtype=np.float32)
    Xs, Ps, Ws, Us, Rs = [], [], [], [], []
    for f in fs:
        z = np.load(f); idx = int(z["idx"])
        g = torch.Generator(device="cpu").manual_seed(770000 + idx)
        W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * (2.0 / 1024) ** 0.5).to(dev)
        X = torch.from_numpy(z["X"]).to(dev); Phi = torch.from_numpy(z["Phi"]).to(dev)
        with torch.no_grad():
            _, us = lt.run(X[None], Phi[None], [W[l][None] for l in range(16)])
        U = torch.stack([u[0] for u in us])
        R = torch.from_numpy(T[f]).to(dev).double() - torch.from_numpy(z["pred"][15]).to(dev).double() - U[15].double()
        Xs.append(X); Ps.append(Phi); Ws.append(W.half()); Us.append(U); Rs.append(R.float())
    return {"X": torch.stack(Xs), "Phi": torch.stack(Ps), "W": torch.stack(Ws), "U": torch.stack(Us), "R": torch.stack(Rs)}


t0 = time.time()
D = {k: load(v) for k, v in sets.items()}
# tf + standardization of the 51 columns per layer, from TRAIN
Z = tf(D["tr"]["X"]); mu = Z.mean(dim=(0, 2)); sd = Z.std(dim=(0, 2)).clamp_min(1e-6)
for d in D.values():
    d["Z"] = (tf(d["X"]) - mu[None, :, None, :]) / sd[None, :, None, :]
    d["Un"] = d["U"] / es[None, :, None]
    del d["X"]
print("data %.0fs, tr %d va %d, |R|tr %.4e va %.4e" % (time.time() - t0, NTR, NVA, float((D["tr"]["R"] ** 2).mean()),
      float((D["va"]["R"] ** 2).mean())), flush=True)

LEAVES = ["x%d" % k for k in range(51)] + ["u", "phi", "one"]
UN = ["tanh", "sq", "abs", "cdf", "neg"]
BIN = ["+", "-", "*"]
GR = ["A", "Q"]


def rand_expr(depth):
    if depth <= 0 or random.random() < 0.3:
        return random.choice(LEAVES)
    r = random.random()
    if r < 0.25:
        return (random.choice(GR), rand_expr(depth - 1))
    if r < 0.5:
        return (random.choice(UN), rand_expr(depth - 1))
    return (random.choice(BIN), rand_expr(depth - 1), rand_expr(depth - 1))


def size(e):
    return 1 if isinstance(e, str) else 1 + sum(size(c) for c in e[1:])


def ev(e, d, l, cache):
    """value (B, n) of expression e at layer l on data d."""
    key = (id(e), l) if not isinstance(e, str) else (e, l)
    if key in cache:
        return cache[key]
    B = d["Z"].shape[0]
    if isinstance(e, str):
        if e == "u":
            v = d["Un"][:, l]
        elif e == "phi":
            v = d["Phi"][:, l]
        elif e == "one":
            v = torch.ones(B, 1024, device=dev)
        elif e == "noise":
            v = torch.randn(B, 1024, device=dev, generator=NOISE_G)
        else:
            v = d["Z"][:, l, :, int(e[1:])]
    else:
        op = e[0]
        if op in GR:
            if l == 0:
                v = torch.zeros(B, 1024, device=dev)
            else:
                a = ev(e[1], d, l - 1, cache)
                Wl = d["W"][:, l].float()
                if op == "Q":
                    Wl = Wl * Wl * 512.0   # E[W^2] = 2/n -> scale to O(1)
                v = torch.bmm(a[:, None, :], Wl)[:, 0] / (math.sqrt(2.0) if op == "A" else 1.0)
        elif op in UN:
            a = ev(e[1], d, l, cache)
            v = {"tanh": torch.tanh, "sq": lambda x: x * x, "abs": torch.abs,
                 "cdf": lambda x: 0.5 * (1 + torch.erf(x * 0.7071067811865476)), "neg": torch.neg}[op](a)
        else:
            a = ev(e[1], d, l, cache); b = ev(e[2], d, l, cache)
            v = a + b if op == "+" else (a - b if op == "-" else a * b)
        v = torch.nan_to_num(v.clamp(-1e4, 1e4))
    cache[key] = v
    return v


NOISE_G = torch.Generator(device=dev); NOISE_G.manual_seed(123)


def transported(e, d, perm=None):
    """v_15 of the side stream fed with es_l * z(f_l); perm: foreign W control (shuffled networks)."""
    cache = {}
    dd = d if perm is None else dict(d, W=d["W"][perm])
    v = None
    with torch.no_grad():
        for l in range(16):
            f = ev(e, d, l, cache) if perm is None else ev(e, dd, l, cache)
            f = (f - f.mean()) / (f.std() + 1e-12) * es[l]
            if v is None:
                v = f
            else:
                v = d["Phi"][:, l] * torch.bmm(v[:, None, :], d["W"][:, l].float())[:, 0] + f
    return v


def fit(e):
    v = transported(e, D["tr"]); R = D["tr"]["R"]
    c = float((v * R).sum() / ((v * v).sum() + 1e-30))
    gtr = 1.0 - float(((R - c * v) ** 2).sum() / (R * R).sum())
    return c, gtr


def val(e, c, perm=None):
    v = transported(e, D["va"], perm); R = D["va"]["R"]
    return 1.0 - float(((R - c * v) ** 2).sum() / (R * R).sum())


def mutate(e):
    if isinstance(e, str) or random.random() < 0.2:
        return rand_expr(2)
    k = random.randrange(1, len(e))
    ch = list(e); ch[k] = mutate(e[k])
    return tuple(ch)


def subtrees(e):
    yield e
    if not isinstance(e, str):
        for c in e[1:]:
            yield from subtrees(c)


def cross(a, b):
    sb = random.choice(list(subtrees(b)))
    if isinstance(a, str) or random.random() < 0.15:
        return sb
    k = random.randrange(1, len(a)); ch = list(a); ch[k] = cross(a[k], b) if random.random() < 0.5 else sb
    return tuple(ch)


def score(e):
    if size(e) > 25:
        return -1, 0, 0
    c, g = fit(e)
    return g - 2e-5 * size(e), c, g


# controls first
for name, e in (("noise", "noise"), ("u", "u"), ("x14 pred", "x14")):
    c, g = fit(e)
    print("control %-8s train %.5f val %.5f" % (name, g, val(e, c)), flush=True)

pop = [rand_expr(3) for _ in range(POP)]
seen = {}
hist = []
for gen in range(GEN):
    scored = []
    for e in pop:
        k = repr(e)
        if k not in seen:
            seen[k] = score(e)
        scored.append((seen[k][0], e))
    scored.sort(key=lambda t: -t[0])
    top = scored[:10]
    best = top[0][1]; s, c, g = seen[repr(best)]
    vb = val(best, c)
    hist.append({"gen": gen, "train": g, "val": vb, "expr": repr(best), "c": c})
    print("gen %2d train %.5f val %.5f size %d c %.3g  %s  (%.0fs)" % (gen, g, vb, size(best), c, repr(best)[:160],
          time.time() - t0), flush=True)
    elite = [e for _, e in scored[:POP // 10]]
    nxt = list(elite)
    while len(nxt) < POP:
        r = random.random()
        a = random.choice(scored[:POP // 3])[1]
        if r < 0.5:
            nxt.append(cross(a, random.choice(scored[:POP // 3])[1]))
        elif r < 0.9:
            nxt.append(mutate(a))
        else:
            nxt.append(rand_expr(3))
    pop = nxt
    json.dump(hist, open(OUT + "/hist_s%d.json" % SEED, "w"), indent=1)

# final: top 10 distinct by val, plus foreign-W control on the best
fin = []
for k, (s, c, g) in sorted(seen.items(), key=lambda t: -t[1][0])[:30]:
    e = eval(k)
    fin.append((val(e, c), g, c, k))
fin.sort(key=lambda t: -t[0])
for v, g, c, k in fin[:10]:
    print("TOP val %.5f train %.5f c %.3g %s" % (v, g, c, k[:200]), flush=True)
e = eval(fin[0][3]); perm = torch.roll(torch.arange(NVA, device=dev), 1)
print("foreign-W control on best: val %.5f" % val(e, fin[0][2], perm), flush=True)
json.dump({"hist": hist, "top": fin[:10]}, open(OUT + "/final_s%d.json" % SEED, "w"), indent=1)
