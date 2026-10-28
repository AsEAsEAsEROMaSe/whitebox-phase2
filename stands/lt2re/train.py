"""19.10.2026 white-box-16: донавчання коректора LT2 (Susupro MIT) на траєкторіях ланцюга sb2 (з FIJA), GPU.
Старт з відправлених ваг (lt2_shipped.npz). Дані: дампи dump_corpus.py (X, Phi, pred), істина m1 з шардів корпусу,
ваги з torch-сіда на CPU (1 потік). Оцінка: mini (дампи dump_mini.py), агрегатне відношення raw MSE (pred+u15)/raw (pred).
Лос (env LOSS): local = середнє по шарах ((g_l - eps_l)/es_l)^2, як у них; final = ((u_15 - d_15)/es_15)^2 (e2e);
mix = local + BETA*final. eps_l = d_l - Phi_l*(d_{l-1} @ W_l), d = truth - pred (float64, як corpus_prep.py).
ПЕРЕДБАЧЕННЯ (до запуску): e2e з warm start, lr 1e-4: mini 0.864 -> 0.85…0.86 (×1.005…1.015 у MSE); шанс ≥×1.02 ~25%.
usage: train.py <out_dir>   env: LOSS, LR, EPOCHS, BS, BETA, NTRAIN, SEED"""
import glob, json, os, sys, time, threading, queue
import numpy as np
import torch
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
from lt2torch import LT2

torch.set_num_threads(1)
dev = "cuda"
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
LOSS = os.environ.get("LOSS", "final"); LR = float(os.environ.get("LR", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "4")); BS = int(os.environ.get("BS", "4"))
BETA = float(os.environ.get("BETA", "1.0")); NTRAIN = int(os.environ.get("NTRAIN", "0"))
SEED = int(os.environ.get("SEED", "0")); EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "0"))
SH = "/home/trim/wb2data/corpus14k"; DUMP = "/home/trim/wb2data/lt2dump"
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

# truth m1 for every dumped corpus index (numeric arrays only)
loc = {}
for fn in sorted(os.listdir(SH)):
    if fn.startswith("shard") and fn.endswith(".npz"):
        with np.load(os.path.join(SH, fn), allow_pickle=False) as z:
            for r, i in enumerate(np.asarray(z["idx"], dtype=np.int64)):
                loc[int(i)] = (fn, r)
files = sorted(glob.glob(DUMP + "/corpus/corpus_*.npz"))
files = [f for f in files if not f.endswith(".tmp.npz") and int(f[-9:-4]) in loc]
if os.environ.get("EXT", "") == "unseen":
    files = [f for f in files if int(f[-9:-4]) >= 2048]
if NTRAIN:
    files = files[:NTRAIN]
need = {}
for f in files:
    need.setdefault(loc[int(f[-9:-4])][0], []).append(f)
TRUTH = {}
for fn, fs in need.items():
    with np.load(os.path.join(SH, fn), allow_pickle=False) as z:
        M = z["moments"]
        for f in fs:
            TRUTH[f] = np.asarray(M[loc[int(f[-9:-4])][1], 0], dtype=np.float32)
print("train nets %d" % len(files), flush=True)

tm = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
mfiles = sorted(f for f in glob.glob(DUMP + "/mini/mini_*.npz") if not f.endswith(".tmp.npz"))
print("mini nets %d" % len(mfiles), flush=True)


def w_corpus(idx):
    g = torch.Generator(device="cpu").manual_seed(770000 + idx)
    return torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * (2.0 / 1024) ** 0.5


def w_mini(s):
    wss, _, _ = np.random.SeedSequence(int(s)).spawn(3)
    r = np.random.default_rng(wss)
    return torch.from_numpy(np.stack([(r.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32)
                                      for _ in range(16)]))


def item(f):
    z = np.load(f)
    W = w_corpus(int(z["idx"]))
    return (torch.from_numpy(z["X"]), torch.from_numpy(z["Phi"]), W, torch.from_numpy(TRUTH[f]), torch.from_numpy(z["pred"]))


def feeder(order, q, part, nparts):
    chunks = [order[i:i + BS] for i in range(0, len(order), BS)]
    for c in chunks[part::nparts]:
        q.put([item(files[j]) for j in c])
    q.put(None)


NFEED = int(os.environ.get("NFEED", "2"))
EXT = os.environ.get("EXT", "")   # "unseen": only idx >= 2048


m = LT2("/home/trim/wb2lean/lt2re/lt2_shipped.npz", dev)
opt = torch.optim.Adam(m.parameters(), lr=LR)
es = m.es


def evaluate():
    a0 = a1 = 0.0; per = []
    m.eval()
    for f in mfiles:
        z = np.load(f); k = int(z["idx"])
        W = w_mini(z["seed"]).to(dev)
        with torch.no_grad():
            gs, us = m.run(torch.from_numpy(z["X"]).to(dev)[None], torch.from_numpy(z["Phi"]).to(dev)[None],
                           [W[l][None] for l in range(16)])
        tr = tm["all_layer_means"][k][-1].astype(np.float64); p = z["pred"][-1].astype(np.float64)
        u = us[-1][0].double().cpu().numpy()
        r0 = np.mean((p - tr) ** 2); r1 = np.mean((p + u - tr) ** 2); a0 += r0; a1 += r1; per.append(r1 / r0)
    m.train()
    return a1 / a0, per


r, per = evaluate()
log = [{"epoch": 0, "mini": r}]
print("epoch 0 mini %.5f" % r, flush=True)
best = (r, 0)
torch.save(m.state_dict(), OUT + "/ep0.pt")
step = 0
for ep in range(1, EPOCHS + 1):
    order = rng.permutation(len(files))
    q = queue.Queue(maxsize=4)
    ths = [threading.Thread(target=feeder, args=(order, q, k, NFEED), daemon=True) for k in range(NFEED)]
    for th in ths:
        th.start()
    t0 = time.time(); tl = 0.0; nb = 0; ended = 0
    while True:
        b = q.get()
        if b is None:
            ended += 1
            if ended == NFEED:
                break
            continue
        X = torch.stack([x[0] for x in b]).to(dev); Phi = torch.stack([x[1] for x in b]).to(dev)
        W = torch.stack([x[2] for x in b]).to(dev)
        T = torch.stack([x[3] for x in b]).to(dev).double(); P = torch.stack([x[4] for x in b]).to(dev).double()
        with torch.no_grad():
            d = T - P; Wd = W.double(); Phd = Phi.double()
            E = d.clone()
            for l in range(1, 16):
                E[:, l] = d[:, l] - Phd[:, l] * torch.bmm(d[:, l - 1][:, None, :], Wd[:, l])[:, 0]
            E = E.float(); D15 = d[:, 15].float()
            del Wd
        gs, us = m.run(X, Phi, [W[:, l] for l in range(16)])
        lf = (((us[-1] - D15) / es[15]) ** 2).mean()
        if LOSS == "final":
            loss = lf
        else:
            ll = sum((((gs[l] - E[:, l]) / es[l]) ** 2).mean() for l in range(16)) / 16
            loss = ll if LOSS == "local" else ll + BETA * lf
        opt.zero_grad(); loss.backward(); opt.step()
        tl += float(lf); nb += 1; step += 1
        if EVAL_EVERY and step % EVAL_EVERY == 0:
            r, _ = evaluate(); print("  step %d mini %.5f" % (step, r), flush=True)
    r, per = evaluate()
    log.append({"epoch": ep, "mini": r, "train_final": tl / max(nb, 1), "wall": time.time() - t0})
    print("epoch %d mini %.5f train_final %.4f wall %.0fs" % (ep, r, tl / max(nb, 1), time.time() - t0), flush=True)
    torch.save(m.state_dict(), OUT + "/ep%d.pt" % ep)
    if r < best[0]:
        best = (r, ep)
    json.dump({"log": log, "best": best, "env": {k: os.environ.get(k) for k in ("LOSS", "LR", "EPOCHS", "BS", "BETA", "NTRAIN", "SEED")}},
              open(OUT + "/log.json", "w"), indent=1)
print("best", best, flush=True)
