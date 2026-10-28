"""21.10.2026 white-box-b9 (= f9): prep дампу ALG058 (s32refire з хуками f0) у memmap для lt2_train_seed.py.
Ціль: залишок ПІСЛЯ власного коректора ALG058 (nn_asset): d_l = truth_l - (pred_l + res_e_l) для l < 15,
d_15 = truth_15 - pred_15 (останній рядок уже містить res_e і max(.,0)). eps_l = d_l - Phi_l * (d_{l-1} @ W_l).
Ваги на диск не пишуться (32 МБ на мережу): тренер відтворює їх із сіду 770000+idx.
Спліти: full = усі, крім останніх NHO файлів; mini = останні NHO (held-out).
usage: prep_alg.py <dump_dir> <prep_dir> [--nho 48]"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(2)
L, N = 16, 1024
SCALE = (2.0 / 1024.0) ** 0.5
ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("prep"); ap.add_argument("--nho", type=int, default=48); ap.add_argument("--rm", action="store_true")   # delete each dump after copy (disk)
a = ap.parse_args(); os.makedirs(a.prep, exist_ok=True)
files = sorted(f for f in glob.glob(os.path.join(a.dump, "corpus_*.npz")) if not f.endswith(".tmp.npz"))
truth = {}
for fn in sorted(os.listdir("/home/trim/wb2data/corpus14k")):
    if fn.startswith("shard") and fn.endswith(".npz"):
        with np.load(os.path.join("/home/trim/wb2data/corpus14k", fn), allow_pickle=False) as z:
            mom = np.asarray(z["moments"][:, 0], dtype=np.float32)
            for r, i in enumerate(np.asarray(z["idx"], dtype=np.int64)):
                truth[int(i)] = mom[r]
splits = {"full": files[:-a.nho], "mini": files[-a.nho:]}
for s, fl in splits.items():
    M = len(fl); F = np.load(fl[0])["X"].shape[-1]
    X = np.lib.format.open_memmap(f"{a.prep}/{s}_X.npy", mode="w+", dtype=np.float32, shape=(M, L, N, F))
    arr = {k: np.zeros((M, L, N), dtype=np.float32) for k in ("Phi", "P", "T", "E")}
    idx = []
    for m, f in enumerate(fl):
        d = np.load(f); k = int(d["idx"]); idx.append(k)
        pred = d["pred"].astype(np.float64); Phi = d["Phi"]
        if "res_e" in d.files:
            pred[:L - 1] += d["res_e"][:L - 1].astype(np.float64)
        T = truth[k].astype(np.float64); dl = T - pred
        g = torch.Generator(device="cpu").manual_seed(770000 + k)
        W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE).numpy()
        eps = dl.copy()
        for l in range(1, L):
            eps[l] = dl[l] - Phi[l] * (dl[l - 1] @ W[l])
        X[m] = d["X"]; arr["Phi"][m] = Phi; arr["P"][m] = pred; arr["T"][m] = T; arr["E"][m] = eps
        if a.rm: os.remove(f)
        if m % 50 == 0: print(s, m, M, k, "raw15 %.3e" % float(np.mean(dl[-1] ** 2)), flush=True)
    X.flush(); del X
    for kk, v in arr.items(): np.save(f"{a.prep}/{s}_{kk}.npy", v)
    json.dump({"idx": idx, "seeds": [770000 + i for i in idx]}, open(f"{a.prep}/{s}_meta.json", "w"))
print("PREPDONE", {s: len(v) for s, v in splits.items()}, flush=True)
