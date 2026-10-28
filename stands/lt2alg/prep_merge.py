"""21.10.2026 white-box-b9: злиття старого prep ALG058 (952 full + 48 mini, X 51 колонка) з новим дампом
(корпус 1000+, X 53 колонки з LT_FE3, беремо перші 51 = старі біт у біт за пломбою a0) у новий prep для lt2_train_seed.py.
full = старий full + усі нові дампи; mini = старий mini (той самий held-out, щоб порівняння з 0.9535 було чесним).
Ціль та сама, що в prep_alg.py. usage: prep_merge.py <old_prep> <dump_dir> <new_prep> [--max N]"""
import argparse, glob, json, os, shutil, sys
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(2)
L, N, F = 16, 1024, 51
SCALE = (2.0 / 1024.0) ** 0.5
ap = argparse.ArgumentParser(); ap.add_argument("old"); ap.add_argument("dump"); ap.add_argument("prep"); ap.add_argument("--max", type=int, default=100000)
a = ap.parse_args(); os.makedirs(a.prep, exist_ok=True)
files = sorted(f for f in glob.glob(os.path.join(a.dump, "corpus_*.npz")) if not f.endswith(".tmp.npz"))[:a.max]
truth = {}
for fn in sorted(os.listdir("/home/trim/wb2data/corpus14k")):
    if fn.startswith("shard") and fn.endswith(".npz"):
        with np.load(os.path.join("/home/trim/wb2data/corpus14k", fn), allow_pickle=False) as z:
            mom = np.asarray(z["moments"][:, 0], dtype=np.float32)
            for r, i in enumerate(np.asarray(z["idx"], dtype=np.int64)):
                truth[int(i)] = mom[r]
for fn in os.listdir(a.old):
    if fn.startswith("mini_"):
        shutil.copyfile(os.path.join(a.old, fn), os.path.join(a.prep, fn))
oX = np.load(f"{a.old}/full_X.npy", mmap_mode="r"); M0 = oX.shape[0]; assert oX.shape[-1] == F
om = json.load(open(f"{a.old}/full_meta.json"))
M = M0 + len(files)
X = np.lib.format.open_memmap(f"{a.prep}/full_X.npy", mode="w+", dtype=np.float32, shape=(M, L, N, F))
for i in range(0, M0, 50):
    X[i:min(i + 50, M0)] = oX[i:min(i + 50, M0)]
arr = {k: np.zeros((M, L, N), dtype=np.float32) for k in ("Phi", "P", "T", "E")}
for k in arr:
    arr[k][:M0] = np.load(f"{a.old}/full_{k}.npy")
idx = list(om["idx"])
for m, f in enumerate(files, start=M0):
    d = np.load(f); k = int(d["idx"]); assert k not in idx; idx.append(k)
    pred = d["pred"].astype(np.float64); Phi = d["Phi"]
    pred[:L - 1] += d["res_e"][:L - 1].astype(np.float64)
    T = truth[k].astype(np.float64); dl = T - pred
    g = torch.Generator(device="cpu").manual_seed(770000 + k)
    W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE).numpy()
    eps = dl.copy()
    for l in range(1, L):
        eps[l] = dl[l] - Phi[l] * (dl[l - 1] @ W[l])
    X[m] = d["X"][..., :F]; arr["Phi"][m] = Phi; arr["P"][m] = pred; arr["T"][m] = T; arr["E"][m] = eps
    if m % 100 == 0: print("full", m, M, k, "raw15 %.3e" % float(np.mean(dl[-1] ** 2)), flush=True)
X.flush(); del X
for kk, v in arr.items(): np.save(f"{a.prep}/full_{kk}.npy", v)
json.dump({"idx": idx, "seeds": [770000 + i for i in idx]}, open(f"{a.prep}/full_meta.json", "w"))
print("MERGEDONE full", M, "mini", len(json.load(open(f"{a.prep}/mini_meta.json"))["idx"]), flush=True)
