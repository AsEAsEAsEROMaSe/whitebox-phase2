"""Bake O(W^2) higher-moment ground truth for the whestbench MLPs.

For each official MLP (width 256, depth 32, bias-free ReLU, He-init) we forward
N i.i.d. N(0, I) samples and accumulate, per layer (post-ReLU), the moment-tensor
components that are O(W^2) in storage — i.e. those indexed by <= 2 distinct neuron
indices. The full rank-3/rank-4 tensors (O(W^3), O(W^4)) are deliberately skipped.

Per layer we store (raw, non-central moments):
  mean   (W,)      E[x_i]
  m2,m3,m4 (W,)    E[x_i^2], E[x_i^3], E[x_i^4]        (diagonals)
  M11   (W, W)     E[x_i x_j]                          (1,1)
  M21   (W, W)     E[x_i^2 x_j]                        (2,1)   note: i is the squared index
  M22   (W, W)     E[x_i^2 x_j^2]                      (2,2)
  M31   (W, W)     E[x_i^3 x_j]                        (3,1)   note: i is the cubed index

Connected/central versions are a cheap deterministic transform downstream — see
`connected()` at the bottom of this file. We store raw moments because (a) they are
all-positive and well-conditioned to accumulate, and (b) it keeps every connected /
central / cumulant combination recoverable.

Precision: each chunk's moment products are fp32 GEMMs (tensor cores; tf32 OFF — its
10-bit mantissa is above the MC noise floor and would corrupt the moments). The
cross-chunk running sums live in fp64, because a pure-fp32 accumulator holding the
sum of ~1e8-1e9 positive terms reaches a magnitude where its ULP swallows per-chunk
contributions. fp64 *GEMMs* are NOT used: they run at ~1/64 rate on this card and the
MC sampling floor (~1/sqrt(N)) sits well above the fp32 arithmetic floor anyway.

Work distribution: each MLP is an independent unit of work claimed via an atomic lock
file in the output dir. Launch one process per GPU (bind with CUDA_VISIBLE_DEVICES);
they self-balance and the run is resumable — re-launching skips finished MLPs and
reclaims locks left by crashed workers. Works across machines on a shared filesystem.

Examples
--------
  # Local smoke test (offline) against the cached official_mini.npz, 2 MLPs, small N:
  uv run python research/bake_moments.py --source npz \
      --npz /workspace/whest-2/research/data/official_mini.npz \
      --out /tmp/moments_test --n-samples 2_000_000 --limit 2 --verify

  # Production, one process per GPU (run this once per visible device):
  CUDA_VISIBLE_DEVICES=0 uv run python research/bake_moments.py \
      --source hf --split full --out /data/whest_moments --n-samples 100_000_000 &
  CUDA_VISIBLE_DEVICES=1 uv run python research/bake_moments.py \
      --source hf --split full --out /data/whest_moments --n-samples 100_000_000 &
  # ... add more GPUs / more machines (shared --out) at will.
"""
from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

import numpy as np

WIDTH, DEPTH = 256, 32
HF_REPO = "aicrowd/arc-whestbench-public-2026"
HF_REVISION = "v1-phase1"
N_SHARDS = {"mini": 3, "full": 28}


# --------------------------------------------------------------------------- #
# MLP sources: yield (global_index, name, weights[32,256,256] f32, official_alm or None)
# --------------------------------------------------------------------------- #
def iter_npz(path: str):
    d = np.load(path)
    W = d["W"]  # (n,32,256,256)
    alm = d["ALM"] if "ALM" in d.files else None
    names = d["names"] if "names" in d.files else None
    for i in range(W.shape[0]):
        nm = str(names[i]) if names is not None else f"mlp_{i:05d}"
        yield i, nm, np.asarray(W[i], dtype=np.float32), (None if alm is None else np.asarray(alm[i]))


def iter_hf(split: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def fetch(shard, tries=6):
        for t in range(tries):
            try:
                return hf_hub_download(HF_REPO, shard, revision=HF_REVISION, repo_type="dataset")
            except Exception as e:  # noqa: BLE001
                if t == tries - 1:
                    raise
                print(f"  retry {shard} ({t + 1}): {type(e).__name__}", flush=True)
                time.sleep(5 * (t + 1))

    gidx = 0
    n = N_SHARDS[split]
    for s in range(n):
        shard = f"data/{split}-{s:05d}-of-{n:05d}.parquet"
        tbl = pq.read_table(fetch(shard), columns=["weights", "all_layer_means"])
        Wc, Mc = tbl.column("weights"), tbl.column("all_layer_means")
        for i in range(tbl.num_rows):
            Ws = np.asarray(Wc[i].as_py(), dtype=np.float32)         # (32,256,256)
            alm = np.asarray(Mc[i].as_py(), dtype=np.float64)        # (32,256)
            yield gidx, f"{split}_{gidx:05d}", Ws, alm
            gidx += 1
        del tbl, Wc, Mc


# --------------------------------------------------------------------------- #
# Core: forward + moment accumulation for one MLP
# --------------------------------------------------------------------------- #
def bake_one(weights_np, n_samples, chunk, device, seed, do_pre=False, do_post=True):
    """Accumulate O(W^2) moments per layer. `do_post` -> moments of y=relu(z) (keys
    mean/M11/.../m4); `do_pre` -> moments of the pre-activation z=x@W (keys prefixed
    `pre_`). The forward pass is shared, so computing both costs ~1 extra GEMM set.

    pre and post are accumulated from the SAME samples (same seed+chunk reproduces the
    identical RNG path), so a pre-only catch-up pass on an already-baked MLP yields pre
    moments coupled to its existing post moments.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # ground truth: real fp32, never tf32
    torch.backends.cudnn.allow_tf32 = False

    Ws = [torch.from_numpy(weights_np[k]).to(device) for k in range(DEPTH)]  # f32 (256,256)
    f64 = torch.float64
    z2 = lambda: torch.zeros(DEPTH, WIDTH, device=device, dtype=f64)
    zM = lambda: torch.zeros(DEPTH, WIDTH, WIDTH, device=device, dtype=f64)
    mk = lambda: dict(mean=z2(), M11=zM(), M21=zM(), M22=zM(), M31=zM())
    post = mk() if do_post else None
    pre = mk() if do_pre else None

    def accumulate(acc, k, v):                 # v: (b, W) activations (pre z or post y)
        v2 = v * v
        v3 = v2 * v
        acc["mean"][k] += v.sum(0, dtype=torch.float32).double()
        acc["M11"][k] += (v.t() @ v).double()
        acc["M21"][k] += (v2.t() @ v).double()
        acc["M22"][k] += (v2.t() @ v2).double()
        acc["M31"][k] += (v3.t() @ v).double()

    gen = torch.Generator(device=device).manual_seed(seed)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        x = torch.randn(b, WIDTH, device=device, dtype=torch.float32, generator=gen)
        for k in range(DEPTH):
            z = x @ Ws[k]                      # pre-activation
            if do_pre:
                accumulate(pre, k, z)
            x = torch.relu(z)                  # post-activation
            if do_post:
                accumulate(post, k, x)
        done += b

    di = np.arange(WIDTH)
    out = {}
    for prefix, acc in (("", post), ("pre_", pre)):
        if acc is None:
            continue
        m = {kk: (vv / n_samples).to(torch.float32).cpu().numpy() for kk, vv in acc.items()}
        m["m2"] = m["M11"][:, di, di]   # E[v^2]
        m["m3"] = m["M21"][:, di, di]   # E[v^2 * v] = E[v^3]
        m["m4"] = m["M22"][:, di, di]   # E[v^2 * v^2] = E[v^4]
        out.update({prefix + kk: vv for kk, vv in m.items()})
    return out


# --------------------------------------------------------------------------- #
# Atomic claim / write
# --------------------------------------------------------------------------- #
def claim(lockpath: Path, reclaim_after: float) -> bool:
    """Try to atomically claim work. Returns True if we own it."""
    try:
        fd = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{socket.gethostname()}:{os.getpid()}:{int(time.time())}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - lockpath.stat().st_mtime
        except FileNotFoundError:
            return False
        if age > reclaim_after:                       # stale lock from a dead worker
            try:
                lockpath.unlink()
                return claim(lockpath, reclaim_after)
            except FileNotFoundError:
                return False
        return False


def save_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "npz"], default="hf")
    ap.add_argument("--split", choices=["mini", "full"], default="full")
    ap.add_argument("--npz", help="path to official_*.npz when --source npz")
    ap.add_argument("--out", required=True, help="output dir (shared across workers)")
    ap.add_argument("--n-samples", type=lambda s: int(float(s.replace("_", ""))), default=100_000_000)
    ap.add_argument("--chunk", type=lambda s: int(float(s.replace("_", ""))), default=2_000_000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed-base", type=int, default=0, help="per-MLP RNG seed = seed_base + global_index")
    ap.add_argument("--limit", type=int, default=None, help="process at most this many MLPs (testing)")
    ap.add_argument("--reclaim-after", type=float, default=3600.0, help="steal locks older than this many s")
    ap.add_argument("--moments", choices=["post", "pre", "both"], default="both",
                    help="which activation statistics to ensure present (default both). "
                         "Already-present components are kept; only the missing ones are computed.")
    ap.add_argument("--verify", action="store_true", help="compare baked mean vs official all_layer_means")
    args = ap.parse_args()
    want_post = args.moments in ("post", "both")
    want_pre = args.moments in ("pre", "both")

    split = args.split if args.source == "hf" else "npz"
    outdir = Path(args.out) / split
    outdir.mkdir(parents=True, exist_ok=True)

    src = iter_npz(args.npz) if args.source == "npz" else iter_hf(args.split)

    t_start = time.time()
    n_done = n_seen = 0
    for gidx, name, Ws, alm in src:
        if args.limit is not None and n_seen >= args.limit:
            break
        n_seen += 1
        out_path = outdir / f"mlp_{gidx:05d}.npz"
        # Decide what's missing for this MLP. Existing components are reused.
        have_post = have_pre = False
        if out_path.exists():
            try:
                have = set(np.load(out_path).files)
                have_post, have_pre = "M11" in have, "pre_M11" in have
            except Exception:  # noqa: BLE001 — corrupt/partial file: re-bake from scratch
                have_post = have_pre = False
        need_post = want_post and not have_post
        need_pre = want_pre and not have_pre
        if not (need_post or need_pre):
            continue
        lock_path = outdir / f"mlp_{gidx:05d}.lock"
        if not claim(lock_path, args.reclaim_after):
            continue
        try:
            t0 = time.time()
            res = bake_one(Ws, args.n_samples, args.chunk, args.device,
                           args.seed_base + gidx, do_pre=need_pre, do_post=need_post)
            # merge into any existing payload so we never drop already-computed parts
            payload = dict(np.load(out_path)) if out_path.exists() else dict(
                global_index=gidx, name=name, n_samples=args.n_samples, width=WIDTH, depth=DEPTH)
            payload.update(res)
            mean_match = np.nan
            if alm is not None:
                payload["official_alm"] = alm.astype(np.float32)
                if need_post:
                    mean_match = float(np.abs(res["mean"] - alm).max())
            save_atomic(out_path, payload)
            n_done += 1
            tag = "+".join([s for s, n in (("post", need_post), ("pre", need_pre)) if n])
            msg = f"[{args.device}] mlp {gidx:05d} ({name}) {tag} in {time.time() - t0:5.1f}s"
            if not np.isnan(mean_match):
                msg += f"  | mean vs official max|Δ|={mean_match:.2e}"
            print(msg, flush=True)
        finally:
            lock_path.unlink(missing_ok=True)

    print(f"[{args.device}] done: baked {n_done} MLP(s) this process in "
          f"{time.time() - t_start:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Downstream helper: raw moments -> connected (cumulant) components, in fp64.
# --------------------------------------------------------------------------- #
def connected(npz, prefix=""):
    """Given a baked .npz (or dict), return per-layer connected moment matrices (fp64).

    Pass prefix="" for post-activation moments (default) or prefix="pre_" for the
    pre-activation ones. For pre-activations the cumulants c21/c22/c31 are exactly the
    leading non-Gaussianity (they vanish for a Gaussian pre-activation).

    cov   (D,W,W)  Cov(x_i, x_j)                       = E[x_i x_j] - E[x_i]E[x_j]
    c21   (D,W,W)  cumulant kappa(x_i,x_i,x_j)         (the (2,1) connected piece)
    c22   (D,W,W)  cumulant kappa(x_i,x_i,x_j,x_j)     (the (2,2) connected piece)
    c31   (D,W,W)  cumulant kappa(x_i,x_i,x_i,x_j)     (the (3,1) connected piece)
    """
    g = (lambda k: np.asarray(npz[prefix + k], dtype=np.float64))
    mu, m2, m3 = g("mean"), g("m2"), g("m3")
    M11, M21, M22, M31 = g("M11"), g("M21"), g("M22"), g("M31")
    M12 = M21.transpose(0, 2, 1)                       # E[x_i x_j^2] (swap squared index)
    mi, mj = mu[:, :, None], mu[:, None, :]            # i along axis 1, j along axis 2
    m2i, m2j, m3i = m2[:, :, None], m2[:, None, :], m3[:, :, None]

    cov = M11 - mi * mj
    var = np.diagonal(cov, axis1=1, axis2=2)           # (D,W)
    vi, vj = var[:, :, None], var[:, None, :]

    # (2,1) central = cumulant kappa(x_i,x_i,x_j) = E[(xi-mi)^2 (xj-mj)]
    c21 = M21 - 2 * mi * M11 - m2i * mj + 2 * mi * mi * mj
    # (2,2): central E[(xi-mi)^2 (xj-mj)^2], then cumulant subtracts var_i var_j + 2 cov^2
    cen22 = (M22 - 2 * mj * M21 - 2 * mi * M12
             + mj * mj * m2i + mi * mi * m2j + 4 * mi * mj * M11 - 3 * mi * mi * mj * mj)
    c22 = cen22 - vi * vj - 2 * cov * cov
    # (3,1): central E[(xi-mi)^3 (xj-mj)], then cumulant subtracts 3 var_i cov_ij
    cen31 = (M31 - mj * m3i - 3 * mi * M21 + 3 * mi * mj * m2i
             + 3 * mi * mi * M11 - 3 * mi ** 3 * mj)
    c31 = cen31 - 3 * vi * cov
    return dict(cov=cov, c21=c21, c22=c22, c31=c31)


def third_order_blocks(npz, weights):
    """The n^2-vs-n^3 tractability experiment, computed entirely from a baked pre+post
    .npz plus the MLP weights (no n^3 tensor ever materialized).

    For each layer ell (0..D-2) the next pre-activation is z' = h^{(ell)} @ W^{(ell+1)}.
    By cumulant multilinearity kappa3(z'_a) = sum_ijk W_ia W_ja W_ka kappa3[h_i,h_j,h_k],
    and that full contraction is *identically* the baked pre-activation skew at ell+1.
    We split it by index-coincidence into the storable blocks and the n^3 remainder:

        full   (D-1, W)  kappa3(z')            = pre-activation 3rd cumulant at ell+1
        b3     (D-1, W)  (3) diagonal block    = sum_i W_ia^3 mu3_i
        b21    (D-1, W)  (2,1) block           = 3 sum_{i!=k} W_ia^2 W_ka c21[i,k]
        b111   (D-1, W)  (1,1,1) block         = full - b3 - b21   (the n^3 contribution)

    `weights` is the MLP's (D, W, W) array (same orientation as the forward pass,
    pre^{(k)} = post^{(k-1)} @ weights[k]). Returns the four (D-1, W) arrays in fp64.
    """
    g = lambda k: np.asarray(npz[k], dtype=np.float64)
    Wt = np.asarray(weights, dtype=np.float64)
    post = connected(npz, prefix="")                       # c21 at every layer
    c21_all = post["c21"]                                  # (D, W, W)
    # post marginal 3rd cumulant (= central 3rd moment) per neuron, per layer
    pm1, pm2, pm3 = g("mean"), g("m2"), g("m3")
    mu3_post = pm3 - 3 * pm1 * pm2 + 2 * pm1 ** 3           # (D, W)
    # full = pre-activation 3rd cumulant per neuron, per layer
    qm1, qm2, qm3 = g("pre_mean"), g("pre_m2"), g("pre_m3")
    kappa3_pre = qm3 - 3 * qm1 * qm2 + 2 * qm1 ** 3         # (D, W)

    D = mu3_post.shape[0]
    full, b3, b21, b111 = ([] for _ in range(4))
    for ell in range(D - 1):
        W = Wt[ell + 1]                                    # (W_in=i, W_out=a)
        c21 = c21_all[ell]                                 # (i, k)
        mu3 = mu3_post[ell]                                # (i,)
        blk3 = (W ** 3 * mu3[:, None]).sum(0)              # (a,)
        CW = c21 @ W                                       # (i, a) = sum_k c21[i,k] W_ka
        unrestricted = ((W ** 2) * CW).sum(0)              # (a,)
        diag_overlap = (W ** 3 * np.diagonal(c21)[:, None]).sum(0)
        blk21 = 3.0 * (unrestricted - diag_overlap)
        f = kappa3_pre[ell + 1]
        full.append(f); b3.append(blk3); b21.append(blk21); b111.append(f - blk3 - blk21)
    return dict(full=np.array(full), b3=np.array(b3), b21=np.array(b21), b111=np.array(b111))


if __name__ == "__main__":
    main()
