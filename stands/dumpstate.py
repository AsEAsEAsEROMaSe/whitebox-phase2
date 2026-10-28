"""Збір стану ланцюга REG (рекордна конфігурація) для стенду масштабу даних (27.09).
Для кожної мережі: pred усіх шарів (16,1024); понейронні ознаки на шарах 10..15 з пре-стану (D21_HOOK: mu, var, D3, D21)
і пост-стану (K_POST_HOOK: mu_post, C_post); ознаки W; для l=15 повні C_pre і D21_pre у fp16 (пари для GNN пізніше).
Режими: mini (parquet mini-100, з істиною 1e9) · gen START END (мережі gen_common, істина з gen_truth).
Ціль пілоту: крива навчання R²_test(похибка виходу) від кількості мереж 20/40/80 на mini-100. Передбачення: R² ≤0.05 і
плоско (шанс 65%) ⟹ масштаб даних не важіль; якщо росте з n ⟹ гнати gen-мережі."""
import os, sys, time, glob, numpy as np, torch
_argv = sys.argv; sys.argv = sys.argv[:1]   # lean_oracle парсить argv при імпорті
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk; from gen_common import make_w
sys.argv = _argv
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
OUT = "/home/trim/wb2data/gen/state_%s.npz"; LAYS = set(range(10, 16)); dt = torch.float32
def rowfeat(M):   # понейронні агрегати матриці n×n (рядок = нейрон)
    return torch.stack([M.sum(1), M.abs().sum(1), (M * M).sum(1), M.T.sum(1), (M.T * M.T).sum(1)], 1)
class Cap:
    def __init__(self): self.pre = {}; self.pst = {}; self.big = {}
    def d21(self, l, D3, D21, live):
        if l in LAYS:
            mu, C, var = live["mu"], live["C"], live["var"]; Coff = C.clone(); Coff.fill_diagonal_(0.0)
            f = [mu, var, mu / var.sqrt(), D3, D3 / var ** 1.5, Coff @ mu, (Coff * Coff).sum(1), Coff.sum(1), (Coff @ (Coff @ torch.ones_like(mu))),
                 D21.sum(1), D21.abs().sum(1), D21.T.sum(1), (D21 * Coff).sum(1), (D21.T * Coff).sum(1), D21 @ mu, D21.T @ mu]
            self.pre[l] = torch.stack(f, 1).detach().float().cpu().numpy()
            if l == 15: self.big["C15"] = C.detach().half().cpu().numpy(); self.big["D21_15"] = D21.detach().half().cpu().numpy()
        return D3, D21, None
    def post(self, l, K, mu, C):
        if l in LAYS:
            Coff = C.clone(); Coff.fill_diagonal_(0.0); var = torch.diagonal(C)
            self.pst[l] = torch.stack([mu, var, (Coff * Coff).sum(1), Coff.sum(1), Coff @ mu], 1).detach().float().cpu().numpy()
            if l == 15: self.big["Cpost15"] = C.detach().half().cpu().numpy()
def run(name, w, gt=None):
    if os.path.exists(OUT % name): return
    t0 = time.time(); wt = torch.as_tensor(w) if not torch.is_tensor(w) else w; wn = wt.numpy() if wt.device.type == "cpu" else wt.cpu().numpy()
    st = lk.build_static_coefs(wn.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K_POST_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(wn, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    cap = Cap(); lk.D21_HOOK = cap.d21; lk.K_POST_HOOK = cap.post; lk.REGEN_STATE.clear()
    hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
    p = lk.lean_k3_predict(wn, dtype=dt, statics=st, **hk, **REG).detach().cpu().double().numpy()
    lk.D21_HOOK = None; lk.K_POST_HOOK = None
    W = torch.as_tensor(wn, device=lk.DEV)
    wf = torch.stack([W.pow(2).sum(2).sqrt(), W.pow(2).sum(1).sqrt(), W.sum(2), W.sum(1)], 2).cpu().numpy()   # (L, n, 4): норми/суми рядків і стовпців
    d = dict(pred=p, lam=np.array(fit.coef), wfeat=wf, name=name, **{"pre%d" % l: v for l, v in cap.pre.items()}, **{"post%d" % l: v for l, v in cap.pst.items()}, **cap.big)
    if gt is not None: d["gt"] = gt
    tmp = (OUT % name) + ".tmp.npz"; np.savez(tmp, **d); os.replace(tmp, OUT % name)
    msg = "" if gt is None else "  mse %.3e" % ((p[-1] - gt[-1]) ** 2).mean()
    print("state %-22s %.0f с%s" % (name, time.time() - t0, msg), flush=True); torch.cuda.empty_cache()
mode = sys.argv[1]; fails = 0
if mode == "mini":
    import pyarrow.parquet as pq
    fs = sorted(glob.glob('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data/mini-*.parquet'))
    for f in fs:
        t = pq.read_table(f)
        for r in range(t.num_rows):
            row = t.slice(r, 1).to_pylist()[0]
            try: run(row['mlp_name'], np.array(row['weights'], dtype=np.float32), np.array(row['all_layer_means'], dtype=np.float64)); fails = 0
            except Exception as e:
                fails += 1; print("FAIL", row['mlp_name'], repr(e), flush=True); torch.cuda.empty_cache()
                if fails >= 3: sys.exit(1)
else:
    for i in range(int(sys.argv[2]), int(sys.argv[3])):
        try: run("g%05d" % i, make_w(i)); fails = 0
        except Exception as e:
            fails += 1; print("FAIL g%05d %r" % (i, e), flush=True); torch.cuda.empty_cache()
            if fails >= 3: sys.exit(1)
print("DONE_STATE")
