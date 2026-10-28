import sys, numpy as np
for I in [int(x) for x in sys.argv[1:]] or [0,1,2]:
    try: d = np.load(f"/home/trim/wb2data/hubfd5_{I}.npz")
    except FileNotFoundError: continue
    pb, gt, n = d["pred_base"], d["gt"], int(d["n"]); e = pb - gt; base = float(d["base"]); ee = e @ e
    print(f"=== дамп {I}")
    for lf in (3, 6, 12):
        if f"pred_{lf}" not in d.files: continue
        F = pb[None, :] - d[f"pred_{lf}"]; S = F.sum(0); m = F.shape[0]
        s = (e @ S) / (S @ S)                      # найкращий спільний масштаб підмножини (сума слідів), s>0 ⟹ зменшити народження
        mse_s = ((e - s * S) @ (e - s * S)) / ee
        c, *_ = np.linalg.lstsq(F.T, e, rcond=None); se = c.std() / np.sqrt(m)
        Q, _ = np.linalg.qr(F.T); ein = Q @ (Q.T @ e)
        cosS = (ein @ S) / np.linalg.norm(ein) / np.linalg.norm(S)
        print(f"L{lf}: ‖Σf‖²/‖e‖² {S@S/ee:.3f} · ‖e−Σf‖²/‖e‖² {((e-S)@(e-S))/ee:.3f} (викинути всі {m} разом, адитивно) · спільний масштаб s*={s:+.3f} ⟹ MSE ×{mse_s:.4f} · mean c {c.mean():+.3f}±{se:.3f} · cos(e_span, Σf) {cosS:+.3f}")
        # екстраполяція: частка e у спані k стовпців, k=8,16,32,64 (середнє по випадкових піднаборах)
        rng = np.random.default_rng(1); out = []
        for k in (8, 16, 32, 64):
            fr = []
            for _ in range(8):
                idx = rng.choice(m, k, replace=False); Qk, _ = np.linalg.qr(F[idx].T); p = Qk @ (Qk.T @ e); fr.append(p @ p / ee)
            out.append(f"k={k}: {np.mean(fr):.3f} (вип {k/n:.3f})")
        print("      частка у спані: " + " · ".join(out))
