"""13.10.2026: аналіз слідів стовпців хабу (hubfd5_{I}.npz)."""
import sys, numpy as np
rng = np.random.default_rng(0)
def frac_in_span(F, e):
    Q, _ = np.linalg.qr(F.T); p = Q @ (Q.T @ e); return float((p @ p) / (e @ e))
for I in [int(x) for x in sys.argv[1:]] or [0, 1, 2]:
    try: d = np.load(f"/home/trim/wb2data/hubfd5_{I}.npz")
    except FileNotFoundError: continue
    pb, gt, n = d["pred_base"], d["gt"], int(d["n"]); e = pb - gt; base = float(d["base"])
    print(f"\n=== дамп {I}  base {base:.4e}  ‖e‖²/n {e@e/n:.4e}")
    Fall = []
    for lf in (3, 6, 12):
        if f"pred_{lf}" not in d.files: continue
        F = pb[None, :] - d[f"pred_{lf}"]; Fall.append(F); m = F.shape[0]
        dl = ((d[f"pred_{lf}"] - gt) ** 2).mean(1) - base
        # 1. частка похибки у спані слідів; контроль: випадкові m напрямків і перемішані сліди (компоненти переставлені)
        fs = frac_in_span(F, e); fr = frac_in_span(rng.standard_normal((m, n)), e)
        Fp = np.array([rng.permutation(f) for f in F]); fp = frac_in_span(Fp, e)
        # 2. оракули: МНК-перезваження, лише знак (викинути шкідливі), ридж по LOO-стовпцях не треба — це стеля
        c, *_ = np.linalg.lstsq(F.T, e, rcond=None); mse_ls = float(((e - F.T @ c) ** 2).mean())
        neg = dl < 0; mse_drop = float(((e - F[neg].sum(0)) ** 2).mean())
        # 3. скільки похибки у топ-k сингулярних напрямках слідів
        U, S, Vt = np.linalg.svd(F, full_matrices=False); ek = (Vt @ e) ** 2 / (e @ e)
        # 4. Грам і знак
        nf = np.linalg.norm(F, axis=1); R = (F @ F.T) / np.outer(nf, nf); np.fill_diagonal(R, 0)
        col = np.abs(R).mean(1); net = R.sum(1); mx = np.abs(R).max(1)
        a = (F @ e) / nf ** 2  # ⟨e,f_j⟩/‖f_j‖²; шкідливий iff a > 1/2
        cs = lambda x, y: np.corrcoef(x, y)[0, 1]
        canc = np.linalg.norm(F.sum(0)) / nf.sum()
        print(f"L{lf}: шкідливих {neg.sum()}/{m} · частка e у спані {fs:.3f} (випадк {fr:.3f}, переміш {fp:.3f}) · топ-8/16 синг {ek[:8].sum():.3f}/{ek[:16].sum():.3f}")
        print(f"      оракул МНК c_j: MSE ×{mse_ls/base:.3f} · лише знак (викинути шкідливі): ×{mse_drop/base:.3f} · c_j: медіана {np.median(c):+.2f} IQR [{np.percentile(c,25):+.2f},{np.percentile(c,75):+.2f}] · ‖Σf‖/Σ‖f‖ {canc:.3f}")
        print(f"      corr(a, колінеарність) {cs(a,col):+.3f} · corr(a, net R) {cs(a,net):+.3f} · corr(a, max|R|) {cs(a,mx):+.3f} · corr(c, net R) {cs(c,net):+.3f} · corr(c, a) {cs(c,a):+.3f} · corr(a,‖f‖) {cs(a,nf):+.3f}")
        print(f"      sign(шкідл) vs net R>0: точність {np.mean((net>0)==neg):.2f} · vs col>медіана: {np.mean((col>np.median(col))==neg):.2f}")
    if len(Fall) > 1:
        F = np.vstack(Fall); m = F.shape[0]
        fs = frac_in_span(F, e); fr = frac_in_span(rng.standard_normal((m, n)), e)
        c, *_ = np.linalg.lstsq(F.T, e, rcond=None); mse_ls = float(((e - F.T @ c) ** 2).mean())
        print(f"усі {m}: частка e у спані {fs:.3f} (випадк {fr:.3f}) · оракул МНК ×{mse_ls/base:.3f}")
