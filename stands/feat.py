"""Ознаки мереж mini-100 фази 2 (27.09) для зіставлення з помережною MSE грейдера (332445/331976, 50 публічних).
Ознаки з ваг (Фробеніус, спектральна норма, розкид норм рядків) і з МК 8192 семплів (медіана |t|, частка мертвих,
PR-ранг Cov(h) на шарах 4/8/12/15), плюс avg_variance і mean(final²) з датасету. Передбачення: MSE/mean(final²)
корелює з медіаною |t| на глибині (насиченість воріт) |r|>0.4 (шанс 40%); інакше розкид ×1.6 = випадковість реалізації."""
import glob, numpy as np, torch, pyarrow.parquet as pq, time
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
fs = sorted(glob.glob('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data/mini-*.parquet'))
out = {}
N = 8192; g = torch.Generator(device=dev); g.manual_seed(1)
for f in fs:
    t = pq.read_table(f)
    for r in range(t.num_rows):
        row = t.slice(r, 1).to_pylist()[0]
        name = row['mlp_name']; W = torch.tensor(np.array(row['weights'], dtype=np.float32), device=dev)
        fm = np.array(row['final_means'], dtype=np.float64); alm = np.array(row['all_layer_means'], dtype=np.float64)
        L, n, _ = W.shape
        fro = W.pow(2).sum((1, 2)).sqrt().cpu().numpy(); spec = torch.linalg.matrix_norm(W, ord=2).cpu().numpy()
        rown = W.pow(2).sum(2).sqrt(); rcv = (rown.std(1) / rown.mean(1)).cpu().numpy(); coln = W.pow(2).sum(1).sqrt(); ccv = (coln.std(1) / coln.mean(1)).cpu().numpy()
        z = torch.randn(N, n, device=dev, generator=g); p = z @ W[0]
        medt, dead, pr, sk = [], [], [], []
        for l in range(L):
            mu = p.mean(0); sd = p.std(0); medt.append(float((mu / sd).abs().median())); dead.append(float(((p > 0).float().mean(0) < 1.0 / N).float().mean()))
            if l in (4, 8, 12, 15):
                h = torch.relu(p); hc = h - h.mean(0); C = (hc.T @ hc) / N; ev = torch.linalg.eigvalsh(C); pr.append(float(ev.sum() ** 2 / (ev * ev).sum()))
            if l + 1 < L: p = torch.relu(p) @ W[l + 1]
        out[name] = dict(avg_var=row['avg_variance'], fm2=float((fm ** 2).mean()), fro=fro, spec=spec, rcv=rcv, ccv=ccv, medt=np.array(medt), dead=np.array(dead), pr=np.array(pr),
                         alm2=(alm ** 2).mean(1))
        print(name, "medt15 %.2f dead15 %.3f pr %s fm2 %.3f avg_var %.3f" % (medt[-1], dead[-1], np.round(pr, 0), out[name]['fm2'], row['avg_variance']), flush=True)
np.save('/home/trim/wb2data/feat_mini100.npy', out, allow_pickle=True)
print("DONE_FEAT")
