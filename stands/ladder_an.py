import numpy as np, glob
fs = sorted(glob.glob('/home/trim/wb2data/ladder_*.npz')); D = [np.load(f) for f in fs]
rungs = [k for k in D[0].files if k not in ('gt', 'top')]
print(f"дампів {len(D)}; MSE top сер {np.mean([((d['top']-d['gt'])**2).mean() for d in D]):.4e}")
print("рунг        | MSE×     | cos(e_r,e_t) | cos(d,e_t) | LOO суміш p=top−c·d: MSE× (геом.сер) | c (сер±sd)")
for r in rungs:
    if not all(r in d.files for d in D): continue
    rows = []; cs = []
    E = [(d['top']-d['gt'], d[r]-d['top'], d[r]-d['gt']) for d in D]
    for h in range(len(D)):
        num = sum(np.dot(E[i][0], E[i][1]) for i in range(len(D)) if i != h); den = sum(np.dot(E[i][1], E[i][1]) for i in range(len(D)) if i != h)
        c = num/den; cs.append(c); et, dd, er = E[h]
        rows.append((((er**2).mean()/(et**2).mean()), np.dot(er,et)/np.linalg.norm(er)/np.linalg.norm(et), np.dot(dd,et)/np.linalg.norm(dd)/np.linalg.norm(et), ((et - c*dd)**2).mean()/(et**2).mean()))
    a = np.array(rows)
    print(f"{r:11s} | {np.exp(np.log(a[:,0]).mean()):8.2f} | {a[:,1].mean():+.3f}       | {a[:,2].mean():+.3f}     | {np.exp(np.log(a[:,3]).mean()):.4f} [{a[:,3].min():.3f}…{a[:,3].max():.3f}] | {np.mean(cs):+.4f}±{np.std(cs):.4f}")
