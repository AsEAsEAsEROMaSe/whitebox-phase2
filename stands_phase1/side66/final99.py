"""Зводить обидва зрізи докупи: n=32 (чутливий режим) і n=256 (робоча точка),
і рахує гейт цінності на ЗМІРЯНОМУ розкиді, а не на припущеному."""
import math, numpy as np
from collections import Counter
L = "/mnt/win_d/llm/White-Box/arena62/logs/"
a = np.load(L+"wfeat99.npz", allow_pickle=True)
b = np.load(L+"wfeat256.npz", allow_pickle=True)
names = list(a["names"])

def loo_best_single(X, y):
    m, p = X.shape; pred = np.empty(m); picks=[]
    for i in range(m):
        tr = np.ones(m, bool); tr[i]=False
        Xt, yt = X[tr], y[tr]; best,bj=-np.inf,0
        for j in range(p):
            c = np.corrcoef(Xt[:,j], yt)[0,1]; c = 0.0 if not np.isfinite(c) else c
            if c*c>best: best,bj=c*c,j
        sx=Xt[:,bj]; bb=np.cov(sx,yt)[0,1]/max(np.var(sx),1e-300)
        pred[i]=yt.mean()+bb*(X[i,bj]-sx.mean()); picks.append(bj)
    return 1.0-float(((y-pred)**2).sum())/float(((y-y.mean())**2).sum()), picks

print(f"{'ознака':<20s}{'sd@32':>10s}{'sd@256':>10s}{'r@32':>9s}{'r@256':>9s}")
for j,nm in enumerate(names):
    r32 = np.corrcoef(a["X"][:,j], a["y"])[0,1]
    r256= np.corrcoef(b["X"][:,j], b["y"])[0,1]
    print(f"{nm:<20s}{a['X'][:,j].std():>10.5f}{b['X'][:,j].std():>10.5f}"
          f"{np.nan_to_num(r32):>+9.3f}{np.nan_to_num(r256):>+9.3f}")

for tag, d in (("n=32 ", a), ("n=256", b)):
    y = d["y"]; X = d["X"]
    r2, picks = loo_best_single(X, y)
    print(f"\n{tag}: M={len(y)} · медіана e={100*math.exp(np.median(y)):.4f}%"
          f" · sd(log e)={y.std():.4f} · LOO-R²={r2:+.4f}"
          f" · обрана ознака: {Counter(names[k] for k in picks).most_common(2)}")

# ── гейт цінності на ЗМІРЯНОМУ розкиді робочої точки
V, E2 = 1.9227e-06, 2.3385e-05
SDL = float(b["y"].std())
rng = np.random.default_rng(0); M=400000
s = 2.0*SDL; g = rng.standard_normal(M)
Ei2 = E2*np.exp(s*g - 0.5*s*s)
wg = V/(V+Ei2.mean())
base = (wg**2*Ei2 + (1-wg)**2*V).mean()
orac = (V*Ei2/(V+Ei2)).mean()
lg = np.log(Ei2); mu = lg.mean()
print(f"\nГЕЙТ ЦІННОСТІ на ЗМІРЯНОМУ sd(log e)={SDL:.4f} (n=256, істина 1e9):")
print(f"  глобальний w={wg:.5f}  MSE={base:.4e}   [пломба: записана база 1.8115e-06]")
for r2v in (0.0, 0.10, 0.20, 0.30, 0.50, 1.00):
    pr = np.exp(mu + r2v*(lg-mu)); wp = V/(V+pr)
    m_ = (wp**2*Ei2 + (1-wp)**2*V).mean()
    print(f"  R²={r2v:.2f} ⟹ MSE={m_:.4e}  ×{base/m_:.4f}"
          f"   бал {1.86762e-07*m_/base:.5e}")
