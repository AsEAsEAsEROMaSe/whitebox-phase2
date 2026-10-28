"""ПЕРЕВІРКА ПРИЧИННОГО ЗАКОНУ НА СПОСТЕРЕЖЕННІ: чи передбачає ρ, де ми гірші.

[pert116, ВТРУЧАННЯ] похибка йде за медіанним |ρ| між нейронами, e ∝ ρ^1.43
   (немонотонно по ручці λ, монотонно по ρ ⟹ це причина, не супутність)
Якщо закон чинний — `ρ` мусить передбачати ПОМЕРЕЖНУ похибку на реальних
мережах, яких ми не чіпали. Це перевірка причинного виміру спостереженням.

🔮 V1 (60%) corr(log ρ, log MSE) > 0.4 на 24 мережах
🔮 V2 (45%) нахил близький до 1.43·2 = 2.9 (MSE = e²)
🔒 КОНТРОЛЬ: `ρ` на шарі 0 (де замикання ТОЧНЕ) не мусить нічого пояснювати
"""
import sys, glob, json, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def rho_profile(W):
    """медіана |ρ| між передактиваціями на КОЖНОМУ шарі, за замиканням"""
    m = np.zeros(NW); C = np.eye(NW); out = []
    iu = np.triu_indices(NW, 1)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        R = cp_/np.outer(s, s)
        out.append(float(np.median(np.abs(R[iu]))))
        if i == D-1: return np.array(out)
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
per = {x["mlp_name"]: x["final_layer_mse"]
       for x in json.load(open("/home/trim/wblogs/s_g3d.json"))["results"]["per_mlp"]}
FLOOR = 1.813e-07     # підлога приладу dev24 (С116)
rows = []
for k, nm in enumerate(NAMES):
    if nm not in per: continue
    W = np.asarray(tb["weights"][k].as_py(), np.float64).reshape(D, NW, NW)
    rp = rho_profile(W)
    mse = per[nm] - FLOOR
    rows.append((nm, rp, mse))
    lg("%-18s ρ₁₅ %.4f  ρ̄(9..15) %.4f  MSE-підлога %.4e" % (nm, rp[15], rp[9:].mean(), mse))

lM = np.log(np.array([r[2] for r in rows]))
print("\n── corr(log ρ на шарі L, log(MSE−підлога)) на %d мережах ──" % len(rows))
print("%5s %10s %12s" % ("шар", "corr", "нахил"))
for L in (0, 1, 4, 8, 11, 13, 15):
    lr = np.log(np.array([r[1][L] for r in rows]))
    c = np.corrcoef(lr, lM)[0, 1]
    sl = np.polyfit(lr, lM, 1)[0]
    print("%5d %10.4f %12.3f" % (L, c, sl))
lr = np.log(np.array([r[1][9:].mean() for r in rows]))
c = np.corrcoef(lr, lM)[0, 1]; sl = np.polyfit(lr, lM, 1)[0]
print("%5s %10.4f %12.3f   ← середнє по 9..15" % ("9-15", c, sl))
print("\n🔮 передбачення з ВТРУЧАННЯ: нахил log MSE / log ρ = 2×1.43 = 2.86")
print("розкид ρ₁₅ по мережах: %.4f … %.4f  (×%.3f)"
      % (min(r[1][15] for r in rows), max(r[1][15] for r in rows),
         max(r[1][15] for r in rows)/min(r[1][15] for r in rows)))
print("розкид MSE−підлога:    ×%.3f" % (max(r[2] for r in rows)/min(r[2] for r in rows)))
lg("ГОТОВО")
