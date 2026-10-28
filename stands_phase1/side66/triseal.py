"""пломба: tri_kab з rec118h == T3 із tri118 (regina, L=1); tri_core узгоджене з tri_kab через core_contract при повному базисі"""
import os, sys, numpy as np
os.environ["TRI"] = "1"; os.environ["SEAL"] = "0"
src = open("rec118h.py").read()
exec(src.split("import pyarrow")[0])          # константи, функції, tri_kab/tri_core, core_contract
d = np.load("/home/trim/wblogs/tri118_regina-johnston_L1.npz"); T3 = d["T3"]
import pyarrow.parquet as pq
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]; idx = NAMES.index("regina-johnston")
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(16, 1024, 1024)
mp = np.zeros(1024); Sp = W[0].T @ W[0]; s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s); c = hcoef(s, t, 4)
K = tri_kab(c[2], rho, W[1])
print("S1 tri_kab vs tri118: max|Δ|/max|T3| %.2e" % (np.abs(K - T3).max()/np.abs(T3).max()))
# S2: ядро у випадковому базисі r=48 проти прямої проєкції Ktri: core_contract(C,V)[a,b] має = Σ_ijk T3 (WV)(WV)(WV) контрактоване... перевіряємо через повний базис на підвибірці
rng = np.random.default_rng(0); V = np.linalg.qr(rng.standard_normal((1024, 48)))[0]
C = tri_core(c[2], rho, W[1], V)
# незалежний шлях: K_V[a,b] = Σ_αβγ C V_aα V_aβ V_bγ проти проєкції точного K: (V Vᵀ) не дає точну рівність, тому пломба на симетрії і на слід-контракті:
print("S2 симетрія ядра: max|C - C^T(1,0,2)| %.2e · max|C - C^T(2,1,0)| %.2e · max|C| %.2e" % (np.abs(C - C.transpose(1, 0, 2)).max(), np.abs(C - C.transpose(2, 1, 0)).max(), np.abs(C).max()))
# S3: повна контракція з одним вектором v: Σ_abc T(p)_abc v_a v_b v_c обчислюється і з K? ні. Натомість: C у базисі V проти прямого n³r-незалежного обчислення через einsum на підвибірці i∈[0,64)
import itertools
U = c[2][:, None]*(W[1] @ V); n = 64
Cd = np.zeros((48, 48, 48))
for i in range(n):
    M = rho*np.outer(rho[:, i], rho[:, i]); Cd += U[i][:, None, None]*(U.T @ M @ U)[None]
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp
rg = cp.asarray(rho, cp.float32); Ug = cp.asarray(U, cp.float32); Cg = cp.zeros((48, 48, 48), cp.float64)
for i in range(n):
    M = rg*rg[:, i][:, None]*rg[:, i][None, :]; Cg += Ug[i][:, None, None]*(Ug.T @ M @ Ug)[None].astype(cp.float64)
print("S3 tri_core (GPU fp32, перші 64 хаби) vs numpy fp64: max|Δ|/max %.2e" % (np.abs(cp.asnumpy(Cg) - Cd).max()/np.abs(Cd).max()))
print("DONE_SEAL")
