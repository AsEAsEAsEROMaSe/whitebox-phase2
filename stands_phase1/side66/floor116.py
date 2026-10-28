"""ПІДЛОГА ШУМУ БАНКУ `gamall` ПОШАРОВО — з власних (mp,sd) банку.
Var(ReLU(p)) = (mp²+s²)Φ(t) + mp·s·φ(t) − m²  ⟹ шум = ‖sd_relu‖/(√N·‖m‖).
🔒 на шарі 0 формула m₀=s₀/√(2π) ТОЧНА ⟹ зміряна «похибка» там мусить дорівнювати
   цій підлозі. Це пломба на сам прилад.
"""
import glob, numpy as np, sys
sys.argv = ["x"]
INV = 0.3989422804014327
def phi(x): return np.exp(-0.5*x*x)*INV
from math import erf as _erf
def Phi(x): return 0.5*(1.0+np.vectorize(_erf)(x/np.sqrt(2.0)))
D = 16
F = sorted(glob.glob("/home/trim/wblogs/gamall_*.npz"))
acc = np.zeros((len(F), D))
for k, f in enumerate(F):
    d = np.load(f, allow_pickle=True); N = int(d["n"])
    for L in range(D):
        mp, s = d["mp_%d" % L], d["sd_%d" % L]
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        v = np.maximum((mp*mp + s*s)*Ph + mp*s*ph - m*m, 0.0)
        acc[k, L] = 100*np.sqrt(v.sum()/N)/np.linalg.norm(d["relu_%d" % L])
n = acc.mean(0)
sig = np.array([0, 0, 0, 0, 0.1204, 0, 0, 0, 0.1750, 0, 0, 0, 0, 0, 0, 0.2207])
print("ПІДЛОГА ШУМУ БАНКУ gamall (N=2²², %d мереж)" % len(F))
print("%4s %14s" % ("шар", "шум %"))
for L in range(D): print("%4d %13.4f%%" % (L, n[L]))
print("\n🔒 шар 0: теорія 0.0715%% · модель %.4f%% · зміряна «похибка» 0.0721%%" % n[0])
