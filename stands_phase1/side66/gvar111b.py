# 🔒 ДВІ ПЛОМБИ НА ПРАВИЛЬНИХ МІСЦЯХ (перша спроба міряла не те):
#  P1 ЛІНІЙНА ознака + АНТИТЕТИКА ⟹ РІВНО 0 (заперечення у fp точне)
#  P2 КВАДРАТИЧНА ознака + ВИБІЛЮВАННЯ ⟹ mean(y²) = ‖w‖² РІВНО
#     (бо z'ᵀz'/N = I ⟹ diag(W0ᵀ·I·W0))
# Якщо обидві тримають — прилад міряє саме те, що я стверджую, і падіння ‖g‖²
# у 15.2× є ВЛАСТИВІСТЮ relu, а не артефактом.
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(99); NPAIR=3072
W0=np.array(W[0,0],dtype=np.float64)
z=rng.standard_normal((NPAIR,256))
cov=(z.T@z)/NPAIR; wh=np.linalg.inv(np.linalg.cholesky(cov)).T
w0=wh@W0; y=z@w0
print(f"🔒P1 лінійна+антитетика   : {np.abs(np.concatenate([y,-y]).mean(0)).max():.3e}  (мусить бути ~1e-16)")
lin_noanti=np.abs(y.mean(0)).max()
print(f"   та сама БЕЗ антитетики : {lin_noanti:.3e}  ⟹ нуль дає антитетика, не вибілювання")
q_true=(W0*W0).sum(0)
print(f"🔒P2 квадратична+вибілюв.  : {np.abs((y*y).mean(0)/q_true-1).max():.3e}  (мусить бути ~1e-14)")
zr=rng.standard_normal((2*NPAIR,256)); yr=zr@W0
print(f"   та сама БЕЗ вибілювання: {np.abs((yr*yr).mean(0)/q_true-1).max():.3e}  ⟹ ×{np.abs((yr*yr).mean(0)/q_true-1).max()/max(np.abs((y*y).mean(0)/q_true-1).max(),1e-18):.1e}")
# 🔑 і ГОЛОВНЕ: relu лежить МІЖ ними ⟹ бере частину точності, але не всю
