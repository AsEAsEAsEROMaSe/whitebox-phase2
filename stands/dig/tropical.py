"""18.10.2026, юзер: поміряти тропічний розклад ReLU-мережі (Zhang–Naitzat–Lim) через W = W⁺ − W⁻.
Рекурсія: p_l = F_l − G_l, F,G ≥ 0 опуклі й однорідні; ReLU(p) = max(F,G) − G;
F_{l+1} = max(F_l,G_l)·W⁺ + G_l·W⁻ ,  G_{l+1} = G_l·W⁺ + max(F_l,G_l)·W⁻ ;  F_0 = ReLU(zW₀), G_0 = ReLU(−zW₀).
Міряємо: E[F], E[G] проти E[h] (гасіння), ріст по шарах, точність тотожності в fp64.
Передбачення: F,G ростуть ~×36 на шар (Σ|w| стовпця ≈ √2048·√(2/π)), тотожність зламається в fp64 приблизно з шару 9–10."""
import os, numpy as np, torch
os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
Wn=np.load('/home/trim/wb2data/dumps_p2/mlp_0000.npz')['weights']; W=[torch.tensor(Wn[l],dtype=torch.float64,device='cuda') for l in range(16)]
Wp=[w.clamp_min(0) for w in W]; Wm=[(-w).clamp_min(0) for w in W]
N=32768; z=torch.randn(N,1024,generator=torch.Generator(device='cuda').manual_seed(777),device='cuda',dtype=torch.float64)
rms=lambda v: float(v.pow(2).mean().sqrt())
p=z@W[0]; h=torch.relu(p); F=torch.relu(p); G=torch.relu(-p)
print("шар  |E[h]|     |E[F]|     |E[G]|   гасіння |E[F]|/|E[h]|  σ(F)/σ(h)  похибка тотожності max|h−(max(F,G)−G)|/|h|max")
for l in range(16):
    if l>0:
        p=h@W[l]; h=torch.relu(p); M=torch.maximum(F,G); F,G=M@Wp[l]+G@Wm[l], G@Wp[l]+M@Wm[l]
    rec=torch.maximum(F,G)-G; Eh,EF,EG=h.mean(0),F.mean(0),G.mean(0)
    print("%2d  %.3e  %.3e  %.3e   %.2e   %.2e   %.2e"%(l,rms(Eh),rms(EF),rms(EG),rms(EF)/rms(Eh),rms(F.std(0))/rms(h.std(0)),float((h-rec).abs().max()/h.abs().max())))
