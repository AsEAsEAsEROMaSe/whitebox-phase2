"""Аналогія з магнітним маятником: чи є чутливість до початкових умов (ефект метелика)?
Пари z, z+eps*u через мережу: розтяг ||dh_l||/||dz|| по шарах і частка перемкнутих воріт.
Контроль: та сама мережа з вагами x1.5 (хаотична фаза) мусить показати експоненційний ріст."""
import numpy as np, torch
dev='cuda'; torch.manual_seed(0)
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_0000.npz')['weights'],device=dev).float()
N=4096; n=W.shape[1]
def run(scale, eps):
    z=torch.randn(N,n,device=dev); u=torch.randn(N,n,device=dev); u=u/u.norm(dim=1,keepdim=True)*z.norm(dim=1,keepdim=True)*eps
    a,b=z,z+u; out=[]
    for l in range(16):
        pa=a@(W[l]*scale); pb=b@(W[l]*scale)
        flip=((pa>0)!=(pb>0)).float().mean().item()
        a=torch.relu(pa) if l<15 else pa; b=torch.relu(pb) if l<15 else pb
        s=((a-b).norm(dim=1)/u.norm(dim=1)/ (a.norm(dim=1)/z.norm(dim=1))).median().item()
        out.append((s,flip))
    return out
for scale,nm in [(1.0,'наша мережа'),(1.5,'контроль: ваги x1.5')]:
    for eps in [1e-3,1e-1]:
        r=run(scale,eps)
        print('%-20s eps %.0e | відносний розтяг L0/4/8/12/15: %s | перемкнуто воріт L15 %.4f'%(nm,eps,' '.join('%.2f'%r[l][0] for l in [0,4,8,12,15]),r[15][1]))
