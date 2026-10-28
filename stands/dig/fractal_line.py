"""Ідея юзера: уся мережа = один «сумарний фрактал» (розбиття входу на лінійні області всіма шарами разом).
Тест фрактальності: на відрізку z0 + s*u (довжина ~ ||z||) рахуємо перетини меж (перемикання будь-яких воріт) при
роздільності 2^12..2^20. Фрактал: кількість росте як res^D (D>0). Нефрактал: насичується.
Також розподіл довжин областей (степеневий хвіст = багатомасштабність) і внесок шарів."""
import numpy as np, torch, math
dev='cuda'; torch.manual_seed(1)
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_0000.npz')['weights'],device=dev).float()
n=1024
def patterns(Z):
    a=Z; P=[]
    for l in range(16):
        p=a@W[l]; P.append(p>0); a=torch.relu(p)
    return torch.stack(P,1)  # B,16,n
for line in range(3):
    z0=torch.randn(n,device=dev); u=torch.randn(n,device=dev); u=u/u.norm()*z0.norm()
    res_counts=[]
    for R in [12,14,16,18,20]:
        M=2**R; flips=0; changes=0; perlayer=torch.zeros(16,device=dev); prev=None; lens=[]; last=0
        chunk=2**14
        for c0 in range(0,M+1,chunk):
            s=torch.arange(c0,min(c0+chunk,M+1),device=dev).float()/M
            P=patterns(z0[None]+s[:,None]*u[None])
            if prev is not None: P=torch.cat([prev[None],P],0)
            d=(P[1:]!=P[:-1])
            flips+=int(d.sum()); ch=d.any(2).any(1); changes+=int(ch.sum()); perlayer+=d.sum(2).sum(0).float()
            if R==20:
                idx=torch.nonzero(ch).flatten().cpu().numpy()+c0-(1 if prev is not None else 0)
                lens+=list(np.diff(np.concatenate([[last],idx]))); last=idx[-1] if len(idx) else last
            prev=P[-1]
        res_counts.append((R,flips,changes))
        if R==20:
            L=np.array(lens[1:],float)/M
            q=np.quantile(L,[0.01,0.1,0.5,0.9,0.99])
            pl=(perlayer/perlayer.sum()).cpu().numpy()
    print('лінія %d: '%line+' | '.join('res 2^%d: перемикань %d, змін області %d'%x for x in res_counts))
    print('   довжини областей (частка відрізка) квантилі 1/10/50/90/99%%: %s ; частка перемикань шари 0/4/8/12/15: %s'%(' '.join('%.1e'%x for x in q),' '.join('%.3f'%pl[l] for l in [0,4,8,12,15])))
