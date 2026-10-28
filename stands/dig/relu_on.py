"""17.10.2026, ідея юзера: на ОДНИХ вагах (дамп d) поступово вмикати ReLU по нейронах, від першого шару, і дивитись середнє виходу.
Порядок вмикання: шар 0 нейрони 0..1023, потім шар 1, ... (лінійний індекс k = 1024*l + j). Вимкнений нейрон = тотожність (пропускає x як є).
На кожному чекпоінті N точок z, середнє виходу m_k (1024). Записуємо норму, середнє по нейронах, cos з повним ReLU-середнім (банк mean[15]),
і відстань до нього. Також частка «вмикань», що змінили середнє більше за шум."""
import sys, os, numpy as np, torch
os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","GPU-eaa8132f-e46f-2090-d943-0481fbe03aef")
d=int(sys.argv[1]) if len(sys.argv)>1 else 0; N=int(sys.argv[2]) if len(sys.argv)>2 else 65536; STEP=int(sys.argv[3]) if len(sys.argv)>3 else 128
ORDER=sys.argv[4] if len(sys.argv)>4 else 'seq'   # seq | rand (випадковий порядок по всій мережі, сід 1)
dev='cuda'; Wn=np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%d)['weights']; W=[torch.tensor(Wn[l],dtype=torch.float32,device=dev) for l in range(16)]
kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%d); m_full=kp['mean'][15].astype(np.float64)
g=torch.Generator(device=dev).manual_seed(777); Z=torch.randn(N,1024,generator=g,device=dev)   # ті самі точки на всіх чекпоінтах
perm=np.random.RandomState(1).permutation(16*1024) if ORDER=='rand' else np.arange(16*1024)   # perm[i] = лінійний індекс нейрона, що вмикається i-тим
def mean_out(k):   # ReLU увімкнено на перших k нейронах у лінійному порядку
    on=np.zeros(16*1024,dtype=bool); on[perm[:k]]=True; h=Z
    for l in range(16):
        p=h@W[l]; m=torch.tensor(on[1024*l:1024*(l+1)],device=dev)
        h=torch.where(m[None,:],torch.relu(p),p)
    return h.double().mean(0).cpu().numpy()
ks=list(range(0,16*1024+1,STEP)); rows=[]; tail={}
for k in ks:
    m=mean_out(k)
    if k>=16*1024-4096: tail[k]=m
    rows.append((k,np.linalg.norm(m),m.mean(),float(m@m_full/(np.linalg.norm(m)*np.linalg.norm(m_full)+1e-30)),np.sqrt(((m-m_full)**2).mean())))
    if k%2048==0: print("k=%5d шар %2d  |m|=%.4e  mean=%.4e  cos=%.4f  rms(m−повне)=%.4e"%(k,k//1024,rows[-1][1],rows[-1][2],rows[-1][3],rows[-1][4]),flush=True)
np.savez('/home/trim/wblogs/relu_on_d%d%s.npz'%(d,'' if ORDER=='seq' else '_'+ORDER),ks=np.array(ks),rows=np.array(rows),m_full=m_full,tail_k=np.array(sorted(tail)),tail_m=np.array([tail[k] for k in sorted(tail)]))
print("DONE")
