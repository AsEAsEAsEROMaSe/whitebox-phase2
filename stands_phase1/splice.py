"""СКЛЕЙКА ПОЛОВИН (ідея юзера С47): беремо 256-вимірний семпл, ріжемо навпіл,
перші 128 від одного семпла + другі 128 від іншого.

ЗАКОННІСТЬ: координати x незалежні ⟹ склейка = ТОЧНО такий самий гаус. Новий валідний
семпл БЕЗ витрати випадковості. (Обчислень не economить: перший шар справді розкладається
на два пів-добутки, але він один із 32 ⟹ ~1.5%, губиться в шумі.)

ЧОМУ ЦЕ МОЖЕ ДАТИ ВИГРАШ: схема «всі n² перехресних склейок із n+n половин» має
дисперсію головних ефектів ~1/n, а дисперсію ВЗАЄМОДІЇ ~1/n². Взаємодія оцінюється
квадратично краще. Але коштує n² прогонів ⟹ при рівному бюджеті виграє ЛИШЕ якщо
взаємодія переважає, а головні ефекти малі.

МІРЯЄМО двофакторний розклад дисперсії (ANOVA) на сітці m×k склейок:
    y_ij = μ + a_i + b_j + e_ij
    частка головних ефектів (A,B) проти частки ВЗАЄМОДІЇ
Незміщені компоненти (модель випадкових ефектів):
    σ²_AB = SS_AB/((m−1)(k−1));  σ²_A = (SS_A/(m−1) − σ²_AB)/k;  σ²_B = аналогічно

ВЕРДИКТ: якщо взаємодія ≫ головних — схема склейок ЖИВА; якщо навпаки — мертва.
Додатково пробуємо 4 блоки по 64 (чи росте взаємодія з дрібнішим дробленням).
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/splice.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def even(W,x):
    a=x
    for w in W: a=torch.relu(a@w)
    b=-x
    for w in W: b=torch.relu(b@w)
    return (a+b)*0.5

M=K=96                                        # сітка 96×96 = 9216 склейок
for net in (0,1):
    W=load(net); log('='*78); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(770000+net)
    HA=torch.randn(M,D//2,device=dev,generator=g)      # перші половини
    HB=torch.randn(K,D//2,device=dev,generator=g)      # другі половини
    Y=torch.empty(M,K,D,dtype=torch.float64,device=dev)
    for i in range(M):
        x=torch.cat([HA[i].unsqueeze(0).expand(K,-1),HB],1)
        Y[i]=even(W,x).double()
    gm=Y.mean((0,1)); ri=Y.mean(1); cj=Y.mean(0)
    SSA=float((K*((ri-gm)**2)).sum()); SSB=float((M*((cj-gm)**2)).sum())
    R=Y-ri.unsqueeze(1)-cj.unsqueeze(0)+gm
    SSAB=float((R*R).sum())
    vAB=SSAB/((M-1)*(K-1))
    vA=(SSA/(M-1)-vAB)/K; vB=(SSB/(K-1)-vAB)/M
    tot=vA+vB+vAB
    log('  сітка {:d}×{:d} = {:d} склейок (×2 прогони на антитетику)'.format(M,K,M*K))
    log('  головний ефект першої половини  {:.4f}'.format(vA/tot))
    log('  головний ефект другої половини  {:.4f}'.format(vB/tot))
    log('  ⟹ ВЗАЄМОДІЯ                     {:.4f}'.format(vAB/tot))
    log('  вердикт: схема n² виграє лише при взаємодії ≫ головних ефектів')

    # чи росте взаємодія з дрібнішим дробленням: 4 блоки по 64
    g2=torch.Generator(device=dev).manual_seed(880000+net)
    NB=4; SZ=D//NB; m2=40
    H=[torch.randn(m2,SZ,device=dev,generator=g2) for _ in range(NB)]
    idx=torch.randint(0,m2,(4000,NB),device=dev,generator=g2)
    x=torch.cat([H[b][idx[:,b]] for b in range(NB)],1)
    y=even(W,x).double()
    vtot=float(y.var(0).sum())
    expl=0.0
    for b in range(NB):
        mb=torch.zeros(m2,D,dtype=torch.float64,device=dev); cn=torch.zeros(m2,device=dev)
        mb.index_add_(0,idx[:,b],y); cn.index_add_(0,idx[:,b],torch.ones(4000,device=dev))
        mb=mb/cn.clamp(min=1).unsqueeze(1)
        expl+=float(((mb-y.mean(0))**2).mean(0).sum())
    log('  4 блоки по 64: сума головних ефектів {:.4f}  ⟹ взаємодія {:.4f}'
        .format(expl/vtot,1-expl/vtot))
log('done.')
