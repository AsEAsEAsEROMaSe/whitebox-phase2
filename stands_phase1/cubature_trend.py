import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/cubature_trend.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
def fwd_sum(X,wts,CH=8192):
    acc=torch.zeros(D,dtype=torch.float64,device=dev)
    for i in range(0,X.shape[0],CH):
        x=X[i:i+CH]
        for li in range(L): x=relu(x@W[li])
        acc+=(x.double()*wts[i:i+CH,None]).sum(0)
    return acc
def gt(N=8_000_000,CH=100_000):
    acc=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(555+c))
        x=torch.cat([z,-z],0)
        cov=(x.t()@x)/x.shape[0]
        ev,V=torch.linalg.eigh(cov.double().cpu()); wh=(V*(1.0/ev.clamp(min=1e-12)).sqrt())@V.t()
        x=x@wh.float().to(dev)
        for li in range(L): x=relu(x@W[li])
        acc+=x.double().sum(0); n+=x.shape[0]
    return acc/n
log('ground truth ...'); G=gt(); log('GT mean {:.6f}'.format(float(G.mean())))
E=torch.eye(D,device=dev)

def mse(e): return float(((e-G)**2).mean())

# --- degree-3 rule: 2d points +-sqrt(d) e_i, equal weights (exact through degree 3) ---
X3=torch.cat([math.sqrt(D)*E,-math.sqrt(D)*E],0)
W3=torch.full((2*D,),1.0/(2*D),dtype=torch.float64,device=dev)
m3=mse(fwd_sum(X3,W3))
log('CUBATURE deg-3 : {:>7} pts  MSE = {:.4e}'.format(2*D,m3))

# --- degree-5 Stroud: 2d^2+1 points ---
A=2.0/(D+2); B=(4.0-D)/(2.0*(D+2)**2); C=1.0/((D+2)**2)
r=math.sqrt(D+2); s=math.sqrt((D+2)/2.0)
iu=torch.triu_indices(D,D,offset=1); ei=E[iu[0]]; ej=E[iu[1]]
pts=[torch.zeros(1,D,device=dev), r*E, -r*E]
wts=[torch.tensor([A],dtype=torch.float64,device=dev),
     torch.full((D,),B,dtype=torch.float64,device=dev),
     torch.full((D,),B,dtype=torch.float64,device=dev)]
for sa in (1,-1):
    for sb in (1,-1):
        pts.append(s*(sa*ei+sb*ej)); wts.append(torch.full((iu.shape[1],),C,dtype=torch.float64,device=dev))
X5=torch.cat(pts,0); W5=torch.cat(wts,0)
m5=mse(fwd_sum(X5,W5))
log('CUBATURE deg-5 : {:>7} pts  MSE = {:.4e}   ({}x more points than deg-3)'.format(
    X5.shape[0],m5,X5.shape[0]//(2*D)))

# --- MC / v2 at both budgets, for the scaling comparison ---
def v2(n_pts,seed):
    z=torch.randn(n_pts//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(seed))
    x=torch.cat([z,-z],0)
    cov=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(cov.double().cpu()); wh=(V*(1.0/ev.clamp(min=1e-12)).sqrt())@V.t()
    x=x@wh.float().to(dev)
    return fwd_sum(x,torch.full((x.shape[0],),1.0/x.shape[0],dtype=torch.float64,device=dev))
v_small=float(np.mean([mse(v2(2*D,100+i)) for i in range(5)]))
v_big  =float(np.mean([mse(v2(131072,200+i)) for i in range(3)]))
log('v2       @ {:>7} pts  MSE = {:.4e}'.format(2*D,v_small))
log('v2       @ {:>7} pts  MSE = {:.4e}'.format(131072,v_big))
log('')
log('=== SCALING: what did 256x more points buy? ===')
log('  cubature deg3 -> deg5 : {:.4e} -> {:.4e}   factor {:.2f}x'.format(m3,m5,m3/max(m5,1e-30)))
log('  v2        512 -> 131k : {:.4e} -> {:.4e}   factor {:.2f}x  (ideal 1/N = 256x)'.format(
    v_small,v_big,v_small/max(v_big,1e-30)))
log('done.')
